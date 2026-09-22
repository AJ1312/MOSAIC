"""End-to-end integration test: a full L0-L4 run on a tiny generated corpus.

Trains real (tiny) branch models on a handful of clips, runs the whole pipeline, and
checks the structural guarantees the system claims — not accuracy, which a corpus this
small cannot support.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_ffmpeg
from mosaic.config import MosaicConfig, apply_tier
from mosaic.fusion import AVCouplingModel, Verdict
from mosaic.l1_hash import HashRegistry
from mosaic.l3_audio import AUDIO_FEATURE_NAMES, extract_audio_features
from mosaic.l3_av_sync import AV_FEATURE_NAMES, extract_av_features
from mosaic.l3_tier0 import TIER0_FEATURE_NAMES, extract_tier0_features
from mosaic.l3_visual import VISUAL_FEATURE_NAMES, extract_visual_features
from mosaic.l0_ingest import ingest
from mosaic.l4_custody import verify_ledger, verify_proof, MerkleTree, commit_video
from mosaic.models import BranchModel
from mosaic.pipeline import ModelBundle, MosaicPipeline

pytestmark = requires_ffmpeg


@pytest.fixture(scope="module")
def trained_bundle(tmp_path_factory):
    """Train tiny models on a small generated corpus.

    Synthetic feature vectors would not exercise the real extractors, so this trains on
    features genuinely extracted from generated media — just very few of them.
    """
    from mosaic.data.corpus import build_corpus, plan_corpus

    out = tmp_path_factory.mktemp("train_corpus")
    specs = plan_corpus(3, seed=11, splits=(("train", 1.0),))
    records = build_corpus(specs, out, workers=4, progress=False)
    assert all(not r.get("error") for r in records)

    cfg = MosaicConfig()
    work = tmp_path_factory.mktemp("train_work")
    V, A, S, T0 = [], [], [], []
    vf, af, ds = [], [], []
    for r in records:
        m = ingest(r["path"], cfg.canonical, workdir=work).media
        V.append(extract_visual_features(m.frames, 24).vector)
        A.append(extract_audio_features(m.audio, m.sample_rate).vector)
        S.append(extract_av_features(m.frames, m.fps, m.audio, m.sample_rate).vector)
        T0.append(extract_tier0_features(m.motion_vectors, m.audio, m.sample_rate).vector)
        vf.append(r["video_fake"])
        af.append(r["audio_fake"])
        ds.append(r["sync_state"] == "desynced")

    V, A, S, T0 = map(np.array, (V, A, S, T0))
    vf, af, ds = map(np.array, (vf, af, ds))
    any_fake = vf | af

    bundle = ModelBundle(
        visual=BranchModel("visual", VISUAL_FEATURE_NAMES, n_bootstrap=5).fit(V, vf),
        audio=BranchModel("audio", AUDIO_FEATURE_NAMES, n_bootstrap=5).fit(A, af),
        av=BranchModel("av", AV_FEATURE_NAMES, n_bootstrap=5).fit(S, ds),
        tier0=BranchModel("tier0", TIER0_FEATURE_NAMES, n_bootstrap=5).fit(T0, any_fake),
        coupling=AVCouplingModel().fit(vf, af, ds),
        audio_cnn=None,
        metadata={"data_is_synthetic": True},
    )
    return bundle, records


def _pipeline(bundle, tmp_path, tier="consumer"):
    from dataclasses import replace

    cfg = apply_tier(MosaicConfig(), tier)
    cfg = replace(cfg, l4=replace(cfg.l4,
                                  ledger_path=str(tmp_path / "ledger.jsonl"),
                                  records_dir=str(tmp_path / "records")))
    registry = HashRegistry(tmp_path / "reg.sqlite", cfg.l1)
    return MosaicPipeline(cfg, bundle, device_profile={"tier": tier}, registry=registry,
                          workdir=tmp_path / "work", synthetic_data=True), cfg


def test_pipeline_runs_end_to_end_and_seals_custody(trained_bundle, tiny_corpus, tmp_path):
    bundle, _ = trained_bundle
    pipe, cfg = _pipeline(bundle, tmp_path)

    results = [pipe.process(r["path"], clip_id=r["clip_id"]) for r in tiny_corpus]
    assert len(results) == len(tiny_corpus)

    for res in results:
        assert isinstance(res.verdict, Verdict)
        assert res.custody_record_id and res.combined_root
        assert "L0" in res.stages_run and "L1" in res.stages_run and "L4" in res.stages_run
        assert res.timings_s["total"] > 0
        # Every run must carry the synthetic-data caveat.
        assert any("SYNTHETIC" in c for c in res.caveats)

    state = verify_ledger(cfg.l4.ledger_path)
    assert state["valid"] and state["n_entries"] == len(tiny_corpus)


def test_custody_record_contains_every_required_field(trained_bundle, tiny_corpus, tmp_path):
    bundle, _ = trained_bundle
    pipe, cfg = _pipeline(bundle, tmp_path)
    res = pipe.process(tiny_corpus[0]["path"], clip_id=tiny_corpus[0]["clip_id"])

    record = json.loads(
        (Path(cfg.l4.records_dir) / f"{res.custody_record_id}.json").read_text())
    for field in ("record_id", "source_sha256", "canonical_sha256", "config_digest", "seed",
                  "device_profile", "environment", "pipeline_description", "ingest",
                  "l1_triage", "l2_provenance", "l3_detection", "verdict", "evidence",
                  "models", "track_commitments", "combined_root", "timestamp", "events",
                  "compute", "ledger_entry", "caveats"):
        assert field in record, f"custody record is missing '{field}'"

    assert record["ingest"]["ffmpeg_command"]
    assert record["track_commitments"]["video"]["root"]
    assert record["timestamp"]["is_qualified"] is False
    assert len(record["events"]) >= 4
    seqs = [e["seq"] for e in record["events"]]
    assert seqs == sorted(seqs), "custody events must be ordered"


def test_merkle_proof_from_a_sealed_record_verifies(trained_bundle, tiny_corpus, tmp_path):
    bundle, _ = trained_bundle
    pipe, cfg = _pipeline(bundle, tmp_path)
    res = pipe.process(tiny_corpus[0]["path"], clip_id=tiny_corpus[0]["clip_id"])
    record = json.loads(
        (Path(cfg.l4.records_dir) / f"{res.custody_record_id}.json").read_text())

    leaves = record["track_commitments"]["video"]["leaf_hashes"]
    tree = MerkleTree(leaves)
    assert tree.root == record["track_commitments"]["video"]["root"]
    assert verify_proof(tree.proof(0))

    # Recomputing the commitment from the media must reproduce the sealed root.
    media = ingest(tiny_corpus[0]["path"], cfg.canonical, workdir=tmp_path / "recheck").media
    assert commit_video(media.frames, media.fps, cfg.l4.chunk_seconds).root == tree.root


def test_exact_reupload_short_circuits_and_is_recorded(trained_bundle, tiny_corpus, tmp_path):
    bundle, _ = trained_bundle
    pipe, cfg = _pipeline(bundle, tmp_path)
    path = tiny_corpus[0]["path"]

    first = pipe.process(path, clip_id="first")
    second = pipe.process(path, clip_id="second")

    assert first.triage.action == "miss"
    assert second.triage.action == "exact_hit" and second.triage.short_circuit
    assert "L3.Tier1" in second.stages_skipped
    assert second.timings_s["total"] < first.timings_s["total"]
    assert any("inherited" in c.lower() for c in second.caveats)


def test_edge_tier_defers_instead_of_guessing(trained_bundle, tiny_corpus, tmp_path):
    bundle, _ = trained_bundle
    pipe, cfg = _pipeline(bundle, tmp_path, tier="edge")
    res = pipe.process(tiny_corpus[0]["path"], clip_id="edge_clip")

    assert res.verdict is Verdict.UNKNOWN
    assert "L3.Tier1" in res.stages_skipped
    assert "deferred_to_cloud" in res.fusion.flags
    assert any("edge" in c.lower() for c in res.caveats)


def test_integrity_clash_forces_escalation(trained_bundle, tiny_corpus, tmp_path):
    """A manifest/watermark contradiction must escalate regardless of detector scores."""
    bundle, _ = trained_bundle
    pipe, cfg = _pipeline(bundle, tmp_path)

    src = Path(tiny_corpus[0]["path"])
    clip = tmp_path / f"clash{src.suffix}"
    clip.write_bytes(src.read_bytes())
    (tmp_path / (clip.name + ".c2pa.json")).write_text(json.dumps(
        {"_simulated": True, "claims_ai_generated": False, "assertions": []}))
    (tmp_path / (clip.name + ".watermark.json")).write_text(json.dumps(
        {"_simulated": True, "watermark_detected": True, "confidence": 0.95}))

    res = pipe.process(clip, clip_id="clash_clip")
    assert res.provenance.integrity_clash
    assert "provenance_clash" in res.fusion.flags
    assert "L3.Tier2" in res.stages_run, "a clash must force Tier-2 escalation"


def test_verdict_summary_names_the_artefacts(trained_bundle, tiny_corpus, tmp_path):
    bundle, _ = trained_bundle
    pipe, cfg = _pipeline(bundle, tmp_path)
    fake = next((r for r in tiny_corpus if r["label_short"] == "FF"), tiny_corpus[0])
    res = pipe.process(fake["path"], clip_id="summary_clip")
    text = res.summary()
    assert "VERDICT:" in text
    assert "video:" in text and "audio:" in text
    # Either named artefacts, or an explicit statement that none passed threshold.
    assert ("artefacts found" in text) or ("no manipulation artefacts" in text)


def test_results_are_reproducible_across_runs(trained_bundle, tiny_corpus, tmp_path):
    bundle, _ = trained_bundle
    pipe_a, _ = _pipeline(bundle, tmp_path / "a")
    pipe_b, _ = _pipeline(bundle, tmp_path / "b")
    path = tiny_corpus[0]["path"]
    ra = pipe_a.process(path, clip_id="x")
    rb = pipe_b.process(path, clip_id="x")
    assert ra.verdict is rb.verdict
    assert abs(ra.fusion.p_video_fake - rb.fusion.p_video_fake) < 1e-9
    assert abs(ra.fusion.p_audio_fake - rb.fusion.p_audio_fake) < 1e-9
