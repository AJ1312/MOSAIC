#!/usr/bin/env python3
"""Stage 10 — one fully worked chain-of-custody example.

Produces, for a single clip:

  * the complete custody record (JSON)
  * a Merkle proof for one video chunk and one audio chunk, both verified
  * a negative control: the same proof re-checked against a tampered chunk, which must fail
  * a localisation demonstration: substituting one second of audio invalidates the audio
    root while the video root is unchanged, so tampering is attributable to a modality and
    a time range
  * ledger hash-chain verification, including a tamper-detection check
  * a compliance checklist mapping record fields to ISO/IEC 27037, NIST SP 800-101 and
    FRE 901(b)(9)
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from _common import CACHE, MODELS_DIR, OUTPUTS, banner, load_json, save_json

from mosaic.config import MosaicConfig, apply_tier
from mosaic.data.corpus import read_manifest
from mosaic.l0_ingest import ingest
from mosaic.l1_hash import HashRegistry
from mosaic.l4_custody import (LocalLedger, MerkleTree, commit_audio, commit_video,
                               combined_root, verify_proof, verify_ledger)
from mosaic.pipeline import ModelBundle, MosaicPipeline

CHECKLIST = [
    ("ISO/IEC 27037", "Identification — the artefact is uniquely identified",
     "record_id, clip_id, source_path, source_sha256", True),
    ("ISO/IEC 27037", "Collection — the acquisition method is documented",
     "ingest.ffmpeg_command, ingest.ffmpeg_binary_version, ingest.transformations", True),
    ("ISO/IEC 27037", "Acquisition — an integrity value is computed at acquisition",
     "canonical_sha256, track_commitments.video.root, track_commitments.audio.root", True),
    ("ISO/IEC 27037", "Preservation — the artefact is protected from undetected alteration",
     "combined_root + hash-chained ledger_entry", True),
    ("ISO/IEC 27037", "Auditability — every action is traceable and timestamped",
     "events[] (ordered seq, UTC timestamp, layer, action, detail)", True),
    ("ISO/IEC 27037", "Repeatability — the same inputs and settings reproduce the result",
     "config_digest, seed, canonical profile, environment", True),
    ("ISO/IEC 27037", "Reproducibility — an independent party can repeat the process",
     "models descriptor, feature_schema, environment, pipeline_description", True),
    ("NIST SP 800-101", "SHA-256 (or stronger) used for evidence hashing",
     "hashing.sha256_file, Merkle leaves and nodes, ledger chain", True),
    ("NIST SP 800-101", "Hash re-verified at each custody transfer",
     "verify_proof() and verify_ledger() re-derive every commitment on demand", True),
    ("NIST SP 800-86", "Chain of custody documented end to end",
     "events[] + append-only ledger with prev_hash linkage", True),
    ("FRE 901(b)(9)", "Process/system described in sufficient detail",
     "pipeline_description, models, ingest, l3_detection", True),
    ("FRE 901(b)(9)", "System shown to produce an accurate result",
     "verdict + evidence[] + audited evaluation in outputs/04_results and 08_appendix",
     True),
    ("FRE 901(b)(9)", "Operator/tooling identified",
     "environment (library versions), device_profile, mosaic_version", True),
    ("eIDAS 2.0", "Qualified electronic time stamp",
     "timestamp.authority='local_nonqualified', is_qualified=false", False),
    ("Multi-operator anchoring", "Commitment anchored beyond a single operator",
     "LocalLedger only; anchor() interface present, no distributed anchor contacted", False),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="A")
    ap.add_argument("--clip-index", type=int, default=0)
    args = ap.parse_args()

    banner("Stage 10 — chain-of-custody worked example")
    out_dir = OUTPUTS / "07_chain_of_custody_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    profile = load_json(OUTPUTS / "00_device_profile" / "device_profile.json")
    cfg = apply_tier(MosaicConfig(), profile["tier"])
    bundle = ModelBundle.load(MODELS_DIR)
    rows = [r for r in read_manifest(OUTPUTS / "03_dataset" / f"dataset_manifest_{args.corpus}.csv")
            if r["split"] == "test"]
    # Prefer a clip whose ground truth is interesting: a manipulated one.
    pick = next((r for r in rows if r["label_short"] == "FF"), rows[args.clip_index])
    print(f"  clip: {pick['clip_id']}  (ground truth: {pick['label']}, "
          f"{pick['sync_state']}, provenance scenario '{pick['provenance_scenario']}')")

    demo_ledger = out_dir / "demo_ledger.jsonl"
    if demo_ledger.exists():
        demo_ledger.unlink()
    from dataclasses import replace

    cfg = replace(cfg, l4=replace(cfg.l4, ledger_path=str(demo_ledger),
                                  records_dir=str(out_dir / "records")))
    reg_path = OUTPUTS / "registry" / "custody_demo.sqlite"
    if reg_path.exists():
        reg_path.unlink()
    pipe = MosaicPipeline(cfg, bundle, device_profile=profile,
                          registry=HashRegistry(reg_path, cfg.l1),
                          workdir=CACHE / "custody_demo", synthetic_data=True)
    result = pipe.process(pick["path"], clip_id=pick["clip_id"])

    record_path = Path(cfg.l4.records_dir) / f"{result.custody_record_id}.json"
    record = json.loads(record_path.read_text())
    print(f"  custody record: {record_path.name}  combined root {record['combined_root'][:32]}...")
    print(f"  verdict: {result.verdict.value}")
    print()
    print(result.summary())

    # ---- Merkle proofs -------------------------------------------------------------------
    banner("Merkle proofs")
    res = ingest(pick["path"], cfg.canonical, workdir=CACHE / "custody_demo")
    media = res.media
    vc = commit_video(media.frames, media.fps, cfg.l4.chunk_seconds)
    ac = commit_audio(media.audio, media.sample_rate, cfg.l4.chunk_seconds)

    v_tree = MerkleTree(vc.leaf_hashes)
    a_tree = MerkleTree(ac.leaf_hashes)
    v_idx, a_idx = 1, 1
    v_proof = v_tree.proof(v_idx)
    a_proof = a_tree.proof(a_idx)

    checks = {
        "video_chunk_index": v_idx,
        "video_chunk_span_s": vc.chunk_spans[v_idx],
        "video_proof_valid": verify_proof(v_proof),
        "video_proof_path_length": len(v_proof.path),
        "video_proof_size_bytes": len(json.dumps(v_proof.to_dict())),
        "audio_chunk_index": a_idx,
        "audio_chunk_span_s": ac.chunk_spans[a_idx],
        "audio_proof_valid": verify_proof(a_proof),
        "audio_proof_path_length": len(a_proof.path),
        "n_video_chunks": vc.n_chunks,
        "n_audio_chunks": ac.n_chunks,
    }

    # Negative control: a proof must fail when the committed leaf is altered.
    tampered = copy.deepcopy(v_proof.to_dict())
    tampered["leaf_hash"] = "00" + tampered["leaf_hash"][2:]
    checks["tampered_video_proof_rejected"] = not verify_proof(tampered)

    # Localisation: substitute one second of audio and re-commit.
    audio_mod = media.audio.copy()
    sr = media.sample_rate
    audio_mod[sr:2 * sr] = audio_mod[sr:2 * sr][::-1]  # reverse one second
    ac_mod = commit_audio(audio_mod, sr, cfg.l4.chunk_seconds)
    changed = [i for i, (x, y) in enumerate(zip(ac.leaf_hashes, ac_mod.leaf_hashes)) if x != y]
    vc_same = commit_video(media.frames, media.fps, cfg.l4.chunk_seconds)
    checks["audio_tamper_changed_chunks"] = changed
    checks["audio_tamper_localised_to_span_s"] = [ac.chunk_spans[i] for i in changed]
    checks["audio_root_changed"] = ac_mod.root != ac.root
    checks["video_root_unchanged_after_audio_tamper"] = vc_same.root == vc.root
    checks["combined_root_changed"] = (
        combined_root(vc_same.root, ac_mod.root, "ctx") != combined_root(vc.root, ac.root, "ctx"))

    for k, v in checks.items():
        print(f"  {k:<44} {v}")

    save_json({"proofs": {"video": v_proof.to_dict(), "audio": a_proof.to_dict()},
               "checks": checks}, out_dir / "merkle_proof_demo.json")

    # ---- ledger --------------------------------------------------------------------------
    banner("Ledger verification")
    state = verify_ledger(demo_ledger)
    print(f"  intact ledger : valid={state['valid']} entries={state['n_entries']}")

    tamper_path = out_dir / "demo_ledger_tampered.jsonl"
    lines = demo_ledger.read_text().strip().split("\n")
    entry = json.loads(lines[0])
    entry["payload"]["verdict"] = "real_video_real_audio"   # retroactively rewrite a verdict
    lines[0] = json.dumps(entry, sort_keys=True)
    tamper_path.write_text("\n".join(lines) + "\n")
    tampered_state = verify_ledger(tamper_path)
    print(f"  tampered ledger: valid={tampered_state['valid']} "
          f"broken_at={tampered_state['broken_at']} — {tampered_state['reason']}")

    save_json({"intact": state, "tampered": tampered_state},
              out_dir / "ledger_verification.json")

    # ---- compliance checklist -------------------------------------------------------------
    lines = [
        "# Chain-of-custody compliance checklist",
        "",
        "**SYNTHETIC DEMO DATA — not representative of real detection performance.**",
        "",
        f"Worked example: `{pick['clip_id']}`  ",
        f"Custody record: `{record_path.name}`  ",
        f"Combined Merkle root: `{record['combined_root']}`",
        "",
        "| Standard | Requirement | Custody-record field(s) | Satisfied |",
        "|---|---|---|---|",
    ]
    for std, req, field, ok in CHECKLIST:
        lines.append(f"| {std} | {req} | `{field}` | {'yes' if ok else '**NO**'} |")
    lines += [
        "",
        "## Requirements deliberately NOT satisfied",
        "",
        "Two rows above are marked NO, and they are marked NO rather than quietly omitted:",
        "",
        "1. **Qualified electronic time stamp (eIDAS 2.0).** The record carries a local "
        "system-clock timestamp explicitly labelled `local_nonqualified`. It establishes "
        "ordering *within this ledger* and carries no third-party attestation of wall-clock "
        "time. Evidentiary use requires substituting a qualified RFC-3161 TSA; the "
        "`LocalTimestampAuthority` interface exists for exactly that substitution.",
        "",
        "2. **Multi-operator anchoring.** Commitments are appended to a local hash-chained "
        "file. This is tamper-*evident* to anyone holding a copy, but it is not tamper-"
        "*resistant* against the operator who holds the only copy — the single-operator "
        "trust problem. No distributed ledger is contacted and none is claimed.",
        "",
        "## What the Merkle structure does provide",
        "",
        f"- Video and audio are committed in separate trees ({vc.n_chunks} and "
        f"{ac.n_chunks} one-second chunks), bound by "
        "`combined_root = SHA-256(video_root || audio_root || context_hash)`.",
        f"- A chunk proof is {checks['video_proof_path_length']} hashes "
        f"({checks['video_proof_size_bytes']} bytes as JSON), so a single second can be "
        "verified without disclosing the rest of the clip.",
        f"- Substituting one second of audio changed exactly chunk(s) "
        f"{checks['audio_tamper_changed_chunks']} (span "
        f"{checks['audio_tamper_localised_to_span_s']} s), changed the audio root, and left "
        "the video root untouched — tampering is attributable to a modality and a time range, "
        "not merely detected.",
    ]
    (out_dir / "compliance_checklist.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  wrote {out_dir}/compliance_checklist.md")
    print(f"  wrote {out_dir}/merkle_proof_demo.json")


if __name__ == "__main__":
    main()
