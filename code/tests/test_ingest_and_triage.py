"""Tests for L0 canonical ingest and L1 hash triage."""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

from conftest import requires_ffmpeg
from mosaic import FEATURE_SCHEMA_VERSION
from mosaic.config import CanonicalProfile, L1Config, MosaicConfig, apply_tier
from mosaic.hashing import audio_phash, video_phash
from mosaic.l0_ingest import IngestError, ingest, probe_summary, ffprobe
from mosaic.l1_hash import HashRegistry, RegistryEntry, triage


# --------------------------------------------------------------------------------------
# L0
# --------------------------------------------------------------------------------------


@requires_ffmpeg
def test_ingest_produces_canonical_profile(ingested):
    p = CanonicalProfile()
    probe = ingested.canonical_probe
    assert probe["width"] == p.width and probe["height"] == p.height
    assert probe["avg_frame_rate"] == p.fps
    assert probe["sample_rate"] == p.audio_sample_rate
    assert probe["channels"] == p.audio_channels


@requires_ffmpeg
def test_ingest_enforces_fixed_analysis_duration(ingested):
    """The K-frame filter: canonical duration must be exactly analysis_seconds.

    Regression test for a real leak. Normalising resolution/codec alone left clip length
    predicting the label at AUC 1.000 on the deliberately-leaky corpus; only fixing the
    duration closed it.
    """
    p = CanonicalProfile()
    assert abs(ingested.canonical_probe["duration_s"] - p.analysis_seconds) < 0.06
    media = ingested.media
    assert abs(media.n_frames - p.analysis_seconds * p.fps) <= 2
    assert abs(media.audio.size / media.sample_rate - p.analysis_seconds) < 0.06


@requires_ffmpeg
def test_ingest_decodes_all_three_signal_types(ingested):
    m = ingested.media
    assert m.frames.ndim == 4 and m.frames.shape[3] == 3
    assert m.has_audio and m.audio.size > 0
    # Motion vectors either present, or absent WITH a stated reason — never silently empty.
    assert m.motion_vectors.available or m.motion_vectors.reason_unavailable


@requires_ffmpeg
def test_ingest_records_command_and_hashes(ingested):
    assert "ffmpeg" in ingested.ffmpeg_command[0]
    assert len(ingested.source_sha256) == 64 and len(ingested.canonical_sha256) == 64
    assert ingested.source_sha256 != ingested.canonical_sha256
    assert any("duration->" in t for t in ingested.transformations)


@requires_ffmpeg
def test_canonicalisation_is_deterministic(sample_media, tmp_path):
    a = ingest(sample_media, CanonicalProfile(), workdir=tmp_path / "a", decode=False)
    b = ingest(sample_media, CanonicalProfile(), workdir=tmp_path / "b", decode=False)
    assert a.canonical_sha256 == b.canonical_sha256


@requires_ffmpeg
def test_differently_encoded_sources_converge_to_one_profile(sample_media, tmp_path):
    """Two different source encodes of the same content must canonicalise identically in shape."""
    variant = tmp_path / "variant.mkv"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i",
                    str(sample_media), "-c:v", "libx264", "-crf", "34", "-vf",
                    "scale=160:120", "-r", "20", "-c:a", "aac", str(variant)],
                   check=True, capture_output=True, timeout=300)
    a = ingest(sample_media, CanonicalProfile(), workdir=tmp_path / "wa")
    b = ingest(variant, CanonicalProfile(), workdir=tmp_path / "wb")
    for key in ("width", "height", "avg_frame_rate", "sample_rate", "channels"):
        assert a.canonical_probe[key] == b.canonical_probe[key]
    assert a.media.n_frames == b.media.n_frames


def test_ingest_rejects_missing_file(tmp_path):
    with pytest.raises(IngestError):
        ingest(tmp_path / "nope.mp4", CanonicalProfile())


# --------------------------------------------------------------------------------------
# Device tiers
# --------------------------------------------------------------------------------------


def test_edge_tier_disables_expensive_stages():
    edge = apply_tier(MosaicConfig(), "edge")
    cloud = apply_tier(MosaicConfig(), "cloud")
    assert edge.l3.tier2_enabled is False
    assert cloud.l3.tier2_enabled is True
    assert edge.l3.visual_frames < cloud.l3.visual_frames


def test_tier_changes_config_digest():
    assert apply_tier(MosaicConfig(), "edge").digest() != apply_tier(MosaicConfig(), "cloud").digest()


def test_unknown_tier_rejected():
    with pytest.raises(ValueError):
        apply_tier(MosaicConfig(), "quantum")


# --------------------------------------------------------------------------------------
# L1 triage
# --------------------------------------------------------------------------------------


@pytest.fixture
def registry(tmp_path) -> HashRegistry:
    return HashRegistry(tmp_path / "reg.sqlite", L1Config())


def _entry(sha: str, clip_id: str, v_ph, a_ph, digest="cfg") -> RegistryEntry:
    return RegistryEntry(canonical_sha256=sha, source_sha256=sha, clip_id=clip_id,
                         verdict="real_video_real_audio", verdict_payload={},
                         video_phash=v_ph, audio_phash=a_ph,
                         feature_schema=FEATURE_SCHEMA_VERSION, config_digest=digest)


def test_miss_on_empty_registry(registry):
    r = triage(registry, "a" * 64, [1, 2, 3], [4, 5],
               feature_schema=FEATURE_SCHEMA_VERSION, config_digest="cfg")
    assert r.action == "miss" and not r.short_circuit


def test_exact_hit_short_circuits(registry):
    registry.put(_entry("b" * 64, "clip1", [1, 2, 3], [4, 5]))
    r = triage(registry, "b" * 64, [1, 2, 3], [4, 5],
               feature_schema=FEATURE_SCHEMA_VERSION, config_digest="cfg")
    assert r.action == "exact_hit" and r.short_circuit
    assert r.exact_match.clip_id == "clip1"


def test_stale_config_forces_reanalysis(registry):
    """A cached verdict from another configuration is not a verdict from this one."""
    registry.put(_entry("c" * 64, "clip1", [1, 2, 3], [4, 5], digest="OLD"))
    r = triage(registry, "c" * 64, [1, 2, 3], [4, 5],
               feature_schema=FEATURE_SCHEMA_VERSION, config_digest="NEW")
    assert r.action == "exact_hit"
    assert not r.short_circuit
    assert "different configuration" in r.reason


def test_stale_feature_schema_forces_reanalysis(registry):
    entry = _entry("d" * 64, "clip1", [1, 2, 3], [4, 5])
    entry.feature_schema = "av-0.0.1-old"
    registry.put(entry)
    r = triage(registry, "d" * 64, [1, 2, 3], [4, 5],
               feature_schema=FEATURE_SCHEMA_VERSION, config_digest="cfg")
    assert not r.short_circuit


def test_perceptual_hit_does_not_short_circuit_by_default(registry, rng):
    base = [int(x) for x in rng.integers(0, 2**63, 16)]
    near = list(base)
    near[0] ^= 0b11          # two bits different out of 1024
    registry.put(_entry("e" * 64, "clip1", base, []))
    r = triage(registry, "f" * 64, near, [],
               feature_schema=FEATURE_SCHEMA_VERSION, config_digest="cfg")
    assert r.action == "perceptual_candidate"
    assert not r.short_circuit, "perceptual similarity is retrieval, never proof"
    assert r.candidates and r.candidates[0]["clip_id"] == "clip1"
    assert "not proof" in r.reason


def test_perceptual_short_circuit_only_when_explicitly_enabled(tmp_path, rng):
    cfg = L1Config(allow_perceptual_shortcircuit=True,
                   perceptual_shortcircuit_threshold=0.05)
    reg = HashRegistry(tmp_path / "r2.sqlite", cfg)
    base = [int(x) for x in rng.integers(0, 2**63, 16)]
    near = list(base)
    near[0] ^= 0b1
    reg.put(_entry("e" * 64, "clip1", base, []))
    r = triage(reg, "f" * 64, near, [], feature_schema=FEATURE_SCHEMA_VERSION,
               config_digest="cfg", config=cfg)
    assert r.short_circuit
    assert "similarity inference, not proof of identity" in r.reason


def test_registry_roundtrip_preserves_hashes(registry):
    v = [111, 222, 333]
    registry.put(_entry("a" * 64, "clip1", v, [7, 8]))
    got = registry.get_exact("a" * 64)
    assert got.video_phash == v and got.audio_phash == [7, 8]
    assert registry.size() == 1
