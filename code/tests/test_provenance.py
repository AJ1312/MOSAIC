"""Tests for L2.

These encode the three rules the layer exists to enforce: absence is not authenticity,
an unavailable detector never reports a negative finding, and contradictions escalate.
"""

from __future__ import annotations

import json

import pytest

from mosaic.config import L2Config
from mosaic.l2_provenance import (C2PAStatus, SidecarSimulatedDetector, UnavailableVendorDetector,
                                  WatermarkStatus, check_provenance, default_registry,
                                  read_c2pa)
from conftest import requires_ffmpeg


@requires_ffmpeg
def test_unsigned_media_reports_no_manifest_not_authentic(sample_media):
    res = read_c2pa(sample_media, L2Config(allow_sidecar_manifest=False))
    assert res.status in (C2PAStatus.NO_MANIFEST, C2PAStatus.FORMAT_UNSUPPORTED)
    assert res.claims_ai_generated is None
    assert not res.is_adverse, "absence of a manifest is not an adverse finding"


@requires_ffmpeg
def test_absent_provenance_produces_an_explicit_note(sample_media):
    res = check_provenance(sample_media, L2Config(watermark_detectors=("synthid",)))
    # A note must exist that explicitly denies any authenticity implication.
    assert any("authentic" in n.lower() and "no implication" in n.lower() for n in res.notes), \
        res.notes
    assert not res.escalate, "missing provenance is not, by itself, grounds for escalation"


def test_unavailable_detector_never_reports_not_detected(tmp_path):
    det = UnavailableVendorDetector("synthid", "gated behind a vendor surface")
    ok, reason = det.available()
    assert not ok
    result = det.detect(tmp_path / "nonexistent.mp4")
    assert result.status is WatermarkStatus.UNAVAILABLE
    assert result.status is not WatermarkStatus.NOT_DETECTED


def test_default_registry_marks_real_vendors_unavailable():
    reg = default_registry()
    for name in ("synthid", "c2pa_soft_binding"):
        ok, reason = reg[name].available()
        assert not ok and reason


def test_sidecar_detector_without_sidecar_is_unavailable_not_negative(tmp_path):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"not really a video")
    res = SidecarSimulatedDetector().detect(media)
    assert res.status is WatermarkStatus.UNAVAILABLE
    assert "NOT a finding" in res.detail or "not" in res.detail.lower()


def _write_sidecars(media, *, claims_ai=None, watermark=None):
    if claims_ai is not None:
        (media.parent / (media.name + ".c2pa.json")).write_text(json.dumps({
            "_simulated": True, "claims_ai_generated": claims_ai,
            "claim_generator": "test", "assertions": [],
        }))
    if watermark is not None:
        (media.parent / (media.name + ".watermark.json")).write_text(json.dumps({
            "_simulated": True, "watermark_detected": watermark, "confidence": 0.9,
            "modality": "both", "detector": "simulated_vendor_detector",
        }))


@requires_ffmpeg
def test_integrity_clash_detected_when_manifest_claims_human_but_watermark_says_ai(
        sample_media, tmp_path):
    media = tmp_path / "clash.mp4"
    media.write_bytes(sample_media.read_bytes())
    _write_sidecars(media, claims_ai=False, watermark=True)
    res = check_provenance(media, L2Config(watermark_detectors=("sidecar_simulated",)))
    assert res.integrity_clash
    assert res.escalate
    assert "INTEGRITY CLASH" in res.clash_detail


@requires_ffmpeg
def test_integrity_clash_detected_in_reverse_direction(sample_media, tmp_path):
    media = tmp_path / "clash2.mp4"
    media.write_bytes(sample_media.read_bytes())
    _write_sidecars(media, claims_ai=True, watermark=False)
    res = check_provenance(media, L2Config(watermark_detectors=("sidecar_simulated",)))
    assert res.integrity_clash and res.escalate


@requires_ffmpeg
def test_agreeing_signals_do_not_clash(sample_media, tmp_path):
    media = tmp_path / "agree.mp4"
    media.write_bytes(sample_media.read_bytes())
    _write_sidecars(media, claims_ai=True, watermark=True)
    res = check_provenance(media, L2Config(watermark_detectors=("sidecar_simulated",)))
    assert not res.integrity_clash
    # Still escalates: the manifest asserts AI generation.
    assert res.escalate


@requires_ffmpeg
def test_simulated_signals_are_labelled(sample_media, tmp_path):
    media = tmp_path / "sim.mp4"
    media.write_bytes(sample_media.read_bytes())
    _write_sidecars(media, claims_ai=False, watermark=True)
    res = check_provenance(media, L2Config(watermark_detectors=("sidecar_simulated",)))
    assert any("SIMULATED" in n for n in res.notes)
    assert all(w.is_simulated for w in res.watermarks if w.status is WatermarkStatus.DETECTED)


def test_unknown_detector_name_is_reported_not_silently_skipped(tmp_path):
    media = tmp_path / "x.mp4"
    media.write_bytes(b"stub")
    res = check_provenance(media, L2Config(watermark_detectors=("does_not_exist",)))
    assert len(res.watermarks) == 1
    assert res.watermarks[0].status is WatermarkStatus.UNAVAILABLE
