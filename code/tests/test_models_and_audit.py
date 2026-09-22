"""Tests for the branch models, evidence attribution, and the audit protocol."""

from __future__ import annotations

import numpy as np
import pytest

from mosaic.audit import (bootstrap_ci, c1_canonical_check, c2_leakage_audit,
                          c3_coherence_probe, c5_multiseed, expected_calibration_error,
                          matched_readout, recall_at_fpr, roc_auc)
from mosaic.evidence import build_branch_findings, build_report
from mosaic.models import AudioCNN, BranchModel, BranchPrediction


@pytest.fixture(scope="module")
def separable_data():
    """Two classes separable on the first few of 12 features."""
    rng = np.random.default_rng(0)
    n = 240
    X = rng.normal(0, 1, (2 * n, 12))
    X[n:, 0] += 2.0
    X[n:, 1] += 1.2
    y = np.array([0] * n + [1] * n)
    idx = rng.permutation(2 * n)
    return X[idx], y[idx]


@pytest.fixture(scope="module")
def fitted_model(separable_data):
    X, y = separable_data
    names = [f"f{i}" for i in range(X.shape[1])]
    cut = int(0.7 * len(y))
    return BranchModel("test", names, seed=0).fit(X[:cut], y[:cut], X[cut:], y[cut:]), X, y


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def test_roc_auc_extremes():
    y = np.array([False, False, True, True])
    assert roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), y) == 1.0
    assert roc_auc(np.array([0.9, 0.8, 0.2, 0.1]), y) == 0.0
    assert roc_auc(np.array([0.5, 0.5, 0.5, 0.5]), y) == 0.5


def test_roc_auc_handles_single_class():
    assert roc_auc(np.array([0.1, 0.9]), np.array([True, True])) == 0.5


def test_recall_at_fpr_is_conservative():
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.normal(0, 1, 2000), rng.normal(6, 1, 2000)])
    labels = np.array([False] * 2000 + [True] * 2000)
    rec, thr = recall_at_fpr(scores, labels, 0.001)
    assert 0.0 <= rec <= 1.0
    rec_loose, _ = recall_at_fpr(scores, labels, 0.05)
    assert rec_loose >= rec, "a looser FPR budget cannot reduce recall"


def test_ece_zero_for_perfect_calibration():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    y = rng.random(20000) < p
    assert expected_calibration_error(p, y) < 0.02


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.normal(0, 1, 200), rng.normal(1.5, 1, 200)])
    labels = np.array([False] * 200 + [True] * 200)
    point, lo, hi = bootstrap_ci(scores, labels, roc_auc, n_boot=300, seed=1)
    assert lo <= point <= hi


# --------------------------------------------------------------------------------------
# Branch model
# --------------------------------------------------------------------------------------


def test_model_separates_and_calibrates(fitted_model):
    model, X, y = fitted_model
    cut = int(0.7 * len(y))
    p = model.predict_proba(X[cut:])
    assert roc_auc(p, y[cut:].astype(bool)) > 0.85
    assert expected_calibration_error(p, y[cut:]) < 0.25


def test_prediction_fields_are_consistent(fitted_model):
    model, X, y = fitted_model
    pred = model.predict(X[0])
    assert pred.available
    assert 0.0 <= pred.p_fake <= 1.0
    assert 0.0 <= pred.vacuity <= 1.0
    assert np.isfinite(pred.llr)
    assert len(pred.contributions) == X.shape[1]


def test_contributions_sum_direction_matches_score(fitted_model):
    """Contributions are the actual log-odds terms, not a post-hoc rationalisation."""
    model, X, y = fitted_model
    fake_like = X[y == 1][0]
    pred = model.predict(fake_like)
    total = sum(c.contribution for c in pred.contributions)
    assert (total > 0) == (pred.p_fake > 0.5)


def test_out_of_distribution_input_raises_vacuity(fitted_model):
    model, X, y = fitted_model
    normal = model.predict(X[0])
    far = model.predict(X[0] + 60.0)
    assert far.ood_score >= normal.ood_score
    assert far.ood_flag
    assert far.vacuity > normal.vacuity


def test_wrong_feature_count_is_reported_not_crashed(fitted_model):
    model, X, y = fitted_model
    pred = model.predict(np.zeros(3))
    assert not pred.available and "expects" in pred.reason_unavailable


def test_nonfinite_features_rejected(fitted_model):
    model, X, y = fitted_model
    bad = X[0].copy()
    bad[2] = np.nan
    assert not model.predict(bad).available


def test_unfitted_model_raises():
    with pytest.raises(RuntimeError):
        BranchModel("x", ["a", "b"]).predict(np.zeros(2))


def test_single_class_training_rejected():
    with pytest.raises(ValueError):
        BranchModel("x", ["a"]).fit(np.zeros((10, 1)), np.zeros(10))


def test_model_save_load_roundtrip(fitted_model, tmp_path):
    model, X, y = fitted_model
    model.save(tmp_path / "m.pkl")
    loaded = BranchModel.load(tmp_path / "m.pkl")
    assert np.allclose(loaded.predict_proba(X[:20]), model.predict_proba(X[:20]))


def test_unavailable_prediction_is_neutral_with_total_vacuity():
    p = BranchPrediction.unavailable("no audio track")
    assert p.p_fake == 0.5 and p.llr == 0.0 and p.vacuity == 1.0
    assert not p.available


# --------------------------------------------------------------------------------------
# Audio CNN
# --------------------------------------------------------------------------------------


def test_audio_cnn_trains_and_predicts():
    rng = np.random.default_rng(0)
    specs, y = [], []
    for i in range(40):
        fake = i % 2 == 0
        s = rng.normal(0, 1, (64, 128)).astype(np.float32)
        if fake:
            s[40:] -= 3.0        # a band-limitation-like pattern
        specs.append(s)
        y.append(fake)
    cnn = AudioCNN(n_mels=64, seed=0).fit(specs, np.array(y), epochs=25)
    p = cnn.predict_proba(specs)
    assert p.shape == (40,)
    assert ((p >= 0.5) == np.array(y)).mean() > 0.7


def test_audio_cnn_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    specs = [rng.normal(0, 1, (64, 128)).astype(np.float32) for _ in range(20)]
    y = np.array([i % 2 == 0 for i in range(20)])
    cnn = AudioCNN(n_mels=64, seed=0).fit(specs, y, epochs=5)
    cnn.save(tmp_path / "cnn.pt")
    loaded = AudioCNN.load(tmp_path / "cnn.pt")
    assert loaded.fitted_
    # Must run without a device mismatch after loading onto CPU.
    assert np.allclose(loaded.predict_proba(specs), cnn.predict_proba(specs), atol=1e-4)


# --------------------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------------------


def test_findings_are_named_and_ranked(fitted_model):
    model, X, y = fitted_model
    pred = model.predict(X[y == 1][0])
    findings = build_branch_findings(pred, "visual", model.train_stats_, top_k=3)
    assert findings
    assert len(findings) <= 3
    mags = [abs(f.contribution_logodds) for f in findings]
    assert mags == sorted(mags, reverse=True)
    assert all(f.description for f in findings)
    assert all(f.direction in ("supports_manipulated", "supports_authentic") for f in findings)


def test_unavailable_branch_yields_no_findings_but_a_note():
    missing = BranchPrediction.unavailable("no audio track")
    assert build_branch_findings(missing, "audio", {}) == []
    report = build_report(missing, missing, missing, {})
    assert any("no findings" in n for n in report.notes)


# --------------------------------------------------------------------------------------
# Audit controls
# --------------------------------------------------------------------------------------


def test_c1_detects_inconsistent_profiles():
    good = [{"width": 256, "height": 256, "avg_frame_rate": 25.0, "video_codec": "h264",
             "pix_fmt": "yuv420p", "audio_codec": "pcm_s16le", "sample_rate": 16000,
             "channels": 1}] * 3
    assert c1_canonical_check(good).passed
    bad = good + [dict(good[0], width=320)]
    result = c1_canonical_check(bad)
    assert not result.passed and "video.width" in result.numbers["inconsistent_fields"]


def test_c1_tolerates_genuinely_single_modality_clips():
    """Audio-only and video-only clips in one corpus must not be read as a profile violation.

    Real corpora are not uniformly multimodal: ASVspoof ships audio-only FLAC and
    AIGVDBench ships silent video. C1 asks whether canonicalisation was applied
    consistently, not whether every file contains every stream.
    """
    video_only = {"width": 256, "height": 256, "avg_frame_rate": 25.0,
                  "video_codec": "h264", "pix_fmt": "yuv420p",
                  "audio_codec": None, "sample_rate": None, "channels": None}
    audio_only = {"width": None, "height": None, "avg_frame_rate": None,
                  "video_codec": None, "pix_fmt": None,
                  "audio_codec": "pcm_s16le", "sample_rate": 16000, "channels": 1}
    result = c1_canonical_check([video_only] * 3 + [audio_only] * 5)
    assert result.passed, result.detail
    assert result.numbers["n_with_video"] == 3
    assert result.numbers["n_with_audio"] == 5


def test_c1_still_catches_violation_within_one_modality():
    audio_only = {"video_codec": None, "audio_codec": "pcm_s16le",
                  "sample_rate": 16000, "channels": 1}
    mixed = [audio_only] * 3 + [dict(audio_only, sample_rate=44100)]
    result = c1_canonical_check(mixed)
    assert not result.passed
    assert "audio.sample_rate" in result.numbers["inconsistent_fields"]


def test_c2_flags_a_leaky_feature():
    rng = np.random.default_rng(0)
    n = 200
    y = np.array([False] * n + [True] * n)
    leaky = np.where(y, 3.0, 0.0)[:, None] + rng.normal(0, 0.2, (2 * n, 1))
    clean = rng.normal(0, 1, (2 * n, 1))
    idx = rng.permutation(2 * n)
    tr, te = idx[:250], idx[250:]
    assert not c2_leakage_audit({"leaky": leaky}, y, tr, te).passed
    assert c2_leakage_audit({"clean": clean}, y, tr, te).passed


def test_c3_floor_near_chance_for_unstructured_features():
    rng = np.random.default_rng(0)
    result = c3_coherence_probe(rng.normal(0, 1, (200, 10)), n_splits=5)
    assert result.passed
    assert 0.3 < result.numbers["floor_auc"] < 0.7


def test_c5_reports_multiple_seeds():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (300, 6))
    y = X[:, 0] + rng.normal(0, 0.5, 300) > 0
    result = c5_multiseed(X, y)
    assert result.passed and result.numbers["n_seeds"] >= 5
    assert result.numbers["std_auc"] >= 0.0


def test_matched_readout_is_deterministic():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (200, 5))
    y = (X[:, 0] > 0).astype(int)
    a = matched_readout(X[:150], y[:150], X[150:])
    b = matched_readout(X[:150], y[:150], X[150:])
    assert np.allclose(a, b)
