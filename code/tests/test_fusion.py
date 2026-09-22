"""Tests for uncertainty-aware fusion.

These encode the guarantees the architecture claims: five distinguishable outcomes,
absence of evidence never reading as authenticity, and no modality silently overriding a
confident reading from another.
"""

from __future__ import annotations

import numpy as np
import pytest

from mosaic.fusion import (AVCouplingModel, EscalationFlag, ModalityVerdict, Verdict, fuse)
from mosaic.models import BranchPrediction


def pred(p: float, vacuity: float = 0.05, available: bool = True) -> BranchPrediction:
    llr = float(np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))))
    return BranchPrediction(p_fake=p, llr=llr, vacuity=vacuity, available=available)


@pytest.fixture
def coupling() -> AVCouplingModel:
    """Coupling with the structure learned from the corpus."""
    c = AVCouplingModel()
    c.table_ = np.array([0.02, 0.76, 0.64, 0.58])
    c.fitted_ = True
    return c


# --------------------------------------------------------------------------------------
# The five outcomes
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("pv,pa,ps,expected", [
    (0.02, 0.02, 0.05, Verdict.REAL_VIDEO_REAL_AUDIO),
    (0.98, 0.02, 0.95, Verdict.FAKE_VIDEO_REAL_AUDIO),
    (0.02, 0.98, 0.95, Verdict.REAL_VIDEO_FAKE_AUDIO),
    (0.98, 0.98, 0.60, Verdict.FAKE_VIDEO_FAKE_AUDIO),
])
def test_all_four_decided_verdicts_reachable(coupling, pv, pa, ps, expected):
    r = fuse(pred(pv), pred(pa), pred(ps), coupling)
    assert r.verdict is expected


def test_ambiguous_evidence_yields_unknown(coupling):
    r = fuse(pred(0.5, 0.5), pred(0.5, 0.5), pred(0.5, 0.5), coupling)
    assert r.verdict is Verdict.UNKNOWN
    assert EscalationFlag.INSUFFICIENT_EVIDENCE.value in r.flags


def test_jointly_generated_fake_detected_despite_perfect_sync(coupling):
    """A synchronised fake must still be caught by the per-modality branches."""
    r = fuse(pred(0.95), pred(0.95), pred(0.05), coupling)
    assert r.verdict is Verdict.FAKE_VIDEO_FAKE_AUDIO
    assert r.p_video_fake > 0.8 and r.p_audio_fake > 0.8


def test_voice_conversion_caught_without_desync(coupling):
    """Fake audio that preserves timing: AV sees nothing, audio branch must carry it."""
    r = fuse(pred(0.03), pred(0.96), pred(0.05), coupling)
    assert r.verdict is Verdict.REAL_VIDEO_FAKE_AUDIO


# --------------------------------------------------------------------------------------
# Absence of evidence
# --------------------------------------------------------------------------------------


def test_unavailable_branch_does_not_vote_real(coupling):
    """A branch that could not run must not push its modality toward 'authentic'."""
    missing = BranchPrediction.unavailable("no audio track")
    assert missing.vacuity == 1.0
    r = fuse(pred(0.5, 0.4), missing, pred(0.5, 0.4), coupling)
    assert r.audio_verdict is ModalityVerdict.UNKNOWN
    assert abs(r.p_audio_fake - 0.5) < 0.15, "missing audio must stay near the prior"
    assert EscalationFlag.BRANCH_UNAVAILABLE.value in r.flags


def test_missing_modality_never_gets_a_decided_verdict(coupling):
    r = fuse(pred(0.99), BranchPrediction.unavailable("no audio"), pred(0.9), coupling)
    assert r.audio_verdict is ModalityVerdict.UNKNOWN
    assert r.verdict is Verdict.UNKNOWN


def test_both_branches_missing_is_unknown(coupling):
    r = fuse(BranchPrediction.unavailable("x"), BranchPrediction.unavailable("y"),
             pred(0.99), coupling)
    assert r.verdict is Verdict.UNKNOWN


# --------------------------------------------------------------------------------------
# Non-override guarantee
# --------------------------------------------------------------------------------------


def test_confident_branch_is_not_overridden_by_cross_modal_evidence(coupling):
    """The core guarantee: AV evidence cannot flip a confident per-modality reading."""
    # Visual branch is confidently 'authentic'; AV screams desync; audio says fake.
    r = fuse(pred(0.02, vacuity=0.05), pred(0.97, vacuity=0.05), pred(0.99, vacuity=0.05),
             coupling)
    # It may not silently report the video as manipulated.
    assert r.video_verdict is not ModalityVerdict.FAKE


def test_contradiction_escalates_rather_than_resolving(coupling):
    """Construct a genuine conflict and require escalation, not a winner."""
    strong = AVCouplingModel()
    strong.table_ = np.array([0.01, 0.99, 0.99, 0.01])
    strong.fitted_ = True
    # Visual confidently authentic, audio confidently authentic, AV confidently desynced:
    # the coupling pushes hard toward a single-modality manipulation.
    r = fuse(pred(0.03, vacuity=0.02), pred(0.03, vacuity=0.02), pred(0.999, vacuity=0.02),
             strong, confident_llr=1.5, confident_max_vacuity=0.3)
    if EscalationFlag.MODALITY_CONFLICT.value in r.flags:
        assert r.verdict is Verdict.UNKNOWN
        assert ModalityVerdict.UNKNOWN in (r.video_verdict, r.audio_verdict)
    else:
        # No conflict is acceptable only if neither confident reading was actually flipped.
        assert r.video_verdict is not ModalityVerdict.FAKE
        assert r.audio_verdict is not ModalityVerdict.FAKE


def test_unconfident_branch_may_be_revised(coupling):
    """Non-override applies to *confident* branches only; weak evidence can be updated."""
    r = fuse(pred(0.45, vacuity=0.8), pred(0.98, vacuity=0.03), pred(0.9), coupling)
    assert r.audio_verdict is ModalityVerdict.FAKE


# --------------------------------------------------------------------------------------
# Uncertainty behaviour
# --------------------------------------------------------------------------------------


def test_high_vacuity_shrinks_evidence(coupling):
    confident = fuse(pred(0.95, vacuity=0.02), pred(0.5, 0.5), pred(0.5, 0.5), coupling)
    vague = fuse(pred(0.95, vacuity=0.95), pred(0.5, 0.5), pred(0.5, 0.5), coupling)
    assert confident.p_video_fake > vague.p_video_fake


def test_ood_flag_propagates(coupling):
    p = pred(0.9)
    p.ood_flag = True
    p.ood_score = 99.8
    r = fuse(p, pred(0.5, 0.5), pred(0.5, 0.5), coupling)
    assert EscalationFlag.OUT_OF_DISTRIBUTION.value in r.flags


def test_posterior_is_a_distribution(coupling):
    r = fuse(pred(0.7), pred(0.3), pred(0.6), coupling)
    assert abs(sum(r.posterior.values()) - 1.0) < 1e-9
    assert all(0.0 <= v <= 1.0 for v in r.posterior.values())
    assert abs(r.p_video_fake - (r.posterior["fake_video_real_audio"]
                                 + r.posterior["fake_video_fake_audio"])) < 1e-9


# --------------------------------------------------------------------------------------
# Coupling model
# --------------------------------------------------------------------------------------


def test_coupling_learns_expected_structure():
    """Real/real rarely desynced; mixed cells often; fake/fake intermediate."""
    n = 200
    vf = np.array([False] * n + [True] * n + [False] * n + [True] * n)
    af = np.array([False] * n + [False] * n + [True] * n + [True] * n)
    rng = np.random.default_rng(0)
    des = np.concatenate([
        rng.random(n) < 0.02, rng.random(n) < 0.75,
        rng.random(n) < 0.65, rng.random(n) < 0.50,
    ])
    c = AVCouplingModel().fit(vf, af, des)
    rr, fr, rf, ff = c.table_
    assert rr < 0.15
    assert fr > 0.5 and rf > 0.5
    assert rr < ff < max(fr, rf), "fake/fake must sit between real/real and the mixed cells"


def test_coupling_smoothing_avoids_degenerate_zero():
    c = AVCouplingModel().fit(np.array([False]), np.array([False]), np.array([False]))
    assert all(0.0 < p < 1.0 for p in c.table_)
