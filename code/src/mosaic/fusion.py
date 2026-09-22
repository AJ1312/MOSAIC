"""Uncertainty-aware fusion of the visual, audio and audiovisual branches.

The problem
-----------
Three branches produce evidence about two independent binary questions — is the video
manipulated, is the audio manipulated — and the third branch does not answer either of
them. Audiovisual desynchronisation says *something* is inconsistent without saying which
stream is at fault, and a jointly generated fake is perfectly synchronised, so AV evidence
is neither necessary nor sufficient for either modality.

The model
---------
Fusion is a four-cell factor graph over the joint state (V, A) in {real, fake}^2:

    score(v, a) = log P(v, a)                              prior
                + log P(visual evidence  | v)              visual branch
                + log P(audio evidence   | a)              audio branch
                + log P(AV evidence      | v, a)           coupling

The coupling factor is where the audiovisual branch enters, and it is a *marginalisation*
rather than a heuristic. The AV branch reports a calibrated probability that the streams
are desynchronised, ``p_desync``. The corpus supplies, for each cell, the rate at which
clips of that kind are actually desynchronised, ``C[v, a]``. Then

    P(AV evidence | v, a) = p_desync * C[v, a] + (1 - p_desync) * (1 - C[v, a])

``C`` is *learned*, not asserted. It comes out with a low value for real/real, high values
for the two single-modality manipulations, and an intermediate value for fake/fake —
which is exactly the structure that stops desynchronisation from being read as a verdict:
observing desync raises the posterior of the two mixed cells relative to both pure ones,
and leaves the question of *which* modality open for the per-modality branches to settle.

Uncertainty
-----------
Every branch probability is tempered toward 0.5 in proportion to that branch's vacuity
before it enters the product:

    p_eff = 0.5 + (p - 0.5) / (1 + vacuity_penalty * vacuity)

An unavailable branch has vacuity 1.0 and contributes almost nothing — never a vote for
"real". This is the mechanism by which "we could not check the audio" and "the audio is
authentic" stay distinguishable all the way to the verdict.

Non-override guarantee
----------------------
Even with a principled coupling term, a strong AV signal combined with a weak per-modality
signal can flip a marginal. That is acceptable when the per-modality branch is unsure and
unacceptable when it is confident. So after fusion, each modality is checked: if a branch
was confident (large |LLR|, low vacuity) and the fused marginal disagrees with it, the
result is not silently accepted. The verdict escalates to MODALITY_CONFLICT and reports
UNKNOWN for that modality, with both the branch's own reading and the fused reading
recorded. A contradiction between modalities is a finding to surface, not a tie for one
side to win.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .models import BranchPrediction

# --------------------------------------------------------------------------------------
# Verdict space
# --------------------------------------------------------------------------------------


class Verdict(str, Enum):
    REAL_VIDEO_REAL_AUDIO = "real_video_real_audio"
    FAKE_VIDEO_REAL_AUDIO = "fake_video_real_audio"
    REAL_VIDEO_FAKE_AUDIO = "real_video_fake_audio"
    FAKE_VIDEO_FAKE_AUDIO = "fake_video_fake_audio"
    UNKNOWN = "unknown_inconclusive"


class ModalityVerdict(str, Enum):
    REAL = "real"
    FAKE = "fake"
    UNKNOWN = "unknown"


#: Cell order used consistently throughout: (video_fake, audio_fake).
CELLS: tuple[tuple[bool, bool], ...] = ((False, False), (True, False), (False, True), (True, True))
CELL_VERDICTS = (
    Verdict.REAL_VIDEO_REAL_AUDIO,
    Verdict.FAKE_VIDEO_REAL_AUDIO,
    Verdict.REAL_VIDEO_FAKE_AUDIO,
    Verdict.FAKE_VIDEO_FAKE_AUDIO,
)
CELL_NAMES = tuple(v.value for v in CELL_VERDICTS)


class EscalationFlag(str, Enum):
    MODALITY_CONFLICT = "modality_conflict"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    OUT_OF_DISTRIBUTION = "out_of_distribution"
    BRANCH_UNAVAILABLE = "branch_unavailable"
    PROVENANCE_CLASH = "provenance_clash"
    DEFERRED_TO_CLOUD = "deferred_to_cloud"


# --------------------------------------------------------------------------------------
# Coupling model
# --------------------------------------------------------------------------------------


class AVCouplingModel:
    """P(desynchronised | video_fake, audio_fake), estimated from labelled data."""

    def __init__(self, smoothing: float = 2.0):
        self.smoothing = smoothing
        # Sensible defaults used only if the model is never fitted; overwritten by fit().
        self.table_ = np.array([0.05, 0.60, 0.60, 0.45])
        self.counts_: dict[str, dict[str, int]] = {}
        self.fitted_ = False

    def fit(self, video_fake: np.ndarray, audio_fake: np.ndarray,
            desynced: np.ndarray) -> "AVCouplingModel":
        video_fake = np.asarray(video_fake).astype(bool)
        audio_fake = np.asarray(audio_fake).astype(bool)
        desynced = np.asarray(desynced).astype(bool)
        table = []
        for vf, af in CELLS:
            m = (video_fake == vf) & (audio_fake == af)
            n = int(m.sum())
            k = int(desynced[m].sum())
            # Laplace smoothing so an unobserved cell does not become a hard 0 or 1.
            table.append((k + self.smoothing) / (n + 2 * self.smoothing))
            self.counts_[f"v{int(vf)}a{int(af)}"] = {"n": n, "desynced": k}
        self.table_ = np.array(table, dtype=np.float64)
        self.fitted_ = True
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "fitted": self.fitted_,
            "p_desync_given_cell": {n: round(float(p), 4) for n, p in zip(CELL_NAMES, self.table_)},
            "counts": self.counts_,
            "smoothing": self.smoothing,
        }


# --------------------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------------------


@dataclass
class FusionResult:
    verdict: Verdict
    video_verdict: ModalityVerdict
    audio_verdict: ModalityVerdict
    posterior: dict[str, float]
    p_video_fake: float
    p_audio_fake: float
    top_posterior: float
    margin: float                      # gap between best and second-best cell
    joint_vacuity: float
    flags: list[str] = field(default_factory=list)
    branch_summary: dict[str, Any] = field(default_factory=dict)
    coupling_used: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "video_verdict": self.video_verdict.value,
            "audio_verdict": self.audio_verdict.value,
            "posterior": {k: round(v, 6) for k, v in self.posterior.items()},
            "p_video_fake": round(self.p_video_fake, 6),
            "p_audio_fake": round(self.p_audio_fake, 6),
            "top_posterior": round(self.top_posterior, 6),
            "margin": round(self.margin, 6),
            "joint_vacuity": round(self.joint_vacuity, 6),
            "flags": list(self.flags),
            "branch_summary": self.branch_summary,
            "coupling_used": self.coupling_used,
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------------------


def _temper(p: float, vacuity: float, penalty: float) -> float:
    """Shrink a probability toward 0.5 in proportion to the branch's vacuity."""
    w = 1.0 / (1.0 + penalty * float(np.clip(vacuity, 0.0, 1.0)))
    return float(np.clip(0.5 + (p - 0.5) * w, 1e-6, 1 - 1e-6))


def fuse(
    visual: BranchPrediction,
    audio: BranchPrediction,
    av: BranchPrediction,
    coupling: AVCouplingModel,
    *,
    prior_fake_video: float = 0.5,
    prior_fake_audio: float = 0.5,
    vacuity_penalty: float = 3.0,
    decide_threshold: float = 0.55,
    confident_llr: float = 1.6,
    confident_max_vacuity: float = 0.30,
    modality_margin: float = 0.15,
    extra_flags: list[str] | None = None,
) -> FusionResult:
    """Fuse three branch predictions into a five-way verdict."""
    flags: list[str] = list(extra_flags or [])
    notes: list[str] = []

    p_v = _temper(visual.p_fake, visual.vacuity, vacuity_penalty)
    p_a = _temper(audio.p_fake, audio.vacuity, vacuity_penalty)
    p_s = _temper(av.p_fake, av.vacuity, vacuity_penalty)

    for pred, label in ((visual, "visual"), (audio, "audio"), (av, "audiovisual")):
        if not pred.available:
            flags.append(EscalationFlag.BRANCH_UNAVAILABLE.value)
            notes.append(f"{label} branch unavailable: {pred.reason_unavailable}")
        if pred.ood_flag:
            flags.append(EscalationFlag.OUT_OF_DISTRIBUTION.value)
            notes.append(
                f"{label} branch input is outside its training distribution "
                f"(Mahalanobis percentile {pred.ood_score:.0f}); its score is discounted"
            )

    # ---- joint scores ------------------------------------------------------------------
    prior = np.array([
        (1 - prior_fake_video) * (1 - prior_fake_audio),
        prior_fake_video * (1 - prior_fake_audio),
        (1 - prior_fake_video) * prior_fake_audio,
        prior_fake_video * prior_fake_audio,
    ])
    prior = prior / prior.sum()

    scores = np.zeros(4)
    for i, (vf, af) in enumerate(CELLS):
        s = np.log(prior[i] + 1e-12)
        s += np.log(p_v if vf else 1 - p_v)
        s += np.log(p_a if af else 1 - p_a)
        c = float(np.clip(coupling.table_[i], 1e-4, 1 - 1e-4))
        # Marginalise the latent synchronisation state.
        s += np.log(p_s * c + (1 - p_s) * (1 - c))
        scores[i] = s

    scores -= scores.max()
    post = np.exp(scores)
    post /= post.sum()

    p_video_fake = float(post[1] + post[3])
    p_audio_fake = float(post[2] + post[3])

    order = np.argsort(post)[::-1]
    top_idx = int(order[0])
    top_p = float(post[top_idx])
    margin = float(post[order[0]] - post[order[1]])

    joint_vacuity = float(np.mean([visual.vacuity, audio.vacuity, av.vacuity]))

    # ---- per-modality verdicts ----------------------------------------------------------
    # Each modality is decided from its own marginal with its own confidence band, rather
    # than inherited from the winning joint cell. The two questions are genuinely separate:
    # a clip can have confidently synthetic audio while the video remains undetermined, and
    # collapsing both to UNKNOWN whenever the *joint* state is unclear discards a finding
    # the audio branch actually made.
    def _modality(p: float) -> ModalityVerdict:
        if p >= 0.5 + modality_margin:
            return ModalityVerdict.FAKE
        if p <= 0.5 - modality_margin:
            return ModalityVerdict.REAL
        return ModalityVerdict.UNKNOWN

    video_verdict = _modality(p_video_fake)
    audio_verdict = _modality(p_audio_fake)

    conflict = False
    for pred, fused_p, name in (
        (visual, p_video_fake, "video"),
        (audio, p_audio_fake, "audio"),
    ):
        if not pred.available:
            continue
        confident = abs(pred.llr) >= confident_llr and pred.vacuity <= confident_max_vacuity
        if not confident:
            continue
        branch_says_fake = pred.p_fake >= 0.5
        fused_says_fake = fused_p >= 0.5
        if branch_says_fake != fused_says_fake:
            conflict = True
            notes.append(
                f"{name} branch is confident this modality is "
                f"{'manipulated' if branch_says_fake else 'authentic'} "
                f"(p={pred.p_fake:.3f}, vacuity={pred.vacuity:.2f}), but fusion with the "
                f"other modalities points the opposite way (p={fused_p:.3f}). "
                "Cross-modal evidence is not permitted to override a confident "
                "single-modality reading; this modality is reported UNKNOWN and escalated."
            )
            if name == "video":
                video_verdict = ModalityVerdict.UNKNOWN
            else:
                audio_verdict = ModalityVerdict.UNKNOWN

    if conflict:
        flags.append(EscalationFlag.MODALITY_CONFLICT.value)

    # ---- decision ----------------------------------------------------------------------
    # A modality whose own branch could not run must not inherit a verdict from the joint
    # posterior alone, however strong the other modalities are.
    if not visual.available:
        video_verdict = ModalityVerdict.UNKNOWN
    if not audio.available:
        audio_verdict = ModalityVerdict.UNKNOWN

    if conflict:
        verdict = Verdict.UNKNOWN
    elif not visual.available and not audio.available:
        verdict = Verdict.UNKNOWN
        flags.append(EscalationFlag.INSUFFICIENT_EVIDENCE.value)
        notes.append("both single-modality branches unavailable; no verdict is supportable")
    elif ModalityVerdict.UNKNOWN in (video_verdict, audio_verdict):
        # One modality is undetermined, so the *joint* state cannot be named — but the
        # other modality's verdict is still reported in its own field.
        verdict = Verdict.UNKNOWN
        flags.append(EscalationFlag.INSUFFICIENT_EVIDENCE.value)
        undecided = "video" if video_verdict is ModalityVerdict.UNKNOWN else "audio"
        notes.append(
            f"the {undecided} modality is undetermined "
            f"(P(manipulated)={p_video_fake if undecided == 'video' else p_audio_fake:.3f} "
            f"lies inside the +/-{modality_margin} indecision band), so no joint state can "
            "be named; the other modality's verdict is reported on its own marginal"
        )
    elif top_p < decide_threshold:
        verdict = Verdict.UNKNOWN
        flags.append(EscalationFlag.INSUFFICIENT_EVIDENCE.value)
        notes.append(
            f"no joint state reaches the decision threshold "
            f"(best {CELL_NAMES[top_idx]} at {top_p:.3f} < {decide_threshold})"
        )
    else:
        verdict = CELL_VERDICTS[top_idx]

    return FusionResult(
        verdict=verdict,
        video_verdict=video_verdict,
        audio_verdict=audio_verdict,
        posterior={n: float(p) for n, p in zip(CELL_NAMES, post)},
        p_video_fake=p_video_fake,
        p_audio_fake=p_audio_fake,
        top_posterior=top_p,
        margin=margin,
        joint_vacuity=joint_vacuity,
        flags=sorted(set(flags)),
        branch_summary={
            "visual": visual.to_dict(),
            "audio": audio.to_dict(),
            "audiovisual": av.to_dict(),
            "tempered": {"p_video_fake": p_v, "p_audio_fake": p_a, "p_desync": p_s},
        },
        coupling_used={n: float(c) for n, c in zip(CELL_NAMES, coupling.table_)},
        notes=notes,
    )
