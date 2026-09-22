"""The six-control audit protocol (after arXiv:2606.31004, "VidAudit").

The audit paper's own survey of 20 recent detection papers found that 2/20 applied C1,
1/20 applied C2 or C5, and 0/20 applied C3. Those omissions are how a 3-feature
clip-length classifier reaches 0.998 LOGO-AUC on a widely-used benchmark while measuring
nothing about content. This module runs all six controls and reports them whether they
pass or fail.

======  =========================================================================
C1      Canonical re-encode. Verify every clip really was normalised to one
        profile, and that container metadata no longer predicts the label.
C2      Leakage audit. Train deliberately trivial classifiers — duration alone,
        container metadata alone, file size alone — that measure nothing about
        content. They MUST fail. Reported even when they do.
C3      Real-vs-real coherence probe. Split authentic clips into two arbitrary
        halves and try to separate them with the same model and features. The
        resulting AUC is the *floor*: any real detector must be scored as a
        margin above it, not against 0.5.
C4      Matched harness. Every compared method is re-fitted through one fixed
        L2-regularised logistic readout, so differences reflect features rather
        than tuning effort.
C5      Multi-seed bootstrap confidence intervals over >= 5 seeds.
C6      Cross-dataset validation on an independently parameterised corpus.
======  =========================================================================

The reported headline is the *audited tuple* — AUC, above-floor margin,
recall@FPR=0.1%, and calibration error — never a bare AUC.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np
from sklearn.linear_model import LogisticRegression

# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC. Returns 0.5 for a degenerate single-class input."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks within ties so tied scores cannot inflate the statistic.
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size)
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def recall_at_fpr(scores: np.ndarray, labels: np.ndarray, target_fpr: float = 0.001
                  ) -> tuple[float, float]:
    """Recall at a target false-positive rate, plus the threshold achieving it.

    The deployable operating point. AUC averages over thresholds nobody would ship.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    neg = scores[~labels]
    pos = scores[labels]
    if neg.size == 0 or pos.size == 0:
        return 0.0, float("nan")
    # Threshold at the (1 - target_fpr) quantile of the negative distribution.
    thr = float(np.quantile(neg, 1.0 - target_fpr))
    return float((pos > thr).mean()), thr


def equal_error_rate(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Equal error rate and the threshold achieving it.

    EER is the primary metric in the anti-spoofing literature (ASVspoof reports it
    throughout), so audio results are reported in EER as well as AUC to be comparable with
    published baselines. ``labels`` is True for the spoof/fake class.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    if labels.all() or (~labels).all():
        return float("nan"), float("nan")
    order = np.argsort(-scores)
    s = scores[order]
    y = labels[order]
    # Sweeping thresholds down the sorted scores: at each cut, everything above is called fake.
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    frr = 1.0 - tp / n_pos          # false rejection of spoof (miss rate)
    far = fp / n_neg                # false acceptance of bonafide as spoof
    idx = int(np.nanargmin(np.abs(far - frr)))
    return float((far[idx] + frr[idx]) / 2.0), float(s[idx])


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    probs = np.clip(np.asarray(probs, dtype=np.float64), 0, 1)
    labels = np.asarray(labels).astype(float)
    if probs.size == 0:
        return float("nan")
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (probs > edges[i]) & (probs <= edges[i + 1]) if i > 0 else (probs <= edges[1])
        if not m.any():
            continue
        ece += m.mean() * abs(labels[m].mean() - probs[m].mean())
    return float(ece)


def bootstrap_ci(scores: np.ndarray, labels: np.ndarray, stat: Callable = roc_auc,
                 n_boot: int = 1000, seed: int = 0, alpha: float = 0.05
                 ) -> tuple[float, float, float]:
    """(point estimate, lower, upper) percentile bootstrap CI."""
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    point = stat(scores, labels)
    rng = np.random.default_rng(seed)
    n = len(scores)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(labels[idx])) < 2:
            continue
        vals.append(stat(scores[idx], labels[idx]))
    if not vals:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


@dataclass
class AuditedTuple:
    """The full reportable result for one method on one task."""

    method: str
    task: str
    n: int
    auc: float
    auc_ci_low: float
    auc_ci_high: float
    floor_auc: float
    above_floor_margin: float
    recall_at_fpr_001pct: float
    ece: float
    accuracy: float
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audited_tuple(method: str, task: str, probs: np.ndarray, labels: np.ndarray,
                  floor_auc: float, *, seed: int = 0, notes: str = "") -> AuditedTuple:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    auc, lo, hi = bootstrap_ci(probs, labels, roc_auc, seed=seed)
    rec, _ = recall_at_fpr(probs, labels, 0.001)
    return AuditedTuple(
        method=method, task=task, n=int(len(labels)), auc=auc, auc_ci_low=lo, auc_ci_high=hi,
        floor_auc=floor_auc, above_floor_margin=auc - floor_auc,
        recall_at_fpr_001pct=rec, ece=expected_calibration_error(probs, labels),
        accuracy=float(((probs >= 0.5) == labels).mean()), notes=notes,
    )


# --------------------------------------------------------------------------------------
# The matched readout (C4)
# --------------------------------------------------------------------------------------


def matched_readout(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                    *, C: float = 0.25, seed: int = 0) -> np.ndarray:
    """One fixed L2-regularised logistic readout, used identically for every method.

    Differences between compared feature sets then reflect the features, not how much
    effort went into tuning each one.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd < 1e-9] = 1.0
    # L2 is the lbfgs default; passing penalty="l2" explicitly is deprecated in
    # scikit-learn >= 1.8.
    clf = LogisticRegression(C=C, solver="lbfgs", max_iter=2000,
                             class_weight="balanced", random_state=seed)
    clf.fit((X_train - mu) / sd, np.asarray(y_train).astype(int))
    return clf.predict_proba((X_test - mu) / sd)[:, 1]


# --------------------------------------------------------------------------------------
# C1 — canonical re-encode
# --------------------------------------------------------------------------------------


@dataclass
class ControlResult:
    control: str
    name: str
    passed: bool
    detail: str
    numbers: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


VIDEO_PROFILE_KEYS = ("width", "height", "avg_frame_rate", "video_codec", "pix_fmt")
AUDIO_PROFILE_KEYS = ("audio_codec", "sample_rate", "channels")


def c1_canonical_check(canonical_probes: list[dict[str, Any]]) -> ControlResult:
    """Verify every canonicalised clip shares one profile, per stream that exists.

    Uniformity is checked *within the clips that actually carry each stream*. Real corpora
    are not uniformly multimodal — ASVspoof ships audio-only FLAC, AIGVDBench ships video
    with no audio track — and comparing a video field across an audio-only clip would
    report a profile violation where there is simply no video stream to normalise. The
    control is about whether canonicalisation was applied consistently, not about whether
    every file happens to contain every modality.
    """
    def uniformity(probes: list[dict[str, Any]], keys) -> tuple[dict, dict]:
        observed: dict[str, set] = {k: set() for k in keys}
        for p in probes:
            for k in keys:
                observed[k].add(p.get(k))
        bad = {k: sorted(str(x) for x in v) for k, v in observed.items() if len(v) > 1}
        prof = {k: (sorted(str(x) for x in v)[0] if len(v) == 1 else None)
                for k, v in observed.items()}
        return bad, prof

    with_video = [p for p in canonical_probes if p.get("video_codec")]
    with_audio = [p for p in canonical_probes if p.get("audio_codec")]
    bad_v, prof_v = uniformity(with_video, VIDEO_PROFILE_KEYS) if with_video else ({}, {})
    bad_a, prof_a = uniformity(with_audio, AUDIO_PROFILE_KEYS) if with_audio else ({}, {})
    inconsistent = {**{f"video.{k}": v for k, v in bad_v.items()},
                    **{f"audio.{k}": v for k, v in bad_a.items()}}
    passed = not inconsistent
    return ControlResult(
        control="C1", name="canonical re-encode",
        passed=passed,
        detail=(f"canonical profile uniform across {len(with_video)} clips with video and "
                f"{len(with_audio)} with audio" if passed else
                f"canonical profile is not uniform: {inconsistent}"),
        numbers={"n_clips": len(canonical_probes), "n_with_video": len(with_video),
                 "n_with_audio": len(with_audio),
                 "video_profile": prof_v, "audio_profile": prof_a,
                 "inconsistent_fields": inconsistent},
    )


# --------------------------------------------------------------------------------------
# C2 — leakage audit
# --------------------------------------------------------------------------------------


def c2_leakage_audit(
    trivial_features: dict[str, np.ndarray],
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    threshold: float = 0.65,
    seed: int = 0,
) -> ControlResult:
    """Train content-blind baselines that must fail.

    Each entry of ``trivial_features`` is a feature matrix that measures nothing about the
    content — duration, container metadata, file size. If any of them separates the classes,
    the corpus (or the pipeline feeding it) leaks, and every downstream number is suspect.
    """
    labels = np.asarray(labels).astype(bool)
    results: dict[str, float] = {}
    for name, X in trivial_features.items():
        X = np.asarray(X, dtype=np.float64).reshape(len(labels), -1)
        probs = matched_readout(X[train_idx], labels[train_idx], X[test_idx], seed=seed)
        results[name] = roc_auc(probs, labels[test_idx])

    worst = max(results.items(), key=lambda kv: abs(kv[1] - 0.5))
    passed = abs(worst[1] - 0.5) < (threshold - 0.5)
    return ControlResult(
        control="C2", name="leakage audit (trivial content-blind baselines)",
        passed=passed,
        detail=(
            f"strongest trivial baseline is '{worst[0]}' at AUC {worst[1]:.3f}; "
            + ("it fails to separate the classes, as required — no metadata leakage detected"
               if passed else
               f"it separates the classes beyond the {threshold} tolerance, indicating the "
               "labels are predictable from metadata alone. Downstream numbers are unsafe.")
        ),
        numbers={"auc_per_baseline": {k: round(v, 4) for k, v in results.items()},
                 "threshold": threshold},
    )


# --------------------------------------------------------------------------------------
# C3 — real-vs-real coherence probe
# --------------------------------------------------------------------------------------


def c3_coherence_probe(X_real: np.ndarray, *, n_splits: int = 5, seed: int = 0,
                       C: float = 0.25) -> ControlResult:
    """Try to separate authentic clips from other authentic clips.

    Whatever AUC this reaches is the detector's *floor*. A model scoring 0.72 against a
    0.68 floor has learned almost nothing; scored against 0.5 it would look respectable.
    This is the control the audit paper found no surveyed work applying.
    """
    X_real = np.asarray(X_real, dtype=np.float64)
    n = X_real.shape[0]
    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    for s in range(n_splits):
        perm = rng.permutation(n)
        pseudo = np.zeros(n, dtype=bool)
        pseudo[perm[: n // 2]] = True     # arbitrary, meaningless split
        idx = rng.permutation(n)
        cut = int(0.7 * n)
        tr, te = idx[:cut], idx[cut:]
        if len(np.unique(pseudo[tr])) < 2 or len(np.unique(pseudo[te])) < 2:
            continue
        probs = matched_readout(X_real[tr], pseudo[tr], X_real[te], C=C, seed=seed + s)
        aucs.append(roc_auc(probs, pseudo[te]))
    if not aucs:
        return ControlResult("C3", "real-vs-real coherence probe", False,
                             "insufficient authentic clips to run the probe", {})
    floor = float(np.mean(aucs))
    passed = floor < 0.62
    return ControlResult(
        control="C3", name="real-vs-real coherence probe",
        passed=passed,
        detail=(f"arbitrary splits of authentic clips are separable at AUC {floor:.3f}; "
                + ("close enough to chance that the feature space is not encoding spurious "
                   "structure among authentic material"
                   if passed else
                   "this is well above chance, meaning the features encode nuisance structure "
                   "and every reported AUC must be read as a margin above this floor")),
        numbers={"floor_auc": round(floor, 4), "per_split": [round(a, 4) for a in aucs],
                 "n_real": int(n)},
    )


# --------------------------------------------------------------------------------------
# C5 — multi-seed stability
# --------------------------------------------------------------------------------------


def c5_multiseed(
    X: np.ndarray, y: np.ndarray, *, seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    test_fraction: float = 0.3,
) -> ControlResult:
    """Refit under several seeds and report the spread, not one lucky split."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(bool)
    aucs: list[float] = []
    for s in seeds:
        rng = np.random.default_rng(s)
        idx = rng.permutation(len(y))
        cut = int((1 - test_fraction) * len(y))
        tr, te = idx[:cut], idx[cut:]
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        probs = matched_readout(X[tr], y[tr], X[te], seed=s)
        aucs.append(roc_auc(probs, y[te]))
    if not aucs:
        return ControlResult("C5", "multi-seed stability", False, "could not fit any seed", {})
    arr = np.array(aucs)
    return ControlResult(
        control="C5", name="multi-seed stability",
        passed=len(aucs) >= 5,
        detail=(f"AUC over {len(aucs)} seeds: mean {arr.mean():.4f}, sd {arr.std():.4f}, "
                f"range [{arr.min():.4f}, {arr.max():.4f}]"),
        numbers={"n_seeds": len(aucs), "mean_auc": round(float(arr.mean()), 4),
                 "std_auc": round(float(arr.std()), 4),
                 "per_seed": [round(a, 4) for a in aucs]},
    )


# --------------------------------------------------------------------------------------
# C6 — cross-dataset
# --------------------------------------------------------------------------------------


def c6_cross_dataset(auc_in_domain: float, auc_cross: float, floor_cross: float
                     ) -> ControlResult:
    drop = auc_in_domain - auc_cross
    return ControlResult(
        control="C6", name="cross-dataset validation",
        passed=auc_cross > floor_cross + 0.05,
        detail=(f"in-domain AUC {auc_in_domain:.4f} -> cross-corpus AUC {auc_cross:.4f} "
                f"(drop {drop:.4f}; cross-corpus floor {floor_cross:.4f}). "
                + ("the model retains a margin above the coherence floor on an "
                   "independently parameterised corpus"
                   if auc_cross > floor_cross + 0.05 else
                   "the model does not retain a meaningful margin off-distribution")),
        numbers={"auc_in_domain": round(auc_in_domain, 4), "auc_cross": round(auc_cross, 4),
                 "drop": round(drop, 4), "floor_cross": round(floor_cross, 4)},
    )
