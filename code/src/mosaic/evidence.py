"""Turning model scores into named, inspectable findings.

A verdict that says only "fake, p=0.87" is not usable evidence. This module converts each
branch's feature contributions into a ranked list of *artefacts found*: what was measured,
what value it took, how that compares with authentic material the model was fitted on, and
how much it moved the decision.

Two deliberate restrictions:

* Contributions come from the linear model's own coefficients applied to standardised
  feature values, so a reported contribution is exactly the term that entered the log-odds.
  Nothing is a post-hoc rationalisation of a score produced some other way.
* Cepstral coefficients are collapsed into a single grouped finding. Forty lines reading
  "cepstral coefficient 12 is unusual" is not evidence anyone can act on; one line saying
  the overall spectral-shape profile departs from recorded speech is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .l3_audio import AUDIO_FEATURE_DESCRIPTIONS, LFCC_GROUP_DESCRIPTION
from .l3_av_sync import AV_FEATURE_DESCRIPTIONS, SuspiciousInterval
from .l3_visual import VISUAL_FEATURE_DESCRIPTIONS
from .models import BranchPrediction

ALL_DESCRIPTIONS: dict[str, str] = {
    **VISUAL_FEATURE_DESCRIPTIONS,
    **AUDIO_FEATURE_DESCRIPTIONS,
    **AV_FEATURE_DESCRIPTIONS,
}


@dataclass
class Finding:
    """One named artefact supporting (or contradicting) the verdict."""

    branch: str
    feature: str
    description: str
    measured: float
    typical_authentic: float | None
    deviation_sigma: float
    contribution_logodds: float
    direction: str            # "supports_manipulated" | "supports_authentic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch": self.branch,
            "feature": self.feature,
            "description": self.description,
            "measured": round(self.measured, 6),
            "typical_authentic": (round(self.typical_authentic, 6)
                                  if self.typical_authentic is not None else None),
            "deviation_sigma": round(self.deviation_sigma, 3),
            "contribution_logodds": round(self.contribution_logodds, 4),
            "direction": self.direction,
        }

    def describe(self) -> str:
        arrow = "higher" if self.measured > (self.typical_authentic or 0) else "lower"
        base = f"[{self.branch}] {self.description}: measured {self.measured:.4g}"
        if self.typical_authentic is not None:
            base += (f", {arrow} than the authentic-reference mean {self.typical_authentic:.4g} "
                     f"({abs(self.deviation_sigma):.1f} sd)")
        return base


@dataclass
class EvidenceReport:
    findings: list[Finding] = field(default_factory=list)
    suspicious_intervals: list[dict[str, float]] = field(default_factory=list)
    provenance_findings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "findings": [f.to_dict() for f in self.findings],
            "suspicious_intervals": self.suspicious_intervals,
            "provenance_findings": self.provenance_findings,
            "notes": self.notes,
        }

    def top(self, n: int = 5) -> list[Finding]:
        return sorted(self.findings, key=lambda f: abs(f.contribution_logodds), reverse=True)[:n]


def _group_lfcc(contributions, branch: str, train_stats) -> Finding | None:
    """Collapse all lfcc_* contributions into one grouped finding."""
    members = [c for c in contributions if c.name.startswith("lfcc_")]
    if not members:
        return None
    total = float(sum(c.contribution for c in members))
    if abs(total) < 1e-9:
        return None
    strongest = max(members, key=lambda c: abs(c.contribution))
    return Finding(
        branch=branch,
        feature="lfcc_profile",
        description=LFCC_GROUP_DESCRIPTION,
        measured=float(strongest.value),
        typical_authentic=(train_stats.get(strongest.name, {}) or {}).get("real_mean"),
        deviation_sigma=float(strongest.z),
        contribution_logodds=total,
        direction="supports_manipulated" if total > 0 else "supports_authentic",
    )


def build_branch_findings(
    prediction: BranchPrediction,
    branch: str,
    train_stats: dict[str, dict[str, float]],
    *,
    top_k: int = 4,
    min_contribution: float = 0.05,
) -> list[Finding]:
    """Rank a branch's feature contributions into named findings."""
    if not prediction.available or not prediction.contributions:
        return []

    grouped = _group_lfcc(prediction.contributions, branch, train_stats)
    scalar = [c for c in prediction.contributions if not c.name.startswith("lfcc_")]

    findings: list[Finding] = []
    for c in scalar:
        if abs(c.contribution) < min_contribution:
            continue
        stats = train_stats.get(c.name, {}) or {}
        findings.append(Finding(
            branch=branch,
            feature=c.name,
            description=ALL_DESCRIPTIONS.get(c.name, c.name.replace("_", " ")),
            measured=c.value,
            typical_authentic=stats.get("real_mean"),
            deviation_sigma=c.z,
            contribution_logodds=c.contribution,
            direction="supports_manipulated" if c.contribution > 0 else "supports_authentic",
        ))
    if grouped is not None and abs(grouped.contribution_logodds) >= min_contribution:
        findings.append(grouped)

    findings.sort(key=lambda f: abs(f.contribution_logodds), reverse=True)
    return findings[:top_k]


def build_report(
    visual: BranchPrediction,
    audio: BranchPrediction,
    av: BranchPrediction,
    stats: dict[str, dict[str, dict[str, float]]],
    *,
    intervals: list[SuspiciousInterval] | None = None,
    provenance: dict[str, Any] | None = None,
    top_k: int = 4,
) -> EvidenceReport:
    """Assemble the full evidence report backing a verdict."""
    report = EvidenceReport()
    for pred, name in ((visual, "visual"), (audio, "audio"), (av, "audiovisual")):
        report.findings.extend(
            build_branch_findings(pred, name, stats.get(name, {}), top_k=top_k)
        )
        if not pred.available:
            report.notes.append(f"{name} branch produced no findings: {pred.reason_unavailable}")
        elif pred.ood_flag:
            report.notes.append(
                f"{name} branch findings are discounted: this clip sits outside the branch's "
                f"training distribution (Mahalanobis percentile {pred.ood_score:.0f})"
            )

    if intervals:
        report.suspicious_intervals = [iv.to_dict() for iv in intervals]

    if provenance:
        c2pa = provenance.get("c2pa", {}) or {}
        report.provenance_findings.append(f"C2PA: {c2pa.get('status')} — {c2pa.get('detail', '')}")
        for wm in provenance.get("watermarks", []) or []:
            report.provenance_findings.append(
                f"watermark[{wm.get('detector')}]: {wm.get('status')} — {wm.get('detail', '')}"
            )
        if provenance.get("integrity_clash"):
            report.provenance_findings.append(str(provenance.get("clash_detail")))
        report.notes.extend(provenance.get("notes", []) or [])

    return report


def summarise_verdict(fusion_result, report: EvidenceReport, max_items: int = 5) -> str:
    """Human-readable summary naming the artefacts that drove the verdict."""
    lines: list[str] = []
    v = fusion_result.verdict.value
    lines.append(f"VERDICT: {v.replace('_', ' ')}")
    lines.append(
        f"  video: {fusion_result.video_verdict.value} (P(manipulated)={fusion_result.p_video_fake:.3f})   "
        f"audio: {fusion_result.audio_verdict.value} (P(manipulated)={fusion_result.p_audio_fake:.3f})"
    )
    lines.append(
        f"  joint confidence {fusion_result.top_posterior:.3f} "
        f"(margin {fusion_result.margin:.3f}, uncertainty {fusion_result.joint_vacuity:.3f})"
    )
    if fusion_result.flags:
        lines.append(f"  flags: {', '.join(fusion_result.flags)}")

    supporting = [f for f in report.top(max_items * 2) if f.direction == "supports_manipulated"]
    if supporting:
        lines.append("  artefacts found (ranked by contribution to the verdict):")
        for f in supporting[:max_items]:
            lines.append(f"    - {f.describe()}  [+{f.contribution_logodds:.2f} log-odds]")
    else:
        lines.append("  no manipulation artefacts exceeded the reporting threshold.")

    if report.suspicious_intervals:
        spans = ", ".join(
            f"{iv['start_s']:.2f}-{iv['end_s']:.2f}s" for iv in report.suspicious_intervals[:4]
        )
        lines.append(f"  suspicious intervals (audiovisual inconsistency): {spans}")

    for note in report.notes[:4]:
        lines.append(f"  note: {note}")
    for note in fusion_result.notes[:4]:
        lines.append(f"  note: {note}")
    return "\n".join(lines)
