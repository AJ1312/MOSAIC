"""L3 Tier-0 — near-zero-cost triage from the compressed bitstream.

Tier-0 reads features the decoder has already produced: codec motion vectors, macroblock
sizes, frame-type ratios and packet sizes, plus a handful of audio statistics computable in
a single pass over the waveform. Extracting them is a *parse*, not a forward pass
(cf. arXiv:2607.19476, arXiv:2311.10788), so the whole stage costs microseconds once L0's
decode has happened.

Its only job is to decide whether the expensive stages need to run at all. Two rules keep
that safe:

* **Tier-0 is never the sole basis for a verdict.** It can exit early only toward "no
  manipulation detected", and only when L2 has raised nothing. It can never conclude
  "manipulated" on its own — that always escalates to the full branches. This asymmetry is
  the leakage-audit lesson made structural: bitstream statistics are exactly the kind of
  feature that correlates with production pipeline rather than with content, so they are
  allowed to save compute and never to convict.
* **An early exit is recorded as an early exit.** The custody record states that Tier-1 and
  Tier-2 did not run, so no reader can mistake a cheap pass for a full analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .l0_ingest import MotionVectorFeatures

TIER0_AUDIO_FEATURES = [
    "a_rms", "a_crest", "a_zcr", "a_silence_frac", "a_hf_ratio", "a_spec_flat",
]

TIER0_FEATURE_DESCRIPTIONS: dict[str, str] = {
    "mv_count_mean": "average number of motion vectors per frame",
    "mv_mag_mean": "average motion-vector magnitude",
    "mv_mag_std": "variability of motion-vector magnitude",
    "mv_mag_p95": "95th-percentile motion-vector magnitude",
    "mv_zero_fraction": "share of zero-motion blocks",
    "mv_angle_entropy": "directional disorder of the motion field",
    "mv_block_size_mean": "average motion-compensation block size",
    "mv_temporal_smoothness": "frame-to-frame stability of overall motion",
    "frame_type_i": "share of intra-coded frames",
    "frame_type_p": "share of predicted frames",
    "frame_type_b": "share of bidirectional frames",
    "pkt_bytes_mean": "average compressed packet size",
    "pkt_bytes_std": "variability of compressed packet size",
    "a_rms": "overall audio level",
    "a_crest": "audio crest factor (peak-to-RMS ratio)",
    "a_zcr": "audio zero-crossing rate",
    "a_silence_frac": "share of near-silent samples",
    "a_hf_ratio": "share of audio energy above 6 kHz",
    "a_spec_flat": "audio spectral flatness",
}

TIER0_FEATURE_NAMES: list[str] = (
    MotionVectorFeatures.feature_names() + TIER0_AUDIO_FEATURES
)


@dataclass
class Tier0Features:
    vector: np.ndarray
    names: list[str]
    mv_available: bool
    mv_reason: str | None = None

    def as_dict(self) -> dict[str, float]:
        return {n: float(v) for n, v in zip(self.names, self.vector)}


def _audio_quick_stats(wave: np.ndarray, sr: int) -> np.ndarray:
    if wave is None or wave.size < 256:
        return np.zeros(len(TIER0_AUDIO_FEATURES))
    x = np.asarray(wave, dtype=np.float64)
    rms = float(np.sqrt((x**2).mean()) + 1e-12)
    crest = float(np.abs(x).max() / rms)
    zcr = float((np.diff(np.sign(x)) != 0).mean())
    silence = float((np.abs(x) < 1e-4).mean())
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    freqs = np.fft.rfftfreq(x.size, 1.0 / sr)
    total = float(spec.sum()) + 1e-12
    hf = float(spec[freqs > 6000].sum() / total)
    gmean = float(np.exp(np.log(spec + 1e-10).mean()))
    flat = gmean / (float(spec.mean()) + 1e-12)
    return np.array([rms, crest, zcr, silence, hf, flat], dtype=np.float64)


def extract_tier0_features(mv: MotionVectorFeatures, wave: np.ndarray, sr: int) -> Tier0Features:
    vector = np.concatenate([mv.to_vector(), _audio_quick_stats(wave, sr)])
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    return Tier0Features(vector=vector, names=list(TIER0_FEATURE_NAMES),
                         mv_available=mv.available, mv_reason=mv.reason_unavailable)


@dataclass
class Tier0Decision:
    p_any_manipulation: float
    action: str              # "early_exit_clean" | "escalate" | "escalate_forced"
    reason: str
    vacuity: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "p_any_manipulation": round(self.p_any_manipulation, 6),
            "action": self.action,
            "reason": self.reason,
            "vacuity": round(self.vacuity, 6),
        }


def tier0_decide(
    p_any: float,
    vacuity: float,
    *,
    real_exit_threshold: float,
    provenance_escalation: bool,
    mv_available: bool,
    enabled: bool = True,
) -> Tier0Decision:
    """Decide whether the cascade may stop at Tier-0."""
    if not enabled:
        return Tier0Decision(p_any, "escalate", "Tier-0 triage disabled by configuration", vacuity)
    if provenance_escalation:
        return Tier0Decision(
            p_any, "escalate_forced",
            "L2 raised a provenance flag; escalation is mandatory regardless of Tier-0 score",
            vacuity,
        )
    if not mv_available:
        return Tier0Decision(
            p_any, "escalate",
            "codec motion vectors unavailable, so Tier-0 evidence is incomplete and cannot "
            "support an early exit",
            vacuity,
        )
    if p_any <= real_exit_threshold and vacuity <= 0.5:
        return Tier0Decision(
            p_any, "early_exit_clean",
            f"bitstream triage found no manipulation indication (p={p_any:.4f} <= "
            f"{real_exit_threshold}) and L2 raised nothing; Tier-1/Tier-2 skipped to save compute",
            vacuity,
        )
    return Tier0Decision(
        p_any, "escalate",
        f"Tier-0 score p={p_any:.4f} exceeds the clean-exit threshold {real_exit_threshold} "
        "(or confidence is insufficient); escalating. Tier-0 alone never convicts.",
        vacuity,
    )
