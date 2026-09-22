"""L3 audiovisual branch — cross-modal synchronisation and consistency.

This branch answers a question neither single-modality branch can: *do the picture and the
sound describe the same event?* It compares a visual articulation signal against the audio
envelope, globally and in sliding windows, and localises the intervals where they diverge.

What this branch's evidence does and does not mean
--------------------------------------------------
Desynchronisation says **something** was manipulated. It does not say **which** modality.
A reenacted face over authentic audio and a dubbed voice over authentic video produce the
same AV signature, and a *jointly generated* fake is perfectly synchronised. This is why
fusion consumes AV evidence as a coupling term over the joint (video, audio) state rather
than as a verdict on either modality — see :mod:`mosaic.fusion`. An AV branch allowed to
override a confident per-modality branch would confidently mislabel exactly the cases that
matter most.

No face detector is used. The visual articulation signal comes from the frame's own
temporal-variance structure: in talking-head footage the mouth region dominates motion
energy, so restricting the frame-difference measure to the most temporally-variable pixels
recovers an articulation proxy without an oracle, and degrades gracefully to whole-frame
motion when no such region exists. A missing or failed ROI is reported, never silently
replaced with a fabricated curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage, signal

AV_FEATURE_DESCRIPTIONS: dict[str, str] = {
    "global_peak_corr": "best achievable agreement between mouth motion and speech energy",
    "global_peak_lag_ms": "audio/video offset at which agreement is best",
    "abs_peak_lag_ms": "magnitude of the best-fitting audio/video offset (large values mean "
                       "the soundtrack does not line up with the visible articulation)",
    "peak_prominence": "how sharply the best offset stands out from other offsets (a real "
                       "recording has one clear alignment; unrelated streams have none)",
    "corr_at_zero_lag": "agreement at zero offset",
    "window_lag_std_ms": "variability of the offset across the clip (a genuine recording has "
                         "one stable offset; a spliced or drifting track does not)",
    "window_lag_median_abs_ms": "typical per-window offset magnitude",
    "frac_windows_desynced": "share of the clip whose local offset exceeds broadcast tolerance",
    "window_corr_mean": "average local audio/video agreement",
    "window_corr_min": "worst local audio/video agreement",
    "window_corr_std": "variability of local audio/video agreement",
    "mutual_information": "statistical dependence between mouth motion and speech energy",
    "envelope_corr": "direct correlation of the two activity curves",
    "suspicious_fraction": "share of the clip flagged as locally inconsistent",
    "roi_motion_concentration": "how concentrated motion is in the most active region "
                                "(speech produces localised mouth motion)",
}

AV_FEATURE_NAMES: list[str] = list(AV_FEATURE_DESCRIPTIONS.keys())

#: Broadcast AV-sync tolerance is asymmetric (audio may lag further than it may lead)
#: but a single symmetric threshold is used here and reported as such.
DESYNC_TOLERANCE_MS = 80.0


@dataclass
class SuspiciousInterval:
    start_s: float
    end_s: float
    local_corr: float
    local_lag_ms: float
    severity: float   # robust sigmas below the clip's own median agreement

    def to_dict(self) -> dict[str, float]:
        return {
            "start_s": round(self.start_s, 3), "end_s": round(self.end_s, 3),
            "local_corr": round(self.local_corr, 4),
            "local_lag_ms": round(self.local_lag_ms, 1),
            "severity_sigma": round(self.severity, 2),
        }


@dataclass
class AVFeatures:
    vector: np.ndarray
    names: list[str]
    available: bool = True
    reason_unavailable: str | None = None
    suspicious_intervals: list[SuspiciousInterval] = field(default_factory=list)
    visual_curve: np.ndarray | None = field(default=None, repr=False)
    audio_curve: np.ndarray | None = field(default=None, repr=False)

    def as_dict(self) -> dict[str, float]:
        return {n: float(v) for n, v in zip(self.names, self.vector)}


def _safe(x, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


# --------------------------------------------------------------------------------------
# Activity curves
# --------------------------------------------------------------------------------------


def visual_activity_curve(frames: np.ndarray, top_fraction: float = 0.05
                          ) -> tuple[np.ndarray, float]:
    """Mouth-aperture proxy, as a curve of length T plus a motion-concentration scalar.

    Two choices here were made from measurement, not intuition:

    **Aperture, not motion.** The obvious visual signal is the frame-to-frame difference,
    but that is mouth *velocity*, while the audio envelope tracks mouth *aperture*.
    Correlating them pairs a signal with the derivative of its counterpart, which peaks a
    quarter-cycle apart — on synchronised clips it produced a spurious ~90 ms offset and
    roughly half the correlation. Instead the aperture proxy uses ROI mean luminance,
    negated: an open mouth exposes a dark cavity, so luminance drops as the mouth opens.

    **A lower-face prior.** Restricting the ROI to the lower half of the activity bounding
    box suppresses blinks and head-motion edges, which otherwise dilute the mouth signal.
    This is a stated assumption about talking-head footage, not a face detector, and it
    degrades to whole-frame behaviour when no clear activity region exists.

    Absolute correlations stay modest (~0.4 on synchronised clips) and that is expected:
    aperture and acoustic energy are genuinely only loosely coupled — a fricative is quiet
    with a near-closed mouth, an open vowel is loud with an open one, but the mapping is
    not one-to-one. This is why production lip-sync systems learn an embedding rather than
    correlating raw signals. The gap between synchronised and desynchronised clips, not
    the absolute value, is what carries information here.
    """
    if frames.ndim == 4:
        luma = frames.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    else:
        luma = frames.astype(np.float32)
    T = luma.shape[0]
    if T < 3:
        return np.zeros(max(T, 1)), 0.0

    # Blur slightly so the ROI follows structure rather than sensor noise.
    blurred = ndimage.gaussian_filter(luma, sigma=(0, 1.0, 1.0))
    temporal_var = blurred.var(axis=0)
    flat = temporal_var.ravel()
    if flat.size == 0 or not np.isfinite(flat).any() or flat.max() <= 0:
        return np.zeros(T), 0.0

    # Locate the active region, then keep only its lower half.
    k0 = max(16, int(0.12 * flat.size))
    coarse = temporal_var >= np.partition(flat, -k0)[-k0]
    ys, _ = np.nonzero(coarse)
    scored = temporal_var.copy()
    if ys.size > 0:
        midline = int(ys.min() + 0.5 * (ys.max() - ys.min()))
        if 0 < midline < scored.shape[0] - 2:
            scored[:midline, :] = 0.0

    k = max(16, int(top_fraction * flat.size))
    thresh = np.partition(scored.ravel(), -k)[-k]
    mask = scored >= thresh
    if mask.sum() < 8:
        mask = temporal_var >= np.partition(flat, -k)[-k]
    if mask.sum() < 8:
        mask = np.ones_like(temporal_var, dtype=bool)

    concentration = _safe(temporal_var[mask].sum() / (temporal_var.sum() + 1e-9))

    curve = -blurred[:, mask].mean(axis=1).astype(np.float64)
    # Remove slow drift from lighting and head translation, keeping articulation-rate motion.
    curve = curve - ndimage.uniform_filter1d(curve, size=25, mode="nearest")
    return curve, concentration


def audio_activity_curve(wave: np.ndarray, sr: int, n_frames: int, fps: float,
                         lo_hz: float = 200.0, hi_hz: float = 1000.0) -> np.ndarray:
    """First-formant-band energy envelope resampled to the video frame rate.

    The default band is 200-1000 Hz rather than the full speech band because jaw opening
    raises the first formant, making F1-band energy a closer acoustic correlate of mouth
    aperture than broadband loudness. Measured on the corpus, it separated synchronised
    from desynchronised clips about 30% better than a 300-3400 Hz band.
    """
    if wave is None or wave.size < 64 or n_frames < 2:
        return np.zeros(max(n_frames, 1))
    x = np.asarray(wave, dtype=np.float64)
    nyq = sr / 2
    lo = max(lo_hz / nyq, 1e-4)
    hi = min(hi_hz / nyq, 0.99)
    if lo < hi:
        sos = signal.butter(4, [lo, hi], btype="band", output="sos")
        x = signal.sosfiltfilt(sos, x)
    env = np.abs(signal.hilbert(x))
    # Low-pass the envelope below the video Nyquist to avoid aliasing on resample.
    sos_env = signal.butter(4, min(fps / 2 * 0.9, sr / 2 * 0.99), btype="low", fs=sr, output="sos")
    env = signal.sosfiltfilt(sos_env, env)
    t_src = np.arange(env.size) / sr
    t_dst = np.arange(n_frames) / fps
    return np.interp(t_dst, t_src, env)


def _normalise_curve(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    x = x - x.mean()
    s = x.std()
    return x / s if s > 1e-12 else x


def _xcorr(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Normalised cross-correlation of two equal-length curves over +/- max_lag."""
    a, b = _normalise_curve(a), _normalise_curve(b)
    n = a.size
    max_lag = int(min(max_lag, n - 2))
    if n < 4 or max_lag < 1:
        return np.zeros(1), np.zeros(1)
    lags = np.arange(-max_lag, max_lag + 1)
    out = np.zeros(lags.size)
    for i, lag in enumerate(lags):
        if lag < 0:
            x, y = a[-lag:], b[:n + lag]
        elif lag > 0:
            x, y = a[:n - lag], b[lag:]
        else:
            x, y = a, b
        if x.size < 4:
            continue
        xs, ys = x - x.mean(), y - y.mean()
        denom = np.sqrt((xs**2).sum() * (ys**2).sum())
        out[i] = (xs * ys).sum() / denom if denom > 1e-12 else 0.0
    return lags, out


def _mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 8) -> float:
    if a.size < 16:
        return 0.0
    hist, _, _ = np.histogram2d(a, b, bins=bins)
    p = hist / (hist.sum() + 1e-12)
    px = p.sum(axis=1, keepdims=True)
    py = p.sum(axis=0, keepdims=True)
    nz = p > 0
    mi = (p[nz] * np.log2(p[nz] / (px @ py)[nz] + 1e-12)).sum()
    return _safe(mi)


# --------------------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------------------


def extract_av_features(
    frames: np.ndarray,
    fps: float,
    wave: np.ndarray,
    sr: int,
    *,
    window_s: float = 1.0,
    hop_s: float = 0.25,
    max_lag_ms: float = 500.0,
    interval_sigma: float = 2.0,
) -> AVFeatures:
    """Extract audiovisual synchronisation features and localise suspicious intervals."""
    n_names = len(AV_FEATURE_NAMES)
    if wave is None or wave.size < int(0.2 * sr):
        return AVFeatures(
            vector=np.zeros(n_names), names=list(AV_FEATURE_NAMES), available=False,
            reason_unavailable="no usable audio track; audiovisual analysis is not applicable",
        )
    T = int(frames.shape[0])
    if T < 8 or fps <= 0:
        return AVFeatures(
            vector=np.zeros(n_names), names=list(AV_FEATURE_NAMES), available=False,
            reason_unavailable=f"too few video frames for temporal analysis (T={T})",
        )

    vis, concentration = visual_activity_curve(frames)
    aud = audio_activity_curve(wave, sr, T, fps)
    if vis.std() < 1e-9 or aud.std() < 1e-9:
        return AVFeatures(
            vector=np.zeros(n_names), names=list(AV_FEATURE_NAMES), available=False,
            reason_unavailable="one activity curve is constant (static video or silent audio)",
            visual_curve=vis, audio_curve=aud,
        )

    feats: dict[str, float] = {}
    max_lag = int(round(max_lag_ms * 1e-3 * fps))

    lags, corr = _xcorr(vis, aud, max_lag)
    best = int(np.argmax(corr))
    feats["global_peak_corr"] = _safe(corr[best])
    feats["global_peak_lag_ms"] = _safe(lags[best] / fps * 1000.0)
    feats["abs_peak_lag_ms"] = abs(feats["global_peak_lag_ms"])
    # Prominence: peak relative to the spread of the whole correlation curve.
    feats["peak_prominence"] = _safe((corr[best] - corr.mean()) / (corr.std() + 1e-9))
    zero_idx = int(np.argmin(np.abs(lags)))
    feats["corr_at_zero_lag"] = _safe(corr[zero_idx])
    feats["envelope_corr"] = _safe(np.corrcoef(vis, aud)[0, 1])
    feats["mutual_information"] = _mutual_information(_normalise_curve(vis), _normalise_curve(aud))
    feats["roi_motion_concentration"] = concentration

    # ---- windowed analysis -------------------------------------------------------------
    win = max(8, int(window_s * fps))
    hop = max(1, int(hop_s * fps))
    win_lags: list[float] = []
    win_corrs: list[float] = []
    win_times: list[tuple[float, float]] = []
    local_max_lag = max(2, int(round(min(max_lag_ms, 300.0) * 1e-3 * fps)))
    for start in range(0, max(1, T - win + 1), hop):
        end = min(T, start + win)
        if end - start < 8:
            break
        wl, wc = _xcorr(vis[start:end], aud[start:end], local_max_lag)
        b = int(np.argmax(wc))
        win_lags.append(wl[b] / fps * 1000.0)
        win_corrs.append(wc[b])
        win_times.append((start / fps, end / fps))

    intervals: list[SuspiciousInterval] = []
    if win_corrs:
        wc_arr = np.array(win_corrs)
        wl_arr = np.array(win_lags)
        feats["window_lag_std_ms"] = _safe(wl_arr.std())
        feats["window_lag_median_abs_ms"] = _safe(np.median(np.abs(wl_arr)))
        feats["frac_windows_desynced"] = _safe((np.abs(wl_arr) > DESYNC_TOLERANCE_MS).mean())
        feats["window_corr_mean"] = _safe(wc_arr.mean())
        feats["window_corr_min"] = _safe(wc_arr.min())
        feats["window_corr_std"] = _safe(wc_arr.std())

        # Suspicious intervals: windows whose agreement falls well below the clip's own
        # median. Referencing the clip to itself rather than to a global threshold means
        # a uniformly hard clip is not flagged end-to-end, and a locally spliced one is.
        med = float(np.median(wc_arr))
        mad = float(np.median(np.abs(wc_arr - med))) * 1.4826
        scale = mad if mad > 1e-6 else (wc_arr.std() + 1e-9)
        for (t0, t1), c, l in zip(win_times, wc_arr, wl_arr):
            severity = (med - c) / scale
            if severity >= interval_sigma or abs(l) > DESYNC_TOLERANCE_MS * 2:
                intervals.append(SuspiciousInterval(
                    start_s=t0, end_s=t1, local_corr=float(c),
                    local_lag_ms=float(l), severity=float(severity),
                ))
        intervals = _merge_intervals(intervals)
        covered = sum(iv.end_s - iv.start_s for iv in intervals)
        feats["suspicious_fraction"] = _safe(covered / max(T / fps, 1e-6))
    else:
        for key in ("window_lag_std_ms", "window_lag_median_abs_ms", "frac_windows_desynced",
                    "window_corr_mean", "window_corr_min", "window_corr_std",
                    "suspicious_fraction"):
            feats[key] = 0.0

    vector = np.array([feats[n] for n in AV_FEATURE_NAMES], dtype=np.float64)
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    return AVFeatures(
        vector=vector, names=list(AV_FEATURE_NAMES), available=True,
        suspicious_intervals=intervals, visual_curve=vis, audio_curve=aud,
    )


def _merge_intervals(intervals: list[SuspiciousInterval]) -> list[SuspiciousInterval]:
    """Merge overlapping windows into contiguous reported intervals."""
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda iv: iv.start_s)
    merged = [ordered[0]]
    for iv in ordered[1:]:
        last = merged[-1]
        if iv.start_s <= last.end_s:
            merged[-1] = SuspiciousInterval(
                start_s=last.start_s,
                end_s=max(last.end_s, iv.end_s),
                local_corr=min(last.local_corr, iv.local_corr),
                local_lag_ms=last.local_lag_ms if abs(last.local_lag_ms) > abs(iv.local_lag_ms) else iv.local_lag_ms,
                severity=max(last.severity, iv.severity),
            )
        else:
            merged.append(iv)
    return merged
