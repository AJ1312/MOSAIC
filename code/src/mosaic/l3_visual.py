"""L3 visual branch — spatial and spatiotemporal forensic features.

The feature set is deliberately *generic*: nothing here knows where a face is, which
manipulation was applied, or that the corpus is synthetic. Every feature measures a
property that manipulated video tends to violate for reasons that hold beyond this
corpus:

**Spatial**
  * High-frequency content and its *spatial uniformity*. A generated or resampled region
    loses fine detail, so the frame develops a localised high-frequency deficit. Measured
    on a grid, without knowing where the region is.
  * Noise-residual statistics. Sensor noise is approximately spatially white; denoising,
    upsampling and blending all leave a residual that is correlated instead.
  * Local-variance discontinuity. A composited region has a boundary along which local
    texture statistics jump — the blending-seam signature.
  * Spectral peakiness in the high-frequency annulus, which is what transposed-convolution
    checkerboard patterns look like in the Fourier domain.
  * Blockiness at the 8-pixel grid, which separates codec artefacts from content.

**Temporal**
  * Second-order dynamics: real motion has continuous acceleration; frame-wise independent
    warping does not (cf. arXiv:2508.00701).
  * Temporal self-similarity structure (cf. arXiv:2604.04029).
  * Flicker and residual-energy instability, which per-frame gain jitter produces.
  * Duplicate-frame rate, which catches naive temporal resampling.

Every feature carries a human-readable description so that a verdict can name the
artefacts it found rather than emitting a bare score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

# --------------------------------------------------------------------------------------
# Feature documentation — used to build the evidence list in a verdict
# --------------------------------------------------------------------------------------

VISUAL_FEATURE_DESCRIPTIONS: dict[str, str] = {
    "hf_energy_mean": "overall high-frequency detail level",
    "hf_energy_std": "frame-to-frame variability of high-frequency detail",
    "hf_ratio": "fraction of spectral energy above the mid-frequency band",
    "hf_grid_min_ratio": "weakest image region's detail level relative to the frame median "
                         "(a localised high-frequency deficit is the classic signature of a "
                         "regenerated or resampled region)",
    "hf_grid_dispersion": "spread of detail level across image regions (uniform in "
                          "unmanipulated frames, uneven when one region was replaced)",
    "noise_residual_std": "sensor-noise residual energy",
    "noise_autocorr_lag1": "spatial correlation of the noise residual (real sensor noise is "
                           "close to uncorrelated; denoising and upsampling make it correlated)",
    "noise_residual_kurtosis": "heavy-tailedness of the noise residual",
    "noise_grid_dispersion": "spread of noise level across image regions (a region with "
                             "suppressed noise indicates local synthesis or denoising)",
    "color_residual_corr": "correlation between colour-channel noise residuals",
    "blockiness": "edge energy aligned to the 8-pixel coding grid",
    "edge_density": "proportion of strong edges in the frame",
    "seam_score": "sharpest discontinuity in the local-texture map (a blending seam "
                  "produces an abrupt change in local variance along a closed contour)",
    "spectral_peakiness": "strength of isolated peaks in the high-frequency spectrum "
                          "(periodic checkerboard patterns left by transposed convolutions)",
    "local_var_mean": "average local texture variance",
    "local_var_dispersion": "spread of local texture variance across the frame",
    "frame_diff_mean": "average inter-frame change",
    "frame_diff_std": "variability of inter-frame change",
    "flicker_index": "instability of overall frame brightness over time",
    "second_order_mean": "average motion acceleration magnitude (real motion is smooth; "
                         "independently perturbed frames are not)",
    "second_order_ratio": "acceleration relative to velocity — high values mean motion "
                          "changes direction implausibly fast between frames",
    "motion_smoothness": "correlation between consecutive inter-frame differences",
    "selfsim_offdiag_mean": "average similarity between non-adjacent frames",
    "selfsim_decay_slope": "how quickly frame similarity decays with temporal distance",
    "selfsim_entropy": "entropy of the temporal self-similarity structure",
    "duplicate_frame_rate": "proportion of near-identical consecutive frames",
    "hf_temporal_std": "instability of high-frequency detail over time",
    "residual_temporal_std": "instability of noise-residual energy over time",
}

VISUAL_FEATURE_NAMES: list[str] = list(VISUAL_FEATURE_DESCRIPTIONS.keys())

# This is an intentionally dataset-agnostic extension of the baseline feature set.
# The first block is always the unchanged full-frame baseline; the remaining blocks
# repeat the same measurements over a fixed spatial grid.  It does not assume that a
# face is present or that a particular corpus uses face swaps.
MULTIREGION_VISUAL_FEATURE_NAMES: list[str] = [
    *VISUAL_FEATURE_NAMES,
    *[f"region_{region}_{name}" for region in range(4) for name in VISUAL_FEATURE_NAMES],
]


@dataclass
class VisualFeatures:
    vector: np.ndarray
    names: list[str]
    n_frames_used: int

    def as_dict(self) -> dict[str, float]:
        return {n: float(v) for n, v in zip(self.names, self.vector)}


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _to_luma(frames: np.ndarray) -> np.ndarray:
    """(T, H, W, 3) uint8 -> (T, H, W) float32 luma in [0, 255]."""
    if frames.ndim == 3:
        return frames.astype(np.float32)
    w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return frames.astype(np.float32) @ w


def _sample_frames(frames: np.ndarray, n: int) -> np.ndarray:
    t = frames.shape[0]
    if t <= n:
        return frames
    idx = np.unique(np.linspace(0, t - 1, n).astype(int))
    return frames[idx]


def _grid_stats(img: np.ndarray, cells: int = 6) -> np.ndarray:
    """Mean of ``img`` over a cells x cells grid."""
    h, w = img.shape[-2:]
    ys = np.linspace(0, h, cells + 1).astype(int)
    xs = np.linspace(0, w, cells + 1).astype(int)
    out = np.empty((cells, cells), dtype=np.float64)
    for i in range(cells):
        for j in range(cells):
            out[i, j] = img[..., ys[i]:ys[i + 1], xs[j]:xs[j + 1]].mean()
    return out


def _safe(x: float, default: float = 0.0) -> float:
    return float(x) if np.isfinite(x) else default


# --------------------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------------------


def extract_visual_features(frames: np.ndarray, n_frames: int = 48) -> VisualFeatures:
    """Extract the full visual feature vector from a clip's frames."""
    sub = _sample_frames(frames, n_frames)
    luma = _to_luma(sub)
    T, H, W = luma.shape
    feats: dict[str, float] = {}

    # ---- high-frequency content --------------------------------------------------------
    # Laplacian magnitude per frame: a cheap, robust detail measure.
    lap = np.abs(ndimage.laplace(luma, mode="reflect"))
    hf_per_frame = lap.mean(axis=(1, 2))
    mean_level = float(luma.mean()) + 1e-6
    feats["hf_energy_mean"] = _safe(hf_per_frame.mean() / mean_level)
    feats["hf_energy_std"] = _safe(hf_per_frame.std() / mean_level)
    feats["hf_temporal_std"] = _safe(hf_per_frame.std() / (hf_per_frame.mean() + 1e-9))

    # Radial spectral split on a mid-clip frame stack.
    spec = np.abs(np.fft.rfft2(luma - luma.mean(axis=(1, 2), keepdims=True), axes=(1, 2)))
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.rfftfreq(W)[None, :]
    radius = np.sqrt(fy**2 + fx**2)
    radius = radius / (radius.max() + 1e-12)
    power = (spec**2).mean(axis=0)
    total = power.sum() + 1e-12
    feats["hf_ratio"] = _safe(power[radius > 0.5].sum() / total)

    # Spectral peakiness in the high-frequency annulus: periodic checkerboard artefacts
    # show up as isolated peaks rather than a smooth spectral falloff.
    hf_band = power[(radius > 0.55) & (radius < 0.98)]
    if hf_band.size > 16:
        med = np.median(hf_band) + 1e-12
        feats["spectral_peakiness"] = _safe(np.percentile(hf_band, 99.9) / med)
    else:
        feats["spectral_peakiness"] = 0.0

    # Per-region detail level: catches a localised deficit without knowing where it is.
    grid_hf = _grid_stats(lap.mean(axis=0), cells=6)
    med_hf = float(np.median(grid_hf)) + 1e-9
    feats["hf_grid_min_ratio"] = _safe(grid_hf.min() / med_hf)
    feats["hf_grid_dispersion"] = _safe(grid_hf.std() / med_hf)

    # ---- noise residual ----------------------------------------------------------------
    smooth = ndimage.gaussian_filter(luma, sigma=(0, 1.2, 1.2))
    resid = luma - smooth
    feats["noise_residual_std"] = _safe(resid.std())
    r = resid.reshape(T, -1)
    r_centered = r - r.mean(axis=1, keepdims=True)
    denom = (r_centered**2).mean(axis=1) + 1e-12
    feats["noise_residual_kurtosis"] = _safe(((r_centered**4).mean(axis=1) / denom**2).mean() - 3.0)

    # Lag-1 horizontal autocorrelation of the residual.
    a = resid[:, :, :-1]
    b = resid[:, :, 1:]
    num = (a * b).mean()
    den = np.sqrt((a**2).mean() * (b**2).mean()) + 1e-12
    feats["noise_autocorr_lag1"] = _safe(num / den)

    grid_noise = _grid_stats(np.abs(resid).mean(axis=0), cells=6)
    feats["noise_grid_dispersion"] = _safe(grid_noise.std() / (np.median(grid_noise) + 1e-9))

    # Colour-channel residual correlation.
    if sub.ndim == 4:
        rgb = sub.astype(np.float32)
        rgb_sm = ndimage.gaussian_filter(rgb, sigma=(0, 1.2, 1.2, 0))
        cres = (rgb - rgb_sm).reshape(T, -1, 3)
        c0, c2 = cres[..., 0].ravel(), cres[..., 2].ravel()
        feats["color_residual_corr"] = _safe(
            np.corrcoef(c0, c2)[0, 1] if c0.size > 8 else 0.0
        )
    else:
        feats["color_residual_corr"] = 0.0

    feats["residual_temporal_std"] = _safe(
        np.abs(resid).mean(axis=(1, 2)).std() / (np.abs(resid).mean() + 1e-9)
    )

    # ---- local variance / seams --------------------------------------------------------
    mean_frame = luma.mean(axis=0)
    win = 9
    mu = ndimage.uniform_filter(mean_frame, win)
    var = np.maximum(ndimage.uniform_filter(mean_frame**2, win) - mu**2, 0.0)
    logvar = np.log1p(var)
    feats["local_var_mean"] = _safe(var.mean())
    feats["local_var_dispersion"] = _safe(var.std() / (var.mean() + 1e-9))
    gy, gx = np.gradient(ndimage.gaussian_filter(logvar, 2.0))
    seam = np.sqrt(gy**2 + gx**2)
    # 99.5th percentile rather than the max: robust to a single hot pixel.
    feats["seam_score"] = _safe(np.percentile(seam, 99.5) / (np.median(seam) + 1e-9))

    # ---- blockiness / edges ------------------------------------------------------------
    dv = np.abs(np.diff(mean_frame, axis=0))
    dh = np.abs(np.diff(mean_frame, axis=1))
    rows_on = dv[7::8].mean() if dv.shape[0] > 8 else dv.mean()
    rows_off = dv[np.setdiff1d(np.arange(dv.shape[0]), np.arange(7, dv.shape[0], 8))].mean()
    cols_on = dh[:, 7::8].mean() if dh.shape[1] > 8 else dh.mean()
    cols_off = dh[:, np.setdiff1d(np.arange(dh.shape[1]), np.arange(7, dh.shape[1], 8))].mean()
    feats["blockiness"] = _safe(
        (rows_on + cols_on) / (rows_off + cols_off + 1e-9) - 1.0
    )
    grad = np.hypot(*np.gradient(mean_frame))
    feats["edge_density"] = _safe((grad > np.percentile(grad, 90)).mean())

    # ---- temporal ----------------------------------------------------------------------
    diffs = np.diff(luma, axis=0)
    dmag = np.abs(diffs).mean(axis=(1, 2))
    feats["frame_diff_mean"] = _safe(dmag.mean())
    feats["frame_diff_std"] = _safe(dmag.std())

    frame_means = luma.mean(axis=(1, 2))
    detrended = frame_means - ndimage.uniform_filter1d(frame_means, size=5, mode="nearest")
    feats["flicker_index"] = _safe(detrended.std() / (frame_means.mean() + 1e-9))

    if T >= 3:
        second = luma[2:] - 2 * luma[1:-1] + luma[:-2]
        smag = np.abs(second).mean(axis=(1, 2))
        feats["second_order_mean"] = _safe(smag.mean())
        feats["second_order_ratio"] = _safe(smag.mean() / (dmag.mean() + 1e-9))
        d0, d1 = diffs[:-1].reshape(T - 2, -1), diffs[1:].reshape(T - 2, -1)
        num = (d0 * d1).sum(axis=1)
        den = np.sqrt((d0**2).sum(axis=1) * (d1**2).sum(axis=1)) + 1e-12
        feats["motion_smoothness"] = _safe((num / den).mean())
    else:
        feats["second_order_mean"] = 0.0
        feats["second_order_ratio"] = 0.0
        feats["motion_smoothness"] = 0.0

    feats["duplicate_frame_rate"] = _safe((dmag < 0.15).mean())

    # Temporal self-similarity over downsampled frame embeddings.
    small = luma[:, ::8, ::8].reshape(T, -1)
    small = small - small.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(small, axis=1, keepdims=True) + 1e-12
    sim = (small / norm) @ (small / norm).T
    iu = np.triu_indices(T, k=2)
    if iu[0].size > 4:
        offdiag = sim[iu]
        feats["selfsim_offdiag_mean"] = _safe(offdiag.mean())
        lags = (iu[1] - iu[0]).astype(np.float64)
        # Slope of similarity vs temporal distance.
        slope = np.polyfit(lags, offdiag, 1)[0] if np.ptp(lags) > 0 else 0.0
        feats["selfsim_decay_slope"] = _safe(slope * T)
        hist, _ = np.histogram(offdiag, bins=16, range=(-1, 1))
        p = hist / (hist.sum() + 1e-12)
        nz = p[p > 0]
        feats["selfsim_entropy"] = _safe(-(nz * np.log2(nz)).sum() / 4.0)
    else:
        feats["selfsim_offdiag_mean"] = 0.0
        feats["selfsim_decay_slope"] = 0.0
        feats["selfsim_entropy"] = 0.0

    vector = np.array([feats[name] for name in VISUAL_FEATURE_NAMES], dtype=np.float64)
    return VisualFeatures(vector=vector, names=list(VISUAL_FEATURE_NAMES), n_frames_used=T)


def extract_multiregion_visual_features(
    frames: np.ndarray, n_frames: int = 48
) -> VisualFeatures:
    """Extract baseline and generic localized visual evidence.

    The baseline full-frame vector is preserved as the first block.  Four fixed
    quadrants then receive the exact same feature extractor, which gives a model
    access to spatially localized evidence while remaining independent of faces,
    identities, manipulation generators, and dataset-specific metadata.  This is
    an architecture variant, not a change to :func:`extract_visual_features`.
    """
    if frames.ndim not in (3, 4) or frames.shape[0] == 0:
        raise ValueError("frames must have shape (T,H,W) or (T,H,W,C) with T > 0")
    _, height, width = frames.shape[:3]
    mid_y, mid_x = height // 2, width // 2
    bounds = ((0, mid_y, 0, mid_x), (0, mid_y, mid_x, width),
              (mid_y, height, 0, mid_x), (mid_y, height, mid_x, width))
    if min(mid_y, mid_x, height - mid_y, width - mid_x) == 0:
        raise ValueError("frames must be at least 2x2 for multiregion extraction")

    baseline = extract_visual_features(frames, n_frames=n_frames)
    vectors = [baseline.vector]
    for y0, y1, x0, x1 in bounds:
        vectors.append(extract_visual_features(frames[:, y0:y1, x0:x1], n_frames=n_frames).vector)
    return VisualFeatures(
        vector=np.concatenate(vectors),
        names=list(MULTIREGION_VISUAL_FEATURE_NAMES),
        n_frames_used=baseline.n_frames_used,
    )
