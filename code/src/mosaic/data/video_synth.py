"""Procedural talking-head rendering and simulated visual manipulations.

**SYNTHETIC DEMO DATA.** See ``mosaic.data.__init__``.

"Real" video is a rendered head whose mouth aperture is driven by an articulation
envelope, with handheld camera drift, blinks, prosodic head motion and per-frame sensor
noise. The properties the visual branch keys on — spatially white sensor noise, a
consistent high-frequency spectrum across the whole frame, smooth second-order motion —
are all present because they are rendered in, not because they are asserted.

"Fake" video applies the manipulation families that visual deepfake detectors are
reported to exploit:

  * **blending seam** — a composited face region whose colour statistics and noise
    characteristics differ from the surrounding frame, with a feathered boundary.
  * **resample blur** — a face region downsampled and re-upsampled, destroying
    high-frequency detail inside the region only. This region-local HF deficit is the
    single most characteristic generated-face signature.
  * **face denoise** — sensor noise removed inside the face region, so the noise-residual
    energy no longer matches the background.
  * **temporal flicker** — per-frame gain/hue jitter confined to the face region.
  * **warp jitter** — temporally inconsistent non-rigid warping.
  * **checkerboard** — the faint periodic pattern left by transposed convolutions.
  * **frame duplication/drop** — irregular temporal sampling.

As with the audio side, these are simulations of documented artefact classes, not the
output of a real generator, and any number derived from them is reported as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage

# --------------------------------------------------------------------------------------
# Scene description
# --------------------------------------------------------------------------------------


@dataclass
class Scene:
    """Randomised per-clip appearance so identity/lighting never encodes the label."""

    skin: np.ndarray          # (3,) RGB base colour
    bg_a: np.ndarray          # (3,) background gradient endpoint
    bg_b: np.ndarray          # (3,) background gradient endpoint
    head_rx: float
    head_ry: float
    head_cx: float
    head_cy: float
    light_dir: np.ndarray     # (2,) illumination direction
    noise_sigma: float        # sensor noise standard deviation (0-255 scale)
    blink_rate: float         # blinks per second
    motion_scale: float       # handheld camera / head motion amplitude
    mouth_rx: float
    mouth_ry: float
    eye_sep: float
    eye_y: float

    @staticmethod
    def random(rng: np.random.Generator, h: int, w: int) -> "Scene":
        skin_base = np.array([
            rng.uniform(120, 225), rng.uniform(85, 175), rng.uniform(70, 150)
        ])
        bg_lo = rng.uniform(25, 110)
        return Scene(
            skin=skin_base,
            bg_a=np.array([bg_lo, bg_lo, bg_lo]) + rng.uniform(-25, 25, 3),
            bg_b=np.array([bg_lo, bg_lo, bg_lo]) + rng.uniform(-45, 45, 3),
            head_rx=float(rng.uniform(0.20, 0.28) * w),
            head_ry=float(rng.uniform(0.26, 0.35) * h),
            head_cx=float(rng.uniform(0.44, 0.56) * w),
            head_cy=float(rng.uniform(0.46, 0.56) * h),
            light_dir=rng.normal(0, 1, 2),
            noise_sigma=float(rng.uniform(1.6, 4.5)),
            blink_rate=float(rng.uniform(0.15, 0.45)),
            motion_scale=float(rng.uniform(0.004, 0.018)),
            mouth_rx=float(rng.uniform(0.055, 0.085) * w),
            mouth_ry=float(rng.uniform(0.030, 0.050) * h),
            eye_sep=float(rng.uniform(0.09, 0.125) * w),
            eye_y=float(rng.uniform(0.10, 0.14) * h),
        )


@dataclass
class RenderedClip:
    frames: np.ndarray                     # (T, H, W, 3) uint8
    fps: float
    face_bbox: tuple[int, int, int, int]   # (x0, y0, x1, y1), union across frames
    mouth_curve: np.ndarray                # (T,) ground-truth mouth aperture
    scene: Scene | None = None
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# Smooth random motion
# --------------------------------------------------------------------------------------


def _smooth_noise(rng: np.random.Generator, n: int, hz: float, fps: float) -> np.ndarray:
    """Band-limited random signal: control points at ``hz``, cubic-ish interpolation."""
    n_ctrl = max(2, int(np.ceil(n / fps * hz)) + 2)
    ctrl = rng.normal(0, 1, n_ctrl)
    x = np.linspace(0, n_ctrl - 1, n)
    out = np.interp(x, np.arange(n_ctrl), ctrl)
    # Two smoothing passes turn the piecewise-linear interpolant into something with
    # continuous second derivatives, which matters: real camera motion has no
    # acceleration discontinuities and the visual branch measures exactly that.
    k = max(1, int(fps / max(hz, 1e-3) / 4))
    kern = np.hanning(2 * k + 1)
    kern /= kern.sum()
    for _ in range(2):
        out = np.convolve(out, kern, mode="same")
    return out


def _blink_curve(rng: np.random.Generator, n: int, fps: float, rate: float) -> np.ndarray:
    """1.0 = eyes open, dips toward 0 during a blink."""
    curve = np.ones(n)
    n_blinks = rng.poisson(rate * n / fps)
    dur = max(2, int(0.13 * fps))
    for _ in range(int(n_blinks)):
        start = int(rng.integers(0, max(1, n - dur)))
        curve[start:start + dur] = np.minimum(
            curve[start:start + dur], 1.0 - np.hanning(dur)
        )
    return curve


# --------------------------------------------------------------------------------------
# Renderer
# --------------------------------------------------------------------------------------


def _ellipse_mask(X: np.ndarray, Y: np.ndarray, cx: float, cy: float,
                  rx: float, ry: float, soft: float = 1.5) -> np.ndarray:
    """Anti-aliased ellipse in [0, 1]."""
    d = ((X - cx) / max(rx, 1e-6)) ** 2 + ((Y - cy) / max(ry, 1e-6)) ** 2
    return np.clip((1.0 - d) * max(rx, ry) / soft, 0.0, 1.0)


def render_clip(
    rng: np.random.Generator,
    drive_envelope: np.ndarray,
    n_frames: int,
    fps: float,
    height: int,
    width: int,
    scene: Scene | None = None,
    *,
    mouth_lead_ms: float = 30.0,
) -> RenderedClip:
    """Render a talking head whose mouth follows ``drive_envelope``.

    ``drive_envelope`` is sampled at the audio rate and resampled here to frame times.
    Passing an envelope that does *not* correspond to the clip's audio is how
    desynchronised and content-mismatched clips are produced.
    """
    scene = scene or Scene.random(rng, height, width)
    Y, X = np.mgrid[0:height, 0:width].astype(np.float32)

    # ---- articulation ------------------------------------------------------------------
    env = np.asarray(drive_envelope, dtype=np.float64)
    if env.size == 0:
        env = np.zeros(n_frames)
    src_t = np.linspace(0, 1, env.size)
    dst_t = np.linspace(0, 1, n_frames)
    # Mouth motion slightly precedes the acoustic result (articulatory anticipation).
    lead = mouth_lead_ms * 1e-3 * fps / max(n_frames, 1)
    mouth = np.interp(np.clip(dst_t + lead, 0, 1), src_t, env)
    k = max(1, int(0.045 * fps))
    kern = np.hanning(2 * k + 1)
    kern /= kern.sum()
    mouth = np.convolve(mouth, kern, mode="same")
    if mouth.max() > 0:
        mouth = mouth / mouth.max()

    # ---- motion tracks -----------------------------------------------------------------
    dx = _smooth_noise(rng, n_frames, 0.7, fps) * scene.motion_scale * width
    dy = _smooth_noise(rng, n_frames, 0.6, fps) * scene.motion_scale * height
    # Prosodic head nod: partly driven by articulation, as in real speech.
    dy = dy + (mouth - mouth.mean()) * scene.motion_scale * height * 0.9
    tilt = _smooth_noise(rng, n_frames, 0.4, fps) * 0.05
    blink = _blink_curve(rng, n_frames, fps, scene.blink_rate)
    brow = _smooth_noise(rng, n_frames, 0.5, fps) * 0.35 + mouth * 0.25

    # ---- static background -------------------------------------------------------------
    gx = X / width
    gy = Y / height
    ramp = (0.6 * gx + 0.4 * gy)[..., None]
    bg = scene.bg_a[None, None, :] * (1 - ramp) + scene.bg_b[None, None, :] * ramp
    # Low-frequency texture so the background is not a perfectly flat gradient.
    tex = _smooth_field(rng, height, width, cells=6) * 12.0
    bg = bg + tex[..., None]

    # Lambertian-ish shading term over the head.
    ld = scene.light_dir / (np.linalg.norm(scene.light_dir) + 1e-9)

    frames = np.empty((n_frames, height, width, 3), dtype=np.uint8)
    xs0, ys0, xs1, ys1 = width, height, 0, 0

    # Facial features (eyes, brows, nose, mouth) only ever occupy a window around the
    # head, so they are composited into that sub-view instead of the full frame. The head
    # and neck still need full-frame masks. This is ~3x cheaper per frame and changes
    # nothing about the output.
    mx_drift = float(np.abs(dx).max()) + 4.0
    my_drift = float(np.abs(dy).max()) + 4.0
    wy0 = max(0, int(scene.head_cy - scene.head_ry * 1.10 - my_drift))
    wy1 = min(height, int(scene.head_cy + scene.head_ry * 1.10 + my_drift))
    wx0 = max(0, int(scene.head_cx - scene.head_rx * 1.15 - mx_drift))
    wx1 = min(width, int(scene.head_cx + scene.head_rx * 1.15 + mx_drift))
    win = (slice(wy0, wy1), slice(wx0, wx1))
    Xw, Yw = X[win], Y[win]

    for t in range(n_frames):
        cx = scene.head_cx + dx[t]
        cy = scene.head_cy + dy[t]
        rx = scene.head_rx * (1.0 + 0.02 * tilt[t])
        ry = scene.head_ry * (1.0 - 0.02 * tilt[t])

        img = bg.copy()

        head = _ellipse_mask(X, Y, cx, cy, rx, ry, soft=2.0)
        nx = (X - cx) / rx
        ny = (Y - cy) / ry
        shade = 1.0 - 0.30 * np.clip(nx * ld[0] + ny * ld[1], -1, 1)
        # A soft rim/ambient-occlusion term keeps the head from looking like a flat disc.
        shade = shade * (1.0 - 0.18 * np.clip(nx**2 + ny**2, 0, 1))
        face = scene.skin[None, None, :] * shade[..., None]
        img = img * (1 - head[..., None]) + face * head[..., None]

        # Neck/shoulders: a large low ellipse anchors the head in the frame.
        neck = _ellipse_mask(X, Y, cx, cy + ry * 1.35, rx * 1.5, ry * 0.75, soft=3.0)
        neck = np.clip(neck - head, 0, 1)
        img = img * (1 - neck[..., None]) + (scene.skin * 0.72)[None, None, :] * neck[..., None]

        # ---- facial features, composited within the face window only -------------------
        sub = img[win]

        def paint(mask: np.ndarray, colour: np.ndarray, strength: float = 1.0) -> None:
            nonlocal sub
            a = (mask * strength)[..., None]
            sub = sub * (1 - a) + colour[None, None, :] * a

        eye_ry = scene.eye_y * 0.30 * blink[t] + 0.5
        for sign in (-1.0, 1.0):
            ex = cx + sign * scene.eye_sep
            ey = cy - scene.eye_y
            paint(_ellipse_mask(Xw, Yw, ex, ey, scene.eye_sep * 0.52, eye_ry, soft=1.0),
                  np.array([238.0, 238.0, 232.0]))
            paint(_ellipse_mask(Xw, Yw, ex, ey, scene.eye_sep * 0.21,
                                min(eye_ry, scene.eye_sep * 0.21), soft=0.8),
                  np.array([32.0, 30.0, 38.0]))
            by = ey - scene.eye_y * 0.62 - brow[t] * 2.5
            paint(_ellipse_mask(Xw, Yw, ex, by, scene.eye_sep * 0.60, 2.2, soft=1.0),
                  scene.skin * 0.42)

        # Nose: a shading lobe rather than a drawn shape.
        paint(_ellipse_mask(Xw, Yw, cx, cy + ry * 0.10, rx * 0.13, ry * 0.16, soft=2.5),
              scene.skin * 0.80, strength=0.18)

        # Mouth: aperture follows the articulation envelope.
        my = cy + ry * 0.42
        open_amt = mouth[t]
        mrx = scene.mouth_rx * (1.0 + 0.10 * open_amt)
        mry = scene.mouth_ry * (0.16 + 0.84 * open_amt)
        paint(_ellipse_mask(Xw, Yw, cx, my, mrx * 1.18, mry + 2.4, soft=1.2),
              scene.skin * np.array([0.92, 0.62, 0.62]))
        inner = _ellipse_mask(Xw, Yw, cx, my, mrx, mry, soft=1.0)
        paint(inner, np.array([48.0, 22.0, 26.0]))
        # Teeth appear only once the mouth is meaningfully open.
        if open_amt > 0.35:
            teeth = _ellipse_mask(Xw, Yw, cx, my - mry * 0.45, mrx * 0.78, mry * 0.26, soft=0.9)
            paint(teeth * inner, np.array([225.0, 222.0, 214.0]))

        img[win] = sub

        # Sensor noise: independent per frame and spatially white. Its absence inside a
        # manipulated region is one of the strongest cues the visual branch has.
        img = img + rng.normal(0, scene.noise_sigma, img.shape)

        frames[t] = np.clip(img, 0, 255).astype(np.uint8)

        xs0 = min(xs0, int(cx - rx)); xs1 = max(xs1, int(cx + rx))
        ys0 = min(ys0, int(cy - ry)); ys1 = max(ys1, int(cy + ry))

    bbox = (max(0, xs0), max(0, ys0), min(width, xs1), min(height, ys1))
    return RenderedClip(frames=frames, fps=fps, face_bbox=bbox,
                        mouth_curve=mouth.astype(np.float32), scene=scene)


def _smooth_field(rng: np.random.Generator, h: int, w: int, cells: int = 6) -> np.ndarray:
    """Low-frequency 2-D random field, upsampled from a coarse grid."""
    coarse = rng.normal(0, 1, (cells, cells))
    return ndimage.zoom(coarse, (h / cells, w / cells), order=3)[:h, :w]


# --------------------------------------------------------------------------------------
# Visual manipulations
# --------------------------------------------------------------------------------------

FAKE_VIDEO_QUALITY_LEVELS = ("low", "medium", "high")
FAKE_VIDEO_QUALITY_WEIGHTS = (0.30, 0.40, 0.30)

_ALL_VIDEO_OPS = (
    "blend_seam", "resample_blur", "face_denoise", "temporal_flicker",
    "warp_jitter", "checkerboard", "frame_dup_drop",
)


def sample_fake_video_ops(rng: np.random.Generator, quality: str | None = None
                          ) -> tuple[str, list[str]]:
    """Pick a visual-fake quality level and its manipulation ops.

    Mirrors the audio side: a spread of easy and hard clips, so that some fakes are
    genuinely near the decision boundary instead of all being trivially separable.
    """
    if quality is None:
        quality = str(rng.choice(FAKE_VIDEO_QUALITY_LEVELS, p=FAKE_VIDEO_QUALITY_WEIGHTS))
    if quality == "low":
        pool = ["blend_seam", "resample_blur", "face_denoise", "temporal_flicker",
                "warp_jitter", "checkerboard"]
        k = int(rng.integers(3, 5))
        ops = list(rng.choice(pool, size=k, replace=False))
    elif quality == "medium":
        pool = ["blend_seam", "resample_blur", "face_denoise", "temporal_flicker", "warp_jitter"]
        ops = list(rng.choice(pool, size=2, replace=False))
    else:
        # A good face model: one subtle, region-local artefact only.
        ops = [str(rng.choice(["resample_blur", "face_denoise", "blend_seam"]))]
    if rng.random() < 0.15:
        ops.append("frame_dup_drop")
    return quality, ops


def _face_slice(bbox: tuple[int, int, int, int], h: int, w: int, pad: float = 0.06):
    x0, y0, x1, y1 = bbox
    px, py = int(pad * w), int(pad * h)
    x0 = max(0, x0 - px); y0 = max(0, y0 - py)
    x1 = min(w, x1 + px); y1 = min(h, y1 + py)
    return slice(y0, y1), slice(x0, x1)


def _area_matrix(n_in: int, n_out: int) -> np.ndarray:
    """Box-average resampling matrix (n_out x n_in): a correctly low-passing downscale."""
    edges = np.linspace(0.0, n_in, n_out + 1)
    m = np.zeros((n_out, n_in))
    for i in range(n_out):
        lo, hi = edges[i], edges[i + 1]
        for j in range(int(np.floor(lo)), min(int(np.ceil(hi)), n_in)):
            m[i, j] = max(0.0, min(hi, j + 1.0) - max(lo, float(j)))
        s = m[i].sum()
        if s > 0:
            m[i] /= s
    return m


def _linear_matrix(n_in: int, n_out: int) -> np.ndarray:
    """Linear-interpolation resampling matrix (n_out x n_in): the upscale direction."""
    pos = np.clip((np.arange(n_out) + 0.5) * n_in / n_out - 0.5, 0, n_in - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, n_in - 1)
    frac = pos - lo
    m = np.zeros((n_out, n_in))
    rows = np.arange(n_out)
    np.add.at(m, (rows, lo), 1.0 - frac)
    np.add.at(m, (rows, hi), frac)
    return m


def _resample_axis(vol: np.ndarray, mat: np.ndarray, axis: int) -> np.ndarray:
    """Apply a resampling matrix along one axis of an N-D volume.

    Separable matrix products dispatch to BLAS, which is roughly two orders of magnitude
    faster here than ``ndimage.zoom`` on a 4-D array — zoom spline-prefilters along
    *every* axis including time, where no resampling is wanted at all.
    """
    return np.moveaxis(np.tensordot(mat, vol, axes=([1], [axis])), 0, axis)


def _feather_mask(h: int, w: int, softness: float = 0.22) -> np.ndarray:
    """Elliptical alpha mask with a feathered edge, as used by face-swap compositors."""
    Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2, (w - 1) / 2
    d = np.sqrt(((X - cx) / (w / 2)) ** 2 + ((Y - cy) / (h / 2)) ** 2)
    return np.clip((1.0 - d) / max(softness, 1e-3), 0, 1)


def apply_visual_manipulations(
    clip: RenderedClip,
    rng: np.random.Generator,
    ops: list[str] | None = None,
    quality: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Apply sampled visual manipulations. Frame count and resolution are preserved."""
    if ops is None:
        quality, ops = sample_fake_video_ops(rng, quality)
    quality = quality or "medium"

    frames = clip.frames.astype(np.float32).copy()
    T, H, W, _ = frames.shape
    ys, xs = _face_slice(clip.face_bbox, H, W)
    fh, fw = ys.stop - ys.start, xs.stop - xs.start
    if fh < 8 or fw < 8:  # degenerate bbox: fall back to the centre of the frame
        ys, xs = slice(H // 4, 3 * H // 4), slice(W // 4, 3 * W // 4)
        fh, fw = ys.stop - ys.start, xs.stop - xs.start

    alpha = _feather_mask(fh, fw, softness={"low": 0.10, "medium": 0.18, "high": 0.30}[quality])
    alpha3 = alpha[..., None]
    applied: list[str] = [f"quality={quality}"]

    # All manipulations operate on the whole (T, fh, fw, 3) face volume at once rather
    # than frame by frame. This is ~30x faster and, more importantly, keeps corpus
    # generation cheap enough to regenerate from scratch whenever a seed changes.
    region = frames[:, ys, xs, :]
    alpha4 = alpha3[None]

    def blend(new: np.ndarray) -> None:
        nonlocal region
        region = region * (1 - alpha4) + new * alpha4

    if "resample_blur" in ops:
        factor = float(rng.uniform(*{"low": (2.6, 4.0), "medium": (1.9, 2.6),
                                     "high": (1.35, 1.9)}[quality]))
        sh, sw = max(2, int(fh / factor)), max(2, int(fw / factor))
        small = _resample_axis(_resample_axis(region, _area_matrix(fh, sh), 1),
                               _area_matrix(fw, sw), 2)
        back = _resample_axis(_resample_axis(small, _linear_matrix(sh, fh), 1),
                              _linear_matrix(sw, fw), 2)
        blend(back)
        applied.append(f"resample_blur(factor={factor:.2f})")

    if "face_denoise" in ops:
        sigma = {"low": 1.6, "medium": 1.1, "high": 0.7}[quality]
        blend(ndimage.gaussian_filter(region, sigma=(0, sigma, sigma, 0)))
        applied.append(f"face_denoise(sigma={sigma})")

    if "blend_seam" in ops:
        # Colour/brightness statistics of the pasted region do not match the surround.
        gain = 1.0 + rng.uniform(*{"low": (0.10, 0.20), "medium": (0.05, 0.10),
                                   "high": (0.02, 0.05)}[quality]) * rng.choice([-1, 1])
        tint = rng.normal(0, {"low": 9.0, "medium": 5.0, "high": 2.2}[quality], 3)
        shift = int(rng.integers(1, {"low": 4, "medium": 3, "high": 2}[quality] + 1))
        blend(np.roll(region, shift=(shift, shift), axis=(1, 2)) * gain + tint)
        applied.append(f"blend_seam(gain={gain:.3f},shift={shift})")

    if "temporal_flicker" in ops:
        amp = {"low": 0.055, "medium": 0.030, "high": 0.014}[quality]
        jitter = rng.normal(1.0, amp, T)[:, None, None, None]
        hue = rng.normal(0, amp * 22, (T, 3))[:, None, None, :]
        blend(region * jitter + hue)
        applied.append(f"temporal_flicker(amp={amp})")

    if "warp_jitter" in ops:
        amp = {"low": 1.5, "medium": 0.9, "high": 0.45}[quality]
        # A fresh displacement field every frame: temporally inconsistent by construction,
        # which is what the second-order-dynamics features detect.
        def field() -> np.ndarray:
            coarse = rng.normal(0, amp, (T, 5, 5))
            up = _resample_axis(coarse, _linear_matrix(5, fh), 1)
            return _resample_axis(up, _linear_matrix(5, fw), 2)

        dy, dx = field(), field()
        Tg, Yg, Xg = np.mgrid[0:T, 0:fh, 0:fw].astype(np.float32)
        coords = [Tg, Yg + dy, Xg + dx]
        warped = np.stack([
            ndimage.map_coordinates(region[..., c], coords, order=1, mode="nearest")
            for c in range(3)
        ], axis=-1)
        blend(warped)
        applied.append(f"warp_jitter(amp={amp})")

    if "checkerboard" in ops:
        amp = {"low": 3.2, "medium": 1.8, "high": 0.9}[quality]
        period = int(rng.choice([2, 4, 8]))
        Yg, Xg = np.mgrid[0:fh, 0:fw]
        half = max(1, period // 2)
        pattern = amp * (((Yg // half + Xg // half) % 2) * 2.0 - 1.0)
        region = region + pattern[None, ..., None] * alpha4
        applied.append(f"checkerboard(period={period},amp={amp})")

    frames[:, ys, xs, :] = region

    if "frame_dup_drop" in ops:
        n_events = int(rng.integers(1, 4))
        order = list(range(T))
        for _ in range(n_events):
            i = int(rng.integers(1, T - 1))
            if rng.random() < 0.5:
                order[i] = order[i - 1]          # duplicate
            else:
                order[i] = order[min(T - 1, i + 1)]  # drop
        frames = frames[order]
        applied.append(f"frame_dup_drop(n={n_events})")

    return np.clip(frames, 0, 255).astype(np.uint8), applied
