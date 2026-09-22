"""Hashing primitives for L1 (triage) and L4 (custody).

Two categorically different tools live in this module and MOSAIC never conflates them:

**SHA-256** answers exactly one question: *are these two byte sequences identical?*
It is used for file identity, chunk commitments, Merkle nodes and the ledger hash chain.
It says nothing whatsoever about whether content is authentic, and a SHA-256 match is
never by itself evidence of authenticity — only of byte-identity with something already
adjudicated.

**Perceptual hashes** answer a fuzzy question: *do these two clips plausibly show the
same content?* They exist here for candidate retrieval only. A perceptual match is a
pointer to a record worth looking at, never proof, and by default it cannot decide a
verdict (see :class:`mosaic.config.L1Config`). The implementations below are simple
DCT-based baselines standing in for TMK+PDQF-class methods (arXiv:1912.07745); they are
not a reproduction of Meta's implementation and are not claimed to match its robustness.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.fft import dct

# --------------------------------------------------------------------------------------
# Exact identity (SHA-256)
# --------------------------------------------------------------------------------------

_CHUNK = 1 << 20


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file's bytes. Exact identity only."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    """SHA-256 over an array's exact bytes plus its dtype/shape.

    dtype and shape are included so that two arrays with identical bytes but different
    interpretations cannot collide into the same commitment.
    """
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------------------
# Perceptual hashing — candidate retrieval only
# --------------------------------------------------------------------------------------


def phash_frame(gray: np.ndarray, hash_bits: int = 64) -> int:
    """64-bit DCT perceptual hash of a single grayscale frame.

    Standard pHash: resize to 32x32, 2-D DCT, keep the low-frequency 8x8 block excluding
    the DC term, threshold at the median.
    """
    side = int(round(hash_bits**0.5))  # 64 bits -> 8x8 low-frequency block
    img = _resize_gray(gray, 32, 32).astype(np.float64)
    d = dct(dct(img, axis=0, norm="ortho"), axis=1, norm="ortho")
    block = d[:side, :side].flatten()
    # Exclude DC (index 0) from the median so overall brightness does not bias the threshold.
    med = np.median(block[1:])
    bits = block > med
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def video_phash(frames: np.ndarray, n_sample: int = 32, hash_bits: int = 64) -> list[int]:
    """Sequence of per-frame perceptual hashes, uniformly sampled across the clip.

    Returned as a *sequence* rather than a single aggregate so that similarity can be
    computed with temporal alignment tolerance (a re-cropped or trimmed re-upload keeps
    most of the sequence intact).
    """
    if frames.ndim == 4:  # (T, H, W, C) -> luma
        frames = frames.mean(axis=3)
    total = frames.shape[0]
    if total == 0:
        return []
    idx = np.unique(np.linspace(0, total - 1, min(n_sample, total)).astype(int))
    return [phash_frame(frames[i], hash_bits) for i in idx]


def audio_phash(
    waveform: np.ndarray,
    sample_rate: int,
    n_segments: int = 16,
    hash_bits: int = 64,
) -> list[int]:
    """Per-segment spectral perceptual hash of an audio track.

    Each segment's log-mel-ish band energies are DCT'd and median-thresholded, mirroring
    the video pHash construction so the two tracks are compared the same way.
    """
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    n = waveform.shape[0]
    if n < n_segments * 256:
        n_segments = max(1, n // 256)
    if n_segments == 0:
        return []
    bounds = np.linspace(0, n, n_segments + 1).astype(int)
    out: list[int] = []
    for i in range(n_segments):
        seg = waveform[bounds[i]:bounds[i + 1]]
        if seg.size < 64:
            continue
        spec = np.abs(np.fft.rfft(seg * np.hanning(seg.size)))
        # Aggregate into hash_bits log-spaced bands (speech energy is log-distributed).
        edges = np.unique(
            np.geomspace(1, max(2, spec.size - 1), hash_bits + 1).astype(int)
        )
        if edges.size < hash_bits + 1:
            edges = np.linspace(1, spec.size - 1, hash_bits + 1).astype(int)
        bands = np.array([
            spec[edges[j]:max(edges[j] + 1, edges[j + 1])].mean() for j in range(hash_bits)
        ])
        bands = np.log1p(bands)
        d = dct(bands, norm="ortho")
        med = np.median(d[1:])
        value = 0
        for b in d > med:
            value = (value << 1) | int(b)
        out.append(value)
    return out


def hamming(a: int, b: int) -> int:
    return int(bin(a ^ b).count("1"))


def sequence_distance(seq_a: list[int], seq_b: list[int], hash_bits: int = 64) -> float:
    """Normalised distance in [0, 1] between two perceptual-hash sequences.

    Each hash in the shorter sequence is matched to its best counterpart within a small
    temporal window of the longer one, which gives tolerance to trimming and frame-rate
    changes without pretending to be a full alignment algorithm.
    """
    if not seq_a or not seq_b:
        return 1.0
    if len(seq_a) > len(seq_b):
        seq_a, seq_b = seq_b, seq_a
    n_a, n_b = len(seq_a), len(seq_b)
    window = max(1, n_b // max(1, n_a))
    dists = []
    for i, ha in enumerate(seq_a):
        centre = int(i * (n_b - 1) / max(1, n_a - 1)) if n_a > 1 else 0
        lo = max(0, centre - window)
        hi = min(n_b, centre + window + 1)
        dists.append(min(hamming(ha, seq_b[j]) for j in range(lo, hi)))
    return float(np.mean(dists) / hash_bits)


def hashes_to_hex(seq: list[int]) -> str:
    return ",".join(f"{h:016x}" for h in seq)


def hashes_from_hex(text: str) -> list[int]:
    if not text:
        return []
    return [int(part, 16) for part in text.split(",") if part]


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def _resize_gray(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Area-average resize without pulling in an image library.

    Deliberately dependency-free: the hash must be reproducible from this file alone,
    with no reliance on an external resampler's version-specific behaviour.
    """
    h, w = img.shape[:2]
    ys = (np.arange(out_h + 1) * h / out_h).astype(int)
    xs = (np.arange(out_w + 1) * w / out_w).astype(int)
    out = np.empty((out_h, out_w), dtype=np.float64)
    for i in range(out_h):
        y0, y1 = ys[i], max(ys[i] + 1, ys[i + 1])
        row = img[y0:y1]
        for j in range(out_w):
            x0, x1 = xs[j], max(xs[j] + 1, xs[j + 1])
            out[i, j] = row[:, x0:x1].mean()
    return out
