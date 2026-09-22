"""Tests for L1 hashing primitives.

The central property under test is that SHA-256 and perceptual hashing answer *different*
questions and are never interchangeable.
"""

from __future__ import annotations

import numpy as np
import pytest

from mosaic.hashing import (audio_phash, hamming, hashes_from_hex, hashes_to_hex,
                            phash_frame, sequence_distance, sha256_array, sha256_bytes,
                            sha256_file, video_phash)


def test_sha256_is_deterministic_and_exact(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"mosaic" * 1000)
    assert sha256_file(p) == sha256_file(p)
    q = tmp_path / "b.bin"
    q.write_bytes(b"mosaic" * 1000 + b"!")
    assert sha256_file(p) != sha256_file(q)


def test_sha256_single_bit_change_changes_digest():
    a = bytearray(b"\x00" * 64)
    b = bytearray(a)
    b[13] ^= 0x01
    assert sha256_bytes(bytes(a)) != sha256_bytes(bytes(b))


def test_sha256_array_includes_dtype_and_shape():
    """Same bytes, different interpretation, must not collide."""
    a = np.zeros(8, dtype=np.uint8)
    b = np.zeros(8, dtype=np.uint8).reshape(2, 4)
    assert sha256_array(a) != sha256_array(b)
    c = np.zeros(4, dtype=np.uint16)  # same 8 bytes as `a`
    assert sha256_array(a) != sha256_array(c)


def test_phash_is_stable_under_mild_compression_noise(rng):
    base = rng.normal(128, 40, (64, 64))
    noisy = np.clip(base + rng.normal(0, 2.0, base.shape), 0, 255)
    assert hamming(phash_frame(base), phash_frame(noisy)) <= 8


def test_phash_differs_for_different_content(rng):
    a = rng.normal(128, 40, (64, 64))
    b = rng.normal(128, 40, (64, 64))
    assert hamming(phash_frame(a), phash_frame(b)) > 8


def test_video_phash_sequence_distance_separates_content(rng):
    clip_a = rng.normal(120, 30, (40, 48, 48))
    clip_b = rng.normal(120, 30, (40, 48, 48))
    a = video_phash(clip_a, 16)
    a_noisy = video_phash(np.clip(clip_a + rng.normal(0, 1.5, clip_a.shape), 0, 255), 16)
    b = video_phash(clip_b, 16)
    assert sequence_distance(a, a_noisy) < sequence_distance(a, b)
    assert sequence_distance(a, a) == 0.0


def test_sequence_distance_handles_empty():
    assert sequence_distance([], [1, 2]) == 1.0
    assert sequence_distance([1, 2], []) == 1.0


def test_audio_phash_separates_content(rng):
    sr = 16000
    t = np.arange(sr) / sr
    tone = np.sin(2 * np.pi * 220 * t)
    other = np.sin(2 * np.pi * 880 * t)
    ha = audio_phash(tone, sr, 8)
    hb = audio_phash(other, sr, 8)
    assert len(ha) > 0
    assert sequence_distance(ha, hb) > 0.1


def test_hash_hex_roundtrip():
    seq = [0, 1, 2**63, 2**64 - 1]
    assert hashes_from_hex(hashes_to_hex(seq)) == seq
    assert hashes_from_hex("") == []
