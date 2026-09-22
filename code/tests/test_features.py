"""Tests for the L3 feature extractors.

Beyond shape/finiteness checks, these verify that each feature family actually responds
to the physical property it claims to measure.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal as sg

from mosaic.data.speech_synth import synthesize_utterance
from mosaic.l3_audio import (AUDIO_FEATURE_NAMES, _f0_track, extract_audio_features,
                             log_mel_spectrogram)
from mosaic.l3_av_sync import (AV_FEATURE_NAMES, extract_av_features, visual_activity_curve)
from mosaic.l3_visual import VISUAL_FEATURE_NAMES, extract_visual_features


# --------------------------------------------------------------------------------------
# Visual
# --------------------------------------------------------------------------------------


def test_visual_features_shape_and_finiteness(rng):
    frames = rng.integers(0, 255, (30, 64, 64, 3), dtype=np.uint8)
    f = extract_visual_features(frames, 16)
    assert f.vector.shape == (len(VISUAL_FEATURE_NAMES),)
    assert np.isfinite(f.vector).all()


def test_visual_detects_localised_high_frequency_deficit(rng):
    """A blurred region must lower the grid-local detail ratio."""
    from scipy import ndimage

    base = rng.integers(0, 255, (24, 96, 96, 3), dtype=np.uint8)
    blurred = base.astype(np.float32).copy()
    blurred[:, 20:60, 20:60] = ndimage.gaussian_filter(
        blurred[:, 20:60, 20:60], sigma=(0, 3, 3, 0))
    fa = extract_visual_features(base, 12).as_dict()
    fb = extract_visual_features(blurred.astype(np.uint8), 12).as_dict()
    assert fb["hf_grid_min_ratio"] < fa["hf_grid_min_ratio"]


def test_visual_detects_temporal_flicker(rng):
    steady = np.tile(rng.integers(60, 200, (1, 48, 48, 3), dtype=np.uint8), (30, 1, 1, 1))
    flickering = (steady.astype(np.float32)
                  * rng.normal(1.0, 0.10, (30, 1, 1, 1))).clip(0, 255).astype(np.uint8)
    assert (extract_visual_features(flickering, 30).as_dict()["flicker_index"]
            > extract_visual_features(steady, 30).as_dict()["flicker_index"])


def test_visual_detects_duplicate_frames(rng):
    moving = rng.integers(0, 255, (30, 48, 48, 3), dtype=np.uint8)
    duplicated = moving.copy()
    duplicated[1::2] = duplicated[0::2][: len(duplicated[1::2])]
    assert (extract_visual_features(duplicated, 30).as_dict()["duplicate_frame_rate"]
            >= extract_visual_features(moving, 30).as_dict()["duplicate_frame_rate"])


def test_visual_handles_short_clips(rng):
    f = extract_visual_features(rng.integers(0, 255, (2, 32, 32, 3), dtype=np.uint8), 8)
    assert np.isfinite(f.vector).all()


# --------------------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------------------


def test_audio_features_shape_and_finiteness(rng):
    wave = rng.normal(0, 0.05, 32000).astype(np.float32)
    f = extract_audio_features(wave, 16000)
    assert f.vector.shape == (len(AUDIO_FEATURE_NAMES),)
    assert np.isfinite(f.vector).all()


def test_audio_unavailable_is_reported_not_zero_filled():
    f = extract_audio_features(np.zeros(10, dtype=np.float32), 16000)
    assert not f.available
    assert f.reason_unavailable


def test_f0_tracker_matches_ground_truth():
    """The pitch tracker is validated against the synthesiser's own F0 contour.

    This is the regression test for a bug that made every 'real' clip in the corpus
    unusable: the noise floor was mixed in before loudness normalisation, so speech sat
    ~30x below the noise and the tracker found no periodicity at all.
    """
    sr, hop = 16000, 160
    ratios, recalls = [], []
    for seed in (5, 17, 33):
        u = synthesize_utterance(np.random.default_rng(seed), 3.0, sr)
        f0, strength, rms = _f0_track(u.wave.astype(float), sr, hop)
        idx = np.clip(np.arange(f0.size) * hop + int(0.02 * sr), 0, u.f0_contour.size - 1)
        gt, gtv = u.f0_contour[idx], u.voiced_mask[idx]
        v = (f0 > 0) & gtv
        recalls.append(v.sum() / max(gtv.sum(), 1))
        ratios.append(f0[v] / gt[v])
    r = np.concatenate(ratios)
    assert np.mean(recalls) > 0.75, "voiced-frame recall too low"
    assert ((r > 0.9) & (r < 1.1)).mean() > 0.75, "too many octave errors"


def test_f0_tracker_returns_aligned_arrays():
    """f0, strength and rms must share one framing, or jitter and shimmer are meaningless."""
    u = synthesize_utterance(np.random.default_rng(1), 2.0, 16000)
    f0, strength, rms = _f0_track(u.wave.astype(float), 16000, 160)
    assert f0.shape == strength.shape == rms.shape


def test_audio_detects_band_limitation():
    u = synthesize_utterance(np.random.default_rng(3), 3.0, 16000)
    sos = sg.butter(8, 5000 / 8000, btype="low", output="sos")
    limited = sg.sosfilt(sos, u.wave.astype(float)).astype(np.float32)
    a = extract_audio_features(u.wave, 16000).as_dict()
    b = extract_audio_features(limited, 16000).as_dict()
    assert b["highband_ratio_6k"] < a["highband_ratio_6k"]


def test_audio_detects_phase_incoherence():
    """Randomised STFT phase must raise the phase-advance incoherence measure."""
    u = synthesize_utterance(np.random.default_rng(9), 3.0, 16000)
    x = u.wave.astype(float)
    f, t, spec = sg.stft(x, nperseg=512, noverlap=384, boundary="zeros", padded=True)
    rng = np.random.default_rng(0)
    scrambled = np.abs(spec) * np.exp(2j * np.pi * rng.random(spec.shape))
    y = sg.istft(scrambled, nperseg=512, noverlap=384, boundary=True)[1][: x.size]
    a = extract_audio_features(x.astype(np.float32), 16000).as_dict()
    b = extract_audio_features(y.astype(np.float32), 16000).as_dict()
    assert b["phase_advance_incoherence"] > a["phase_advance_incoherence"]


def test_log_mel_spectrogram_shape(rng):
    spec = log_mel_spectrogram(rng.normal(0, 0.05, 16000), 16000, n_mels=64)
    assert spec.shape[0] == 64 and spec.shape[1] > 10
    assert np.isfinite(spec).all()


# --------------------------------------------------------------------------------------
# Audiovisual
# --------------------------------------------------------------------------------------


def _synthetic_talking_clip(rng, n=75, fps=25.0, sr=16000, shift_ms=0.0):
    """Video whose ROI brightness tracks the audio *energy*, with a controllable offset.

    One shared envelope both amplitude-modulates the audio and darkens the mouth region,
    so the two streams are coupled exactly by construction. Driving the video from the
    synthesiser's articulation envelope instead would not work as a unit test: articulatory
    openness and acoustic energy are only loosely related (a fricative is quiet with a
    nearly closed mouth), so the measured lag would be dominated by that mismatch rather
    than by the offset under test.
    """
    # Smooth positive envelope at the syllabic rate, defined on the audio timebase.
    n_ctrl = max(4, int(n / fps * 4))
    ctrl = np.abs(rng.normal(0, 1, n_ctrl)) + 0.15
    t_sr = np.linspace(0, 1, int(n / fps * sr))
    env_sr = np.interp(t_sr, np.linspace(0, 1, n_ctrl), ctrl)
    env_sr = sg.sosfiltfilt(sg.butter(4, 8, "low", fs=sr, output="sos"), env_sr)
    env_sr = np.clip(env_sr, 0, None)

    carrier = sg.sosfiltfilt(sg.butter(4, [200, 1000], "band", fs=sr, output="sos"),
                             rng.normal(0, 1, t_sr.size))
    wave = (carrier * env_sr).astype(float)
    wave /= (np.abs(wave).max() + 1e-9)

    env_frames = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, env_sr.size), env_sr)
    env_frames /= (env_frames.max() + 1e-9)
    frames = np.full((n, 48, 48, 3), 120, dtype=np.float32)
    frames += rng.normal(0, 2.0, frames.shape)
    for t in range(n):
        frames[t, 30:42, 18:30] -= env_frames[t] * 90

    if shift_ms:
        k = int(shift_ms * 1e-3 * sr)
        shifted = np.zeros_like(wave)
        if k > 0:
            shifted[k:] = wave[: wave.size - k]
        else:
            shifted[: wave.size + k] = wave[-k:]
        wave = shifted
    return np.clip(frames, 0, 255).astype(np.uint8), wave.astype(np.float32)


def test_av_features_shape(rng):
    frames, wave = _synthetic_talking_clip(rng)
    f = extract_av_features(frames, 25.0, wave, 16000)
    assert f.vector.shape == (len(AV_FEATURE_NAMES),)
    assert np.isfinite(f.vector).all()


def test_av_unavailable_without_audio(rng):
    frames, _ = _synthetic_talking_clip(rng)
    f = extract_av_features(frames, 25.0, np.zeros(10, dtype=np.float32), 16000)
    assert not f.available and f.reason_unavailable


@pytest.mark.parametrize("shift_ms", [200.0, -240.0, 320.0])
def test_av_recovers_injected_offset(shift_ms):
    """A known audio offset must be recovered, in the right direction and roughly right size."""
    frames, aligned = _synthetic_talking_clip(np.random.default_rng(4))
    _, offset = _synthetic_talking_clip(np.random.default_rng(4), shift_ms=shift_ms)
    a = extract_av_features(frames, 25.0, aligned, 16000).as_dict()
    b = extract_av_features(frames, 25.0, offset, 16000).as_dict()
    assert a["abs_peak_lag_ms"] <= 80.0, "aligned clip should show near-zero offset"
    assert b["abs_peak_lag_ms"] > a["abs_peak_lag_ms"]
    # Recovered within one frame period (40 ms at 25 fps) of the injected value.
    assert abs(abs(b["global_peak_lag_ms"]) - abs(shift_ms)) <= 45.0


def test_av_correlation_higher_when_synchronised(rng):
    frames, aligned = _synthetic_talking_clip(np.random.default_rng(6))
    _, unrelated = _synthetic_talking_clip(np.random.default_rng(99))
    a = extract_av_features(frames, 25.0, aligned, 16000).as_dict()
    b = extract_av_features(frames, 25.0, unrelated, 16000).as_dict()
    assert a["global_peak_corr"] > b["global_peak_corr"]


def test_visual_activity_curve_is_aperture_not_velocity(rng):
    """The curve must track mouth openness, not its rate of change.

    Regression test: using the frame difference put the visual signal a quarter-cycle out
    of phase with the audio envelope, producing a spurious ~90 ms offset on clips that were
    perfectly synchronised.
    """
    frames, _ = _synthetic_talking_clip(np.random.default_rng(11))
    curve, concentration = visual_activity_curve(frames)
    assert curve.shape[0] == frames.shape[0]
    assert 0.0 <= concentration <= 1.0
    # Darkening the ROI raises the curve, so it correlates with the drawn openness.
    roi_mean = frames[:, 30:42, 18:30, :].mean(axis=(1, 2, 3))
    assert np.corrcoef(curve, -roi_mean)[0, 1] > 0.5
