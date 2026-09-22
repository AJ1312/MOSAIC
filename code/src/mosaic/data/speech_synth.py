"""Procedural speech synthesis and simulated vocoder artefacts.

**SYNTHETIC DEMO DATA.** See ``mosaic.data.__init__``.

"Real" audio here is a source-filter speech simulation: a jittered glottal pulse train
driving a time-varying formant cascade, with aspiration noise, per-phone amplitude
envelopes, a short room impulse response and a realistic noise floor. It is not recorded
speech, but it reproduces the properties the audio branch actually keys on — cycle-to-cycle
jitter and shimmer, a full-band harmonic structure, coherent STFT phase, and a 4-8 Hz
syllabic modulation peak.

"Fake" audio simulates the artefact families that neural vocoders and TTS systems are
repeatedly reported to leave behind:

  * **mel bottleneck + Griffin-Lim resynthesis** — magnitude-only reconstruction with
    iteratively estimated phase. This is the single most faithful part of the simulation:
    Griffin-Lim genuinely destroys inter-frame phase coherence in the same way a
    magnitude-domain vocoder does, and it is what the group-delay and phase-linearity
    features detect.
  * **band limitation** — the steep high-frequency cliff typical of systems whose
    acoustic model was trained at a lower bandwidth.
  * **spectral over-smoothing** — loss of fine harmonic detail from the mel projection.
  * **periodicity flattening** — removal of natural jitter/shimmer, the "too clean" tell.
  * **digital silence flooring** — synthesised silence with no noise floor at all.

These are *simulations of documented artefact classes*, not samples from a real TTS or
voice-conversion model, and are labelled as such wherever results derived from them appear.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import signal

# --------------------------------------------------------------------------------------
# Speaker / phone inventory
# --------------------------------------------------------------------------------------

# (F1, F2, F3) in Hz for vowel-like targets, plus a voicing flag and a bandwidth scale.
_VOWELS: dict[str, tuple[float, float, float]] = {
    "a": (730.0, 1090.0, 2440.0),
    "i": (270.0, 2290.0, 3010.0),
    "u": (300.0, 870.0, 2240.0),
    "e": (530.0, 1840.0, 2480.0),
    "o": (570.0, 840.0, 2410.0),
    "ae": (660.0, 1720.0, 2410.0),
    "schwa": (500.0, 1500.0, 2500.0),
}

# Consonant-like phones: (kind, F1, F2, F3)
#   "fric" = unvoiced fricative (noise excitation, high-frequency emphasis)
#   "stop" = closure + burst
#   "nasal" = voiced, heavily damped, low first formant
_CONSONANTS: dict[str, tuple[str, float, float, float]] = {
    "s": ("fric", 4000.0, 6000.0, 7500.0),
    "f": ("fric", 1200.0, 3500.0, 6000.0),
    "sh": ("fric", 2200.0, 3800.0, 5200.0),
    "t": ("stop", 1800.0, 3000.0, 4200.0),
    "k": ("stop", 1300.0, 2200.0, 3200.0),
    "p": ("stop", 900.0, 1700.0, 2800.0),
    "m": ("nasal", 300.0, 1100.0, 2300.0),
    "n": ("nasal", 320.0, 1600.0, 2600.0),
    "l": ("nasal", 400.0, 1200.0, 2700.0),
}


@dataclass
class Speaker:
    """A synthetic voice. Randomised per clip so identity is never a class shortcut."""

    f0_base: float          # Hz
    f0_range: float         # semitone-ish span of the intonation contour
    formant_scale: float    # vocal-tract length scaling (0.85 = longer tract, 1.15 = shorter)
    jitter: float           # cycle-to-cycle period perturbation, fraction
    shimmer: float          # cycle-to-cycle amplitude perturbation, fraction
    breathiness: float      # aspiration-noise level relative to the glottal source
    speech_rate: float      # phones per second
    noise_floor_db: float   # background noise level, dBFS
    reverb_t60: float       # seconds

    @staticmethod
    def random(rng: np.random.Generator) -> "Speaker":
        # Two loose voice populations so the corpus is not a single timbre.
        if rng.random() < 0.5:
            f0 = float(rng.uniform(85.0, 155.0))     # lower-pitched
            fscale = float(rng.uniform(0.86, 1.0))
        else:
            f0 = float(rng.uniform(165.0, 255.0))    # higher-pitched
            fscale = float(rng.uniform(1.0, 1.18))
        return Speaker(
            f0_base=f0,
            f0_range=float(rng.uniform(1.5, 4.5)),
            formant_scale=fscale,
            jitter=float(rng.uniform(0.004, 0.016)),
            shimmer=float(rng.uniform(0.02, 0.07)),
            breathiness=float(rng.uniform(0.02, 0.12)),
            speech_rate=float(rng.uniform(6.5, 10.5)),
            noise_floor_db=float(rng.uniform(-62.0, -44.0)),
            reverb_t60=float(rng.uniform(0.05, 0.30)),
        )


@dataclass
class Utterance:
    """A synthesised utterance plus the ground-truth articulation track."""

    wave: np.ndarray            # (N,) float32
    sample_rate: int
    envelope: np.ndarray        # (N,) float32, articulatory openness in [0, 1]
    phones: list[dict[str, Any]] = field(default_factory=list)
    speaker: Speaker | None = None
    # Per-sample ground-truth F0 and voicing, kept so the pitch tracker can be validated
    # against what was actually synthesised rather than against the speaker's nominal
    # base frequency (which the intonation contour departs from by design).
    f0_contour: np.ndarray | None = None
    voiced_mask: np.ndarray | None = None


# --------------------------------------------------------------------------------------
# Phone sequence
# --------------------------------------------------------------------------------------


def _build_phone_sequence(rng: np.random.Generator, duration_s: float, speaker: Speaker
                          ) -> list[dict[str, Any]]:
    """Alternate consonant-ish and vowel-ish phones into words separated by pauses."""
    phones: list[dict[str, Any]] = []
    t = 0.0
    base = 1.0 / speaker.speech_rate
    while t < duration_s:
        # A word of 2-6 phones.
        n_phones = int(rng.integers(2, 7))
        for i in range(n_phones):
            if t >= duration_s:
                break
            if i % 2 == 0 and rng.random() < 0.75:
                name = str(rng.choice(list(_CONSONANTS)))
                kind, f1, f2, f3 = _CONSONANTS[name]
                dur = base * float(rng.uniform(0.45, 0.85))
                openness = 0.25 if kind == "nasal" else (0.15 if kind == "stop" else 0.35)
            else:
                name = str(rng.choice(list(_VOWELS)))
                f1, f2, f3 = _VOWELS[name]
                kind = "vowel"
                dur = base * float(rng.uniform(0.9, 1.8))
                # Open vowels move the jaw further: this drives the mouth aperture.
                openness = float(np.clip(0.35 + f1 / 900.0 * 0.65, 0.3, 1.0))
            dur = min(dur, duration_s - t)
            if dur <= 0:
                break
            phones.append({
                "name": name, "kind": kind, "start": t, "dur": dur,
                "f1": f1 * speaker.formant_scale,
                "f2": f2 * speaker.formant_scale,
                "f3": f3 * speaker.formant_scale,
                "openness": openness,
                "amp": float(rng.uniform(0.7, 1.0)),
            })
            t += dur
        # Inter-word pause.
        if t < duration_s:
            pause = float(rng.uniform(0.06, 0.28))
            pause = min(pause, duration_s - t)
            if pause > 0:
                phones.append({
                    "name": "sil", "kind": "silence", "start": t, "dur": pause,
                    "f1": 500.0, "f2": 1500.0, "f3": 2500.0,
                    "openness": 0.05, "amp": 0.0,
                })
                t += pause
    return phones


def _tracks_from_phones(phones: list[dict[str, Any]], n: int, sr: int
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rasterise the phone sequence into per-sample formant/amplitude/openness tracks."""
    f1 = np.full(n, 500.0)
    f2 = np.full(n, 1500.0)
    f3 = np.full(n, 2500.0)
    amp = np.zeros(n)
    openness = np.zeros(n)
    kind_voiced = np.zeros(n)
    kind_fric = np.zeros(n)
    for ph in phones:
        i0 = int(ph["start"] * sr)
        i1 = min(n, int((ph["start"] + ph["dur"]) * sr))
        if i1 <= i0:
            continue
        f1[i0:i1] = ph["f1"]
        f2[i0:i1] = ph["f2"]
        f3[i0:i1] = ph["f3"]
        # Raised-cosine attack/decay avoids clicks and mimics articulatory inertia.
        seg = i1 - i0
        ramp = np.ones(seg)
        edge = max(1, int(0.012 * sr))
        if seg > 2 * edge:
            w = 0.5 * (1 - np.cos(np.linspace(0, np.pi, edge)))
            ramp[:edge] = w
            ramp[-edge:] = w[::-1]
        else:
            ramp = np.hanning(seg + 2)[1:-1] if seg > 1 else np.ones(seg)
        amp[i0:i1] = ph["amp"] * ramp
        openness[i0:i1] = ph["openness"] * ramp
        if ph["kind"] in ("vowel", "nasal"):
            kind_voiced[i0:i1] = 1.0
        elif ph["kind"] in ("fric", "stop"):
            kind_fric[i0:i1] = 1.0

    # Articulators cannot move instantaneously: low-pass the formant and openness tracks.
    def smooth(x: np.ndarray, ms: float) -> np.ndarray:
        k = max(1, int(ms * 1e-3 * sr))
        kern = np.hanning(k * 2 + 1)
        kern /= kern.sum()
        return np.convolve(x, kern, mode="same")

    return (
        np.stack([smooth(f1, 22.0), smooth(f2, 22.0), smooth(f3, 22.0)]),
        smooth(amp, 6.0),
        smooth(openness, 30.0),
        np.stack([smooth(kind_voiced, 10.0), smooth(kind_fric, 10.0)]),
    )


# --------------------------------------------------------------------------------------
# Source-filter synthesis
# --------------------------------------------------------------------------------------


def _f0_contour(rng: np.random.Generator, n: int, sr: int, speaker: Speaker,
                flatten: bool = False) -> np.ndarray:
    """Intonation contour with declination, micro-prosody and cycle jitter."""
    t = np.arange(n) / sr
    dur = max(t[-1], 1e-6)
    # Overall declination across the utterance.
    contour = np.full(n, speaker.f0_base) * (1.0 - 0.18 * t / dur)
    # A couple of slow prosodic swings.
    for _ in range(int(rng.integers(2, 5))):
        freq = float(rng.uniform(0.3, 1.6))
        phase = float(rng.uniform(0, 2 * np.pi))
        depth = speaker.f0_range / 12.0 * float(rng.uniform(0.3, 1.0))
        contour *= 2.0 ** (depth * np.sin(2 * np.pi * freq * t + phase))
    if not flatten:
        # Micro-prosody: fast, low-amplitude random walk (natural pitch is never smooth).
        walk = np.cumsum(rng.normal(0, 1, n))
        k = max(1, int(0.03 * sr))
        kern = np.hanning(2 * k + 1)
        kern /= kern.sum()
        walk = np.convolve(walk, kern, mode="same")
        walk /= (np.abs(walk).max() + 1e-9)
        contour *= 2.0 ** (0.02 * walk)
    return np.clip(contour, 60.0, 400.0)


def _glottal_excitation(rng: np.random.Generator, f0: np.ndarray, amp: np.ndarray,
                        sr: int, speaker: Speaker, flatten: bool = False) -> np.ndarray:
    """Impulse-train excitation with per-cycle jitter and shimmer."""
    n = f0.size
    exc = np.zeros(n)
    jitter = 0.0 if flatten else speaker.jitter
    shimmer = 0.0 if flatten else speaker.shimmer
    pos = 0.0
    while pos < n - 1:
        i = int(pos)
        period = sr / max(f0[i], 1e-6)
        if jitter > 0:
            period *= 1.0 + rng.normal(0.0, jitter)
        period = max(period, 2.0)
        gain = amp[i]
        if shimmer > 0:
            gain *= 1.0 + rng.normal(0.0, shimmer)
        # Fractional-delay impulse so the period is not quantised to the sample grid
        # (quantisation would itself be an artefact the detector could latch onto).
        frac = pos - i
        if i + 1 < n:
            exc[i] += gain * (1.0 - frac)
            exc[i + 1] += gain * frac
        pos += period
    # Shape impulses into glottal pulses: a 2-pole low-pass approximates the open phase.
    b, a = signal.butter(2, min(0.98, 700.0 / (sr / 2)), btype="low")
    return signal.lfilter(b, a, exc)


def _time_varying_formants(x: np.ndarray, formants: np.ndarray, sr: int,
                           block_ms: float = 8.0, bandwidths=(80.0, 110.0, 160.0)
                           ) -> np.ndarray:
    """Cascade of three resonators whose centre frequencies track the phone sequence.

    Processed blockwise with carried filter state so coefficients can vary over time
    without introducing discontinuities.
    """
    n = x.size
    block = max(16, int(block_ms * 1e-3 * sr))
    out = np.zeros(n)
    states = [np.zeros(2) for _ in bandwidths]
    nyq = sr / 2
    for start in range(0, n, block):
        end = min(n, start + block)
        seg = x[start:end]
        mid = (start + end) // 2
        for k, bw in enumerate(bandwidths):
            fc = float(np.clip(formants[k, min(mid, n - 1)], 90.0, nyq * 0.95))
            # Standard 2-pole resonator.
            r = np.exp(-np.pi * bw / sr)
            theta = 2 * np.pi * fc / sr
            a = np.array([1.0, -2 * r * np.cos(theta), r * r])
            b = np.array([1.0 - r, 0.0, 0.0])
            seg, states[k] = signal.lfilter(b, a, seg, zi=states[k])
        out[start:end] = seg
    return out


def _reverb(x: np.ndarray, sr: int, t60: float, rng: np.random.Generator,
            wet: float = 0.28) -> np.ndarray:
    """Short exponentially-decaying noise impulse response.

    The direct path must dominate. Normalising the whole IR to unit energy — the obvious
    thing to write — makes the diffuse tail carry an order of magnitude more energy than
    the direct sound, which is a cathedral rather than a room and smears the periodicity
    that pitch, jitter and shimmer measurement depend on. Here the tail is normalised to
    unit energy first and then scaled to ``wet`` *relative* to a unit direct path, giving
    a realistic direct-to-reverberant ratio.
    """
    if t60 <= 0.01:
        return x
    length = int(t60 * sr)
    if length < 8:
        return x
    tail = rng.normal(0, 1, length) * np.exp(-6.9 * np.arange(length) / length)
    tail /= np.sqrt((tail**2).sum()) + 1e-12
    ir = wet * tail
    ir[0] += 1.0
    ir /= np.sqrt((ir**2).sum()) + 1e-12
    return signal.fftconvolve(x, ir, mode="full")[: x.size]


def synthesize_utterance(
    rng: np.random.Generator,
    duration_s: float,
    sample_rate: int,
    speaker: Speaker | None = None,
    *,
    flatten_periodicity: bool = False,
) -> Utterance:
    """Synthesise one 'real' utterance."""
    speaker = speaker or Speaker.random(rng)
    n = int(duration_s * sample_rate)
    phones = _build_phone_sequence(rng, duration_s, speaker)
    formants, amp, openness, kinds = _tracks_from_phones(phones, n, sample_rate)
    voiced, fric = kinds[0], kinds[1]

    f0 = _f0_contour(rng, n, sample_rate, speaker, flatten=flatten_periodicity)
    exc = _glottal_excitation(rng, f0, amp, sample_rate, speaker, flatten=flatten_periodicity)
    exc *= voiced

    # Aspiration/frication noise, high-pass shaped.
    noise = rng.normal(0, 1, n)
    b_hp, a_hp = signal.butter(2, min(0.98, 1500.0 / (sample_rate / 2)), btype="high")
    noise_hp = signal.lfilter(b_hp, a_hp, noise)
    exc = exc + noise_hp * amp * (fric * 0.55 + voiced * speaker.breathiness)

    voiced_out = _time_varying_formants(exc, formants, sample_rate)
    # Lip radiation is approximately a first-order differentiator.
    out = np.diff(voiced_out, prepend=voiced_out[:1])
    out = _reverb(out, sample_rate, speaker.reverb_t60, rng)

    # Bring the speech to the reference level *before* mixing in the noise floor.
    # ``noise_floor_db`` is specified in dBFS, so it is only meaningful once the signal
    # sits at a known level: adding it to the raw filter output instead means the floor
    # is referenced to nothing in particular. The lip-radiation differentiator leaves the
    # raw output around -90 dBFS, so mixing at that point buries the speech under a noise
    # floor tens of times louder than itself.
    out = _normalise(out)

    # Noise floor: real recordings are never digitally silent.
    floor_amp = 10 ** (speaker.noise_floor_db / 20.0)
    pink = np.cumsum(rng.normal(0, 1, n))
    pink -= pink.mean()
    pink /= (np.abs(pink).max() + 1e-9)
    out = out + floor_amp * (0.6 * pink + 0.4 * rng.normal(0, 1, n))

    out = _normalise(out)
    return Utterance(
        wave=out.astype(np.float32),
        sample_rate=sample_rate,
        envelope=_articulation_envelope(openness, amp),
        phones=phones,
        speaker=speaker,
        f0_contour=f0.astype(np.float32),
        voiced_mask=(voiced > 0.5),
    )


def _articulation_envelope(openness: np.ndarray, amp: np.ndarray) -> np.ndarray:
    """Ground-truth mouth-opening signal that the video renderer will follow."""
    env = openness * (0.35 + 0.65 * np.clip(amp, 0, 1))
    m = env.max()
    return (env / m if m > 0 else env).astype(np.float32)


def _normalise(x: np.ndarray, target_rms: float = 0.06) -> np.ndarray:
    """Fix loudness across the whole corpus.

    Loudness is held constant on purpose: if fake clips were systematically louder or
    quieter than real ones, a detector could reach high accuracy by measuring RMS and
    learning nothing about synthesis artefacts. This is a corpus-level confound control,
    the audio counterpart of L0's canonical re-encode.
    """
    x = x - float(np.mean(x))
    rms = float(np.sqrt((x**2).mean()) + 1e-12)
    x = x * (target_rms / rms)
    # Soft-limit only the samples that would clip, then restore RMS. Rescaling the whole
    # signal by its peak (the naive approach) lets one outlier sample dictate the loudness
    # of the entire clip, which would make loudness correlate with processing history —
    # precisely the confound this function exists to remove.
    peak = float(np.abs(x).max())
    if peak > 0.95:
        over = np.abs(x) > 0.95
        x[over] = np.sign(x[over]) * (0.95 + 0.04 * np.tanh((np.abs(x[over]) - 0.95) / 0.04))
        rms = float(np.sqrt((x**2).mean()) + 1e-12)
        x = x * (target_rms / rms)
    return np.clip(x, -0.999, 0.999)


# --------------------------------------------------------------------------------------
# Simulated vocoder / TTS artefacts
# --------------------------------------------------------------------------------------


def _mel_matrix(n_fft: int, sr: int, n_mels: int) -> np.ndarray:
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sr / 2, n_bins)
    mel_pts = np.linspace(hz_to_mel(0.0), hz_to_mel(sr / 2), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    fb = np.zeros((n_mels, n_bins))
    for i in range(n_mels):
        lo, ctr, hi = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        left = (fft_freqs - lo) / max(ctr - lo, 1e-9)
        right = (hi - fft_freqs) / max(hi - ctr, 1e-9)
        fb[i] = np.clip(np.minimum(left, right), 0, None)
    return fb


def _stft_kw(n_fft: int, hop: int) -> dict[str, Any]:
    # ``boundary='zeros'`` + ``padded=True`` is the configuration that satisfies NOLA, so
    # the analysis/synthesis round-trip is genuinely invertible. Without it the first and
    # last frames reconstruct incorrectly and the resulting edge transient dominates the
    # signal's peak, which then corrupts loudness normalisation downstream.
    return dict(nperseg=n_fft, noverlap=n_fft - hop, window="hann",
                boundary="zeros", padded=True)


def _istft_kw(n_fft: int, hop: int) -> dict[str, Any]:
    # ``istft`` takes ``boundary`` as a bool and has no ``padded`` argument.
    return dict(nperseg=n_fft, noverlap=n_fft - hop, window="hann", boundary=True)


def _match_ltas(new_mag: np.ndarray, ref_mag: np.ndarray, strength: float = 1.0
                ) -> np.ndarray:
    """Rescale each frequency bin so the long-term average spectrum matches ``ref_mag``.

    Without this, the mel projection/pseudo-inverse leaves a large *global* spectral-tilt
    signature, and the whole fake class becomes separable on a single feature
    (``spectral_slope`` alone reached AUC 1.000 before this was added) regardless of how
    subtle the intended artefact was. That is not how a competent vocoder fails: it
    reproduces the spectral envelope well and loses *fine structure* — individual
    harmonics, and phase. Matching the long-term spectrum removes the global giveaway and
    leaves exactly the local artefact that is supposed to be under test.
    """
    ref = ref_mag.mean(axis=1) + 1e-10
    cur = new_mag.mean(axis=1) + 1e-10
    gain = (ref / cur) ** strength
    return new_mag * gain[:, None]


def _griffin_lim(mag: np.ndarray, n_fft: int, hop: int, iters: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Magnitude-only reconstruction with iteratively estimated phase.

    This is what makes the simulated fake audio genuinely detectable by phase features
    rather than by an arbitrary tell: Griffin-Lim converges to a signal whose STFT
    magnitude matches the target but whose inter-frame phase advance is inconsistent
    with any real excitation, exactly the failure mode magnitude-domain vocoders exhibit.
    """
    fwd = _stft_kw(n_fft, hop)
    inv = _istft_kw(n_fft, hop)
    angles = np.exp(2j * np.pi * rng.random(mag.shape))
    x = signal.istft(mag * angles, **inv)[1]
    for _ in range(iters):
        _, _, s = signal.stft(x, **fwd)
        # Match frame count in case the istft/stft round-trip changes it by one.
        m = min(s.shape[1], mag.shape[1])
        angles = np.exp(1j * np.angle(s[:, :m]))
        x = signal.istft(mag[:, :m] * angles, **inv)[1]
    return x


#: Difficulty gradient for simulated fakes. A corpus where every fake carries every
#: artefact at full strength makes the audio task trivial and would make the uncertainty
#: machinery untestable — there would never be a genuinely ambiguous clip. Sampling a
#: quality level per clip produces the spread of easy and hard cases that fusion,
#: calibration and abstention actually need in order to be exercised.
FAKE_AUDIO_QUALITY_LEVELS = ("low", "medium", "high")
FAKE_AUDIO_QUALITY_WEIGHTS = (0.30, 0.40, 0.30)


def sample_fake_audio_ops(rng: np.random.Generator, quality: str | None = None
                          ) -> tuple[str, list[str]]:
    """Pick a fake-audio quality level and the artefact ops that go with it."""
    if quality is None:
        quality = str(rng.choice(FAKE_AUDIO_QUALITY_LEVELS, p=FAKE_AUDIO_QUALITY_WEIGHTS))
    if quality == "low":
        # A crude, magnitude-domain synthesiser: everything is wrong at once.
        pool = ["mel_griffinlim", "band_limit", "spectral_oversmooth", "silence_floor"]
        k = int(rng.integers(3, 5))
        ops = list(rng.choice(pool, size=k, replace=False))
        if "mel_griffinlim" not in ops:
            ops[0] = "mel_griffinlim"
    elif quality == "medium":
        pool = ["band_limit", "spectral_oversmooth", "silence_floor"]
        base = "mel_griffinlim" if rng.random() < 0.5 else "mel_phase_preserve"
        ops = [base] + list(rng.choice(pool, size=1, replace=False))
    else:
        # A good modern neural vocoder. Critically, it reconstructs *coherent* phase —
        # HiFi-GAN-class models do not suffer Griffin-Lim's phase problem — so the phase
        # features that catch crude fakes see nothing here and only a faint magnitude
        # over-smoothing remains. These are the clips that should land near the decision
        # boundary with high uncertainty, and they are the reason the corpus is not
        # saturated: without them every fake would be separable on a single feature.
        ops = ["mel_phase_preserve"]
        if rng.random() < 0.35:
            ops.append("spectral_oversmooth")
    return quality, ops


def apply_vocoder_artifacts(
    wave: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    ops: list[str] | None = None,
    quality: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Apply a sampled subset of simulated synthesis artefacts.

    Returns (processed_wave, ops_applied). Duration and loudness are preserved so that
    neither becomes a shortcut feature.
    """
    if ops is None:
        quality, ops = sample_fake_audio_ops(rng, quality)
    quality = quality or "medium"

    n_orig = wave.size
    x = wave.astype(np.float64).copy()
    applied: list[str] = [f"quality={quality}"]

    n_fft, hop = 512, 128

    if "mel_griffinlim" in ops:
        _, _, spec = signal.stft(x, **_stft_kw(n_fft, hop))
        mag = np.abs(spec)
        # More mel bands = less spectral detail lost = a harder clip to catch.
        n_mels = {"low": (48, 70), "medium": (70, 96), "high": (96, 130)}[quality]
        n_mels = int(rng.integers(*n_mels))
        iters = {"low": (18, 30), "medium": (30, 50), "high": (50, 90)}[quality]
        fb = _mel_matrix(n_fft, sample_rate, n_mels)
        mel = fb @ mag
        # Pseudo-inverse back to linear: irreversibly loses fine harmonic structure.
        approx = np.clip(np.linalg.pinv(fb) @ mel, 0, None)
        # Crude systems get the envelope wrong too; better ones do not.
        approx = _match_ltas(approx, mag, strength={"low": 0.4, "medium": 0.85, "high": 1.0}[quality])
        x = _griffin_lim(approx, n_fft, hop, int(rng.integers(*iters)), rng)
        applied.append(f"mel_griffinlim(n_mels={n_mels})")

    if "mel_phase_preserve" in ops:
        # Mel bottleneck applied to the magnitude only, with the original phase reused.
        # This models a high-quality neural vocoder: the spectral envelope is slightly
        # over-smoothed, but phase remains physically consistent.
        _, _, spec = signal.stft(x, **_stft_kw(n_fft, hop))
        mag, ph = np.abs(spec), np.angle(spec)
        n_mels = int(rng.integers(*{"low": (48, 70), "medium": (72, 100),
                                    "high": (100, 136)}[quality]))
        fb = _mel_matrix(n_fft, sample_rate, n_mels)
        approx = np.clip(np.linalg.pinv(fb) @ (fb @ mag), 0, None)
        approx = _match_ltas(approx, mag, strength=1.0)
        x = signal.istft(approx * np.exp(1j * ph), **_istft_kw(n_fft, hop))[1]
        applied.append(f"mel_phase_preserve(n_mels={n_mels})")

    if "spectral_oversmooth" in ops:
        _, _, spec = signal.stft(x, **_stft_kw(n_fft, hop))
        mag, phase = np.abs(spec), np.angle(spec)
        width = int(rng.integers(3, 7))
        kern = np.hanning(width * 2 + 1)
        kern /= kern.sum()
        smoothed = np.apply_along_axis(lambda c: np.convolve(c, kern, mode="same"), 0, mag)
        x = signal.istft(smoothed * np.exp(1j * phase), **_istft_kw(n_fft, hop))[1]
        applied.append(f"spectral_oversmooth(width={width})")

    if "band_limit" in ops:
        cutoff = float(rng.uniform(*{"low": (4800.0, 6200.0), "medium": (6200.0, 7000.0),
                                     "high": (7000.0, 7600.0)}[quality]))
        nyq = sample_rate / 2
        if cutoff < nyq * 0.98:
            b, a = signal.butter(8, cutoff / nyq, btype="low")
            x = signal.lfilter(b, a, x)
            applied.append(f"band_limit(cutoff={cutoff:.0f}Hz)")

    if "silence_floor" in ops:
        # Gate low-energy regions to (near-)digital silence: synthetic silence has no
        # room tone, which is a strong and genuinely diagnostic cue.
        frame = max(1, int(0.02 * sample_rate))
        n_frames = x.size // frame
        env = np.abs(x[: n_frames * frame].reshape(n_frames, frame)).max(axis=1)
        thr = np.percentile(env, 25)
        gate = np.repeat((env > thr).astype(np.float64), frame)
        gate = np.concatenate([gate, np.ones(x.size - gate.size)])
        k = max(1, int(0.005 * sample_rate))
        kern = np.hanning(2 * k + 1)
        kern /= kern.sum()
        gate = np.convolve(gate, kern, mode="same")
        x = x * (gate + 0.0015)
        applied.append("silence_floor")

    # Restore exact duration: length must never encode the label.
    if x.size < n_orig:
        x = np.pad(x, (0, n_orig - x.size))
    x = x[:n_orig]
    return _normalise(x).astype(np.float32), applied


def resynthesize_same_content(utt: Utterance, rng: np.random.Generator
                              ) -> tuple[np.ndarray, list[str]]:
    """A voice-conversion-style fake: same timing and articulation, synthetic voice.

    This is the case that stops the audiovisual branch from being a proxy for the audio
    branch — the audio is fake but still lip-synchronous, so AV analysis sees nothing
    wrong and only the audio branch's own evidence can catch it.
    """
    return apply_vocoder_artifacts(utt.wave, utt.sample_rate, rng)


def time_shift(wave: np.ndarray, sample_rate: int, shift_ms: float) -> np.ndarray:
    """Shift audio in time (positive = audio lags video), preserving length."""
    shift = int(round(shift_ms * 1e-3 * sample_rate))
    if shift == 0:
        return wave.copy()
    out = np.zeros_like(wave)
    if shift > 0:
        out[shift:] = wave[: wave.size - shift]
    else:
        out[: wave.size + shift] = wave[-shift:]
    return out


def nonlinear_timewarp(wave: np.ndarray, sample_rate: int, rng: np.random.Generator,
                       max_drift_ms: float = 220.0) -> np.ndarray:
    """Slow non-linear drift between audio and video timelines, length preserved."""
    n = wave.size
    t = np.arange(n)
    drift = max_drift_ms * 1e-3 * sample_rate
    phase = float(rng.uniform(0, 2 * np.pi))
    cycles = float(rng.uniform(0.5, 1.5))
    warp = t + drift * np.sin(2 * np.pi * cycles * t / n + phase)
    warp = np.clip(warp, 0, n - 1)
    return np.interp(t, np.arange(n), wave[warp.astype(int)]).astype(np.float32)
