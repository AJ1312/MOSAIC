"""Builds the labelled synthetic audio-video corpus.

**SYNTHETIC DEMO DATA.** See ``mosaic.data.__init__``.

Design decisions that make this corpus useful rather than circular
-----------------------------------------------------------------

**1. Source encodings vary independently of the label.** Every clip is written with a
randomly sampled container, resolution, frame rate, CRF and audio codec, drawn from the
same distribution regardless of class. Real uploads are heterogeneous, and if fake clips
were systematically encoded differently a detector could reach near-perfect accuracy by
reading container metadata alone. Holding the encode distribution label-independent means
L0 canonicalisation has real work to do and the C2 leakage audit is a genuine test rather
than a formality.

**2. A deliberately leaky variant exists for the motivation figure.** ``leak_mode=True``
correlates duration/CRF/frame-rate with the label on purpose. Running the trivial
metadata-only classifier on the leaky corpus before and after canonicalisation reproduces
the shape of the clip-length-leak finding (arXiv:2606.31004) on our own data, and shows
L0 closing it.

**3. Synchronisation is not a proxy for either modality.** The naive corpus design — real
clips synced, all fake clips desynced — would let the audiovisual branch alone solve the
whole task, and the fusion logic would never be exercised. Instead:

  =========================  ===========================================================
  class                      synchronisation
  =========================  ===========================================================
  real video + real audio    always synchronised
  fake video + real audio    70% desynced (reenactment), 30% synced (appearance-only swap)
  real video + fake audio    70% desynced (dubbed), 30% synced (voice conversion keeps timing)
  fake video + fake audio    50% synced (jointly generated), 50% desynced (assembled)
  =========================  ===========================================================

The two cases that matter most are the ones that punish over-reliance on any single
branch: a *jointly generated* fake is perfectly lip-synced, so AV analysis sees nothing
wrong and only the per-modality branches can catch it; and a *voice-converted* fake keeps
the original timing, so again sync is clean while the audio is synthetic. Conversely, a
desynced clip tells you *something* was manipulated but not *which* modality — which is
exactly why AV evidence enters fusion as a coupling term rather than a verdict.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import SYNTHETIC_WARNING
from .speech_synth import (
    Speaker,
    apply_vocoder_artifacts,
    nonlinear_timewarp,
    sample_fake_audio_ops,
    synthesize_utterance,
    time_shift,
)
from .video_synth import Scene, apply_visual_manipulations, render_clip, sample_fake_video_ops

# --------------------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------------------

#: The four ground-truth combinations. UNKNOWN is a *verdict*, never a label.
CLASSES = ("real_video_real_audio", "fake_video_real_audio",
           "real_video_fake_audio", "fake_video_fake_audio")
CLASS_SHORT = {"real_video_real_audio": "RR", "fake_video_real_audio": "FR",
               "real_video_fake_audio": "RF", "fake_video_fake_audio": "FF"}


@dataclass
class CorpusVariant:
    """Knobs that define a generator 'family'.

    Variant A is the primary corpus. Variant B shifts speaker population, scene
    statistics, manipulation strength and source-encode distribution, giving a genuine
    distribution shift for the C6 cross-dataset control — a model tuned on A has not seen
    B's parameterisation.
    """

    name: str = "A"
    f0_shift: float = 1.0
    speech_rate_shift: float = 1.0
    noise_sigma_scale: float = 1.0
    quality_weights_video: tuple[float, float, float] = (0.30, 0.40, 0.30)
    quality_weights_audio: tuple[float, float, float] = (0.30, 0.40, 0.30)
    render_size: int = 320
    source_crf_range: tuple[int, int] = (18, 30)
    source_fps_choices: tuple[int, ...] = (24, 25, 30)
    source_scale_choices: tuple[int, ...] = (256, 320, 384)
    source_containers: tuple[str, ...] = ("mp4", "mkv", "mov")
    source_audio_codecs: tuple[str, ...] = ("aac", "libmp3lame", "pcm_s16le")


VARIANT_A = CorpusVariant(name="A")
VARIANT_B = CorpusVariant(
    name="B",
    f0_shift=1.18,
    speech_rate_shift=0.85,
    noise_sigma_scale=1.5,
    # Variant B leans toward subtler manipulations: a harder, differently-distributed set.
    quality_weights_video=(0.15, 0.35, 0.50),
    quality_weights_audio=(0.15, 0.35, 0.50),
    render_size=288,
    source_crf_range=(20, 34),
    source_fps_choices=(20, 30),
    source_scale_choices=(224, 288, 352),
    source_containers=("mp4", "mkv"),
    source_audio_codecs=("aac", "libopus"),
)


@dataclass
class ClipSpec:
    clip_id: str
    label: str
    split: str
    seed: int
    variant: str = "A"
    duration_s: float = 4.0
    leak_mode: bool = False


@dataclass
class ClipRecord:
    """One row of the dataset manifest."""

    clip_id: str
    path: str
    label: str
    label_short: str
    video_fake: bool
    audio_fake: bool
    sync_state: str            # "synced" | "desynced"
    desync_mode: str | None
    desync_ms: float | None
    split: str
    variant: str
    seed: int
    is_synthetic: bool
    video_quality: str | None
    audio_quality: str | None
    video_ops: list[str]
    audio_ops: list[str]
    source_container: str
    source_codec: str
    source_audio_codec: str
    source_crf: int
    source_fps: int
    source_scale: int
    duration_s: float
    provenance_scenario: str
    file_sha256: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------------------
# Provenance scenarios (simulated — see l2_provenance for how these are consumed)
# --------------------------------------------------------------------------------------

PROVENANCE_SCENARIOS = (
    "none",             # no manifest, no watermark — the expected common case
    "c2pa_only",
    "watermark_only",
    "both_agree",
    "clash",            # manifest and watermark contradict -> Integrity Clash
)
PROVENANCE_WEIGHTS = (0.60, 0.12, 0.12, 0.08, 0.08)


def _sample_provenance(rng: np.random.Generator, video_fake: bool, audio_fake: bool
                       ) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
    """Sample a provenance scenario and build the matching sidecar payloads.

    Returns (scenario, c2pa_sidecar, watermark_sidecar). Either sidecar may be None.
    Both are explicitly marked simulated; nothing here is cryptographically signed and
    L2 refuses to treat them as verified.
    """
    scenario = str(rng.choice(PROVENANCE_SCENARIOS, p=PROVENANCE_WEIGHTS))
    is_fake = video_fake or audio_fake

    def manifest(claim_ai: bool) -> dict[str, Any]:
        return {
            "_simulated": True,
            "_note": "SIMULATED C2PA-shaped manifest. Not cryptographically signed.",
            "claim_generator": "MosaicSyntheticCorpus/0.1",
            "assertions": [
                {"label": "c2pa.actions", "data": {"actions": [
                    {"action": "c2pa.created",
                     "digitalSourceType": (
                         "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
                         if claim_ai else
                         "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture")},
                ]}},
            ],
            "claims_ai_generated": claim_ai,
        }

    def watermark(detected: bool, modality: str = "both") -> dict[str, Any]:
        return {
            "_simulated": True,
            "_note": ("SIMULATED watermark-detector response. No real vendor detector "
                      "(SynthID or otherwise) was queried."),
            "detector": "simulated_vendor_detector",
            "modality": modality,
            "watermark_detected": detected,
            "confidence": float(rng.uniform(0.82, 0.99)) if detected else float(rng.uniform(0.01, 0.15)),
        }

    if scenario == "none":
        return scenario, None, None
    if scenario == "c2pa_only":
        return scenario, manifest(claim_ai=is_fake), None
    if scenario == "watermark_only":
        return scenario, None, watermark(detected=is_fake)
    if scenario == "both_agree":
        return scenario, manifest(claim_ai=is_fake), watermark(detected=is_fake)
    # Integrity Clash: the manifest asserts human capture while the watermark says AI
    # (or the reverse). Both "pass" on their own; only cross-checking catches it.
    if rng.random() < 0.5:
        return scenario, manifest(claim_ai=False), watermark(detected=True)
    return scenario, manifest(claim_ai=True), watermark(detected=False)


# --------------------------------------------------------------------------------------
# Single-clip generation
# --------------------------------------------------------------------------------------


def _variant(name: str) -> CorpusVariant:
    return {"A": VARIANT_A, "B": VARIANT_B}[name]


def _sample_speaker(rng: np.random.Generator, var: CorpusVariant) -> Speaker:
    sp = Speaker.random(rng)
    return Speaker(
        f0_base=sp.f0_base * var.f0_shift,
        f0_range=sp.f0_range,
        formant_scale=sp.formant_scale,
        jitter=sp.jitter,
        shimmer=sp.shimmer,
        breathiness=sp.breathiness,
        speech_rate=sp.speech_rate * var.speech_rate_shift,
        noise_floor_db=sp.noise_floor_db,
        reverb_t60=sp.reverb_t60,
    )


def _sample_scene(rng: np.random.Generator, var: CorpusVariant, h: int, w: int) -> Scene:
    sc = Scene.random(rng, h, w)
    sc.noise_sigma = sc.noise_sigma * var.noise_sigma_scale
    return sc


def _decide_sync(rng: np.random.Generator, video_fake: bool, audio_fake: bool) -> bool:
    """True if the clip should end up desynchronised. See module docstring."""
    if not video_fake and not audio_fake:
        return False
    if video_fake and audio_fake:
        return rng.random() < 0.50
    return rng.random() < 0.70


def generate_clip(spec: ClipSpec, out_dir: str | Path) -> ClipRecord:
    """Generate one labelled clip and write it to disk with its provenance sidecars."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    var = _variant(spec.variant)
    rng = np.random.default_rng(spec.seed)

    video_fake = spec.label in ("fake_video_real_audio", "fake_video_fake_audio")
    audio_fake = spec.label in ("real_video_fake_audio", "fake_video_fake_audio")

    sr = 16000
    fps_render = 25.0
    n_frames = int(spec.duration_s * fps_render)
    size = var.render_size

    # ---- content ----------------------------------------------------------------------
    speaker = _sample_speaker(rng, var)
    utt_main = synthesize_utterance(rng, spec.duration_s, sr, speaker)

    desynced = _decide_sync(rng, video_fake, audio_fake)
    desync_mode: str | None = None
    desync_ms: float | None = None

    # The envelope that drives the mouth, and the audio that is actually written.
    drive_env = utt_main.envelope
    audio = utt_main.wave

    if audio_fake:
        if desynced:
            # Dubbed with different content: a second utterance, different words. Some
            # dubs come from a full text-to-speech system rather than voice conversion,
            # and those have an unnaturally regular excitation — flattened jitter and
            # shimmer — which is what the voice-quality features are there to catch.
            other = synthesize_utterance(
                rng, spec.duration_s, sr, _sample_speaker(rng, var),
                flatten_periodicity=bool(rng.random() < 0.5),
            )
            audio, audio_ops = apply_vocoder_artifacts(
                other.wave, sr, rng,
                quality=str(rng.choice(("low", "medium", "high"), p=var.quality_weights_audio)))
            desync_mode = "content_mismatch_dub"
        else:
            # Voice conversion: same timing and articulation, synthetic voice.
            audio, audio_ops = apply_vocoder_artifacts(
                utt_main.wave, sr, rng,
                quality=str(rng.choice(("low", "medium", "high"), p=var.quality_weights_audio)))
        audio_quality = next((o.split("=")[1] for o in audio_ops if o.startswith("quality=")), None)
    else:
        audio_ops, audio_quality = [], None
        if desynced:
            # Real audio, manipulated video: reenactment drives the mouth from other content.
            mode = rng.random()
            if mode < 0.55:
                other = synthesize_utterance(rng, spec.duration_s, sr, _sample_speaker(rng, var))
                drive_env = other.envelope
                desync_mode = "reenactment_mismatch"
            elif mode < 0.85:
                desync_ms = float(rng.choice([-1, 1]) * rng.uniform(120.0, 420.0))
                audio = time_shift(audio, sr, desync_ms)
                desync_mode = "constant_offset"
            else:
                audio = nonlinear_timewarp(audio, sr, rng)
                desync_mode = "nonlinear_drift"

    if audio_fake and desynced and desync_mode == "content_mismatch_dub" and rng.random() < 0.35:
        # Some dubs are also globally offset.
        desync_ms = float(rng.choice([-1, 1]) * rng.uniform(120.0, 420.0))
        audio = time_shift(audio, sr, desync_ms)
        desync_mode = "content_mismatch_dub+offset"

    # Natural capture jitter, well within perceptual tolerance, applied to every clip so
    # that "exactly zero lag" is never itself a cue.
    jitter_ms = float(rng.uniform(-25.0, 25.0))
    audio = time_shift(audio, sr, jitter_ms)

    # ---- render -----------------------------------------------------------------------
    scene = _sample_scene(rng, var, size, size)
    clip = render_clip(rng, drive_env, n_frames, fps_render, size, size, scene)
    frames = clip.frames
    video_ops: list[str] = []
    video_quality: str | None = None
    if video_fake:
        quality = str(rng.choice(("low", "medium", "high"), p=var.quality_weights_video))
        frames, video_ops = apply_visual_manipulations(clip, rng, quality=quality)
        video_quality = quality

    # ---- source encode (label-independent by construction) -----------------------------
    if spec.leak_mode:
        # Deliberate confound for the motivation experiment ONLY.
        is_fake = video_fake or audio_fake
        crf = 28 if is_fake else 20
        src_fps = 24 if is_fake else 30
        src_scale = 256 if is_fake else 384
        container = "mp4" if is_fake else "mkv"
        acodec = "aac" if is_fake else "pcm_s16le"
        duration_out = spec.duration_s * (0.8 if is_fake else 1.0)
    else:
        crf = int(rng.integers(*var.source_crf_range))
        src_fps = int(rng.choice(var.source_fps_choices))
        src_scale = int(rng.choice(var.source_scale_choices))
        container = str(rng.choice(var.source_containers))
        acodec = str(rng.choice(var.source_audio_codecs))
        duration_out = spec.duration_s

    path = out_dir / f"{spec.clip_id}.{container}"
    _mux(frames, fps_render, audio, sr, path,
         crf=crf, fps_out=src_fps, scale=src_scale, acodec=acodec,
         duration_s=duration_out)

    # ---- provenance sidecars -----------------------------------------------------------
    scenario, c2pa_side, wm_side = _sample_provenance(rng, video_fake, audio_fake)
    if c2pa_side is not None:
        (out_dir / f"{path.name}.c2pa.json").write_text(json.dumps(c2pa_side, indent=2))
    if wm_side is not None:
        (out_dir / f"{path.name}.watermark.json").write_text(json.dumps(wm_side, indent=2))

    from ..hashing import sha256_file

    return ClipRecord(
        clip_id=spec.clip_id,
        path=str(path),
        label=spec.label,
        label_short=CLASS_SHORT[spec.label],
        video_fake=video_fake,
        audio_fake=audio_fake,
        sync_state="desynced" if desynced else "synced",
        desync_mode=desync_mode,
        desync_ms=desync_ms,
        split=spec.split,
        variant=spec.variant,
        seed=spec.seed,
        is_synthetic=True,
        video_quality=video_quality,
        audio_quality=audio_quality,
        video_ops=video_ops,
        audio_ops=audio_ops,
        source_container=container,
        source_codec="libx264",
        source_audio_codec=acodec,
        source_crf=crf,
        source_fps=src_fps,
        source_scale=src_scale,
        duration_s=duration_out,
        provenance_scenario=scenario,
        file_sha256=sha256_file(path),
    )


def _mux(frames: np.ndarray, fps_in: float, audio: np.ndarray, sr: int, path: Path,
         *, crf: int, fps_out: int, scale: int, acodec: str, duration_s: float) -> None:
    """Write frames + audio to a container using ffmpeg."""
    import soundfile as sf

    tmp = Path(tempfile.mkdtemp(prefix="mosaic_mux_"))
    try:
        raw = tmp / "frames.raw"
        raw.write_bytes(np.ascontiguousarray(frames).tobytes())
        wav = tmp / "audio.wav"
        sf.write(wav, audio, sr, subtype="PCM_16")

        h, w = frames.shape[1], frames.shape[2]
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps_in),
            "-i", str(raw),
            "-i", str(wav),
            "-t", f"{duration_s:.3f}",
            "-c:v", "libx264", "-crf", str(crf), "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-vf", f"scale={scale}:{scale}", "-r", str(fps_out),
            "-c:a", acodec,
        ]
        if acodec in ("aac", "libmp3lame", "libopus"):
            cmd += ["-b:a", "96k"]
        cmd += [str(path)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0 or not path.exists():
            raise RuntimeError(f"ffmpeg mux failed for {path.name}: {proc.stderr.strip()[:500]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------------------
# Corpus assembly
# --------------------------------------------------------------------------------------


def plan_corpus(
    n_per_class: int,
    *,
    variant: str = "A",
    splits: tuple[tuple[str, float], ...] = (("train", 0.6), ("val", 0.15), ("test", 0.25)),
    seed: int = 1337,
    duration_s: float = 4.0,
    leak_mode: bool = False,
    prefix: str = "",
) -> list[ClipSpec]:
    """Plan a class-balanced corpus with disjoint per-clip seeds.

    Class balance is exact, and split assignment is stratified within class so that no
    split is accidentally enriched for one condition.
    """
    rng = np.random.default_rng(seed)
    specs: list[ClipSpec] = []
    counter = 0
    for label in CLASSES:
        idx = np.arange(n_per_class)
        rng.shuffle(idx)
        offset = 0
        for split_name, frac in splits:
            n = int(round(frac * n_per_class))
            if split_name == splits[-1][0]:
                n = n_per_class - offset
            for _ in range(n):
                counter += 1
                specs.append(ClipSpec(
                    clip_id=f"{prefix}{variant}_{CLASS_SHORT[label]}_{counter:05d}",
                    label=label,
                    split=split_name,
                    # Distinct, deterministic seed per clip.
                    seed=int(seed * 1_000_003 + counter),
                    variant=variant,
                    duration_s=duration_s,
                    leak_mode=leak_mode,
                ))
            offset += n
    return specs


def _worker(args: tuple[dict[str, Any], str]) -> dict[str, Any]:
    spec_dict, out_dir = args
    spec = ClipSpec(**spec_dict)
    try:
        return generate_clip(spec, out_dir).to_dict()
    except Exception as exc:  # a failed clip is recorded, never silently dropped
        return ClipRecord(
            clip_id=spec.clip_id, path="", label=spec.label,
            label_short=CLASS_SHORT[spec.label],
            video_fake=spec.label.startswith("fake"),
            audio_fake=spec.label.endswith("fake_audio"),
            sync_state="unknown", desync_mode=None, desync_ms=None,
            split=spec.split, variant=spec.variant, seed=spec.seed, is_synthetic=True,
            video_quality=None, audio_quality=None, video_ops=[], audio_ops=[],
            source_container="", source_codec="", source_audio_codec="",
            source_crf=0, source_fps=0, source_scale=0, duration_s=spec.duration_s,
            provenance_scenario="none",
            error=f"{type(exc).__name__}: {exc}",
        ).to_dict()


def build_corpus(
    specs: list[ClipSpec],
    out_dir: str | Path,
    *,
    workers: int | None = None,
    progress: bool = True,
) -> list[dict[str, Any]]:
    """Generate all clips, in parallel across processes."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, (os.cpu_count() or 2) - 2)

    payload = [(asdict(s), str(out_dir)) for s in specs]
    records: list[dict[str, Any]] = []
    if workers <= 1:
        for i, item in enumerate(payload, 1):
            records.append(_worker(item))
            if progress and i % 10 == 0:
                print(f"  ... {i}/{len(payload)} clips", flush=True)
        return records

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_worker, item): item for item in payload}
        for i, fut in enumerate(as_completed(futures), 1):
            records.append(fut.result())
            if progress and i % 25 == 0:
                print(f"  ... {i}/{len(payload)} clips", flush=True)
    records.sort(key=lambda r: r["clip_id"])
    return records


def write_manifest(records: list[dict[str, Any]], path: str | Path) -> Path:
    """Write the dataset manifest as CSV, with the synthetic-data warning as a header."""
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError("no records to write")

    fields = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write(f"# {SYNTHETIC_WARNING}\n")
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            row = {k: (json.dumps(v) if isinstance(v, list) else v) for k, v in rec.items()}
            writer.writerow(row)
    return path


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Read a manifest written by :func:`write_manifest`."""
    import csv

    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        lines = [ln for ln in fh if not ln.startswith("#")]
    for row in csv.DictReader(lines):
        rec = dict(row)
        for key in ("video_ops", "audio_ops"):
            if rec.get(key):
                try:
                    rec[key] = json.loads(rec[key])
                except json.JSONDecodeError:
                    rec[key] = []
        for key in ("video_fake", "audio_fake", "is_synthetic"):
            rec[key] = str(rec.get(key)).lower() == "true"
        for key in ("source_crf", "source_fps", "source_scale", "seed"):
            rec[key] = int(rec[key]) if rec.get(key) else 0
        for key in ("duration_s", "desync_ms"):
            rec[key] = float(rec[key]) if rec.get(key) else None
        rows.append(rec)
    return rows
