"""L0 — Canonical ingest.

Every input is re-encoded through a *single fixed profile* before any other layer runs.
This is control C1 of the VidAudit protocol (arXiv:2606.31004) implemented as an
architectural default rather than an evaluation-time afterthought.

Why it matters here specifically: a detector trained on non-canonicalised media can
learn container-level or codec-level shortcuts (clip length, bitrate, sample rate,
encoder string) that correlate with class purely because real and fake clips came from
different production pipelines. The June-2026 audit found a 3-feature clip-length
classifier reaching 0.998 LOGO-AUC on an unaudited benchmark. Canonicalising first
closes that channel at the source; ``mosaic.audit`` then *proves* it is closed by running
the trivial baseline and reporting that it fails.

One decode pass produces everything downstream needs:
  * RGB frames                (visual branch, perceptual hash, AV branch)
  * PCM audio                 (audio branch, AV branch, audio perceptual hash)
  * codec motion vectors      (Tier-0 near-zero-cost triage)
  * container/stream metadata (leakage auditing, custody record)
  * embedded C2PA/EXIF-style metadata blocks (L2)
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import CanonicalProfile
from .hashing import sha256_file


class IngestError(RuntimeError):
    """Raised when an input cannot be canonicalised. Never silently swallowed."""


# --------------------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------------------


@dataclass
class MotionVectorFeatures:
    """Statistics over codec motion vectors, extracted during the canonical decode.

    These come from the compressed bitstream the encoder already produced — reading them
    is a parse, not a forward pass (cf. arXiv:2607.19476, arXiv:2311.10788), which is what
    makes Tier-0 near-zero-cost.
    """

    available: bool
    n_frames_with_mvs: int = 0
    mv_count_mean: float = 0.0
    mv_mag_mean: float = 0.0
    mv_mag_std: float = 0.0
    mv_mag_p95: float = 0.0
    mv_zero_fraction: float = 0.0
    mv_angle_entropy: float = 0.0
    mv_block_size_mean: float = 0.0
    mv_temporal_smoothness: float = 0.0
    frame_type_i: float = 0.0
    frame_type_p: float = 0.0
    frame_type_b: float = 0.0
    pkt_bytes_mean: float = 0.0
    pkt_bytes_std: float = 0.0
    reason_unavailable: str | None = None

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.mv_count_mean, self.mv_mag_mean, self.mv_mag_std, self.mv_mag_p95,
            self.mv_zero_fraction, self.mv_angle_entropy, self.mv_block_size_mean,
            self.mv_temporal_smoothness, self.frame_type_i, self.frame_type_p,
            self.frame_type_b, self.pkt_bytes_mean, self.pkt_bytes_std,
        ], dtype=np.float64)

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "mv_count_mean", "mv_mag_mean", "mv_mag_std", "mv_mag_p95",
            "mv_zero_fraction", "mv_angle_entropy", "mv_block_size_mean",
            "mv_temporal_smoothness", "frame_type_i", "frame_type_p",
            "frame_type_b", "pkt_bytes_mean", "pkt_bytes_std",
        ]


@dataclass
class MediaBundle:
    """Decoded canonical media plus everything L1-L4 need from the decode pass."""

    frames: np.ndarray            # (T, H, W, 3) uint8
    fps: float
    audio: np.ndarray             # (N,) float32 in [-1, 1]
    sample_rate: int
    duration_s: float
    has_audio: bool
    motion_vectors: MotionVectorFeatures

    @property
    def n_frames(self) -> int:
        return int(self.frames.shape[0])


@dataclass
class IngestResult:
    """Full record of the L0 transformation — every field lands in the custody record."""

    source_path: str
    canonical_path: str
    source_sha256: str
    canonical_sha256: str
    profile: dict[str, Any]
    ffmpeg_binary_version: str
    ffmpeg_command: list[str]
    source_probe: dict[str, Any]
    canonical_probe: dict[str, Any]
    embedded_metadata: dict[str, Any]
    transformations: list[str]
    ingest_wall_s: float
    media: MediaBundle | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "media"}
        return d


# --------------------------------------------------------------------------------------
# ffmpeg / ffprobe
# --------------------------------------------------------------------------------------


def _require(binary: str) -> str:
    path = shutil.which(binary)
    if path is None:
        raise IngestError(
            f"{binary} not found on PATH. L0 canonical ingest cannot run without it, and "
            "MOSAIC will not fall back to un-canonicalised input because doing so would "
            "reintroduce exactly the codec confound C1 exists to remove."
        )
    return path


def ffmpeg_version() -> str:
    _require("ffmpeg")
    out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=20)
    return out.stdout.splitlines()[0].strip() if out.stdout else "unknown"


def ffprobe(path: str | Path) -> dict[str, Any]:
    """Full container + stream metadata as a dict."""
    _require("ffprobe")
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise IngestError(f"ffprobe failed on {path}: {out.stderr.strip()}")
    return json.loads(out.stdout or "{}")


def probe_summary(probe: dict[str, Any]) -> dict[str, Any]:
    """Condense an ffprobe dump to the fields that matter for auditing and custody."""
    fmt = probe.get("format", {}) or {}
    streams = probe.get("streams", []) or []
    v = next((s for s in streams if s.get("codec_type") == "video"), {})
    a = next((s for s in streams if s.get("codec_type") == "audio"), {})

    def _num(x, cast=float):
        try:
            return cast(x)
        except (TypeError, ValueError):
            return None

    def _rate(text):
        if not text or "/" not in str(text):
            return _num(text)
        num, den = str(text).split("/")
        n, d = _num(num), _num(den)
        return round(n / d, 6) if n is not None and d else None

    return {
        "container": fmt.get("format_name"),
        "duration_s": _num(fmt.get("duration")),
        "size_bytes": _num(fmt.get("size"), int),
        "bit_rate": _num(fmt.get("bit_rate"), int),
        "n_streams": len(streams),
        "video_codec": v.get("codec_name"),
        "width": _num(v.get("width"), int),
        "height": _num(v.get("height"), int),
        "pix_fmt": v.get("pix_fmt"),
        "avg_frame_rate": _rate(v.get("avg_frame_rate")),
        "nb_frames": _num(v.get("nb_frames"), int),
        "video_bit_rate": _num(v.get("bit_rate"), int),
        "has_audio": bool(a),
        "audio_codec": a.get("codec_name"),
        "sample_rate": _num(a.get("sample_rate"), int),
        "channels": _num(a.get("channels"), int),
        "audio_bit_rate": _num(a.get("bit_rate"), int),
        "tags": {k: v_ for k, v_ in (fmt.get("tags") or {}).items()},
    }


def extract_embedded_metadata(path: str | Path) -> dict[str, Any]:
    """Container-level metadata blocks, including any C2PA-shaped payload.

    This does *not* verify anything cryptographically — that is L2's job. It only reports
    what is present, so that "no manifest found" is recorded as an observation rather than
    inferred from a failed parse.
    """
    probe = ffprobe(path)
    fmt = probe.get("format", {}) or {}
    tags = dict(fmt.get("tags") or {})
    stream_tags: list[dict[str, Any]] = []
    for s in probe.get("streams", []) or []:
        st = dict(s.get("tags") or {})
        if st:
            stream_tags.append({"index": s.get("index"), "codec_type": s.get("codec_type"), "tags": st})

    c2pa_keys = [k for k in tags if "c2pa" in k.lower() or "jumbf" in k.lower()]
    return {
        "format_tags": tags,
        "stream_tags": stream_tags,
        "c2pa_shaped_keys": c2pa_keys,
        "has_c2pa_shaped_metadata": bool(c2pa_keys),
    }


# --------------------------------------------------------------------------------------
# Canonical re-encode
# --------------------------------------------------------------------------------------


def canonicalise(
    src: str | Path,
    dst: str | Path,
    profile: CanonicalProfile,
    *,
    overwrite: bool = True,
) -> tuple[list[str], list[str]]:
    """Re-encode ``src`` to ``dst`` under the single canonical profile.

    Returns (ffmpeg_command, transformations_applied).
    """
    _require("ffmpeg")
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    src_probe = probe_summary(ffprobe(src))
    has_audio = src_probe["has_audio"]
    has_video = src_probe["video_codec"] is not None

    if not has_video and not has_audio:
        raise IngestError(f"{src} contains neither a video nor an audio stream")

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    cmd += ["-y"] if overwrite else ["-n"]
    cmd += ["-i", str(src)]
    if has_video:
        cmd += profile.ffmpeg_video_args()
    else:
        # Audio-only source (e.g. an ASVspoof FLAC). Canonicalising the audio track alone
        # is the correct behaviour: inventing a video track to satisfy a fixed profile
        # would fabricate content, and the downstream branches already handle an absent
        # modality by reporting it unavailable rather than guessing.
        cmd += ["-vn"]
    if has_audio:
        cmd += profile.ffmpeg_audio_args()
    else:
        cmd += ["-an"]
    # Applied after the pad filters so every output is exactly analysis_seconds long.
    cmd += profile.ffmpeg_duration_args()
    # Strip all container metadata: a detector must never be able to read the encoder
    # string or creation time. Provenance metadata is captured separately *before* this
    # step and handed to L2, so nothing is lost — it is just removed from the pixels' path.
    cmd += ["-map_metadata", "-1", "-fflags", "+bitexact", "-flags:v", "+bitexact"]
    cmd += [str(dst)]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0 or not dst.exists():
        raise IngestError(f"canonical re-encode failed for {src}: {proc.stderr.strip()}")

    transformations = ([
        f"scale->{profile.width}x{profile.height}",
        f"fps->{profile.fps}",
        f"video_codec->{profile.video_codec}(crf={profile.crf},preset={profile.preset})",
        f"pix_fmt->{profile.pix_fmt}",
        f"gop->{profile.gop_size}(fixed, sc_threshold=0)",
        f"duration->{profile.analysis_seconds}s(fixed K-frame window; trim+clone-pad)",
        "container_metadata->stripped",
    ] if has_video else ["video->absent_in_source(audio-only ingest)",
                         "container_metadata->stripped"])
    if has_audio:
        transformations += [
            f"audio_codec->{profile.audio_codec}",
            f"audio_rate->{profile.audio_sample_rate}",
            f"audio_channels->{profile.audio_channels}",
        ]
    else:
        transformations.append("audio->absent_in_source(none written)")
    return cmd, transformations


# --------------------------------------------------------------------------------------
# Decode: frames + audio + motion vectors in one pass
# --------------------------------------------------------------------------------------


# FFmpeg AVPictureType. PyAV exposes ``frame.pict_type`` as a plain int on some builds
# and as a named enum on others, so both are handled.
_PICT_TYPES = {0: "NONE", 1: "I", 2: "P", 3: "B", 4: "S", 5: "SI", 6: "SP", 7: "BI"}


def _pict_type_letter(pict_type) -> str:
    name = getattr(pict_type, "name", None)
    if name:
        return str(name).upper()
    try:
        return _PICT_TYPES.get(int(pict_type), "NONE")
    except (TypeError, ValueError):
        return str(pict_type).split(".")[-1].upper()


def _mv_entropy(angles: np.ndarray, bins: int = 16) -> float:
    if angles.size == 0:
        return 0.0
    hist, _ = np.histogram(angles, bins=bins, range=(-np.pi, np.pi))
    p = hist.astype(np.float64)
    total = p.sum()
    if total <= 0:
        return 0.0
    p /= total
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum() / np.log2(bins))


def decode_bundle(path: str | Path, profile: CanonicalProfile) -> MediaBundle:
    """Decode canonical media: RGB frames, PCM audio, and codec motion vectors.

    Motion vectors are requested via the codec's ``export_mvs`` flag. If the codec or
    build does not provide them we record *why* they are unavailable rather than
    substituting a synthetic stand-in.
    """
    import av
    from av.codec.context import Flags2
    from av.sidedata.sidedata import Type as SideDataType

    path = Path(path)
    frames: list[np.ndarray] = []
    mv_mags: list[float] = []
    mv_counts: list[int] = []
    mv_angles_all: list[np.ndarray] = []
    mv_block_sizes: list[float] = []
    mv_frame_means: list[float] = []
    frame_types: list[str] = []
    pkt_sizes: list[int] = []
    n_frames_with_mvs = 0
    mv_reason: str | None = None

    with av.open(str(path)) as container:
        if not container.streams.video:
            # Audio-only media is a first-class input, not an error: the audio branch runs
            # and the visual / audiovisual branches report themselves unavailable.
            audio, sr, has_audio = _decode_audio(path, profile)
            if not has_audio:
                raise IngestError(f"{path} yielded neither video frames nor audio samples")
            return MediaBundle(
                frames=np.zeros((0, profile.height, profile.width, 3), dtype=np.uint8),
                fps=float(profile.fps), audio=audio, sample_rate=sr, has_audio=True,
                duration_s=float(audio.size / sr) if sr else 0.0,
                motion_vectors=MotionVectorFeatures(
                    available=False,
                    reason_unavailable="source has no video stream (audio-only media)"),
            )
        vstream = container.streams.video[0]
        # Single-threaded decode on purpose. Frame-threaded decoding spawns worker threads
        # per container, and across a long sequential run those accumulate faster than they
        # are reaped, eventually surfacing as EAGAIN ("Resource temporarily unavailable")
        # from the scaler mid-run. Canonical clips are small enough that threading buys
        # nothing measurable here, and a pipeline that dies 100 clips into a batch is worse
        # than one that decodes a few milliseconds slower.
        vstream.thread_type = "NONE"
        vstream.codec_context.thread_count = 1
        try:
            vstream.codec_context.flags2 |= Flags2.export_mvs
        except Exception as exc:  # pragma: no cover - build-dependent
            mv_reason = f"export_mvs flag unsupported by this PyAV/FFmpeg build: {exc}"

        fps = float(vstream.average_rate) if vstream.average_rate else float(profile.fps)

        for packet in container.demux(vstream):
            if packet.size:
                pkt_sizes.append(int(packet.size))
            for frame in packet.decode():
                frames.append(frame.to_ndarray(format="rgb24"))
                frame_types.append(_pict_type_letter(frame.pict_type))
                mvs = None
                try:
                    mvs = frame.side_data.get(SideDataType.MOTION_VECTORS)
                except Exception:
                    mvs = None
                if mvs is None:
                    continue
                arr = mvs.to_ndarray()
                if arr.size == 0:
                    continue
                n_frames_with_mvs += 1
                # motion_x/motion_y are in units of 1/motion_scale pixels.
                scale = arr["motion_scale"].astype(np.float64)
                scale[scale == 0] = 1.0
                dx = arr["motion_x"].astype(np.float64) / scale
                dy = arr["motion_y"].astype(np.float64) / scale
                mag = np.hypot(dx, dy)
                mv_mags.append(mag)
                mv_counts.append(int(arr.size))
                mv_angles_all.append(np.arctan2(dy, dx))
                mv_block_sizes.append(float(np.mean(arr["w"].astype(np.float64) * arr["h"].astype(np.float64))))
                mv_frame_means.append(float(mag.mean()))

        # Audio is decoded in a second open() because seeking the same container after a
        # full video demux is not reliable across formats.
    audio, sr, has_audio = _decode_audio(path, profile)

    if not frames:
        raise IngestError(f"decoded zero frames from {path}")

    if mv_mags:
        all_mags = np.concatenate(mv_mags)
        all_angles = np.concatenate(mv_angles_all)
        smooth = 0.0
        if len(mv_frame_means) > 2:
            series = np.array(mv_frame_means)
            diffs = np.abs(np.diff(series))
            smooth = float(diffs.mean() / (series.mean() + 1e-9))
        n_ft = max(1, len(frame_types))
        mv = MotionVectorFeatures(
            available=True,
            n_frames_with_mvs=n_frames_with_mvs,
            mv_count_mean=float(np.mean(mv_counts)),
            mv_mag_mean=float(all_mags.mean()),
            mv_mag_std=float(all_mags.std()),
            mv_mag_p95=float(np.percentile(all_mags, 95)),
            mv_zero_fraction=float((all_mags < 1e-6).mean()),
            mv_angle_entropy=_mv_entropy(all_angles),
            mv_block_size_mean=float(np.mean(mv_block_sizes)),
            mv_temporal_smoothness=smooth,
            frame_type_i=frame_types.count("I") / n_ft,
            frame_type_p=frame_types.count("P") / n_ft,
            frame_type_b=frame_types.count("B") / n_ft,
            pkt_bytes_mean=float(np.mean(pkt_sizes)) if pkt_sizes else 0.0,
            pkt_bytes_std=float(np.std(pkt_sizes)) if pkt_sizes else 0.0,
        )
    else:
        n_ft = max(1, len(frame_types))
        mv = MotionVectorFeatures(
            available=False,
            reason_unavailable=mv_reason or (
                "codec produced no MOTION_VECTORS side data (all-intra encode or "
                "decoder did not export them)"
            ),
            frame_type_i=frame_types.count("I") / n_ft,
            frame_type_p=frame_types.count("P") / n_ft,
            frame_type_b=frame_types.count("B") / n_ft,
            pkt_bytes_mean=float(np.mean(pkt_sizes)) if pkt_sizes else 0.0,
            pkt_bytes_std=float(np.std(pkt_sizes)) if pkt_sizes else 0.0,
        )

    stack = np.stack(frames).astype(np.uint8)
    duration = stack.shape[0] / fps if fps else 0.0
    return MediaBundle(
        frames=stack,
        fps=fps,
        audio=audio,
        sample_rate=sr,
        duration_s=float(duration),
        has_audio=has_audio,
        motion_vectors=mv,
    )


def _decode_audio(path: Path, profile: CanonicalProfile) -> tuple[np.ndarray, int, bool]:
    """Decode the audio track to mono float32 at the canonical sample rate."""
    import av

    with av.open(str(path)) as container:
        if not container.streams.audio:
            return np.zeros(0, dtype=np.float32), profile.audio_sample_rate, False
        astream = container.streams.audio[0]
        astream.thread_type = "NONE"
        astream.codec_context.thread_count = 1
        resampler = av.AudioResampler(
            format="fltp", layout="mono", rate=profile.audio_sample_rate
        )
        chunks: list[np.ndarray] = []
        for frame in container.decode(astream):
            for res in resampler.resample(frame):
                chunks.append(res.to_ndarray().reshape(-1))
        for res in resampler.resample(None):  # flush
            chunks.append(res.to_ndarray().reshape(-1))

    if not chunks:
        return np.zeros(0, dtype=np.float32), profile.audio_sample_rate, False
    wave = np.concatenate(chunks).astype(np.float32)
    return wave, profile.audio_sample_rate, True


# --------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------


def ingest(
    src: str | Path,
    profile: CanonicalProfile,
    *,
    workdir: str | Path | None = None,
    decode: bool = True,
) -> IngestResult:
    """Run L0 on one input file.

    Order matters: embedded provenance metadata is captured from the **original** file
    before re-encoding strips it, because canonicalisation would otherwise destroy the
    very manifest L2 needs to examine.
    """
    t0 = time.perf_counter()
    src = Path(src)
    if not src.exists():
        raise IngestError(f"input does not exist: {src}")

    source_sha = sha256_file(src)
    source_probe = probe_summary(ffprobe(src))
    embedded = extract_embedded_metadata(src)

    if workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix="mosaic_l0_"))
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    dst = workdir / f"{src.stem}.canonical.{profile.container}"

    cmd, transformations = canonicalise(src, dst, profile)
    canonical_sha = sha256_file(dst)
    canonical_probe = probe_summary(ffprobe(dst))

    media = decode_bundle(dst, profile) if decode else None
    elapsed = time.perf_counter() - t0

    return IngestResult(
        source_path=str(src),
        canonical_path=str(dst),
        source_sha256=source_sha,
        canonical_sha256=canonical_sha,
        profile=profile.__dict__.copy(),
        ffmpeg_binary_version=ffmpeg_version(),
        ffmpeg_command=cmd,
        source_probe=source_probe,
        canonical_probe=canonical_probe,
        embedded_metadata=embedded,
        transformations=transformations,
        ingest_wall_s=round(elapsed, 4),
        media=media,
    )
