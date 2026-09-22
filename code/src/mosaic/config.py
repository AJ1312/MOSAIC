"""Configuration objects for the MOSAIC-AV pipeline.

Every configuration that can influence a verdict is hashable to a stable digest
(:func:`config_digest`) which is written into the chain-of-custody record, so a
verdict can always be tied back to the exact settings that produced it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Literal

# --------------------------------------------------------------------------------------
# Canonical media profile (L0)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalProfile:
    """The single mandatory re-encode profile applied to *every* input before anything runs.

    This directly implements control C1 of the VidAudit six-control protocol
    (arXiv:2606.31004) as an architectural default rather than an eval-time afterthought:
    if every clip — real or fake, from any source — passes through an identical codec
    configuration, container-level and codec-level confounds cannot leak class information
    into the detector.
    """

    width: int = 256
    height: int = 256
    fps: int = 25
    video_codec: str = "libx264"
    # Fixed CRF (not bitrate) so quality, not file size, is held constant.
    crf: int = 23
    preset: str = "medium"
    pix_fmt: str = "yuv420p"
    gop_size: int = 25  # 1-second GOP: makes per-second chunking codec-aligned for L4.

    audio_codec: str = "pcm_s16le"  # lossless in the canonical intermediate
    audio_sample_rate: int = 16000
    audio_channels: int = 1

    container: str = "mkv"  # Matroska tolerates pcm_s16le; mp4 does not.

    #: Fixed analysis duration — the "K-frame filter". Every clip is trimmed (and padded
    #: if unavoidably short) to exactly this many seconds.
    #:
    #: This is not cosmetic. Normalising resolution, frame rate and codec does NOT remove a
    #: clip-length confound, and clip length is precisely the channel that let a 3-feature
    #: classifier reach 0.998 LOGO-AUC on an unaudited benchmark (arXiv:2606.31004).
    #: Measured on this project's deliberately-leaky corpus, a duration-only classifier
    #: scored AUC 1.000 both before and after re-encoding until this was added. Holding
    #: duration fixed is what actually closes the channel, and the C2 audit re-runs
    #: afterwards to prove it.
    analysis_seconds: float = 3.0

    def ffmpeg_video_args(self) -> list[str]:
        # tpad clones the final frame for clips shorter than the analysis window; -t then
        # truncates everything to exactly analysis_seconds.
        vf = (f"scale={self.width}:{self.height}:flags=bicubic,fps={self.fps},"
              f"tpad=stop_mode=clone:stop_duration={self.analysis_seconds}")
        return [
            "-c:v", self.video_codec,
            "-crf", str(self.crf),
            "-preset", self.preset,
            "-pix_fmt", self.pix_fmt,
            "-g", str(self.gop_size),
            "-keyint_min", str(self.gop_size),
            "-sc_threshold", "0",  # deterministic GOP boundaries -> stable L4 chunking
            "-vf", vf,
        ]

    def ffmpeg_audio_args(self) -> list[str]:
        return [
            "-c:a", self.audio_codec,
            "-ar", str(self.audio_sample_rate),
            "-ac", str(self.audio_channels),
            "-af", f"apad=whole_dur={self.analysis_seconds}",
        ]

    def ffmpeg_duration_args(self) -> list[str]:
        return ["-t", f"{self.analysis_seconds:.3f}"]


# --------------------------------------------------------------------------------------
# Layer configs
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class L1Config:
    """Hash triage.

    ``allow_perceptual_shortcircuit`` defaults to **False** on purpose. MOSAIC's stated
    principle is that perceptual hashing is a *candidate retrieval* mechanism, not proof.
    Allowing a near-duplicate perceptual hit to inherit a cached verdict is a genuine
    compute win but carries false-short-circuit risk; we benchmark both settings and
    report the risk rather than enabling it silently.
    """

    registry_path: str = "outputs/registry/mosaic_registry.sqlite"
    phash_frames: int = 32          # frames sampled for the video perceptual hash
    phash_bits_per_frame: int = 64
    audio_phash_segments: int = 16
    # Hamming distance (as a fraction of total bits) below which two clips are "candidates".
    perceptual_candidate_threshold: float = 0.10
    # Much stricter bar required before a perceptual hit may short-circuit compute.
    perceptual_shortcircuit_threshold: float = 0.02
    allow_perceptual_shortcircuit: bool = False


@dataclass(frozen=True)
class L2Config:
    """Provenance and watermark cross-check."""

    # Ordered registry of watermark detectors to consult. Only detectors that are
    # genuinely available in this environment may return a decision; everything else
    # must return UNAVAILABLE. See l2_provenance.WatermarkDetectorRegistry.
    watermark_detectors: tuple[str, ...] = ("sidecar_simulated",)
    # Accept a C2PA sidecar (<file>.c2pa.json) as a *simulated, non-cryptographic*
    # manifest source when the media itself carries no embedded manifest.
    allow_sidecar_manifest: bool = True
    trust_anchors_dir: str | None = None  # dir of PEM roots for real C2PA verification


@dataclass(frozen=True)
class L3Config:
    """Cascaded detection."""

    # ---- Tier-0 (near-zero cost) early exit -------------------------------------------
    # A Tier-0 exit is only permitted when the clip looks clean AND L2 raised no flag.
    tier0_enabled: bool = True
    tier0_real_exit_threshold: float = 0.06   # P(any manipulation) below this -> exit
    tier0_fake_escalate_threshold: float = 0.35

    # ---- Tier-1 (branch models) --------------------------------------------------------
    visual_frames: int = 48        # frames sampled for visual features
    audio_frame_ms: float = 25.0
    audio_hop_ms: float = 10.0
    n_mels: int = 64
    n_lfcc: int = 20

    # ---- Tier-2 (expensive: windowed localisation) -------------------------------------
    tier2_enabled: bool = True
    tier2_uncertainty_threshold: float = 0.25  # branch vacuity above this -> escalate
    tier2_margin_threshold: float = 0.55       # top joint posterior below this -> escalate
    av_window_s: float = 1.0
    av_hop_s: float = 0.25
    av_max_lag_ms: float = 500.0
    # A window is flagged as a suspicious interval when its local sync score falls this
    # many robust-sigmas below the clip's own median.
    av_interval_sigma: float = 2.0

    # An MLLM reasoning tier is deliberately NOT implemented; see docs/IMPLEMENTED_VS_STUBBED.md.
    mllm_enabled: bool = False


@dataclass(frozen=True)
class FusionConfig:
    """Uncertainty-aware fusion of the visual / audio / audiovisual branches."""

    prior_fake_video: float = 0.5
    prior_fake_audio: float = 0.5
    # Vacuity (subjective-logic uncertainty mass) is used to temper each branch's evidence:
    #   weight = 1 / (1 + vacuity_penalty * vacuity)
    vacuity_penalty: float = 3.0
    # Minimum top-cell posterior required to emit a decided (non-UNKNOWN) verdict.
    decide_threshold: float = 0.55
    # A branch is "confident" when |LLR| exceeds this and its vacuity is below
    # ``confident_max_vacuity``. Confident branches cannot be overridden by other
    # modalities; a contradiction escalates to MODALITY_CONFLICT instead.
    confident_llr: float = 1.6
    confident_max_vacuity: float = 0.30
    # Half-width of the per-modality indecision band. A marginal inside 0.5 +/- this is
    # reported UNKNOWN for that modality rather than rounded to the nearer side.
    modality_margin: float = 0.15
    # Mahalanobis percentile above which a feature vector is treated as out-of-distribution.
    ood_percentile: float = 99.0


@dataclass(frozen=True)
class L4Config:
    """Chain of custody."""

    chunk_seconds: float = 1.0
    ledger_path: str = "outputs/custody/ledger.jsonl"
    records_dir: str = "outputs/custody/records"
    # There is no qualified RFC-3161 timestamp authority available offline. The local
    # authority is clearly labelled as NON-QUALIFIED everywhere it appears.
    timestamp_authority: Literal["local_nonqualified", "rfc3161"] = "local_nonqualified"
    rfc3161_url: str | None = None


@dataclass(frozen=True)
class MosaicConfig:
    seed: int = 1337
    canonical: CanonicalProfile = field(default_factory=CanonicalProfile)
    l1: L1Config = field(default_factory=L1Config)
    l2: L2Config = field(default_factory=L2Config)
    l3: L3Config = field(default_factory=L3Config)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    l4: L4Config = field(default_factory=L4Config)
    # Filled in at runtime by device.profile_device(); recorded in every custody record.
    device_tier: str = "consumer"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        return config_digest(self)


# --------------------------------------------------------------------------------------
# Device-tier overrides
# --------------------------------------------------------------------------------------


def apply_tier(cfg: MosaicConfig, tier: str) -> MosaicConfig:
    """Return a copy of ``cfg`` adapted to a device tier.

    The three tiers follow the MOSAIC device table:
      * ``cloud``    — full cascade, densest sampling.
      * ``consumer`` — full cascade, moderate sampling (this is what an M-series laptop gets).
      * ``edge``     — L0/L1 plus Tier-0 triage only; anything uncertain is explicitly
                       marked "deferred to cloud" rather than given a false-confidence verdict.
    """
    from dataclasses import replace

    if tier == "cloud":
        l3 = replace(cfg.l3, visual_frames=96, tier2_enabled=True, av_hop_s=0.125)
    elif tier == "consumer":
        l3 = replace(cfg.l3, visual_frames=48, tier2_enabled=True, av_hop_s=0.25)
    elif tier == "edge":
        # Edge devices run triage only. Tier-1/Tier-2 are disabled; the pipeline emits
        # DEFER_TO_CLOUD instead of a verdict it cannot justify.
        l3 = replace(cfg.l3, visual_frames=16, tier2_enabled=False, av_hop_s=0.5)
    else:
        raise ValueError(f"unknown device tier: {tier!r}")
    return replace(cfg, l3=l3, device_tier=tier)


# --------------------------------------------------------------------------------------
# Serialisation helpers
# --------------------------------------------------------------------------------------


def _canonical_json(obj: Any) -> str:
    if is_dataclass(obj) and not isinstance(obj, type):
        obj = asdict(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_digest(obj: Any) -> str:
    """Stable SHA-256 digest of any config object. Recorded in the custody record."""
    return hashlib.sha256(_canonical_json(obj).encode("utf-8")).hexdigest()


def load_config(path: str | Path | None = None) -> MosaicConfig:
    """Load a config, optionally overlaying a YAML file over the defaults."""
    cfg = MosaicConfig()
    if path is None:
        return cfg
    import yaml
    from dataclasses import replace

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    sub = {
        "canonical": CanonicalProfile,
        "l1": L1Config,
        "l2": L2Config,
        "l3": L3Config,
        "fusion": FusionConfig,
        "l4": L4Config,
    }
    updates: dict[str, Any] = {}
    for key, value in raw.items():
        if key in sub and isinstance(value, dict):
            current = getattr(cfg, key)
            updates[key] = replace(current, **value)
        elif hasattr(cfg, key):
            updates[key] = value
        else:
            raise KeyError(f"unknown config key: {key!r}")
    return replace(cfg, **updates)
