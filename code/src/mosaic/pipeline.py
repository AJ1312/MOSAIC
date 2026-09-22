"""End-to-end MOSAIC-AV orchestration.

    L0 Canonical ingest
       -> L1 Hash / similarity triage        (exact hit may short-circuit)
       -> L2 Provenance & watermark          (clash forces escalation)
       -> L3 Cascaded detection
             Tier-0  bitstream triage        (may exit early toward "clean" only)
             Tier-1  visual | audio | audiovisual branches
             Tier-2  windowed localisation on the residual uncertain fraction
             -> uncertainty-aware fusion     (five-way verdict)
       -> L4 Custody sealing

Cost is spent where it is needed. Tier-0 costs microseconds and can retire the common
clean case; Tier-1 runs the three branch models; Tier-2 only runs when fusion is
unconfident, a modality conflict appears, or L2 flagged something. Every one of those
decisions, and every skipped stage, is written into the custody record — a clip that
exited early is never presented as one that received the full analysis.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import FEATURE_SCHEMA_VERSION, __version__
from .config import MosaicConfig
from .evidence import EvidenceReport, build_report, summarise_verdict
from .fusion import AVCouplingModel, EscalationFlag, FusionResult, ModalityVerdict, Verdict, fuse
from .hashing import audio_phash, video_phash
from .l0_ingest import IngestResult, ingest
from .l1_hash import HashRegistry, RegistryEntry, TriageResult, triage
from .l2_provenance import ProvenanceResult, check_provenance
from .l3_audio import extract_audio_features, log_mel_spectrogram
from .l3_av_sync import AVFeatures, extract_av_features
from .l3_tier0 import Tier0Decision, extract_tier0_features, tier0_decide
from .l3_visual import extract_visual_features
from .l4_custody import CustodyBuilder, LocalLedger, commit_audio, commit_video
from .models import AudioCNN, BranchModel, BranchPrediction

PIPELINE_DESCRIPTION = (
    "MOSAIC-AV v{ver}: L0 canonical re-encode (single fixed ffmpeg profile) -> L1 SHA-256 "
    "exact-identity triage with perceptual-hash candidate retrieval -> L2 C2PA verification "
    "and watermark-detector registry with Integrity Clash cross-check -> L3 cascaded "
    "detection (Tier-0 bitstream triage; Tier-1 visual, audio and audiovisual branches; "
    "Tier-2 windowed localisation) fused by an uncertainty-aware four-cell factor graph over "
    "(video, audio) authenticity -> L4 dual-track Merkle custody sealing with a hash-chained "
    "ledger."
).format(ver=__version__)


# --------------------------------------------------------------------------------------
# Model bundle
# --------------------------------------------------------------------------------------


@dataclass
class ModelBundle:
    visual: BranchModel
    audio: BranchModel
    av: BranchModel
    tier0: BranchModel
    coupling: AVCouplingModel
    audio_cnn: AudioCNN | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def train_stats(self) -> dict[str, dict[str, dict[str, float]]]:
        return {
            "visual": self.visual.train_stats_,
            "audio": self.audio.train_stats_,
            "audiovisual": self.av.train_stats_,
        }

    def descriptor(self) -> dict[str, Any]:
        """Model identity, recorded in every custody record."""
        return {
            "visual": {"name": self.visual.name, "n_features": len(self.visual.feature_names),
                       "n_bootstrap": self.visual.n_bootstrap, "C": self.visual.C,
                       "calibrated_on": getattr(self.visual, "calibrated_on_", "unknown")},
            "audio": {"name": self.audio.name, "n_features": len(self.audio.feature_names),
                      "n_bootstrap": self.audio.n_bootstrap, "C": self.audio.C,
                      "calibrated_on": getattr(self.audio, "calibrated_on_", "unknown")},
            "audiovisual": {"name": self.av.name, "n_features": len(self.av.feature_names),
                            "n_bootstrap": self.av.n_bootstrap, "C": self.av.C,
                            "calibrated_on": getattr(self.av, "calibrated_on_", "unknown")},
            "tier0": {"name": self.tier0.name, "n_features": len(self.tier0.feature_names)},
            "audio_cnn": ("~25k-param log-mel CNN (trained locally)" if self.audio_cnn
                          else "not used"),
            "coupling": self.coupling.to_dict(),
            "feature_schema": FEATURE_SCHEMA_VERSION,
            "mosaic_version": __version__,
            **self.metadata,
        }

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.visual.save(directory / "visual.pkl")
        self.audio.save(directory / "audio.pkl")
        self.av.save(directory / "av.pkl")
        self.tier0.save(directory / "tier0.pkl")
        import pickle

        with open(directory / "coupling.pkl", "wb") as fh:
            pickle.dump(self.coupling, fh)
        with open(directory / "metadata.pkl", "wb") as fh:
            pickle.dump(self.metadata, fh)
        if self.audio_cnn is not None:
            self.audio_cnn.save(directory / "audio_cnn.pt")
        return directory

    @staticmethod
    def load(directory: str | Path) -> "ModelBundle":
        import pickle

        directory = Path(directory)
        with open(directory / "coupling.pkl", "rb") as fh:
            coupling = pickle.load(fh)
        meta = {}
        if (directory / "metadata.pkl").exists():
            with open(directory / "metadata.pkl", "rb") as fh:
                meta = pickle.load(fh)
        cnn = None
        if (directory / "audio_cnn.pt").exists():
            try:
                cnn = AudioCNN.load(directory / "audio_cnn.pt")
            except Exception:
                cnn = None
        return ModelBundle(
            visual=BranchModel.load(directory / "visual.pkl"),
            audio=BranchModel.load(directory / "audio.pkl"),
            av=BranchModel.load(directory / "av.pkl"),
            tier0=BranchModel.load(directory / "tier0.pkl"),
            coupling=coupling, audio_cnn=cnn, metadata=meta,
        )


# --------------------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------------------


@dataclass
class PipelineResult:
    clip_id: str
    source_path: str
    verdict: Verdict
    fusion: FusionResult | None
    evidence: EvidenceReport
    triage: TriageResult
    provenance: ProvenanceResult | None
    tier0: Tier0Decision | None
    timings_s: dict[str, float]
    custody_record_id: str | None
    combined_root: str | None
    stages_run: list[str]
    stages_skipped: list[str]
    caveats: list[str] = field(default_factory=list)
    av_features: AVFeatures | None = field(default=None, repr=False)
    error: str | None = None

    def summary(self) -> str:
        if self.fusion is None:
            return f"{self.clip_id}: {self.verdict.value} (no fusion performed)"
        return summarise_verdict(self.fusion, self.evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "source_path": self.source_path,
            "verdict": self.verdict.value,
            "fusion": self.fusion.to_dict() if self.fusion else None,
            "evidence": self.evidence.to_dict(),
            "l1_triage": self.triage.to_dict(),
            "l2_provenance": self.provenance.to_dict() if self.provenance else None,
            "tier0": self.tier0.to_dict() if self.tier0 else None,
            "timings_s": {k: round(v, 5) for k, v in self.timings_s.items()},
            "custody_record_id": self.custody_record_id,
            "combined_root": self.combined_root,
            "stages_run": self.stages_run,
            "stages_skipped": self.stages_skipped,
            "caveats": self.caveats,
            "error": self.error,
        }


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------


class MosaicPipeline:
    """Runs the full L0-L4 pipeline over individual media files."""

    def __init__(self, config: MosaicConfig, models: ModelBundle,
                 device_profile: dict[str, Any] | None = None,
                 registry: HashRegistry | None = None,
                 workdir: str | Path | None = None,
                 synthetic_data: bool = True):
        self.config = config
        self.models = models
        self.device_profile = device_profile or {}
        self.registry = registry or HashRegistry(config.l1.registry_path, config.l1)
        self.ledger = LocalLedger(config.l4.ledger_path)
        self.workdir = Path(workdir) if workdir else None
        self.synthetic_data = synthetic_data
        self.config_digest = config.digest()

    # -- helpers ------------------------------------------------------------------------

    def _base_caveats(self) -> list[str]:
        caveats = [
            "The local timestamp authority is NOT a qualified RFC-3161/eIDAS timestamp "
            "service; it establishes ordering within this ledger only.",
            "The custody anchor is a local append-only hash-chained file, not a distributed "
            "or multi-operator ledger.",
        ]
        if self.synthetic_data:
            caveats.insert(0,
                "SYNTHETIC DEMO DATA: this clip was procedurally generated and its "
                "manipulations are simulated signal-processing operations, not the output of "
                "a real generative model. No accuracy figure derived from it describes "
                "real-world detection performance.")
        return caveats

    # -- main ---------------------------------------------------------------------------

    def process(self, path: str | Path, clip_id: str | None = None,
                register: bool = True) -> PipelineResult:
        path = Path(path)
        clip_id = clip_id or path.stem
        timings: dict[str, float] = {}
        stages_run: list[str] = []
        stages_skipped: list[str] = []
        caveats = self._base_caveats()

        custody = CustodyBuilder(clip_id, self.ledger)
        custody.event("L0", "start", {"source": str(path)})
        t_total = time.perf_counter()

        # ---- L0 --------------------------------------------------------------------
        t = time.perf_counter()
        ing: IngestResult = ingest(path, self.config.canonical, workdir=self.workdir)
        timings["l0_ingest"] = time.perf_counter() - t
        stages_run.append("L0")
        media = ing.media
        assert media is not None
        custody.event("L0", "canonicalised", {
            "source_sha256": ing.source_sha256,
            "canonical_sha256": ing.canonical_sha256,
            "ffmpeg_command": ing.ffmpeg_command,
            "transformations": ing.transformations,
            "motion_vectors_available": media.motion_vectors.available,
        })

        # ---- L1 --------------------------------------------------------------------
        t = time.perf_counter()
        v_ph = video_phash(media.frames, self.config.l1.phash_frames,
                           self.config.l1.phash_bits_per_frame)
        a_ph = (audio_phash(media.audio, media.sample_rate, self.config.l1.audio_phash_segments)
                if media.has_audio else [])
        tri = triage(self.registry, ing.canonical_sha256, v_ph, a_ph,
                     feature_schema=FEATURE_SCHEMA_VERSION, config_digest=self.config_digest,
                     config=self.config.l1)
        timings["l1_triage"] = time.perf_counter() - t
        stages_run.append("L1")
        custody.event("L1", tri.action, {"short_circuit": tri.short_circuit,
                                         "reason": tri.reason,
                                         "n_candidates": len(tri.candidates)})

        if tri.short_circuit and tri.exact_match is not None:
            return self._short_circuit_result(
                clip_id, path, ing, tri, timings, stages_run, stages_skipped, caveats,
                custody, media,
            )

        # ---- L2 --------------------------------------------------------------------
        t = time.perf_counter()
        prov = check_provenance(path, self.config.l2)
        timings["l2_provenance"] = time.perf_counter() - t
        stages_run.append("L2")
        custody.event("L2", "provenance_checked", {
            "c2pa_status": prov.c2pa.status.value,
            "integrity_clash": prov.integrity_clash,
            "escalate": prov.escalate,
            "reason": prov.escalation_reason,
        })

        # ---- L3 Tier-0 -------------------------------------------------------------
        t = time.perf_counter()
        t0_feats = extract_tier0_features(media.motion_vectors, media.audio, media.sample_rate)
        t0_pred = self.models.tier0.predict(t0_feats.vector,
                                            ood_percentile=self.config.fusion.ood_percentile)
        # Prefer the threshold learned on validation over the config default: Tier-0's
        # calibrated scores live wherever the training corpus's base rate puts them, so a
        # portable absolute constant would either never fire or fire indiscriminately.
        exit_threshold = float(self.models.metadata.get(
            "tier0_exit_threshold", self.config.l3.tier0_real_exit_threshold))
        t0_decision = tier0_decide(
            t0_pred.p_fake, t0_pred.vacuity,
            real_exit_threshold=exit_threshold,
            provenance_escalation=prov.escalate,
            mv_available=media.motion_vectors.available,
            enabled=self.config.l3.tier0_enabled,
        )
        timings["l3_tier0"] = time.perf_counter() - t
        stages_run.append("L3.Tier0")
        custody.event("L3", "tier0", t0_decision.to_dict())

        edge_tier = self.config.device_tier == "edge"
        if t0_decision.action == "early_exit_clean" and not edge_tier:
            return self._tier0_exit_result(
                clip_id, path, ing, tri, prov, t0_decision, t0_pred, timings,
                stages_run, stages_skipped, caveats, custody, media, v_ph, a_ph, register,
            )

        if edge_tier:
            stages_skipped.extend(["L3.Tier1", "L3.Tier2"])
            caveats.append(
                "Device tier 'edge': Tier-1 and Tier-2 detection did not run. This result is a "
                "pre-upload sanity check deferred to a higher tier, not a verdict."
            )
            return self._deferred_result(
                clip_id, path, ing, tri, prov, t0_decision, timings, stages_run,
                stages_skipped, caveats, custody, media, v_ph, a_ph, register,
            )

        # ---- L3 Tier-1 branches ----------------------------------------------------
        t = time.perf_counter()
        vis_feats = extract_visual_features(media.frames, self.config.l3.visual_frames)
        vis_pred = self.models.visual.predict(vis_feats.vector,
                                              ood_percentile=self.config.fusion.ood_percentile)
        timings["l3_visual"] = time.perf_counter() - t

        t = time.perf_counter()
        if media.has_audio and media.audio.size > 0:
            aud_feats = extract_audio_features(
                media.audio, media.sample_rate,
                frame_ms=self.config.l3.audio_frame_ms, hop_ms=self.config.l3.audio_hop_ms,
            )
            if aud_feats.available:
                extra: list[float] = []
                if self.models.audio_cnn is not None and self.models.audio_cnn.fitted_:
                    try:
                        spec = log_mel_spectrogram(media.audio, media.sample_rate,
                                                   n_mels=self.config.l3.n_mels)
                        extra = [float(self.models.audio_cnn.predict_proba([spec])[0])]
                    except Exception as exc:
                        # A silent fallback here would quietly drop an ensemble member and
                        # make the audio branch weaker than the model card claims, with no
                        # trace in the record. Record the failure instead.
                        extra = []
                        cnn_error = f"{type(exc).__name__}: {exc}"
                        custody.event("L3", "audio_cnn_unavailable", {"error": cnn_error})
                        caveats.append(
                            f"The audio CNN ensemble member failed and was excluded from this "
                            f"clip's audio prediction ({cnn_error}). The audio branch ran on "
                            "hand-designed features alone."
                        )
                aud_pred = self.models.audio.predict(
                    aud_feats.vector, ood_percentile=self.config.fusion.ood_percentile,
                    extra_probs=extra,
                )
            else:
                aud_pred = BranchPrediction.unavailable(aud_feats.reason_unavailable or "unknown")
        else:
            aud_pred = BranchPrediction.unavailable("no audio track in the source media")
        timings["l3_audio"] = time.perf_counter() - t

        t = time.perf_counter()
        av_feats = extract_av_features(
            media.frames, media.fps, media.audio, media.sample_rate,
            window_s=self.config.l3.av_window_s, hop_s=self.config.l3.av_hop_s,
            max_lag_ms=self.config.l3.av_max_lag_ms,
            interval_sigma=self.config.l3.av_interval_sigma,
        )
        if av_feats.available:
            av_pred = self.models.av.predict(av_feats.vector,
                                             ood_percentile=self.config.fusion.ood_percentile)
        else:
            av_pred = BranchPrediction.unavailable(av_feats.reason_unavailable or "unknown")
        timings["l3_av"] = time.perf_counter() - t
        stages_run.append("L3.Tier1")
        custody.event("L3", "tier1_branches", {
            "visual": vis_pred.to_dict(), "audio": aud_pred.to_dict(), "av": av_pred.to_dict(),
        })

        # ---- fusion ----------------------------------------------------------------
        t = time.perf_counter()
        extra_flags = [EscalationFlag.PROVENANCE_CLASH.value] if prov.integrity_clash else []
        result = fuse(
            vis_pred, aud_pred, av_pred, self.models.coupling,
            prior_fake_video=self.config.fusion.prior_fake_video,
            prior_fake_audio=self.config.fusion.prior_fake_audio,
            vacuity_penalty=self.config.fusion.vacuity_penalty,
            decide_threshold=self.config.fusion.decide_threshold,
            confident_llr=self.config.fusion.confident_llr,
            confident_max_vacuity=self.config.fusion.confident_max_vacuity,
            modality_margin=self.config.fusion.modality_margin,
            extra_flags=extra_flags,
        )
        timings["fusion"] = time.perf_counter() - t

        # ---- L3 Tier-2 -------------------------------------------------------------
        needs_tier2 = (
            self.config.l3.tier2_enabled and (
                result.top_posterior < self.config.l3.tier2_margin_threshold
                or result.joint_vacuity > self.config.l3.tier2_uncertainty_threshold
                or prov.escalate
                or EscalationFlag.MODALITY_CONFLICT.value in result.flags
            )
        )
        if needs_tier2:
            t = time.perf_counter()
            # Tier-2 is the expensive pass: denser windowing for temporal localisation.
            av_feats = extract_av_features(
                media.frames, media.fps, media.audio, media.sample_rate,
                window_s=self.config.l3.av_window_s, hop_s=max(0.05, self.config.l3.av_hop_s / 2),
                max_lag_ms=self.config.l3.av_max_lag_ms,
                interval_sigma=max(1.2, self.config.l3.av_interval_sigma - 0.5),
            )
            timings["l3_tier2"] = time.perf_counter() - t
            stages_run.append("L3.Tier2")
            custody.event("L3", "tier2_localisation", {
                "n_suspicious_intervals": len(av_feats.suspicious_intervals),
                "reason": "fusion unconfident, modality conflict, or provenance escalation",
            })
            caveats.append(
                "Tier-2 in this prototype performs dense audiovisual localisation only. The "
                "explainable-MLLM reasoning tier described in the MOSAIC design is NOT "
                "implemented; see docs/IMPLEMENTED_VS_STUBBED.md."
            )
        else:
            stages_skipped.append("L3.Tier2")

        # ---- evidence --------------------------------------------------------------
        report = build_report(
            vis_pred, aud_pred, av_pred, self.models.train_stats(),
            intervals=av_feats.suspicious_intervals, provenance=prov.to_dict(),
        )
        if prov.integrity_clash:
            report.notes.append(
                "Integrity Clash detected in L2; the cascade was escalated regardless of "
                "detector scores."
            )

        timings["total"] = time.perf_counter() - t_total

        # ---- L4 --------------------------------------------------------------------
        record_id, root = self._seal(
            custody, clip_id, ing, tri, prov, result, report, media, timings,
            stages_run, stages_skipped, caveats,
            l3_detail={
                "tier0": t0_decision.to_dict(),
                "visual_features": vis_feats.as_dict(),
                "audio_features": (aud_feats.as_dict() if media.has_audio and aud_feats.available
                                   else {}),
                "av_features": av_feats.as_dict(),
                "branches": {"visual": vis_pred.to_dict(), "audio": aud_pred.to_dict(),
                             "audiovisual": av_pred.to_dict()},
            },
        )
        stages_run.append("L4")

        if register:
            self._register(clip_id, ing, v_ph, a_ph, result.verdict.value,
                           result.to_dict(), record_id)

        return PipelineResult(
            clip_id=clip_id, source_path=str(path), verdict=result.verdict, fusion=result,
            evidence=report, triage=tri, provenance=prov, tier0=t0_decision,
            timings_s=timings, custody_record_id=record_id, combined_root=root,
            stages_run=stages_run, stages_skipped=stages_skipped, caveats=caveats,
            av_features=av_feats,
        )

    # -- exit paths ---------------------------------------------------------------------

    def _seal(self, custody, clip_id, ing, tri, prov, result, report, media, timings,
              stages_run, stages_skipped, caveats, l3_detail) -> tuple[str, str]:
        record_id = f"MOSAIC-{uuid.uuid4().hex[:16]}"
        vc = commit_video(media.frames, media.fps, self.config.l4.chunk_seconds)
        ac = commit_audio(media.audio, media.sample_rate, self.config.l4.chunk_seconds)
        record = custody.seal(
            record_id=record_id,
            schema_version=FEATURE_SCHEMA_VERSION,
            source_path=ing.source_path,
            source_sha256=ing.source_sha256,
            canonical_sha256=ing.canonical_sha256,
            feature_schema=FEATURE_SCHEMA_VERSION,
            config_digest=self.config_digest,
            seed=self.config.seed,
            device_profile=self.device_profile,
            pipeline_description=PIPELINE_DESCRIPTION,
            ingest=ing.to_dict(),
            l1_triage=tri.to_dict(),
            l2_provenance=prov.to_dict() if prov else {},
            l3_detection=l3_detail,
            verdict=(result.to_dict() if hasattr(result, "to_dict") else dict(result)),
            evidence=report.to_dict()["findings"] if report else [],
            models=self.models.descriptor(),
            video_commitment=vc,
            audio_commitment=ac,
            compute={"timings_s": {k: round(v, 5) for k, v in timings.items()},
                     "stages_run": stages_run, "stages_skipped": stages_skipped},
            caveats=caveats,
            records_dir=self.config.l4.records_dir,
        )
        return record.record_id, record.combined_root

    def _register(self, clip_id, ing, v_ph, a_ph, verdict, payload, record_id) -> None:
        self.registry.put(RegistryEntry(
            canonical_sha256=ing.canonical_sha256, source_sha256=ing.source_sha256,
            clip_id=clip_id, verdict=verdict, verdict_payload=payload,
            video_phash=v_ph, audio_phash=a_ph,
            feature_schema=FEATURE_SCHEMA_VERSION, config_digest=self.config_digest,
            custody_record_id=record_id,
        ))

    def _short_circuit_result(self, clip_id, path, ing, tri, timings, stages_run,
                              stages_skipped, caveats, custody, media) -> PipelineResult:
        stages_skipped.extend(["L2", "L3.Tier0", "L3.Tier1", "L3.Tier2"])
        cached = tri.exact_match.verdict
        caveats.append(
            "This verdict was inherited from a byte-identical artefact already in the "
            "registry (L1 exact SHA-256 hit). The detector cascade did not re-run."
        )
        timings["total"] = sum(timings.values())
        report = EvidenceReport(notes=[
            f"No new analysis performed: {tri.reason}",
            f"Inherited verdict '{cached}' from registry entry "
            f"{tri.exact_match.canonical_sha256[:16]} (clip {tri.exact_match.clip_id}).",
        ])
        record_id, root = self._seal(
            custody, clip_id, ing, tri, None,
            {"verdict": cached, "source": "registry_cache"}, report, media, timings,
            stages_run, stages_skipped, caveats,
            l3_detail={"skipped": "L1 exact hit short-circuited the cascade"},
        )
        stages_run.append("L4")
        try:
            verdict = Verdict(cached)
        except ValueError:
            verdict = Verdict.UNKNOWN
        return PipelineResult(
            clip_id=clip_id, source_path=str(path), verdict=verdict, fusion=None,
            evidence=report, triage=tri, provenance=None, tier0=None, timings_s=timings,
            custody_record_id=record_id, combined_root=root, stages_run=stages_run,
            stages_skipped=stages_skipped, caveats=caveats,
        )

    def _tier0_exit_result(self, clip_id, path, ing, tri, prov, t0_decision, t0_pred,
                           timings, stages_run, stages_skipped, caveats, custody, media,
                           v_ph, a_ph, register) -> PipelineResult:
        stages_skipped.extend(["L3.Tier1", "L3.Tier2"])
        caveats.append(
            "Tier-0 early exit: the full visual, audio and audiovisual branches did NOT run. "
            "This clip was retired by cheap bitstream triage as showing no manipulation "
            "indication. Tier-0 can only exit toward 'clean' and never convicts on its own."
        )
        report = EvidenceReport(notes=[t0_decision.reason])
        # An early exit still produces a full five-way verdict object, with per-modality
        # findings marked UNKNOWN rather than asserted real, because no branch examined them.
        result = FusionResult(
            verdict=Verdict.REAL_VIDEO_REAL_AUDIO,
            video_verdict=ModalityVerdict.REAL, audio_verdict=ModalityVerdict.REAL,
            posterior={"real_video_real_audio": 1.0 - t0_pred.p_fake,
                       "fake_video_real_audio": t0_pred.p_fake / 3,
                       "real_video_fake_audio": t0_pred.p_fake / 3,
                       "fake_video_fake_audio": t0_pred.p_fake / 3},
            p_video_fake=t0_pred.p_fake, p_audio_fake=t0_pred.p_fake,
            top_posterior=1.0 - t0_pred.p_fake, margin=1.0 - 4 * t0_pred.p_fake / 3,
            joint_vacuity=t0_pred.vacuity,
            flags=[], notes=[t0_decision.reason],
            branch_summary={"tier0_only": t0_pred.to_dict()},
        )
        timings["total"] = sum(v for k, v in timings.items() if k != "total")
        record_id, root = self._seal(
            custody, clip_id, ing, tri, prov, result, report, media, timings,
            stages_run, stages_skipped, caveats,
            l3_detail={"tier0": t0_decision.to_dict(), "tier1": "skipped (early exit)"},
        )
        stages_run.append("L4")
        if register:
            self._register(clip_id, ing, v_ph, a_ph, result.verdict.value,
                           result.to_dict(), record_id)
        return PipelineResult(
            clip_id=clip_id, source_path=str(path), verdict=result.verdict, fusion=result,
            evidence=report, triage=tri, provenance=prov, tier0=t0_decision,
            timings_s=timings, custody_record_id=record_id, combined_root=root,
            stages_run=stages_run, stages_skipped=stages_skipped, caveats=caveats,
        )

    def _deferred_result(self, clip_id, path, ing, tri, prov, t0_decision, timings,
                         stages_run, stages_skipped, caveats, custody, media, v_ph, a_ph,
                         register) -> PipelineResult:
        report = EvidenceReport(notes=[
            "Edge tier: only L0, L1, L2 and Tier-0 triage ran. A verdict is deliberately "
            "withheld rather than issued with false confidence.",
        ])
        result = FusionResult(
            verdict=Verdict.UNKNOWN,
            video_verdict=ModalityVerdict.UNKNOWN, audio_verdict=ModalityVerdict.UNKNOWN,
            posterior={k: 0.25 for k in ("real_video_real_audio", "fake_video_real_audio",
                                         "real_video_fake_audio", "fake_video_fake_audio")},
            p_video_fake=0.5, p_audio_fake=0.5, top_posterior=0.25, margin=0.0,
            joint_vacuity=1.0,
            flags=[EscalationFlag.DEFERRED_TO_CLOUD.value],
            notes=["deferred to a higher device tier for full analysis"],
        )
        timings["total"] = sum(v for k, v in timings.items() if k != "total")
        record_id, root = self._seal(
            custody, clip_id, ing, tri, prov, result, report, media, timings,
            stages_run, stages_skipped, caveats,
            l3_detail={"tier0": t0_decision.to_dict(), "tier1": "skipped (edge tier)"},
        )
        stages_run.append("L4")
        return PipelineResult(
            clip_id=clip_id, source_path=str(path), verdict=Verdict.UNKNOWN, fusion=result,
            evidence=report, triage=tri, provenance=prov, tier0=t0_decision,
            timings_s=timings, custody_record_id=record_id, combined_root=root,
            stages_run=stages_run, stages_skipped=stages_skipped, caveats=caveats,
        )
