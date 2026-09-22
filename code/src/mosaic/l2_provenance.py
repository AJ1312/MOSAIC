"""L2 — Provenance and watermark cross-check.

Three rules govern this layer, and every function here is written to keep them true.

**1. Absence of provenance is never evidence of authenticity.** The overwhelming majority
of real-world media carries no manifest and no watermark. "No manifest found" is recorded
as an observation about the *file*, never as a signal about the *content*. The status enum
below deliberately separates the states that naive implementations collapse together:
a manifest that verified, a manifest that failed to verify, no manifest present, a
container the reader cannot parse, and a reader that is not installed. Only the second is
adverse; the last two are not findings about the media at all.

**2. Unavailable vendor verification is reported as unavailable.** No real SynthID, Kling,
Sora or Midjourney detector is reachable from this environment — SynthID detection is gated
behind Google's own surfaces and returns nothing for non-Google content by design. The
registry below therefore contains one detector that reads a clearly-labelled *simulated*
sidecar, and stubs that return UNAVAILABLE with a reason. A stub never returns "no
watermark detected", because that is a claim about the content that nothing here can support.

**3. Contradictions are first-class findings.** A file can carry a valid manifest asserting
human authorship while its pixels carry a watermark asserting AI generation — the
"Integrity Clash" failure mode (arXiv:2603.02378). Both signals pass independently; only
cross-checking catches it. A clash is logged as a high-priority evidentiary artefact and
forces cascade escalation, independent of which signal one might prefer to believe.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .config import L2Config


class C2PAStatus(str, Enum):
    VERIFIED_VALID = "verified_valid"
    VERIFIED_INVALID = "verified_invalid"
    NO_MANIFEST = "no_manifest_present"
    FORMAT_UNSUPPORTED = "format_unsupported_by_reader"
    READER_UNAVAILABLE = "c2pa_reader_unavailable"
    SIDECAR_SIMULATED = "sidecar_simulated_not_cryptographically_verified"
    ERROR = "error"


class WatermarkStatus(str, Enum):
    DETECTED = "watermark_detected"
    NOT_DETECTED = "watermark_not_detected"
    UNAVAILABLE = "detector_unavailable"
    ERROR = "error"


@dataclass
class C2PAResult:
    status: C2PAStatus
    claims_ai_generated: bool | None = None     # None = unknown / not asserted
    validation_state: str | None = None
    manifest_summary: dict[str, Any] = field(default_factory=dict)
    detail: str = ""
    is_simulated: bool = False

    @property
    def is_adverse(self) -> bool:
        return self.status is C2PAStatus.VERIFIED_INVALID

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "claims_ai_generated": self.claims_ai_generated,
            "validation_state": self.validation_state,
            "manifest_summary": self.manifest_summary,
            "detail": self.detail,
            "is_simulated": self.is_simulated,
        }


@dataclass
class WatermarkResult:
    detector: str
    status: WatermarkStatus
    confidence: float | None = None
    modality: str = "unknown"
    detail: str = ""
    is_simulated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector": self.detector,
            "status": self.status.value,
            "confidence": self.confidence,
            "modality": self.modality,
            "detail": self.detail,
            "is_simulated": self.is_simulated,
        }


@dataclass
class ProvenanceResult:
    c2pa: C2PAResult
    watermarks: list[WatermarkResult]
    integrity_clash: bool
    clash_detail: str | None
    escalate: bool
    escalation_reason: str | None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "c2pa": self.c2pa.to_dict(),
            "watermarks": [w.to_dict() for w in self.watermarks],
            "integrity_clash": self.integrity_clash,
            "clash_detail": self.clash_detail,
            "escalate": self.escalate,
            "escalation_reason": self.escalation_reason,
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------------------
# C2PA
# --------------------------------------------------------------------------------------

_AI_SOURCE_TYPES = (
    "trainedAlgorithmicMedia",
    "compositeWithTrainedAlgorithmicMedia",
    "algorithmicMedia",
)


def read_c2pa(path: str | Path, config: L2Config | None = None) -> C2PAResult:
    """Attempt genuine C2PA verification, then fall back to a labelled sidecar."""
    config = config or L2Config()
    path = Path(path)

    try:
        import c2pa
    except Exception as exc:
        return _sidecar_or(
            path, config,
            C2PAResult(status=C2PAStatus.READER_UNAVAILABLE,
                       detail=f"c2pa library not importable: {exc}. No claim is made about "
                              "the presence or absence of a manifest."),
        )

    try:
        supported = set(c2pa.Reader.get_supported_mime_types())
    except Exception:
        supported = set()
    suffix = path.suffix.lower().lstrip(".")
    if supported and suffix and suffix not in supported and f"video/{suffix}" not in supported:
        return _sidecar_or(
            path, config,
            C2PAResult(
                status=C2PAStatus.FORMAT_UNSUPPORTED,
                detail=(f"the C2PA reader does not support the '{suffix}' container "
                        f"(SDK {getattr(c2pa, 'sdk_version', lambda: '?')()}). This says nothing "
                        "about whether a manifest exists or whether the content is authentic."),
            ),
        )

    try:
        try:
            reader = c2pa.Reader(str(path))
        except TypeError:
            reader = c2pa.Reader.from_file(str(path))
        
        raw = reader.json()
        manifest = json.loads(raw) if raw else {}
        try:
            state = str(reader.get_validation_state())
        except Exception:
            state = None
        valid = True
        try:
            valid = bool(reader.is_valid())
        except Exception:
            pass

            active = manifest.get("active_manifest")
            manifests = manifest.get("manifests", {}) or {}
            am = manifests.get(active, {}) if active else {}
            claims_ai = _manifest_claims_ai(am)
            return C2PAResult(
                status=C2PAStatus.VERIFIED_VALID if valid else C2PAStatus.VERIFIED_INVALID,
                claims_ai_generated=claims_ai,
                validation_state=state,
                manifest_summary={
                    "active_manifest": active,
                    "claim_generator": am.get("claim_generator"),
                    "title": am.get("title"),
                    "n_assertions": len(am.get("assertions", []) or []),
                    "signature_issuer": (am.get("signature_info") or {}).get("issuer"),
                },
                detail="manifest read and validated by the c2pa SDK",
            )
    except Exception as exc:
        name = type(exc).__name__
        if "ManifestNotFound" in name or "no JUMBF" in str(exc):
            return _sidecar_or(
                path, config,
                C2PAResult(
                    status=C2PAStatus.NO_MANIFEST,
                    detail="no embedded C2PA manifest found. This is the expected common case "
                           "and is NOT evidence that the content is authentic.",
                ),
            )
        return _sidecar_or(
            path, config,
            C2PAResult(status=C2PAStatus.ERROR,
                       detail=f"C2PA read failed ({name}): {str(exc)[:300]}"),
        )


def _manifest_claims_ai(manifest: dict[str, Any]) -> bool | None:
    """Whether the manifest asserts AI generation. None when it makes no such assertion."""
    assertions = manifest.get("assertions") or []
    for a in assertions:
        if a.get("label") != "c2pa.actions":
            continue
        for action in (a.get("data") or {}).get("actions", []) or []:
            dst = str(action.get("digitalSourceType", ""))
            if any(t in dst for t in _AI_SOURCE_TYPES):
                return True
            if "digitalCapture" in dst:
                return False
    return None


def _sidecar_or(path: Path, config: L2Config, fallback: C2PAResult) -> C2PAResult:
    """Use a ``<file>.c2pa.json`` sidecar if permitted, clearly marked as simulated."""
    if not config.allow_sidecar_manifest:
        return fallback
    sidecar = Path(str(path) + ".c2pa.json")
    if not sidecar.exists():
        return fallback
    try:
        data = json.loads(sidecar.read_text())
    except Exception as exc:
        fallback.detail += f" (a sidecar exists but could not be parsed: {exc})"
        return fallback
    return C2PAResult(
        status=C2PAStatus.SIDECAR_SIMULATED,
        claims_ai_generated=data.get("claims_ai_generated"),
        validation_state="not_verified",
        manifest_summary={
            "claim_generator": data.get("claim_generator"),
            "n_assertions": len(data.get("assertions", []) or []),
            "source": str(sidecar.name),
        },
        detail=("SIMULATED manifest read from a sidecar file. It is NOT cryptographically "
                "signed and NOT verified; it is used only to exercise cross-check logic. "
                f"Embedded-manifest status was: {fallback.status.value}."),
        is_simulated=True,
    )


# --------------------------------------------------------------------------------------
# Watermark detector registry
# --------------------------------------------------------------------------------------


class WatermarkDetector:
    """Pluggable watermark detector interface.

    Designed as a registry rather than a hardcoded vendor call so that additional
    detectors can be added as cross-vendor APIs appear, without changing L2's logic.
    """

    name = "base"
    is_simulated = False

    def available(self) -> tuple[bool, str]:
        raise NotImplementedError

    def detect(self, path: Path) -> WatermarkResult:
        raise NotImplementedError


class SidecarSimulatedDetector(WatermarkDetector):
    """Reads a ``<file>.watermark.json`` sidecar. Always labelled simulated."""

    name = "sidecar_simulated"
    is_simulated = True

    def available(self) -> tuple[bool, str]:
        return True, "reads a local simulated sidecar; no vendor service is contacted"

    def detect(self, path: Path) -> WatermarkResult:
        sidecar = Path(str(path) + ".watermark.json")
        if not sidecar.exists():
            return WatermarkResult(
                detector=self.name, status=WatermarkStatus.UNAVAILABLE,
                detail="no simulated watermark sidecar accompanies this file, so no watermark "
                       "determination was made. This is NOT a finding that the content is "
                       "unwatermarked or authentic.",
                is_simulated=True,
            )
        try:
            data = json.loads(sidecar.read_text())
        except Exception as exc:
            return WatermarkResult(detector=self.name, status=WatermarkStatus.ERROR,
                                   detail=f"sidecar unreadable: {exc}", is_simulated=True)
        detected = bool(data.get("watermark_detected"))
        return WatermarkResult(
            detector=self.name,
            status=WatermarkStatus.DETECTED if detected else WatermarkStatus.NOT_DETECTED,
            confidence=data.get("confidence"),
            modality=str(data.get("modality", "unknown")),
            detail="SIMULATED watermark-detector response read from a local sidecar. No real "
                   "vendor detector was queried and this must not be read as vendor evidence.",
            is_simulated=True,
        )


class UnavailableVendorDetector(WatermarkDetector):
    """A real vendor detector that cannot be reached from this environment.

    Returns UNAVAILABLE with a reason — never NOT_DETECTED. Reporting "no watermark" for a
    detector that never ran would be a fabricated negative result, and negative watermark
    findings are exactly the kind of evidence that gets over-read as proof of authenticity.
    """

    def __init__(self, name: str, reason: str):
        self.name = name
        self.reason = reason

    def available(self) -> tuple[bool, str]:
        return False, self.reason

    def detect(self, path: Path) -> WatermarkResult:
        return WatermarkResult(detector=self.name, status=WatermarkStatus.UNAVAILABLE,
                               detail=self.reason)


def default_registry() -> dict[str, WatermarkDetector]:
    return {
        "sidecar_simulated": SidecarSimulatedDetector(),
        "synthid": UnavailableVendorDetector(
            "synthid",
            "SynthID verification is gated behind Google's Gemini app and Detector Portal and "
            "has no offline API. It also returns nothing for non-Google content by design. No "
            "determination is made.",
        ),
        "c2pa_soft_binding": UnavailableVendorDetector(
            "c2pa_soft_binding",
            "No soft-binding/fingerprint resolver service is configured, so durable-content-"
            "credential lookup could not be attempted.",
        ),
    }


# --------------------------------------------------------------------------------------
# Cross-check
# --------------------------------------------------------------------------------------


def check_provenance(
    path: str | Path,
    config: L2Config | None = None,
    registry: dict[str, WatermarkDetector] | None = None,
) -> ProvenanceResult:
    """Run C2PA verification and every configured watermark detector, then cross-check."""
    config = config or L2Config()
    registry = registry if registry is not None else default_registry()
    path = Path(path)
    notes: list[str] = []

    c2pa_res = read_c2pa(path, config)

    watermarks: list[WatermarkResult] = []
    for key in config.watermark_detectors:
        det = registry.get(key)
        if det is None:
            watermarks.append(WatermarkResult(
                detector=key, status=WatermarkStatus.UNAVAILABLE,
                detail=f"detector '{key}' is not present in the registry",
            ))
            continue
        ok, reason = det.available()
        if not ok:
            watermarks.append(WatermarkResult(detector=det.name,
                                              status=WatermarkStatus.UNAVAILABLE, detail=reason))
            continue
        try:
            watermarks.append(det.detect(path))
        except Exception as exc:
            watermarks.append(WatermarkResult(detector=det.name, status=WatermarkStatus.ERROR,
                                              detail=f"{type(exc).__name__}: {exc}"))

    # ---- Integrity Clash ---------------------------------------------------------------
    clash = False
    clash_detail: str | None = None
    manifest_claims_ai = c2pa_res.claims_ai_generated
    wm_detected = [w for w in watermarks if w.status is WatermarkStatus.DETECTED]
    wm_absent = [w for w in watermarks if w.status is WatermarkStatus.NOT_DETECTED]

    if manifest_claims_ai is False and wm_detected:
        clash = True
        clash_detail = (
            f"INTEGRITY CLASH: the manifest asserts human capture "
            f"(digitalSourceType=digitalCapture) while watermark detector "
            f"'{wm_detected[0].detector}' reports an AI-generation watermark present. Both "
            "signals verify independently; their disagreement is itself the finding."
        )
    elif manifest_claims_ai is True and wm_absent:
        clash = True
        clash_detail = (
            f"INTEGRITY CLASH: the manifest asserts AI generation while watermark detector "
            f"'{wm_absent[0].detector}' reports no watermark. This is weaker than the converse "
            "— watermarks are removable, and absence is a weak signal — but the disagreement "
            "is still recorded and escalated rather than resolved in either direction."
        )

    # ---- escalation --------------------------------------------------------------------
    escalate = False
    reasons: list[str] = []
    if clash:
        escalate = True
        reasons.append("integrity clash between manifest and watermark")
    if c2pa_res.is_adverse:
        escalate = True
        reasons.append("C2PA manifest present but failed validation")
    if manifest_claims_ai is True:
        escalate = True
        reasons.append("manifest itself asserts AI generation")
    if wm_detected:
        escalate = True
        reasons.append(f"watermark detected by {wm_detected[0].detector}")

    if c2pa_res.status in (C2PAStatus.NO_MANIFEST, C2PAStatus.FORMAT_UNSUPPORTED,
                           C2PAStatus.READER_UNAVAILABLE):
        notes.append(
            "No usable provenance was found. Under MOSAIC this carries NO implication that "
            "the content is authentic: the absence of a manifest is the normal state of "
            "real-world media and the detector cascade decides the verdict on its own."
        )
    if any(w.is_simulated for w in watermarks) or c2pa_res.is_simulated:
        notes.append(
            "One or more provenance signals in this record are SIMULATED (local sidecar). "
            "They exercise the cross-check logic and must not be cited as vendor evidence."
        )

    return ProvenanceResult(
        c2pa=c2pa_res,
        watermarks=watermarks,
        integrity_clash=clash,
        clash_detail=clash_detail,
        escalate=escalate,
        escalation_reason="; ".join(reasons) if reasons else None,
        notes=notes,
    )
