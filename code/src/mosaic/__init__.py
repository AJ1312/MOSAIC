"""MOSAIC-AV — Multi-layer Origin-Synthesis Authentication with Immutable Chain-of-custody,
extended to multimodal audio-video deepfake authentication.

Pipeline
--------
    L0  Canonical ingest              (single-profile re-encode; video + audio; metadata; motion vectors)
    L1  Hash / similarity triage      (SHA-256 exact identity; perceptual hash = candidate retrieval only)
    L2  Provenance & watermark        (C2PA verification; watermark detector registry; Integrity Clash)
    L3  Cascaded detection            (Tier-0 triage -> visual | audio | audiovisual branches -> fusion)
    L4  Immutable chain of custody    (dual-track Merkle over video+audio chunks; hash-chained ledger)

Design principles inherited from the MOSAIC design document (see
``MOSAIC_deepfake_video_research.md``) and enforced throughout this package:

1. SHA-256 is for *exact identity only* — never similarity, never proof of authenticity.
2. Perceptual hashing is for *candidate retrieval*, not proof. It never decides a verdict
   on its own unless explicitly enabled, and that setting is benchmarked as a risk.
3. Absence of provenance NEVER means "real". It is the expected common case.
4. Watermark / C2PA contradictions ("Integrity Clash") force cascade escalation.
5. Compute-efficient cascade with early exits.
6. Everything — input, model, config, decision, timestamp, hash, transformation — is
   recorded in the chain-of-custody record.
7. Never fabricate unavailable results. Unsupported vendor watermark / C2PA verification
   is reported as UNAVAILABLE, never as a negative or positive finding.
"""

__version__ = "0.1.0"

# Bump when feature extraction changes in a way that invalidates cached registry verdicts.
FEATURE_SCHEMA_VERSION = "av-1.0.0"

__all__ = ["__version__", "FEATURE_SCHEMA_VERSION"]
