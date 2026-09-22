"""Synthetic audio-video corpus generation for the MOSAIC-AV prototype.

**SYNTHETIC DEMO DATA — not representative of real detection performance.**

Every clip produced by this package is procedurally generated. No real recorded speech
and no real generative-model output is involved. Real deepfake corpora (GenVidBench,
FaceForensics++, Celeb-DF-v2, DFDC, ASVspoof, AV-Deepfake1M) are all credential-gated;
``scripts/01_acquire_data.py`` attempts them, logs the outcome, and falls back here.

What this corpus *is* good for:
  * exercising the full L0-L4 pipeline end to end on media with known ground truth;
  * validating that the audit controls (C1-C6) actually fire;
  * measuring compute, latency and cascade behaviour on real media files;
  * testing that fusion resolves the five-way verdict space correctly, including the
    adversarial cases where one modality's evidence must not override another's.

What it is emphatically *not* good for:
  * any claim about accuracy against real generators. The manipulations here are
    hand-specified signal-processing operations, so a detector trained on them is
    detecting *those operations*, not "deepfakes". Numbers from this corpus are
    reported throughout as capability checks, never as detection performance.
"""

SYNTHETIC_WARNING = (
    "**SYNTHETIC DEMO DATA — not representative of real detection performance.** "
    "Clips are procedurally generated; manipulations are specified signal-processing "
    "operations, not outputs of real generative models."
)

__all__ = ["SYNTHETIC_WARNING"]
