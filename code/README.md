# MOSAIC Implementation Codebase

This directory contains the reference Python implementation for the MOSAIC architecture.

## Structure

- `src/mosaic/`: Core algorithmic modules:
  - `l0_ingest.py`: Stream canonicalization via ffmpeg.
  - `l1_hash.py`: Content and metadata cryptographic and perceptual hashing.
  - `l2_provenance.py`: C2PA manifest parsing and cryptographic signature verification.
  - `l3_visual.py`: Spatial-temporal feature extraction (temporal gradient variance, flow acceleration).
  - `l3_audio.py`: Acoustic spectral feature extraction (CQCC, spectral flux, pitch jitter).
  - `l3_av_sync.py`: Cross-modal synchrony checks.
  - `l3_tier0.py`: Fast screening cascades.
  - `fusion.py`: Asymmetric fusion safety rules.
  - `models.py`: Classifier wrappers and vacuity-aware shrinkage calibration.
  - `l4_custody.py` & `hashing.py`: Chunk-level Merkle tree chain-of-custody logging.
  - `evidence.py`: VidAudit audit protocol specifications.
  - `pipeline.py`: Unified end-to-end inference pipeline.
- `tests/`: Comprehensive unit, integration, and security tamper tests.
- `scripts/`: Operational scripts for pipeline execution, evaluation, and custody audits.

## Usage

```bash
# Install package in editable mode
pip install -e .

# Run all test suites
pytest tests/

# Execute chain-of-custody verification attack simulations
python scripts/audit_block_custody_attack.py
```
