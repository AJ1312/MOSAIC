# MOSAIC: An Auditable Architecture for Multimodal Deepfake Forensics

[![IEEE Access](https://img.shields.io/badge/IEEE_Access-Submitted-blue.svg)](https://ieeeaccess.ieee.org/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/pytest-passing-brightgreen.svg)](code/tests)

This repository contains the complete research codebase and reproducible evaluation pipelines for **MOSAIC** (*Multimodal Open-Science Auditable Integrity Checker*).

---

## Overview

Modern deep generative models can synthesize hyper-realistic audiovisual media. However, commercial and academic deepfake detectors often fail in real-world deployment due to opaque black-box models, container shortcuts, and uncalibrated overconfidence under distribution shift. 

**MOSAIC** addresses these challenges through a six-stage auditable forensic architecture:
1. **L0 Canonicalization:** Strips container and encoding metadata shortcuts by re-encoding streams to uniform audio-video representations.
2. **L1 Identity & Near-Duplicate Hashing:** Binds SHA-256 byte hashes, PDQ perceptual image hashes, and Chromaprint acoustic fingerprints.
3. **L2 Provenance Verification:** Inspects C2PA JUMBF manifests, cryptographically verifying manifest signatures and asset hash bindings.
4. **L3 Multi-Scale Feature Extraction:** Extracts interpretable spatial-temporal video artifacts (inter-frame gradient divergence, optical flow acceleration) and acoustic-spectral cues (CQCCs, spectral flux, voice pitch continuity).
5. **L4 Asymmetric Fusion & Vacuity Calibration:** Employs Subjective Logic vacuity shrinkage to attenuate out-of-distribution confidence toward an indecision band, refusing unsupported classifications under domain drift.
6. **L5 Tamper-Evident Chain-of-Custody:** Anchors every analytical intermediate into a chunk-level Merkle tree, generating cryptographic proof of analysis integrity verifiable under a trusted root anchor (compliant with ISO/IEC 27037 and FRE 901(b)(9)).

---

## Repository Structure

```
MOSAIC/
├── code/                         # Implementation source code & reproduction scripts
│   ├── src/mosaic/               # Core MOSAIC library (L0-L5 stages, fusion, models)
│   ├── tests/                    # Pytest test suite and test fixtures
│   ├── scripts/                  # End-to-end pipeline execution & verification scripts
│   ├── requirements.txt          # Python dependencies
│   ├── pyproject.toml            # Project packaging metadata
│   └── README.md                 # Code execution guide
│
├── LICENSE                       # MIT License
└── README.md                     # Project overview (this file)
```

---

## Quick Start

### 1. Installation

Clone the repository and install the dependencies:
```bash
git clone https://github.com/AJ1312/MOSAIC.git
cd MOSAIC/code
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### 2. Run Test Suite

Verify the implementation and forensic integrity checks:
```bash
pytest tests/ -v
```

### 3. Chain-of-Custody Demonstration

Run an end-to-end demonstration of the Merkle tree chain-of-custody and tamper-detection engine:
```bash
python scripts/10_custody_demo.py
```

---

## Citation

If you find this work or codebase useful in your research, please cite our paper:

```bibtex
@article{sharma2026mosaic,
  author    = {Sharma, Ajitesh and Mahesh, Rayban Pranav and Kaila, Marmik Pradip and Priya R, Padma},
  title     = {{MOSAIC}: An Auditable Architecture for Multimodal Deepfake Forensics},
  journal   = {IEEE Access},
  year      = {2026},
  note      = {Submitted for peer review}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
