"""Shared pytest fixtures for the MOSAIC-AV test suite."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _have(binary: str) -> bool:
    return shutil.which(binary) is not None


requires_ffmpeg = pytest.mark.skipif(
    not (_have("ffmpeg") and _have("ffprobe")),
    reason="ffmpeg/ffprobe not available",
)


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(20260817)


@pytest.fixture(scope="session")
def tiny_corpus(tmp_path_factory) -> list[dict]:
    """A very small labelled corpus covering all four classes, generated once."""
    from mosaic.data.corpus import build_corpus, plan_corpus

    out = tmp_path_factory.mktemp("tiny_corpus")
    specs = plan_corpus(1, seed=7, splits=(("test", 1.0),))
    records = build_corpus(specs, out, workers=2, progress=False)
    assert all(not r.get("error") for r in records), [r.get("error") for r in records]
    return records


@pytest.fixture(scope="session")
def sample_media(tmp_path_factory) -> Path:
    """One short synthetic clip with both video and audio, written via ffmpeg."""
    out = tmp_path_factory.mktemp("sample") / "sample.mp4"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=4",
        "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=44100:duration=4",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=300)
    return out


@pytest.fixture(scope="session")
def ingested(sample_media, tmp_path_factory):
    """L0 output for the sample clip."""
    from mosaic.config import CanonicalProfile
    from mosaic.l0_ingest import ingest

    work = tmp_path_factory.mktemp("ingest_work")
    return ingest(sample_media, CanonicalProfile(), workdir=work)
