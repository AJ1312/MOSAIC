"""Device profiling and tier selection.

Every results file records which tier's configuration was actually used, so a reported
number can never be silently attributed to the wrong hardware path.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class DeviceProfile:
    os_name: str
    os_version: str
    machine: str
    cpu_brand: str
    cpu_count_logical: int
    cpu_count_physical: int | None
    ram_gb: float
    python_version: str
    torch_version: str | None
    accelerator: str            # "cuda" | "mps" | "cpu" | "none"
    accelerator_name: str | None
    accelerator_vram_gb: float | None
    ffmpeg_version: str | None
    tier: str
    tier_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sysctl(key: str) -> str | None:
    try:
        out = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def _cpu_brand() -> str:
    if platform.system() == "Darwin":
        return _sysctl("machdep.cpu.brand_string") or platform.processor() or "unknown"
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _physical_cores() -> int | None:
    if platform.system() == "Darwin":
        v = _sysctl("hw.physicalcpu")
        return int(v) if v and v.isdigit() else None
    try:
        return len({
            line.split(":")[1].strip()
            for line in open("/proc/cpuinfo", encoding="utf-8")
            if line.lower().startswith("core id")
        }) or None
    except Exception:
        return None


def _ram_gb() -> float:
    if platform.system() == "Darwin":
        v = _sysctl("hw.memsize")
        if v and v.isdigit():
            return round(int(v) / 1024**3, 2)
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 2)
    except Exception:
        return 0.0


def _ffmpeg_version() -> str | None:
    if shutil.which("ffmpeg") is None:
        return None
    try:
        out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=10)
        return out.stdout.splitlines()[0].strip() if out.stdout else None
    except Exception:
        return None


def _accelerator() -> tuple[str, str | None, float | None, str | None]:
    """Return (kind, name, vram_gb, torch_version)."""
    try:
        import torch
    except Exception:
        return "none", None, None, None

    tv = torch.__version__
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return "cuda", props.name, round(props.total_memory / 1024**3, 2), tv
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        # Apple silicon uses unified memory: there is no separate VRAM figure to report,
        # and inventing one would be a fabricated measurement.
        return "mps", "Apple Silicon GPU (unified memory)", None, tv
    return "cpu", None, None, tv


def select_tier(accel: str, vram_gb: float | None, cores: int, ram_gb: float) -> tuple[str, str]:
    """Pick a device tier.

    Thresholds follow the MOSAIC design document, extended with an explicit Apple-silicon
    rule: unified-memory GPUs report no VRAM, so they are tiered on core count and RAM.
    """
    if accel == "cuda" and vram_gb is not None and vram_gb >= 16:
        return "cloud", f"CUDA GPU with {vram_gb} GB VRAM >= 16 GB"
    if accel == "cuda":
        return "consumer", f"CUDA GPU present with {vram_gb} GB VRAM < 16 GB"
    if accel == "mps":
        if cores >= 8 and ram_gb >= 16:
            return "consumer", (
                f"Apple-silicon MPS accelerator, {cores} logical cores, {ram_gb} GB unified RAM "
                "(no discrete VRAM to threshold on; tiered by cores/RAM)"
            )
        return "edge", f"Apple-silicon MPS but only {cores} cores / {ram_gb} GB RAM"
    if cores >= 8 and ram_gb >= 16:
        return "consumer", f"CPU-only with {cores} logical cores and {ram_gb} GB RAM"
    return "edge", f"CPU-only with {cores} logical cores and {ram_gb} GB RAM below consumer threshold"


def profile_device() -> DeviceProfile:
    accel, accel_name, vram, tv = _accelerator()
    cores = os.cpu_count() or 1
    ram = _ram_gb()
    tier, reason = select_tier(accel, vram, cores, ram)
    return DeviceProfile(
        os_name=platform.system(),
        os_version=platform.release(),
        machine=platform.machine(),
        cpu_brand=_cpu_brand(),
        cpu_count_logical=cores,
        cpu_count_physical=_physical_cores(),
        ram_gb=ram,
        python_version=platform.python_version(),
        torch_version=tv,
        accelerator=accel,
        accelerator_name=accel_name,
        accelerator_vram_gb=vram,
        ffmpeg_version=_ffmpeg_version(),
        tier=tier,
        tier_reason=reason,
    )


def save_profile(profile: DeviceProfile, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
    return path


def torch_device(profile: DeviceProfile | None = None):
    """Best available torch device. Falls back to CPU without complaint."""
    import torch

    profile = profile or profile_device()
    if profile.accelerator == "cuda":
        return torch.device("cuda")
    if profile.accelerator == "mps":
        return torch.device("mps")
    return torch.device("cpu")
