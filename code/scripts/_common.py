"""Shared helpers for the MOSAIC-AV pipeline scripts: paths, IO, figures, tables."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
MODELS_DIR = OUTPUTS / "models"
CACHE = OUTPUTS / "cache"

SYNTHETIC_BANNER = (
    "**SYNTHETIC DEMO DATA — not representative of real detection performance.**"
)

# --------------------------------------------------------------------------------------
# Figure style — one scheme across every figure
# --------------------------------------------------------------------------------------

PALETTE = {
    "real": "#3d7ea6",
    "fake": "#c1553b",
    "video": "#4f7a4a",
    "audio": "#8a6bab",
    "av": "#c98a2e",
    "neutral": "#6b7280",
    "accent": "#2f4858",
}
CLASS_COLORS = {
    "RR": "#3d7ea6", "FR": "#c1553b", "RF": "#c98a2e", "FF": "#7a3b8f",
    "UNK": "#6b7280",
}


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.autolayout": False,
    })


def save_figure(fig, out_dir: Path, name: str, caption: str = "") -> dict[str, str]:
    """Save a figure as both 300-DPI PNG and vector PDF."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    if caption:
        (out_dir / f"{name}.caption.txt").write_text(caption, encoding="utf-8")
    return {"png": str(png.relative_to(ROOT)), "pdf": str(pdf.relative_to(ROOT)),
            "caption": caption}


def save_table(rows: list[dict[str, Any]], out_dir: Path, name: str, caption: str = "",
               synthetic: bool = True) -> dict[str, str]:
    """Save a table as both CSV and a pre-formatted Markdown table."""
    import csv

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    md_path = out_dir / f"{name}.md"
    if not rows:
        csv_path.write_text("", encoding="utf-8")
        md_path.write_text("_(empty table)_\n", encoding="utf-8")
        return {"csv": str(csv_path.relative_to(ROOT)), "md": str(md_path.relative_to(ROOT))}

    fields = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(v) for k, v in r.items()})

    lines = []
    if synthetic:
        lines.append(SYNTHETIC_BANNER)
        lines.append("")
    if caption:
        lines.append(f"_{caption}_")
        lines.append("")
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("|" + "|".join("---" for _ in fields) + "|")
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(k)) for k in fields) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"csv": str(csv_path.relative_to(ROOT)), "md": str(md_path.relative_to(ROOT)),
            "caption": caption}


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        if not np.isfinite(v):
            return "n/a"
        return f"{v:.4f}" if abs(v) < 1000 else f"{v:.1f}"
    if isinstance(v, (list, dict)):
        return json.dumps(v)
    return "" if v is None else str(v)


def save_json(obj: Any, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return path


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78, flush=True)


def require(path: Path, hint: str) -> Path:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"missing required input: {path}\n  -> run {hint} first")
    return path


# --------------------------------------------------------------------------------------
# Cached-feature fusion (used by evaluation, ablation and audit)
# --------------------------------------------------------------------------------------


def load_cached(corpus: str):
    """Load the cached feature arrays and the manifest rows for one corpus."""
    from mosaic.data.corpus import read_manifest

    npz = np.load(CACHE / f"features_{corpus}.npz", allow_pickle=False)
    rows = {r["clip_id"]: r for r in read_manifest(
        OUTPUTS / "03_dataset" / f"dataset_manifest_{corpus}.csv")}
    clip_ids = [str(c) for c in npz["clip_ids"]]
    meta = [rows[c] for c in clip_ids]
    return npz, clip_ids, meta


def fuse_from_cache(bundle, npz, cfg, idx, *, use_visual=True, use_audio=True,
                    use_av=True, use_cnn=True, use_uncertainty=True,
                    modality_margin=None):
    """Run the branch models and fusion over cached features.

    Equivalent to the L3 stage of the pipeline without re-decoding media, so evaluation,
    ablation and the audit controls can all reuse one feature extraction pass. The ``use_*``
    switches are what the ablation study toggles.
    """
    from mosaic.fusion import fuse
    from mosaic.models import BranchPrediction

    logmel = npz["logmel"]
    out = []
    for i in idx:
        if use_visual:
            vis = bundle.visual.predict(npz["visual"][i],
                                        ood_percentile=cfg.fusion.ood_percentile)
        else:
            vis = BranchPrediction.unavailable("visual branch disabled (ablation)")

        if use_audio and bool(npz["audio_available"][i]):
            extra = []
            if use_cnn and bundle.audio_cnn is not None and bundle.audio_cnn.fitted_:
                extra = [float(bundle.audio_cnn.predict_proba([logmel[i]])[0])]
            aud = bundle.audio.predict(npz["audio"][i],
                                       ood_percentile=cfg.fusion.ood_percentile,
                                       extra_probs=extra)
        else:
            aud = BranchPrediction.unavailable(
                "audio branch disabled (ablation)" if not use_audio else "no usable audio")

        if use_av and bool(npz["av_available"][i]):
            av = bundle.av.predict(npz["av"][i], ood_percentile=cfg.fusion.ood_percentile)
        else:
            av = BranchPrediction.unavailable(
                "audiovisual branch disabled (ablation)" if not use_av
                else "audiovisual analysis not applicable")

        if not use_uncertainty:
            # Ablation: strip vacuity so every branch enters fusion at full weight.
            for p in (vis, aud, av):
                if p.available:
                    p.vacuity = 0.0

        out.append(fuse(
            vis, aud, av, bundle.coupling,
            prior_fake_video=cfg.fusion.prior_fake_video,
            prior_fake_audio=cfg.fusion.prior_fake_audio,
            vacuity_penalty=cfg.fusion.vacuity_penalty if use_uncertainty else 0.0,
            decide_threshold=cfg.fusion.decide_threshold,
            confident_llr=cfg.fusion.confident_llr,
            confident_max_vacuity=cfg.fusion.confident_max_vacuity,
            modality_margin=(cfg.fusion.modality_margin if modality_margin is None
                             else modality_margin),
        ))
    return out
