#!/usr/bin/env python3
"""Stage 06 — evaluation: audited tuples, ROC/PR curves, and the five-way confusion matrix.

Reports the audited tuple (AUC with bootstrap CI, above-floor margin, recall@FPR=0.1%,
calibration error) rather than a bare AUC, for each of the three modality questions, and
the exact five-way verdict behaviour including abstention.
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
from _common import (MODELS_DIR, OUTPUTS, PALETTE, banner, fuse_from_cache, load_cached,
                     save_figure, save_json, save_table, setup_style)
import matplotlib.pyplot as plt

from mosaic.audit import audited_tuple, c3_coherence_probe, roc_auc
from mosaic.config import MosaicConfig
from mosaic.fusion import CELL_NAMES, Verdict
from mosaic.pipeline import ModelBundle

FIVE_WAY = list(CELL_NAMES) + [Verdict.UNKNOWN.value]
SHORT = {"real_video_real_audio": "RR", "fake_video_real_audio": "FR",
         "real_video_fake_audio": "RF", "fake_video_fake_audio": "FF",
         "unknown_inconclusive": "UNK"}


def roc_points(scores, labels):
    scores = np.asarray(scores)
    labels = np.asarray(labels).astype(bool)
    order = np.argsort(-scores)
    tp = np.cumsum(labels[order])
    fp = np.cumsum(~labels[order])
    tpr = tp / max(labels.sum(), 1)
    fpr = fp / max((~labels).sum(), 1)
    return np.concatenate([[0], fpr, [1]]), np.concatenate([[0], tpr, [1]])


def pr_points(scores, labels):
    scores = np.asarray(scores)
    labels = np.asarray(labels).astype(bool)
    order = np.argsort(-scores)
    tp = np.cumsum(labels[order])
    prec = tp / np.arange(1, len(scores) + 1)
    rec = tp / max(labels.sum(), 1)
    return rec, prec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="A")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    banner("Stage 06 — evaluation")
    setup_style()
    cfg = MosaicConfig()
    bundle = ModelBundle.load(MODELS_DIR)
    npz, clip_ids, meta = load_cached(args.corpus)

    split = np.array([m["split"] for m in meta])
    sel = np.nonzero(split == args.split)[0]
    video_fake = np.array([meta[i]["video_fake"] for i in sel], dtype=bool)
    audio_fake = np.array([meta[i]["audio_fake"] for i in sel], dtype=bool)
    desynced = np.array([meta[i]["sync_state"] == "desynced" for i in sel], dtype=bool)
    true_label = [meta[i]["label"] for i in sel]
    print(f"  corpus {args.corpus} / {args.split}: {len(sel)} clips")

    results = fuse_from_cache(bundle, npz, cfg, sel)
    p_video = np.array([r.p_video_fake for r in results])
    p_audio = np.array([r.p_audio_fake for r in results])
    p_desync = np.array([r.branch_summary["audiovisual"]["p_fake"] for r in results])
    pred_label = [r.verdict.value for r in results]

    # ---- C3 floors, computed on this split's authentic clips ---------------------------
    floor_v = c3_coherence_probe(npz["visual"][sel][~video_fake]).numbers["floor_auc"]
    floor_a = c3_coherence_probe(npz["audio"][sel][~audio_fake]).numbers["floor_auc"]
    floor_s = c3_coherence_probe(npz["av"][sel][~desynced]).numbers["floor_auc"]

    tuples = [
        audited_tuple("MOSAIC-AV fused", "video track manipulated", p_video, video_fake, floor_v),
        audited_tuple("MOSAIC-AV fused", "audio track manipulated", p_audio, audio_fake, floor_a),
        audited_tuple("AV branch", "tracks desynchronised", p_desync, desynced, floor_s),
    ]
    # Single-branch references, so the fused numbers can be read against their inputs.
    p_v_only = np.array([r.branch_summary["visual"]["p_fake"] for r in results])
    p_a_only = np.array([r.branch_summary["audio"]["p_fake"] for r in results])
    tuples.append(audited_tuple("visual branch alone", "video track manipulated",
                                p_v_only, video_fake, floor_v))
    tuples.append(audited_tuple("audio branch alone", "audio track manipulated",
                                p_a_only, audio_fake, floor_a))

    rows = [t.to_dict() for t in tuples]
    save_table(rows, OUTPUTS / "04_results", "audited_tuple",
               caption=("Audited tuple per task. 'above_floor_margin' is AUC minus the C3 "
                        "real-vs-real coherence floor — the honest headline, since scoring "
                        "against 0.5 credits a model for structure that exists among "
                        "authentic clips alone."))
    print("\n  audited tuples:")
    for t in tuples:
        print(f"    {t.method:<22} {t.task:<28} AUC {t.auc:.4f} "
              f"[{t.auc_ci_low:.4f},{t.auc_ci_high:.4f}]  floor {t.floor_auc:.3f}  "
              f"margin {t.above_floor_margin:+.4f}  R@FPR0.1% {t.recall_at_fpr_001pct:.3f}  "
              f"ECE {t.ece:.4f}")

    # ---- five-way confusion ------------------------------------------------------------
    cm = np.zeros((4, 5), dtype=int)
    true_idx = {n: i for i, n in enumerate(CELL_NAMES)}
    pred_idx = {n: i for i, n in enumerate(FIVE_WAY)}
    for t, p in zip(true_label, pred_label):
        cm[true_idx[t], pred_idx[p]] += 1

    cm_rows = []
    for i, tname in enumerate(CELL_NAMES):
        row = {"true": SHORT[tname]}
        row.update({f"pred_{SHORT[p]}": int(cm[i, j]) for j, p in enumerate(FIVE_WAY)})
        row["n"] = int(cm[i].sum())
        row["exact_correct"] = float(cm[i, i] / max(cm[i].sum(), 1))
        cm_rows.append(row)
    save_table(cm_rows, OUTPUTS / "04_results", "five_way_confusion",
               caption="Five-way verdict confusion. UNK is abstention, not an error class.")

    exact = float(np.mean([t == p for t, p in zip(true_label, pred_label)]))
    unknown_rate = float(np.mean([p == Verdict.UNKNOWN.value for p in pred_label]))
    decided = [(t, p) for t, p in zip(true_label, pred_label) if p != Verdict.UNKNOWN.value]
    exact_decided = float(np.mean([t == p for t, p in decided])) if decided else float("nan")

    # Per-modality accuracy among decided clips.
    vid_acc = float(np.mean((p_video >= 0.5) == video_fake))
    aud_acc = float(np.mean((p_audio >= 0.5) == audio_fake))

    summary = {
        "corpus": args.corpus, "split": args.split, "n": int(len(sel)),
        "exact_five_way_accuracy_all": round(exact, 4),
        "exact_five_way_accuracy_when_decided": round(exact_decided, 4),
        "abstention_rate": round(unknown_rate, 4),
        "video_marginal_accuracy": round(vid_acc, 4),
        "audio_marginal_accuracy": round(aud_acc, 4),
        "coherence_floors": {"visual": floor_v, "audio": floor_a, "audiovisual": floor_s},
        "audited_tuples": rows,
        "verdict_counts": dict(Counter(pred_label)),
    }
    save_json(summary, OUTPUTS / "04_results" / "evaluation_summary.json")
    print(f"\n  exact five-way accuracy      : {exact:.4f} "
          f"({exact_decided:.4f} among decided clips)")
    print(f"  abstention rate (UNKNOWN)    : {unknown_rate:.4f}")
    print(f"  video marginal accuracy      : {vid_acc:.4f}")
    print(f"  audio marginal accuracy      : {aud_acc:.4f}")

    # ---- risk / coverage trade-off ------------------------------------------------------
    # The per-modality indecision band is an operating-point choice, so it is reported as a
    # curve rather than asserted as a constant. Each point re-derives the verdicts at a
    # different band width; the configured value is marked.
    sweep_rows = []
    for margin in (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30):
        res_m = fuse_from_cache(bundle, npz, cfg, sel, modality_margin=margin)
        pred_m = np.array([r.verdict.value for r in res_m])
        decided = pred_m != Verdict.UNKNOWN.value
        sweep_rows.append({
            "modality_margin": margin,
            "is_configured": bool(abs(margin - cfg.fusion.modality_margin) < 1e-9),
            "coverage": round(float(decided.mean()), 4),
            "exact_five_way_all": round(float((pred_m == np.array(true_label)).mean()), 4),
            "exact_five_way_when_decided": (
                round(float((pred_m[decided] == np.array(true_label)[decided]).mean()), 4)
                if decided.any() else float("nan")),
        })
    save_table(sweep_rows, OUTPUTS / "04_results", "risk_coverage_sweep",
               caption=("Verdict accuracy against coverage as the per-modality indecision "
                        "band widens. The configured operating point is flagged."))
    print("\n  risk/coverage trade-off (per-modality indecision band):")
    for r in sweep_rows:
        mark = " <- configured" if r["is_configured"] else ""
        print(f"    margin {r['modality_margin']:.2f}  coverage {r['coverage']:.3f}  "
              f"accuracy(decided) {r['exact_five_way_when_decided']:.4f}{mark}")

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    cov = [r["coverage"] for r in sweep_rows]
    acc = [r["exact_five_way_when_decided"] for r in sweep_rows]
    ax.plot(cov, acc, "o-", color=PALETTE["accent"], lw=1.6, ms=5)
    for r in sweep_rows:
        if r["is_configured"]:
            ax.plot(r["coverage"], r["exact_five_way_when_decided"], "o",
                    ms=11, mfc="none", mec=PALETTE["fake"], mew=2)
            ax.annotate(f"configured\n(band ±{r['modality_margin']})",
                        (r["coverage"], r["exact_five_way_when_decided"]),
                        textcoords="offset points", xytext=(8, -24), fontsize=7,
                        color=PALETTE["fake"])
    for r in sweep_rows:
        ax.annotate(f"{r['modality_margin']:.2f}",
                    (r["coverage"], r["exact_five_way_when_decided"]),
                    textcoords="offset points", xytext=(4, 6), fontsize=6,
                    color=PALETTE["neutral"])
    ax.set_xlabel("coverage (fraction of clips given a decided verdict)")
    ax.set_ylabel("exact five-way accuracy among decided clips")
    ax.set_title("Risk vs coverage — SYNTHETIC DATA")
    save_figure(fig, OUTPUTS / "04_results", "risk_coverage",
                caption=("Widening the indecision band trades coverage for accuracy on the "
                         "clips the system does decide."))

    # ---- figures -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9))
    for scores, labels, name, colour in (
        (p_video, video_fake, "video manipulated", PALETTE["video"]),
        (p_audio, audio_fake, "audio manipulated", PALETTE["audio"]),
        (p_desync, desynced, "tracks desynchronised", PALETTE["av"]),
    ):
        fpr, tpr = roc_points(scores, labels)
        axes[0].plot(fpr, tpr, color=colour, lw=1.8,
                     label=f"{name} (AUC {roc_auc(scores, labels):.3f})")
        rec, prec = pr_points(scores, labels)
        axes[1].plot(rec, prec, color=colour, lw=1.8, label=name)
    axes[0].plot([0, 1], [0, 1], ls="--", lw=1, color=PALETTE["neutral"], label="chance")
    axes[0].set_xlabel("false positive rate")
    axes[0].set_ylabel("true positive rate")
    axes[0].set_title("ROC")
    axes[0].legend(loc="lower right")
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].set_title("Precision-recall")
    axes[1].legend(loc="lower left")
    fig.suptitle("MOSAIC-AV detection performance — SYNTHETIC DATA", fontsize=10)
    save_figure(fig, OUTPUTS / "04_results", "roc_pr_curves",
                caption="ROC and PR curves per modality question on synthetic corpus A test split.")

    # Confusion matrix heatmap.
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(5), [SHORT[p] for p in FIVE_WAY])
    ax.set_yticks(range(4), [SHORT[t] for t in CELL_NAMES])
    ax.set_xlabel("predicted verdict")
    ax.set_ylabel("true class")
    ax.set_title("Five-way verdict confusion (row-normalised)")
    ax.grid(False)
    for i in range(4):
        for j in range(5):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.55 else "#22303c")
    fig.colorbar(im, ax=ax, fraction=0.045, label="row fraction")
    save_figure(fig, OUTPUTS / "04_results", "five_way_confusion",
                caption="UNK column is deliberate abstention under the fusion decision rule.")

    # Calibration.
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    for probs, labels, name, colour in ((p_video, video_fake, "video", PALETTE["video"]),
                                        (p_audio, audio_fake, "audio", PALETTE["audio"])):
        edges = np.linspace(0, 1, 9)
        xs, ys = [], []
        for k in range(8):
            m = (probs > edges[k]) & (probs <= edges[k + 1]) if k else (probs <= edges[1])
            if m.sum() >= 3:
                xs.append(probs[m].mean())
                ys.append(labels[m].mean())
        ax.plot(xs, ys, "o-", color=colour, lw=1.6, ms=4, label=name)
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=PALETTE["neutral"], label="perfect")
    ax.set_xlabel("predicted P(manipulated)")
    ax.set_ylabel("observed fraction manipulated")
    ax.set_title("Calibration")
    ax.legend()
    save_figure(fig, OUTPUTS / "04_results", "calibration",
                caption="Reliability diagram for the fused per-modality probabilities.")

    print(f"\n  figures + tables -> {OUTPUTS / '04_results'}")


if __name__ == "__main__":
    main()
