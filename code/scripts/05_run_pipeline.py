#!/usr/bin/env python3
"""Stage 05 — run the complete L0-L4 pipeline end to end on the held-out test split.

This is the integration run: real files on disk, real ffmpeg canonicalisation, real
registry lookups, real provenance checks, the full cascade, and a sealed custody record
per clip written to an append-only hash-chained ledger. Timings here are wall-clock for
the whole pipeline, not for feature extraction alone.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

import numpy as np
from _common import CACHE, MODELS_DIR, OUTPUTS, banner, load_json, save_json

from mosaic.config import MosaicConfig, apply_tier
from mosaic.data.corpus import read_manifest
from mosaic.l1_hash import HashRegistry
from mosaic.pipeline import ModelBundle, MosaicPipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="A")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="0 = all clips in the split")
    ap.add_argument("--tier", default=None, help="force a device tier")
    args = ap.parse_args()

    banner("Stage 05 — full L0-L4 pipeline run")
    profile = load_json(OUTPUTS / "00_device_profile" / "device_profile.json")
    tier = args.tier or profile["tier"]
    cfg = apply_tier(MosaicConfig(), tier)
    print(f"  device tier: {tier}   config digest: {cfg.digest()[:16]}")

    bundle = ModelBundle.load(MODELS_DIR)
    rows = [r for r in read_manifest(OUTPUTS / "03_dataset" / f"dataset_manifest_{args.corpus}.csv")
            if r["split"] == args.split]
    if args.limit:
        rows = rows[: args.limit]
    print(f"  corpus {args.corpus} / split '{args.split}': {len(rows)} clips")

    # A fresh registry per run keeps L1 statistics interpretable: every clip is a genuine
    # miss, so the short-circuit rate reported later is measured deliberately in stage 09
    # rather than accumulating silently here.
    registry_path = OUTPUTS / "registry" / f"run_{args.corpus}_{args.split}.sqlite"
    if registry_path.exists():
        registry_path.unlink()
    ledger_path = OUTPUTS / "custody" / "ledger.jsonl"
    if ledger_path.exists():
        ledger_path.unlink()

    registry = HashRegistry(registry_path, cfg.l1)
    pipe = MosaicPipeline(cfg, bundle, device_profile=profile, registry=registry,
                          workdir=CACHE / "pipeline_work", synthetic_data=True)

    results = []
    t_start = time.perf_counter()
    for i, row in enumerate(rows, 1):
        res = pipe.process(row["path"], clip_id=row["clip_id"])
        results.append({
            "clip_id": row["clip_id"],
            "true_label": row["label"],
            "true_video_fake": row["video_fake"],
            "true_audio_fake": row["audio_fake"],
            "true_sync": row["sync_state"],
            "provenance_scenario": row["provenance_scenario"],
            **res.to_dict(),
        })
        if i % 25 == 0:
            print(f"    ... {i}/{len(rows)} clips  "
                  f"({(time.perf_counter() - t_start) / i:.3f}s/clip)", flush=True)
    total = time.perf_counter() - t_start

    save_json(results, CACHE / f"pipeline_results_{args.corpus}_{args.split}.json")

    verdicts = Counter(r["verdict"] for r in results)
    correct = sum(1 for r in results if r["verdict"] == r["true_label"])
    unknown = sum(1 for r in results if r["verdict"] == "unknown_inconclusive")
    stage_times = {}
    for key in ("l0_ingest", "l1_triage", "l2_provenance", "l3_tier0", "l3_visual",
                "l3_audio", "l3_av", "l3_tier2", "fusion", "total"):
        vals = [r["timings_s"].get(key) for r in results if r["timings_s"].get(key) is not None]
        if vals:
            stage_times[key] = {"mean_s": round(float(np.mean(vals)), 5),
                                "median_s": round(float(np.median(vals)), 5),
                                "n": len(vals)}

    tier2_run = sum(1 for r in results if "L3.Tier2" in r["stages_run"])
    tier0_exit = sum(1 for r in results if "L3.Tier1" in r["stages_skipped"])
    clashes = sum(1 for r in results
                  if (r.get("l2_provenance") or {}).get("integrity_clash"))
    conflicts = sum(1 for r in results
                    if "modality_conflict" in ((r.get("fusion") or {}).get("flags") or []))

    summary = {
        "corpus": args.corpus, "split": args.split, "device_tier": tier,
        "config_digest": cfg.digest(),
        "n_clips": len(results),
        "wall_seconds_total": round(total, 2),
        "wall_seconds_per_clip": round(total / max(len(results), 1), 4),
        "verdict_counts": dict(verdicts),
        "exact_five_way_accuracy": round(correct / max(len(results), 1), 4),
        "abstention_rate_unknown": round(unknown / max(len(results), 1), 4),
        "tier2_escalation_rate": round(tier2_run / max(len(results), 1), 4),
        "tier0_early_exit_rate": round(tier0_exit / max(len(results), 1), 4),
        "integrity_clashes_detected": clashes,
        "modality_conflicts_flagged": conflicts,
        "stage_times": stage_times,
        "custody_records_written": sum(1 for r in results if r["custody_record_id"]),
    }
    save_json(summary, OUTPUTS / "04_results" / "pipeline_run_summary.json")

    from mosaic.l4_custody import verify_ledger

    ledger_state = verify_ledger(ledger_path)
    save_json(ledger_state, OUTPUTS / "07_chain_of_custody_demo" / "ledger_verification.json")

    print(f"\n  processed {len(results)} clips in {total:.1f}s "
          f"({total / max(len(results), 1):.3f}s/clip)")
    print(f"  exact five-way accuracy : {summary['exact_five_way_accuracy']:.4f}")
    print(f"  abstention (UNKNOWN)    : {summary['abstention_rate_unknown']:.4f}")
    print(f"  Tier-0 early exits      : {summary['tier0_early_exit_rate']:.4f}")
    print(f"  Tier-2 escalations      : {summary['tier2_escalation_rate']:.4f}")
    print(f"  integrity clashes       : {clashes}")
    print(f"  modality conflicts      : {conflicts}")
    print(f"  custody records written : {summary['custody_records_written']}")
    print(f"  ledger                  : {ledger_state['n_entries']} entries, "
          f"valid={ledger_state['valid']}")
    print(f"\n  verdict distribution: {dict(verdicts)}")


if __name__ == "__main__":
    main()
