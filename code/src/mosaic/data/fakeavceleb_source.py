"""FakeAVCeleb dataset source reader for MOSAIC.

FakeAVCeleb organises clips by condition folder:

    RealVideo-RealAudio/   -> label_video=real, label_audio=real
    FakeVideo-RealAudio/   -> label_video=fake, label_audio=real
    RealVideo-FakeAudio/   -> label_video=real, label_audio=fake
    FakeVideo-FakeAudio/   -> label_video=fake, label_audio=fake

Sub-structure within each condition:
    <condition>/<ethnicity>/<gender>/<celebrity_id>/<video>.mp4

Reference
---------
Khalid, Tariq, Kim, Woo — "FakeAVCeleb: A Novel Audio-Video Multimodal
Deepfake Dataset", NeurIPS Datasets & Benchmarks Track, 2021.
https://sites.google.com/view/fakeavceleb/home  (CC BY 4.0)

Usage
-----
>>> from mosaic.data.fakeavceleb_source import build_fakeavceleb_manifest
>>> records = build_fakeavceleb_manifest(Path("/data/fakeavceleb/FakeAVCeleb"))
"""
from __future__ import annotations

import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Label mapping: folder name -> (label_video, label_audio)
# ---------------------------------------------------------------------------
CONDITION_MAP: dict[str, tuple[str, str]] = {
    "RealVideo-RealAudio": ("real", "real"),
    "FakeVideo-RealAudio": ("fake", "real"),
    "RealVideo-FakeAudio": ("real", "fake"),
    "FakeVideo-FakeAudio": ("fake", "fake"),
}

# Short attack-id tags stored in the manifest (readable, no slashes)
_ATTACK_TAG: dict[str, str] = {
    "RealVideo-RealAudio": "FAC_RV_RA",
    "FakeVideo-RealAudio": "FAC_FV_RA",
    "RealVideo-FakeAudio": "FAC_RV_FA",
    "FakeVideo-FakeAudio": "FAC_FV_FA",
}

# ---------------------------------------------------------------------------
# Sampling defaults
# ---------------------------------------------------------------------------
# 250 per condition  -> 1,000 clips total (well-balanced 4-cell design)
# 50  per condition  ->   200 clips total (fast smoke-test / pilot)
DEFAULT_PER_CONDITION: int = 250
PILOT_PER_CONDITION: int = 50


def build_fakeavceleb_manifest(
    root: Path,
    per_condition: int = DEFAULT_PER_CONDITION,
    seed: int = 1337,
    split_fractions: tuple[float, float, float] = (0.60, 0.20, 0.20),
    stage: int = 3,
) -> list[dict]:
    """Scan the FakeAVCeleb root and return a list of manifest dicts.

    Each returned dict has the same keys as MANIFEST_FIELDS in
    scripts/real_01_acquire.py so it can be passed straight to
    write_manifest() or written directly to a dedicated CSV.

    Parameters
    ----------
    root:
        Path to the ``FakeAVCeleb/`` folder that contains the four
        condition sub-directories (``RealVideo-RealAudio/`` etc.).
    per_condition:
        Maximum number of clips sampled from each of the 4 conditions.
        Set to ``PILOT_PER_CONDITION`` (50) for a quick smoke-test.
    seed:
        RNG seed for reproducible sampling and split assignment.
    split_fractions:
        ``(train, dev, test)`` fractions.  Must sum to 1.0.
    stage:
        Stage number written into the manifest (default 3 so as not to
        conflict with the existing Stage 1/2 audio-video manifests).

    Returns
    -------
    list[dict]
        One dict per selected clip, keys: source_dataset, file_path,
        sha256, modality, label_video, label_audio, label_av_sync,
        split, stage, attack_id, speaker, download_method, bytes.
    """
    if abs(sum(split_fractions) - 1.0) > 1e-6:
        raise ValueError(f"split_fractions must sum to 1.0, got {split_fractions}")

    rng = random.Random(seed)
    records: list[dict] = []

    for condition_dir, (label_video, label_audio) in CONDITION_MAP.items():
        cond_path = root / condition_dir
        if not cond_path.exists():
            print(f"  [WARN] FakeAVCeleb: condition dir not found: {cond_path}")
            continue

        all_mp4s = sorted(cond_path.rglob("*.mp4"))
        if not all_mp4s:
            print(f"  [WARN] FakeAVCeleb: no .mp4 files in {cond_path}")
            continue

        sample = rng.sample(all_mp4s, min(per_condition, len(all_mp4s)))

        # Assign deterministic train/dev/test labels
        n = len(sample)
        n_train = int(n * split_fractions[0])
        n_dev   = int(n * split_fractions[1])
        n_test  = n - n_train - n_dev
        splits_list: list[str] = (
            ["train"] * n_train + ["dev"] * n_dev + ["test"] * n_test
        )
        rng.shuffle(splits_list)

        for mp4_path, split_label in zip(sample, splits_list):
            # Extract speaker / ethnicity from path depth
            # Expected: <root>/<condition>/<ethnicity>/<gender>/<speaker_id>/<file.mp4>
            parts = mp4_path.parts
            try:
                cond_idx = next(i for i, p in enumerate(parts) if p == condition_dir)
                ethnicity  = parts[cond_idx + 1] if cond_idx + 1 < len(parts) else "unknown"
                speaker_id = parts[cond_idx + 3] if cond_idx + 3 < len(parts) else "unknown"
            except StopIteration:
                ethnicity = "unknown"
                speaker_id = "unknown"

            # FakeAVCeleb clips always carry both streams (Wav2Lip ensures sync)
            records.append({
                "source_dataset": "FakeAVCeleb",
                "file_path":      str(mp4_path.resolve()),
                "sha256":         "",          # populated by write_manifest
                "modality":       "av",        # both audio + video present
                "label_video":    label_video,
                "label_audio":    label_audio,
                "label_av_sync":  "synced",    # Wav2Lip synchronised
                "split":          split_label,
                "stage":          stage,
                "attack_id":      _ATTACK_TAG[condition_dir],
                "speaker":        speaker_id,
                "download_method": "official_fakeavceleb_download_script",
                "bytes":          mp4_path.stat().st_size if mp4_path.exists() else 0,
            })

    # ---- Summary print ------------------------------------------------
    total = len(records)
    print(f"  FakeAVCeleb manifest: {total} clips across {len(CONDITION_MAP)} conditions")
    from collections import Counter
    for cond, cnt in sorted(Counter(r["attack_id"] for r in records).items()):
        print(f"    {cond}: {cnt} clips")
    split_counts = Counter(r["split"] for r in records)
    print(f"  Split counts: train={split_counts['train']}, "
          f"dev={split_counts['dev']}, test={split_counts['test']}")

    return records
