"""Acquisition of REAL deepfake corpora. No synthetic content is produced or accepted here.

Every function in this module either returns real clips from a cited, licensed corpus or
records an honest failure. There is no fallback path that manufactures data: if a modality
cannot be obtained, it is reported unevaluated.

Acquisition strategy, and why it differs by stage
-------------------------------------------------
The corpora are published as single large archives (ASVspoof2019 LA 7.64 GB, In-The-Wild
8.16 GB, AIGVDBench Real 94 GB). Measured against both hosts, throughput caps near
2 MB/s aggregate regardless of concurrency, and each range request costs ~1 s of latency.
That gives a clean crossover:

* **Selective range extraction** (Stage 1, and video at both stages) — pull only the clips
  needed straight out of the remote archive. Reading a 94 GB archive's index costs ~2.5 MB.
  Cheaper whenever the wanted clip count is below roughly 2,300.
* **Bulk download** (Stage 2 audio) — past that crossover, per-request latency dominates
  and fetching the whole archive once is faster.

Both paths are resumable: completed clips are checkpointed and skipped on restart.
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .remote_zip import RemoteZip, TransferStats, parallel_extract

# --------------------------------------------------------------------------------------
# Source registry — every URL is a cited, licensed, publicly reachable corpus
# --------------------------------------------------------------------------------------

ASVSPOOF_LA_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip"
IN_THE_WILD_URL = ("https://huggingface.co/datasets/mueller91/In-The-Wild/"
                   "resolve/main/release_in_the_wild.zip")
AIGVD_BASE = "https://huggingface.co/datasets/AIGVDBench/AIGVDBench/resolve/main/"

SOURCES: dict[str, dict[str, Any]] = {
    "ASVspoof2019-LA": {
        "modality": "audio",
        "citation": "Todisco et al., ASVspoof 2019, Edinburgh DataShare 10283/3336",
        "licence": "CC BY 4.0 / ODC-BY 1.0",
        "url": ASVSPOOF_LA_URL,
        "access": "open, no login",
    },
    "In-The-Wild": {
        "modality": "audio",
        "citation": "Muller et al. 2022, 'Does Audio Deepfake Detection Generalize?'",
        "licence": "open, research use",
        "url": IN_THE_WILD_URL,
        "access": "open, no login (Hugging Face)",
    },
    "AIGVDBench": {
        "modality": "video",
        "citation": "AIGVDBench, Hugging Face AIGVDBench/AIGVDBench",
        "licence": "see dataset card",
        "url": AIGVD_BASE,
        "access": "open, no login (Hugging Face); gated=False",
    },
}

#: AIGVDBench generator archives, smallest first — the fake class is stratified across
#: these so no single generator dominates the sample.
AIGVD_GENERATORS: list[tuple[str, str]] = [
    ("Open-Sora", "AIGVDBench/OpenSource/T2V/Open-Sora.zip"),
    ("pika", "AIGVDBench/ClosedSource/pika.zip"),
    ("vidu", "AIGVDBench/ClosedSource/vidu.zip"),
    ("Cogvideox1.5", "AIGVDBench/OpenSource/T2V/Cogvideox1.5.zip"),
    ("AnimateDiff", "AIGVDBench/OpenSource/T2V/AnimateDiff.zip"),
    ("SVD", "AIGVDBench/OpenSource/I2V/SVD.zip"),
    ("Luma", "AIGVDBench/ClosedSource/Luma.zip"),
    ("HunyuanVideo", "AIGVDBench/OpenSource/T2V/HunyuanVideo.zip"),
]
AIGVD_REAL = "AIGVDBench/Real/Real.zip"


# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------


@dataclass
class Attempt:
    """One acquisition attempt, logged whether it succeeded, failed or was skipped."""

    dataset: str
    modality: str
    method: str                 # "range_extraction" | "bulk_download" | "probe"
    status: str                 # "ok" | "skip" | "fail"
    files: int = 0
    bytes_pulled: int = 0
    reason: str = ""
    citation: str = ""
    licence: str = ""
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AcquisitionLog:
    """Accumulates attempts and writes the markdown log the protocol requires."""

    def __init__(self, path: Path, budget_gb: float):
        self.path = Path(path)
        self.budget_bytes = int(budget_gb * 1e9)
        self.attempts: list[Attempt] = []
        self.running_bytes = 0

    def add(self, attempt: Attempt) -> Attempt:
        self.attempts.append(attempt)
        self.running_bytes += attempt.bytes_pulled
        print(f"  [{attempt.status.upper():4s}] {attempt.dataset:18s} "
              f"{attempt.modality:6s} {attempt.method:17s} "
              f"files={attempt.files:5d} pulled={attempt.bytes_pulled/1e6:8.1f} MB  "
              f"running={self.running_bytes/1e6:8.1f} MB", flush=True)
        if attempt.reason:
            print(f"         {attempt.reason[:150]}", flush=True)
        return attempt

    def budget_remaining(self) -> int:
        return max(0, self.budget_bytes - self.running_bytes)

    def over_budget(self) -> bool:
        return self.running_bytes >= self.budget_bytes

    def write(self, stage: int) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        L = [f"# Dataset acquisition log — Stage {stage}", "",
             "**REAL DATA ONLY.** No synthetic, generated, TTS or fabricated content is "
             "produced or accepted by this pipeline at any stage. Where a modality could "
             "not be obtained from a real, licensed, reachable source within budget, it is "
             "recorded below as skipped and reported downstream as *unevaluated* — the gap "
             "is never filled.", "",
             f"Footprint budget: {self.budget_bytes/1e9:.1f} GB. "
             f"Total pulled: {self.running_bytes/1e9:.3f} GB.", "",
             "| dataset | modality | method | status | files | MB pulled | reason |",
             "|---|---|---|---|---|---|---|"]
        for a in self.attempts:
            L.append(f"| {a.dataset} | {a.modality} | {a.method} | **{a.status}** | "
                     f"{a.files} | {a.bytes_pulled/1e6:.1f} | {a.reason[:220]} |")
        L += ["", "## Sources and licences", ""]
        for name, meta in SOURCES.items():
            L.append(f"- **{name}** ({meta['modality']}) — {meta['citation']}; "
                     f"licence: {meta['licence']}; access: {meta['access']}")
        self.path.write_text("\n".join(L) + "\n", encoding="utf-8")
        return self.path


# --------------------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------------------


class Checkpoint:
    """Records completed work so an interrupted multi-hour run resumes instead of restarting."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done: dict[str, dict] = {}
        if self.path.exists():
            try:
                self.done = json.loads(self.path.read_text())
            except json.JSONDecodeError:
                self.done = {}

    def has(self, key: str) -> bool:
        return key in self.done

    def adopt(self, key: str, dest: Path, record: dict) -> bool:
        """Treat an already-downloaded file as done, rebuilding its record if needed.

        The checkpoint is an index, not the source of truth — the bytes on disk are. If the
        checkpoint is cleared (or lost) while the media survives, re-fetching gigabytes that
        are already present would be pure waste, so a present file of the right size is
        accepted and its manifest record reconstructed.
        """
        if not dest.exists() or dest.stat().st_size == 0:
            return False
        if key not in self.done:
            record = dict(record)
            record["bytes"] = dest.stat().st_size
            record["download_method"] = record.get("download_method", "") + "+resumed_from_disk"
            self.done[key] = record
        return True

    def mark(self, key: str, payload: dict) -> None:
        self.done[key] = payload
        if len(self.done) % 25 == 0:
            self.flush()

    def flush(self) -> None:
        self.path.write_text(json.dumps(self.done, indent=1))


# --------------------------------------------------------------------------------------
# ASVspoof2019 LA
# --------------------------------------------------------------------------------------


def parse_asvspoof_protocol(text: str) -> list[dict[str, str]]:
    """Parse a CM protocol file: speaker, utt_id, -, attack_id, label."""
    rows = []
    for line in text.strip().split("\n"):
        parts = line.split()
        if len(parts) >= 5:
            rows.append({"speaker": parts[0], "utt_id": parts[1],
                         "attack": parts[3], "label": parts[4]})
    return rows


def stratified_pick(rows: list[dict], n_bonafide: int, n_spoof: int, seed: int = 1337
                    ) -> list[dict]:
    """Balanced bonafide/spoof pick, with spoof spread evenly across attack IDs.

    Sampling by attack matters: ASVspoof's spoof class is a union of distinct synthesis
    systems, and taking the first N encountered would over-represent whichever attack the
    protocol happens to list first.
    """
    import random

    rng = random.Random(seed)
    bona = [r for r in rows if r["label"] == "bonafide"]
    spoof = [r for r in rows if r["label"] == "spoof"]
    rng.shuffle(bona)
    picked = bona[:n_bonafide]

    by_attack: dict[str, list[dict]] = {}
    for r in spoof:
        by_attack.setdefault(r["attack"], []).append(r)
    for v in by_attack.values():
        rng.shuffle(v)
    attacks = sorted(by_attack)
    per = max(1, n_spoof // max(len(attacks), 1))
    chosen: list[dict] = []
    for a in attacks:
        chosen.extend(by_attack[a][:per])
    # Top up round-robin if integer division left a shortfall.
    i = 0
    while len(chosen) < n_spoof and attacks:
        a = attacks[i % len(attacks)]
        pool = by_attack[a]
        if len(pool) > per:
            chosen.append(pool[per + (i // len(attacks))])
        i += 1
        if i > 10 * n_spoof:
            break
    return picked + chosen[:n_spoof]


ASVSPOOF_SPLITS = {
    "train": ("LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
              "LA/ASVspoof2019_LA_train/flac"),
    "dev": ("LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
            "LA/ASVspoof2019_LA_dev/flac"),
    "eval": ("LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
             "LA/ASVspoof2019_LA_eval/flac"),
}


def acquire_asvspoof_remote(
    out_dir: Path, plan: dict[str, tuple[int, int]], log: AcquisitionLog,
    checkpoint: Checkpoint, *, seed: int = 1337, workers: int = 5,
    local_archive: Path | None = None,
) -> list[dict[str, Any]]:
    """Extract ASVspoof2019 LA clips, from a local archive when present or remotely otherwise.

    Stage 2 wants thousands of clips. Past roughly 2,300 the per-request latency of range
    extraction exceeds the cost of fetching the 7.64 GB archive once, so Stage 2 downloads
    it (resumably) and reads members locally; Stage 1's few hundred clips stay remote.
    """
    out_dir = Path(out_dir)
    records: list[dict[str, Any]] = []
    extra_fetched = [0]
    t0 = time.perf_counter()

    use_local = local_archive is not None and Path(local_archive).exists()
    if use_local:
        return _acquire_asvspoof_local(Path(local_archive), out_dir, plan, log, checkpoint,
                                       seed=seed, t0=t0)
    try:
        rz = RemoteZip(ASVSPOOF_LA_URL).open()
    except Exception as exc:
        log.add(Attempt("ASVspoof2019-LA", "audio", "range_extraction", "fail",
                        reason=f"cannot open remote archive: {type(exc).__name__}: {exc}",
                        citation=SOURCES["ASVspoof2019-LA"]["citation"],
                        licence=SOURCES["ASVspoof2019-LA"]["licence"]))
        return records

    n_files = 0
    try:
        for split, (n_bona, n_spoof) in plan.items():
            proto_path, flac_dir = ASVSPOOF_SPLITS[split]
            rows = parse_asvspoof_protocol(rz.read(proto_path).decode())
            picks = stratified_pick(rows, n_bona, n_spoof, seed=seed)
            pending, meta_by_member = [], {}
            for r in picks:
                key = f"asvspoof/{split}/{r['utt_id']}"
                dest = out_dir / "asvspoof2019_la" / split / f"{r['utt_id']}.flac"
                base = {"source_dataset": "ASVspoof2019-LA", "file_path": str(dest),
                        "modality": "audio",
                        "label_audio": "fake" if r["label"] == "spoof" else "real",
                        "label_video": "", "label_av_sync": "", "split": split,
                        "attack_id": r["attack"], "speaker": r["speaker"],
                        "download_method": "range_extraction"}
                if checkpoint.adopt(key, dest, base):
                    records.append(checkpoint.done[key])
                    continue
                member = f"{flac_dir}/{r['utt_id']}.flac"
                pending.append(member)
                meta_by_member[member] = (key, dest, r)

            def sink(member: str, data: bytes) -> bool:
                key, dest, r = meta_by_member[member]
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                rec = {"source_dataset": "ASVspoof2019-LA", "file_path": str(dest),
                       "modality": "audio",
                       "label_audio": "fake" if r["label"] == "spoof" else "real",
                       "label_video": "", "label_av_sync": "", "split": split,
                       "attack_id": r["attack"], "speaker": r["speaker"],
                       "download_method": "range_extraction_parallel", "bytes": len(data)}
                records.append(rec)
                checkpoint.mark(key, rec)
                return True

            if pending:
                n_got, agg = parallel_extract(
                    ASVSPOOF_LA_URL, pending, sink, workers=workers,
                    on_done=lambda d, t: print(f"    {split}: {d}/{t} clips", flush=True))
                n_files += n_got
                extra_fetched[0] += agg.bytes_fetched
    except StopIteration:
        pass
    finally:
        checkpoint.flush()
        stats = rz.stats.as_dict()
        rz.close()

    total_fetched = rz.stats.bytes_fetched + extra_fetched[0]
    log.add(Attempt(
        "ASVspoof2019-LA", "audio", "range_extraction_parallel", "ok" if records else "fail",
        files=len(records), bytes_pulled=total_fetched,
        reason=(f"selective extraction from a {rz.archive_bytes/1e9:.2f} GB archive; "
                f"{total_fetched/1e6:.1f} MB transferred across {workers} parallel handles, "
                f"{time.perf_counter()-t0:.0f}s. Stratified across attack IDs."),
        citation=SOURCES["ASVspoof2019-LA"]["citation"],
        licence=SOURCES["ASVspoof2019-LA"]["licence"],
        seconds=round(time.perf_counter() - t0, 1)))
    return records


def _acquire_asvspoof_local(archive: Path, out_dir: Path, plan: dict[str, tuple[int, int]],
                            log: AcquisitionLog, checkpoint: Checkpoint, *, seed: int,
                            t0: float) -> list[dict[str, Any]]:
    """Read the wanted members out of a locally downloaded LA.zip."""
    import zipfile

    records: list[dict[str, Any]] = []
    written = 0
    with zipfile.ZipFile(archive) as zf:
        for split, (n_bona, n_spoof) in plan.items():
            proto_path, flac_dir = ASVSPOOF_SPLITS[split]
            rows = parse_asvspoof_protocol(zf.read(proto_path).decode())
            for r in stratified_pick(rows, n_bona, n_spoof, seed=seed):
                key = f"asvspoof/{split}/{r['utt_id']}"
                dest = out_dir / "asvspoof2019_la" / split / f"{r['utt_id']}.flac"
                base = {"source_dataset": "ASVspoof2019-LA", "file_path": str(dest),
                        "modality": "audio",
                        "label_audio": "fake" if r["label"] == "spoof" else "real",
                        "label_video": "", "label_av_sync": "", "split": split,
                        "attack_id": r["attack"], "speaker": r["speaker"],
                        "download_method": "local_archive"}
                if checkpoint.adopt(key, dest, base):
                    records.append(checkpoint.done[key])
                    continue
                try:
                    data = zf.read(f"{flac_dir}/{r['utt_id']}.flac")
                except KeyError:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                rec = {"source_dataset": "ASVspoof2019-LA", "file_path": str(dest),
                       "modality": "audio",
                       "label_audio": "fake" if r["label"] == "spoof" else "real",
                       "label_video": "", "label_av_sync": "", "split": split,
                       "attack_id": r["attack"], "speaker": r["speaker"],
                       "download_method": "local_archive", "bytes": len(data)}
                records.append(rec)
                checkpoint.mark(key, rec)
                written += 1
                if written % 500 == 0:
                    print(f"    {split}: {written} clips extracted locally", flush=True)
    checkpoint.flush()
    log.add(Attempt(
        "ASVspoof2019-LA", "audio", "bulk_download+local_extract",
        "ok" if records else "fail", files=len(records),
        bytes_pulled=archive.stat().st_size,
        reason=(f"archive downloaded once ({archive.stat().st_size/1e9:.2f} GB) and members "
                f"read locally; faster than range extraction above ~2,300 clips. "
                f"Stratified across attack IDs. {time.perf_counter()-t0:.0f}s."),
        citation=SOURCES["ASVspoof2019-LA"]["citation"],
        licence=SOURCES["ASVspoof2019-LA"]["licence"],
        seconds=round(time.perf_counter() - t0, 1)))
    return records


# --------------------------------------------------------------------------------------
# AIGVDBench (video)
# --------------------------------------------------------------------------------------


def acquire_aigvd_remote(
    out_dir: Path, n_real: int, n_fake: int, log: AcquisitionLog, checkpoint: Checkpoint,
    *, seed: int = 1337, split: str = "test", max_clip_bytes: int = 12_000_000,
    workers: int = 5,
) -> list[dict[str, Any]]:
    """Selective extraction of real and generated video from AIGVDBench.

    The fake class is stratified across generator archives so the sample is not a single
    generator's fingerprint. Real and generated clips share video IDs in this corpus (the
    generators are conditioned on the same source clips), so pairing is preserved where
    possible — that removes scene content as a confound between the classes.
    """
    import random

    rng = random.Random(seed)
    out_dir = Path(out_dir)
    records: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    total_fetched = 0

    # ---- fake: stratified across generators -------------------------------------------
    per_gen = max(1, n_fake // len(AIGVD_GENERATORS))
    picked_ids: list[str] = []
    for gen_name, rel in AIGVD_GENERATORS:
        if len([r for r in records if r["label_video"] == "fake"]) >= n_fake:
            break
        try:
            rz = RemoteZip(AIGVD_BASE + rel).open()
        except Exception as exc:
            log.add(Attempt(f"AIGVDBench/{gen_name}", "video", "range_extraction", "fail",
                            reason=f"{type(exc).__name__}: {str(exc)[:120]}"))
            continue
        try:
            infos = [i for i in rz.infolist()
                     if i.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))
                     and i.file_size <= max_clip_bytes]
            rng.shuffle(infos)
            pending, want, adopted = [], per_gen, 0
            for info in infos:
                # Adopted (already-on-disk) clips count toward the requested total. Counting
                # only newly-fetched ones lets a directory populated by a larger run silently
                # inflate a smaller run's sample — which is how Stage 1's video set grew from
                # 54 to 67 clips after Stage 2 had populated the same directory.
                if len(pending) + adopted >= want:
                    break
                key = f"aigvd/fake/{gen_name}/{info.filename}"
                dest = out_dir / "aigvdbench" / "fake" / gen_name / Path(info.filename).name
                base = {"source_dataset": "AIGVDBench", "file_path": str(dest),
                        "modality": "video", "label_video": "fake", "label_audio": "",
                        "label_av_sync": "", "split": split, "attack_id": gen_name,
                        "speaker": "", "download_method": "range_extraction"}
                if checkpoint.adopt(key, dest, base):
                    records.append(checkpoint.done[key])
                    picked_ids.append(Path(info.filename).name)
                    adopted += 1
                    continue
                pending.append(info.filename)

            def sink(member: str, data: bytes, _g=gen_name) -> bool:
                dest = out_dir / "aigvdbench" / "fake" / _g / Path(member).name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                rec = {"source_dataset": "AIGVDBench", "file_path": str(dest),
                       "modality": "video", "label_video": "fake", "label_audio": "",
                       "label_av_sync": "", "split": split, "attack_id": _g,
                       "speaker": "", "download_method": "range_extraction_parallel",
                       "bytes": len(data)}
                records.append(rec)
                checkpoint.mark(f"aigvd/fake/{_g}/{member}", rec)
                picked_ids.append(Path(member).name)
                return True

            if pending:
                parallel_extract(AIGVD_BASE + rel, pending, sink, workers=workers)
        finally:
            total_fetched += rz.stats.bytes_fetched
            rz.close()

    # ---- real: prefer the same video IDs so scene content is matched -------------------
    try:
        rz = RemoteZip(AIGVD_BASE + AIGVD_REAL).open()
    except Exception as exc:
        log.add(Attempt("AIGVDBench/Real", "video", "range_extraction", "fail",
                        reason=f"{type(exc).__name__}: {str(exc)[:150]}"))
        checkpoint.flush()
        return records
    try:
        by_name = {Path(i.filename).name: i for i in rz.infolist()
                   if i.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))}
        # picked_ids is appended from worker threads, so its order reflects completion
        # order, not the seeded selection. Sorting restores run-to-run determinism.
        ordered = [by_name[n] for n in sorted(set(picked_ids)) if n in by_name]
        rest = [i for n, i in by_name.items() if n not in set(picked_ids)]
        rng.shuffle(rest)
        ordered += rest
        pending, adopted_real = [], 0
        for info in ordered:
            if len(pending) + adopted_real >= n_real:
                break
            if info.file_size > max_clip_bytes:
                continue
            key = f"aigvd/real/{info.filename}"
            dest = out_dir / "aigvdbench" / "real" / Path(info.filename).name
            base = {"source_dataset": "AIGVDBench", "file_path": str(dest),
                    "modality": "video", "label_video": "real", "label_audio": "",
                    "label_av_sync": "", "split": split, "attack_id": "real",
                    "speaker": "", "download_method": "range_extraction"}
            if checkpoint.adopt(key, dest, base):
                records.append(checkpoint.done[key]); adopted_real += 1; continue
            pending.append(info.filename)

        def sink_real(member: str, data: bytes) -> bool:
            dest = out_dir / "aigvdbench" / "real" / Path(member).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            rec = {"source_dataset": "AIGVDBench", "file_path": str(dest),
                   "modality": "video", "label_video": "real", "label_audio": "",
                   "label_av_sync": "", "split": split, "attack_id": "real",
                   "speaker": "", "download_method": "range_extraction_parallel",
                   "bytes": len(data)}
            records.append(rec)
            checkpoint.mark(f"aigvd/real/{member}", rec)
            return True

        if pending:
            parallel_extract(AIGVD_BASE + AIGVD_REAL, pending, sink_real, workers=workers,
                             on_done=lambda d, t: print(f"    real: {d}/{t}", flush=True))
    finally:
        total_fetched += rz.stats.bytes_fetched
        rz.close()
        checkpoint.flush()

    n_real_got = len([r for r in records if r["label_video"] == "real"])
    n_fake_got = len([r for r in records if r["label_video"] == "fake"])
    log.add(Attempt(
        "AIGVDBench", "video", "range_extraction", "ok" if records else "fail",
        files=len(records), bytes_pulled=total_fetched,
        reason=(f"{n_real_got} real + {n_fake_got} generated clips, fake stratified across "
                f"{len(AIGVD_GENERATORS)} generator archives, real preferentially matched to "
                f"the same video IDs. Extracted from remote archives totalling >100 GB "
                f"without downloading them; {time.perf_counter()-t0:.0f}s."),
        citation=SOURCES["AIGVDBench"]["citation"],
        licence=SOURCES["AIGVDBench"]["licence"],
        seconds=round(time.perf_counter() - t0, 1)))
    return records


# --------------------------------------------------------------------------------------
# In-The-Wild (Stage 2 cross-dataset target)
# --------------------------------------------------------------------------------------


def acquire_in_the_wild(
    out_dir: Path, n_per_class: int, log: AcquisitionLog, checkpoint: Checkpoint,
    *, seed: int = 1337, local_archive: Path | None = None,
) -> list[dict[str, Any]]:
    """Real-world audio deepfakes for the C6 cross-dataset control.

    Reads a locally downloaded archive when available, otherwise falls back to remote
    range extraction. Nothing here is generated locally.
    """
    import csv
    import random
    import zipfile

    rng = random.Random(seed)
    out_dir = Path(out_dir)
    records: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    use_local = local_archive is not None and Path(local_archive).exists()
    try:
        if use_local:
            zf = zipfile.ZipFile(local_archive)
            reader: Callable[[str], bytes] = zf.read
            method = "local_archive"
            fetched = 0
        else:
            rz = RemoteZip(IN_THE_WILD_URL).open()
            reader = rz.read
            method = "range_extraction"
    except Exception as exc:
        log.add(Attempt("In-The-Wild", "audio", "range_extraction", "fail",
                        reason=f"{type(exc).__name__}: {str(exc)[:150]}"))
        return records

    try:
        meta = reader("release_in_the_wild/meta.csv").decode()
        rows = list(csv.DictReader(io.StringIO(meta)))
        bona = [r for r in rows if r["label"].strip().lower().startswith("bona")]
        spoof = [r for r in rows if r["label"].strip().lower() == "spoof"]
        rng.shuffle(bona); rng.shuffle(spoof)
        picks = [(r, "real") for r in bona[:n_per_class]] + \
                [(r, "fake") for r in spoof[:n_per_class]]
        for r, lab in picks:
            key = f"itw/{r['file']}"
            dest = out_dir / "in_the_wild" / lab / r["file"]
            base = {"source_dataset": "In-The-Wild", "file_path": str(dest),
                    "modality": "audio", "label_audio": lab, "label_video": "",
                    "label_av_sync": "", "split": "cross_dataset_eval",
                    "attack_id": "in_the_wild", "speaker": r.get("speaker", ""),
                    "download_method": method}
            if checkpoint.adopt(key, dest, base):
                records.append(checkpoint.done[key]); continue
            if log.over_budget():
                break
            try:
                data = reader(f"release_in_the_wild/{r['file']}")
            except Exception:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            rec = {"source_dataset": "In-The-Wild", "file_path": str(dest),
                   "modality": "audio", "label_audio": lab, "label_video": "",
                   "label_av_sync": "", "split": "cross_dataset_eval",
                   "attack_id": "in_the_wild", "speaker": r.get("speaker", ""),
                   "download_method": method, "bytes": len(data)}
            records.append(rec); checkpoint.mark(key, rec)
    finally:
        checkpoint.flush()
        if use_local:
            zf.close()
            fetched = sum(r.get("bytes", 0) for r in records)
        else:
            fetched = rz.stats.bytes_fetched
            rz.close()

    log.add(Attempt(
        "In-The-Wild", "audio", method, "ok" if records else "fail",
        files=len(records), bytes_pulled=fetched,
        reason=(f"real-world (YouTube/podcast sourced) audio deepfakes; the standard "
                f"cross-domain generalisation target against lab-recorded ASVspoof. "
                f"{time.perf_counter()-t0:.0f}s."),
        citation=SOURCES["In-The-Wild"]["citation"],
        licence=SOURCES["In-The-Wild"]["licence"],
        seconds=round(time.perf_counter() - t0, 1)))
    return records


# --------------------------------------------------------------------------------------
# Honest probes for sources that cannot be acquired
# --------------------------------------------------------------------------------------


def probe_gated_sources(log: AcquisitionLog, stage: int) -> None:
    """Record, rather than quietly omit, every source that could not be acquired."""
    # The acquisition path must remain usable in a minimal offline/reproducibility
    # environment. Hugging Face Hub is optional: absence of the client prevents probing
    # gated catalog entries, but must not invalidate already acquired open corpora.
    try:
        from huggingface_hub import HfApi
    except ImportError:
        log.add(Attempt(
            dataset="gated-source-probe", modality="mixed", method="probe",
            status="skip", reason="optional huggingface_hub client is not installed; "
            "no gated source was accessed or bypassed", citation="", licence=""))
        return

    api = HfApi()

    def hf_state(rid: str) -> str:
        try:
            info = api.dataset_info(rid)
            return f"reachable, gated={info.gated}"
        except Exception as exc:
            return f"unreachable: {type(exc).__name__}"

    lavdf = hf_state("ControlNet/LAV-DF")
    log.add(Attempt(
        "LAV-DF", "audio-visual", "probe", "skip",
        reason=(f"HF ControlNet/LAV-DF {lavdf}. The repo exposes only a single 25.6 GB "
                "full-corpus archive (LAV-DF.tar) covering all 136K videos, with no "
                "per-clip access, and requires accepting terms (gated=auto) which needs a "
                "user token this environment does not hold. Per the acquisition protocol, "
                "full-corpus archives are not downloaded. AV modality therefore has no "
                "usable source and is reported UNEVALUATED — not substituted.")))

    avd1m = hf_state("ControlNet/AV-Deepfake1M")
    log.add(Attempt(
        "AV-Deepfake1M", "audio-visual", "probe", "skip",
        reason=(f"HF ControlNet/AV-Deepfake1M {avd1m}. gated=manual: access requires a "
                "human-completed approval form, which an agent cannot and should not "
                "complete on the user's behalf. Not attempted.")))

    log.add(Attempt(
        "FakeAVCeleb", "audio-visual", "probe", "skip",
        reason=("Requires a manually signed Google Form request approved by the authors. "
                "Gated; not attempted, per protocol.")))

    for name in ("FaceForensics++", "Celeb-DF-v2", "DFDC"):
        log.add(Attempt(
            name, "video", "probe", "skip",
            reason=("No ungated programmatic source: distribution is behind an author-"
                    "approved EULA request form (FF++/Celeb-DF) or competition-rule "
                    "acceptance with credentials (DFDC). Not on the HF Hub under any "
                    "resolvable public id. Legacy comparison therefore not run.")))

    for name, note in (
        ("RobustSora", "no resolvable public Hugging Face dataset id; the de-watermarked "
                       "confound control could not be run"),
        ("Chameleon", "no resolvable public Hugging Face dataset id; the commercial-"
                      "generator split could not be run"),
        ("GenVidBench", "community mirrors exist (e.g. jian-0/GenVidBench) but the corpus "
                        "is distributed as multi-part .rar and split .7z volumes (single "
                        "members up to 49 GB) which do not support selective random access "
                        "the way ZIP does; extracting any subset requires downloading whole "
                        "volumes, exceeding budget. AIGVDBench (ZIP-based) is used for video "
                        "instead, and is a benchmark named in the same protocol."),
    ):
        log.add(Attempt(name, "video", "probe", "skip", reason=note))
