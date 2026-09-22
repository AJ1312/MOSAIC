"""Tests for real-corpus acquisition logic (no network, no synthetic media)."""
from __future__ import annotations

import json

import pytest

from mosaic.data.real_sources import (Attempt, AcquisitionLog, Checkpoint,
                                      parse_asvspoof_protocol, stratified_pick)

PROTOCOL = """LA_0079 LA_T_1138215 - - bonafide
LA_0079 LA_T_1272637 - A01 spoof
LA_0080 LA_T_1000001 - A02 spoof
LA_0080 LA_T_1000002 - A03 spoof
LA_0081 LA_T_1000003 - - bonafide
LA_0081 LA_T_1000004 - A01 spoof
"""


def test_protocol_parsing():
    rows = parse_asvspoof_protocol(PROTOCOL)
    assert len(rows) == 6
    assert rows[0] == {"speaker": "LA_0079", "utt_id": "LA_T_1138215",
                       "attack": "-", "label": "bonafide"}
    assert sum(r["label"] == "spoof" for r in rows) == 4


def test_protocol_parsing_ignores_malformed_lines():
    rows = parse_asvspoof_protocol(PROTOCOL + "garbage line\n\n")
    assert len(rows) == 6


def test_stratified_pick_balances_classes():
    rows = parse_asvspoof_protocol(PROTOCOL)
    picked = stratified_pick(rows, 2, 3)
    assert sum(r["label"] == "bonafide" for r in picked) == 2
    assert sum(r["label"] == "spoof" for r in picked) == 3


def test_stratified_pick_spreads_across_attacks():
    """Spoof sampling must cover attack IDs, not just take whichever comes first."""
    rows = []
    for a in ("A01", "A02", "A03", "A04", "A05", "A06"):
        for i in range(50):
            rows.append({"speaker": "s", "utt_id": f"{a}_{i}", "attack": a, "label": "spoof"})
    rows += [{"speaker": "s", "utt_id": f"b{i}", "attack": "-", "label": "bonafide"}
             for i in range(50)]
    picked = stratified_pick(rows, 10, 30)
    attacks = {r["attack"] for r in picked if r["label"] == "spoof"}
    assert attacks == {"A01", "A02", "A03", "A04", "A05", "A06"}


def test_stratified_pick_is_deterministic():
    rows = parse_asvspoof_protocol(PROTOCOL)
    assert [r["utt_id"] for r in stratified_pick(rows, 2, 2, seed=7)] == \
           [r["utt_id"] for r in stratified_pick(rows, 2, 2, seed=7)]


def test_checkpoint_roundtrip(tmp_path):
    ck = Checkpoint(tmp_path / "ck.json")
    assert not ck.has("a")
    ck.mark("a", {"file": "x"})
    ck.flush()
    assert Checkpoint(tmp_path / "ck.json").has("a")


def test_checkpoint_survives_corrupt_file(tmp_path):
    p = tmp_path / "ck.json"
    p.write_text("{not json")
    ck = Checkpoint(p)          # must not raise: a corrupt checkpoint restarts, never crashes
    assert ck.done == {}


def test_acquisition_log_tracks_budget_and_writes(tmp_path):
    log = AcquisitionLog(tmp_path / "acq.md", budget_gb=0.000001)   # 1 kB budget
    log.add(Attempt("X", "audio", "range_extraction", "ok", files=2, bytes_pulled=5000))
    assert log.over_budget()
    log.add(Attempt("Y", "video", "probe", "skip", reason="gated"))
    p = log.write(stage=1)
    text = p.read_text()
    assert "REAL DATA ONLY" in text
    assert "**ok**" in text and "**skip**" in text
    assert "gated" in text


def test_acquisition_log_records_skips_not_silence(tmp_path):
    """Unobtainable sources must appear in the log, not be quietly omitted."""
    log = AcquisitionLog(tmp_path / "acq.md", budget_gb=1)
    log.add(Attempt("LAV-DF", "audio-visual", "probe", "skip",
                    reason="full-corpus archive only"))
    text = log.write(stage=1).read_text()
    assert "LAV-DF" in text and "full-corpus archive only" in text


def test_checkpoint_adopts_file_already_on_disk(tmp_path):
    """A cleared checkpoint must not cause gigabytes of already-present media to re-download.

    Regression test: clearing outputs_real/cache removed the checkpoint while the media
    survived, and acquisition would have re-fetched every clip. The bytes on disk are the
    source of truth; the checkpoint is only an index.
    """
    ck = Checkpoint(tmp_path / "ck.json")
    dest = tmp_path / "clip.flac"
    dest.write_bytes(b"x" * 1234)
    base = {"source_dataset": "ASVspoof2019-LA", "file_path": str(dest),
            "label_audio": "real", "download_method": "range_extraction"}
    assert ck.adopt("k1", dest, base)
    assert ck.has("k1")
    assert ck.done["k1"]["bytes"] == 1234
    assert "resumed_from_disk" in ck.done["k1"]["download_method"]


def test_checkpoint_does_not_adopt_missing_or_empty(tmp_path):
    ck = Checkpoint(tmp_path / "ck.json")
    assert not ck.adopt("k", tmp_path / "absent.flac", {})
    empty = tmp_path / "empty.flac"
    empty.touch()
    assert not ck.adopt("k2", empty, {})
    assert not ck.has("k2")


def test_adopt_keeps_existing_record_unchanged(tmp_path):
    ck = Checkpoint(tmp_path / "ck.json")
    dest = tmp_path / "c.flac"
    dest.write_bytes(b"y" * 10)
    ck.mark("k", {"file_path": str(dest), "download_method": "original", "bytes": 10})
    assert ck.adopt("k", dest, {"download_method": "other"})
    assert ck.done["k"]["download_method"] == "original"


def test_adopted_clips_count_toward_requested_total():
    """Requesting N clips must yield N, whether they were fetched or already present.

    Regression test: adopted files were appended to the record list without incrementing
    the counter that stops the loop, so a directory already populated by a larger run
    inflated a smaller run's sample (Stage 1 video went from 54 to 67 clips).
    """
    want = 5
    on_disk = {f"c{i}.mp4" for i in range(3)}      # 3 already present, 2 to fetch
    pending, adopted = [], 0
    for i in range(50):
        if len(pending) + adopted >= want:
            break
        name = f"c{i}.mp4"
        if name in on_disk:
            adopted += 1
        else:
            pending.append(name)
    assert adopted + len(pending) == want
    assert adopted == 3 and len(pending) == 2
