"""L1 — Hash and similarity triage.

L1 exists to make the rest of the pipeline affordable. A viral clip is re-uploaded,
re-cropped and re-compressed thousands of times; running the full cascade on every copy is
wasted compute. So hashing is used here as a **compute-gating decision**, not as a
detection method or a similarity score that means anything on its own.

The two hash types answer different questions and are never conflated:

**SHA-256 — exact identity.** A match means the canonical bytes are identical to something
already adjudicated, so the previous verdict applies to *this exact artefact*. Even then
the cached verdict is only reused when it was produced by the same feature schema and
configuration digest; otherwise the clip is re-analysed, because a verdict from a
different model version is not a verdict about this pipeline.

**Perceptual hash — candidate retrieval only.** A near match means "here is a record worth
looking at", never "this is the same content". By default a perceptual hit does *not*
short-circuit anything: it is attached to the custody record as a lineage pointer and the
full cascade runs anyway. Allowing perceptual short-circuiting is available behind
``L1Config.allow_perceptual_shortcircuit`` and is benchmarked as a risk, because a false
short-circuit means a genuinely different video silently inherits someone else's verdict —
the most safety-critical failure this layer can produce.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import L1Config
from .hashing import hashes_from_hex, hashes_to_hex, sequence_distance


@dataclass
class RegistryEntry:
    canonical_sha256: str
    source_sha256: str
    clip_id: str
    verdict: str
    verdict_payload: dict[str, Any]
    video_phash: list[int]
    audio_phash: list[int]
    feature_schema: str
    config_digest: str
    custody_record_id: str | None = None
    created_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "canonical_sha256": self.canonical_sha256,
            "source_sha256": self.source_sha256,
            "clip_id": self.clip_id,
            "verdict": self.verdict,
            "feature_schema": self.feature_schema,
            "config_digest": self.config_digest,
            "custody_record_id": self.custody_record_id,
            "created_utc": self.created_utc,
        }


@dataclass
class TriageResult:
    """What L1 decided, and why."""

    action: str                       # "exact_hit" | "perceptual_candidate" | "miss"
    short_circuit: bool               # whether the expensive cascade may be skipped
    reason: str
    exact_match: RegistryEntry | None = None
    candidates: list[dict[str, Any]] = field(default_factory=list)
    lookup_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "short_circuit": self.short_circuit,
            "reason": self.reason,
            "exact_match": self.exact_match.to_dict() if self.exact_match else None,
            "candidates": self.candidates,
            "lookup_ms": round(self.lookup_ms, 3),
        }


_SCHEMA = """
CREATE TABLE IF NOT EXISTS registry (
    canonical_sha256 TEXT PRIMARY KEY,
    source_sha256    TEXT NOT NULL,
    clip_id          TEXT NOT NULL,
    verdict          TEXT NOT NULL,
    verdict_payload  TEXT NOT NULL,
    video_phash      TEXT NOT NULL,
    audio_phash      TEXT NOT NULL,
    feature_schema   TEXT NOT NULL,
    config_digest    TEXT NOT NULL,
    custody_record_id TEXT,
    created_utc      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_source ON registry(source_sha256);
CREATE INDEX IF NOT EXISTS idx_schema ON registry(feature_schema);
"""


class HashRegistry:
    """SQLite-backed registry of adjudicated clips."""

    def __init__(self, path: str | Path, config: L1Config | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.config = config or L1Config()
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "HashRegistry":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- writing ------------------------------------------------------------------------

    def put(self, entry: RegistryEntry) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO registry VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                entry.canonical_sha256, entry.source_sha256, entry.clip_id, entry.verdict,
                json.dumps(entry.verdict_payload), hashes_to_hex(entry.video_phash),
                hashes_to_hex(entry.audio_phash), entry.feature_schema, entry.config_digest,
                entry.custody_record_id,
                entry.created_utc or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            ),
        )
        self.conn.commit()

    # -- reading ------------------------------------------------------------------------

    def _row_to_entry(self, row: sqlite3.Row) -> RegistryEntry:
        return RegistryEntry(
            canonical_sha256=row["canonical_sha256"],
            source_sha256=row["source_sha256"],
            clip_id=row["clip_id"],
            verdict=row["verdict"],
            verdict_payload=json.loads(row["verdict_payload"]),
            video_phash=hashes_from_hex(row["video_phash"]),
            audio_phash=hashes_from_hex(row["audio_phash"]),
            feature_schema=row["feature_schema"],
            config_digest=row["config_digest"],
            custody_record_id=row["custody_record_id"],
            created_utc=row["created_utc"],
        )

    def get_exact(self, canonical_sha256: str) -> RegistryEntry | None:
        cur = self.conn.execute(
            "SELECT * FROM registry WHERE canonical_sha256 = ?", (canonical_sha256,)
        )
        row = cur.fetchone()
        return self._row_to_entry(row) if row else None

    def size(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM registry").fetchone()[0])

    def find_candidates(self, video_phash: list[int], audio_phash: list[int],
                        threshold: float | None = None, limit: int = 5) -> list[dict[str, Any]]:
        """Linear scan for perceptually similar entries.

        A linear scan is honest about what this prototype is: at registry sizes beyond a
        few tens of thousands this needs an LSH or inverted index, which is noted as a
        scaling limitation rather than hidden behind a benchmark run at small N.
        """
        threshold = self.config.perceptual_candidate_threshold if threshold is None else threshold
        out: list[dict[str, Any]] = []
        for row in self.conn.execute("SELECT * FROM registry"):
            entry = self._row_to_entry(row)
            d_v = sequence_distance(video_phash, entry.video_phash)
            d_a = (sequence_distance(audio_phash, entry.audio_phash)
                   if audio_phash and entry.audio_phash else 1.0)
            # Video distance gates retrieval; audio is reported alongside it because a
            # re-upload with substituted audio is precisely the case worth surfacing.
            if d_v <= threshold:
                out.append({
                    "canonical_sha256": entry.canonical_sha256,
                    "clip_id": entry.clip_id,
                    "verdict": entry.verdict,
                    "video_phash_distance": round(d_v, 5),
                    "audio_phash_distance": round(d_a, 5),
                    "feature_schema": entry.feature_schema,
                    "config_digest": entry.config_digest,
                    "custody_record_id": entry.custody_record_id,
                })
        out.sort(key=lambda c: c["video_phash_distance"])
        return out[:limit]


# --------------------------------------------------------------------------------------
# Triage
# --------------------------------------------------------------------------------------


def triage(
    registry: HashRegistry,
    canonical_sha256: str,
    video_phash: list[int],
    audio_phash: list[int],
    *,
    feature_schema: str,
    config_digest: str,
    config: L1Config | None = None,
) -> TriageResult:
    """Decide whether the expensive cascade can be skipped for this clip."""
    config = config or registry.config
    t0 = time.perf_counter()

    exact = registry.get_exact(canonical_sha256)
    if exact is not None:
        stale = (exact.feature_schema != feature_schema or exact.config_digest != config_digest)
        if stale:
            result = TriageResult(
                action="exact_hit",
                short_circuit=False,
                reason=(
                    "byte-identical canonical form found, but it was adjudicated under a "
                    f"different configuration (schema {exact.feature_schema} vs {feature_schema}, "
                    f"config {exact.config_digest[:12]} vs {config_digest[:12]}). A verdict from "
                    "another model version is not a verdict from this one, so the cascade re-runs."
                ),
                exact_match=exact,
            )
        else:
            result = TriageResult(
                action="exact_hit",
                short_circuit=True,
                reason=(
                    "canonical bytes are identical (SHA-256) to an entry adjudicated under "
                    "this exact feature schema and configuration; the cached verdict applies "
                    "to this artefact by identity, not by similarity"
                ),
                exact_match=exact,
            )
        result.lookup_ms = (time.perf_counter() - t0) * 1000
        return result

    candidates = registry.find_candidates(video_phash, audio_phash)
    if candidates:
        best = candidates[0]
        allow = config.allow_perceptual_shortcircuit
        tight = best["video_phash_distance"] <= config.perceptual_shortcircuit_threshold
        same_cfg = (best["feature_schema"] == feature_schema
                    and best["config_digest"] == config_digest)
        if allow and tight and same_cfg:
            reason = (
                f"perceptual distance {best['video_phash_distance']:.4f} is below the "
                f"short-circuit threshold {config.perceptual_shortcircuit_threshold} and "
                "perceptual short-circuiting is explicitly enabled. NOTE: this is a "
                "similarity inference, not proof of identity; false short-circuits are "
                "possible and are measured in the efficiency benchmark."
            )
            short = True
        else:
            reason = (
                f"{len(candidates)} perceptually similar entr"
                f"{'y' if len(candidates) == 1 else 'ies'} found (best distance "
                f"{best['video_phash_distance']:.4f}). Recorded as lineage candidates only — "
                "perceptual similarity is retrieval, not proof, so the full cascade runs."
            )
            short = False
        result = TriageResult(action="perceptual_candidate", short_circuit=short,
                              reason=reason, candidates=candidates)
        result.lookup_ms = (time.perf_counter() - t0) * 1000
        return result

    result = TriageResult(
        action="miss", short_circuit=False,
        reason="no exact or perceptual match in the registry; full cascade required",
    )
    result.lookup_ms = (time.perf_counter() - t0) * 1000
    return result
