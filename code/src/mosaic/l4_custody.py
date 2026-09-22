"""L4 — Immutable chain of custody.

Custody begins at **first verifier ingestion**, not at generation. That is the whole point:
watermarking and C2PA require cooperation from whoever produced the content, so they cannot
cover legacy footage, screen recordings, or anything an adversary has deliberately stripped.
A verifier-side custody record applies to all of it, retroactively.

Structure
---------
The clip is chunked on a fixed time grid (aligned to the canonical GOP, so video chunk
boundaries fall on keyframes). Video and audio are committed **separately** — a Merkle tree
per track — and the two roots are then bound together with the analysis context into a
single combined root::

    combined_root = SHA-256( video_root || audio_root || context_hash )

Separate track trees are what make the multimodal extension meaningful for evidence: a
proof can demonstrate that *this second of audio* is intact without disclosing the video,
and a substituted soundtrack invalidates the audio root while leaving the video root
untouched, which localises the tampering to a modality and a time range. Committing an
interleaved container digest instead would only be able to say "something changed".

The ledger is an append-only JSONL file in which each entry carries the hash of the previous
entry, so any retroactive edit breaks the chain from that point forward and
:func:`verify_ledger` reports the exact index where it breaks.

Honest limits
-------------
* ``LocalLedger`` is a local append-only log, not a distributed or multi-operator anchor.
  The interface is deliberately shaped so a real anchor can be substituted; no distributed
  ledger is contacted, and none is claimed.
* ``LocalTimestampAuthority`` is **not** a qualified RFC-3161 timestamp authority. It
  records the local clock and labels itself non-qualified in every record it touches. It
  provides ordering evidence, not trusted time.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .hashing import sha256_bytes

# --------------------------------------------------------------------------------------
# Merkle tree
# --------------------------------------------------------------------------------------

#: Domain-separation prefixes prevent a leaf from being reinterpreted as an internal node
#: (the classic second-preimage attack on naive Merkle constructions).
_LEAF_PREFIX = b"\x00"
_NODE_PREFIX = b"\x01"


def _leaf_hash(data: bytes) -> str:
    return hashlib.sha256(_LEAF_PREFIX + data).hexdigest()


def _node_hash(left: str, right: str) -> str:
    return hashlib.sha256(_NODE_PREFIX + bytes.fromhex(left) + bytes.fromhex(right)).hexdigest()


@dataclass
class MerkleProof:
    leaf_index: int
    leaf_hash: str
    path: list[dict[str, str]]     # [{"hash": ..., "side": "left"|"right"}, ...]
    root: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MerkleTree:
    """Binary Merkle tree over an ordered list of leaf hashes."""

    def __init__(self, leaves: list[str]):
        if not leaves:
            raise ValueError("MerkleTree requires at least one leaf")
        self.leaves = list(leaves)
        self.levels: list[list[str]] = [list(leaves)]
        while len(self.levels[-1]) > 1:
            prev = self.levels[-1]
            nxt: list[str] = []
            for i in range(0, len(prev), 2):
                left = prev[i]
                # Odd node is promoted by duplication, the standard construction.
                right = prev[i + 1] if i + 1 < len(prev) else prev[i]
                nxt.append(_node_hash(left, right))
            self.levels.append(nxt)

    @property
    def root(self) -> str:
        return self.levels[-1][0]

    def proof(self, index: int) -> MerkleProof:
        if not 0 <= index < len(self.leaves):
            raise IndexError(f"leaf index {index} out of range (n={len(self.leaves)})")
        path: list[dict[str, str]] = []
        idx = index
        for level in self.levels[:-1]:
            sibling = idx ^ 1
            if sibling >= len(level):
                sibling = idx      # duplicated odd node
            path.append({
                "hash": level[sibling],
                "side": "right" if sibling > idx else "left",
            })
            idx //= 2
        return MerkleProof(leaf_index=index, leaf_hash=self.leaves[index], path=path,
                           root=self.root)


def verify_proof(proof: MerkleProof | dict[str, Any]) -> bool:
    """Recompute a Merkle root from a leaf and its path."""
    if isinstance(proof, dict):
        proof = MerkleProof(
            leaf_index=proof["leaf_index"], leaf_hash=proof["leaf_hash"],
            path=proof["path"], root=proof["root"],
        )
    current = proof.leaf_hash
    for step in proof.path:
        if step["side"] == "left":
            current = _node_hash(step["hash"], current)
        else:
            current = _node_hash(current, step["hash"])
    return current == proof.root


# --------------------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------------------


@dataclass
class TrackCommitment:
    track: str                     # "video" | "audio"
    n_chunks: int
    chunk_seconds: float
    root: str
    leaf_hashes: list[str]
    chunk_spans: list[tuple[float, float]]
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "track": self.track,
            "n_chunks": self.n_chunks,
            "chunk_seconds": self.chunk_seconds,
            "root": self.root,
            "leaf_hashes": self.leaf_hashes,
            "chunk_spans": [[round(a, 4), round(b, 4)] for a, b in self.chunk_spans],
            "note": self.note,
        }


def commit_video(frames: np.ndarray, fps: float, chunk_seconds: float) -> TrackCommitment:
    """Merkle-commit the decoded video track in fixed-duration chunks."""
    n_frames = int(frames.shape[0])
    if n_frames == 0 or fps <= 0:
        raise ValueError("cannot commit an empty video track")
    per_chunk = max(1, int(round(chunk_seconds * fps)))
    leaves: list[str] = []
    spans: list[tuple[float, float]] = []
    for start in range(0, n_frames, per_chunk):
        end = min(n_frames, start + per_chunk)
        block = np.ascontiguousarray(frames[start:end])
        # Shape and dtype are hashed with the pixels so a reshaped buffer cannot collide.
        header = f"video|{block.dtype}|{block.shape}|{start}|{end}".encode()
        leaves.append(_leaf_hash(header + block.tobytes()))
        spans.append((start / fps, end / fps))
    tree = MerkleTree(leaves)
    return TrackCommitment(
        track="video", n_chunks=len(leaves), chunk_seconds=chunk_seconds, root=tree.root,
        leaf_hashes=leaves, chunk_spans=spans,
        note="committed over decoded canonical RGB frames, so the commitment is invariant "
             "to container remuxing but not to re-encoding",
    )


def commit_audio(wave: np.ndarray, sample_rate: int, chunk_seconds: float) -> TrackCommitment:
    """Merkle-commit the decoded audio track in fixed-duration chunks."""
    if wave is None or wave.size == 0 or sample_rate <= 0:
        return TrackCommitment(
            track="audio", n_chunks=0, chunk_seconds=chunk_seconds,
            root=sha256_bytes(b"no-audio-track"), leaf_hashes=[], chunk_spans=[],
            note="no audio track present; the root is a fixed sentinel digest and no audio "
                 "chunk proofs can be produced",
        )
    per_chunk = max(1, int(round(chunk_seconds * sample_rate)))
    leaves: list[str] = []
    spans: list[tuple[float, float]] = []
    for start in range(0, wave.size, per_chunk):
        end = min(wave.size, start + per_chunk)
        block = np.ascontiguousarray(wave[start:end])
        header = f"audio|{block.dtype}|{block.shape}|{start}|{end}|{sample_rate}".encode()
        leaves.append(_leaf_hash(header + block.tobytes()))
        spans.append((start / sample_rate, end / sample_rate))
    tree = MerkleTree(leaves)
    return TrackCommitment(
        track="audio", n_chunks=len(leaves), chunk_seconds=chunk_seconds, root=tree.root,
        leaf_hashes=leaves, chunk_spans=spans,
        note="committed over decoded canonical PCM samples",
    )


def combined_root(video_root: str, audio_root: str, context_hash: str) -> str:
    """Bind both track roots and the analysis context into one commitment."""
    return hashlib.sha256(
        b"mosaic-av-combined-root|" + video_root.encode() + b"|" + audio_root.encode()
        + b"|" + context_hash.encode()
    ).hexdigest()


# --------------------------------------------------------------------------------------
# Timestamping
# --------------------------------------------------------------------------------------


@dataclass
class Timestamp:
    utc_iso: str
    monotonic_ns: int
    authority: str
    is_qualified: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LocalTimestampAuthority:
    """Local, explicitly NON-QUALIFIED timestamp source."""

    name = "local_nonqualified"

    def stamp(self) -> Timestamp:
        return Timestamp(
            utc_iso=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            monotonic_ns=time.monotonic_ns(),
            authority=self.name,
            is_qualified=False,
            detail=("Local system clock. This is NOT an RFC-3161 qualified timestamp and NOT "
                    "an eIDAS qualified electronic time stamp. It establishes ordering within "
                    "this ledger only, and carries no third-party attestation of wall-clock "
                    "time. A qualified TSA must be substituted for evidentiary use."),
        )


# --------------------------------------------------------------------------------------
# Ledger
# --------------------------------------------------------------------------------------


class LocalLedger:
    """Append-only, hash-chained JSONL ledger.

    Each entry stores the digest of the previous entry, so the file is tamper-evident:
    editing entry *k* invalidates every entry from *k* onward, and verification reports
    the first index that fails.
    """

    GENESIS = "0" * 64

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def _entries(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def head(self) -> str:
        entries = self._entries()
        return entries[-1]["entry_hash"] if entries else self.GENESIS

    @staticmethod
    def _digest(payload: dict[str, Any], prev_hash: str, index: int) -> str:
        body = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(f"{index}|{prev_hash}|{body}".encode()).hexdigest()

    def append(self, payload: dict[str, Any]) -> dict[str, Any]:
        entries = self._entries()
        index = len(entries)
        prev_hash = entries[-1]["entry_hash"] if entries else self.GENESIS
        entry = {
            "index": index,
            "prev_hash": prev_hash,
            "payload": payload,
            "entry_hash": self._digest(payload, prev_hash, index),
        }
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
        return entry

    def verify(self) -> dict[str, Any]:
        entries = self._entries()
        prev = self.GENESIS
        for i, entry in enumerate(entries):
            if entry.get("index") != i:
                return {"valid": False, "n_entries": len(entries), "broken_at": i,
                        "reason": f"entry index {entry.get('index')} does not match position {i}"}
            if entry.get("prev_hash") != prev:
                return {"valid": False, "n_entries": len(entries), "broken_at": i,
                        "reason": "prev_hash does not match the preceding entry's hash"}
            expected = self._digest(entry["payload"], prev, i)
            if expected != entry.get("entry_hash"):
                return {"valid": False, "n_entries": len(entries), "broken_at": i,
                        "reason": "entry_hash does not match a recomputation of the payload"}
            prev = entry["entry_hash"]
        return {"valid": True, "n_entries": len(entries), "broken_at": None,
                "head": prev, "reason": "hash chain is intact"}


def verify_ledger(path: str | Path) -> dict[str, Any]:
    return LocalLedger(path).verify()


# --------------------------------------------------------------------------------------
# Custody record
# --------------------------------------------------------------------------------------


@dataclass
class CustodyRecord:
    """The complete, exportable record for one adjudicated artefact.

    Field-for-field mapping to the evidentiary standards MOSAIC targets:

    ================================  ==================================================
    requirement                        field(s)
    ================================  ==================================================
    ISO/IEC 27037 identification       ``record_id``, ``source_path``, ``source_sha256``
    ISO/IEC 27037 collection           ``ingest`` (ffmpeg command, binary version)
    ISO/IEC 27037 acquisition          ``canonical_sha256``, ``track_commitments``
    ISO/IEC 27037 preservation         ``combined_root``, ``ledger_entry``
    ISO/IEC 27037 auditability         ``events`` (ordered, timestamped)
    ISO/IEC 27037 repeatability        ``config_digest``, ``seed``, ``environment``
    ISO/IEC 27037 reproducibility      ``models``, ``feature_schema``
    NIST SP 800-101 hashing            SHA-256 throughout; re-verified at each transfer
    NIST SP 800-86 chain of custody    ``events`` + hash-chained ledger
    FRE 901(b)(9) process/system       ``pipeline_description``, ``models``, ``environment``
    eIDAS 2.0 qualified timestamp      ``timestamp`` — NOT satisfied; see ``is_qualified``
    ================================  ==================================================
    """

    record_id: str
    schema_version: str
    clip_id: str
    source_path: str
    source_sha256: str
    canonical_sha256: str
    feature_schema: str
    config_digest: str
    seed: int
    device_profile: dict[str, Any]
    environment: dict[str, Any]
    pipeline_description: str
    ingest: dict[str, Any]
    l1_triage: dict[str, Any]
    l2_provenance: dict[str, Any]
    l3_detection: dict[str, Any]
    verdict: dict[str, Any]
    evidence: list[dict[str, Any]]
    models: dict[str, Any]
    track_commitments: dict[str, Any]
    combined_root: str
    timestamp: dict[str, Any]
    events: list[dict[str, Any]]
    compute: dict[str, Any]
    ledger_entry: dict[str, Any] | None = None
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.record_id}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return path


def environment_snapshot() -> dict[str, Any]:
    """Everything needed to attribute a result to a software environment."""
    import numpy
    import scipy
    import sklearn

    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }
    for mod in ("torch", "av", "soundfile", "c2pa"):
        try:
            m = __import__(mod)
            env[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            env[mod] = "not installed"
    return env


class CustodyBuilder:
    """Accumulates timestamped events during a pipeline run, then seals the record."""

    def __init__(self, clip_id: str, ledger: LocalLedger,
                 tsa: LocalTimestampAuthority | None = None):
        self.clip_id = clip_id
        self.ledger = ledger
        self.tsa = tsa or LocalTimestampAuthority()
        self.events: list[dict[str, Any]] = []
        self._t0 = time.perf_counter()

    def event(self, layer: str, action: str, detail: Any = None) -> None:
        """Record one ordered, timestamped step. Called at every layer boundary."""
        self.events.append({
            "seq": len(self.events),
            "layer": layer,
            "action": action,
            "utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "elapsed_s": round(time.perf_counter() - self._t0, 6),
            "detail": detail,
        })

    def seal(
        self,
        *,
        record_id: str,
        schema_version: str,
        source_path: str,
        source_sha256: str,
        canonical_sha256: str,
        feature_schema: str,
        config_digest: str,
        seed: int,
        device_profile: dict[str, Any],
        pipeline_description: str,
        ingest: dict[str, Any],
        l1_triage: dict[str, Any],
        l2_provenance: dict[str, Any],
        l3_detection: dict[str, Any],
        verdict: dict[str, Any],
        evidence: list[dict[str, Any]],
        models: dict[str, Any],
        video_commitment: TrackCommitment,
        audio_commitment: TrackCommitment,
        compute: dict[str, Any],
        caveats: list[str],
        records_dir: str | Path,
    ) -> CustodyRecord:
        context_hash = sha256_bytes(json.dumps(
            {"config": config_digest, "schema": feature_schema, "models": models,
             "verdict": verdict.get("verdict")},
            sort_keys=True, default=str,
        ).encode())
        root = combined_root(video_commitment.root, audio_commitment.root, context_hash)
        stamp = self.tsa.stamp()
        self.event("L4", "sealed", {"combined_root": root})

        record = CustodyRecord(
            record_id=record_id,
            schema_version=schema_version,
            clip_id=self.clip_id,
            source_path=source_path,
            source_sha256=source_sha256,
            canonical_sha256=canonical_sha256,
            feature_schema=feature_schema,
            config_digest=config_digest,
            seed=seed,
            device_profile=device_profile,
            environment=environment_snapshot(),
            pipeline_description=pipeline_description,
            ingest=ingest,
            l1_triage=l1_triage,
            l2_provenance=l2_provenance,
            l3_detection=l3_detection,
            verdict=verdict,
            evidence=evidence,
            models=models,
            track_commitments={
                "video": video_commitment.to_dict(),
                "audio": audio_commitment.to_dict(),
                "context_hash": context_hash,
            },
            combined_root=root,
            timestamp=stamp.to_dict(),
            events=self.events,
            compute=compute,
            caveats=caveats,
        )
        entry = self.ledger.append({
            "record_id": record_id,
            "clip_id": self.clip_id,
            "canonical_sha256": canonical_sha256,
            "combined_root": root,
            "verdict": verdict.get("verdict"),
            "timestamp": stamp.to_dict(),
        })
        record.ledger_entry = entry
        record.save(records_dir)
        return record
