"""Tests for L4: Merkle commitments, proofs, and the hash-chained ledger."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mosaic.l4_custody import (LocalLedger, LocalTimestampAuthority, MerkleTree,
                               combined_root, commit_audio, commit_video, verify_ledger,
                               verify_proof)


# --------------------------------------------------------------------------------------
# Merkle tree
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 9, 17, 64])
def test_every_leaf_has_a_valid_proof(n):
    leaves = [f"{i:064x}" for i in range(n)]
    tree = MerkleTree(leaves)
    for i in range(n):
        assert verify_proof(tree.proof(i)), f"proof failed for leaf {i} of {n}"


def test_proof_rejects_altered_leaf():
    tree = MerkleTree([f"{i:064x}" for i in range(9)])
    proof = tree.proof(4).to_dict()
    proof["leaf_hash"] = "ff" + proof["leaf_hash"][2:]
    assert not verify_proof(proof)


def test_proof_rejects_altered_sibling():
    tree = MerkleTree([f"{i:064x}" for i in range(9)])
    proof = tree.proof(4).to_dict()
    proof["path"][0]["hash"] = "ab" + proof["path"][0]["hash"][2:]
    assert not verify_proof(proof)


def test_proof_rejects_flipped_sibling_side():
    tree = MerkleTree([f"{i:064x}" for i in range(8)])
    proof = tree.proof(3).to_dict()
    proof["path"][0]["side"] = "left" if proof["path"][0]["side"] == "right" else "right"
    assert not verify_proof(proof)


def test_leaf_and_node_domains_are_separated():
    """A leaf must not be reinterpretable as an internal node."""
    from mosaic.l4_custody import _leaf_hash, _node_hash

    h = f"{7:064x}"
    assert _leaf_hash(bytes.fromhex(h) + bytes.fromhex(h)) != _node_hash(h, h)


def test_empty_tree_rejected():
    with pytest.raises(ValueError):
        MerkleTree([])


# --------------------------------------------------------------------------------------
# Track commitments
# --------------------------------------------------------------------------------------


def test_commitments_are_deterministic(rng):
    frames = rng.integers(0, 255, (50, 16, 16, 3), dtype=np.uint8)
    wave = rng.normal(0, 0.1, 16000).astype(np.float32)
    a = commit_video(frames, 25.0, 1.0)
    b = commit_video(frames, 25.0, 1.0)
    assert a.root == b.root
    assert commit_audio(wave, 16000, 1.0).root == commit_audio(wave, 16000, 1.0).root


def test_audio_tamper_localises_to_one_chunk_and_leaves_video_intact(rng):
    frames = rng.integers(0, 255, (75, 16, 16, 3), dtype=np.uint8)
    wave = rng.normal(0, 0.1, 3 * 16000).astype(np.float32)

    v0 = commit_video(frames, 25.0, 1.0)
    a0 = commit_audio(wave, 16000, 1.0)

    tampered = wave.copy()
    tampered[16000:32000] = tampered[16000:32000][::-1]  # rewrite second 1
    a1 = commit_audio(tampered, 16000, 1.0)
    v1 = commit_video(frames, 25.0, 1.0)

    changed = [i for i, (x, y) in enumerate(zip(a0.leaf_hashes, a1.leaf_hashes)) if x != y]
    assert changed == [1], "audio tampering must localise to exactly the affected chunk"
    assert a1.root != a0.root
    assert v1.root == v0.root, "video root must be unaffected by audio tampering"
    assert combined_root(v1.root, a1.root, "ctx") != combined_root(v0.root, a0.root, "ctx")


def test_video_tamper_localises(rng):
    frames = rng.integers(0, 255, (75, 16, 16, 3), dtype=np.uint8)
    v0 = commit_video(frames, 25.0, 1.0)
    tampered = frames.copy()
    tampered[60, 0, 0, 0] ^= 0xFF          # one pixel in chunk 2
    v1 = commit_video(tampered, 25.0, 1.0)
    changed = [i for i, (x, y) in enumerate(zip(v0.leaf_hashes, v1.leaf_hashes)) if x != y]
    assert changed == [2]


def test_missing_audio_is_recorded_not_faked():
    c = commit_audio(np.zeros(0, dtype=np.float32), 16000, 1.0)
    assert c.n_chunks == 0
    assert "no audio track" in c.note
    assert c.root  # a sentinel digest, not an empty string


def test_combined_root_binds_all_three_inputs():
    base = combined_root("a" * 64, "b" * 64, "ctx")
    assert base != combined_root("c" * 64, "b" * 64, "ctx")
    assert base != combined_root("a" * 64, "c" * 64, "ctx")
    assert base != combined_root("a" * 64, "b" * 64, "other")


# --------------------------------------------------------------------------------------
# Ledger
# --------------------------------------------------------------------------------------


def test_ledger_chain_verifies(tmp_path):
    ledger = LocalLedger(tmp_path / "l.jsonl")
    for i in range(5):
        ledger.append({"record": i})
    state = verify_ledger(tmp_path / "l.jsonl")
    assert state["valid"] and state["n_entries"] == 5


def test_ledger_detects_retroactive_edit(tmp_path):
    path = tmp_path / "l.jsonl"
    ledger = LocalLedger(path)
    for i in range(5):
        ledger.append({"verdict": f"v{i}"})

    lines = path.read_text().strip().split("\n")
    entry = json.loads(lines[2])
    entry["payload"]["verdict"] = "tampered"
    lines[2] = json.dumps(entry, sort_keys=True)
    path.write_text("\n".join(lines) + "\n")

    state = verify_ledger(path)
    assert not state["valid"]
    assert state["broken_at"] == 2


def test_ledger_detects_deleted_entry(tmp_path):
    path = tmp_path / "l.jsonl"
    ledger = LocalLedger(path)
    for i in range(5):
        ledger.append({"record": i})
    lines = path.read_text().strip().split("\n")
    del lines[2]
    path.write_text("\n".join(lines) + "\n")
    assert not verify_ledger(path)["valid"]


def test_ledger_head_advances(tmp_path):
    ledger = LocalLedger(tmp_path / "l.jsonl")
    assert ledger.head() == LocalLedger.GENESIS
    e1 = ledger.append({"a": 1})
    assert ledger.head() == e1["entry_hash"]
    e2 = ledger.append({"a": 2})
    assert e2["prev_hash"] == e1["entry_hash"]


# --------------------------------------------------------------------------------------
# Timestamp honesty
# --------------------------------------------------------------------------------------


def test_local_timestamp_declares_itself_non_qualified():
    stamp = LocalTimestampAuthority().stamp()
    assert stamp.is_qualified is False
    assert "NOT" in stamp.detail
    assert stamp.authority == "local_nonqualified"
