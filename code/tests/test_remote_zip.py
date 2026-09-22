"""Tests for range-request ZIP extraction, against a local HTTP server.

Deliberately not network-dependent: a local server that genuinely honours (and can be made
to refuse) Range requests exercises the same code path as Edinburgh DataShare or Hugging
Face, without making the suite depend on either being up.
"""
from __future__ import annotations

import io
import threading
import zipfile
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler

import pytest

from mosaic.data.remote_zip import HTTPRangeFile, RemoteReadError, RemoteZip, parallel_extract


class RangeHandler(SimpleHTTPRequestHandler):
    """Serves one in-memory blob, honouring Range unless told not to."""

    payload = b""
    allow_range = True

    def log_message(self, *a):  # keep test output clean
        pass

    def do_GET(self):
        rng = self.headers.get("Range")
        if rng and self.allow_range:
            start, _, end = rng.replace("bytes=", "").partition("-")
            s = int(start)
            e = int(end) if end else len(self.payload) - 1
            e = min(e, len(self.payload) - 1)
            body = self.payload[s:e + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {s}-{e}/{len(self.payload)}")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(200)
            self.send_header("Content-Length", str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)


@pytest.fixture(scope="module")
def zip_server():
    # Members are incompressible and the archive totals ~2.4 MB, so it is many blocks
    # long. A toy archive smaller than one block cannot exercise selective extraction at
    # all — a single fetch would pull the whole thing, which is the opposite of the
    # behaviour under test.
    import random

    rnd = random.Random(20260819)
    members = {f"dir/file_{i:03d}.bin": bytes(rnd.getrandbits(8) for _ in range(40_000))
               for i in range(60)}
    members["dir/meta.txt"] = b"label,value\nreal,1\nfake,0\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    RangeHandler.payload = buf.getvalue()
    RangeHandler.allow_range = True

    srv = HTTPServer(("127.0.0.1", 0), RangeHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    url = f"http://127.0.0.1:{srv.server_port}/archive.zip"
    yield url, members
    srv.shutdown()


def test_range_file_reports_true_size(zip_server):
    url, _ = zip_server
    f = HTTPRangeFile(url)
    assert f.size == len(RangeHandler.payload)


def test_range_file_seek_and_read(zip_server):
    url, _ = zip_server
    f = HTTPRangeFile(url)
    f.seek(10)
    assert f.read(20) == RangeHandler.payload[10:30]
    f.seek(-16, 2)
    assert f.read() == RangeHandler.payload[-16:]


def test_range_file_large_read_bypasses_cache(zip_server):
    url, _ = zip_server
    f = HTTPRangeFile(url, block_size=1024)
    f.seek(0)
    data = f.read(len(RangeHandler.payload))
    assert data == RangeHandler.payload


def test_server_without_range_support_is_rejected(zip_server):
    """A server that ignores Range must fail loudly, not silently return wrong bytes."""
    url, _ = zip_server
    RangeHandler.allow_range = False
    try:
        with pytest.raises(RemoteReadError):
            HTTPRangeFile(url)
    finally:
        RangeHandler.allow_range = True


def test_remote_zip_lists_and_extracts(zip_server):
    url, members = zip_server
    with RemoteZip(url) as rz:
        names = set(rz.namelist())
        assert names == set(members)
        for name in ("dir/file_000.bin", "dir/file_017.bin", "dir/meta.txt"):
            assert rz.read(name) == members[name]


def test_remote_zip_transfers_far_less_than_the_archive(zip_server):
    """The whole point: pull a few members without downloading the archive."""
    url, members = zip_server
    with RemoteZip(url, block_size=1 << 16) as rz:
        for name in list(members)[:3]:
            rz.read(name)
        # Three ~40 kB members out of a ~2.4 MB archive: the transfer must be a fraction
        # of the whole, which is the entire justification for this code path.
        assert rz.stats.bytes_fetched < 0.5 * rz.archive_bytes
        assert rz.stats.requests > 0


def test_extract_to_writes_file(zip_server, tmp_path):
    url, members = zip_server
    with RemoteZip(url) as rz:
        n = rz.extract_to("dir/file_005.bin", tmp_path / "out.bin")
    assert (tmp_path / "out.bin").read_bytes() == members["dir/file_005.bin"]
    assert n == len(members["dir/file_005.bin"])


def test_parallel_extract_matches_serial(zip_server):
    url, members = zip_server
    wanted = [n for n in members if n.endswith(".bin")][:12]
    got: dict[str, bytes] = {}

    def sink(name, data):
        got[name] = data
        return True

    n, stats = parallel_extract(url, wanted, sink, workers=4)
    assert n == len(wanted)
    for name in wanted:
        assert got[name] == members[name]
    assert stats.requests > 0


def test_parallel_extract_empty_is_noop(zip_server):
    url, _ = zip_server
    n, stats = parallel_extract(url, [], lambda *_: True, workers=4)
    assert n == 0 and stats.requests == 0


def test_unopened_archive_raises():
    with pytest.raises(RemoteReadError):
        RemoteZip("http://127.0.0.1:1/none.zip").namelist()
