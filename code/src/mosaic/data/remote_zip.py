"""Selective extraction from a remote ZIP archive over HTTP range requests.

Why this exists
---------------
The real corpora this project needs are published as single monolithic archives:
ASVspoof2019 LA is a 7.64 GB zip on Edinburgh DataShare, In-The-Wild is an 8.16 GB zip
on Hugging Face. Downloading either in full blows the Stage-1 footprint budget several
times over, and Stage 1 needs roughly 280 clips totalling ~50 MB.

Both servers advertise ``Accept-Ranges: bytes``, so the archive can be treated as a
random-access file. This module implements a seekable file-like object backed by HTTP
range requests, which ``zipfile.ZipFile`` then drives directly — meaning the standard
library handles central-directory parsing, ZIP64 (both archives exceed 4 GB, so ZIP64 is
mandatory), and per-member decompression. Hand-rolling that parsing would be a large
source of subtle bugs for no benefit.

Reads are served from a block cache so that ``zipfile``'s many small seeks collapse into
a modest number of range requests.
"""

from __future__ import annotations

import io
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from typing import Iterable

DEFAULT_BLOCK = 1 << 19          # 512 KiB — one block usually covers a whole clip
DEFAULT_UA = "mosaic-av/0.1 (research prototype; contact: repository owner)"


class RemoteReadError(RuntimeError):
    """Raised when the remote archive cannot be read. Never silently swallowed."""


@dataclass
class TransferStats:
    """Byte accounting, so acquisition logs can state exactly what was pulled."""

    requests: int = 0
    bytes_fetched: int = 0
    retries: int = 0

    def as_dict(self) -> dict:
        return {"range_requests": self.requests,
                "bytes_fetched": self.bytes_fetched,
                "megabytes_fetched": round(self.bytes_fetched / 1e6, 3),
                "retries": self.retries}


class HTTPRangeFile(io.RawIOBase):
    """A seekable read-only file over HTTP, served by range requests with a block cache."""

    def __init__(self, url: str, *, block_size: int = DEFAULT_BLOCK, timeout: int = 60,
                 headers: dict | None = None, max_retries: int = 4,
                 stats: TransferStats | None = None):
        self.url = url
        self.block_size = block_size
        self.timeout = timeout
        self.max_retries = max_retries
        self.stats = stats or TransferStats()
        self._headers = {"User-Agent": DEFAULT_UA}
        if headers:
            self._headers.update(headers)
        self._pos = 0
        self._cache: dict[int, bytes] = {}
        # A persistent session matters more than bandwidth here. Edinburgh DataShare adds
        # ~3.7 s of connection setup to every fresh request; reusing one keep-alive
        # connection cuts that to ~1 s and roughly triples effective throughput. Over the
        # few hundred range requests an acquisition run needs, that is the difference
        # between minutes and hours.
        self._session = None
        try:
            import requests

            self._session = requests.Session()
            self._session.headers.update(self._headers)
        except ImportError:
            self._session = None
        self._size = self._probe_size()

    # -- HTTP ---------------------------------------------------------------------------

    def _probe_size(self) -> int:
        try:
            status, headers, _ = self._request("bytes=0-0")
            if status != 206:
                raise RemoteReadError(
                    f"server did not honour a range request (status {status}); "
                    "selective extraction is not possible for this URL"
                )
            cr = headers.get("Content-Range", "")
            if "/" not in cr:
                raise RemoteReadError(f"missing Content-Range in response: {cr!r}")
            total = cr.rsplit("/", 1)[1]
            if not total.isdigit():
                raise RemoteReadError(f"server reported an unknown total size: {cr!r}")
            return int(total)
        except RemoteReadError:
            raise
        except Exception as exc:
            raise RemoteReadError(f"cannot probe {self.url}: {type(exc).__name__}: {exc}") from exc

    def _request(self, rng: str):
        """One range request. Returns (status, headers, body)."""
        if self._session is not None:
            r = self._session.get(self.url, headers={"Range": rng}, timeout=self.timeout)
            return r.status_code, r.headers, r.content
        req = urllib.request.Request(self.url, headers={**self._headers, "Range": rng})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return r.status, r.headers, r.read()

    def _fetch(self, start: int, end: int) -> bytes:
        """Fetch an inclusive byte range, with bounded retries on transient failures."""
        end = min(end, self._size - 1)
        if end < start:
            return b""
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                status, _, data = self._request(f"bytes={start}-{end}")
                if status not in (200, 206):
                    raise RemoteReadError(f"unexpected status {status}")
                self.stats.requests += 1
                self.stats.bytes_fetched += len(data)
                return data
            except Exception as exc:                       # transient network failure
                last = exc
                self.stats.retries += 1
                time.sleep(min(2 ** attempt, 8))
        raise RemoteReadError(
            f"range {start}-{end} of {self.url} failed after {self.max_retries} attempts: "
            f"{type(last).__name__}: {last}")

    def _block(self, index: int) -> bytes:
        blk = self._cache.get(index)
        if blk is None:
            start = index * self.block_size
            blk = self._fetch(start, start + self.block_size - 1)
            self._cache[index] = blk
        return blk

    # -- file protocol ------------------------------------------------------------------

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._size + offset
        else:
            raise ValueError(f"bad whence {whence}")
        self._pos = max(0, min(self._pos, self._size))
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self._size - self._pos
        size = max(0, min(size, self._size - self._pos))
        if size == 0:
            return b""
        out = bytearray()
        pos = self._pos
        remaining = size
        # Large reads bypass the block cache: caching a whole member would blow memory
        # for no reuse, since members are read exactly once.
        if remaining > 4 * self.block_size:
            data = self._fetch(pos, pos + remaining - 1)
            self._pos = pos + len(data)
            return bytes(data)
        while remaining > 0:
            idx = pos // self.block_size
            off = pos % self.block_size
            blk = self._block(idx)
            if not blk:
                break
            take = blk[off:off + remaining]
            if not take:
                break
            out += take
            pos += len(take)
            remaining -= len(take)
        self._pos = pos
        return bytes(out)

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)

    @property
    def size(self) -> int:
        return self._size


@dataclass
class RemoteZip:
    """A remote ZIP archive from which individual members can be extracted."""

    url: str
    headers: dict | None = None
    block_size: int = DEFAULT_BLOCK
    stats: TransferStats = field(default_factory=TransferStats)
    _fp: HTTPRangeFile | None = field(default=None, repr=False)
    _zf: zipfile.ZipFile | None = field(default=None, repr=False)

    def open(self) -> "RemoteZip":
        self._fp = HTTPRangeFile(self.url, headers=self.headers, stats=self.stats,
                                 block_size=self.block_size)
        # A large read buffer keeps central-directory parsing (tens of MB for a corpus with
        # ~120k members) from becoming tens of thousands of tiny range requests. It is tied
        # to the block size so that shrinking the block genuinely shrinks the transfer
        # rather than being overridden by the buffer.
        self._zf = zipfile.ZipFile(
            io.BufferedReader(self._fp, buffer_size=max(1024, self.block_size * 8)))
        return self

    def __enter__(self) -> "RemoteZip":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._zf is not None:
            self._zf.close()
        if self._fp is not None:
            if getattr(self._fp, "_session", None) is not None:
                self._fp._session.close()
            self._fp.close()

    @property
    def archive_bytes(self) -> int:
        return self._fp.size if self._fp else 0

    def namelist(self) -> list[str]:
        self._check()
        return self._zf.namelist()

    def infolist(self) -> list[zipfile.ZipInfo]:
        self._check()
        return self._zf.infolist()

    def read(self, name: str) -> bytes:
        self._check()
        return self._zf.read(name)

    def extract_to(self, name: str, dest) -> int:
        """Extract one member to ``dest``. Returns bytes written."""
        from pathlib import Path

        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = self.read(name)
        dest.write_bytes(data)
        return len(data)

    def _check(self) -> None:
        if self._zf is None:
            raise RemoteReadError("archive is not open; call open() or use as a context manager")


# --------------------------------------------------------------------------------------
# Parallel extraction
# --------------------------------------------------------------------------------------


def parallel_extract(url: str, members: list[str], sink, *, workers: int = 4,
                     headers: dict | None = None, on_done=None) -> tuple[int, TransferStats]:
    """Extract many members concurrently, one archive handle per worker thread.

    Extraction of small members is latency-bound, not bandwidth-bound: each clip costs one
    round trip regardless of its size, and measured single-threaded throughput was ~18 s
    per clip against Edinburgh DataShare. ``zipfile`` is not safe to share across threads,
    so each worker opens its own handle — the central directory is re-read per worker
    (a few MB, once), which pays for itself after a handful of clips.

    ``sink(name, data) -> bool`` writes one member and returns True on success.
    Returns (n_written, aggregate transfer stats).
    """
    from concurrent.futures import ThreadPoolExecutor
    from queue import Queue

    if not members:
        return 0, TransferStats()

    workers = max(1, min(workers, len(members)))
    handles: Queue = Queue()
    opened: list[RemoteZip] = []
    for _ in range(workers):
        rz = RemoteZip(url, headers=headers).open()
        opened.append(rz)
        handles.put(rz)

    written = 0
    lock = __import__("threading").Lock()

    def one(name: str) -> bool:
        nonlocal written
        rz = handles.get()
        try:
            data = rz.read(name)
        except Exception:
            return False
        finally:
            handles.put(rz)
        ok = bool(sink(name, data))
        if ok:
            with lock:
                written += 1
                if on_done and written % 20 == 0:
                    on_done(written, len(members))
        return ok

    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(one, members))
    finally:
        agg = TransferStats()
        for rz in opened:
            agg.requests += rz.stats.requests
            agg.bytes_fetched += rz.stats.bytes_fetched
            agg.retries += rz.stats.retries
            rz.close()
    return written, agg
