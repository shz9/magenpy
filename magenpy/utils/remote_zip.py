"""Read-only ZIP access optimized for remote Zarr stores.

The standard Python ZIP reader expects a seekable file. For remote archives,
each seek may become a separate HTTP request, and a general-purpose block
cache can download substantially more matrix data than the caller requested.
This module provides the three layers needed to avoid those costs:

* :class:`_RemoteRangeReader` translates reads into HTTP byte-range requests,
  caches the ZIP tail, and plans coalesced reads for Zarr initialization data.
* :class:`_RemoteRangeFile` exposes that reader through Python's seekable
  binary-file interface so that :class:`zipfile.ZipFile` can consume it.
* :class:`ReadOnlyZipStore` exposes ZIP members as a read-only mapping that
  can be passed directly to Zarr.

Small metadata and index entries are cached together because they are usually
read during matrix initialization. Compressed ``matrix/data`` payloads form
barriers between those cached ranges and are fetched exactly when requested.
"""

from __future__ import annotations

import io
import re
import threading
import time
import zipfile
from bisect import bisect_right
from collections.abc import MutableMapping
from urllib.request import Request, urlopen

ZIP_LOCAL_HEADER_SIZE = 30
ZIP_LOCAL_HEADER_SLACK = 64
# Python's synchronous ZIP parser benefits from fetching the trailing Zarr
# initialization entries with the central directory. Two MiB covers this
# region in the largest UK Biobank LD archive while remaining a small fraction
# of the complete remote ZIP.
DEFAULT_TAIL_SIZE = 2 * 1_024 * 1_024
INITIALIZATION_ENTRY = re.compile(
    r"(?:^|/)(?:\.zmetadata|matrix/indptr/[^/.][^/]*|metadata/[^/]+/[^/.][^/]*)$"
)
MATRIX_DATA_ENTRY = re.compile(r"(?:^|/)matrix/data/[^/.][^/]*$")


def _containing(ranges, offset, length):
    """Find a cached byte range that completely contains a requested span.

    :param ranges: Iterable of ``(offset, data)`` tuples, where ``offset`` is
        the absolute position in the remote object and ``data`` is bytes-like.
    :param offset: Absolute start offset of the requested span.
    :param length: Number of requested bytes.
    :return: The first containing ``(offset, data)`` tuple, or ``None`` when
        the requested span is not fully cached.
    """

    end = offset + length
    return next(
        (
            byte_range
            for byte_range in ranges
            if byte_range[0] <= offset and byte_range[0] + len(byte_range[1]) >= end
        ),
        None,
    )


class _RemoteRangeReader:
    """Serve random reads from a remote object using HTTP byte ranges.

    The reader keeps only deliberately reusable ranges: the ZIP tail, planned
    Zarr initialization ranges, matrix local-file headers, or a complete file
    returned by a server that ignored the ``Range`` header. Ordinary matrix
    payload reads are not cached.

    Access to position-independent state and the range cache is serialized by
    a re-entrant lock. This also prevents duplicate requests when a single
    :class:`zipfile.ZipFile` is shared between threads.
    """

    def __init__(
        self,
        url,
        headers=None,
        coalesce_gap=8_192,
        tail_size=DEFAULT_TAIL_SIZE,
        request=None,
    ):
        """Initialize a remote byte-range reader.

        :param url: HTTP(S) URL of the remote ZIP archive.
        :param headers: Optional headers included in direct HTTP requests,
            such as an authorization header.
        :param coalesce_gap: Maximum number of bytes allowed between adjacent
            initialization entries before their planned reads are combined.
        :param tail_size: Number of bytes requested from the end of the object
            during ZIP discovery. The default also covers the trailing Zarr
            initialization entries in the expected LD archive layout, which
            lets later metadata reads reuse the initial response.
        :param request: Optional callable accepting an HTTP ``Range`` value
            and returning ``(status, headers, data)``. This allows callers to
            supply a persistent or authenticated HTTP client.
        """

        self.url = url
        self.headers = dict(headers or {})
        self.coalesce_gap = coalesce_gap
        self.tail_size = tail_size
        self.length = None
        self.ranges = []
        self.planned = {}
        self.matrix_headers = {}
        self.data_barriers = []
        self.request_count = 0
        self.bytes_transferred = 0
        self.request_log = []
        self._request = request or self._http_request
        self._lock = threading.RLock()

    def _http_request(self, range_header):
        """Execute one byte-range GET with Python's standard HTTP client.

        :param range_header: Value for the HTTP ``Range`` header.
        :return: A ``(status, headers, data)`` tuple.
        """

        headers = dict(self.headers)
        headers["Range"] = range_header
        headers.setdefault("Accept-Encoding", "identity")
        request = Request(self.url, headers=headers, method="GET")
        with urlopen(request) as response:
            return response.status, dict(response.headers.items()), response.read()

    def _request_range(self, range_header):
        """Execute and record one HTTP byte-range request.

        Besides normalizing response-header names, this method maintains the
        request count, transferred-byte count, and per-request timing log used
        for diagnostics and benchmarks.

        :param range_header: Value for the HTTP ``Range`` header.
        :return: A ``(status, lowercase_headers, data)`` tuple.
        :raises OSError: If the server returns a status other than 200 or 206.
        """

        started_at = time.monotonic()
        status, headers, data = self._request(range_header)
        if status not in (200, 206):
            raise OSError(f"HTTP range request failed for {self.url}: status {status}")
        self.request_count += 1
        self.bytes_transferred += len(data)
        self.request_log.append(
            (range_header, len(data), time.monotonic() - started_at)
        )
        return status, {key.lower(): value for key, value in headers.items()}, data

    def get_length(self):
        """Return the remote object's length and cache its trailing bytes.

        The first call requests a suffix of the archive. ZIP readers need this
        region to locate the end-of-central-directory record and, for typical
        archives, subsequent tail reads can therefore be served from memory.
        Later calls return the cached length without performing I/O.

        :return: Total remote object length in bytes.
        """

        with self._lock:
            if self.length is not None:
                return self.length

            # A suffix request obtains both the file length and the ZIP tail.
            status, headers, data = self._request_range(f"bytes=-{self.tail_size}")
            content_range = headers.get("content-range", "")
            match = re.search(r"/(\d+)$", content_range)
            length = (
                int(match.group(1))
                if match
                else int(headers.get("content-length", len(data)))
            )
            self.length = length
            offset = max(0, length - len(data)) if status == 206 else 0
            self.ranges.append((offset, data))
            return length

    @staticmethod
    def _filename_length(info):
        """Return a ZIP member's encoded filename length.

        ZIP local headers store the byte length rather than the character
        length. The UTF-8 flag determines which encoding the ZIP reader uses.

        :param info: :class:`zipfile.ZipInfo` describing an archive member.
        :return: Encoded filename length in bytes.
        """

        encoding = "utf-8" if info.flag_bits & 0x800 else "cp437"
        return len(info.orig_filename.encode(encoding))

    def register_initialization_entries(self, entries):
        """Plan cacheable reads from the parsed ZIP central directory.

        Zarr initialization repeatedly opens consolidated metadata, ``indptr``
        chunks, and variant-metadata chunks. Plans cover each such member's
        local header and compressed payload, plus a small allowance for local
        header fields that can differ from their central-directory versions.
        Nearby plans are combined to reduce HTTP round trips.

        ``matrix/data`` entries are recorded as barriers. No combined range is
        allowed to cross their local-header offsets, which prevents a metadata
        request from accidentally downloading a large matrix payload. Only a
        matrix member's small local header is cached; its compressed data is
        still fetched exactly by :meth:`read`.

        :param entries: Iterable of :class:`zipfile.ZipInfo` objects, normally
            returned by :meth:`zipfile.ZipFile.infolist`.
        """

        plans = []
        data_entries = []

        # Classify relevant members and estimate their complete local spans.
        for info in entries:
            offset = info.header_offset
            header_length = (
                ZIP_LOCAL_HEADER_SIZE + self._filename_length(info) + len(info.extra)
            )
            if MATRIX_DATA_ENTRY.search(info.filename):
                data_entries.append((offset, header_length + ZIP_LOCAL_HEADER_SLACK))
            elif INITIALIZATION_ENTRY.search(info.filename):
                plans.append(
                    (
                        offset,
                        header_length + info.compress_size + ZIP_LOCAL_HEADER_SLACK,
                    )
                )

        self.data_barriers = sorted(offset for offset, _ in data_entries)
        self.matrix_headers = dict(data_entries)

        # Truncate initialization ranges at the next matrix-data member.
        capped_plans = []
        for offset, length in sorted(plans):
            barrier_index = bisect_right(self.data_barriers, offset)
            if barrier_index < len(self.data_barriers):
                length = min(length, self.data_barriers[barrier_index] - offset)
            if length >= ZIP_LOCAL_HEADER_SIZE:
                capped_plans.append((offset, length))

        # Merge nearby initialization ranges when no matrix barrier intervenes.
        groups = []
        for offset, length in capped_plans:
            end = offset + length
            previous = groups[-1] if groups else None
            barrier_index = (
                bisect_right(self.data_barriers, previous[0])
                if previous is not None
                else 0
            )
            crosses_matrix_data = (
                previous is not None
                and barrier_index < len(self.data_barriers)
                and self.data_barriers[barrier_index] < offset
            )
            if (
                previous is not None
                and not crosses_matrix_data
                and offset <= previous[1] + self.coalesce_gap
            ):
                previous[1] = max(previous[1], end)
                previous[2].append(offset)
            else:
                groups.append([offset, end, [offset]])

        # Every entry offset points to its group's shared fetch plan.
        for offset, end, entry_offsets in groups:
            plan = (offset, end - offset)
            for entry_offset in entry_offsets:
                self.planned[entry_offset] = plan

    def read(self, offset, length):
        """Read an absolute byte span from the remote ZIP.

        Cached spans are returned immediately. A 30-byte read at the local
        header of a planned initialization entry triggers its coalesced group
        fetch. A matrix-data local header receives a small cached read with
        slack, while all other reads—including matrix payloads—remain exact
        and uncached.

        :param offset: Absolute byte offset within the remote archive.
        :param length: Maximum number of bytes to return.
        :return: Requested bytes, truncated at end-of-file when necessary.
        """

        if length <= 0:
            return b""

        with self._lock:
            total_length = self.get_length()
            length = min(length, total_length - offset)
            cached = _containing(self.ranges, offset, length)
            if cached is not None:
                start = offset - cached[0]
                return cached[1][start : start + length]

            plan = self.planned.get(offset) if length == ZIP_LOCAL_HEADER_SIZE else None
            if plan is not None:
                byte_range = self._fetch_range(*plan, cache=True)
            elif offset in self.matrix_headers and length == ZIP_LOCAL_HEADER_SIZE:
                byte_range = self._fetch_range(
                    offset,
                    self.matrix_headers[offset],
                    cache=True,
                )
            else:
                byte_range = self._fetch_range(offset, length, cache=False)

            start = offset - byte_range[0]
            return byte_range[1][start : start + length]

    def _fetch_range(self, offset, requested_length, cache):
        """Fetch one bounded byte span and optionally retain it.

        Servers are allowed to ignore ``Range`` and return the complete object
        with status 200. Such a response is recognized as starting at offset
        zero and retained because it can satisfy every subsequent read.

        :param offset: Absolute start offset within the remote object.
        :param requested_length: Desired number of bytes before EOF clipping.
        :param cache: Whether to retain the returned range for later reads.
        :return: An ``(absolute_response_offset, data)`` tuple.
        """

        total_length = self.get_length()
        length = min(requested_length, total_length - offset)
        cached = _containing(self.ranges, offset, length)
        if cached is not None:
            return cached

        status, headers, data = self._request_range(
            f"bytes={offset}-{offset + length - 1}"
        )
        content_range = headers.get("content-range", "")
        match = re.match(r"bytes\s+(\d+)-", content_range, re.IGNORECASE)
        response_offset = (
            int(match.group(1))
            if match
            else (0 if status == 200 and len(data) == total_length else offset)
        )
        byte_range = (response_offset, data)
        if cache or response_offset == 0 and len(data) == total_length:
            self.ranges.append(byte_range)
        return byte_range


class _RemoteRangeFile(io.RawIOBase):
    """Adapt :class:`_RemoteRangeReader` to a seekable binary file.

    :class:`zipfile.ZipFile` performs ordinary ``seek`` and ``read`` calls.
    This lightweight adapter tracks the current position while delegating all
    actual byte retrieval and caching decisions to the range reader.
    """

    def __init__(self, reader):
        """Initialize the file adapter at position zero.

        :param reader: :class:`_RemoteRangeReader` that supplies archive bytes.
        """

        super().__init__()
        self.reader = reader
        self.position = 0
        self.name = reader.url

    def readable(self):
        """Return ``True`` because the adapter supports binary reads."""

        return True

    def seekable(self):
        """Return ``True`` because absolute and relative seeks are supported."""

        return True

    def tell(self):
        """Return the current absolute byte position."""

        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        """Move the current position using standard file seek semantics.

        :param offset: Position relative to ``whence``.
        :param whence: One of ``io.SEEK_SET``, ``io.SEEK_CUR``, or
            ``io.SEEK_END``.
        :return: The resulting absolute position.
        :raises ValueError: If ``whence`` is unsupported or the resulting
            position would be negative.
        """

        if whence == io.SEEK_CUR:
            offset += self.position
        elif whence == io.SEEK_END:
            offset += self.reader.get_length()
        elif whence != io.SEEK_SET:
            raise ValueError(f"Unsupported whence value: {whence}")
        if offset < 0:
            raise ValueError("Negative seek position")
        self.position = offset
        return offset

    def read(self, size=-1):
        """Read bytes at the current position and advance it.

        :param size: Maximum bytes to read. A negative value or ``None`` reads
            through the end of the remote object.
        :return: Bytes returned by the underlying range reader.
        """

        if size is None or size < 0:
            size = self.reader.get_length() - self.position
        data = self.reader.read(self.position, size)
        self.position += len(data)
        return data

    def readinto(self, buffer):
        """Read bytes into a writable buffer.

        :param buffer: Writable bytes-like object to fill.
        :return: Number of bytes written into ``buffer``.
        """

        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


class ReadOnlyZipStore(MutableMapping):
    """Expose a remote ZIP archive as a read-only Zarr-compatible mapping.

    Mapping keys are ZIP member names relative to an optional archive prefix.
    Values are the uncompressed member bytes returned by
    :class:`zipfile.ZipFile`. Member discovery happens once from the central
    directory, after which the range reader is given enough layout information
    to optimize Zarr initialization reads.

    Although this class implements :class:`collections.abc.MutableMapping` for
    compatibility with Zarr 2's store normalization, mutation methods always
    raise :class:`TypeError`.
    """

    def __init__(
        self,
        url,
        prefix=None,
        headers=None,
        coalesce_gap=8_192,
        tail_size=DEFAULT_TAIL_SIZE,
        request=None,
    ):
        """Open a remote ZIP archive and prepare its range-read plans.

        :param url: HTTP(S) URL of the remote ZIP archive.
        :param prefix: Optional directory containing the Zarr store inside the
            archive. If no archive members use the requested prefix, the store
            automatically falls back to the archive root.
        :param headers: Optional headers for direct HTTP requests.
        :param coalesce_gap: Maximum gap between initialization entries that
            may be fetched as a single range.
        :param tail_size: Size of the initial ZIP-tail suffix request. The
            default is tuned to include trailing initialization metadata in
            remote LD archives.
        :param request: Optional range-request callable returning
            ``(status, headers, data)``. Useful for persistent sessions and
            authenticated clients.
        :raises zipfile.BadZipFile: If the remote object is not a valid ZIP.
        """

        # Layer the ZIP parser over the seekable remote range adapter.
        self.reader = _RemoteRangeReader(
            url,
            headers=headers,
            coalesce_gap=coalesce_gap,
            tail_size=tail_size,
            request=request,
        )
        self._file = _RemoteRangeFile(self.reader)
        self._zip = zipfile.ZipFile(self._file, mode="r")

        # The central directory supplies offsets and sizes for range planning.
        self.reader.register_initialization_entries(self._zip.infolist())

        # Present member names relative to the requested Zarr root. Archives
        # that already store Zarr keys at their root need no prefix.
        self.prefix = str(prefix or "").strip("/")
        archive_files = [
            info.filename.rstrip("/")
            for info in self._zip.infolist()
            if not info.is_dir()
        ]
        prefix_with_separator = f"{self.prefix}/" if self.prefix else ""
        if self.prefix and not any(
            name.startswith(prefix_with_separator) for name in archive_files
        ):
            self.prefix = ""
            prefix_with_separator = ""
        self._names = {
            name[len(prefix_with_separator) :]
            for name in archive_files
            if name.startswith(prefix_with_separator)
        }

    @property
    def request_count(self):
        """Return the number of HTTP range requests issued by this store."""

        return self.reader.request_count

    @property
    def bytes_transferred(self):
        """Return the cumulative response-body bytes downloaded by the store."""

        return self.reader.bytes_transferred

    def __getitem__(self, key):
        """Read and decompress one ZIP member.

        :param key: Member name relative to the configured archive prefix.
        :return: Uncompressed member contents as ``bytes``.
        :raises KeyError: If the member does not exist.
        """

        key = str(key).lstrip("/")
        archive_key = f"{self.prefix}/{key}" if self.prefix else key
        try:
            with self._zip.open(archive_key, mode="r") as member:
                return member.read()
        except KeyError as exc:
            raise KeyError(key) from exc

    def __contains__(self, key):
        """Return whether a relative member name exists in the store."""

        return str(key).lstrip("/") in self._names

    def __iter__(self):
        """Iterate over member names relative to the archive prefix."""

        return iter(self._names)

    def __len__(self):
        """Return the number of files exposed by the mapping."""

        return len(self._names)

    def __setitem__(self, key, value):
        """Reject member writes because the remote store is read-only."""

        raise TypeError("ReadOnlyZipStore does not support writes.")

    def __delitem__(self, key):
        """Reject member deletion because the remote store is read-only."""

        raise TypeError("ReadOnlyZipStore does not support writes.")

    def close(self):
        """Close the ZIP parser and its seekable file adapter."""

        self._zip.close()
        self._file.close()
