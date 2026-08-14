"""Server-side memory accounting for Milvus.

``list_loaded_segments`` reports ``mem_size=0``, so per-segment memory is not usable.
The loaded-data accounting lives in the ``internal_cache_loaded_bytes`` metric family
on the Milvus metrics endpoint (port 9091), broken down by ``data_type``
(``vector_index``, ``vector_field``, ``scalar_index``, ...) and ``location``
(``memory``, ``disk``, ``mixed``). This module polls that endpoint to report the
resident index size and, by sampling in the background, the peak process memory of a
benchmark phase.

The peak figures describe the process that serves the polled endpoint, which is the
whole server for a standalone deployment but only one node of a cluster.
"""

import logging
import re
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, fields
from urllib.parse import urlsplit

log = logging.getLogger(__name__)

MILVUS_METRICS_PORT = 9091
MILVUS_METRICS_TIMEOUT = 5
# Number of consecutive identical samples that mark the loaded data as settled, and the
# wall-clock safety net for waiting on it. The deadline is generous because the wait also
# covers loading a large index from object storage.
MILVUS_INDEX_STABLE_CHECKS = 3
MILVUS_INDEX_STABLE_DEADLINE_SECONDS = 600

# name{label="value",...} value
_METRIC_LINE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>\S+)$")
_LABEL = re.compile(r'(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)="(?P<value>[^"]*)"')


def metrics_url_from_uri(uri: str, port: int = MILVUS_METRICS_PORT) -> str:
    """Derive the Prometheus endpoint from a Milvus connection uri."""
    host = urlsplit(uri).hostname or "localhost"
    return f"http://{host}:{port}/metrics"


def _fmt_bytes(num: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(num) < 1024 or unit == "GiB":
            return f"{num:.1f} {unit}" if unit != "B" else f"{num:.0f} B"
        num /= 1024
    return f"{num:.1f} GiB"


@dataclass
class MilvusMemorySnapshot:
    """One sample of the server-side memory counters, all values in bytes."""

    rss: float = 0.0
    jemalloc_allocated: float = 0.0
    jemalloc_resident: float = 0.0
    vector_index_mem: float = 0.0
    vector_index_disk: float = 0.0
    vector_field_mem: float = 0.0
    vector_field_disk: float = 0.0
    scalar_index_mem: float = 0.0
    scalar_field_mem: float = 0.0
    other_mem: float = 0.0
    loaded_rows: float = 0.0
    loaded_segments: float = 0.0
    loading: float = 0.0

    @property
    def index_memory(self) -> float:
        """Resident vector-index bytes (mmapped indexes are reported separately)."""
        return self.vector_index_mem

    def report(self, row_count: int | None = None) -> str:
        """Full memory use: process totals plus the loaded-data breakdown."""
        per_vector = ""
        if row_count:
            per_vector = f" ({self.vector_index_mem / row_count:.1f} B/vector over {row_count} rows)"
        # loaded_rows above the dataset size means superseded segments are still
        # resident, so the index bytes cover more than the final index.
        return (
            f"rss={_fmt_bytes(self.rss)}, "
            f"jemalloc_allocated={_fmt_bytes(self.jemalloc_allocated)}, "
            f"jemalloc_resident={_fmt_bytes(self.jemalloc_resident)}, "
            f"loaded_rows={self.loaded_rows:.0f}, loaded_segments={self.loaded_segments:.0f}, "
            f"vector_index={_fmt_bytes(self.vector_index_mem)}{per_vector}, "
            f"vector_index_mmap={_fmt_bytes(self.vector_index_disk)}, "
            f"vector_field={_fmt_bytes(self.vector_field_mem)}, "
            f"vector_field_mmap={_fmt_bytes(self.vector_field_disk)}, "
            f"scalar_index={_fmt_bytes(self.scalar_index_mem)}, "
            f"scalar_field={_fmt_bytes(self.scalar_field_mem)}, "
            f"other={_fmt_bytes(self.other_mem)}"
        )


class MilvusMemoryMonitor:
    """Polls the Milvus metrics endpoint for index size and per-phase peak memory.

    Holds configuration only; the sampling thread lives inside ``monitor()`` so the
    monitor stays picklable when the client is copied into runner processes.
    """

    def __init__(self, metrics_url: str, interval: float = 1.0, name: str = "Milvus"):
        self.metrics_url = metrics_url
        self.interval = interval
        self.name = name
        # Sampling is frequent, so an unreachable endpoint is reported once per process.
        self.fetch_error_logged = False

    def _fetch(self) -> str:
        with urllib.request.urlopen(self.metrics_url, timeout=MILVUS_METRICS_TIMEOUT) as resp:  # noqa: S310
            return resp.read().decode()

    def snapshot(self) -> MilvusMemorySnapshot | None:
        """Read the current counters, or None if the endpoint is unreachable."""
        try:
            text = self._fetch()
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            if not self.fetch_error_logged:
                self.fetch_error_logged = True
                log.warning(f"{self.name} cannot read memory metrics from {self.metrics_url}: {e}")
            return None

        samples: list[tuple[str, dict[str, str], float]] = []
        for line in text.splitlines():
            if not line or line.startswith("#"):
                continue
            m = _METRIC_LINE.match(line)
            if m is None:
                continue
            try:
                value = float(m.group("value"))
            except ValueError:  # NaN placeholders and non-numeric values
                continue
            labels = {lm.group("key"): lm.group("value") for lm in _LABEL.finditer(m.group("labels") or "")}
            samples.append((m.group("name"), labels, value))

        def total(name: str, **labels: str) -> float:
            return sum(v for n, lb, v in samples if n == name and all(lb.get(k) == w for k, w in labels.items()))

        loaded = "internal_cache_loaded_bytes"
        return MilvusMemorySnapshot(
            rss=total("process_resident_memory_bytes"),
            jemalloc_allocated=total("milvus_jemalloc_allocated_bytes"),
            jemalloc_resident=total("milvus_jemalloc_resident_bytes"),
            vector_index_mem=total(loaded, data_type="vector_index", location="memory"),
            vector_index_disk=total(loaded, data_type="vector_index", location="disk"),
            vector_field_mem=total(loaded, data_type="vector_field", location="memory"),
            vector_field_disk=total(loaded, data_type="vector_field", location="disk"),
            scalar_index_mem=total(loaded, data_type="scalar_index", location="memory"),
            scalar_field_mem=total(loaded, data_type="scalar_field", location="memory"),
            other_mem=total(loaded, data_type="other", location="memory"),
            loaded_rows=total("milvus_querynode_entity_num", segment_state="Sealed"),
            loaded_segments=total("milvus_querynode_segment_num", segment_state="Sealed", segment_level="L1"),
            loading=total("internal_cache_loading_bytes"),
        )

    def _settled_snapshot(self, expected_rows: int | None = None) -> MilvusMemorySnapshot | None:
        """Sample until the loaded data settles.

        Settled means nothing is being loaded, the expected rows are resident, and the
        index size stops changing. The wait matters twice: a compaction leaves the
        superseded segments resident for a while after the new one is loaded, which
        counts their indexes twice, and a server that has just been restarted reports
        a partial index until it finishes loading the collection from storage.
        """
        deadline = time.monotonic() + MILVUS_INDEX_STABLE_DEADLINE_SECONDS
        interval = max(self.interval, 1.0)
        previous = None
        stable = 0
        waiting_logged = False
        while True:
            snapshot = self.snapshot()
            if snapshot is None:
                return None
            loading_done = snapshot.loading == 0
            rows_covered = expected_rows is None or snapshot.loaded_rows >= expected_rows
            if not (loading_done and rows_covered):
                stable = 0
                if not waiting_logged:
                    waiting_logged = True
                    log.info(
                        f"{self.name} waiting for the collection to finish loading "
                        f"(loading={_fmt_bytes(snapshot.loading)}, loaded_rows={snapshot.loaded_rows:.0f})",
                    )
            elif snapshot.vector_index_mem == previous:
                stable += 1
                if stable >= MILVUS_INDEX_STABLE_CHECKS:
                    return snapshot
            else:
                stable = 0
            previous = snapshot.vector_index_mem
            if time.monotonic() >= deadline:
                log.warning(
                    f"{self.name} loaded data still unsettled after "
                    f"{MILVUS_INDEX_STABLE_DEADLINE_SECONDS}s "
                    f"(loading={_fmt_bytes(snapshot.loading)}, loaded_rows={snapshot.loaded_rows:.0f}, "
                    f"expected_rows={expected_rows}), reporting the last sample.",
                )
                return snapshot
            time.sleep(interval)

    def log_memory(self, label: str, row_count: int | None = None):
        """Log memory use once the collection is fully loaded and settled.

        ``row_count`` is both the expected resident row count to wait for and the
        divisor for the per-vector index size.
        """
        snapshot = self._settled_snapshot(expected_rows=row_count)
        if snapshot is None:
            return
        log.info(f"{self.name} memory [{label}] {snapshot.report(row_count)}")

    @contextmanager
    def monitor(self, phase: str):
        """Sample memory in the background and log the peak when the phase ends."""
        peak = MilvusMemorySnapshot()
        counters = [f.name for f in fields(MilvusMemorySnapshot)]
        samples = 0
        stop = threading.Event()

        def poll():
            nonlocal samples
            while True:
                snapshot = self.snapshot()
                if snapshot is not None:
                    samples += 1
                    for counter in counters:
                        setattr(peak, counter, max(getattr(peak, counter), getattr(snapshot, counter)))
                if stop.wait(self.interval):
                    return

        thread = threading.Thread(target=poll, name=f"milvus-memory-{phase}", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=MILVUS_METRICS_TIMEOUT + self.interval)
            if samples == 0:
                log.warning(f"{self.name} memory [{phase}] no samples collected from {self.metrics_url}")
            else:
                log.info(
                    f"{self.name} memory [{phase}] peak: "
                    f"rss={_fmt_bytes(peak.rss)}, "
                    f"jemalloc_allocated={_fmt_bytes(peak.jemalloc_allocated)}, "
                    f"jemalloc_resident={_fmt_bytes(peak.jemalloc_resident)}, "
                    f"vector_index={_fmt_bytes(peak.vector_index_mem)}, "
                    f"vector_field={_fmt_bytes(peak.vector_field_mem)} "
                    f"({samples} samples every {self.interval}s)",
                )
