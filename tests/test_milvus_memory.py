"""Unit tests for the Milvus memory reporting. No running Milvus needed."""

import logging
import urllib.error

import pytest

from vectordb_bench.backend.clients.milvus import memory
from vectordb_bench.backend.clients.milvus.memory import (
    MilvusMemoryMonitor,
    metrics_url_from_uri,
)

METRICS = """
# HELP process_resident_memory_bytes Resident memory size in bytes.
# TYPE process_resident_memory_bytes gauge
process_resident_memory_bytes 9.239e+08
milvus_jemalloc_allocated_bytes 2.9e+08
milvus_jemalloc_resident_bytes 1.09e+09
internal_cache_loaded_bytes{data_type="vector_index",location="memory"} 9.0624975e+07
internal_cache_loaded_bytes{data_type="vector_index",location="disk"} 0
internal_cache_loaded_bytes{data_type="vector_field",location="memory"} 0
internal_cache_loaded_bytes{data_type="vector_field",location="disk"} 3.072e+08
internal_cache_loaded_bytes{data_type="scalar_index",location="memory"} 1.213319e+06
internal_cache_loaded_bytes{data_type="scalar_field",location="memory"} 800000
internal_cache_loaded_bytes{data_type="other",location="memory"} 261553
internal_cache_loading_bytes{data_type="vector_index",location="memory"} 0
internal_cache_loading_bytes{data_type="scalar_index",location="memory"} 0
milvus_querynode_entity_num{collection_name="VDBBench",segment_state="Growing"} 0
milvus_querynode_entity_num{collection_name="VDBBench",segment_state="Sealed"} 50000
milvus_querynode_segment_num{segment_level="L0",segment_state="Sealed"} 0
milvus_querynode_segment_num{segment_level="L1",segment_state="Sealed"} 1
some_metric_without_value_that_should_be_skipped
"""


class TestMetricsUrl:
    def test_derives_metrics_port_from_uri(self):
        assert metrics_url_from_uri("http://localhost:19530") == "http://localhost:9091/metrics"
        assert metrics_url_from_uri("https://10.0.0.1:19531") == "http://10.0.0.1:9091/metrics"
        assert metrics_url_from_uri("http://localhost:19530", port=8080) == "http://localhost:8080/metrics"

    def test_falls_back_to_localhost(self):
        assert metrics_url_from_uri("") == "http://localhost:9091/metrics"


class TestSnapshot:
    @pytest.fixture
    def monitor(self, monkeypatch: pytest.MonkeyPatch) -> MilvusMemoryMonitor:
        monitor = MilvusMemoryMonitor("http://localhost:9091/metrics", interval=0.01)
        monkeypatch.setattr(monitor, "_fetch", lambda: METRICS)
        return monitor

    def test_parses_counters(self, monitor: MilvusMemoryMonitor):
        snapshot = monitor.snapshot()

        assert snapshot.rss == 923_900_000
        assert snapshot.jemalloc_allocated == 290_000_000
        assert snapshot.jemalloc_resident == 1_090_000_000
        assert snapshot.vector_index_mem == 90_624_975
        assert snapshot.vector_index_disk == 0
        assert snapshot.vector_field_mem == 0
        assert snapshot.vector_field_disk == 307_200_000
        assert snapshot.scalar_index_mem == 1_213_319
        assert snapshot.scalar_field_mem == 800_000
        assert snapshot.other_mem == 261_553
        assert snapshot.loaded_rows == 50_000
        assert snapshot.loaded_segments == 1
        assert snapshot.loading == 0
        assert snapshot.index_memory == snapshot.vector_index_mem

    def test_report_includes_bytes_per_vector(self, monitor: MilvusMemoryMonitor):
        report = monitor.snapshot().report(row_count=50000)

        assert "vector_index=86.4 MiB (1812.5 B/vector over 50000 rows)" in report
        assert "vector_field_mmap=293.0 MiB" in report
        assert "loaded_rows=50000, loaded_segments=1" in report

    def test_report_includes_process_memory(self, monitor: MilvusMemoryMonitor):
        report = monitor.snapshot().report()

        assert "rss=881.1 MiB" in report
        assert "jemalloc_allocated=276.6 MiB" in report
        assert "jemalloc_resident=1.0 GiB" in report

    def test_report_without_row_count(self, monitor: MilvusMemoryMonitor):
        assert "B/vector" not in monitor.snapshot().report()

    def test_index_memory_waits_for_compaction_to_release_segments(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ):
        # The old segment is still resident in the first samples, doubling the index.
        double = METRICS.replace('location="memory"} 9.0624975e+07', 'location="memory"} 1.8124995e+08')
        pages = [double, double, METRICS, METRICS, METRICS, METRICS]
        monitor = MilvusMemoryMonitor("http://localhost:9091/metrics", interval=0.01)
        monkeypatch.setattr(monitor, "_fetch", lambda: pages.pop(0))
        monkeypatch.setattr(memory.time, "sleep", lambda _: None)

        with caplog.at_level(logging.INFO):
            monitor.log_memory("index", row_count=50000)

        assert "vector_index=86.4 MiB (1812.5 B/vector over 50000 rows)" in caplog.text
        assert "172.9 MiB" not in caplog.text

    def test_loaded_memory_waits_for_load_from_storage(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ):
        # A restarted server serves a partial index while it reads from storage.
        loading = (
            METRICS.replace('data_type="vector_index",location="memory"} 9.0624975e+07', "")
            .replace('internal_cache_loading_bytes{data_type="vector_index",location="memory"} 0', "")
            .replace('segment_state="Sealed"} 50000', 'segment_state="Sealed"} 20000')
            + 'internal_cache_loaded_bytes{data_type="vector_index",location="memory"} 3.6e+07\n'
            + 'internal_cache_loading_bytes{data_type="vector_index",location="memory"} 5.4e+07\n'
        )
        pages = [loading, loading, *[METRICS] * 4]
        monitor = MilvusMemoryMonitor("http://localhost:9091/metrics", interval=0.01)
        monkeypatch.setattr(monitor, "_fetch", lambda: pages.pop(0))
        monkeypatch.setattr(memory.time, "sleep", lambda _: None)

        with caplog.at_level(logging.INFO):
            monitor.log_memory("loaded", row_count=50000)

        assert "waiting for the collection to finish loading" in caplog.text
        assert "memory [loaded] rss=881.1 MiB" in caplog.text
        assert "vector_index=86.4 MiB (1812.5 B/vector over 50000 rows)" in caplog.text

    def test_settling_waits_for_expected_rows(self, monkeypatch: pytest.MonkeyPatch):
        partial = METRICS.replace('segment_state="Sealed"} 50000', 'segment_state="Sealed"} 20000')
        pages = [partial, partial, *[METRICS] * 4]
        monitor = MilvusMemoryMonitor("http://localhost:9091/metrics", interval=0.01)
        monkeypatch.setattr(monitor, "_fetch", lambda: pages.pop(0))
        monkeypatch.setattr(memory.time, "sleep", lambda _: None)

        assert monitor._settled_snapshot(expected_rows=50000).loaded_rows == 50_000

    def test_monitor_logs_peak(self, monitor: MilvusMemoryMonitor, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.INFO), monitor.monitor("insert"):
            pass

        assert "memory [insert] peak: rss=881.1 MiB" in caplog.text
        assert "vector_index=86.4 MiB" in caplog.text


class TestUnreachableEndpoint:
    @pytest.fixture
    def monitor(self, monkeypatch: pytest.MonkeyPatch) -> MilvusMemoryMonitor:
        def refuse():
            raise urllib.error.URLError("connection refused")

        monitor = MilvusMemoryMonitor("http://localhost:1/metrics", interval=0.01)
        monkeypatch.setattr(monitor, "_fetch", refuse)
        return monitor

    def test_snapshot_returns_none(self, monitor: MilvusMemoryMonitor):
        assert monitor.snapshot() is None

    def test_error_is_logged_once(self, monitor: MilvusMemoryMonitor, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.WARNING):
            monitor.snapshot()
            monitor.snapshot()

        assert caplog.text.count("cannot read memory metrics") == 1

    def test_monitor_reports_no_samples(self, monitor: MilvusMemoryMonitor, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.WARNING), monitor.monitor("search_serial"):
            pass

        assert "memory [search_serial] no samples collected" in caplog.text

    def test_memory_report_is_skipped(self, monitor: MilvusMemoryMonitor, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.INFO):
            monitor.log_memory("index", row_count=50000)
            monitor.log_memory("loaded", row_count=50000)

        assert "memory [index]" not in caplog.text
        assert "memory [loaded]" not in caplog.text
