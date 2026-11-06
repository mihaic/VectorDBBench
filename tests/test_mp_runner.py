import pytest

from vectordb_bench.backend.runner.mp_runner import MultiProcessingSearchRunner


def test_concurrent_search_window_spans_all_workers():
    # QPS is computed over the workers' own search windows, not the time spent collecting results.
    windows = [(100.0, 130.0), (100.5, 130.5), (99.8, 129.9)]

    assert MultiProcessingSearchRunner._search_window(windows) == pytest.approx(30.7)
