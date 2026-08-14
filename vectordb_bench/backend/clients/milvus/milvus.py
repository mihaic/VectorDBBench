"""Wrapper around the Milvus vector database over VectorDB"""

import logging
import time
from collections.abc import Iterable
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any

from pymilvus import DataType, MilvusClient, MilvusException

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import BenchmarkPhase, VectorDB
from .config import MilvusIndexConfig
from .memory import MilvusMemoryMonitor, metrics_url_from_uri

log = logging.getLogger(__name__)

MILVUS_LOAD_REQS_SIZE = 1.5 * 1024 * 1024

# Force-merge target size in MB. Intentionally huge so the server's memory-aware
# force-merge policy decides the real target segment size/count (which may be more
# than one segment when memory is limited).
MILVUS_COMPACT_TARGET_SIZE_MB = 2**31 - 1
# Seconds between polls while waiting for sort / index / segment quiescence.
MILVUS_OPTIMIZE_POLL_INTERVAL = 5
# Number of consecutive identical persistent-segment samples required before the
# segment layout is considered quiescent (no auto/mix compaction in flight).
MILVUS_OPTIMIZE_STABLE_CHECKS = 3
# Wall-clock safety net only. Convergence (a force-merge round that no longer
# reduces the segment count) plus full load is the primary termination signal;
# this deadline merely prevents waiting forever if the cluster never settles.
MILVUS_OPTIMIZE_DEADLINE_SECONDS = 3600 * 3


class Milvus(VectorDB):
    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: MilvusIndexConfig,
        collection_name: str = "VDBBench",
        drop_old: bool = False,
        name: str = "Milvus",
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        """Initialize wrapper around the milvus vector database."""
        self.name = name
        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = collection_name
        self.batch_size = int(MILVUS_LOAD_REQS_SIZE / (dim * 4))
        self.with_scalar_labels = with_scalar_labels
        self.memory_monitor_impl = self._init_memory_monitor()

        self._primary_field = "pk"
        self._scalar_id_field = "id"
        self._scalar_label_field = "label"
        self._vector_field = "vector"
        self._vector_index_name = "vector_idx"
        self._scalar_id_index_name = "id_sort_idx"
        self._scalar_labels_index_name = "labels_idx"

        client = MilvusClient(
            uri=self.db_config.get("uri"),
            user=self.db_config.get("user"),
            password=self.db_config.get("password"),
            timeout=30,
        )

        if drop_old and client.has_collection(self.collection_name):
            log.info(f"{self.name} client drop_old collection: {self.collection_name}")
            client.drop_collection(self.collection_name)

        if not client.has_collection(self.collection_name):
            schema = MilvusClient.create_schema()
            schema.add_field(self._primary_field, DataType.INT64, is_primary=True)
            schema.add_field(self._scalar_id_field, DataType.INT64)
            schema.add_field(self._vector_field, DataType.FLOAT_VECTOR, dim=dim)

            if self.with_scalar_labels:
                is_partition_key = db_case_config.use_partition_key
                log.info(f"with_scalar_labels, add a new varchar field, as partition_key: {is_partition_key}")
                schema.add_field(
                    self._scalar_label_field,
                    DataType.VARCHAR,
                    max_length=256,
                    is_partition_key=is_partition_key,
                )

            log.info(f"{self.name} create collection: {self.collection_name}")

            index_params = self._build_index_params()
            client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                num_shards=self.db_config.get("num_shards", 1),
                consistency_level="Session",
            )
            client.create_index(self.collection_name, index_params)
            client.load_collection(
                self.collection_name,
                replica_number=self.db_config.get("replica_number", 1),
            )

        client.close()

    def _init_memory_monitor(self) -> MilvusMemoryMonitor | None:
        """Build the memory monitor if --log-memory was passed, else None."""
        if not self.db_config.get("log_memory"):
            return None
        metrics_url = self.db_config.get("memory_metrics_uri") or metrics_url_from_uri(self.db_config.get("uri", ""))
        log.info(f"{self.name} logging memory from {metrics_url}")
        return MilvusMemoryMonitor(
            metrics_url=metrics_url,
            interval=self.db_config.get("memory_sample_interval", 1.0),
            name=self.name,
        )

    def memory_monitor(self, phase: BenchmarkPhase) -> AbstractContextManager:
        if self.memory_monitor_impl is None:
            return nullcontext()
        if phase in (BenchmarkPhase.SEARCH_SERIAL, BenchmarkPhase.SEARCH_CONCURRENT):
            # Also report the index here, for runs that skip the load stage and
            # therefore never reach optimize().
            self.memory_monitor_impl.log_index_memory()
        return self.memory_monitor_impl.monitor(phase)

    def _build_index_params(self):
        index_params = MilvusClient.prepare_index_params()
        vec_idx = self.case_config.index_param()
        index_params.add_index(
            field_name=self._vector_field,
            index_name=self._vector_index_name,
            index_type=vec_idx.get("index_type", ""),
            metric_type=vec_idx.get("metric_type", ""),
            params=vec_idx.get("params", {}),
        )
        index_params.add_index(
            field_name=self._scalar_id_field,
            index_name=self._scalar_id_index_name,
            index_type="STL_SORT",
        )
        if self.with_scalar_labels:
            index_params.add_index(
                field_name=self._scalar_label_field,
                index_name=self._scalar_labels_index_name,
                index_type="BITMAP",
            )
        return index_params

    @contextmanager
    def init(self):
        """
        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
            >>>     self.search_embedding()
        """
        self.client: MilvusClient | None = None
        self.client = MilvusClient(
            uri=self.db_config.get("uri"),
            user=self.db_config.get("user"),
            password=self.db_config.get("password"),
            timeout=60,
        )
        yield
        self.client.close()
        self.client = None

    def _persistent_data_segments(self) -> list:
        """Flushed, non-L0 persistent (data) segments for the collection.

        L0 segments only hold deletes and are never force-merge targets, so they
        are excluded from both convergence and load-coverage accounting.
        """
        segments = self.client.list_persistent_segments(self.collection_name)
        return [
            s
            for s in segments
            if s.level_name != "L0" and s.state_name in ("Flushed", "Sealed")
        ]

    def _wait_for_segments_sorted(self, deadline: float):
        while True:
            segments = self.client.list_persistent_segments(self.collection_name)
            unsorted = [s for s in segments if not s.is_sorted]
            if not unsorted:
                log.info(f"{self.name} all persistent segments are sorted.")
                return
            if time.time() >= deadline:
                log.warning(f"{self.name} timed out waiting for {len(unsorted)} segments to be sorted.")
                return
            log.debug(f"{self.name} waiting for {len(unsorted)} segments to be sorted...")
            time.sleep(MILVUS_OPTIMIZE_POLL_INTERVAL)

    def _wait_for_index(self, deadline: float):
        while True:
            info = self.client.describe_index(self.collection_name, self._vector_index_name)
            if info.get("pending_index_rows", -1) == 0:
                return
            if time.time() >= deadline:
                log.warning(
                    f"{self.name} timed out waiting for index, "
                    f"pending_index_rows={info.get('pending_index_rows', -1)}.",
                )
                return
            time.sleep(MILVUS_OPTIMIZE_POLL_INTERVAL)

    def _wait_for_compaction(self, compaction_id: int, deadline: float):
        while True:
            state = self.client.get_compaction_state(compaction_id)
            if state == "Completed":
                return
            if time.time() >= deadline:
                log.warning(f"{self.name} timed out waiting for compaction {compaction_id}, state={state}.")
                return
            time.sleep(MILVUS_OPTIMIZE_POLL_INTERVAL)

    def _wait_until_segments_stable(self, deadline: float) -> list:
        """Wait until the persistent segment layout quiesces.

        Returns once the persistent (non-L0) segment-id set is identical across
        ``MILVUS_OPTIMIZE_STABLE_CHECKS`` consecutive samples while all segments
        are sorted and the index has no pending rows. A stable set means no
        auto/mix compaction is currently rewriting segments.
        """
        prev_ids: frozenset | None = None
        stable = 0
        while True:
            self._wait_for_segments_sorted(deadline)
            self._wait_for_index(deadline)

            segments = self._persistent_data_segments()
            ids = frozenset(s.segment_id for s in segments)
            if ids == prev_ids:
                stable += 1
                if stable >= MILVUS_OPTIMIZE_STABLE_CHECKS:
                    log.info(
                        f"{self.name} segment layout stable: "
                        f"{len(segments)} persistent segments, "
                        f"{sum(s.num_rows for s in segments)} rows.",
                    )
                    return segments
            else:
                stable = 0
                prev_ids = ids

            if time.time() >= deadline:
                log.warning(
                    f"{self.name} timed out waiting for segment layout to stabilize, "
                    f"{len(segments)} persistent segments.",
                )
                return segments
            time.sleep(MILVUS_OPTIMIZE_POLL_INTERVAL)

    def _force_merge_to_convergence(self, deadline: float) -> list:
        """Force-merge until no further compaction is possible.

        The memory-aware force-merge policy converges to a target segment count
        that may be greater than one. Convergence is detected as a fixpoint: a
        force-merge round that no longer reduces the persistent segment count
        (or a ``compact()`` call that returns no job because nothing is eligible).
        """
        segments = self._wait_until_segments_stable(deadline)
        while True:
            prev_count = len(segments)
            try:
                compaction_id = self.client.compact(
                    self.collection_name,
                    target_size=MILVUS_COMPACT_TARGET_SIZE_MB,
                )
            except Exception as e:
                if hasattr(e, "code") and e.code().name == "PERMISSION_DENIED":
                    log.warning(f"{self.name} skip force merge due to compact permission denied.")
                    return segments
                raise

            if compaction_id is not None and compaction_id > 0:
                log.info(f"{self.name} force merge started compaction {compaction_id}.")
                self._wait_for_compaction(compaction_id, deadline)
            else:
                log.info(f"{self.name} force merge found nothing to compact (compaction_id={compaction_id}).")

            segments = self._wait_until_segments_stable(deadline)
            if len(segments) >= prev_count:
                log.info(
                    f"{self.name} force merge converged at {len(segments)} persistent segments; "
                    f"further compaction not possible.",
                )
                return segments
            if time.time() >= deadline:
                log.warning(f"{self.name} force merge stopped at deadline with {len(segments)} segments.")
                return segments

    def _wait_for_load_complete(self, deadline: float, data_size: int | None):
        """Wait until every persistent segment is loaded and rows are covered.

        Ensures the query view is search-ready: all persistent (non-L0) segments
        appear in the loaded segment set (i.e. unloadedSealedSegmentNum == 0) and
        the loaded rows cover the expected row count.
        """
        self.client.refresh_load(self.collection_name)
        self._wait_for_index(deadline)
        while True:
            persistent = self._persistent_data_segments()
            persistent_ids = frozenset(s.segment_id for s in persistent)
            persistent_rows = sum(s.num_rows for s in persistent)
            expected_rows = data_size if data_size is not None else persistent_rows

            loaded = self.client.list_loaded_segments(self.collection_name)
            loaded_ids = frozenset(s.segment_id for s in loaded)
            loaded_rows = sum(s.num_rows for s in loaded)

            all_loaded = persistent_ids <= loaded_ids
            rows_covered = loaded_rows >= expected_rows and persistent_rows >= expected_rows
            if all_loaded and rows_covered:
                log.info(
                    f"{self.name} load complete: {len(loaded)} loaded segments, "
                    f"{loaded_rows} loaded rows, {len(persistent)} persistent segments.",
                )
                return

            if time.time() >= deadline:
                log.warning(
                    f"{self.name} timed out waiting for full load: "
                    f"persistent={len(persistent)} ({persistent_rows} rows), "
                    f"loaded={len(loaded)} ({loaded_rows} rows), "
                    f"unloaded_persistent={len(persistent_ids - loaded_ids)}.",
                )
                return
            time.sleep(MILVUS_OPTIMIZE_POLL_INTERVAL)

    def _optimize(self, data_size: int | None = None):
        log.info(f"{self.name} optimizing before search")
        deadline = time.time() + MILVUS_OPTIMIZE_DEADLINE_SECONDS
        try:
            self.client.flush(self.collection_name)

            if self.case_config.is_gpu_index:
                log.debug("skip force merge compaction for gpu index type.")
                self._wait_for_index(deadline)
            else:
                self._force_merge_to_convergence(deadline)
                log.info(f"{self.name} force merge compaction completed.")

            self._wait_for_load_complete(deadline, data_size)
        except Exception as e:
            log.warning(f"{self.name} optimize error: {e}")
            raise e from None

    def optimize(self, data_size: int | None = None):
        assert self.client, "Please call self.init() before"
        log.info(
            f"before {self.client.list_loaded_segments(self.collection_name)} "
            f"{self.client.list_indexes(self.collection_name, self._vector_field)}",
        )
        self._optimize(data_size=data_size)
        log.info(
            f"after {self.client.list_loaded_segments(self.collection_name)} "
            f"{self.client.list_indexes(self.collection_name, self._vector_field)}",
        )
        if self.memory_monitor_impl is not None:
            # The collection is merged and fully loaded here, so the loaded-bytes
            # counters attribute all of it to this collection's final index.
            self.memory_monitor_impl.log_index_memory(row_count=data_size)

    def need_normalize_cosine(self) -> bool:
        """Wheather this database need to normalize dataset to support COSINE"""
        if self.case_config.is_gpu_index:
            log.info("current gpu_index only supports IP / L2, cosine dataset need normalize.")
            return True
        if self.case_config.is_svs_index:
            log.info("current SVS indexes only supports IP / L2, cosine dataset need normalize.")
            return True

        return False

    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception]:
        """Insert embeddings into Milvus. should call self.init() first"""
        assert self.client is not None
        assert len(embeddings) == len(metadata)
        insert_count = 0
        try:
            for batch_start_offset in range(0, len(embeddings), self.batch_size):
                batch_end_offset = min(batch_start_offset + self.batch_size, len(embeddings))
                batch_data = []
                for i in range(batch_start_offset, batch_end_offset):
                    row = {
                        self._primary_field: metadata[i],
                        self._scalar_id_field: metadata[i],
                        self._vector_field: embeddings[i],
                    }
                    if self.with_scalar_labels:
                        row[self._scalar_label_field] = labels_data[i]
                    batch_data.append(row)
                res = self.client.insert(self.collection_name, batch_data)
                insert_count += res["insert_count"]
        except MilvusException as e:
            log.info(f"Failed to insert data: {e}")
            return insert_count, e
        return insert_count, None

    def prepare_filter(self, filters: Filter):
        if filters.type == FilterOp.NonFilter:
            self.expr = ""
        elif filters.type == FilterOp.NumGE:
            self.expr = f"{self._scalar_id_field} >= {filters.int_value}"
        elif filters.type == FilterOp.StrEqual:
            self.expr = f"{self._scalar_label_field} == '{filters.label_value}'"
        else:
            msg = f"Not support Filter for Milvus - {filters}"
            raise ValueError(msg)

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        config_overwrite: dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> list[int]:
        """Perform a search on a query embedding and return results."""
        assert self.client is not None

        search_params = self.case_config.search_param()
        params = search_params.get("params", {})
        if config_overwrite:
            params = {**params, **config_overwrite}
        search_params["params"] = self.case_config.adjust_search_params(params)

        res = self.client.search(
            collection_name=self.collection_name,
            data=[query],
            anns_field=self._vector_field,
            search_params=search_params,
            limit=k,
            filter=self.expr,
        )

        return [result[self._primary_field] for result in res[0]]
