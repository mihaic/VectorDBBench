"""VectorDBBench client for valkey-search (HNSW and SVS backends)."""
import logging
from contextlib import contextmanager
from typing import Any

import numpy as np
import redis
from redis.commands.search.query import Query

from ..api import DBCaseConfig, VectorDB

log = logging.getLogger(__name__)
INDEX_NAME = "vdbbench"


class ValkeySearch(VectorDB):
    """Valkey-search client supporting both HNSW and SVS algorithms."""

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig,
        drop_old: bool = False,
        extra_index_params: dict | None = None,
        **kwargs,
    ):
        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = INDEX_NAME
        self.dim = dim
        self.extra_index_params = extra_index_params or {}

        conn = redis.Redis(
            host=self.db_config["host"],
            port=self.db_config["port"],
            password=self.db_config.get("password"),
            db=0,
        )

        if drop_old:
            conn.flushall()
            log.info(f"Flushed all data for clean benchmark")

        self._make_index(dim, conn)
        conn.close()

    def _make_index(self, dim: int, conn: redis.Redis):
        """Create the FT index with HNSW or SVS algorithm."""
        try:
            conn.ft(INDEX_NAME).info()
            log.info(f"Index {INDEX_NAME} already exists")
            return
        except redis.exceptions.ResponseError:
            pass

        index_params = self.case_config.index_param()
        algorithm = index_params["algorithm"]
        params = index_params["params"]

        # Build the vector schema args as flat key-value pairs.
        # We use raw execute_command because redis-py's VectorField
        # only knows about HNSW and FLAT — it doesn't support SVS.
        vector_kv = [
            "TYPE", "FLOAT32",
            "DIM", str(dim),
            "DISTANCE_METRIC", "L2",
        ]
        for k, v in params.items():
            vector_kv.extend([str(k), str(v)])
        # Add extra index params (e.g., COMPRESSION LVQ4X8)
        for k, v in self.extra_index_params.items():
            vector_kv.extend([str(k), str(v)])

        cmd = [
            "FT.CREATE", INDEX_NAME,
            "ON", "HASH",
            "PREFIX", "1", f"{INDEX_NAME}:",
            "SCHEMA",
            "id", "TAG",
            "metadata", "NUMERIC",
            "vector", "VECTOR", algorithm, str(len(vector_kv)),
        ] + vector_kv

        conn.execute_command(*cmd)
        log.info(
            f"Created {algorithm} index: dim={dim}, params={params}"
        )

    @contextmanager
    def init(self):
        self.conn = redis.Redis(
            host=self.db_config["host"],
            port=self.db_config["port"],
            password=self.db_config.get("password"),
            db=0,
        )
        yield
        self.conn.close()
        self.conn = None

    def ready_to_search(self) -> bool:
        return True

    def optimize(self, data_size: int | None = None):
        """No optimization needed — valkey-search indexes online."""
        pass

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> tuple[int, Exception]:
        batch_size = 1000
        try:
            with self.conn.pipeline(transaction=False) as pipe:
                for i, embedding in enumerate(embeddings):
                    vec_bytes = np.array(embedding, dtype=np.float32).tobytes()
                    key = f"{INDEX_NAME}:{metadata[i]}"
                    pipe.hset(
                        key,
                        mapping={
                            "id": str(metadata[i]),
                            "metadata": metadata[i],
                            "vector": vec_bytes,
                        },
                    )
                    if (i + 1) % batch_size == 0:
                        pipe.execute()
                pipe.execute()
                result_len = len(embeddings)
        except Exception as e:
            return 0, e
        return result_len, None

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> list[int]:
        assert self.conn is not None

        query_vector = np.array(query, dtype=np.float32).tobytes()

        # Build KNN clause with algorithm-specific runtime params.
        search_params = self.case_config.search_param()
        algo = search_params.get("algorithm", "HNSW")
        params = search_params.get("params", {})

        knn_extra = ""
        if algo == "HNSW" and "EF_RUNTIME" in params:
            knn_extra = f" EF_RUNTIME {params['EF_RUNTIME']}"
        elif algo == "SVS" and params:
            (name, value), = params.items()
            knn_extra = f" {name} {value}"

        query_str = f"*=>[KNN {k} @vector $vec{knn_extra}]"

        query_obj = (
            Query(query_str)
            .return_fields("id")
            .paging(0, k)
            .dialect(2)
        )
        query_params = {"vec": query_vector}

        res = self.conn.ft(INDEX_NAME).search(query_obj, query_params)
        results = []
        for doc in res.docs:
            # doc.id is the Redis key (e.g. "vdbbench:42").
            # The "id" field stores the metadata ID as a string.
            try:
                results.append(int(doc["id"]))
            except (KeyError, ValueError):
                # Fallback: extract numeric ID from Redis key
                try:
                    results.append(int(doc.id.split(":")[-1]))
                except (ValueError, AttributeError):
                    pass
        return results
