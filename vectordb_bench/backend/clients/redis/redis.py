import logging
from contextlib import contextmanager
from typing import Any

import numpy as np
import redis
from redis.commands.search.field import NumericField, TagField, VectorField
try:
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
except ImportError:
    from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from vectordb_bench.backend.filter import Filter, FilterOp
from ..api import DBCaseConfig, VectorDB

log = logging.getLogger(__name__)
INDEX_NAME = "index"  # Vector Index Name


class Redis(VectorDB):

    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig,
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = INDEX_NAME
        self.with_scalar_labels = with_scalar_labels
        self._filter = "*"
        self._vector_field = "vector"
        self._label_field = "label"
        self._numeric_field = "metadata"

        # Create a redis connection, if db has password configured, add it to the connection here and in init():
        password = self.db_config["password"]
        conn = redis.Redis(
            host=self.db_config["host"],
            port=self.db_config["port"],
            password=password,
            db=0,
        )

        if drop_old:
            try:
                conn.ft(INDEX_NAME).info()
                conn.ft(INDEX_NAME).dropindex()
            except redis.exceptions.ResponseError:
                drop_old = False
                log.info(f"Redis client drop_old collection: {self.collection_name}")

        self.make_index(dim, conn)
        conn.close()
        conn = None

    def make_index(self, vector_dimensions: int, conn: redis.Redis):
        try:
            # check to see if index exists
            conn.ft(INDEX_NAME).info()
        except Exception:
            schema = [
                NumericField(self._numeric_field),
                VectorField(
                    self._vector_field,  # Vector Field Name
                    "HNSW",  # Vector Index Type: FLAT or HNSW
                    {
                        "TYPE": "FLOAT32",  # FLOAT32 or FLOAT64
                        "DIM": vector_dimensions,  # Number of Vector Dimensions
                        "DISTANCE_METRIC": "COSINE",  # Vector Search Distance Metric
                        "M": self.case_config.index_param()["params"]["M"],
                        "EF_CONSTRUCTION": self.case_config.index_param()["params"]["efConstruction"],
                    },
                ),
            ]
            if self.with_scalar_labels:
                schema.append(TagField(self._label_field))

            definition = IndexDefinition(index_type=IndexType.HASH)

            rs = conn.ft(INDEX_NAME)
            rs.create_index(schema, definition=definition)

    @contextmanager
    def init(self) -> None:
        """create and destory connections to database.

        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
        """
        self.conn = redis.Redis(
            host=self.db_config["host"],
            port=self.db_config["port"],
            password=self.db_config["password"],
            db=0,
        )
        yield
        self.conn.close()
        self.conn = None

    def optimize(self, data_size: int | None = None):
        pass

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[int, Exception]:
        """Insert embeddings into the database.
        Should call self.init() first.
        """

        batch_size = 1000  # Adjust this as needed, but don't make too big
        try:
            with self.conn.pipeline(transaction=False) as pipe:
                for i, embedding in enumerate(embeddings):
                    ndarr_emb = np.array(embedding).astype(np.float32).tobytes()
                    mapping={
                        self._vector_field: ndarr_emb,
                        self._numeric_field: metadata[i],
                    }
                    if self.with_scalar_labels:
                        assert labels_data is not None
                        mapping[self._label_field] = labels_data[i]
                    pipe.hset(
                        metadata[i],
                        mapping=mapping,
                    )
                    # Execute the pipe so we don't keep too much in memory at once
                    if i % batch_size == 0:
                        pipe.execute()

                pipe.execute()
                result_len = i + 1
        except redis.exceptions.RedisError as e:
            return 0, e

        return result_len, None

    def prepare_filter(self, filters: Filter):
        if filters.type == FilterOp.NonFilter:
            self._filter = "*"
        elif filters.type == FilterOp.NumGE:
            self._filter = f"@{self._numeric_field}:[{filters.int_value} +inf]"
        elif filters.type == FilterOp.StrEqual:
            self._filter = f"@{self._label_field}:{{ {filters.label_value} }}"
        else:
            msg = f"Not support Filter for Redis - {filters}"
            raise ValueError(msg)

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        timeout: int | None = None,
        config_overwrite: dict[str, int] | None = None,
        **kwargs: Any,
    ) -> list[int]:
        assert self.conn is not None

        query_vector = np.array(query).astype(np.float32).tobytes()
        search_params = self.case_config.search_param()["params"]
        if config_overwrite is not None and "ef" in config_overwrite:
            ef_runtime = config_overwrite["ef"]
        else:
            ef_runtime = self.case_config.search_param()["params"]["ef"]
        if config_overwrite is not None and "filtering_batch_size" in config_overwrite:
            filtering_batch_size = config_overwrite["filtering_batch_size"]
        else:
            filtering_batch_size = search_params.get("filtering_batch_size")
        is_filtering = self._filter != "*"
        if is_filtering and filtering_batch_size is not None:
            filtering_params = f" HYBRID_POLICY BATCHES BATCH_SIZE {filtering_batch_size}"
        else:
            filtering_params = ""
        query_obj = (
            Query(f"{self._filter}=>[KNN {k} @{self._vector_field} $vec EF_RUNTIME {ef_runtime}{filtering_params}]")
            .paging(0, k)
        )
        query_params = {"vec": query_vector}
        res = self.conn.ft(INDEX_NAME).search(query_obj, query_params)
        return [int(doc["id"]) for doc in res.docs]
