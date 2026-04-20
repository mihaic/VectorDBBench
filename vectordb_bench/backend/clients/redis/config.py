from typing import Literal

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType

SVS_VAMANA_COMPRESSION_OPTIONS = ["LeanVec4x8", "LVQ8"]


class RedisConfig(DBConfig):
    password: SecretStr | None = None
    host: SecretStr
    port: int | None = None

    def to_dict(self) -> dict:
        return {
            "host": self.host.get_secret_value(),
            "port": self.port,
            "password": self.password.get_secret_value() if self.password is not None else None,
        }


class RedisIndexConfig(BaseModel):
    """Base config for milvus"""

    metric_type: MetricType | None = None
    use_float16: bool = False
    filtering_batch_size: int | None = None
    calibration_target: float | None = None
    calibration_limit: int = 1000
    hybrid_policy: Literal["ADHOC_BF", "BATCHES"] = "BATCHES"

    def parse_metric(self) -> str:
        if not self.metric_type:
            return ""
        return self.metric_type.value


class RedisHNSWConfig(RedisIndexConfig, DBCaseConfig):
    M: int
    efConstruction: int
    ef: int | None = None
    index: IndexType = IndexType.HNSW
    calibration_param: Literal["ef", "filtering_batch_size"] = "ef"

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {"M": self.M, "EF_CONSTRUCTION": self.efConstruction},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "params": {
                "ef": self.ef,
                "calibration_target": self.calibration_target,
                "calibration_param": self.calibration_param,
                "calibration_limit": self.calibration_limit,
                "filtering_batch_size": self.filtering_batch_size,
                "hybrid_policy": self.hybrid_policy,
            },
        }

    def knn_runtime_param(self, config_overwrite: dict | None = None) -> str:
        ef = config_overwrite["ef"] if config_overwrite is not None and "ef" in config_overwrite else self.ef
        return f"EF_RUNTIME {ef}"


class RedisSVSVAMANAConfig(RedisIndexConfig, DBCaseConfig):
    graph_max_degree: int
    construction_window_size: int
    search_window_size: int | None = None
    compression: Literal["LeanVec4x8", "LVQ8"] | None = None
    index: IndexType = IndexType.SVS_VAMANA_REDIS
    calibration_param: Literal["search_window_size", "filtering_batch_size"] = "search_window_size"

    def index_param(self) -> dict:
        params: dict = {
            "GRAPH_MAX_DEGREE": self.graph_max_degree,
            "CONSTRUCTION_WINDOW_SIZE": self.construction_window_size,
        }
        if self.compression is not None:
            params["COMPRESSION"] = self.compression
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": params,
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "params": {
                "search_window_size": self.search_window_size,
                "calibration_target": self.calibration_target,
                "calibration_param": self.calibration_param,
                "calibration_limit": self.calibration_limit,
                "filtering_batch_size": self.filtering_batch_size,
                "hybrid_policy": self.hybrid_policy,
            },
        }

    def knn_runtime_param(self, config_overwrite: dict | None = None) -> str:
        sws = config_overwrite["search_window_size"] if config_overwrite is not None and "search_window_size" in config_overwrite else self.search_window_size
        return f"SEARCH_WINDOW_SIZE {sws}"
