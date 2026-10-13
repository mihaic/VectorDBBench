from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, MetricType


class ValkeySearchConfig(DBConfig):
    """Connection config for Valkey with valkey-search module."""
    password: SecretStr | None = None
    host: SecretStr
    port: int | None = None

    def to_dict(self) -> dict:
        return {
            "host": self.host.get_secret_value(),
            "port": self.port,
            "password": self.password.get_secret_value()
            if self.password is not None
            else None,
        }


class ValkeySearchHNSWConfig(BaseModel, DBCaseConfig):
    """HNSW index config for valkey-search."""
    M: int = 16
    efConstruction: int = 200
    ef: int = 10
    metric_type: MetricType | None = None

    def index_param(self) -> dict:
        return {
            "algorithm": "HNSW",
            "params": {
                "M": self.M,
                "EF_CONSTRUCTION": self.efConstruction,
            },
        }

    def search_param(self) -> dict:
        return {
            "algorithm": "HNSW",
            "params": {"EF_RUNTIME": self.ef},
        }


class ValkeySearchSVSConfig(BaseModel, DBCaseConfig):
    """SVS Vamana index config for valkey-search."""
    graph_max_degree: int = 64
    construction_window_size: int = 128
    search_window_size: int = 10
    alpha: float = 1.2
    # The SVS C API build of valkey-search uses ALGORITHM SVS_VAMANA and
    # takes the query-time search window as EF_RUNTIME.
    algorithm_name: str = "SVS"
    runtime_param: str = "SEARCH_WINDOW_SIZE"
    metric_type: MetricType | None = None

    def index_param(self) -> dict:
        return {
            "algorithm": self.algorithm_name,
            "params": {
                "GRAPH_MAX_DEGREE": self.graph_max_degree,
                "CONSTRUCTION_WINDOW_SIZE": self.construction_window_size,
                "SEARCH_WINDOW_SIZE": self.search_window_size,
                "ALPHA": self.alpha,
            },
        }

    def search_param(self) -> dict:
        return {
            "algorithm": "SVS",
            "params": {
                self.runtime_param: self.search_window_size,
            },
        }
