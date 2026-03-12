from typing import Literal

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType


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

    def parse_metric(self) -> str:
        if not self.metric_type:
            return ""
        return self.metric_type.value


class RedisHNSWConfig(RedisIndexConfig, DBCaseConfig):
    M: int
    efConstruction: int
    ef: int | None = None
    filtering_batch_size: int | None = None
    index: IndexType = IndexType.HNSW
    calibration_target: float | None = None
    calibration_param: Literal["ef", "filtering_batch_size"] = "ef"
    use_float16: bool = False

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {"M": self.M, "efConstruction": self.efConstruction},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "params": {
                "ef": self.ef,
                "calibration_target": self.calibration_target,
                "calibration_param": self.calibration_param,
                "filtering_batch_size": self.filtering_batch_size,
            },
        }
