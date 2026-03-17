from typing import Annotated, TypedDict, Unpack

import click
from pydantic import SecretStr

from ....cli.cli import (
    CommonTypedDict,
    HNSWFlavor2,
    cli,
    click_parameter_decorators_from_typed_dict,
    run,
)
from .. import DB
from .config import RedisHNSWConfig, SVS_VAMANA_COMPRESSION_OPTIONS


class RedisTypedDict(TypedDict):
    host: Annotated[str, click.option("--host", type=str, help="Db host", required=True)]
    password: Annotated[str, click.option("--password", type=str, help="Db password")]
    port: Annotated[int, click.option("--port", type=int, default=6379, help="Db Port")]
    use_float16: Annotated[
        bool,
        click.option(
            "--use-float16/--no-use-float16",
            is_flag=True,
            default=False,
            help="Store and query vectors as FLOAT16 instead of FLOAT32",
        ),
    ]
    filtering_batch_size: Annotated[
        int | None,
        click.option(
            "--filtering-batch-size",
            type=int,
            default=None,
            help="Batch size for hybrid filtering policy (HYBRID_POLICY BATCHES)",
        ),
    ]
    ssl: Annotated[
        bool,
        click.option(
            "--ssl/--no-ssl",
            is_flag=True,
            show_default=True,
            default=True,
            help="Enable or disable SSL for Redis",
        ),
    ]
    ssl_ca_certs: Annotated[
        str,
        click.option(
            "--ssl-ca-certs",
            show_default=True,
            help="Path to certificate authority file to use for SSL",
        ),
    ]
    cmd: Annotated[
        bool,
        click.option(
            "--cmd",
            is_flag=True,
            show_default=True,
            default=False,
            help="Cluster Mode Disabled (CMD) for Redis doesn't use Cluster conn",
        ),
    ]


class RedisHNSWTypedDict(CommonTypedDict, RedisTypedDict, HNSWFlavor2): ...


class RedisSVSVAMANATypedDict(CommonTypedDict, RedisTypedDict):
    graph_max_degree: Annotated[
        int,
        click.option(
            "--graph-max-degree",
            type=int,
            required=True,
            help="SVS-VAMANA GRAPH_MAX_DEGREE (equivalent to HNSW M)",
        ),
    ]
    construction_window_size: Annotated[
        int,
        click.option(
            "--construction-window-size",
            type=int,
            required=True,
            help="SVS-VAMANA CONSTRUCTION_WINDOW_SIZE (equivalent to HNSW EF_CONSTRUCTION)",
        ),
    ]
    search_window_size: Annotated[
        int | None,
        click.option(
            "--search-window-size",
            type=int,
            default=None,
            help="SVS-VAMANA SEARCH_WINDOW_SIZE (equivalent to HNSW EF_RUNTIME)",
        ),
    ]
    compression: Annotated[
        str | None,
        click.option(
            "--compression",
            type=click.Choice(SVS_VAMANA_COMPRESSION_OPTIONS, case_sensitive=True),
            default=None,
            help="SVS-VAMANA compression type (LeanVec4x8 or LVQ8)",
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(RedisHNSWTypedDict)
def Redis(**parameters: Unpack[RedisHNSWTypedDict]):
    from .config import RedisConfig

    run(
        db=DB.Redis,
        db_config=RedisConfig(
            db_label=parameters["db_label"],
            password=SecretStr(parameters["password"]) if parameters["password"] else None,
            host=SecretStr(parameters["host"]),
            port=parameters["port"],
            ssl=parameters["ssl"],
            ssl_ca_certs=parameters["ssl_ca_certs"],
            cmd=parameters["cmd"],
        ),
        db_case_config=RedisHNSWConfig(
            M=parameters["m"],
            efConstruction=parameters["ef_construction"],
            ef=parameters["ef_runtime"],
            filtering_batch_size=parameters["filtering_batch_size"],
            calibration_target=parameters["calibrate"],
            calibration_param=parameters["calibration_param"] or "ef",
            calibration_limit=parameters["calibration_limit"],
            use_float16=parameters["use_float16"],
        ),
        **parameters,
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(RedisSVSVAMANATypedDict)
def RedisSVSVAMANA(**parameters: Unpack[RedisSVSVAMANATypedDict]):
    from .config import RedisConfig, RedisSVSVAMANAConfig

    run(
        db=DB.Redis,
        db_config=RedisConfig(
            db_label=parameters["db_label"],
            password=SecretStr(parameters["password"]) if parameters["password"] else None,
            host=SecretStr(parameters["host"]),
            port=parameters["port"],
            ssl=parameters["ssl"],
            ssl_ca_certs=parameters["ssl_ca_certs"],
            cmd=parameters["cmd"],
        ),
        db_case_config=RedisSVSVAMANAConfig(
            graph_max_degree=parameters["graph_max_degree"],
            construction_window_size=parameters["construction_window_size"],
            search_window_size=parameters["search_window_size"],
            compression=parameters["compression"],
            filtering_batch_size=parameters["filtering_batch_size"],
            calibration_target=parameters["calibrate"],
            calibration_param=parameters["calibration_param"] or "search_window_size",
            calibration_limit=parameters["calibration_limit"],
            use_float16=parameters["use_float16"],
        ),
        **parameters,
    )
