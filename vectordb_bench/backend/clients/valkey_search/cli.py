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
from .config import ValkeySearchHNSWConfig, ValkeySearchSVSConfig


class ValkeySearchTypedDict(TypedDict):
    host: Annotated[str, click.option("--host", type=str, help="Valkey host", required=True)]
    password: Annotated[str, click.option("--password", type=str, help="Valkey password")]
    port: Annotated[int, click.option("--port", type=int, default=6379, help="Valkey port")]


class ValkeySearchHNSWTypedDict(CommonTypedDict, ValkeySearchTypedDict, HNSWFlavor2):
    data_type: Annotated[
        str,
        click.option(
            "--data-type",
            type=click.Choice(["FLOAT32", "FLOAT16", "BFLOAT16"]),
            default="FLOAT32",
            help="Vector TYPE of the HNSW index; stored and query vectors use it",
        ),
    ]


class ValkeySearchSVSTypedDict(CommonTypedDict, ValkeySearchTypedDict):
    graph_max_degree: Annotated[
        int,
        click.option(
            "--graph-max-degree",
            type=int,
            default=64,
            help="SVS graph max degree",
        ),
    ]
    construction_window_size: Annotated[
        int,
        click.option(
            "--construction-window-size",
            type=int,
            default=128,
            help="SVS construction window size",
        ),
    ]
    search_window_size: Annotated[
        int,
        click.option(
            "--search-window-size",
            type=int,
            default=10,
            help="SVS search window size",
        ),
    ]
    alpha: Annotated[
        float,
        click.option(
            "--alpha",
            type=float,
            default=1.2,
            help="SVS alpha parameter",
        ),
    ]
    svs_algorithm_name: Annotated[
        str,
        click.option(
            "--svs-algorithm-name",
            type=str,
            default="SVS",
            help="FT.CREATE algorithm name (SVS_VAMANA for the SVS C API build)",
        ),
    ]
    svs_runtime_param: Annotated[
        str,
        click.option(
            "--svs-runtime-param",
            type=str,
            default="SEARCH_WINDOW_SIZE",
            help="KNN query-time search window parameter (EF_RUNTIME for the SVS C API build)",
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(ValkeySearchHNSWTypedDict)
def ValkeySearchHNSW(**parameters: Unpack[ValkeySearchHNSWTypedDict]):
    """Run ValkeySearch HNSW benchmark."""
    from .config import ValkeySearchConfig

    run(
        db=DB.ValkeySearchHNSW,
        db_config=ValkeySearchConfig(
            db_label=parameters["db_label"],
            password=SecretStr(parameters["password"]) if parameters["password"] else None,
            host=SecretStr(parameters["host"]),
            port=parameters["port"],
        ),
        db_case_config=ValkeySearchHNSWConfig(
            M=parameters["m"],
            efConstruction=parameters["ef_construction"],
            ef=parameters["ef_runtime"],
            data_type=parameters["data_type"],
        ),
        **parameters,
    )


@cli.command()
@click_parameter_decorators_from_typed_dict(ValkeySearchSVSTypedDict)
def ValkeySearchSVS(**parameters: Unpack[ValkeySearchSVSTypedDict]):
    """Run ValkeySearch SVS benchmark."""
    from .config import ValkeySearchConfig

    run(
        db=DB.ValkeySearchSVS,
        db_config=ValkeySearchConfig(
            db_label=parameters["db_label"],
            password=SecretStr(parameters["password"]) if parameters["password"] else None,
            host=SecretStr(parameters["host"]),
            port=parameters["port"],
        ),
        db_case_config=ValkeySearchSVSConfig(
            graph_max_degree=parameters["graph_max_degree"],
            construction_window_size=parameters["construction_window_size"],
            search_window_size=parameters["search_window_size"],
            alpha=parameters["alpha"],
            algorithm_name=parameters["svs_algorithm_name"],
            runtime_param=parameters["svs_runtime_param"],
        ),
        **parameters,
    )
