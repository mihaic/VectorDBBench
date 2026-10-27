#!/usr/bin/env python3
"""
Non-pipelined loader for Cohere 1M dataset.

Supports HNSW, SVS (no compression), and SVS with LVQ or LeanVec compression.
Inserts vectors one-at-a-time to avoid pipeline deadlock with SVS buffer flushing.

Usage:
    python3 load_cohere_simple.py --algorithm hnsw --flush-db
    python3 load_cohere_simple.py --algorithm svs --flush-db
    python3 load_cohere_simple.py --algorithm svs --compression LVQ4X8 --flush-db
    python3 load_cohere_simple.py --algorithm svs --svs-algorithm-name SVS_VAMANA \
        --compression LEANVEC4X8 --reduce 192 --flush-db
"""
import argparse
import logging
import time
from pathlib import Path

import numpy as np
import redis
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

INDEX_NAME = "vdbbench"


def create_index(
    conn: redis.Redis,
    dim: int,
    algorithm: str,
    compression: str | None = None,
    reduce: int | None = None,
    training_threshold: int | None = None,
    svs_algorithm_name: str = "SVS",
):
    """Create index matching VectorDBBench schema."""
    try:
        conn.ft(INDEX_NAME).info()
        log.info(f"Index {INDEX_NAME} already exists")
        return
    except redis.exceptions.ResponseError:
        pass

    if algorithm == "hnsw":
        vector_kv = [
            "TYPE", "FLOAT32",
            "DIM", str(dim),
            "DISTANCE_METRIC", "COSINE",
            "M", "32",
            "EF_CONSTRUCTION", "128",
        ]
        cmd = [
            "FT.CREATE", INDEX_NAME,
            "ON", "HASH",
            "PREFIX", "1", f"{INDEX_NAME}:",
            "SCHEMA",
            "id", "TAG",
            "metadata", "NUMERIC",
            "vector", "VECTOR", "HNSW", str(len(vector_kv)),
        ] + vector_kv
        log.info(f"Creating HNSW index: dim={dim}, M=32, EF_CONSTRUCTION=128")

    elif algorithm == "svs":
        vector_kv = [
            "TYPE", "FLOAT32",
            "DIM", str(dim),
            "DISTANCE_METRIC", "COSINE",
            "GRAPH_MAX_DEGREE", "64",
            "CONSTRUCTION_WINDOW_SIZE", "128",
            "SEARCH_WINDOW_SIZE", "50",
        ]
        if compression and compression != "NONE":
            vector_kv.extend(["COMPRESSION", compression])
        # LeanVec reduced dimension count; required for LEANVEC compression
        if reduce is not None:
            vector_kv.extend(["REDUCE", str(reduce)])
        # SQ8/LVQ/LEANVEC stay uncompressed until this many vectors
        if training_threshold is not None:
            vector_kv.extend(["TRAINING_THRESHOLD", str(training_threshold)])
        cmd = [
            "FT.CREATE", INDEX_NAME,
            "ON", "HASH",
            "PREFIX", "1", f"{INDEX_NAME}:",
            "SCHEMA",
            "id", "TAG",
            "metadata", "NUMERIC",
            "vector", "VECTOR", svs_algorithm_name, str(len(vector_kv)),
        ] + vector_kv
        comp_str = compression or "NONE"
        log.info(
            f"Creating {svs_algorithm_name} index: dim={dim}, GRAPH_MAX_DEGREE=64, "
            f"COMPRESSION={comp_str}, REDUCE={reduce}, TRAINING_THRESHOLD={training_threshold}"
        )

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    conn.execute_command(*cmd)
    log.info("Index created successfully")


def load_dataset(conn: redis.Redis, parquet_path: Path, max_vectors: int | None = None):
    """Load vectors from parquet file using simple one-at-a-time insertion."""
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    log.info(f"Loading from {parquet_path} ({parquet_path.stat().st_size / (1024**3):.2f} GB)")
    log.info("Using non-pipelined insertion (compatible with synchronous flush)")

    start_time = time.time()
    total_inserted = 0

    parquet_file = pq.ParquetFile(parquet_path)
    log.info(f"Reading parquet with {parquet_file.num_row_groups} row groups...")

    for batch in parquet_file.iter_batches(batch_size=1000):
        chunk = batch.to_pandas()
        vectors = np.stack(chunk['emb'].values)
        ids = chunk['id'].values

        for vec, vec_id in zip(vectors, ids):
            vec_bytes = vec.astype(np.float32).tobytes()
            key = f"{INDEX_NAME}:{vec_id}"

            # Non-pipelined: wait for each response
            conn.hset(key, mapping={
                "id": str(vec_id),
                "metadata": int(vec_id),
                "vector": vec_bytes,
            })

            total_inserted += 1

            if total_inserted % 1000 == 0:
                elapsed = time.time() - start_time
                rate = total_inserted / elapsed
                log.info(f"  {total_inserted:,} vectors ({rate:.1f} vec/s)")

            if max_vectors and total_inserted >= max_vectors:
                log.info(f"Reached max_vectors limit: {max_vectors}")
                break

        if max_vectors and total_inserted >= max_vectors:
            break

    total_duration = time.time() - start_time
    avg_rate = total_inserted / total_duration

    log.info(f"✓ Complete: {total_inserted:,} vectors in {total_duration:.1f}s ({avg_rate:.1f} vec/s)")
    return total_inserted


def main():
    parser = argparse.ArgumentParser(description="Load Cohere 1M into Valkey (HNSW or SVS)")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6399)
    parser.add_argument(
        "--parquet",
        default="/tmp/vectordb_bench/dataset/cohere/cohere_medium_1m/shuffle_train.parquet",
    )
    parser.add_argument("--algorithm", choices=["hnsw", "svs"], default="svs")
    parser.add_argument("--compression", default=None, help="SVS compression: NONE, LVQ4X8, LEANVEC4X8, etc.")
    parser.add_argument("--reduce", type=int, help="LeanVec reduced dimension count (required for LEANVEC)")
    parser.add_argument("--training-threshold", type=int, help="Vectors before SQ8/LVQ/LEANVEC conversion")
    parser.add_argument(
        "--svs-algorithm-name",
        default="SVS",
        help="FT.CREATE algorithm name (SVS_VAMANA for the SVS C API build)",
    )
    parser.add_argument("--max-vectors", type=int, help="Stop after N vectors (for testing)")
    parser.add_argument("--flush-db", action="store_true")

    args = parser.parse_args()

    log.info(f"Connecting to Valkey at {args.host}:{args.port}")
    conn = redis.Redis(host=args.host, port=args.port, db=0)

    if args.flush_db:
        log.info("Flushing database...")
        conn.flushall()

    # Detect dimension
    parquet_path = Path(args.parquet)
    parquet_file = pq.ParquetFile(parquet_path)
    first_batch = parquet_file.read_row_group(0, columns=['emb']).to_pandas()
    dim = len(first_batch['emb'].values[0])
    log.info(f"Detected dimension: {dim}")

    create_index(
        conn,
        dim,
        args.algorithm,
        args.compression,
        args.reduce,
        args.training_threshold,
        args.svs_algorithm_name,
    )
    load_dataset(conn, parquet_path, args.max_vectors)

    dbsize = conn.dbsize()
    log.info(f"Final DBSIZE: {dbsize:,} keys")


if __name__ == "__main__":
    main()
