import concurrent.futures
import itertools
import logging
import math
import multiprocessing as mp
import time
import traceback

import numpy as np

from vectordb_bench.backend.dataset import DatasetManager
from vectordb_bench.backend.filter import Filter, non_filter

from ... import config
from ...metric import calc_ndcg, calc_recall, get_ideal_dcg
from ...models import LoadTimeoutError
from .. import utils
from ..clients import api
from ..clients.api import CalibrationType

NUM_PER_BATCH = config.NUM_PER_BATCH
LOAD_MAX_TRY_COUNT = config.LOAD_MAX_TRY_COUNT

log = logging.getLogger(__name__)


class SerialInsertRunner:
    def __init__(
        self,
        db: api.VectorDB,
        dataset: DatasetManager,
        normalize: bool,
        filters: Filter = non_filter,
        timeout: float | None = None,
    ):
        self.timeout = timeout if isinstance(timeout, int | float) else None
        self.dataset = dataset
        self.db = db
        self.normalize = normalize
        self.filters = filters

    def endless_insert_data(self, all_embeddings: list, all_metadata: list, left_id: int = 0) -> int:
        with self.db.init():
            # unique id for endlessness insertion
            all_metadata = [i + left_id for i in all_metadata]

            num_batches = math.ceil(len(all_embeddings) / NUM_PER_BATCH)
            log.info(
                f"({mp.current_process().name:16}) Start inserting {len(all_embeddings)} "
                f"embeddings in batch {NUM_PER_BATCH}"
            )
            count = 0
            for batch_id in range(num_batches):
                retry_count = 0
                already_insert_count = 0
                metadata = all_metadata[batch_id * NUM_PER_BATCH : (batch_id + 1) * NUM_PER_BATCH]
                embeddings = all_embeddings[batch_id * NUM_PER_BATCH : (batch_id + 1) * NUM_PER_BATCH]

                log.debug(
                    f"({mp.current_process().name:16}) batch [{batch_id:3}/{num_batches}], "
                    f"Start inserting {len(metadata)} embeddings"
                )
                while retry_count < LOAD_MAX_TRY_COUNT:
                    insert_count, error = self.db.insert_embeddings(
                        embeddings=embeddings[already_insert_count:],
                        metadata=metadata[already_insert_count:],
                    )
                    already_insert_count += insert_count
                    if error is not None:
                        retry_count += 1
                        time.sleep(10)

                        log.info(f"Failed to insert data, try {retry_count} time")
                        if retry_count >= LOAD_MAX_TRY_COUNT:
                            raise error
                    else:
                        break
                log.debug(
                    f"({mp.current_process().name:16}) batch [{batch_id:3}/{num_batches}], "
                    f"Finish inserting {len(metadata)} embeddings"
                )

                assert already_insert_count == len(metadata)
                count += already_insert_count
            log.info(
                f"({mp.current_process().name:16}) Finish inserting {len(all_embeddings)} embeddings in "
                f"batch {NUM_PER_BATCH}"
            )
        return count

    def run_endlessness(self) -> int:
        """run forever util DB raises exception or crash"""
        # datasets for load tests are quite small, can fit into memory
        # only 1 file
        data_df = next(iter(self.dataset))
        all_embeddings, all_metadata = (
            np.stack(data_df[self.dataset.data.train_vector_field]).tolist(),
            data_df[self.dataset.data.train_id_field].tolist(),
        )

        start_time = time.perf_counter()
        max_load_count, times = 0, 0
        try:
            while time.perf_counter() - start_time < self.timeout:
                count = self.endless_insert_data(
                    all_embeddings,
                    all_metadata,
                    left_id=max_load_count,
                )
                max_load_count += count
                times += 1
                log.info(
                    f"Loaded {times} entire dataset, current max load counts={utils.numerize(max_load_count)}, "
                    f"{max_load_count}"
                )
        except Exception as e:
            log.info(
                f"Capacity case load reach limit, insertion counts={utils.numerize(max_load_count)}, "
                f"{max_load_count}, err={e}"
            )
            traceback.print_exc()
            return max_load_count
        else:
            raise LoadTimeoutError(self.timeout)


class SerialSearchRunner:
    def __init__(
        self,
        db: api.VectorDB,
        test_data: list[list[float]],
        ground_truth: list[list[int]],
        db_case_config: api.DBCaseConfig,
        k: int = 100,
        filters: Filter = non_filter,
    ):
        self.db = db
        self.k = k
        self.filters = filters
        self.db_case_config = db_case_config

        if isinstance(test_data[0], np.ndarray):
            self.test_data = [query.tolist() for query in test_data]
        else:
            self.test_data = test_data
        self.ground_truth = ground_truth

    def _get_db_search_res(self, emb: list[float], retry_idx: int = 0, config_overwrite: dict[str, int] | None = None) -> list[int]:
        try:
            results = self.db.search_embedding(emb, self.k, config_overwrite=config_overwrite)
        except Exception as e:
            log.warning(f"Serial search failed, retry_idx={retry_idx}, Exception: {e}")
            if retry_idx < config.MAX_SEARCH_RETRY:
                return self._get_db_search_res(emb=emb, retry_idx=retry_idx + 1, config_overwrite=config_overwrite)

            msg = f"Serial search failed and retried more than {config.MAX_SEARCH_RETRY} times"
            raise RuntimeError(msg) from e

        return results

    def _calibrate(
        self,
        test_data: list,
        ground_truth: list[list[int]],
        calibration_param: str,
        min_value: int,
        recall: float,
        max_value: int = 1000,
        extra_params: dict[str, tuple[CalibrationType, float | int]] | None = None,
    ) -> tuple[int, float, float]:
        """Calibrate search for a given recall target.

        Args:
            extra_params: optional per-step overrides applied alongside the calibration param.
                Each entry maps a param name to (CalibrationType, value).
                MULTIPLIER entries are computed as int(current * value) at every binary-search step.
                ABSOLUTE entries are applied as-is at every step.

        Returns:
            (calibrated_value, achieved_recall, avg_latency) where avg_latency is the
            average query latency measured at the final calibrated parameter value.
        """
        if min_value > max_value:
            raise ValueError(
                f"{min_value=} cannot be greater than {max_value=}"
            )
        lower_bound = min_value
        upper_bound = max_value
        lower_bound_visited = False
        upper_bound_visited = False
        current = (lower_bound + upper_bound) // 2
        previous = current
        current_recall = 0
        current_avg_latency = 0.0
        previous_avg_latency = 0.0
        while True:
            previous_recall = current_recall
            previous_avg_latency = current_avg_latency
            config_overwrite = {calibration_param: current}
            if extra_params:
                for param_name, (param_type, param_value) in extra_params.items():
                    if param_type == CalibrationType.MULTIPLIER:
                        config_overwrite[param_name] = int(current * param_value)
                    else:
                        config_overwrite[param_name] = param_value
            recalls = []
            latencies = []
            for idx, emb in enumerate(test_data):
                s = time.perf_counter()
                results = self._get_db_search_res(emb, config_overwrite=config_overwrite)
                latencies.append(time.perf_counter() - s)
                recalls.append(calc_recall(self.k, ground_truth[idx][: self.k], results))
            current_recall = float(np.mean(recalls))
            current_avg_latency = float(np.mean(latencies))
            if np.isclose(current_recall, recall):
                return current, current_recall, current_avg_latency
            if current_recall > recall:
                upper_bound = current
                upper_bound_visited = True
            else:
                lower_bound = current
                lower_bound_visited = True
            next_value = (lower_bound + upper_bound) // 2
            if (
                (lower_bound_visited and next_value == lower_bound)
                or (upper_bound_visited and next_value == upper_bound)
            ):
                if abs(previous_recall - recall) < abs(current_recall - recall):
                    return previous, previous_recall, previous_avg_latency
                else:
                    return current, current_recall, current_avg_latency
            previous = current
            current = next_value

    @staticmethod
    def _generate_extra_param_combos(
        extra_params_spec: dict[str, tuple[CalibrationType, tuple[float | int, ...]]] | None,
    ) -> list[dict[str, tuple[CalibrationType, float | int]] | None]:
        """Expand a per-parameter value spec into a flat list of single-value combinations.

        Each entry in the spec maps a parameter name to (CalibrationType, (v1, v2, ...)).
        The returned list contains one dict per Cartesian-product combination, where each
        dict maps a parameter name to (CalibrationType, single_value).

        Returns ``[None]`` when *extra_params_spec* is empty or ``None``, meaning "run
        calibration once with no extra parameter overrides".
        """
        if not extra_params_spec:
            return [None]

        param_names = list(extra_params_spec.keys())
        per_param_choices: list[list[tuple[CalibrationType, float | int]]] = []
        for name in param_names:
            param_type, values = extra_params_spec[name]
            per_param_choices.append([(param_type, v) for v in values])

        combos = []
        for prod in itertools.product(*per_param_choices):
            combo: dict[str, tuple[CalibrationType, float | int]] = {
                name: prod[i] for i, name in enumerate(param_names)
            }
            combos.append(combo)
        return combos

    @staticmethod
    def _resolve_extra_params(
        calibration_param: str,
        calibrated_value: int,
        extra_params: dict[str, tuple[CalibrationType, float | int]] | None,
    ) -> dict[str, int | float]:
        """Build the final config_overwrite dict for a calibrated value + extra params combo.

        MULTIPLIER entries are computed as ``int(calibrated_value * multiplier)``.
        ABSOLUTE entries are used as-is.
        """
        result: dict[str, int | float] = {calibration_param: calibrated_value}
        if extra_params:
            for param_name, (param_type, param_value) in extra_params.items():
                if param_type == CalibrationType.MULTIPLIER:
                    result[param_name] = int(calibrated_value * param_value)
                else:
                    result[param_name] = param_value
        return result

    def search(self, args: tuple[list, list[list[int]]]) -> tuple[float, float, float, float]:
        log.info(f"{mp.current_process().name:14} start search the entire test_data to get recall and latency")
        with self.db.init():
            self.db.prepare_filter(self.filters)
            test_data, ground_truth = args
            ideal_dcg = get_ideal_dcg(self.k)

            log.debug(f"test dataset size: {len(test_data)}")
            log.debug(f"ground truth size: {len(ground_truth)}")

            config_overwrite: dict | None = None

            if (
                ground_truth is not None
                and (calibration_target := self.db_case_config.search_param()["params"]["calibration_target"]) is not None
            ):
                calibration_param = self.db_case_config.search_param()["params"]["calibration_param"]
                calibration_limit = self.db_case_config.search_param()["params"]["calibration_limit"]
                extra_params_spec = getattr(self.db_case_config, "calibration_extra_params", None)
                combos = self._generate_extra_param_combos(extra_params_spec)

                best_avg_latency = math.inf
                for combo_idx, combo in enumerate(combos):
                    log.info(
                        f"{mp.current_process().name:14} calibrating combo {combo_idx + 1}/{len(combos)}: "
                        f"{calibration_param=!s} to {calibration_target=} ({calibration_limit=})"
                        + (f", extra={combo}" if combo else "")
                    )
                    value, recall, avg_latency = self._calibrate(
                        test_data, ground_truth, calibration_param, self.k,
                        calibration_target, calibration_limit, extra_params=combo,
                    )
                    resolved = self._resolve_extra_params(calibration_param, value, combo)
                    log.info(
                        f"{mp.current_process().name:14} combo {combo_idx + 1}/{len(combos)}: "
                        f"calibrated to {recall=!s} at {value}, {avg_latency=:.4f}, params={resolved}"
                    )
                    if avg_latency < best_avg_latency:
                        best_avg_latency = avg_latency
                        config_overwrite = resolved

            latencies, recalls_list, ndcgs_list = [], [], []
            for idx, emb in enumerate(test_data):
                s = time.perf_counter()
                try:
                    results = self._get_db_search_res(emb, config_overwrite=config_overwrite)
                except Exception as e:
                    log.warning(f"VectorDB search_embedding error: {e}")
                    raise e from None

                latencies.append(time.perf_counter() - s)

                if ground_truth is not None:
                    gt = ground_truth[idx]
                    recalls_list.append(calc_recall(self.k, gt[: self.k], results))
                    ndcgs_list.append(calc_ndcg(gt[: self.k], results, ideal_dcg))
                else:
                    recalls_list.append(0)
                    ndcgs_list.append(0)

                if len(latencies) % 100 == 0:
                    log.debug(
                        f"({mp.current_process().name:14}) search_count={len(latencies):3}, "
                        f"latest_latency={latencies[-1]}, latest recall={recalls_list[-1]}"
                    )

        avg_latency = round(np.mean(latencies), 4)
        avg_recall = round(np.mean(recalls_list), 4)
        avg_ndcg = round(np.mean(ndcgs_list), 4)
        cost = round(np.sum(latencies), 4)
        p99 = round(np.percentile(latencies, 99), 4)
        p95 = round(np.percentile(latencies, 95), 4)
        log.info(
            f"{mp.current_process().name:14} search entire test_data: "
            f"cost={cost}s, "
            f"queries={len(latencies)}, "
            f"avg_recall={avg_recall}, "
            f"avg_ndcg={avg_ndcg}, "
            f"avg_latency={avg_latency}, "
            f"p99={p99}, "
            f"p95={p95}"
        )
        return (avg_recall, avg_ndcg, p99, p95, config_overwrite)

    def _run_in_subprocess(self) -> tuple[float, float, float, float, dict | None]:
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.search, (self.test_data, self.ground_truth))
            return future.result()

    @utils.time_it
    def run(self) -> tuple[float, float, float, float]:
        log.info(f"{mp.current_process().name:14} start serial search")
        if self.test_data is None:
            msg = "empty test_data"
            raise RuntimeError(msg)

        return self._run_in_subprocess()

    @utils.time_it
    def run_with_cost(self) -> tuple[tuple[float, float, float, float], float]:
        """
        Search all test data in serial.
        Returns:
            tuple[tuple[float, float, float, float], float]: (avg_recall, avg_ndcg, p99_latency, p95_latency), cost
        """
        log.info(f"{mp.current_process().name:14} start serial search")
        if self.test_data is None:
            msg = "empty test_data"
            raise RuntimeError(msg)

        return self._run_in_subprocess()
