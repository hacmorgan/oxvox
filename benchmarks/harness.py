"""
Timing and correctness harness for the oxvox neighbour-search benchmark

How a measurement is taken:

- One forked worker process per `(competitor, radius)` block. The worker builds the
  index once and then answers every `(k, Q)` combination in the block against it, so
  the build is timed once and never charged to a query, and a crash or a runaway
  configuration takes down only that block.
- Each timed configuration gets one warm-up query followed by the median of three
  timed queries; configurations whose warm-up alone is slower than
  `full_repeat_seconds` are timed once, which is recorded in the run's `repeats`.
- Configurations are visited in ascending query count, and the next query count is
  skipped (with a `skipped` marker giving the projection) when scaling the previous
  measurement to it would exceed `max_run_seconds`. Combinations whose output arrays
  would not fit in `max_output_gb` are skipped the same way.
- Correctness is checked on a fixed subsample of the queries against the reference
  competitor (scipy): the exact methods must reproduce its distances, and the
  approximate one gets a recall number. Distance ties make index comparison ambiguous,
  so index sets are only compared on rows whose distances are all distinct.
- Nothing in the parent process ever calls the Rust extension. The first parallel
  operation in a process starts rayon's global thread pool, and a process forked after
  that inherits the pool's bookkeeping without its threads, so the child's first
  parallel build waits forever on workers that do not exist. Every Rust call therefore
  happens in a worker, including the radius calibration, through `call_in_worker`.
- Peak RSS comes from `resource.getrusage` inside the worker, so it covers that
  index's build and queries and nothing else. The worker inherits the parent's pages
  (the dataset among them), so `peak_rss_mb` starts a couple of hundred megabytes above
  zero and `added_rss_mb`, the growth over the worker's starting point, is the number
  to compare between competitors.
"""

import json
import logging
import multiprocessing
import platform
import queue as queue_module
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import numpy.typing as npt

from benchmarks.competitors import COMPETITORS, REFERENCE_COMPETITOR

logger = logging.getLogger("benchmarks.harness")

# Number of query points the correctness check uses, per (competitor, radius, k)
NUM_CHECK_QUERIES = 2048

# Distance tolerance for calling two neighbour answers the same, in metres. The
# implementations accumulate squared distances in float32 over coordinates up to a few
# hundred metres, so agreement to ~1e-3 m is as much as can be asked of them
DISTANCE_TOLERANCE = 1e-3

# Sentinel a worker sends when it has finished its block
_WORKER_DONE = "__worker_done__"

# Datasets shared with forked workers by key, so that no array is ever pickled into a
# child process (fork gives the child the parent's pages copy-on-write instead)
_SHARED_ARRAYS: dict[str, npt.NDArray[np.float32]] = {}


class RadiusPlan(TypedDict):
    """
    One column of the benchmark grid: a search radius and what it was chosen for

    `density_target` is the expected number of points inside a search sphere on the
    uniform cloud of the same size, which is how the synthetic radii are picked; it is
    0 for a radius chosen directly (the real scans), where `chosen_for` says so
    """

    radius: float
    density_target: float
    chosen_for: str


class MeasurementOptions(TypedDict):
    """
    Limits that keep a full-grid run inside a sane wall-clock and memory budget
    """

    max_run_seconds: float
    max_build_seconds: float
    full_repeat_seconds: float
    max_output_gb: float
    num_check_queries: int
    inactivity_timeout_seconds: float


class RunRecord(TypedDict, total=False):
    """
    One row of the results: one competitor answering one workload at one grid point

    The `check_*` and `spec` fields only travel from a worker to the parent, which
    folds them into correctness records and the competitor table; they never reach the
    results JSON.
    """

    competitor: str
    workload: str
    num_neighbours: int
    num_queries: int
    radius: float
    density_target: float
    status: str
    detail: str
    build_seconds: float
    query_seconds: float
    query_seconds_samples: list[float]
    repeats: int
    peak_rss_mb: float
    added_rss_mb: float
    index_stats: dict[str, float] | None
    correctness: dict[str, float]
    spec: dict[str, Any]
    check_indices: npt.NDArray[np.int32]
    check_distances: npt.NDArray[np.float32]
    check_counts: npt.NDArray[np.int64]


def default_options(
    max_run_seconds: float = 60.0,
    max_build_seconds: float = 300.0,
    full_repeat_seconds: float = 3.0,
    max_output_gb: float = 6.0,
    num_check_queries: int = NUM_CHECK_QUERIES,
) -> MeasurementOptions:
    """
    Measurement limits, with the defaults the committed results were produced with

    Args:
        max_run_seconds: A configuration projected to take longer than this is skipped
        max_build_seconds: A whole block is skipped when its index build is projected
            to take longer than this (only the graph backend, whose build is a full
            self-query, ever gets near it)
        full_repeat_seconds: Configurations whose warm-up is slower than this are timed
            once instead of three times
        max_output_gb: Configurations whose output arrays would exceed this are skipped
        num_check_queries: Query points used for the correctness check

    Returns:
        The options dict passed to `benchmark_group`
    """
    return MeasurementOptions(
        max_run_seconds=max_run_seconds,
        max_build_seconds=max_build_seconds,
        full_repeat_seconds=full_repeat_seconds,
        max_output_gb=max_output_gb,
        num_check_queries=num_check_queries,
        # A worker gets this long to produce its next record before being killed; the
        # generous margin over max_run_seconds covers index builds, which are not
        # projected in advance
        inactivity_timeout_seconds=max(600.0, max_run_seconds * 8.0),
    )


def environment_meta() -> dict[str, Any]:
    """
    Machine, library and oxvox build details, recorded with every results file

    Returns:
        Dict of metadata, all of it safe to commit (no paths, no host identity)
    """
    import oxvox.nns

    versions: dict[str, str] = {}
    for module_name in ("numpy", "scipy", "open3d", "plotly"):
        try:
            versions[module_name] = __import__(module_name).__version__
        except ImportError:
            versions[module_name] = "not installed"

    try:
        import importlib.metadata

        oxvox_version = importlib.metadata.version("oxvox")
    except Exception:
        oxvox_version = "unknown"

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        commit = "unknown"

    return {
        "date": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "cpu_count": multiprocessing.cpu_count(),
        "platform": platform.platform(terse=True),
        "python": platform.python_version(),
        "oxvox_version": oxvox_version,
        "oxvox_commit": commit,
        "oxvox_methods": list(oxvox.nns.available_methods()),
        "library_versions": versions,
    }


def compare_neighbours(
    reference: tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]],
    candidate: tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]],
    tolerance: float = DISTANCE_TOLERANCE,
) -> dict[str, float]:
    """
    Compare one competitor's neighbours against the reference implementation's

    Ties are the difficulty: when two search points sit at the same distance from a
    query, every implementation is free to return either, so indices can differ while
    both answers are correct. Distances cannot, so the primary measure is positional
    agreement of the sorted distance vectors; index sets are only compared on rows
    whose distances are all distinct by more than the tolerance

    Args:
        reference: `(indices, distances)` from the reference competitor
        candidate: `(indices, distances)` from the competitor under test
        tolerance: Distance difference, in metres, still counted as agreement

    Returns:
        Dict with `positional_recall` (1.0 when every reference neighbour was matched
        at the same sorted position), `distance_recall` (the same thing allowing the
        candidate to have skipped neighbours, which only an approximate method does),
        `max_distance_error` over matched positions, `index_agreement` over
        unambiguous rows, and the row counts behind all of them.

        One caveat on `index_agreement`: a search point just beyond the k-th
        neighbour, at the same distance as it, makes membership itself a tie-break, and
        that cannot be detected from the returned k neighbours alone. Rows are only
        excluded for ties that *are* visible, i.e. between the returned distances
    """
    reference_indices, reference_distances = reference
    candidate_indices, candidate_distances = candidate

    reference_found = reference_indices >= 0
    candidate_found = candidate_indices >= 0
    num_reference = int(reference_found.sum())

    # Positional agreement of the sorted distance vectors
    both_found = reference_found & candidate_found
    distance_error = np.abs(
        np.where(both_found, candidate_distances, 0.0)
        - np.where(both_found, reference_distances, 0.0)
    )
    matched = both_found & (distance_error <= tolerance)
    max_distance_error = float(distance_error[matched].max()) if matched.any() else 0.0
    positional_recall = float(matched.sum() / num_reference) if num_reference else 1.0

    # An approximate method that misses one neighbour shifts every later position, so
    # positional agreement understates its recall. Where positions disagree, fall back
    # to matching the two sorted distance vectors as multisets, row by row
    distance_recall = positional_recall
    if positional_recall < 1.0:
        distance_recall = _multiset_distance_recall(
            reference_distances, reference_found, candidate_distances, candidate_found, tolerance
        )

    # Rows where no two reference distances are within the tolerance of each other, and
    # where the candidate found the same number of neighbours: only here is a differing
    # index set a genuine disagreement rather than a different tie-break
    padded_distances = np.where(reference_found, reference_distances, 0.0)
    gaps = np.diff(padded_distances, axis=1)
    both_positions_found = reference_found[:, :-1] & reference_found[:, 1:]
    if gaps.size:
        unambiguous = (~both_positions_found | (gaps > tolerance)).all(axis=1)
    else:
        unambiguous = np.ones(len(reference_indices), dtype=bool)
    unambiguous &= reference_found.sum(axis=1) == candidate_found.sum(axis=1)

    identical_rows = 0
    for row in np.flatnonzero(unambiguous):
        reference_set = set(reference_indices[row][reference_found[row]].tolist())
        candidate_set = set(candidate_indices[row][candidate_found[row]].tolist())
        identical_rows += int(reference_set == candidate_set)

    num_unambiguous = int(unambiguous.sum())
    return {
        "check_rows": float(len(reference_indices)),
        "reference_neighbours": float(num_reference),
        "positional_recall": positional_recall,
        "distance_recall": distance_recall,
        "max_distance_error": max_distance_error,
        "unambiguous_rows": float(num_unambiguous),
        "index_agreement": (float(identical_rows / num_unambiguous) if num_unambiguous else 1.0),
        # Neighbours the candidate returned that the reference did not, which for an
        # exact method must be zero and for an approximate one should be too (it may
        # miss neighbours, but must never invent one)
        "extra_neighbours": float(
            max(int(candidate_found.sum()) - num_reference, 0) / max(num_reference, 1)
        ),
    }


def _multiset_distance_recall(
    reference_distances: npt.NDArray[np.float32],
    reference_found: npt.NDArray[np.bool_],
    candidate_distances: npt.NDArray[np.float32],
    candidate_found: npt.NDArray[np.bool_],
    tolerance: float,
) -> float:
    """
    Fraction of the reference neighbours the candidate also returned, ignoring position

    Both distance vectors are already sorted, so each row is one merge pass: a
    reference distance counts as found when the candidate has an unclaimed distance
    within the tolerance of it

    Args:
        reference_distances: Reference distances (C, k), -1 padded
        reference_found: Which reference entries are real neighbours
        candidate_distances: Candidate distances (C, k), -1 padded
        candidate_found: Which candidate entries are real neighbours
        tolerance: Distance difference still counted as the same neighbour

    Returns:
        Matched reference neighbours divided by reference neighbours
    """
    num_reference = int(reference_found.sum())
    if not num_reference:
        return 1.0

    matched = 0
    for row in range(len(reference_distances)):
        wanted = reference_distances[row][reference_found[row]]
        offered = candidate_distances[row][candidate_found[row]]
        wanted_position = 0
        offered_position = 0
        while wanted_position < len(wanted) and offered_position < len(offered):
            difference = float(offered[offered_position]) - float(wanted[wanted_position])
            if abs(difference) <= tolerance:
                matched += 1
                wanted_position += 1
                offered_position += 1
            elif difference < 0:
                offered_position += 1
            else:
                wanted_position += 1
    return matched / num_reference


def compare_counts(
    reference: npt.NDArray[np.int64], candidate: npt.NDArray[np.int64]
) -> dict[str, float]:
    """
    Compare neighbour counts against the reference implementation's

    Args:
        reference: Counts from the reference competitor (C,)
        candidate: Counts from the competitor under test (C,)

    Returns:
        Dict with the exact-agreement fraction and the largest absolute difference
    """
    difference = candidate.astype(np.int64) - reference.astype(np.int64)
    return {
        "check_rows": float(len(reference)),
        "count_agreement": float((difference == 0).mean()),
        "max_count_difference": float(np.abs(difference).max()) if len(difference) else 0.0,
        "mean_count": float(reference.mean()) if len(reference) else 0.0,
    }


def _median(values: list[float]) -> float:
    """
    Median of a list of timings
    """
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _time_queries(call: Any, options: MeasurementOptions) -> tuple[float, list[float]]:
    """
    Warm up, then time a query call

    Each result is dropped before the next call, so two sets of (possibly
    multi-gigabyte) output arrays are never alive at once

    Args:
        call: Zero-argument callable running the query being timed
        options: Measurement limits, for the repeat-count decision

    Returns:
        Median query time in seconds and every timed sample
    """
    start = time.perf_counter()
    result = call()
    warm_up_seconds = time.perf_counter() - start
    del result

    repeats = 3 if warm_up_seconds < options["full_repeat_seconds"] else 1
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = call()
        samples.append(time.perf_counter() - start)
        del result
    return _median(samples), samples


def call_in_worker(function: Any, *arguments: Any, timeout: float = 1800.0) -> Any:
    """
    Run a function in a forked worker process and return its result

    This exists to keep the parent process free of the Rust extension: the first
    parallel operation in a process starts rayon's global thread pool, and anything
    forked afterwards inherits that pool's bookkeeping but none of its threads, so the
    child's next parallel build blocks forever. Anything that touches the extension
    runs in a worker instead, this function included

    Args:
        function: Callable to run in the worker. It and its arguments are pickled, so
            they must be importable module-level objects
        arguments: Positional arguments for `function`
        timeout: Seconds to wait before giving up on the worker

    Returns:
        Whatever `function` returned
    """
    context = multiprocessing.get_context("fork")
    result_queue = context.Queue()
    worker = context.Process(
        target=_call_and_reply, args=(result_queue, function, arguments), daemon=True
    )
    worker.start()
    try:
        return result_queue.get(timeout=timeout)
    except queue_module.Empty as error:
        raise RuntimeError(f"worker running {function!r} produced no result") from error
    finally:
        worker.join(timeout=30.0)
        if worker.is_alive():
            worker.kill()
            worker.join(timeout=30.0)


def _call_and_reply(result_queue: Any, function: Any, arguments: tuple[Any, ...]) -> None:
    """
    Worker entry point for `call_in_worker`: call the function, send the result back

    Args:
        result_queue: Queue the result is pushed onto
        function: Callable to run
        arguments: Positional arguments for `function`
    """
    result_queue.put(function(*arguments))


def _worker_block(
    result_queue: Any,
    dataset_key: str,
    competitor_key: str,
    radius_plan: RadiusPlan,
    num_neighbours_values: list[int],
    query_counts: list[int],
    query_order_key: str,
    options: MeasurementOptions,
    run_count_workload: bool,
) -> None:
    """
    Build one index in a forked worker and time every configuration against it

    Runs in the child process; every record is pushed onto `result_queue` as soon as it
    is measured, so a worker that is killed for running long still contributes its
    finished measurements

    Args:
        result_queue: Queue the records are pushed onto
        dataset_key: Key of the search points in `_SHARED_ARRAYS`
        competitor_key: Key into `COMPETITORS`
        radius_plan: Radius for this block and the density target it was chosen for
        num_neighbours_values: Values of k to time
        query_counts: Query batch sizes to time, ascending
        query_order_key: Key in `_SHARED_ARRAYS` of the permutation that picks the
            query points out of the search points
        options: Measurement limits
        run_count_workload: Whether to time `count_neighbours` as well
    """
    points = _SHARED_ARRAYS[dataset_key]
    query_order = _SHARED_ARRAYS[query_order_key]
    radius = radius_plan["radius"]

    # The worker inherits the parent's resident pages, so what this index and its
    # queries actually cost is the growth over this starting point, not the raw peak
    baseline_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    def record(**fields: Any) -> None:
        peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        result_queue.put(
            RunRecord(
                competitor=competitor_key,
                radius=radius,
                density_target=radius_plan["density_target"],
                peak_rss_mb=round(peak_rss_mb, 1),
                added_rss_mb=round(peak_rss_mb - baseline_rss_mb, 1),
                **fields,
            )
        )

    # Build the index once for the whole block
    try:
        start = time.perf_counter()
        competitor = COMPETITORS[competitor_key](points, radius)
        build_seconds = time.perf_counter() - start
    except Exception as error:  # a competitor that cannot build still reports why
        record(
            workload="build",
            num_neighbours=0,
            num_queries=0,
            status="failed",
            detail=f"{type(error).__name__}: {error}",
        )
        result_queue.put(_WORKER_DONE)
        return

    index_stats = None
    if hasattr(competitor, "grid_stats"):
        index_stats = competitor.grid_stats()
    record(
        workload="build",
        num_neighbours=0,
        num_queries=0,
        status="ok",
        build_seconds=build_seconds,
        index_stats=index_stats,
        spec=competitor.spec(),
    )

    check_queries = points[query_order[: options["num_check_queries"]]]

    # kNN workload, k by k, query counts ascending so that a cheap measurement can
    # veto an expensive one
    for num_neighbours in num_neighbours_values:
        check_result = competitor.find_neighbours(check_queries, num_neighbours)
        record(
            workload="check",
            num_neighbours=num_neighbours,
            num_queries=len(check_queries),
            status="ok",
            check_indices=check_result[0],
            check_distances=check_result[1],
        )

        last_measurement: tuple[int, float] | None = None
        for num_queries in query_counts:
            output_gb = num_queries * num_neighbours * 16 / 1e9
            if output_gb > options["max_output_gb"]:
                record(
                    workload="find",
                    num_neighbours=num_neighbours,
                    num_queries=num_queries,
                    status="skipped",
                    detail=f"output arrays would need about {output_gb:.1f} GB",
                )
                continue
            if last_measurement is not None:
                previous_queries, previous_seconds = last_measurement
                projected = previous_seconds * num_queries / previous_queries
                if projected > options["max_run_seconds"]:
                    record(
                        workload="find",
                        num_neighbours=num_neighbours,
                        num_queries=num_queries,
                        status="skipped",
                        detail=f"projected {projected:.0f} s from the smaller batch",
                    )
                    continue

            queries = points[query_order[:num_queries]]
            median_seconds, samples = _time_queries(
                lambda: competitor.find_neighbours(queries, num_neighbours), options
            )
            last_measurement = (num_queries, median_seconds)
            record(
                workload="find",
                num_neighbours=num_neighbours,
                num_queries=num_queries,
                status="ok",
                build_seconds=build_seconds,
                query_seconds=median_seconds,
                query_seconds_samples=samples,
                repeats=len(samples),
            )
            del queries

    # Neighbour-count workload
    if run_count_workload and competitor.supports_count:
        check_counts = competitor.count_neighbours(check_queries)
        record(
            workload="check-count",
            num_neighbours=0,
            num_queries=len(check_queries),
            status="ok",
            check_counts=np.asarray(check_counts, dtype=np.int64),
        )

        last_measurement = None
        for num_queries in query_counts:
            if last_measurement is not None:
                previous_queries, previous_seconds = last_measurement
                projected = previous_seconds * num_queries / previous_queries
                if projected > options["max_run_seconds"]:
                    record(
                        workload="count",
                        num_neighbours=0,
                        num_queries=num_queries,
                        status="skipped",
                        detail=f"projected {projected:.0f} s from the smaller batch",
                    )
                    continue
            queries = points[query_order[:num_queries]]
            median_seconds, samples = _time_queries(
                lambda: competitor.count_neighbours(queries), options
            )
            last_measurement = (num_queries, median_seconds)
            record(
                workload="count",
                num_neighbours=0,
                num_queries=num_queries,
                status="ok",
                build_seconds=build_seconds,
                query_seconds=median_seconds,
                query_seconds_samples=samples,
                repeats=len(samples),
            )
            del queries

    result_queue.put(_WORKER_DONE)


def _run_block(
    dataset_key: str,
    competitor_key: str,
    radius_plan: RadiusPlan,
    num_neighbours_values: list[int],
    query_counts: list[int],
    query_order_key: str,
    options: MeasurementOptions,
    run_count_workload: bool,
) -> list[dict[str, Any]]:
    """
    Run one `(competitor, radius)` block in a forked worker and collect its records

    Args:
        dataset_key: Key of the search points in `_SHARED_ARRAYS`
        competitor_key: Key into `COMPETITORS`
        radius_plan: Radius for this block and the density target behind it
        num_neighbours_values: Values of k to time
        query_counts: Query batch sizes to time, ascending
        query_order_key: Key of the query permutation in `_SHARED_ARRAYS`
        options: Measurement limits
        run_count_workload: Whether to time `count_neighbours` as well

    Returns:
        Every record the worker produced, plus a `timeout` record if it was killed
    """
    context = multiprocessing.get_context("fork")
    result_queue = context.Queue()
    worker = context.Process(
        target=_worker_block,
        args=(
            result_queue,
            dataset_key,
            competitor_key,
            radius_plan,
            num_neighbours_values,
            query_counts,
            query_order_key,
            options,
            run_count_workload,
        ),
        daemon=True,
    )
    worker.start()

    records: list[dict[str, Any]] = []
    deadline = time.monotonic() + options["inactivity_timeout_seconds"]
    while True:
        if time.monotonic() > deadline:
            records.append(
                RunRecord(
                    competitor=competitor_key,
                    workload="find",
                    num_neighbours=0,
                    num_queries=0,
                    radius=radius_plan["radius"],
                    density_target=radius_plan["density_target"],
                    status="timeout",
                    detail=(
                        "worker produced no result for "
                        f"{options['inactivity_timeout_seconds']:.0f} s and was killed"
                    ),
                )
            )
            worker.terminate()
            break
        try:
            message = result_queue.get(timeout=2.0)
        except queue_module.Empty:
            if not worker.is_alive():
                break
            continue
        if message == _WORKER_DONE:
            break
        records.append(message)
        deadline = time.monotonic() + options["inactivity_timeout_seconds"]

    worker.join(timeout=30.0)
    if worker.is_alive():
        worker.kill()
        worker.join(timeout=30.0)
    if worker.exitcode not in (0, None) and not any(
        record.get("status") == "timeout" for record in records
    ):
        records.append(
            RunRecord(
                competitor=competitor_key,
                workload="find",
                num_neighbours=0,
                num_queries=0,
                radius=radius_plan["radius"],
                density_target=radius_plan["density_target"],
                status="failed",
                detail=f"worker exited with code {worker.exitcode}",
            )
        )
    return records


def _projected_self_query_seconds(
    measured_find_seconds: dict[tuple[str, int, int], float], num_points: int
) -> float | None:
    """
    Project how long a full self-query over the cloud would take, from measurements

    The graph backend's build *is* a self-query for `graph_degree` neighbours over the
    whole cloud, answered with the hybrid grid, so on a cloud dense enough for that to
    run for an hour it has to be skipped before it is started rather than waited out.
    Scaling the hybrid backend's own measured query time up to N queries is the closest
    honest estimate available before the build happens

    Args:
        measured_find_seconds: `(competitor, k, Q) -> median seconds` measured so far
            at this radius
        num_points: Points in the cloud, i.e. the number of self-queries the build does

    Returns:
        Projected seconds, or None when nothing comparable has been measured yet
    """
    projections = [
        seconds * num_points / num_queries
        for (competitor, num_neighbours, num_queries), seconds in measured_find_seconds.items()
        if competitor == "oxvox-hybrid" and num_neighbours >= 8
    ]
    return min(projections) if projections else None


def benchmark_group(
    dataset_label: str,
    description: str,
    points: npt.NDArray[np.float32],
    radius_plans: list[RadiusPlan],
    num_neighbours_values: list[int],
    query_counts: list[int],
    competitor_keys: list[str],
    options: MeasurementOptions,
    run_count_workload: bool = True,
    seed: int = 7,
) -> dict[str, Any]:
    """
    Benchmark every competitor on one `(dataset, N)` pair across the radius grid

    Args:
        dataset_label: Short name of the dataset, as it appears in the report
        description: One-line description of how the cloud was made
        points: The search points (N, 3), float32
        radius_plans: The radii to sweep, with the density target behind each
        num_neighbours_values: Values of k to time
        query_counts: Query batch sizes to time (each is capped at N)
        competitor_keys: Keys into `COMPETITORS`, reference first
        options: Measurement limits
        run_count_workload: Whether to time `count_neighbours` as well
        seed: Seed for the random choice of query points

    Returns:
        A results dict with `meta`, `dataset`, `radius_plans` and `runs`, ready to be
        written to JSON
    """
    num_points = len(points)
    query_counts = sorted({min(count, num_points) for count in query_counts})

    # One query permutation for the whole group, so every batch size is a prefix of the
    # same spatially unsorted sample of the cloud's own points (self-query, the usual
    # real workload) and every competitor sees exactly the same queries
    rng = np.random.default_rng(seed=seed)
    query_order = rng.permutation(num_points).astype(np.int64)

    dataset_key = f"{dataset_label}-points"
    query_order_key = f"{dataset_label}-query-order"
    _SHARED_ARRAYS[dataset_key] = points
    _SHARED_ARRAYS[query_order_key] = query_order

    # The reference competitor has to run first: the others' correctness is measured
    # against the answers it returns
    ordered_keys = [key for key in competitor_keys if key == REFERENCE_COMPETITOR]
    ordered_keys += [key for key in competitor_keys if key != REFERENCE_COMPETITOR]

    runs: list[dict[str, Any]] = []
    specs: dict[str, Any] = {}
    realised_density: dict[float, float] = {}
    for radius_plan in radius_plans:
        reference_neighbours: dict[int, tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]] = {}
        reference_counts: npt.NDArray[np.int64] | None = None

        # Query times measured at this radius, used to project the cost of the blocks
        # that have not run yet
        measured_find_seconds: dict[tuple[str, int, int], float] = {}

        for competitor_key in ordered_keys:
            # The graph backend builds by self-querying the whole cloud, which on a
            # very dense cloud costs more than the whole rest of the grid put together
            if competitor_key == "oxvox-graph":
                projected = _projected_self_query_seconds(measured_find_seconds, num_points)
                if projected is not None and projected > options["max_build_seconds"]:
                    runs.append(
                        RunRecord(
                            competitor=competitor_key,
                            workload="build",
                            num_neighbours=0,
                            num_queries=0,
                            radius=radius_plan["radius"],
                            density_target=radius_plan["density_target"],
                            status="skipped",
                            detail=(
                                f"build is a self-query projected to take {projected:.0f} s"
                            ),
                        )
                    )
                    continue

            block_start = time.monotonic()
            records = _run_block(
                dataset_key,
                competitor_key,
                radius_plan,
                num_neighbours_values,
                query_counts,
                query_order_key,
                options,
                run_count_workload,
            )
            logger.info(
                "%s N=%d r=%.4g (%g pts/sphere) %s: %d records in %.1f s",
                dataset_label,
                num_points,
                radius_plan["radius"],
                radius_plan["density_target"],
                competitor_key,
                len(records),
                time.monotonic() - block_start,
            )

            for record in records:
                workload = record.get("workload")

                # Correctness records carry the check answers rather than a timing;
                # they are folded into the timing records and never stored raw
                if workload == "check":
                    indices = record.pop("check_indices")
                    distances = record.pop("check_distances")
                    num_neighbours = record["num_neighbours"]
                    if competitor_key == REFERENCE_COMPETITOR:
                        reference_neighbours[num_neighbours] = (indices, distances)
                        continue
                    if num_neighbours in reference_neighbours:
                        runs.append(
                            RunRecord(
                                competitor=competitor_key,
                                workload="check",
                                num_neighbours=num_neighbours,
                                num_queries=record["num_queries"],
                                radius=record["radius"],
                                density_target=record["density_target"],
                                status="ok",
                                correctness=compare_neighbours(
                                    reference_neighbours[num_neighbours], (indices, distances)
                                ),
                            )
                        )
                    continue

                if workload == "check-count":
                    counts = record.pop("check_counts")
                    if competitor_key == REFERENCE_COMPETITOR:
                        reference_counts = counts
                        # The realised density of this cloud at this radius: the mean
                        # number of neighbours a point has, excluding its own copy
                        realised_density[radius_plan["radius"]] = float(counts.mean()) - 1.0
                        continue
                    if reference_counts is not None:
                        runs.append(
                            RunRecord(
                                competitor=competitor_key,
                                workload="check-count",
                                num_neighbours=0,
                                num_queries=record["num_queries"],
                                radius=record["radius"],
                                density_target=record["density_target"],
                                status="ok",
                                correctness=compare_counts(reference_counts, counts),
                            )
                        )
                    continue

                if workload == "build" and "spec" in record:
                    specs[competitor_key] = record.pop("spec")
                if workload == "find" and record.get("status") == "ok":
                    measured_find_seconds[
                        (competitor_key, record["num_neighbours"], record["num_queries"])
                    ] = record["query_seconds"]
                runs.append(record)

    del _SHARED_ARRAYS[dataset_key]
    del _SHARED_ARRAYS[query_order_key]

    return {
        "meta": environment_meta(),
        "dataset": {
            "label": dataset_label,
            "description": description,
            "num_points": num_points,
            "num_check_queries": min(options["num_check_queries"], num_points),
        },
        "radius_plans": [
            {
                **radius_plan,
                "realised_points_per_sphere": realised_density.get(radius_plan["radius"]),
            }
            for radius_plan in radius_plans
        ],
        "query_counts": query_counts,
        "num_neighbours_values": num_neighbours_values,
        "competitors": specs,
        "options": options,
        "runs": runs,
    }


def write_results(results: dict[str, Any], output_dir: Path) -> Path:
    """
    Write one group's results to JSON, named after the dataset and point count

    Args:
        results: Return value of `benchmark_group`
        output_dir: Directory to write into (created if missing)

    Returns:
        Path of the file written
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    label = results["dataset"]["label"].replace(" ", "-")
    path = output_dir / f"{label}-{results['dataset']['num_points']}.json"
    path.write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    logger.info("wrote %s (%d runs)", path.name, len(results["runs"]))
    return path
