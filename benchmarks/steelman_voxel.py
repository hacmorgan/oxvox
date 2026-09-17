#!/usr/bin/env python3


"""
Time OxVoxNNS build and query on the historical performance-test scenarios

Written to run unchanged against both the released 0.7.2 wheel and the current build
(positional arguments only, and the `method` argument is passed only when supported),
so the two can be compared like for like. Results are printed as JSON, one object per
scenario, so they can be diffed or collected by the item 5 benchmark harness

Usage:
    steelman_voxel.py [--method voxel|kdtree|kiddo] [--scenario NAME ...] [--threads N]
"""

import argparse
import inspect
import json
import sys
from time import perf_counter

import numpy as np

from oxvox.nns import OxVoxNNS


def make_scenarios(rng: np.random.Generator) -> dict[str, dict]:
    """
    The scenarios the original performance tests used, with fixed seeds
    """
    uniform_4m = (rng.random((4_000_000, 3), dtype=np.float32) * 15).astype(np.float32)
    clusters_4m = np.concatenate(
        [
            rng.random((400_000, 3), dtype=np.float32) * 0.5
            + rng.random((1, 3), dtype=np.float32) * 15
            for _ in range(10)
        ]
    )
    many_clusters_1m = np.concatenate(
        [
            rng.random((10_000, 3), dtype=np.float32)
            + rng.random((1, 3), dtype=np.float32) * 15
            for _ in range(100)
        ]
    )
    return {
        "4m-uniform-k1": {"points": uniform_4m, "radius": 0.05, "k": 1},
        "4m-uniform-k8": {"points": uniform_4m, "radius": 0.05, "k": 8},
        "4m-clusters-k8": {"points": clusters_4m, "radius": 0.05, "k": 8},
        "1m-many-clusters-k8": {"points": many_clusters_1m, "radius": 0.05, "k": 8},
        "1m-many-clusters-k64": {"points": many_clusters_1m, "radius": 0.2, "k": 64},
    }


def time_scenario(name: str, scenario: dict, method: str | None, threads: int) -> dict:
    """
    Build the index, then run a self-query for neighbours and a count, timing each
    """
    points, radius, k = scenario["points"], scenario["radius"], scenario["k"]
    queries = points[::4]  # 25 % of the cloud as queries keeps the runs to seconds

    supports_method = "method" in inspect.signature(OxVoxNNS.__init__).parameters
    start = perf_counter()
    nns = OxVoxNNS(points, radius, method) if (method and supports_method) else OxVoxNNS(points, radius)
    build_time = perf_counter() - start

    start = perf_counter()
    indices, distances = nns.find_neighbours(queries, k, threads)
    find_time = perf_counter() - start

    start = perf_counter()
    counts = nns.count_neighbours(queries, threads)
    count_time = perf_counter() - start

    return {
        "scenario": name,
        "method": getattr(nns, "method", "voxel-0.7.2"),
        "num_points": int(len(points)),
        "num_queries": int(len(queries)),
        "radius": radius,
        "k": k,
        "build_s": round(build_time, 3),
        "find_s": round(find_time, 3),
        "count_s": round(count_time, 3),
        # Cheap fingerprints so the two versions can be checked for identical answers
        "found_total": int((indices >= 0).sum()),
        "distance_sum": float(np.sum(distances[distances >= 0], dtype=np.float64)),
        "count_total": int(counts.sum(dtype=np.int64)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default=None)
    parser.add_argument("--scenario", action="append", default=None)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(seed=2024)
    scenarios = make_scenarios(rng)
    for name in args.scenario or scenarios:
        result = time_scenario(name, scenarios[name], args.method, args.threads)
        print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
