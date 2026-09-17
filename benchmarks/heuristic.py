"""
Score the `method="auto"` rule against the measured benchmark grid

    python -m benchmarks.heuristic

`auto` has to choose a backend from what is knowable before an index exists: the cloud
and the search radius, never `num_neighbours` or the query batch size, which the caller
only supplies later. This module replays the measured grid through the real rule
(`oxvox.nns.choose_auto_method`, so there is only ever one copy of it) and reports:

- the hit rate: how often the rule names the backend that actually turned out fastest
- the regret: how much slower than the fastest backend the rule's choice was, which is
  what a caller actually pays for a miss
- the same two numbers for every fixed single-backend policy, and for an oracle that is
  allowed to pick the best backend per (dataset, N, radius) but not per k or per query
  batch, which is the ceiling any construction-time rule could reach

A hit rate well below 100% with a regret near 1.0 is a good rule on a grid where
several backends are within noise of each other; a high hit rate with a bad tail is
not, and a rule that cannot beat the best fixed choice should not exist.
"""

import argparse
import json
import logging
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from oxvox.nns import EXACT_METHODS, choose_auto_method

from benchmarks.report import load_results

logger = logging.getLogger("benchmarks.heuristic")

# Competitor keys of the oxvox backends `auto` can choose between. The grid also
# measures the voxel backend at two cells per radius, which `auto` cannot ask for, so
# that configuration is left out rather than counted as a backend the rule failed to
# pick
OXVOX_EXACT_COMPETITORS = {f"oxvox-{method}": method for method in EXACT_METHODS}

# Name given to the construction-time oracle in the report
ORACLE = "oracle"


def collect_measurements(
    groups: list[dict[str, Any]],
) -> dict[tuple[str, int, float, int, int], dict[str, float]]:
    """
    Every kNN measurement of an `auto`-selectable backend, by grid point

    Args:
        groups: Parsed results files

    Returns:
        `(dataset, N, radius, k, Q) -> {method: query seconds}`, restricted to the grid
        points where every candidate backend ran, since only those compare fairly
    """
    measurements: dict[tuple[str, int, float, int, int], dict[str, float]] = defaultdict(dict)
    for group in groups:
        dataset = group["dataset"]["label"]
        num_points = group["dataset"]["num_points"]
        for record in group["runs"]:
            if record["workload"] != "find" or record.get("status") != "ok":
                continue
            method = OXVOX_EXACT_COMPETITORS.get(record["competitor"])
            if method is None:
                continue
            measurements[
                (
                    dataset,
                    num_points,
                    record["radius"],
                    record["num_neighbours"],
                    record["num_queries"],
                )
            ][method] = record["query_seconds"]

    candidates = set(OXVOX_EXACT_COMPETITORS.values())
    return {
        grid_point: times
        for grid_point, times in measurements.items()
        if candidates.issubset(times)
    }


def oracle_choices(
    measurements: dict[tuple[str, int, float, int, int], dict[str, float]],
) -> dict[tuple[str, int, float], str]:
    """
    The best backend per cloud and radius: the ceiling for any construction-time rule

    The oracle sees a whole column of k and query-batch measurements at once and picks
    the backend with the lowest mean regret over it. It still cannot pick per k or per
    batch size, because neither is knowable when the index is built, which is exactly
    what makes it the right yardstick for `auto`

    Args:
        measurements: Output of `collect_measurements`

    Returns:
        `(dataset, N, radius) -> method`
    """
    columns: dict[tuple[str, int, float], list[dict[str, float]]] = defaultdict(list)
    for grid_point, times in measurements.items():
        columns[grid_point[:3]].append(times)

    choices: dict[tuple[str, int, float], str] = {}
    for column, rows in columns.items():
        regrets: dict[str, list[float]] = defaultdict(list)
        for times in rows:
            fastest = min(times.values())
            for method, seconds in times.items():
                regrets[method].append(seconds / fastest)
        choices[column] = min(regrets, key=lambda method: statistics.mean(regrets[method]))
    return choices


def evaluate(groups: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Score the rule, every fixed single-backend policy, and the build-time oracle

    Args:
        groups: Parsed results files

    Returns:
        Dict with a score per policy, the median regret per dataset, and the oracle's
        choice per grid column
    """
    measurements = collect_measurements(groups)
    if not measurements:
        raise SystemExit("no grid point has a measurement for every candidate backend")
    oracle = oracle_choices(measurements)

    methods = sorted({method for times in measurements.values() for method in times})
    policies = ["auto", *methods, ORACLE]
    scores: dict[str, dict[str, Any]] = {}
    per_dataset: dict[str, dict[str, float]] = defaultdict(dict)

    for policy in policies:
        regrets: list[float] = []
        hits = 0
        dataset_regrets: dict[str, list[float]] = defaultdict(list)
        dataset_hits: dict[str, list[int]] = defaultdict(list)

        for grid_point, times in sorted(measurements.items()):
            if policy == "auto":
                choice = choose_auto_method(num_points=grid_point[1], methods=tuple(times))
            elif policy == ORACLE:
                choice = oracle[grid_point[:3]]
            else:
                choice = policy
            fastest = min(times.values())
            regret = times[choice] / fastest
            regrets.append(regret)
            hits += times[choice] == fastest
            dataset_regrets[grid_point[0]].append(regret)
            dataset_hits[grid_point[0]].append(int(times[choice] == fastest))

        ordered = sorted(regrets)
        scores[policy] = {
            "grid_points": len(ordered),
            "hit_rate": hits / len(ordered),
            "mean_regret": statistics.mean(ordered),
            "median_regret": statistics.median(ordered),
            "p90_regret": ordered[int(0.9 * (len(ordered) - 1))],
            "worst_regret": ordered[-1],
        }
        for dataset, values in dataset_regrets.items():
            per_dataset[dataset][policy] = statistics.median(values)
            per_dataset[dataset][f"{policy} hit rate"] = statistics.mean(dataset_hits[dataset])

    return {
        "policies": scores,
        "per_dataset": dict(per_dataset),
        "oracle_choices": {
            f"{dataset} N={num_points} r={radius:.5g}": method
            for (dataset, num_points, radius), method in sorted(oracle.items())
        },
    }


def main(argv: list[str] | None = None) -> int:
    """
    Print the rule's score against the results on disk

    Args:
        argv: Command-line arguments, defaulting to `sys.argv[1:]`

    Returns:
        Process exit status
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="directory holding the results JSON (default: benchmarks/results)",
    )
    parser.add_argument(
        "--json", action="store_true", help="print the scores as JSON instead of a table"
    )
    arguments = parser.parse_args(argv)

    scores = evaluate(load_results(arguments.results_dir))
    if arguments.json:
        print(json.dumps(scores, indent=1, sort_keys=True))
        return 0

    print(
        f"{'policy':10} {'grid points':>11} {'fastest':>8} {'mean':>7} "
        f"{'median':>7} {'p90':>7} {'worst':>7}"
    )
    for policy, score in scores["policies"].items():
        print(
            f"{policy:10} {score['grid_points']:>11} {score['hit_rate']:>7.0%} "
            f"{score['mean_regret']:>6.2f}x {score['median_regret']:>6.2f}x "
            f"{score['p90_regret']:>6.2f}x {score['worst_regret']:>6.2f}x"
        )

    print("\nmedian regret per dataset, by policy")
    for dataset, policies in sorted(scores["per_dataset"].items()):
        rendered = "  ".join(
            f"{policy} {value:.2f}x"
            for policy, value in sorted(policies.items())
            if not policy.endswith("hit rate")
        )
        print(f"  {dataset:14} {rendered}")

    oracle_counts: dict[str, int] = defaultdict(int)
    for method in scores["oracle_choices"].values():
        oracle_counts[method] += 1
    print(
        "\nthe oracle's per-cloud choices: "
        + ", ".join(f"{method} {count}" for method, count in sorted(oracle_counts.items()))
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
