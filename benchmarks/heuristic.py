"""
Score the `method="auto"` rule against the measured benchmark grid

    python -m benchmarks.heuristic

`auto` has to choose a backend from what is knowable before any index exists: the
point count and the grid occupancy (both cheap), never k or the query batch, which the
caller only supplies later. This module replays the measured grid through the real rule
(`oxvox.nns.choose_auto_method`, so there is one copy of it) and reports two things:

- the hit rate, i.e. how often the rule names the method that actually turned out
  fastest at that grid point
- the regret, i.e. how much slower than the fastest method the rule's choice was,
  which is what a user actually pays for a miss

A hit rate well below 100% with a regret near 1.0 is a good rule on a grid where
several methods are within noise of each other; a high hit rate with a bad tail is not.
"""

import argparse
import json
import logging
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, TypedDict

from oxvox.nns import EXACT_METHODS, choose_auto_method

from benchmarks.report import load_results

logger = logging.getLogger("benchmarks.heuristic")

# Competitor keys of the oxvox backends `auto` can choose between. The grid also
# measures the voxel backend at two cells per radius, which `auto` cannot ask for, so
# that configuration is left out of the comparison rather than counted as a method the
# rule failed to pick
OXVOX_EXACT_COMPETITORS = {f"oxvox-{method}": method for method in EXACT_METHODS}


class GridFeatures(TypedDict):
    """
    The features the rule is allowed to see, per (dataset, N, radius) grid column
    """


    num_points: int
    mean_points_per_cell: float
    max_points_per_cell: int


def extract_features(groups: list[dict[str, Any]]) -> dict[tuple[str, int, float], GridFeatures]:
    """
    Pull the grid occupancy the rule needs out of the recorded voxel index builds

    The voxel backend reports its own occupancy after building, and `grid_occupancy`
    in the Rust extension computes exactly the same numbers without building anything,
    so the recorded stats stand in for what `auto` would compute at construction time

    Args:
        groups: Parsed results files

    Returns:
        (dataset, N, radius) -> features
    """
    features: dict[tuple[str, int, float], GridFeatures] = {}
    for group in groups:
        dataset = group["dataset"]["label"]
        num_points = group["dataset"]["num_points"]
        for record in group["runs"]:
            if (
                record["workload"] != "build"
                or record["competitor"] != "oxvox-voxel"
                or not record.get("index_stats")
            ):
                continue
            stats = record["index_stats"]
            features[(dataset, num_points, record["radius"])] = GridFeatures(
                num_points=num_points,
                mean_points_per_cell=stats["mean_points_per_cell"],
                max_points_per_cell=int(stats["max_points_per_cell"]),
            )
    return features


def evaluate(groups: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Replay the grid through the rule and every fixed choice it could have made instead

    Args:
        groups: Parsed results files

    Returns:
        Dict with the rule's overall score, its score per dataset, and the score of
        every fixed single-method policy for comparison
    """
    features = extract_features(groups)

    # (dataset, N, radius, k, Q) -> {method: seconds}
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
            key = (
                dataset,
                num_points,
                record["radius"],
                record["num_neighbours"],
                record["num_queries"],
            )
            measurements[key][method] = record["query_seconds"]

    policies = ["auto", *sorted({method for times in measurements.values() for method in times})]
    scores: dict[str, dict[str, Any]] = {}
    per_dataset: dict[str, dict[str, float]] = defaultdict(dict)

    for policy in policies:
        regrets: list[float] = []
        hits = 0
        considered = 0
        dataset_regrets: dict[str, list[float]] = defaultdict(list)
        dataset_hits: dict[str, list[int]] = defaultdict(list)

        for key, times in sorted(measurements.items()):
            dataset, num_points, radius = key[0], key[1], key[2]
            if policy == "auto":
                feature = features.get((dataset, num_points, radius))
                if feature is None:
                    continue
                choice = choose_auto_method(**feature)
            else:
                choice = policy
            if choice not in times:
                continue
            best = min(times.values())
            considered += 1
            regrets.append(times[choice] / best)
            hits += times[choice] == best
            dataset_regrets[dataset].append(times[choice] / best)
            dataset_hits[dataset].append(int(times[choice] == best))

        if not considered:
            continue
        ordered = sorted(regrets)
        scores[policy] = {
            "grid_points": considered,
            "hit_rate": hits / considered,
            "median_regret": statistics.median(ordered),
            "p90_regret": ordered[int(0.9 * (len(ordered) - 1))],
            "worst_regret": ordered[-1],
        }
        for dataset, values in dataset_regrets.items():
            per_dataset[dataset][policy] = statistics.median(values)
            per_dataset[dataset][f"{policy} hit rate"] = statistics.mean(dataset_hits[dataset])

    return {"policies": scores, "per_dataset": dict(per_dataset)}


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

    print(f"{'policy':10} {'grid points':>12} {'hit rate':>9} {'median':>8} {'p90':>8} {'worst':>8}")
    for policy, score in scores["policies"].items():
        print(
            f"{policy:10} {score['grid_points']:>12} {score['hit_rate']:>8.0%} "
            f"{score['median_regret']:>7.2f}x {score['p90_regret']:>7.2f}x "
            f"{score['worst_regret']:>7.2f}x"
        )
    print("\nmedian regret per dataset, by policy")
    for dataset, policies in sorted(scores["per_dataset"].items()):
        rendered = "  ".join(
            f"{policy} {value:.2f}x"
            for policy, value in sorted(policies.items())
            if not policy.endswith("hit rate")
        )
        print(f"  {dataset:18} {rendered}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
