"""
Run the oxvox neighbour-search benchmark grid and write the results as JSON

    python -m benchmarks.run --quick        # smoke grid, finishes in a few minutes
    python -m benchmarks.run                # the full grid the committed results use

The full grid sweeps, for each of five synthetic generators and any real pointclouds
given on the command line:

- point counts N of 1e4, 1e5, 1e6 and 4e6 (real clouds run at their native size and at
  a 1e6 random subsample)
- four search radii, calibrated per cloud so that a sphere of that radius really holds
  about 1, 10, 100 and 1000 of its points (the closed-form radius only works for the
  uniform box: a cylinder shell or a tight cluster is locally hundreds of times denser,
  so reusing the uniform radius would put every other dataset in the same very dense
  regime). Real clouds use radii that mean something for a laser scan instead: 1 cm,
  5 cm and 20 cm, with the density they realise recorded alongside
- k of 1, 8 and 64 neighbours
- query batches of 1e3, 1e5 and N points, drawn from the cloud's own points

Not every combination is measured: the harness projects each configuration's cost from
the next smaller query batch and skips the ones that would take longer than
`--max-run-seconds`, recording a marker that says so. Without that, the densest
4M-point configurations alone would run for hours.

Real pointclouds are passed as `--real-pointcloud "real scan A=/path/to/cloud"`. Only
the label reaches the results, never the path. `.npy` files holding an (N, 3) array or
a structured array with x/y/z fields are read natively; any other format needs
`--pointcloud-loader some.module:function`, a callable taking the path and returning
such an array, so site-specific formats never have to be known here.
"""

import argparse
import importlib
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from functools import partial

from benchmarks import generators, harness
from benchmarks.competitors import available_competitors

logger = logging.getLogger("benchmarks.run")

# The full grid
FULL_POINT_COUNTS = (10_000, 100_000, 1_000_000, 4_000_000)
FULL_DENSITY_TARGETS = (1.0, 10.0, 100.0, 1000.0)
FULL_NEIGHBOUR_COUNTS = (1, 8, 64)
FULL_QUERY_COUNTS = (1_000, 100_000, 4_000_000)

# The smoke grid: small clouds, two generators, enough to prove the harness works
QUICK_POINT_COUNTS = (10_000, 100_000)
QUICK_DENSITY_TARGETS = (1.0, 100.0)
QUICK_NEIGHBOUR_COUNTS = (1, 8)
QUICK_QUERY_COUNTS = (1_000, 100_000)
QUICK_DATASETS = ("uniform", "clusters")

# Radii for the real laser scans, in metres: a hair over the scan's own point spacing,
# a typical feature scale, and a coarse neighbourhood
REAL_SCAN_RADII = (0.01, 0.05, 0.2)

# Points a real cloud is subsampled to, so it can be compared with the synthetic grid
REAL_SUBSAMPLE_SIZE = 1_000_000

# Radius calibration: how close the realised neighbour count has to get to the target,
# how many tries it gets, and how many query points each try measures
CALIBRATION_TOLERANCE = 0.2
CALIBRATION_MAX_ITERATIONS = 10
CALIBRATION_SAMPLE_SIZE = 2048


def load_pointcloud(path: Path, loader: str | None = None) -> npt.NDArray[np.float32]:
    """
    Load a pointcloud from disk as an (N, 3) float32 array

    Args:
        path: File to load. `.npy` holds either an (N, 3) array or a structured array
            with x/y/z fields
        loader: For any other format, `module.path:function` naming a callable that
            takes the path and returns such an array. Imported lazily, so it is never a
            dependency of the benchmark

    Returns:
        The points, as the engines want them
    """
    if loader is not None:
        module_name, _, function_name = loader.partition(":")
        if not module_name or not function_name:
            raise SystemExit(f"--pointcloud-loader must look like module.path:function, got {loader!r}")
        array = getattr(importlib.import_module(module_name), function_name)(str(path))
    elif path.suffix == ".npy":
        array = np.load(path)
    else:
        raise SystemExit(
            f"cannot read {path.suffix} pointclouds natively; convert to .npy or pass "
            "--pointcloud-loader module.path:function"
        )

    if array.dtype.names is not None:
        array = np.stack([array["x"], array["y"], array["z"]], axis=1)
    array = np.ascontiguousarray(array, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3:
        raise SystemExit(f"expected an (N, 3) pointcloud, got shape {array.shape}")
    return array


def subsample(
    points: npt.NDArray[np.float32], num_points: int, seed: int = 11
) -> npt.NDArray[np.float32]:
    """
    Take a random subsample of a pointcloud, without replacement

    Args:
        points: The cloud to subsample (N, 3)
        num_points: How many points to keep
        seed: Seed for the random generator

    Returns:
        The subsampled cloud, C-contiguous
    """
    rng = np.random.default_rng(seed=seed)
    chosen = rng.choice(len(points), size=num_points, replace=False)
    return np.ascontiguousarray(points[np.sort(chosen)])


def calibrate_radius(
    points: npt.NDArray[np.float32], density_target: float, seed: int = 13
) -> tuple[float, float]:
    """
    Find the radius whose sphere holds `density_target` of this cloud's points

    The density target is what makes a column of the grid comparable across datasets,
    and it cannot be computed in closed form for anything but the uniform box: a
    cylinder shell or a cluster of points is locally hundreds of times denser than a
    uniform cloud of the same size in the same box, so the analytic radius would put
    every one of those datasets in the same extremely dense regime. This measures the
    realised neighbour count instead and walks the radius towards the target, fitting
    the local exponent (3 for a solid volume, nearer 2 for a surface) from the last two
    measurements so that surface-like clouds converge as fast as volume-like ones

    Args:
        points: The cloud to calibrate against (N, 3)
        density_target: Wanted mean number of neighbours within the radius, not
            counting the query point's own copy
        seed: Seed for the random choice of sample points

    Returns:
        The calibrated radius, and the mean neighbour count it realised
    """
    from oxvox.nns import OxVoxNNS

    rng = np.random.default_rng(seed=seed)
    sample_size = min(CALIBRATION_SAMPLE_SIZE, len(points))
    sample = np.ascontiguousarray(
        points[rng.choice(len(points), size=sample_size, replace=False)]
    )

    # Start from the radius a uniform cloud of the same size filling the same bounding
    # box would need, which is the right order of magnitude even when it is wrong
    extent = np.maximum(np.ptp(points, axis=0), 1e-6)
    volume_per_point = float(np.prod(extent)) / len(points)
    radius = float((density_target * volume_per_point * 3.0 / (4.0 * np.pi)) ** (1.0 / 3.0))

    previous: tuple[float, float] | None = None
    realised = 0.0
    for _ in range(CALIBRATION_MAX_ITERATIONS):
        counts = OxVoxNNS(points, radius).count_neighbours(sample)
        realised = float(counts.mean()) - 1.0
        if abs(realised - density_target) <= CALIBRATION_TOLERANCE * density_target:
            break

        # Local dimensionality from the last two measurements, so the step size suits
        # the cloud rather than assuming a solid volume
        exponent = 3.0
        if previous is not None and previous[1] > 0 and realised > 0 and previous[0] != radius:
            measured = np.log(realised / previous[1]) / np.log(radius / previous[0])
            exponent = float(np.clip(measured, 1.5, 3.5))
        previous = (radius, realised)

        ratio = max(density_target, 0.5) / max(realised, 0.05)
        radius *= float(np.clip(ratio ** (1.0 / exponent), 0.25, 4.0))

    return radius, realised


def synthetic_radius_plans(
    points: npt.NDArray[np.float32], density_targets: tuple[float, ...]
) -> list[harness.RadiusPlan]:
    """
    Radii for a synthetic group, calibrated to one density target each

    Args:
        points: The cloud the radii are calibrated against
        density_targets: Wanted mean neighbours per search sphere

    Returns:
        The radius plans, ascending
    """
    plans: list[harness.RadiusPlan] = []
    for target in density_targets:
        radius, realised = calibrate_radius(points, target)
        logger.info(
            "calibrated r=%.4g for ~%g points per sphere (realised %.3g)",
            radius,
            target,
            realised,
        )
        plans.append(
            harness.RadiusPlan(
                radius=radius,
                density_target=target,
                chosen_for=f"calibrated to ~{target:g} points per sphere on this cloud",
            )
        )
    return plans


def real_radius_plans() -> list[harness.RadiusPlan]:
    """
    Radii for a real laser scan, chosen as physical lengths rather than densities

    Returns:
        The radius plans, ascending
    """
    return [
        harness.RadiusPlan(
            radius=radius,
            density_target=0.0,
            chosen_for=f"{radius * 100:g} cm, a laser-scan feature scale",
        )
        for radius in REAL_SCAN_RADII
    ]


def _ignore_points_real_radius_plans(
    points: npt.NDArray[np.float32],
) -> list[harness.RadiusPlan]:
    """
    Radius plans for a real scan, which need no calibration

    Args:
        points: The cloud, ignored: the real radii are physical lengths

    Returns:
        The radius plans, ascending
    """
    return real_radius_plans()


def _parse_real_pointcloud(argument: str) -> tuple[str, Path]:
    """
    Split a `--real-pointcloud LABEL=PATH` argument

    Args:
        argument: The raw command-line value

    Returns:
        The label to publish and the path to read (which is never published)
    """
    label, separator, path = argument.partition("=")
    if not separator or not label.strip():
        raise argparse.ArgumentTypeError("expected LABEL=PATH, e.g. 'real scan A=/data/a.npy'")
    return label.strip(), Path(path).expanduser()


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse the command line

    Args:
        argv: Arguments to parse, defaulting to `sys.argv[1:]`

    Returns:
        The parsed arguments
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run the smoke grid (two generators, up to 1e5 points) instead of the full one",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="synthetic generator to run (repeatable); defaults to all of them",
    )
    parser.add_argument(
        "--competitor",
        action="append",
        dest="competitors",
        help="competitor key to run (repeatable); defaults to every importable one",
    )
    parser.add_argument(
        "--real-pointcloud",
        action="append",
        dest="real_pointclouds",
        type=_parse_real_pointcloud,
        default=[],
        metavar="LABEL=PATH",
        help="a real pointcloud to benchmark; only LABEL is written to the results",
    )
    parser.add_argument(
        "--pointcloud-loader",
        default=None,
        metavar="MODULE:FUNCTION",
        help=(
            "For real pointclouds that are not .npy: a callable, given as "
            "module.path:function, that takes the file path and returns an (N, 3) array "
            "or a structured array with x/y/z fields"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="where the results JSON goes (default: benchmarks/results)",
    )
    parser.add_argument(
        "--max-run-seconds",
        type=float,
        default=60.0,
        help="skip any configuration projected to take longer than this (default: 60)",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=4_000_000,
        help="cap on the largest query batch, which otherwise is N (default: 4e6)",
    )
    parser.add_argument(
        "--max-output-gb",
        type=float,
        default=6.0,
        help="skip configurations whose output arrays would exceed this (default: 6)",
    )
    parser.add_argument(
        "--no-count-workload",
        action="store_true",
        help="skip the count_neighbours workload",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Run the benchmark grid

    Args:
        argv: Command-line arguments, defaulting to `sys.argv[1:]`

    Returns:
        Process exit status
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    arguments = parse_arguments(argv)

    competitor_keys = arguments.competitors or available_competitors()
    logger.info("competitors: %s", ", ".join(competitor_keys))

    point_counts = QUICK_POINT_COUNTS if arguments.quick else FULL_POINT_COUNTS
    density_targets = QUICK_DENSITY_TARGETS if arguments.quick else FULL_DENSITY_TARGETS
    neighbour_counts = list(QUICK_NEIGHBOUR_COUNTS if arguments.quick else FULL_NEIGHBOUR_COUNTS)
    query_counts = list(QUICK_QUERY_COUNTS if arguments.quick else FULL_QUERY_COUNTS)
    query_counts = [min(count, arguments.max_queries) for count in query_counts]
    dataset_names = arguments.datasets or list(
        QUICK_DATASETS if arguments.quick else generators.GENERATORS
    )

    options = harness.default_options(
        max_run_seconds=arguments.max_run_seconds,
        max_output_gb=arguments.max_output_gb,
    )

    # Every (dataset, N) group to run, as a lazy loader so that only one cloud is in
    # memory at a time (the 4M-point clouds and the real scans are hundreds of MB)
    groups: list[dict[str, Any]] = []
    for dataset_name in dataset_names:
        for num_points in point_counts:
            groups.append(
                {
                    "label": dataset_name,
                    "num_points": num_points,
                    "load": lambda name=dataset_name, count=num_points: generators.generate(
                        name, count
                    ),
                    "plan": partial(synthetic_radius_plans, density_targets=density_targets),
                }
            )
    for label, path in arguments.real_pointclouds:
        real_points = load_pointcloud(path, loader=arguments.pointcloud_loader)
        logger.info("loaded %s: %d points", label, len(real_points))
        sizes = [len(real_points)]
        if not arguments.quick and len(real_points) > REAL_SUBSAMPLE_SIZE:
            sizes.append(REAL_SUBSAMPLE_SIZE)
        for size in sizes:
            groups.append(
                {
                    "label": label,
                    "num_points": size,
                    "load": lambda cloud=real_points, count=size: generators.GeneratedCloud(
                        points=cloud if count == len(cloud) else subsample(cloud, count),
                        description=(
                            f"real laser scan, {count} points"
                            + ("" if count == len(cloud) else " (random subsample)")
                        ),
                    ),
                    "plan": _ignore_points_real_radius_plans,
                }
            )

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_start = time.monotonic()
    for group_number, group in enumerate(groups, start=1):
        logger.info(
            "group %d/%d: %s at %d points (%.1f min elapsed)",
            group_number,
            len(groups),
            group["label"],
            group["num_points"],
            (time.monotonic() - benchmark_start) / 60.0,
        )
        cloud = group["load"]()

        # The radius calibration queries the Rust extension, which must not happen in
        # this process: see `harness.call_in_worker`
        radius_plans = harness.call_in_worker(group["plan"], cloud["points"])
        results = harness.benchmark_group(
            dataset_label=group["label"],
            description=cloud["description"],
            points=cloud["points"],
            radius_plans=radius_plans,
            num_neighbours_values=neighbour_counts,
            query_counts=query_counts,
            competitor_keys=competitor_keys,
            options=options,
            run_count_workload=not arguments.no_count_workload,
        )
        harness.write_results(results, arguments.output_dir)
        del cloud, results

    logger.info(
        "finished %d groups in %.1f min", len(groups), (time.monotonic() - benchmark_start) / 60.0
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
