#!/usr/bin/env -S pytest -vv

"""
Tests for the benchmark's synthetic pointcloud generators

The generators decide what the benchmark measures, so the properties the harness
relies on are worth pinning down: the exact point count, the dtype and layout the Rust
engines require, and reproducibility from the seed.
"""

import numpy as np
import pytest

from benchmarks import generators

# Point count every generator is exercised with, small enough to keep the tests quick
NUM_POINTS = 5_000


@pytest.mark.parametrize("name", sorted(generators.GENERATORS))
def test_generator_returns_a_usable_cloud(name: str) -> None:
    """
    Every generator returns the exact number of C-contiguous float32 points asked for
    """
    cloud = generators.generate(name, NUM_POINTS)
    points = cloud["points"]

    assert points.shape == (NUM_POINTS, 3)
    assert points.dtype == np.float32
    assert points.flags["C_CONTIGUOUS"]
    assert np.isfinite(points).all()
    assert cloud["description"]


@pytest.mark.parametrize("name", sorted(generators.GENERATORS))
def test_generators_are_seeded(name: str) -> None:
    """
    Two calls with the same arguments give the same cloud, point for point
    """
    first = generators.generate(name, NUM_POINTS)["points"]
    second = generators.generate(name, NUM_POINTS)["points"]
    assert np.array_equal(first, second)


def test_generators_are_distinguishable() -> None:
    """
    The families really are different shapes, not the same cloud under five names
    """
    extents = {
        name: np.ptp(generators.generate(name, NUM_POINTS)["points"], axis=0)
        for name in generators.GENERATORS
    }

    # The sheet and the cylinder shell are thin in one direction; the uniform box is not
    assert extents["sheet"][2] < extents["sheet"][0] / 10
    assert extents["cylinder"][0] < extents["cylinder"][2] / 2
    assert extents["uniform"].min() > generators.DEFAULT_EXTENT * 0.9


def test_uniform_radius_hits_its_density_target() -> None:
    """
    The radius picked for a density target really holds that many points on the uniform
    cloud, which is what makes the target a meaningful column of the benchmark grid
    """
    from oxvox.nns import OxVoxNNS

    points = generators.uniform_box(200_000)["points"]
    for target in (1.0, 10.0, 100.0):
        radius = generators.uniform_radius_for_density(len(points), target)
        counts = OxVoxNNS(points, radius).count_neighbours(points[:2_000])

        # Every query is one of the search points, so it counts itself as well
        realised = counts.mean() - 1.0
        assert realised == pytest.approx(target, rel=0.15)
