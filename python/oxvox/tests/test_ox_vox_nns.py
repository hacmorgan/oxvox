#!/usr/bin/env -S pytest -vvv


"""
Unit tests for rust binding test library

Run this test script to verify that functions can be compiled and run, and produce
expected results

n.b. The rust module needs to be compiled the first time this is run, but pytest will
hide the output of the rust compiler, so it may appear to hang for a little while.
Subsequent compilations should be much shorter
"""


import pickle

import numpy as np
import numpy.lib.recfunctions as rf
import pytest

from oxvox.nns import OxVoxNNS


TEST_ARRAY = np.arange(9, dtype=np.float32).reshape((3, 3))
ORIGIN = np.array([0, 0, 0], dtype=np.float32)


TEST_SEARCH_POINTS = np.array(
    [
        [0.1, 0.1, 0.2],  # Point 0: Point 1's neighbour
        [0.2, 0.2, 0.1],  # Point 1: Point 0's neighbour
        [3.2, 1.2, 1.1],  # Point 2: A neighbour to Points 0 and 1 if r > 3 (ish)
        [3.3, 1.1, 1.0],  # Point 3: Similar to Point 2, also Point 2's neighbour
    ],
    dtype=np.float32,
)


def test_find_neighbours() -> None:
    """
    Test a simple case of finding neighbours in a small pointcloud
    """
    query_points = TEST_SEARCH_POINTS[0].reshape(1, -1)
    num_neighbours = 3
    max_dist = 4.0
    voxel_size = 0.3

    nns = OxVoxNNS(TEST_SEARCH_POINTS, max_dist)
    indices, distances = nns.find_neighbours(query_points, num_neighbours)

    assert np.all(indices == [0, 1, 2])
    assert np.allclose(distances, [0.0, 0.173, 3.410], atol=0.001)


def test_count_neighbours() -> None:
    """
    Test a simple case of counting neighbours in a small pointcloud
    """
    query_points = TEST_SEARCH_POINTS
    num_neighbours = 3
    max_dist = 2.0
    voxel_size = 0.3

    nns = OxVoxNNS(TEST_SEARCH_POINTS, 2.0)
    assert np.all(nns.count_neighbours(query_points) == [2] * 4)

    nns = OxVoxNNS(TEST_SEARCH_POINTS, 4.0)
    assert np.all(nns.count_neighbours(query_points) == [4] * 4)

    counts_none = nns.count_neighbours(query_points, distance_weight_factor=None)
    counts_zero = nns.count_neighbours(query_points, distance_weight_factor=0)

    # p=0 must reproduce the plain count exactly, but the two differ in dtype: None
    # gives an exact integer count, while any given weighting factor (including 0)
    # gives a distance-weighted float32 sum
    assert np.all(counts_zero == counts_none)
    assert counts_none.dtype == np.uint32
    assert counts_zero.dtype == np.float32


def test_count_neighbours_linear_weighting() -> None:
    """
    Test p=1 (linear falloff) against hand-computed weighted counts
    """
    search_radius = 2.0
    nns = OxVoxNNS(TEST_SEARCH_POINTS, search_radius)

    weighted_counts = nns.count_neighbours(
        TEST_SEARCH_POINTS, distance_weight_factor=1.0
    )

    # Hand-computed pairwise distances: only points 0-1 and 2-3 are mutual neighbours
    # within the 2.0 search radius (see TEST_SEARCH_POINTS comments above); each point
    # is also its own neighbour, at distance 0
    distance_01 = np.sqrt(0.03)  # sqrt((-0.1) ** 2 + (-0.1) ** 2 + 0.1 ** 2)
    distance_23 = np.sqrt(0.03)  # sqrt((-0.1) ** 2 + 0.1 ** 2 + 0.1 ** 2)
    expected = np.array(
        [
            1.0 + (1 - distance_01 / search_radius),
            1.0 + (1 - distance_01 / search_radius),
            1.0 + (1 - distance_23 / search_radius),
            1.0 + (1 - distance_23 / search_radius),
        ],
        dtype=np.float32,
    )

    assert np.allclose(weighted_counts, expected, atol=1e-4)

    # Hard-coded literal for at least one radius, so this test isn't purely
    # self-referential against the numpy computation above
    assert np.allclose(weighted_counts[0], 1.913397, atol=1e-4)


def test_count_neighbours_coincident_point_contributes_one() -> None:
    """
    A query point exactly coincident with a search point must contribute exactly 1.0
    to the weighted count, for any non-negative distance_weight_factor, since
    (1 - 0 / search_radius) ** p == 1 for all p
    """
    search_points = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    query_points = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    nns = OxVoxNNS(search_points, 1.0)

    for distance_weight_factor in (0.0, 0.5, 1.0, 2.0, 10.0):
        result = nns.count_neighbours(
            query_points, distance_weight_factor=distance_weight_factor
        )
        assert np.allclose(result, [1.0])


def test_count_neighbours_point_at_radius_contributes_nothing() -> None:
    """
    A search point at exactly the search radius must contribute 0 to the weighted sum
    (the comparison against the radius is strictly less-than), while a coincident
    point still contributes 1.0, and the plain count only includes the coincident point
    """
    search_points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    query_points = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    search_radius = 1.0
    nns = OxVoxNNS(search_points, search_radius)

    plain_count = nns.count_neighbours(query_points, distance_weight_factor=None)
    weighted_count = nns.count_neighbours(query_points, distance_weight_factor=1.0)

    assert np.all(plain_count == [1])
    assert np.allclose(weighted_count, [1.0])


def test_count_neighbours_negative_weight_factor_raises() -> None:
    """
    A negative distance_weight_factor doesn't correspond to a valid kernel, and must
    raise ValueError
    """
    nns = OxVoxNNS(TEST_SEARCH_POINTS, 2.0)
    with pytest.raises(ValueError):
        nns.count_neighbours(TEST_SEARCH_POINTS, distance_weight_factor=-1.0)


def test_count_neighbours_randomised_brute_force() -> None:
    """
    Randomised brute-force cross-check of the weighted kernel against a pure-numpy
    pairwise-distance reference implementation, for several values of p
    """
    rng = np.random.default_rng(seed=0)
    points = rng.random((1000, 3), dtype=np.float32)
    search_radius = 0.15

    nns = OxVoxNNS(points, search_radius)

    # Full pairwise distance matrix via broadcasting (1000 x 1000 x 3 is small enough)
    distances = np.linalg.norm(
        points[:, None, :] - points[None, :, :], axis=-1
    ).astype(np.float32)

    # The plain (unweighted) count must match the brute-force count exactly, and p=0
    # must reproduce it exactly too, despite the two paths comparing distances
    # differently internally (squared distances vs sqrt'd distances)
    expected_plain_counts = np.sum(distances < search_radius, axis=1)
    plain_counts = nns.count_neighbours(points, distance_weight_factor=None)
    zero_weighted_counts = nns.count_neighbours(points, distance_weight_factor=0.0)
    assert plain_counts.dtype == np.uint32
    assert np.array_equal(plain_counts, expected_plain_counts)
    assert np.array_equal(zero_weighted_counts, expected_plain_counts)

    # Weighted sums must match the brute-force kernel for several exponents
    for distance_weight_factor in (0.5, 1.0, 2.0):
        expected = np.sum(
            np.clip(1 - distances / search_radius, 0, None) ** distance_weight_factor
            * (distances < search_radius),
            axis=1,
        )
        actual = nns.count_neighbours(
            points, distance_weight_factor=distance_weight_factor
        )
        assert np.allclose(actual, expected, atol=1e-3, rtol=1e-4)


def test_pickle_roundtrip_preserves_query_results() -> None:
    """
    Pickling and unpickling an OxVoxNNS must be transparent: querying the unpickled
    object must give exactly the same results as querying the original, for both
    find_neighbours and count_neighbours

    This is the closest thing we have to a regression test for the pyo3 0.29 Bound-API
    migration, since __getstate__/__setstate__/__getnewargs__ are the least-exercised
    and most-changed surface in that migration
    """
    rng = np.random.default_rng(seed=2)
    search_points = rng.random((200, 3), dtype=np.float32)
    query_points = rng.random((50, 3), dtype=np.float32)
    search_radius = 0.2

    nns = OxVoxNNS(search_points, search_radius)
    unpickled_nns = pickle.loads(pickle.dumps(nns))

    original_indices, original_distances = nns.find_neighbours(query_points, 5)
    unpickled_indices, unpickled_distances = unpickled_nns.find_neighbours(
        query_points, 5
    )
    assert np.array_equal(original_indices, unpickled_indices)
    assert np.array_equal(original_distances, unpickled_distances)

    original_counts = nns.count_neighbours(query_points, distance_weight_factor=1.0)
    unpickled_counts = unpickled_nns.count_neighbours(
        query_points, distance_weight_factor=1.0
    )
    assert np.array_equal(original_counts, unpickled_counts)


def test_count_neighbours_plain_count_clustered_and_negative_coords() -> None:
    """
    Regression test for the plain count on a harder distribution: dense Gaussian
    clusters straddling the origin planes (negative and positive coordinates), which
    exercises many voxels and the voxel-neighbourhood logic, checked exactly against a
    brute-force reference
    """
    rng = np.random.default_rng(seed=1)
    cluster_centres = rng.uniform(-1.0, 1.0, size=(5, 3)).astype(np.float32)
    points = np.concatenate(
        [
            centre + rng.normal(scale=0.05, size=(400, 3)).astype(np.float32)
            for centre in cluster_centres
        ]
    )
    search_radius = 0.04
    query_points = points[::3]

    nns = OxVoxNNS(points, search_radius)
    counts = nns.count_neighbours(query_points, distance_weight_factor=None)

    distances = np.linalg.norm(
        query_points[:, None, :] - points[None, :, :], axis=-1
    ).astype(np.float32)
    expected_counts = np.sum(distances < search_radius, axis=1)

    assert np.array_equal(counts, expected_counts)
    # Sanity: the clusters are dense enough that this isn't a trivial all-ones test
    assert counts.max() > 5
