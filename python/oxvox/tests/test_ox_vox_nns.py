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

import pickle

import numpy as np
import pytest

from oxvox.nns import EXACT_METHODS, OxVoxNNS, available_methods

# Every exact search method compiled into this build gets the same test suite; the
# approximate "graph" method gets its own recall tests below
METHODS = [method for method in available_methods() if method in EXACT_METHODS]


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


@pytest.mark.parametrize("method", METHODS)
def test_find_neighbours(method: str) -> None:
    """
    Test a simple case of finding neighbours in a small pointcloud
    """
    query_points = TEST_SEARCH_POINTS[0].reshape(1, -1)
    num_neighbours = 3
    max_dist = 4.0

    nns = OxVoxNNS(TEST_SEARCH_POINTS, max_dist, method=method)
    assert nns.method == method
    assert len(nns) == len(TEST_SEARCH_POINTS)
    indices, distances = nns.find_neighbours(query_points, num_neighbours)

    assert np.all(indices == [0, 1, 2])
    assert np.allclose(distances, [0.0, 0.173, 3.410], atol=0.001)


@pytest.mark.parametrize("method", METHODS)
def test_count_neighbours(method: str) -> None:
    """
    Test a simple case of counting neighbours in a small pointcloud
    """
    query_points = TEST_SEARCH_POINTS

    nns = OxVoxNNS(TEST_SEARCH_POINTS, 2.0, method=method)
    assert np.all(nns.count_neighbours(query_points) == [2] * 4)

    nns = OxVoxNNS(TEST_SEARCH_POINTS, 4.0, method=method)
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


@pytest.mark.parametrize("method", METHODS)
def test_count_neighbours_randomised_brute_force(method: str) -> None:
    """
    Randomised brute-force cross-check of the weighted kernel against a pure-numpy
    pairwise-distance reference implementation, for several values of p
    """
    rng = np.random.default_rng(seed=0)
    points = rng.random((1000, 3), dtype=np.float32)
    search_radius = 0.15

    nns = OxVoxNNS(points, search_radius, method=method)

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


@pytest.mark.parametrize("method", METHODS)
def test_pickle_roundtrip_preserves_query_results(method: str) -> None:
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

    nns = OxVoxNNS(search_points, search_radius, method=method)
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


@pytest.mark.parametrize("method", METHODS)
def test_count_neighbours_plain_count_clustered_and_negative_coords(method: str) -> None:
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

    nns = OxVoxNNS(points, search_radius, method=method)
    counts = nns.count_neighbours(query_points, distance_weight_factor=None)

    distances = np.linalg.norm(
        query_points[:, None, :] - points[None, :, :], axis=-1
    ).astype(np.float32)
    expected_counts = np.sum(distances < search_radius, axis=1)

    assert np.array_equal(counts, expected_counts)
    # Sanity: the clusters are dense enough that this isn't a trivial all-ones test
    assert counts.max() > 5


def test_unknown_method_raises() -> None:
    """
    An unrecognised method name must raise ValueError from the Rust engine
    """
    with pytest.raises(ValueError):
        OxVoxNNS(TEST_SEARCH_POINTS, 1.0, method="octree")  # type: ignore[arg-type]


def test_auto_method_resolves_to_voxel() -> None:
    """
    Until the benchmark-derived heuristic lands, "auto" means "voxel"
    """
    nns = OxVoxNNS(TEST_SEARCH_POINTS, 1.0, method="auto")
    assert nns.method == "voxel"


def test_grid_stats_only_for_voxel_method() -> None:
    """
    Grid occupancy statistics are reported for the voxel method and None otherwise
    """
    rng = np.random.default_rng(seed=2)
    points = rng.random((500, 3), dtype=np.float32)

    stats = OxVoxNNS(points, 0.5, method="voxel").grid_stats()
    assert stats is not None
    assert set(stats) == {"num_cells", "max_points_per_cell", "mean_points_per_cell"}
    assert 1 <= stats["num_cells"] <= 8
    assert stats["max_points_per_cell"] >= stats["mean_points_per_cell"]

    assert OxVoxNNS(points, 0.5, method="kdtree").grid_stats() is None

    hybrid_stats = OxVoxNNS(points, 0.5, method="hybrid", subtree_threshold=16).grid_stats()
    assert hybrid_stats is not None and hybrid_stats["num_subtrees"] >= 1


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("cells_per_radius", [1, 2])
def test_methods_agree_with_brute_force_knn(method: str, cells_per_radius: int) -> None:
    """
    Every method must return the same k nearest neighbours (as sets, with matching
    sorted distances) as a brute-force numpy reference, on a cloud straddling the origin
    """
    rng = np.random.default_rng(seed=4)
    points = (rng.random((3000, 3), dtype=np.float32) - 0.5) * 2
    queries = (rng.random((200, 3), dtype=np.float32) - 0.5) * 2.2
    search_radius = 0.25
    num_neighbours = 12

    nns = OxVoxNNS(points, search_radius, method=method, cells_per_radius=cells_per_radius)
    indices, distances = nns.find_neighbours(queries, num_neighbours)

    all_distances = np.linalg.norm(queries[:, None, :] - points[None, :, :], axis=-1)
    for q in range(len(queries)):
        in_range = np.flatnonzero(all_distances[q] < search_radius)
        expected = in_range[np.argsort(all_distances[q, in_range])][:num_neighbours]
        found = indices[q][indices[q] >= 0]
        assert len(found) == len(expected)
        assert np.allclose(np.sort(distances[q][: len(found)]), all_distances[q, expected], atol=1e-5)
        assert np.all(np.diff(distances[q][: len(found)]) >= 0), "results must be nearest-first"
        assert set(found.tolist()) == set(expected.tolist())
        assert np.all(indices[q][len(found):] == -1)
        assert np.all(distances[q][len(found):] == -1.0)


def test_structured_and_float64_inputs_are_accepted() -> None:
    """
    Structured arrays with x/y/z fields and float64 arrays are converted for the engine
    """
    rng = np.random.default_rng(seed=6)
    points = rng.random((100, 3))
    structured = np.zeros(100, dtype=[("x", np.float64), ("y", np.float64), ("z", np.float64), ("i", np.int32)])
    structured["x"], structured["y"], structured["z"] = points.T

    from_unstructured = OxVoxNNS(points, 0.3).count_neighbours(points)
    from_structured = OxVoxNNS(structured, 0.3).count_neighbours(structured)
    assert np.array_equal(from_unstructured, from_structured)


def test_graph_method_is_approximate_but_safe() -> None:
    """
    The graph method must never return a point outside the radius, a duplicate, or
    non-trailing padding, and on uniform data with k <= graph_degree its recall against
    an exact method must be very high
    """
    rng = np.random.default_rng(seed=9)
    points = rng.random((5000, 3), dtype=np.float32)
    queries = rng.random((300, 3), dtype=np.float32)
    search_radius = 0.15
    num_neighbours = 8

    graph = OxVoxNNS(points, search_radius, method="graph", graph_degree=16)
    exact = OxVoxNNS(points, search_radius, method="kdtree")
    assert graph.method == "graph"

    approx_indices, approx_distances = graph.find_neighbours(queries, num_neighbours)
    exact_indices, _ = exact.find_neighbours(queries, num_neighbours)

    hits = wanted = 0
    for q in range(len(queries)):
        found = approx_indices[q][approx_indices[q] >= 0]
        assert len(set(found.tolist())) == len(found), "duplicate neighbour"
        assert np.all(approx_indices[q][len(found):] == -1), "padding must be trailing"
        distances = np.linalg.norm(points[found] - queries[q], axis=1)
        assert np.all(distances < search_radius)
        assert np.allclose(np.sort(approx_distances[q][: len(found)]), approx_distances[q][: len(found)])
        expected = set(exact_indices[q][exact_indices[q] >= 0].tolist())
        hits += len(expected & set(found.tolist()))
        wanted += len(expected)
    assert hits / wanted >= 0.99

    # Approximate counts can only ever undercount
    assert np.all(graph.count_neighbours(queries) <= exact.count_neighbours(queries))

    # And it pickles like everything else
    unpickled = pickle.loads(pickle.dumps(graph))
    assert np.array_equal(unpickled.find_neighbours(queries, num_neighbours)[0], approx_indices)
