"""
Uniform wrappers around every neighbour-search implementation the benchmark compares

Each wrapper builds an index over a fixed set of search points and answers two
workloads against it:

- `find_neighbours`: the (up to) k nearest search points within the radius of each
  query, as `(indices, distances)` with oxvox's padding semantics, i.e. int32 indices
  padded with -1 and float32 euclidean distances padded with -1.0, nearest first
- `count_neighbours`: how many search points lie within the radius of each query,
  for the implementations that expose it

Normalising the padding here (scipy pads with `inf` distances and an out-of-range
index; Open3D pads with -1 indices and zero *squared* distances) is what makes the
correctness comparison in `harness.py` a plain array comparison.
"""

from typing import Callable, TypedDict

import numpy as np
import numpy.typing as npt

# Indices and distances of the neighbours found for a batch of query points
NeighbourArrays = tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]


class CompetitorSpec(TypedDict):
    """
    Everything needed to construct one competitor, as stored in the results JSON
    """

    label: str
    library: str
    exact: bool
    supports_count: bool
    supports_thread_count: bool


class Competitor:
    """
    Base class: an index over search points that answers the two benchmark workloads
    """

    label = "competitor"
    library = "unknown"
    exact = True
    supports_count = False
    supports_thread_count = False

    def __init__(self, points: npt.NDArray[np.float32], radius: float) -> None:
        """
        Build the index (the benchmark times this constructor)

        Args:
            points: Search points to index (N, 3), float32
            radius: Search radius every query will use
        """
        raise NotImplementedError

    def find_neighbours(
        self, query_points: npt.NDArray[np.float32], num_neighbours: int
    ) -> NeighbourArrays:
        """
        Find up to `num_neighbours` neighbours within the radius of each query point

        Args:
            query_points: Points to find the neighbours of (Q, 3), float32
            num_neighbours: Maximum number of neighbours per query point, a.k.a. k

        Returns:
            Neighbour indices (Q, k) int32, -1 padded, and euclidean distances
            (Q, k) float32, -1.0 padded, both nearest first
        """
        raise NotImplementedError

    def count_neighbours(self, query_points: npt.NDArray[np.float32]) -> npt.NDArray[np.int64]:
        """
        Count the search points within the radius of each query point

        Args:
            query_points: Points to count the neighbours of (Q, 3), float32

        Returns:
            Neighbour count per query point (Q,)
        """
        raise NotImplementedError

    def spec(self) -> CompetitorSpec:
        """
        The static description of this competitor, for the results JSON
        """
        return CompetitorSpec(
            label=self.label,
            library=self.library,
            exact=self.exact,
            supports_count=self.supports_count,
            supports_thread_count=self.supports_thread_count,
        )


class ScipyKdTree(Competitor):
    """
    scipy.spatial.cKDTree, queried with a distance upper bound and all worker threads
    """

    label = "scipy cKDTree"
    library = "scipy"
    exact = True
    supports_count = True

    def __init__(self, points: npt.NDArray[np.float32], radius: float) -> None:
        from scipy.spatial import cKDTree

        self._radius = radius
        self._num_points = len(points)
        self._tree = cKDTree(points)

    def find_neighbours(
        self, query_points: npt.NDArray[np.float32], num_neighbours: int
    ) -> NeighbourArrays:
        distances, indices = self._tree.query(
            query_points,
            k=num_neighbours,
            distance_upper_bound=self._radius,
            workers=-1,
        )

        # scipy returns 1D arrays for k == 1, and marks "not found" with an infinite
        # distance and an index one past the end of the tree. The padding is rewritten
        # in place and the arrays narrowed afterwards: at four million queries and
        # k=64 scipy's float64/int64 outputs are already four gigabytes, so making a
        # copy per step would double the benchmark's peak memory
        distances = np.atleast_2d(distances.T).T
        indices = np.atleast_2d(indices.T).T
        missing = ~np.isfinite(distances)
        distances[missing] = -1.0
        indices[missing] = -1
        return indices.astype(np.int32), distances.astype(np.float32)

    def count_neighbours(self, query_points: npt.NDArray[np.float32]) -> npt.NDArray[np.int64]:
        counts = self._tree.query_ball_point(
            query_points, r=self._radius, return_length=True, workers=-1
        )
        return np.asarray(counts, dtype=np.int64)


class Open3DHybrid(Competitor):
    """
    Open3D's `NearestNeighborSearch.hybrid_search`, the same radius-bounded kNN query

    Open3D parallelises internally over all cores and offers no thread count, and it
    returns squared distances, which are converted here
    """

    label = "Open3D hybrid"
    library = "open3d"
    exact = True
    supports_count = False

    def __init__(self, points: npt.NDArray[np.float32], radius: float) -> None:
        import open3d as o3d

        self._core = o3d.core
        self._radius = radius
        self._index = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(points))
        if not self._index.hybrid_index(radius):
            raise RuntimeError("Open3D failed to build its hybrid index")

    def find_neighbours(
        self, query_points: npt.NDArray[np.float32], num_neighbours: int
    ) -> NeighbourArrays:
        indices, squared_distances, _counts = self._index.hybrid_search(
            self._core.Tensor(query_points), self._radius, num_neighbours
        )
        indices = indices.numpy().astype(np.int32)
        distances = np.sqrt(squared_distances.numpy(), dtype=np.float32)

        # Open3D pads missing neighbours with index -1 and squared distance 0
        distances[indices < 0] = -1.0
        return indices, distances


class OxVox(Competitor):
    """
    One of oxvox's own backends, selected by `method` (and `cells_per_radius`)
    """

    library = "oxvox"
    supports_count = True
    supports_thread_count = True

    def __init__(
        self,
        points: npt.NDArray[np.float32],
        radius: float,
        method: str = "voxel",
        cells_per_radius: int = 1,
        label: str | None = None,
    ) -> None:
        from oxvox.nns import EXACT_METHODS, OxVoxNNS

        self.label = label or f"oxvox {method}"
        self.exact = method in EXACT_METHODS
        self._method = method
        self._nns = OxVoxNNS(
            points, radius, method=method, cells_per_radius=cells_per_radius
        )

    def find_neighbours(
        self, query_points: npt.NDArray[np.float32], num_neighbours: int
    ) -> NeighbourArrays:
        return self._nns.find_neighbours(query_points, num_neighbours)

    def count_neighbours(self, query_points: npt.NDArray[np.float32]) -> npt.NDArray[np.int64]:
        return self._nns.count_neighbours(query_points).astype(np.int64)

    def grid_stats(self) -> dict[str, float] | None:
        """
        The underlying voxel grid's occupancy, where the backend has one
        """
        return self._nns.grid_stats()


def _oxvox_factory(
    method: str, cells_per_radius: int = 1, label: str | None = None
) -> Callable[[npt.NDArray[np.float32], float], Competitor]:
    """
    Build a zero-configuration constructor for one oxvox backend

    Args:
        method: oxvox `method` name
        cells_per_radius: Grid cells per search radius, for the grid-based methods
        label: Name to show in the report, defaulting to "oxvox <method>"

    Returns:
        A callable taking `(points, radius)`, as every competitor constructor does
    """

    def construct(points: npt.NDArray[np.float32], radius: float) -> Competitor:
        return OxVox(
            points,
            radius,
            method=method,
            cells_per_radius=cells_per_radius,
            label=label,
        )

    return construct


# Every competitor the benchmark can run, by the key used on the command line and in
# the results JSON. Order is the order they appear in the report
COMPETITORS: dict[str, Callable[[npt.NDArray[np.float32], float], Competitor]] = {
    "scipy": ScipyKdTree,
    "open3d": Open3DHybrid,
    "oxvox-voxel": _oxvox_factory("voxel", label="oxvox voxel"),
    "oxvox-voxel-c2": _oxvox_factory("voxel", cells_per_radius=2, label="oxvox voxel (c=2)"),
    "oxvox-kdtree": _oxvox_factory("kdtree"),
    "oxvox-hybrid": _oxvox_factory("hybrid"),
    "oxvox-graph": _oxvox_factory("graph"),
    "oxvox-kiddo": _oxvox_factory("kiddo"),
}

# The reference implementation every exact method's answers are checked against
REFERENCE_COMPETITOR = "scipy"


def available_competitors() -> list[str]:
    """
    Competitor keys whose library is importable in this environment

    oxvox's "kiddo" backend only exists in builds compiled with the `kiddo-baseline`
    cargo feature, and scipy/Open3D are optional extras, so the grid adapts to what is
    installed rather than failing

    Returns:
        Keys of `COMPETITORS` that can actually be constructed here
    """
    import importlib.util

    from oxvox.nns import available_methods

    oxvox_methods = set(available_methods())
    usable: list[str] = []
    for key in COMPETITORS:
        if key.startswith("oxvox-"):
            method = key.removeprefix("oxvox-").removesuffix("-c2")
            if method in oxvox_methods:
                usable.append(key)
        elif importlib.util.find_spec(key) is not None:
            usable.append(key)
    return usable
