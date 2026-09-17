"""
Python wrapper around Rust NNS engine for typing stubs and interpreter help
"""

from typing import Literal

import numpy as np
import numpy.typing as npt
from numpy.lib.recfunctions import structured_to_unstructured

from oxvox._oxvox import OxVoxNNSEngine, available_methods

# Search methods the Rust engine can be built with, plus "auto" which picks one
Method = Literal["voxel", "kdtree", "hybrid", "graph", "kiddo", "auto"]

# Methods that return exactly the brute-force answer (everything except "graph")
EXACT_METHODS = ("voxel", "kdtree", "hybrid", "kiddo")


class OxVoxNNS:
    """
    Radius-bounded nearest neighbour search implemented in Rust, with a choice of
    spatial index behind it

    Methods:
        "voxel": Uniform grid of cells of side ``search_radius / cells_per_radius``.
            Search points are stored sorted by cell, and a query scans the cells that
            could hold a neighbour, skipping cells whose bounding box is already
            further away than the current best candidate. Consistent performance on
            dense, clustered clouds where KD-trees struggle
        "kdtree": Bucketed KD-tree with median splits on the widest axis. Cheaper on
            sparse or very non-uniform clouds
        "hybrid": The voxel grid, with a KD subtree inside every cell holding more than
            ``subtree_threshold`` points, so dense cells are searched in logarithmic
            rather than linear time. Exact
        "graph": Approximate. Each search point is linked to its ``graph_degree``
            nearest neighbours at build time; a query finds its nearest search point
            exactly (via the hybrid grid) then floods outwards along graph edges. Never
            returns a point outside the radius, but may miss some neighbours
        "kiddo": The ``kiddo`` crate's KD-tree, only present in builds compiled with
            the ``kiddo-baseline`` cargo feature (used for benchmarking)
        "auto": Currently resolves to "voxel"; a data-driven heuristic is planned
    """

    def __init__(
        self,
        search_points: npt.NDArray[np.floating],
        search_radius: float,
        method: Method = "voxel",
        cells_per_radius: int = 1,
        subtree_threshold: int = 64,
        graph_degree: int = 16,
    ) -> None:
        """
        Construct neighbour searcher object

        n.b. this class (and the rust object it constructs internally) can be pickled,
        allowing queries to be done in async/parallel contexts if required

        Args:
            search_points: Points to search for neighbours amongst. Can be given as a 2D
                unstructured (i.e. conventional) array with 3 columns, or as a
                structured array with "x", "y" and "z" columns at minimum
            search_radius: Maximum distance between points before they are no longer
                considered neighbours
            method: Which spatial index to build, see the class docstring
            cells_per_radius: For the grid-based methods, how many grid cells span one
                search radius. 1 gives the classic 27-cell neighbourhood; 2 gives
                smaller cells with less over-scan per query but more cell lookups
            subtree_threshold: For "hybrid" and "graph", cells with more points than
                this get a KD subtree instead of a linear scan
            graph_degree: For "graph", how many nearest neighbours each search point is
                linked to. Recall is high for `num_neighbours <= graph_degree` and
                degrades gradually above it
        """
        # The rust engine strictly expects 3-column unstructured arrays of 32-bit
        # floats, so we must convert structured arrays to unstructured and ensure we
        # only have 32-bit values
        search_points = self._sanitise_points(search_points)

        # "auto" is a placeholder until the benchmark-derived heuristic lands
        resolved_method = "voxel" if method == "auto" else method

        # Construct internal rust neighbour searcher
        self.engine = OxVoxNNSEngine(
            search_points,
            search_radius,
            method=resolved_method,
            cells_per_radius=cells_per_radius,
            subtree_threshold=subtree_threshold,
            graph_degree=graph_degree,
        )

    @property
    def method(self) -> str:
        """
        Name of the spatial index this searcher was built with
        """
        return self.engine.method

    def __len__(self) -> int:
        """
        Number of indexed search points
        """
        return len(self.engine)

    def find_neighbours(
        self,
        query_points: npt.NDArray[np.floating],
        num_neighbours: int,
        num_threads: int = 0,
        epsilon: float = 0,
        progress: bool = False,
    ) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
        """
        Find neighbours in search points within range for all given query points

        Args:
            query_points: Points to search for neighbours of. Can be given as a 2D
                unstructured (i.e. conventional) array with 3 columns, or as a
                structured array with "x", "y" and "z" columns at minimum
            num_neighbours: Maximum number of neighbours to find, a.k.a. `k`
            num_threads: Number of parallel CPU threads to use in queries. Uses all
                available CPUs if set to 0
            epsilon: Once `num_neighbours` neighbours closer than this distance have
                been found for a query point, its search stops early (an approximate
                mode). A small positive value can help prevent the search getting
                bogged down in extremely dense regions
            progress: Show a progress bar over query points

        Returns:
            Indices of neighbouring search points. -1 where neighbours can't be found
            Distance to query point for each search point index. -1.0 where neighbours
                can't be found
        """
        return self.engine.find_neighbours(
            self._sanitise_points(query_points),
            num_neighbours,
            num_threads=num_threads,
            epsilon=epsilon,
            progress=progress,
        )

    def count_neighbours(
        self,
        query_points: npt.NDArray[np.floating],
        num_threads: int = 0,
        distance_weight_factor: float | None = None,
        progress: bool = False,
    ) -> npt.NDArray[np.uint32 | np.float32]:
        """
        Count neighbours in search points within range for all given query points

        Each neighbour's contribution to the count can optionally be weighted by its
        distance from the query point, using the normalised kernel
        `w = (1 - distance / search_radius) ** distance_weight_factor`. This kernel is
        exactly 1 at zero distance and falls to 0 at the search radius; neighbours at or
        beyond the search radius never contribute

        Args:
            query_points: Points to search for neighbours of. Can be given as a 2D
                unstructured (i.e. conventional) array with 3 columns, or as a
                structured array with "x", "y" and "z" columns at minimum
            num_threads: Number of parallel CPU threads to use in queries. Uses all
                available CPUs if set to 0
            distance_weight_factor: Exponent applied to the normalised distance kernel
                described above. If `None` (the default), neighbours are counted
                exactly, with every in-range neighbour contributing 1 regardless of
                distance. A value of 0 reproduces that same exact count. A value of 1
                gives linear falloff with distance. Larger values concentrate the
                weight closer to the query point. Must be `None` or non-negative;
                negative values raise `ValueError`
            progress: Show a progress bar over query points

        Returns:
            Neighbour count for each query point (Q,). An array of `uint32` exact
            counts when `distance_weight_factor` is `None`, otherwise an array of
            `float32` distance-weighted sums
        """
        counts = self.engine.count_neighbours(
            self._sanitise_points(query_points),
            num_threads=num_threads,
            distance_weight_factor=distance_weight_factor,
            progress=progress,
        )
        return counts.astype(np.uint32) if distance_weight_factor is None else counts

    def grid_stats(self) -> dict[str, float] | None:
        """
        Occupancy statistics of the voxel grid, or None for methods without a grid

        Returns:
            Dict with `num_cells`, `max_points_per_cell` and `mean_points_per_cell`
            (plus `num_subtrees` for the "hybrid" method)
        """
        return self.engine.grid_stats()

    @staticmethod
    def _sanitise_points(points: npt.NDArray[np.floating]) -> npt.NDArray[np.float32]:
        """
        Prepare pointcloud arrays to be used by rust engine, which expects C-contiguous
        (N, 3) arrays of 32-bit floats

        Args:
            points: Pointcloud to be sanitised

        Returns:
            Pointcloud, ready to be passed into rust engine
        """
        if points.dtype.names is not None:
            points = structured_to_unstructured(points[["x", "y", "z"]], dtype=np.float32)
        return np.ascontiguousarray(points, dtype=np.float32)


__all__ = ["OxVoxNNS", "Method", "EXACT_METHODS", "available_methods"]
