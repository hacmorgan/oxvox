"""
Python wrapper around Rust NNS engine for typing stubs and interpreter help
"""


from typing import Tuple
from functools import partial

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import numpy.typing as npt

from oxvox._oxvox import OxVoxNNSEngine


class OxVoxNNS:
    """
    A hybrid-ish nearest neighbour search implemented in rust, tailored towards
    consistent performance, especially on difficult inputs for KDTrees
    """

    def __init__(
        self, search_points: npt.NDArray[np.floating], search_radius: float
    ) -> None:
        """
        Construct neighbour searcher object

        Internally this groups search points by voxel and constructs a lookup for points
        by their voxel coordinates

        n.b. this class (and the rust object it constructs internally) can be pickled,
        allowing queries to be done in async/parallel contexts if required

        Args:
            search_points: Points to search for neighbours amongst. Can be given as a 2D
                unstructured (i.e. conventional) array with 3 columns, or as a
                structured array with "x", "y" and "z" columns at minimum
            search_radius: Maximum distance between points before they are no longer
                considered neighbours
        """
        # The rust engine strictly expects 3-column unstructured arrays of 32-bit
        # floats, so we must convert structured arrays to unstructured and enure we only
        # have 32-bit values
        search_points = self._sanitise_points(search_points)

        # Construct internal rust neighbour searcher
        self.engine = OxVoxNNSEngine(search_points, search_radius)

    def find_neighbours(
        self,
        query_points: npt.NDArray[np.floating],
        num_neighbours: int,
        num_threads: int = 0,
        # epsilon: float = np.finfo(np.float32).eps,
        epsilon: float = 0,
    ) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
        """
        Find neighbours in search points within range for all given query points

        Args:
            query_points: Points to search for neighbours of. Can be given as a 2D
                unstructured (i.e. conventional) array with 3 columns, or as a
                structured array with "x", "y" and "z" columns at minimum
            num_neighbours: Maximum number of neighbours to find, a.k.a. `k`
            num_threads: Number of parallel CPU threads to use in queries. Uses all
                available CPUs if set to 0
            epsilon: Any neighbours within this distance of the query point are accepted
                automatically (skips sorting). Even at its default value of float32 eps
                (0.00000012), this can help prevent the search from getting bogged down
                in extremely dense regions

        Returns:
            Indices of neighbouring search points. -1 where neighbours can't be found
            Distance to query point for each search point index. -1.0 where neighbours
                can't be found
        """
        return self.engine.find_neighbours(
            self._sanitise_points(query_points),
            num_neighbours,
            num_threads,
            epsilon,
        )

    def count_neighbours(
        self,
        query_points: npt.NDArray[np.floating],
        num_threads: int = 0,
    ) -> npt.NDArray[np.uint32]:
        """
        Count neighbours in search points within range for all given query points

        Args:
            query_points: Points to search for neighbours of. Can be given as a 2D
                unstructured (i.e. conventional) array with 3 columns, or as a
                structured array with "x", "y" and "z" columns at minimum
            num_neighbours: Maximum number of neighbours to find, a.k.a. `k`
            num_threads: Number of parallel CPU threads to use in queries. Uses all
                available CPUs if set to 0
            epsilon: Any neighbours within this distance of the query point are accepted
                automatically (skips sorting). Even at its default value of float32 eps
                (0.00000012), this can help prevent the search from getting bogged down
                in extremely dense regions

        Returns:
            Indices of neighbouring search points. -1 where neighbours can't be found
            Distance to query point for each search point index. -1.0 where neighbours
                can't be found
        """
        return self.engine.count_neighbours(
            self._sanitise_points(query_points),
            num_threads,
        )

    @staticmethod
    def _sanitise_points(points: npt.NDArray[np.floating]) -> npt.NDArray[np.float32]:
        """
        Prepare pointcloud arrays to be used by rust engine, which expects arrays of
        32-bit floats

        Args:
            points: Pointcloud to be sanitised

        Returns:
            Pointcloud, ready to be passed into rust engine
        """
        return (
            structured_to_unstructured(points[["x", "y", "z"]], dtype=np.float32)
            if points.dtype.names is not None
            else points.astype(np.float32)
        )
