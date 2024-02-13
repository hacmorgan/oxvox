"""
Compare different nearest neighbour search algorithms under various conditions
"""


import sys
from time import time

import numpy as np
import numpy.typing as npt
import plotly
from open3d.core.nns import NearestNeighborSearch as Open3DNNS
from scipy.spatial import KDTree as ScipyKDTree, cKDTree as ScipycKDTree
from sklearn.neighbors import KDTree as SKLearnKDTree


def run_scipy_kdtree(points: npt.NDArray[np.float32], num_neighbours: int) -> None:
    """
    Run nearest neighbour search using scipy KDTree
    """
    tree = ScipyKDTree(points)
    tree.query(points, k=num_neighbours)
    
    
def main() -> int:
    """
    Compare performance of NNS algorithms   
    """
    
    num_neighbours = 1000
    num_search_points = 10_000  # also try 100_000, 1_000_000
    point_coord_range = 15  # also try larger and smaller values
    
    # Generate some test inputs
    TEST_POINTCLOUDS = [
        np.random.random((num_search_points, 3)).astype(np.float32) * point_coord_range,  # Random point coordinaes between 0 and 15
        # add more as necessary
    ]
    
    # Test each pointcloud on each KDTree algorithm
    for test_pointcloud in TEST_POINTCLOUDS:
        
        # Scipy native Python implementation
        start = time() 
        run_scipy_kdtree(test_pointcloud, num_neighbours)
        compute_time_scipy = time() - start
        
        # Scipy C implementation
        # TODO
        
        # Sklearn implementation
        # TODO
        
        # Open3D implementation
        # TODO
        
        # Rust implementation
        # TODO
        
    # Plot results (e.g. different algos as different colour lines, x-axis: num points, y-axis: processing time)
    # TODO
    
    return 0


if __name__ == "__main__":
    sys.exit(main())