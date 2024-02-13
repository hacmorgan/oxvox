#!/usr/bin/env python3


"""
Unit tests for rust binding test library

Run this test script to verify that functions can be compiled and run, and produce 
expected results

n.b. The rust module needs to be compiled the first time this is run, but pytest will
hide the output of the rust compiler, so it may appear to hang for a little while.
Subsequent compilations should be much shorter
"""


import numpy as np
from time import time
import sys
import numpy.lib.recfunctions as rf
import pytest

from sklearn.neighbors import KDTree

import rustimport

rustimport.settings.release_mode = True
import rustimport.import_hook
import ox_vox_nns


TEST_ARRAY = np.arange(9, dtype=np.float32).reshape((3, 3))
ORIGIN = np.array([0, 0, 0], dtype=np.float32)


def compare_performance_find_neighbours() -> int:
    """
    Compare performance of NNS algorithms
    """
    # OxVoxNNS
    num_points = 1_000_000
    search_points = np.random.random((num_points, 3)).astype(np.float32) * 15
    query_points = search_points
    num_neighbours = 800
    max_dist = 0.05
    voxel_size = 0.1
    start = time()
    indices, distances = ox_vox_nns.find_neighbours(
        search_points,
        query_points,
        num_neighbours,
        max_dist,
        voxel_size,
    )
    print(f"Found neighbours using OxVoxNNS in {time()-start}s")

    # SKLearn
    start = time()
    distances, indices = KDTree(search_points, metric="euclidean").query(
        query_points, num_neighbours, return_distance=True, dualtree=False
    )
    print(f"Found neighbours using KDTree in {time()-start}s")

    return 0


if __name__ == "__main__":
    sys.exit(compare_performance_find_neighbours())
