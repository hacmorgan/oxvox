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
from typing import Tuple
import numpy.typing as npt
import sys
import numpy.lib.recfunctions as rf
from p_tqdm import p_imap
from functools import partial

from sklearn.neighbors import KDTree

from ox_vox_nns import OxVoxNNS


NUM_POINTS = 1_000_000
TEST_ARRAY = np.random.random((NUM_POINTS, 3)).astype(np.float32) * 15




def compare_performance_find_neighbours(test_data: npt.NDArray[np.float32]) -> int:
    """
    Compare performance of NNS algorithms
    """
    # NNS parameters and test data
    search_points = test_data
    query_points = test_data
    num_neighbours = 800

    # OxVoxNNS
    max_dist = 0.05
    voxel_size = 0.1
    start = time()
    indices, distances = OxVoxNNS(search_points, max_dist, voxel_size).find_neighbours(
        query_points,
        num_neighbours,
    )
    print(f"Found neighbours using OxVoxNNS in {time()-start}s")

    # # SKLearn (Compiled C++)
    # start = time()
    # distances, indices = KDTree(search_points, metric="euclidean").query(
    #     query_points, num_neighbours, return_distance=True, dualtree=False
    # )
    # print(f"Found neighbours using KDTree in {time()-start}s")
    
    # SKLearn with python parallelism
    batch_size = 5000
    start = time()
    # Construct output arrays up front (we will fill them in in chunks)
    indices = np.full(
        (NUM_POINTS, num_neighbours), fill_value=-1
    )
    distances = np.full(
        (NUM_POINTS, num_neighbours), fill_value=-1
    )
    # Map processing function across batches of query points
    batch_start_indices = range(0, NUM_POINTS, batch_size)
    query_point_batches = (
        query_points[i : i + batch_size, :] for i in batch_start_indices
    )
    process_batch_preconfigured = partial(process_batch, tree=KDTree(search_points, metric="euclidean"), num_neighbours=num_neighbours)
    processed_batches = p_imap(process_batch_preconfigured, query_point_batches)
    # Insert reults back into array
    for batch_idx, (indices_batch, distances_batch) in zip(batch_start_indices, processed_batches):
        indices[batch_idx : batch_idx + batch_size, :] = indices_batch
        distances[batch_idx : batch_idx + batch_size, :] = distances_batch
    print(f"Found neighbours using parallel KDTree in {time()-start}s")

    # Scipy (Native Python - TODO)

    # Open3d (C++ with parallelism - TODO)

    return 0


def process_batch(
    query_points_chunk: npt.NDArray[np.float32], tree: KDTree, num_neighbours: int
) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:

    # Find neighbours for this chunk of points
    return tree.query(query_points_chunk, num_neighbours, return_distance=True, dualtree=False)

    # print(f"Found neighbours using OxVoxNNS in {time()-start}s")

    # # SKLearn (Compiled C++)
    # start = time()
    # distances, indices = KDTree(search_points, metric="euclidean").query(
    #     query_points, num_neighbours, return_distance=True, dualtree=False
    # )
    # print(f"Found neighbours using KDTree in {time()-start}s")

    # # Scipy (Native Python - TODO)

    # # Open2d (C++ with parallelism - TODO)

    # return 0


if __name__ == "__main__":
    sys.exit(compare_performance_find_neighbours(test_data=TEST_ARRAY))
