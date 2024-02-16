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
import json
import sys
import cloudpickle
import pickle
import tqdm
import numpy.lib.recfunctions as rf
from p_tqdm import p_imap
from functools import partial

# Competitor NNS algorithms
from scipy.spatial import KDTree as SciPyKDTree
from sklearn.neighbors import KDTree as SkLearnKDTree
import open3d as o3d

# Our NNS algorithm
from ox_vox_nns.ox_vox_nns import OxVoxNNS

from abyss.bedrock.io.convenience import easy_load

SEARCH_POINTS_IRL = rf.structured_to_unstructured(
    easy_load("/home/hmo/tmp/DEEPL-1947/4.bin")[["x", "y", "z"]]
).astype(np.float32)[::4]
SEARCH_POINTS_1M_UNIFORM = np.random.random((16_000_000, 3)).astype(np.float32) * 15
SEARCH_POINTS_1M_CLUSTERS = np.vstack(
    [
        # Generate a cluster of random points
        np.random.random((1_600_000, 3)).astype(np.float32)
        # Move cluster somewhere randomly
        + np.random.random((1, 3)).astype(np.float32) * 15
    ]
    * 10
)


TEST_INPUTS = {
    "irl": {
        "search_points": SEARCH_POINTS_IRL,
        "query_points": SEARCH_POINTS_IRL,
        "num_neighbours": 800,
        "max_dist": 0.05,
        "voxel_size": 0.05,
        "batch_size": 40_000,
    },
    "1m-uniform": {
        "search_points": SEARCH_POINTS_1M_UNIFORM,
        "query_points": SEARCH_POINTS_1M_UNIFORM,
        "num_neighbours": 800,
        "max_dist": 0.05,
        "voxel_size": 0.1,
        "batch_size": 40_000,
    },
    "1m-clusters": {
        "search_points": SEARCH_POINTS_1M_CLUSTERS,
        "query_points": SEARCH_POINTS_1M_CLUSTERS,
        "num_neighbours": 800,
        "max_dist": 0.05,
        "voxel_size": 0.05,
        "batch_size": 40_000,
    },
}


def compare_performance() -> int:
    """
    Compare performance of NNS algorithms
    """
    # Construct dict for output data
    results = {}

    for data_name, params in TEST_INPUTS.items():

        results[data_name] = {}

        for algo_name, algo in {
            # "scipy": _scipy_nns,
            # "sklearn": _sklearn_nns,
            "oxvox": _oxvox_nns,
            "open3d": _o3d_nns,
            # "sklearn-multiproc": _sklearn_nns_multiproc,
            # "oxvox-multiproc": _oxvox_nns_multiproc,
        }.items():

            print(
                f"Searching for nearest neighbours in dataset {data_name} using algorithm: {algo_name}... "
            )
            sys.stdout.flush()
            start = time()
            indices, distances = algo(**params)
            compute_time = time() - start
            print(f"Done in {compute_time}s")
            sys.stdout.flush()
            results[data_name][algo_name] = compute_time

    print(json.dumps(results))

    return 0


"""
Wrappers around NNS implementations (with common usage semantics) below
"""


def _scipy_nns(
    search_points: npt.NDArray[np.float32],
    query_points: npt.NDArray[np.float32],
    num_neighbours: int,
    **kwargs,
) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    """
    Run nearest neighbour search using scipy
    """
    distances, indices = SciPyKDTree(search_points).query(
        query_points, k=num_neighbours
    )
    return indices, distances


def _sklearn_nns(
    search_points: npt.NDArray[np.float32],
    query_points: npt.NDArray[np.float32],
    num_neighbours: int,
    **kwargs,
) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    """
    Run nearest neighbour search using sklearn (single-threaded)
    """
    distances, indices = SkLearnKDTree(search_points, metric="euclidean").query(
        query_points, k=num_neighbours
    )
    return indices, distances


def _o3d_nns(
    search_points: npt.NDArray[np.float32],
    query_points: npt.NDArray[np.float32],
    num_neighbours: int,
    max_dist: float,
    batch_size: int,
    **kwargs,
) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    """
    Run nearest neighbour search using open3d
    """
    nns = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(search_points))
    nns.hybrid_index()
    # Construct output arrays up front (we will fill them in in chunks)
    num_points = len(query_points)
    # indices = np.full((num_points, num_neighbours), fill_value=-1)
    # distances = np.full((num_points, num_neighbours), fill_value=-1)

    # Construct generator of chunks of query points
    for chunk_offset in range(0, len(query_points), batch_size):
        query_chunk = query_points[chunk_offset : chunk_offset + batch_size, :]
        chunk_indices, chunk_distances, _ = nns.hybrid_search(
            query_chunk, radius=max_dist, max_knn=num_neighbours
        )
        # indices[chunk_offset : chunk_offset + batch_size] = chunk_indices
        # distances[chunk_offset : chunk_offset + batch_size] = chunk_distances

    # return indices, distances
    return None, None


def _oxvox_nns(
    search_points: npt.NDArray[np.float32],
    query_points: npt.NDArray[np.float32],
    num_neighbours: int,
    max_dist: float,
    **kwargs,
) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    """
    Run nearest neighbour search using OxVoxNNS
    """
    nns = OxVoxNNS(search_points, max_dist)
    return nns.find_neighbours(query_points, num_neighbours, False)


def _sklearn_nns_multiproc(
    search_points: npt.NDArray[np.float32],
    query_points: npt.NDArray[np.float32],
    num_neighbours: int,
    batch_size: int,
    **kwargs,
) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    """
    Run nearest neighbour search using sklearn (multiprocessed)
    """
    # Construct tree with search points
    tree = SkLearnKDTree(search_points, metric="euclidean")

    # Construct output arrays up front (we will fill them in in chunks)
    num_points = len(query_points)
    indices = np.full((num_points, num_neighbours), fill_value=-1)
    distances = np.full((num_points, num_neighbours), fill_value=-1)

    # Construct generator of chunks of query points
    query_chunk_offsets = range(0, len(query_points), batch_size)
    query_chunks = (query_points[i : i + batch_size, :] for i in query_chunk_offsets)

    # Map query across chunks of query points
    processed_chunks = p_imap(
        lambda query_chunk: tree.query(query_chunk, k=num_neighbours),
        query_chunks,
        tqdm=partial(tqdm.tqdm, disable=True),  # Disable tqdm bar
    )

    # Insert values back into output array
    for (chunk_indices, chunk_distances), chunk_offset in zip(
        processed_chunks, query_chunk_offsets
    ):
        indices[chunk_offset : chunk_offset + batch_size] = chunk_indices
        distances[chunk_offset : chunk_offset + batch_size] = chunk_distances

    return indices, distances


def _oxvox_nns_multiproc(
    search_points: npt.NDArray[np.float32],
    query_points: npt.NDArray[np.float32],
    num_neighbours: int,
    max_dist: float,
    batch_size: int,
    voxel_size: float,
) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    """
    Run nearest neighbour search using sklearn (multiprocessed)
    """
    # Construct OxVoxNNS object with search points
    nns = OxVoxNNS(search_points, max_dist, voxel_size)

    # Construct output arrays up front (we will fill them in in chunks)
    num_points = len(query_points)
    indices = np.full((num_points, num_neighbours), fill_value=-1)
    distances = np.full((num_points, num_neighbours), fill_value=-1)

    # Construct generator of chunks of query points
    query_chunk_offsets = range(0, len(query_points), batch_size)
    query_chunks = (query_points[i : i + batch_size, :] for i in query_chunk_offsets)

    # Map query across chunks of query points
    processed_chunks = p_imap(
        lambda query_chunk: nns.find_neighbours(query_chunk, num_neighbours),
        query_chunks,
        tqdm=partial(tqdm.tqdm, disable=True),  # Disable tqdm bar
    )

    # Insert values back into output array
    for (chunk_indices, chunk_distances), chunk_offset in zip(
        processed_chunks, query_chunk_offsets
    ):
        indices[chunk_offset : chunk_offset + batch_size] = chunk_indices
        distances[chunk_offset : chunk_offset + batch_size] = chunk_distances

    return indices, distances


if __name__ == "__main__":
    sys.exit(compare_performance())
