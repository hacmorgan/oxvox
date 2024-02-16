#!/usr/bin/env -S python3 -m pytest -vvv


"""
Unit tests for rust binding test library

Run this test script to verify that functions can be compiled and run, and produce 
expected results

n.b. The rust module needs to be compiled the first time this is run, but pytest will
hide the output of the rust compiler, so it may appear to hang for a little while.
Subsequent compilations should be much shorter
"""


import numpy as np
import numpy.lib.recfunctions as rf
import pytest

import ox_vox_nns


TEST_ARRAY = np.arange(9, dtype=np.float32).reshape((3, 3))
ORIGIN = np.array([0, 0, 0], dtype=np.float32)


# def test_voxelise_points() -> None:
#     """
#     Test point voxelisation
#     """
#     search_points = TEST_ARRAY
#     voxel_size = 3
#     voxel_indices, points_by_voxel = ox_vox_nns.voxelise_points(
#         search_points, voxel_size
#     )
#     assert np.all(voxel_indices == np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]))
#     assert points_by_voxel == {(0, 0, 0): [0], (1, 1, 1): [1], (2, 2, 2): [2]}


# def test_voxel_offsets() -> None:
#     """
#     Test construction of offsets for voxel coordinates
#     """
#     offsets = ox_vox_nns.construct_voxel_offsets(2)
#     assert offsets[1] == [
#         [-1, -1, -1],
#         [-1, -1, 0],
#         [-1, -1, 1],
#         [-1, 0, -1],
#         [-1, 0, 0],
#         [-1, 0, 1],
#         [-1, 1, -1],
#         [-1, 1, 0],
#         [-1, 1, 1],
#         [0, -1, -1],
#         [0, -1, 0],
#         [0, -1, 1],
#         [0, 0, -1],
#         [0, 0, 1],
#         [0, 1, -1],
#         [0, 1, 0],
#         [0, 1, 1],
#         [1, -1, -1],
#         [1, -1, 0],
#         [1, -1, 1],
#         [1, 0, -1],
#         [1, 0, 0],
#         [1, 0, 1],
#         [1, 1, -1],
#         [1, 1, 0],
#         [1, 1, 1],
#     ]


# def test_find_neighbours() -> None:
#     """
#     Test a simple case of finding neighbours in a small pointcloud
#     """
#     search_points = np.array(
#         [
#             [0.1, 0.1, 0.2],  # Point 0: Point 1's neighbour
#             [0.2, 0.2, 0.1],  # Point 1: Point 0's neighbour
#             [3.2, 1.2, 1.1],  # Point 2: A neighbour to Points 0 and 1 if r > 3 (ish)
#             [3.3, 1.1, 1.0],  # Point 3: Similar to Point 2, also Point 2's neighbour
#         ],
#         dtype=np.float32,
#     )
#     query_points = search_points[0].reshape(1, -1)
#     num_neighbours = 3
#     max_dist = 4.0
#     voxel_size = 0.3
#     indices, distances = ox_vox_nns.find_neighbours(
#         search_points,
#         query_points,
#         num_neighbours,
#         max_dist,
#         voxel_size,
#     )
#     assert np.all(indices == [0, 1, 2])
#     assert np.allclose(distances, [0.0, 0.173, 3.410], atol=0.001)


def test_find_neighbours() -> None:
    """
    Test a simple case of finding neighbours in a small pointcloud
    """
    search_points = np.array(
        [
            [0.1, 0.1, 0.2],  # Point 0: Point 1's neighbour
            [0.2, 0.2, 0.1],  # Point 1: Point 0's neighbour
            [3.2, 1.2, 1.1],  # Point 2: A neighbour to Points 0 and 1 if r > 3 (ish)
            [3.3, 1.1, 1.0],  # Point 3: Similar to Point 2, also Point 2's neighbour
        ],
        dtype=np.float32,
    )
    query_points = search_points[0].reshape(1, -1)
    num_neighbours = 3
    max_dist = 4.0
    voxel_size = 0.3

    nns = ox_vox_nns.OxVoxNNS(search_points, max_dist)
    indices, distances = nns.find_neighbours(query_points, num_neighbours, True)

    assert np.all(indices == [0, 1, 2])
    assert np.allclose(distances, [0.0, 0.173, 3.410], atol=0.001)
