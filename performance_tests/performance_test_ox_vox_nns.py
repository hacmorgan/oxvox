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
from typing import Tuple, Dict, Callable, Any
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import sys
import cloudpickle
import pickle
from colorsys import hsv_to_rgb
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
SEARCH_POINTS_4M_UNIFORM = np.random.random((4_000_000, 3)).astype(np.float32) * 15
SEARCH_POINTS_4M_CLUSTERS = np.vstack(
    [
        # Generate a cluster of random points
        np.random.random((400_000, 3)).astype(np.float32) * 0.5
        # Move cluster somewhere randomly
        + np.random.random((1, 3)).astype(np.float32) * 15
    ]
    * 10
)
SEARCH_POINTS_1M_LOTS_CLUSTERS = np.vstack(
    [
        # Generate a cluster of random points
        np.random.random((10_000, 3)).astype(np.float32)
        # Move cluster somewhere randomly
        + np.random.random((1, 3)).astype(np.float32) * 15
    ]
    * 100
)


TEST_INPUTS = {
    "4m-uniform": {
        "search_points": SEARCH_POINTS_4M_UNIFORM,
        "query_points": SEARCH_POINTS_4M_UNIFORM,
    },
    # "4m-uniform-800": {
    #     "search_points": SEARCH_POINTS_4M_UNIFORM,
    #     "query_points": SEARCH_POINTS_4M_UNIFORM,
    #     "num_neighbours": 800,
    #     "max_dist": 0.05,
    #     "batch_size": 40_000,
    #     # "parallel": False,
    #     "parallel": True,
    #     "epsilon": 0.000,
    # },
    # "4m-clusters": {
    #     "search_points": SEARCH_POINTS_4M_CLUSTERS,
    #     "query_points": SEARCH_POINTS_4M_CLUSTERS,
    #     # "num_neighbours": 800,
    #     "num_neighbours": 8,
    #     "max_dist": 0.05,
    #     "voxel_size": 0.05,
    #     "batch_size": 40_000,
    # },
    # "4m-lots-clusters": {
    #     "search_points": SEARCH_POINTS_1M_LOTS_CLUSTERS,
    #     "query_points": SEARCH_POINTS_1M_LOTS_CLUSTERS,
    #     "num_neighbours": 800,
    #     # "num_neighbours": 8,
    #     "max_dist": 0.05,
    #     # "max_dist": 0.01,
    #     "parallel": True,
    #     "epsilon": 0.00,
    #     # "epsilon": 0.005,
    # },
    # "irl-1": {
    #     "search_points": SEARCH_POINTS_IRL,
    #     "query_points": SEARCH_POINTS_IRL,
    #     "num_neighbours": 1,
    #     "max_dist": 0.05,
    #     "voxel_size": 0.05,
    #     "batch_size": 40_000,
    #     "parallel": True,
    #     "epsilon": 0.000,
    # },
    # "irl-800": {
    #     "search_points": SEARCH_POINTS_IRL,
    #     "query_points": SEARCH_POINTS_IRL,
    #     "num_neighbours": 800,
    #     "max_dist": 0.05,
    #     "voxel_size": 0.05,
    #     "batch_size": 40_000,
    # },
}


def compare_performance_exact() -> int:
    """
    Compare performance of exact NNS algorithms
    """
    # Construct dict for output data
    results_exact = {}
    results_approx = {}

    for data_name, params in TEST_INPUTS.items():

        # Compare exact algos
        results_exact[data_name] = {}
        for algo_name, algo in {
            "scipy": _scipy_nns,
            "sklearn": _sklearn_nns,
            "Open3D": _o3d_nns,
            "OxVoxExact": partial(_oxvox_nns, sparse=False),
        }.items():
            results_exact[data_name][algo_name] = _run_test(
                data_name=data_name, algo_name=algo_name, algo=algo, params=params
            )
        with open(f"{data_name}_exact.html", mode="w", encoding="utf-8") as vis_fd:
            vis_fd.write(generate_plot(results_exact[data_name]))

        # Compare approximate algos
        results_approx[data_name] = {}
        for algo_name, algo in {
            "scipy_eps": partial(_scipy_nns, epsilon=0.005),
            "OxVoxApprox": partial(_oxvox_nns, sparse=False, epsilon=0.005),
            "OxVoxSparse": partial(_oxvox_nns, sparse=True),
        }.items():
            results_approx[data_name][algo_name] = _run_test(
                data_name=data_name, algo_name=algo_name, algo=algo, params=params
            )
        with open(f"{data_name}_approx.html", mode="w", encoding="utf-8") as vis_fd:
            vis_fd.write(generate_plot(results_approx[data_name]))

    # Dump results as json for later review
    print(json.dumps(results_exact) + "\n" * 3)
    print(json.dumps(results_approx))

    return 0


def _run_test(
    data_name: str,
    algo_name: str,
    algo: Callable,
    params: Dict[str, Any],
):
    """
    Run tests for a given algorithm and dataset, for various values of k and r
    """
    results = np.zeros((4, 4))
    for j, num_neighbours in enumerate(
        [
            1,
            #     10,
            #     100,
            #     300,
        ]
    ):
        for i, search_radius in enumerate(
            [
                0.001,
                # 0.005,
                # 0.1,
                # 0.2,
            ]
        ):
            print(
                f"Searching for {num_neighbours} nearest neighbours within a"
                f" {search_radius} unit radius in dataset {data_name} using"
                f" algorithm: {algo_name}... "
            )
            sys.stdout.flush()
            start = time()
            indices, distances = algo(
                **params, num_neighbours=num_neighbours, max_dist=search_radius
            )
            compute_time = time() - start
            print(f"Done in {compute_time}s")
            sys.stdout.flush()
            results[j, i] = compute_time
    return results


def generate_plot(test_results: Dict[str, npt.NDArray[np.floating]]) -> str:
    """
    Plot results of tests with plotly

    Args:
        test_results: Compute times from experiments, indexed by experiment name

    Returns:
        Plotly visualisation HTML contents
    """
    # fig = make_subplots(
    #     rows=1,
    #     cols=2,
    #     specs=[[{"is_3d": True}, {"is_3d": True}]],
    #     subplot_titles=[
    #         "Color corresponds to z",
    #         "Color corresponds to distance to origin",
    #     ],
    # )

    # fig.add_trace(go.Surface(x=x, y=y, z=z, colorbar_x=-0.07), 1, 1)
    # fig.add_trace(go.Surface(x=x, y=y, z=z, surfacecolor=x**2 + y**2 + z**2), 1, 2)
    # fig.update_layout(title_text="Ring cyclide")
    # fig.show()

    # Compute a distinct colour for each surface in the plot
    # colours = (
    #     np.array(hsv_to_rgb(h=hue, s=1, v=1)) * 255
    #     for hue in np.arange(0, 1, len(test_results))
    # )

    fig = go.Figure(
        data=[
            go.Surface(
                z=results,
                name=name,
                showscale=False,
                opacity=0.9,
                # surfacecolor=colour.tolist(),
            )
            for name, results in test_results.items()
        ]
    )
    fig.update_layout(
        scene={"xaxis_title": "Search radius", "yaxis_title": "Num neighbours"}
    )

    view = False
    embed_plotly = False
    output_path = None

    if view:
        fig.show()

    # Generate visualisation as html string
    html_str = fig.to_html(include_plotlyjs=True if embed_plotly else "cdn")

    # Make link to parent dir for navigation
    html_str = html_str.replace(
        "<body>",
        "<body><a href=.>Parent directory</a>\n",
    )

    # Dump to file
    if output_path:
        with open(output_path, mode="w", encoding="utf-8") as output_fd:
            output_fd.write(html_str)

    return html_str


"""
Wrappers around NNS implementations (with common usage semantics) below
"""


def _scipy_nns(
    search_points: npt.NDArray[np.float32],
    query_points: npt.NDArray[np.float32],
    num_neighbours: int,
    epsilon: float = 0,
    **kwargs,
) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    """
    Run nearest neighbour search using scipy
    """
    distances, indices = SciPyKDTree(search_points).query(
        query_points, k=num_neighbours, workers=-1, eps=epsilon
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
    **kwargs,
) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    """
    Run nearest neighbour search using open3d
    """
    nns = o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(search_points))
    nns.hybrid_index()

    # Construct generator of chunks of query points
    indices, distances, _ = nns.hybrid_search(
        query_points, radius=max_dist, max_knn=num_neighbours
    )
    return indices, distances


def _oxvox_nns(
    search_points: npt.NDArray[np.float32],
    query_points: npt.NDArray[np.float32],
    num_neighbours: int,
    max_dist: float,
    sparse: bool,
    num_threads: int = 0,
    epsilon: float = 0,
    l2_distance: bool = True,
    **kwargs,
) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    """
    Run nearest neighbour search using OxVoxNNS
    """
    start = time()
    nns = OxVoxNNS(search_points, max_dist)
    print(f"Constructed NNS searcher in {time() -start}s")
    return nns.find_neighbours(
        query_points, num_neighbours, sparse, l2_distance, num_threads, epsilon
    )


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
    Run nearest neighbour search using OxVox with Python multiprocessing
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
    sys.exit(compare_performance_exact())

    # sys.exit(
    #     sys.stdout.write(
    #         generate_plot(
    #             {
    #                 "foo": np.eye(4) + 3,
    #                 "bar": np.eye(4) + 4,
    #             }
    #         )
    #     )
    # )
