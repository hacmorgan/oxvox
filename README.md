# OxVoxNNS - **Ox**idised **Vox**elised **N**earest **N**eighbour **S**earch

[![PyPI](https://img.shields.io/pypi/v/cibuildwheel.svg)](https://pypi.org/project/ox-vox-nns/)
[![Actions Status](https://github.com/hacmorgan/OxVoxNNS/workflows/CI/badge.svg)](https://github.com/hacmorgan/OxVoxNNS/actions)

A hybrid-ish nearest neighbour search implemented in rust, tailored towards consistent performance, especially on difficult inputs for KDTrees


## Okay but why?
Suppose we are searching for `k` neighbours within radius `r`, OxVox can operate in two different modes depending on desired behaviour and performance (see *Performance* section below).

### Inexact Mode (OxVoxApprox)
Inexact mode is the bread and butter of this module. Instead of finding the `k` *nearest* neighbours, OxVoxApprox finds `k` roughly evenly distributed neighbours within radius `r`, avoiding a lot of expensive L2 distance computations and neighbour sorting.

### Exact Mode
Exact Mode behaves like a conventional *nearest* neighbour search, with decent performance in many situations, though typically worse than open3d's highly performant NearestNeighbourSearch


## Installation
### Precompiled (from PyPI, recommended)
```
pip install ox_vox_nns
```

### Manual
Checkout this repo and enter a virtual environment, then run
```
maturin develop --release
```


## Usage
Basic usage, query a block of query points in *inexact* mode:
```
from ox_vox_nns.ox_vox_nns import OxVoxNNS

indices, distances = ox_vox_nns.OxVoxNNS(
    search_points,   # (S, 3) ndarray
    max_dist,        # float
).find_neighbours(
    query_points,    # (Q, 3) ndarray
    num_neighbours,  # int
    False,           # False => inexact mode
)
```

More complex usage, using a single NNS object for multiple *exact* mode queries (e.g. to distribute the `nns` object and perform queries in parallel, or to query from a large number of query points in batches/chunks)
```
from ox_vox_nns.ox_vox_nns import OxVoxNNS

nns = ox_vox_nns.OxVoxNNS(
    search_points,
    max_dist,
    voxel_size,
)

for query_points_chunk in query_points_chunks:
    chunk_indices, chunk_distances = nns.find_neighbours(
        query_points,
        num_neighbours,
        True,
    )
```


## Performance
See `performance_test_ox_vox_nns.py` for test code.

More rigourous testing is still TODO but will be required in order to construct a rubric for users to gauge more reliably whether OxVox will help them or not.

Rough testing suggests that OxVoxNNS outperforms KDTrees under certain circumstances, particularly with dense clusters and/or big gaps. As a rule of thumb, KDTrees will beat OxVox for more uniformally distributed pointclouds, but OxVoxApprox will win at some point as the points become more clustered and the tree search slows down more significantly
