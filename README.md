# OxVoxNNS - Oxidised Voxelised Nearest Neighbour Search
A performant (for large numbers of query points) nearest neighbour search implemented in rust

# Usage
Basic usage:

```
from ox_vox_nns.ox_vox_nns import OxVoxNNS

indices, distances = ox_vox_nns.OxVoxNNS(
    search_points,
    max_dist,
    voxel_size,
).find_neighbours(
    query_points,
    num_neighbours
)
```

More complex usage, using a single NNS object for multiple queries

```
from ox_vox_nns.ox_vox_nns import OxVoxNNS

nns = ox_vox_nns.OxVoxNNS(
    search_points,
    max_dist,
    voxel_size,
)

for query_points_chunk in query_points_chunks:
    chunk_indices, chunk_dictances = nns.find_neighbours(
        query_points,
        num_neighbours
    )
```

# Installation
## Precompiled (from PyPI)
Currently only available for linux x86-64
```
pip install ox_vox_nns
```

## Manual
Checkout this repo and enter a virtual environment, then run
```
maturin develop --release
```

# Performance
See `performance_test_ox_vox_nns.py` for test code.

Rough testing suggests that OxVoxNNS outperforms KDTrees under certain circumstances, particularly with higher numbers of query points. Rigourous testing is a WIP
