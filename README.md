# OxVoxNNS - **Ox**idised **Vox**elised **N**earest **N**eighbour **S**earch

[![PyPI](https://img.shields.io/pypi/v/cibuildwheel.svg)](https://pypi.org/project/ox-vox-nns/)
[![Actions Status](https://github.com/hacmorgan/OxVoxNNS/workflows/CI/badge.svg)](https://github.com/hacmorgan/OxVoxNNS/actions)

A hybrid-ish nearest neighbour search implemented in rust, tailored towards consistent performance, especially on difficult inputs for KDTrees


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
Basic usage, query a block of query points in **sparse** mode:
```
import numpy as np
from ox_vox_nns.ox_vox_nns import OxVoxNNS

NUM_POINTS = 100_000
TEST_POINTS = np.random.random((NUM_POINTS, 3))

indices, distances = ox_vox_nns.OxVoxNNS(
    search_points=TEST_POINTS,
    max_dist=0.05,
).find_neighbours(
    query_points=TEST_POINTS,
    num_neighbours=1000,
    sparse=True,
)
```

More complex usage, using a single NNS object for multiple *exact* mode queries (e.g. to distribute the `nns` object and perform queries in parallel, or to query from a large number of query points in batches/chunks)
```
# same imports and test data as above

nns = ox_vox_nns.OxVoxNNS(TEST_POINTS, 0.1)

for query_points_chunk in query_points_chunks:
    chunk_indices, chunk_distances = nns.find_neighbours(
        query_points=query_points_chunk,
        num_neighbours=1,
        sparse=False,
    )
```


## Performance
As a rough heuristic:

- Open3D will edge out OxVox on **easier data**, e.g. uniformally distributed points, though OxVox does outperform SciPy and SKLearn's KDTree implementations
- OxVox in exact mode will outperform even Open3D on harder inputs, e.g. with clusters of very dense points
- OxVox in sparse mode or with `epsilon > 0.0` will dramatically outperform KDTrees too. This is not really a fair comparison, but if you don't strictly need the exact `k` nearest neighbours it can be very helpful

See `performance_test_ox_vox_nns.py` for test code.

More thorough tests and interactive visualisations are still being developed to help a prospective user decide quickly if OxVox is worth trying
