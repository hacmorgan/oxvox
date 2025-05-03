# OxVox - **Ox**idised **Vox**elised toolkit

[![PyPI](https://img.shields.io/pypi/v/cibuildwheel.svg)](https://pypi.org/project/oxvox/)
[![Actions Status](https://github.com/hacmorgan/oxvox/workflows/CI/badge.svg)](https://github.com/hacmorgan/oxvox/actions)

A collection of operations on arrays and pointclouds implemented in Rust


## Installation
### Precompiled (from PyPI, recommended)
```bash
pip install oxvox
```

### Manual
Checkout this repo and enter a virtual environment, then run
```bash
maturin develop --release
```


## Usage
### Indexing by field
```python
In [7]: from oxvox.indexing import indices_by_field
   ...: 
   ...: TEST_ARRAY = np.array(
   ...:     [
   ...:         (2, "a"),
   ...:         (3, "b"),
   ...:         (2, "a"),
   ...:         (3, "c"),
   ...:         (4, "a"),
   ...:         (4, "c"),
   ...:     ],
   ...:     dtype=[
   ...:         ("score", np.int32),
   ...:         ("initial", "|O"),
   ...:     ],
   ...: )
   ...: 
   ...: for row_values, row_indices in indices_by_field(arr=TEST_ARRAY, fields=["score"]):
   ...:     print(f"Unique value: {row_values.tolist()} at row indices {row_indices}")
   ...: 
Unique value: (2,) at row indices [0 2]
Unique value: (3,) at row indices [1 3]
Unique value: (4,) at row indices [4 5]
```

### Nearest Neighbour Search (NNS)
Basic usage, query a block of query points in **sparse** mode:
```python
import numpy as np
from oxvox.nns import OxVoxNNS

NUM_POINTS = 100_000
TEST_POINTS = np.random.random((NUM_POINTS, 3))

indices, distances = OxVoxNNS(
    search_points=TEST_POINTS,
    max_dist=0.05,
).find_neighbours(
    query_points=TEST_POINTS,
    num_neighbours=1000,
    sparse=True,
)
```

More complex usage, using a single NNS object for multiple *exact* mode queries (e.g. to distribute the `nns` object and perform queries in parallel, or to query from a large number of query points in batches/chunks)
```python
# same imports and test data as above

nns = ox_vox_nns.OxVoxNNS(TEST_POINTS, 0.1)

for query_points_chunk in query_points_chunks:
    chunk_indices, chunk_distances = nns.find_neighbours(
        query_points=query_points_chunk,
        num_neighbours=1,
        sparse=False,
    )
```


## Tests
All test files are executable for spot-testing functionality

To run all tests:
```bash
make test
```


## Performance
See performance tests under `performance_tests` directory


## Building & Pushing to PyPI
1. Get modifications made to existing workflow
```bash
diff .github/workflow{_templates,s}/CI.yml > /tmp/CI.patch
```

2. Generate updated CI YAML
```bash
maturin generate-ci github > .github/workflows/CI.yml
```

3. Apply changes to CI YAML (do manually if application of patch fails)
```bash
patch .github/workflows/CI.yml /tmp/CI.patch
```

4. Update `cargo.toml`
```toml
[package]
name = "oxvox"
version = "0.7.1"
...
```

5. Tag with version number and push
```bash
git commit -am "Push version 0.7.1"
git tag 0.7.1
git push --tags
```

