# OxVox - **Ox**idised **Vox**elised toolkit

[![PyPI](https://img.shields.io/pypi/v/oxvox.svg)](https://pypi.org/project/oxvox/)
[![Actions Status](https://github.com/hacmorgan/oxvox/workflows/CI/badge.svg)](https://github.com/hacmorgan/oxvox/actions)

A collection of operations on arrays and pointclouds implemented in Rust

Requires Python 3.10 or newer.


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
`OxVoxNNS` builds a spatial index over `search_points` and answers radius-bounded
k-nearest-neighbour queries against it. Construct one `OxVoxNNS` per pointcloud and
reuse it for as many queries as needed (e.g. to query in batches, or to distribute the
object across processes: it pickles).

Two indices are available via the `method` argument:

- `"voxel"` (default): a uniform grid of cells of side `search_radius / cells_per_radius`,
  stored sorted by cell so each cell is one contiguous slice. A query scans only the
  cells that could hold a neighbour, skipping any whose bounding box is already farther
  away than its current best candidate. Its cost per query is governed by local
  density, not by the shape of the cloud, so it stays consistent on dense, clustered
  data where KD-trees degrade.
- `"kdtree"`: a bucketed KD-tree (median splits on the widest axis, 16-point leaves).
  Cheaper on sparse or very non-uniform clouds, or when the search radius is large
  relative to the point spacing.

`method="auto"` currently resolves to `"voxel"`; a heuristic derived from benchmarks is
planned. `grid_stats()` exposes the voxel grid's occupancy (cell count, max and mean
points per cell) for making that choice yourself.
```python
import numpy as np
from oxvox.nns import OxVoxNNS

NUM_POINTS = 100_000
SEARCH_RADIUS = 0.05
search_points = np.random.random((NUM_POINTS, 3)).astype(np.float32)
query_points = np.random.random((NUM_POINTS, 3)).astype(np.float32)

nns = OxVoxNNS(search_points, SEARCH_RADIUS)            # method="voxel"
tree = OxVoxNNS(search_points, SEARCH_RADIUS, method="kdtree")

# Find up to `num_neighbours` nearest neighbours per query point. `indices`/`distances`
# are (Q, num_neighbours) arrays, padded with -1 where fewer neighbours were found
indices, distances = nns.find_neighbours(query_points, num_neighbours=10)

# Count neighbours within the search radius per query point (a uint32 exact count)
counts = nns.count_neighbours(query_points)

# Or weight each neighbour's contribution by how close it is, via the normalised
# kernel w = (1 - distance / search_radius) ** distance_weight_factor. This is 1 at
# zero distance and falls smoothly to 0 at the search radius; distance_weight_factor=0
# reproduces the exact count above, larger values concentrate weight near the query
weighted_counts = nns.count_neighbours(query_points, distance_weight_factor=1.0)
```

`find_neighbours` and `count_neighbours` both take a `num_threads` argument (default
`0`, meaning "use all available CPUs") and a `progress` flag; `find_neighbours` also
takes an `epsilon`: once `num_neighbours` neighbours closer than `epsilon` have been
found for a query point, its search stops early, which can help avoid getting bogged
down in very dense regions of the search points (results are then approximate).

Both methods return identical results (up to tie ordering); they are tested against a
brute-force reference and against each other.


## Tests
All test files are executable for spot-testing functionality

To run all tests (Rust unit tests, then Python tests):
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

4. Update the version in `Cargo.toml` (this is the single source of truth for the
   package version; `pyproject.toml` picks it up via `dynamic = ["version"]`)
```toml
[package]
name = "oxvox"
version = "1.0.0"
...
```

5. Commit `Cargo.lock` along with the version bump (it is tracked, not gitignored, so
   every release builds from exactly the dependency versions that were tested)

6. Tag with version number and push
```bash
git commit -am "Push version 1.0.0"
git tag 1.0.0
git push --tags
```
