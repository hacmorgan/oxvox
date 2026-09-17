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

Several indices are available via the `method` argument. If you have no reason to
choose, use `method="auto"`, which picks the one the benchmark says wins most often:

- `"kdtree"`: a bucketed KD-tree (median splits on the widest axis, 16-point leaves).
  The fastest oxvox backend at most of the benchmark's grid points, on every cloud
  shape measured, and what `"auto"` chooses.
- `"voxel"` (the historical default): a uniform grid of cells of side
  `search_radius / cells_per_radius`, stored sorted by cell so each cell is one
  contiguous slice. A query scans only the cells that could hold a neighbour, skipping
  any whose bounding box is already farther away than its current best candidate. Its
  cost per query is governed by local density rather than by the shape of the cloud,
  which is why it exists, but the benchmark does not support the conclusion that this
  wins: it is a median 1.6x slower than the KD-tree, and on a dense real scan searched
  at 20 cm (about 40 000 points per sphere) it is up to 178x slower, because every
  query then scans tens of thousands of candidates linearly.
- `"hybrid"`: the voxel grid, with a KD subtree inside every cell that holds more than
  `subtree_threshold` points (default 64). Dense cells are searched in logarithmic
  rather than linear time, which does remove the grid's collapse on dense data (10.6 s
  to 1.8 s on that same scan) without ever beating the plain KD-tree. Exact.
- `"graph"`: **approximate**. Each search point is linked to its `graph_degree` (default
  16) nearest neighbours at build time. A query finds its single nearest search point
  exactly through the hybrid grid, then floods outwards along graph edges in
  best-first order. It never returns a point outside the radius and never a duplicate,
  but it can miss neighbours; recall is high for `num_neighbours <= graph_degree` and
  degrades gradually above it. Counts can only ever undercount.
- `"kiddo"`: the `kiddo` crate's KD-tree, present only in builds compiled with the
  `kiddo-baseline` cargo feature. It is a benchmark baseline, not part of released
  wheels.

`method="auto"` resolves to `"kdtree"`. That is the whole rule, and it is what the
measurements support: which backend wins a particular query is decided mostly by
`num_neighbours` and the size of the query batch, neither of which is known when the
index is built, and rules keyed on what *is* knowable (point count, grid occupancy)
scored no better than always choosing the KD-tree. See
`oxvox.nns.choose_auto_method` for the numbers, and `grid_stats()` if you want to look
at the grid's occupancy (cell count, max and mean points per cell) and decide yourself.

### Which method should I use?
From the benchmark in [`benchmarks/`](benchmarks/README.md) (five synthetic cloud
families and two 16M-point laser scans, point counts from 1e4 to 1.6e7, k in
{1, 8, 64}, query batches from 1e3 to 4e6; full results and charts in
[`benchmarks/results/report.html`](benchmarks/results/report.html)):

- **Use `"kdtree"` (or `"auto"`) unless you have measured otherwise.** It is the
  fastest exact oxvox backend at 56% of grid points and the fastest or within a couple
  of percent of it at the median; its worst case against the best backend of the moment
  is 1.5x at the 90th percentile.
- Against **scipy's cKDTree** it is a median 2.1x faster on queries (10th percentile
  1.1x, 90th 3.1x) and a median 5.6x faster to build (8.4x from a million points up);
  scipy only wins on tiny query batches,
  where it can be up to 3.3x faster because oxvox's per-call thread pool costs more
  than the query.
- Against **Open3D's hybrid search** it is a median 1.9x faster, and up to 22x faster
  on dense clouds with small k, but Open3D wins at `num_neighbours=64` on clustered
  data (up to 3.4x).
- For `count_neighbours`, `"kdtree"` is fastest on 60% of measurements and a median
  3-5x faster than scipy's `query_ball_point`.
- `"graph"` stays an experiment: its kNN recall is high (median 1.0000, minimum 0.9853,
  at least 0.99 on 98% of checks) but it is several times slower than the exact
  backends, and its counts are badly low on dense clouds.

### Example
```python
import numpy as np
from oxvox.nns import OxVoxNNS

NUM_POINTS = 100_000
SEARCH_RADIUS = 0.05
search_points = np.random.random((NUM_POINTS, 3)).astype(np.float32)
query_points = np.random.random((NUM_POINTS, 3)).astype(np.float32)

nns = OxVoxNNS(search_points, SEARCH_RADIUS, method="auto")   # picks "kdtree"
grid = OxVoxNNS(search_points, SEARCH_RADIUS, method="voxel")  # the historical default

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

### A note on `fork`
Queries run on a rayon thread pool, and thread pools do not survive `fork`. In a process
that has already run an oxvox query, a child process created by `multiprocessing`'s
default `fork` start method on Linux cannot run one: its first parallel operation waits
on worker threads that do not exist in the child, forever. If you want to query from
several processes, either use the `"spawn"` start method
(`multiprocessing.get_context("spawn")`), or keep every oxvox call in the child
processes and none in the parent.

The exact methods (`voxel`, `kdtree`, `hybrid`) return identical results up to tie
ordering; they are tested against a brute-force reference and against each other.
`oxvox.nns.EXACT_METHODS` lists them, and `oxvox.nns.available_methods()` lists every
method compiled into the installed build.


## Tests
All test files are executable for spot-testing functionality

To run all tests (Rust unit tests, then Python tests):
```bash
make test
```


## Performance
`benchmarks/` holds the comparison harness (every oxvox backend against scipy and
Open3D, over synthetic and real pointclouds), the committed results, and the report
built from them:

```bash
pip install -e ".[bench]"
maturin develop --release --features kiddo-baseline
python -m benchmarks.run --quick        # smoke grid, a few minutes
python -m benchmarks.report             # benchmarks/results/report.html
```

See [`benchmarks/README.md`](benchmarks/README.md) for the full grid and how a
measurement is taken, and [`benchmarks/results/report.html`](benchmarks/results/report.html)
for the charts and tables behind the advice above.


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
