# Changelog

## 1.1.0 (unreleased)

### Added
- `OxVoxNNS(..., method="kdtree")`: a hand-rolled bucketed KD-tree as an alternative to
  the voxel grid. Both backends share one query engine and are tested against a
  brute-force reference; results are identical up to tie ordering.
- `method="hybrid"`: the voxel grid with a KD subtree inside every cell holding more
  than `subtree_threshold` points (default 64). Exact; on the 4M-point dense-cluster
  scenario it cuts the grid's query time from 0.72 s to 0.47 s (the plain KD-tree does
  it in 0.14 s), and matches the grid exactly on uniform data where no cell is dense.
- `method="graph"` (approximate): a kNN graph of degree `graph_degree` (default 16)
  built with the hybrid grid; queries enter at their exact nearest search point and
  flood outwards best-first. Guarantees: never a point outside the radius, never a
  duplicate, trailing -1 padding, counts never exceed the exact count. Recall is
  measured, not promised. Across the benchmark grid its kNN recall has a median of
  1.0000 and a minimum of 0.9853, and is at least 0.99 on 98% of the checks, but it is
  a median 3.2x slower than the fastest exact backend (90th percentile 14x, worst 64x)
  and its counts are badly low on dense clouds (agreeing with scipy on as few as 2.3%
  of query points), because a flood that stops at the k-th best neighbour has no reason
  to visit every point in the radius. It is an experiment, not a recommendation.
- `oxvox.nns.EXACT_METHODS`, and `grid_stats()` now also reports `num_subtrees`
  (hybrid) / `graph_degree` (graph).
- `OxVoxNNS(..., cells_per_radius=n)`: for the voxel method, how many grid cells span
  one search radius (default 1, the classic 27-cell neighbourhood).
- `OxVoxNNS.method`, `len(nns)`, `OxVoxNNS.grid_stats()` (cell count, max and mean points
  per cell) and `oxvox.nns.available_methods()`.
- `progress=True` on `find_neighbours`/`count_neighbours` shows a progress bar; bars are
  now off by default.
- Cargo feature `kiddo-baseline` compiles in the `kiddo` crate as `method="kiddo"` for
  benchmarking (not in released wheels). **It stays behind the feature flag**: the plan
  was to promote it to a standard backend if it matched or beat the hand-rolled
  KD-tree, and over the 708 measured grid points it does not. Query time, kiddo
  relative to `kdtree`: median 1.07x, 10th percentile 0.69x, 90th percentile 2.25x;
  `kdtree` is more than 5% faster at 52% of grid points, kiddo at 33%, and the rest are
  within 5%. Per cloud family, kiddo only wins on the uniform box (median 0.91x) and
  loses on the cylinder shell (1.24x), the planar sheet (1.23x) and both real laser
  scans (1.25x and 1.31x). Builds are a median 1.13x slower (up to 3.34x); its one
  clear win is memory, using a median 0.71x the resident memory of our tree. As a fixed
  choice it would be the wrong default (fastest at 39% of grid points against 56%), so
  a permanent third-party dependency in every released wheel is not justified.
- `benchmarks/`: a comparison harness for every backend against scipy's `cKDTree` and
  Open3D's hybrid search, over five seeded synthetic pointcloud families (uniform box,
  Gaussian clusters, cylinder shell with axial density falloff, near-planar sheet, mixed
  scene) and any real pointclouds given on the command line. Build and query time are
  measured separately (warm-up, then the median of three) in one forked worker per
  index, radii are calibrated per cloud so a density column really means that many
  points per search sphere, every exact implementation's answers are checked against
  scipy's, and peak RSS is recorded. `python -m benchmarks.run [--quick]` writes the
  JSON, `python -m benchmarks.report` builds `benchmarks/results/report.html`, and
  `python -m benchmarks.heuristic` scores the `method="auto"` rule against the results.
  `performance_tests/` and `benchmarks/steelman_voxel.py` are removed; their scenarios
  are generator families now.
- `method="auto"` now chooses a backend instead of standing in for `"voxel"`, through
  `oxvox.nns.choose_auto_method`. It resolves to `"kdtree"`, and that is the whole
  rule, because that is all the measurements support. Scored over the 707 grid points
  where all four exact backends ran: always-kdtree is fastest at 56% of them (median
  1.00x, p90 1.48x of the best backend at that point), always-kiddo at 39% (1.09x,
  2.31x), always-voxel at 3% (1.71x, 4.88x), always-hybrid at 2% (1.82x, 3.86x). The
  ceiling for any construction-time rule - an oracle that may choose per cloud and
  radius, but not per `num_neighbours` or per query batch, since neither is known when
  the index is built - is 70% (1.00x, 1.20x). Rules keyed on the grid occupancy (mean
  and maximum points per cell, both cheap to compute before building) bought two points
  of hit rate and a worse worst case, so none ships; `python -m benchmarks.heuristic`
  re-scores the rule against the recorded results.
- README documents that a rayon thread pool does not survive `fork`: a process that has
  already run an oxvox query cannot run one in a child created by `multiprocessing`'s
  default start method on Linux (the child waits on worker threads that do not exist).
  Use the `"spawn"` start method, or keep every oxvox call in the children.

### Changed
- **The voxel grid is no longer the method to reach for, and the benchmark says so.**
  Measured against the hand-rolled KD-tree over the whole grid, the grid is a median
  1.6x slower, is the fastest exact backend at 3% of grid points (all of them
  sub-millisecond runs), and collapses on genuinely dense data: on a 16M-point laser
  scan searched at 20 cm (about 40 000 points per sphere) it takes 10.6 s to the
  KD-tree's 0.06 s for a million single-neighbour queries. `hybrid` removes that
  collapse (1.8 s on the same case) without ever beating the tree. The grid keeps its
  place - it is still exact, still the backend whose cost depends only on local density,
  and `cells_per_radius=2` halves its clustered-data cost - but `auto` does not pick it
  and the README no longer recommends it.
- Voxel backend rewritten for speed. Points are stored sorted by cell (CSR layout)
  instead of a hashmap of per-cell vectors; the scan compares squared distances and
  keeps only the k best candidates in a bounded heap (was: sqrt per candidate and a heap
  of every candidate in range); cells whose bounding box is beyond the current best
  candidate are skipped; queries are processed in parallel chunks that write straight
  into the output arrays. Voxel coordinates now use `floor`, so cells on the x/y/z = 0
  planes are no longer twice as wide (which made queries near those planes scan up to
  8x more candidates). Measured against 0.7.2 on the historical 4M-point scenarios:
  queries 3-11x faster, builds up to 1.6x faster, identical neighbours and distances.
- Radius comparisons are now done on squared distances (`d² < r²` rather than
  `sqrt(d²) < r`). A search point whose distance is within float32 rounding of the
  radius can therefore flip in or out compared with 1.0.0; over 1.5 billion counted
  neighbours in the clustered benchmark, 240 flipped.
- Rust engine restructured into `src/index/{mod,voxel,kdtree}.rs`; `src/nns.rs` removed.
- Input arrays are made C-contiguous before being passed to Rust, so sliced/transposed
  views work.

### Breaking
- `OxVoxNNSEngine` constructor and method signatures changed (keyword arguments
  `method`, `cells_per_radius`, `progress`). The Python `OxVoxNNS` wrapper is
  backwards compatible for positional use. Pickles from 1.0.0 do not load in 1.1.0.

## 1.0.0 (2026-09-17)

### Added
- Distance-weighted neighbour counts: `OxVoxNNS.count_neighbours` takes an optional
  `distance_weight_factor` (`p`). Each neighbour within the search radius contributes
  `(1 - distance / search_radius) ** p` instead of a flat 1, via a normalised kernel
  that is 1 at zero distance and falls to 0 at the search radius. `p=None` (the
  default) keeps the old exact-count behaviour (`uint32` output); any given `p`
  (including `0`) returns a `float32` weighted sum. Negative `p` raises `ValueError`.
- `cp310-abi3` wheels: one wheel per platform now covers Python 3.10 and every later
  CPython release, so new Python versions no longer need a new oxvox release.
- The GIL is now released for the duration of `find_neighbours` and
  `count_neighbours`, so other Python threads aren't blocked while a query runs.

### Changed
- Minimum supported Python is now 3.10 (3.8 and 3.9 are dropped). This was already
  true in practice: the package uses PEP 604 `X | Y` type syntax, which raises at
  import on 3.8/3.9 despite the previous `requires-python = ">=3.8"` declaration, so
  every previously published cp38/cp39 wheel was broken on import.
- `num_threads` is now honoured on every call to `find_neighbours`/`count_neighbours`.
  Previously, only the first call's `num_threads` value ever took effect, because
  rayon's global thread pool was configured once per process; every query now builds
  and installs its own thread pool for the duration of that call.
- `indices_by_field` (the Rust engine behind `oxvox.indexing.indices_by_field`) is now
  a single O(n+u) sequential pass with a per-id write cursor, replacing the previous
  O(n·u) loop (one pass per unique id). Output is unchanged.
- Migrated to pyo3 0.29 / numpy 0.29 (from pyo3 0.18 / numpy 0.18) and ndarray 0.17.2
  (from 0.15.6), and moved to Rust edition 2024.

### Fixed
- The Linux CI job's test step ran `make test -vvv`; GNU make treats `-v` as
  `--version` and exits without running any target, so Linux CI has never actually
  run the test suite. It now runs `make test`.
- `oxvox.OxVoxNNSEngine` (the pyclass backing `OxVoxNNS`) declares `module = "oxvox"`,
  which pickle requires to look the class back up on unpickling, but the top-level
  `oxvox` package never re-exported it. Every attempt to pickle an `OxVoxNNS` has
  raised since this project's first commit; it is now re-exported from
  `oxvox/__init__.py` and covered by a pickle round-trip regression test.
- macOS CI runners updated from the removed `macos-13` / deprecated `macos-14` to
  `macos-15-intel` (x86_64) and `macos-latest` (aarch64).

### Breaking
- Python 3.8 and 3.9 are no longer supported.
- **Pickle format**: pickles are not guaranteed to load across oxvox releases. This is
  not a regression from 0.7.x, where pickling `OxVoxNNS` never worked at all (see Fixed).
