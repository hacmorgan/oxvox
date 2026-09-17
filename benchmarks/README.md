# oxvox benchmarks

Everything here measures oxvox's neighbour-search backends against each other and
against scipy's `cKDTree` and Open3D's hybrid search. It is not shipped in the wheel.

The results that back the README's method advice and the `method="auto"` rule live in
`results/`, one JSON file per (dataset, point count) group, with `results/report.html`
built from them.

## Running it

```bash
pip install -e ".[bench]"                                # scipy, open3d, plotly
maturin develop --release --features kiddo-baseline      # adds method="kiddo"
python -m benchmarks.run --quick                         # smoke grid, a few minutes
python -m benchmarks.run                                 # the full grid, about two hours
python -m benchmarks.report                              # results/report.html
python -m benchmarks.heuristic                           # score the method="auto" rule
```

`--quick` runs two generators at up to 100k points, which is enough to check the
harness end to end. The full grid sweeps point counts up to 4e6, four density targets,
k in {1, 8, 64} and query batches of 1e3, 1e5 and N.

Each dataset's four radii are *calibrated* so that a search sphere really holds about
1, 10, 100 and 1000 of that cloud's points: the closed-form radius only works for the
uniform box, and a cylinder shell or a tight Gaussian cluster is locally hundreds of
times denser, so reusing the uniform radius would put every other dataset in the same
very dense regime and collapse the density axis. The real scans keep physical radii
(1 cm, 5 cm, 20 cm) and the density they realise is recorded next to the timings.

The `kiddo-baseline` feature is optional: without it the `kiddo` backend is simply
absent from the grid, as are scipy and Open3D if they are not installed.

## How the committed results were produced

On 20 cores, in two passes (the second with tighter limits, to keep the whole thing
inside two hours; every file records the limits it ran under in its `options` key):

```bash
# the uniform family, with the full query-batch ladder up to 4e6 and a 60 s cap
python -m benchmarks.run --dataset uniform

# every other family plus the real scans, capped at 1e6 queries and 20 s per run
python -m benchmarks.run \
    --dataset clusters --dataset cylinder --dataset sheet --dataset scene \
    --real-pointcloud "real scan A=<path>" --real-pointcloud "real scan B=<path>" \
    --max-run-seconds 20 --max-queries 1000000
```

The real scans are two 16M-point laser scans that are not public; only their labels,
point counts, realised neighbour densities and timings are in the results.

## Real pointclouds

```bash
python -m benchmarks.run --real-pointcloud "real scan A=/path/to/scan.npy"
```

Only the label is written to the results: no paths, no coordinates, no field names,
nothing but the point count, the realised neighbour density and the timings. `.npy`
files holding an `(N, 3)` array (or a structured array with `x`/`y`/`z` fields) work
anywhere; other formats are handed to `abyss.bedrock.io.convenience.easy_load`, which
is imported lazily and is not a dependency of anything here.

## The pieces

| File | What it does |
|------|--------------|
| `generators.py` | Five seeded synthetic families: uniform box, Gaussian clusters, cylinder shell with axial density falloff, near-planar sheet, and a mixed scene. All fill the same 100 m box |
| `competitors.py` | One interface over scipy, Open3D and every oxvox method, normalising the three padding conventions onto oxvox's (`-1` indices, `-1.0` distances, nearest first) |
| `harness.py` | Times builds and queries, checks answers against scipy, records peak RSS, writes the JSON |
| `run.py` | The grid CLI |
| `report.py` | Builds `results/report.html` |
| `heuristic.py` | Scores the `method="auto"` rule against the measured grid (hit rate and regret), and against every fixed single-method policy |
| `test_generators.py` | Unit tests for the generators (run by `make test`) |

## How a measurement is taken

- One forked worker process per (competitor, radius): it builds the index once and
  answers every (k, Q) combination against it, so a build is never charged to a query
  and a runaway configuration takes down only its own block.
- Warm-up query, then the median of three; a configuration whose warm-up alone takes
  longer than three seconds is timed once, and the run records how many repeats it got.
- Query points are a seeded random sample of the cloud's own points (the self-query
  workload), the same sample for every implementation, with each batch size a prefix of
  it.
- Configurations are visited in ascending query count and projected from the previous
  one; anything projected past `--max-run-seconds` (60 by default) is skipped with a
  marker that says what the projection was. The report lists every skip. The graph
  backend's build is itself a full self-query, so it is projected from the hybrid
  backend's measured query time and skipped when it would exceed `--max-build-seconds`.
- Correctness: on a fixed 2048-query subsample, every exact method must reproduce
  scipy's distances, and the approximate backend's recall is measured. Ties make index
  comparison ambiguous, so index sets are only compared on rows whose distances are all
  distinct.
