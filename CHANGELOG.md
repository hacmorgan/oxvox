# Changelog

## 1.0.0 (unreleased)

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
