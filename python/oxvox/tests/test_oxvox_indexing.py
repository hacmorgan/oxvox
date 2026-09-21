#!/usr/bin/env -S pytest -vvv

"""
Unit tests for indexing operations
"""

import numpy as np
import pytest

from oxvox._oxvox import indices_by_field as indices_by_field_rust
from oxvox.indexing import indices_by_field
from oxvox.util import _default_dtype


PLATFORM_DTYPE = _default_dtype()


TEST_ARRAY = np.array(
    [
        ("foo", 2, 0.5, 15),
        ("bar", 3, 1.5, 15),
        ("baz", 2, 2.5, 30),
        ("bat", 3, 3.5, 30),
        ("bet", 4, 4.5, 15),
        ("bot", 4, 5.5, 30),
    ],
    dtype=[
        ("field", "|O"),
        ("a", np.int32),
        ("b", np.float32),
        ("c", np.int32),
    ],
)


def test_indices_by_field() -> None:
    """
    Test that indices_by_field returns the correct row indices for each value in a given field
    """

    # Test a single field
    for computed, expected in zip(
        sorted(list(indices_by_field(arr=TEST_ARRAY, fields="a"))),
        [
            (2, [0, 2]),
            (3, [1, 3]),
            (4, [4, 5]),
        ],
    ):
        assert computed[0].tolist() == expected[0]
        assert computed[1].tolist() == expected[1]

    # Test multiple fields
    for computed, expected in zip(
        sorted(
            list(indices_by_field(arr=TEST_ARRAY, fields=["a", "c"])),
            key=lambda x: tuple(x[0]),
        ),
        [
            ((2, 15), [0]),
            ((2, 30), [2]),
            ((3, 15), [1]),
            ((3, 30), [3]),
            ((4, 15), [4]),
            ((4, 30), [5]),
        ],
    ):
        assert computed[0].tolist() == expected[0]
        assert computed[1].tolist() == expected[1]


def _reference_indices(values: np.ndarray) -> dict:
    """
    Brute-force oracle: the row indices of every distinct value, via np.flatnonzero
    """
    return {value: np.flatnonzero(values == value).tolist() for value in np.unique(values)}


@pytest.mark.parametrize("num_rows", [1, 2, 9, 10, 1024])
def test_single_valued_field_returns_every_row(num_rows: int) -> None:
    """
    Regression test for the 0.7.2 off-by-one: a field with a single value is one group
    holding all n rows, and the old early exit dropped the final row and reported row 0
    twice. Every row must come back exactly once, in order
    """
    arr = np.zeros(num_rows, dtype=[("group", np.int32), ("payload", np.float32)])
    arr["group"] = 7

    results = list(indices_by_field(arr=arr, fields="group"))
    assert len(results) == 1
    value, indices = results[0]
    assert value == 7
    assert indices.tolist() == list(range(num_rows))


def test_indices_by_field_matches_flatnonzero_oracle() -> None:
    """
    Randomised property test: over varied row counts and group counts, including the
    boundary where one group holds n - 1 rows, the row indices for every value must equal
    a brute-force np.flatnonzero reference
    """
    rng = np.random.default_rng(seed=2026)
    for trial in range(300):
        num_rows = int(rng.integers(1, 200))
        num_groups = int(rng.integers(1, min(num_rows, 12) + 1))
        values = rng.integers(0, num_groups, size=num_rows)
        # Every few trials, force the near-degenerate layout the 0.7.2 bug lived in
        if trial % 5 == 0 and num_rows > 1:
            values[:] = 3
            values[int(rng.integers(0, num_rows))] = 4
        arr = np.zeros(num_rows, dtype=[("group", np.int64)])
        arr["group"] = values

        computed = {value: indices.tolist() for value, indices in indices_by_field(arr=arr, fields="group")}
        assert computed == _reference_indices(values), f"trial {trial}"


def test_rust_engine_rejects_counts_that_do_not_match_the_rows() -> None:
    """
    Direct callers of the Rust function must get ValueError, never padded output, when
    counts are overstated or understated, when ids are out of range or negative, and an
    empty input must round-trip cleanly
    """
    row_ids = np.array([0, 1, 0, 1], dtype=np.int64)

    exact = indices_by_field_rust(row_ids, np.array([2, 2], dtype=np.int64))
    assert {key: value.tolist() for key, value in exact.items()} == {0: [0, 2], 1: [1, 3]}

    with pytest.raises(ValueError, match="appears 2 times but its count is 3"):
        indices_by_field_rust(row_ids, np.array([3, 2], dtype=np.int64))
    with pytest.raises(ValueError, match="more times than its count"):
        indices_by_field_rust(row_ids, np.array([1, 2], dtype=np.int64))
    with pytest.raises(ValueError, match="out of range"):
        indices_by_field_rust(np.array([0, 5], dtype=np.int64), np.array([1, 1], dtype=np.int64))
    with pytest.raises(ValueError, match="out of range"):
        indices_by_field_rust(np.array([-1], dtype=np.int64), np.array([1], dtype=np.int64))
    # A group that never appears with a non-zero count is an overstated count too
    with pytest.raises(ValueError, match="appears 0 times but its count is 1"):
        indices_by_field_rust(np.array([0], dtype=np.int64), np.array([1, 1], dtype=np.int64))

    empty = indices_by_field_rust(np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    assert dict(empty) == {}
