#!/usr/bin/env python3

"""
Performance test the OxVox indexing module compared to native python and numpy_indexed
"""

import numpy as np
import numpy.lib.recfunctions as rfn
import numpy_indexed as npi
from time import time

from oxvox.indexing import indices_by_field


BIGASS_ARRAY = rfn.unstructured_to_structured(
    # np.random.randint(0, 100, (99999, 3)),
    np.random.randint(0, 100, (1_000_000, 3)),
    names=["a", "b", "c"],
)


def test_indices_by_field_native():
    start = time()
    for unique_values in np.unique(BIGASS_ARRAY[["a", "b"]]):
        values = BIGASS_ARRAY[
            (BIGASS_ARRAY["a"] == unique_values[0])
            & (BIGASS_ARRAY["b"] == unique_values[1])
        ]
    print(f"Native indices_by_field took {time() - start}s")


def test_indices_by_field_npi():
    start = time()
    for chunk in npi.group_by(BIGASS_ARRAY[["a", "b"]]).split(BIGASS_ARRAY):
        pass
    print(f"Numpy-indexed indices_by_field took {time() - start}s")


def test_indices_by_field_oxvox():
    start = time()
    for unique_values, indices in indices_by_field(BIGASS_ARRAY, ["a", "b"]):
        pass
    print(f"OxVox indices_by_field took {time() - start}s")


if __name__ == "__main__":
    test_indices_by_field_native()
    test_indices_by_field_npi()
    test_indices_by_field_oxvox()
