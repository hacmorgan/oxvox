"""
Array indexing operations
"""

from typing import Tuple
from functools import partial

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import numpy.typing as npt

from oxvox._oxvox import OxVoxIndexing


def indices_by_field(arr: npt.NDArray[Any], fields: str | tuple[str, ...]) -> npt.NDArray[np.int32]:
    """
    Compute row indices for each value in a given field in a structured array

    This can be useful for modifying an array in-place. The following snippet also does
    this, but requires a full pass through the array for each unique value, making it
    O(n*u) where n is the number of rows and u is the number of unique values

        for value in np.unique(arr[fields]):
            arr[arr[fields] == value] = some_updated_value

    Similarly, we can use numpy_indexed to do this in O(n+u), so long as we accept the
    output array being sorted by the fields we are splitting by

        chunks = []
        for chunk in npi.group_by(arr[fields]).split(arr):
            chunk["some_field"] = some_updated_value
            chunks.append(chunk)
        return np.concatenate(chunks)

    Args:
        arr: Structured array to compute row indices for
        fields: Field(s) to split by

    Returns:
        Row indices for each value in the given field(s)
    """
    # We first use numpy_indexed (because it handles structured arrays nicely) to get:
    # - An array the same length as the input array, with a unique ID for each unique
    #       set of values in the given field(s)
    # - A set of unique values in the given field(s)
    # - A count of how many times each unique value appears in the given field(s)
    grouper = npi.group_by(arr[fields])
    unique_ids = grouper.inverse
    unique_values = grouper.unique
    counts = grouper.count

    # We quickly create a lookup from unique array values to their unique IDs
    value_to_index = dict(enumerate(unique_values))

    # Now we use the rust engine to compute the row indices for each unique ID
    indices_by_field = indices_by_field(unique_ids, counts)

    # Remap the indices to the original array values
    return {value: indices_by_field[value_to_index[value]] for value in unique_values}
