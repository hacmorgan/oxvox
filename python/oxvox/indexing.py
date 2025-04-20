"""
Array indexing operations
"""

from typing import Any, Iterator

import numpy as np
import numpy.typing as npt
import numpy_indexed as npi

from oxvox._oxvox import indices_by_field as indices_by_field_rust


def indices_by_field(
    arr: npt.NDArray[Any], fields: str | tuple[str, ...]
) -> Iterator[tuple[Any, npt.NDArray[np.int32]]]:
    """
    Compute row indices for each value in a given field in a structured array

    This can be useful for modifying an array in-place. If this is not a requirement,
    consider using the numpy_indexed library directly (used internally by this function)

    Args:
        arr: Structured array to compute row indices for
        fields: Field(s) to split by

    Yields:
        Value(s) in the given field(s) in the input array
        Row indices with those values

    Examples:

        The following snippet modifies in-place, but requires a full pass through the
        array for each unique value, making it O(n*u) where n is the number of rows and
        u is the number of unique values:

            for value in np.unique(arr[fields]):
                arr[arr[fields] == value] = some_updated_value

        Similarly, we can use numpy_indexed to do this in O(n+u), so long as we accept
        the output array being sorted by the fields we are splitting by:

            chunks = []
            for chunk in npi.group_by(arr[fields]).split(arr):
                chunk["some_field"] = some_updated_value
                chunks.append(chunk)
            return np.concatenate(chunks)

        With oxvox we can do this in O(n+u) time, in parallel, and without sorting the
        output array:

            for value, indices in indices_by_field(arr, fields):
                arr[indices] = some_updated_value

        n.b. If you don't need the values in the array, and are happy with unique
        sequential IDs as dictionary keys, you can use the underlying
        `indices_by_field_rust` directly:

            for offset, indices in enumerate(
                indices_by_field_rust(unique_ids, counts).values()
            ):
                arr[indices] += offset
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

    # We quickly create a lookup from unique IDs to their unique array values
    index_to_value = dict(enumerate(unique_values))

    # Now we use the rust engine to compute the row indices for each unique ID
    for unique_id, indices in indices_by_field_rust(unique_ids, counts).items():
        yield index_to_value[unique_id], indices
