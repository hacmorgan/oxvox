"""
Utility functions
"""


from functools import lru_cache

import numpy as np


@lru_cache
def _default_dtype() -> np.dtype:
    """
    Return a sensible default integer type to use for row indices for this platform
    """
    return np.uint32 if np.intp == np.int32 else np.uint64