"""
oxvox: performant operations on array and pointcloud data, written in Rust
"""

from oxvox._oxvox import OxVoxNNSEngine

# The Rust OxVoxNNSEngine pyclass declares module = "oxvox" (required for it to
# pickle/unpickle correctly), so it must be importable from this top-level package,
# not just from the private oxvox._oxvox extension module
__all__ = ["OxVoxNNSEngine"]
