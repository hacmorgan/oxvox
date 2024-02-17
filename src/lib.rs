use std::collections::HashMap;

use bincode::{deserialize, serialize};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::types::{PyBytes, PyModule};
use pyo3::{pyclass, pymethods, pymodule, PyResult, Python};
use serde::{Deserialize, Serialize};

mod nns;

#[derive(Serialize, Deserialize)]
#[pyclass(module = "ox_vox_nns")] // required for python to serialise correctly
struct OxVoxNNS {
    search_points: Array2<f32>, // (N, 3)
    sorted_indices: nns::SortedIndices,
}

#[pymethods]
impl OxVoxNNS {
    /// Construct OxVoxNNS object
    ///
    /// Args:
    ///     search_points: Points to search for neighbours amongst
    ///     max_dist: Maximum distance to neighbouring point for it to be considered
    #[new]
    fn new(search_points: PyReadonlyArray2<f32>) -> Self {
        // Convert search points to rust ndarray
        // TODO: check if we can get rid of the .to_owned without breaking pickleabaility (test in python)
        let search_points = search_points.as_array().to_owned();

        // Perform initial passes (one-time stuff)
        let sorted_indices = nns::initialise_nns(search_points.view());

        // Construct the NNS object with computed values required for querying
        OxVoxNNS {
            search_points,
            sorted_indices,
        }
    }

    /// Find neighbours of query points within search points
    ///
    /// Args:
    ///     query_points: Points to search for neighbours of (Q, 3)
    ///     num_neighbours: Maximum number of neighbours to search for
    ///
    /// Returns:
    ///     Indices of neighbouring points for each query point (Q, num_neighbours)
    ///     Distance from query point to search point for each search point in indices (Q, num_neighbours)
    pub fn find_neighbours<'py>(
        &self,
        py: Python<'py>,
        query_points: PyReadonlyArray2<'py, f32>,
        num_neighbours: i32,
        max_dist: f32,
    ) -> (&'py PyArray2<i32>, &'py PyArray2<f32>) {
        // Convert query points to rust ndarray
        let query_points = query_points.as_array();

        // Run find_neighbours function
        let (indices, distances) = nns::find_neighbours(
            query_points,
            &self.search_points,
            num_neighbours,
            max_dist,
            &self.sorted_indices,
        );

        (indices.into_pyarray(py), distances.into_pyarray(py))
    }

    /// Implement deserialisation (unpickling) for OxVoxNNS objects
    pub fn __setstate__(&mut self, state: &PyBytes) -> PyResult<()> {
        *self = deserialize(state.as_bytes()).unwrap();
        Ok(())
    }

    /// Implement serialisation (pickling) for OxVoxNNS objects
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<&'py PyBytes> {
        Ok(PyBytes::new(py, &serialize(&self).unwrap()))
    }

    /// How to construct a new OxVoxNNS object from an existing one (needed for pickling)
    pub fn __getnewargs__<'py>(&self, py: Python<'py>) -> &'py PyArray2<f32> {
        &self.search_points.clone().into_pyarray(py)
    }
}

#[pymodule]
fn ox_vox_nns<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    // All our python interface is in the OxVoxNNS class
    m.add_class::<OxVoxNNS>()?;

    // Return a successful PyResult if the module compiled successfully
    Ok(())
}
