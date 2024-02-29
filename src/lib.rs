use std::collections::HashMap;

use bincode::{deserialize, serialize};
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::types::{PyBytes, PyModule};
use pyo3::{pyclass, pymethods, pymodule, PyResult, Python};
use serde::{Deserialize, Serialize};

mod nns;

#[derive(Serialize, Deserialize)]
#[pyclass(module = "ox_vox_nns")] // module = "blah" required for python to serialise correctly
struct OxVoxEngine {
    search_points: Array2<f32>,                          // (N, 3)
    points_by_voxel: HashMap<(i32, i32, i32), Vec<i32>>, // maps voxel_coords -> indices of search points in that voxel
    voxel_offsets: Array2<i32>,                          // (27, 3)
    max_dist: f32,
}

#[pymethods]
impl OxVoxEngine {
    /// Construct OxVoxNNS object
    ///
    /// Args:
    ///     search_points: Points to search for neighbours amongst
    ///     max_dist: Maximum distance to neighbouring point for it to be considered (i.e. search radius)
    #[new]
    fn new(search_points: PyReadonlyArray2<f32>, max_dist: f32) -> Self {
        // Convert search points to rust ndarray
        let search_points = search_points.as_array().to_owned();

        // Perform initial passes (one-time stuff)
        let (points_by_voxel, voxel_offsets) = nns::initialise_nns(&search_points, max_dist);

        // Construct the NNS object with computed values required for querying
        OxVoxEngine {
            search_points,
            points_by_voxel,
            voxel_offsets,
            max_dist,
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
        sparse: bool,
        l2_distance: bool,
        num_threads: usize,
        epsilon: f32,
    ) -> (&'py PyArray2<i32>, &'py PyArray2<f32>) {
        // Convert query points to rust ndarray
        let query_points = query_points.as_array();

        // Run find_neighbours function
        let (indices, distances) = if num_threads != 1 {

            // Set number of threads in global thread pool
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build_global()
                .unwrap_or(());

            // Query for neighbours
            nns::find_neighbours(
                query_points,
                &self.search_points,
                &self.points_by_voxel,
                &self.voxel_offsets,
                num_neighbours,
                self.max_dist,
                epsilon,
                sparse,
                l2_distance,
            )
        } else {
            nns::find_neighbours_singlethread(
                query_points,
                &self.search_points,
                &self.points_by_voxel,
                &self.voxel_offsets,
                num_neighbours,
                self.max_dist,
                epsilon,
                sparse,
                l2_distance,
            )
        };

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
    pub fn __getnewargs__<'py>(&self, py: Python<'py>) -> (&'py PyArray2<f32>, f32) {
        (&self.search_points.clone().into_pyarray(py), self.max_dist)
    }
}

#[pymodule]
#[pyo3(name="_ox_vox_nns")]
fn ox_vox_nns<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    // All our python interface is in the OxVoxNNS class
    m.add_class::<OxVoxEngine>()?;

    // Return a successful PyResult if the module compiled successfully
    Ok(())
}
