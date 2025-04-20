use std::collections::HashMap;
use rayon::prelude::*;

use bincode::{deserialize, serialize};
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyReadonlyArray1};
use pyo3::types::{PyBytes, PyModule, PyDict};
use pyo3::{pyclass, pymethods, pymodule, pyfunction, PyResult, Python, wrap_pyfunction, IntoPy};
use ndarray::{Array1, ShapeBuilder};
use serde::{Deserialize, Serialize};

mod nns;

#[derive(Serialize, Deserialize)]
#[pyclass(module = "oxvox")] // module = "blah" required for python to serialise correctly
struct OxVoxNNSEngine {
    search_points: Array2<f32>,                          // (N, 3)
    points_by_voxel: HashMap<(i32, i32, i32), Vec<i32>>, // maps voxel_coords -> indices of search points in that voxel
    voxel_offsets: Array2<i32>,                          // (27, 3)
    max_dist: f32,
}

/*
TODOs
- how to convert hashmap to python dict?
- how to correctly return a dictionary of arrays?
*/

/// Rust engine for computing row indices for each unique value in a field or fields in a pointcloud
///
/// Args:
///     unique_ids: Array of unique IDs for each point in the pointcloud (same length as the pointcloud)
///     counts: Array of counts for each unique ID
///
/// Returns:
#[pyfunction]
pub fn indices_by_field<'py>(
    py: Python<'py>,
    unique_ids: PyReadonlyArray1<'py, i64>,
    counts: PyReadonlyArray1<'py, i64>,
) -> PyResult<&'py PyDict> {
    let unique_ids = unique_ids.as_array().to_owned();
    let counts = counts.as_array().to_owned();

    // Construct a hashmap mapping each unique ID to a list of indices
    let mut indices_by_id = HashMap::new();
    counts.iter().enumerate().for_each(|(id_, count)| {
        indices_by_id.insert(id_, Array1::<u64>::zeros(*count as usize));
    });

    // Populate the hashmap in parallel
    indices_by_id.par_iter_mut().for_each(|(id_, indices_arr)| {

        // Our arrays have already been allocated with the correct size, so we simply
        // track how far we are through the array
        let mut hashmap_inner_idx = 0;

        // Iterate over unique IDs and fill in the array with indices
        for idx in 0..unique_ids.len() {

            // If we find a match, add the index to the array
            if unique_ids[idx] == *id_ as i64 {
                indices_arr[hashmap_inner_idx] = idx as u64;
                hashmap_inner_idx += 1;
            }

            // If we've filled the array, break
            if hashmap_inner_idx == unique_ids.len() - 1 {
                break;
            }
        }
    });

    let dict = PyDict::new(py);
    for (k, v) in indices_by_id.iter() {
        dict.set_item(k.into_py(py), v.clone().into_pyarray(py))?;
    }

    Ok(dict)

}

#[pymethods]
impl OxVoxNNSEngine {
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
        OxVoxNNSEngine {
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
        num_threads: usize,
        epsilon: f32,
    ) -> PyResult<(&'py PyArray2<i32>, &'py PyArray2<f32>)> {
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
            )
        };

        Ok((indices.into_pyarray(py), distances.into_pyarray(py)))
    }

    /// Find how many neighbours exist within the search radius for each query point
    ///
    /// Args:
    ///     query_points: Points to search for neighbours of (Q, 3)
    ///     num_threads: Numper of parallel threads to use
    ///
    /// Returns:
    ///     Number of neighbours within radius for each query point (Q,)
    pub fn count_neighbours<'py>(
        &self,
        py: Python<'py>,
        query_points: PyReadonlyArray2<'py, f32>,
        num_threads: usize,
    ) -> &'py PyArray1<u32> {
        // Convert query points to rust ndarray
        let query_points = query_points.as_array();

        // Run find_neighbours function
        let counts = if num_threads != 1 {
            // Set number of threads in global thread pool
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build_global()
                .unwrap_or(());

            // Query for neighbours
            nns::count_neighbours(
                query_points,
                &self.search_points,
                &self.points_by_voxel,
                &self.voxel_offsets,
                self.max_dist,
            )
        } else {
            nns::count_neighbours_singlethread(
                query_points,
                &self.search_points,
                &self.points_by_voxel,
                &self.voxel_offsets,
                self.max_dist,
            )
        };

        counts.into_pyarray(py)
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
#[pyo3(name = "_oxvox")]
fn oxvox<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {

    // All our python interface is in the OxVoxEngine class
    m.add_class::<OxVoxNNSEngine>()?;
    m.add_function(wrap_pyfunction!(indices_by_field, m)?)?;

    // Return a successful PyResult if the module compiled successfully
    Ok(())
}
