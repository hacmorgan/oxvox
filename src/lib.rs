use std::collections::HashMap;

use bincode::{deserialize, serialize};
use ndarray::Array1;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyBytes, PyBytesMethods, PyDict, PyDictMethods, PyModule, PyModuleMethods};
use pyo3::{Bound, PyResult, Python, pyclass, pyfunction, pymethods, pymodule, wrap_pyfunction};
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

/// Build a rayon thread pool for a single query call
///
/// Args:
///     num_threads: Number of threads to use. `0` selects rayon's default (the number
///         of logical CPUs), matching the Python-facing `num_threads=0` convention
///
/// Returns:
///     A thread pool scoped to this call; every call gets its own pool, so
///     `num_threads` is honoured independently each time, rather than only on the
///     first call (as it was when we built rayon's global pool once per process)
fn _build_thread_pool(num_threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("failed to build rayon thread pool")
}

/// Rust engine for computing row indices for each unique value in a field or fields in a pointcloud
///
/// Runs in O(n+u) time: one pass over `counts` to size each unique id's output array,
/// then a single sequential pass over `unique_ids`, writing each row index into its
/// id's array via a per-id write cursor
///
/// Args:
///     unique_ids: Array of unique IDs for each point in the pointcloud (same length as the pointcloud)
///     counts: Array of counts for each unique ID
///
/// Returns:
///     Dict mapping each unique ID to the row indices in the pointcloud with that ID
#[pyfunction]
pub fn indices_by_field<'py>(
    py: Python<'py>,
    unique_ids: PyReadonlyArray1<'py, i64>,
    counts: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyDict>> {
    let unique_ids = unique_ids.as_array();
    let counts = counts.as_array();

    // Allocate one output array per unique id, sized exactly by its count
    let mut indices_by_id: Vec<Array1<u64>> = counts
        .iter()
        .map(|&count| Array1::<u64>::zeros(count as usize))
        .collect();

    // Track how far each id's output array has been filled, so the whole sweep over
    // unique_ids below is a single O(n) pass rather than one pass per unique id
    let mut write_cursors: Vec<usize> = vec![0; counts.len()];

    for (row, &id) in unique_ids.iter().enumerate() {
        let id = id as usize;
        let cursor = &mut write_cursors[id];
        indices_by_id[id][*cursor] = row as u64;
        *cursor += 1;
    }

    let dict = PyDict::new(py);
    for (id, indices) in indices_by_id.into_iter().enumerate() {
        dict.set_item(id, indices.into_pyarray(py))?;
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
    ///     num_threads: Number of parallel threads to use for this call. `0` uses
    ///         rayon's default (all available CPUs)
    ///     epsilon: Neighbours within this distance are accepted without further sorting
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
    ) -> PyResult<(Bound<'py, PyArray2<i32>>, Bound<'py, PyArray2<f32>>)> {
        // Convert query points to rust ndarray
        let query_points = query_points.as_array();

        // Release the GIL for the duration of the parallel search, so other Python
        // threads aren't blocked while we crunch through the query, and scope this
        // call's rayon thread pool to just this call, so num_threads is honoured on
        // every call, not just the first
        let (indices, distances) = py.detach(|| {
            let pool = _build_thread_pool(num_threads);
            pool.install(|| {
                nns::find_neighbours(
                    query_points,
                    &self.search_points,
                    &self.points_by_voxel,
                    &self.voxel_offsets,
                    num_neighbours,
                    self.max_dist,
                    epsilon,
                )
            })
        });

        Ok((indices.into_pyarray(py), distances.into_pyarray(py)))
    }

    /// Find how many neighbours exist within the search radius for each query point,
    /// optionally weighting each neighbour's contribution by its distance from the query point
    ///
    /// Args:
    ///     query_points: Points to search for neighbours of (Q, 3)
    ///     num_threads: Number of parallel threads to use for this call. `0` uses
    ///         rayon's default (all available CPUs)
    ///     distance_weight_factor: If `None`, count neighbours exactly. If `Some(p)` (must
    ///         be non-negative), each neighbour within the search radius instead
    ///         contributes `(1 - distance / search_radius).powf(p)`, so contributions run
    ///         from 1 at zero distance down to 0 at the search radius
    ///
    /// Returns:
    ///     Number of neighbours (or distance-weighted sum) within radius for each query
    ///     point (Q,)
    pub fn count_neighbours<'py>(
        &self,
        py: Python<'py>,
        query_points: PyReadonlyArray2<'py, f32>,
        num_threads: usize,
        distance_weight_factor: Option<f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        // A negative weighting factor doesn't correspond to a sensible kernel, so reject
        // it here, before it ever reaches the engine
        if let Some(p) = distance_weight_factor {
            if p < 0.0 {
                return Err(PyValueError::new_err(
                    "distance_weight_factor must be non-negative",
                ));
            }
        }

        // Convert query points to rust ndarray
        let query_points = query_points.as_array();

        // Release the GIL for the duration of the parallel search, and scope this
        // call's rayon thread pool to just this call (see find_neighbours above)
        let counts = py.detach(|| {
            let pool = _build_thread_pool(num_threads);
            pool.install(|| {
                nns::count_neighbours(
                    query_points,
                    &self.search_points,
                    &self.points_by_voxel,
                    &self.voxel_offsets,
                    self.max_dist,
                    distance_weight_factor,
                )
            })
        });

        Ok(counts.into_pyarray(py))
    }

    /// Implement deserialisation (unpickling) for OxVoxNNS objects
    pub fn __setstate__(&mut self, state: &Bound<'_, PyBytes>) -> PyResult<()> {
        *self = deserialize(state.as_bytes()).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(())
    }

    /// Implement serialisation (pickling) for OxVoxNNS objects
    pub fn __getstate__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(
            py,
            &serialize(&self).map_err(|e| PyValueError::new_err(e.to_string()))?,
        ))
    }

    /// How to construct a new OxVoxNNS object from an existing one (needed for pickling)
    pub fn __getnewargs__<'py>(&self, py: Python<'py>) -> (Bound<'py, PyArray2<f32>>, f32) {
        (self.search_points.clone().into_pyarray(py), self.max_dist)
    }
}

#[pymodule]
#[pyo3(name = "_oxvox")]
fn oxvox(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // All our python interface is in the OxVoxEngine class
    m.add_class::<OxVoxNNSEngine>()?;
    m.add_function(wrap_pyfunction!(indices_by_field, m)?)?;

    // Return a successful PyResult if the module compiled successfully
    Ok(())
}
