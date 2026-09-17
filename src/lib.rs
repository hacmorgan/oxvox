use bincode::{deserialize, serialize};
use ndarray::{Array1, Array2, ArrayView2};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyBytes, PyBytesMethods, PyDict, PyDictMethods, PyModule, PyModuleMethods};
use pyo3::{Bound, PyResult, Python, pyclass, pyfunction, pymethods, pymodule, wrap_pyfunction};
use serde::{Deserialize, Serialize};

mod index;

use index::graph::KnnGraph;
use index::hybrid::HybridGrid;
use index::kdtree::KdTree;
use index::voxel::VoxelGrid;

#[cfg(feature = "kiddo-baseline")]
use index::kiddo::KiddoTree;

/// Indices and distances of a query's found neighbours, as returned to Python
type NeighbourArrays<'py> = (Bound<'py, PyArray2<i32>>, Bound<'py, PyArray2<f32>>);

/// Names of the available search methods, as accepted by `OxVoxNNSEngine::new`
const METHOD_VOXEL: &str = "voxel";
const METHOD_KDTREE: &str = "kdtree";
const METHOD_HYBRID: &str = "hybrid";
const METHOD_GRAPH: &str = "graph";
#[cfg(feature = "kiddo-baseline")]
const METHOD_KIDDO: &str = "kiddo";

/// The spatial index behind an engine, one variant per search method
#[derive(Serialize, Deserialize)]
enum Backend {
    Voxel(VoxelGrid),
    KdTree(KdTree),
    Hybrid(HybridGrid),
    Graph(KnnGraph),
    #[cfg(feature = "kiddo-baseline")]
    Kiddo(KiddoTree),
}

impl Backend {
    /// Build the index named by `method`
    fn build(
        method: &str,
        search_points: ArrayView2<f32>,
        max_dist: f32,
        cells_per_radius: u32,
        subtree_threshold: usize,
        graph_degree: usize,
    ) -> PyResult<Self> {
        match method {
            METHOD_VOXEL => Ok(Backend::Voxel(VoxelGrid::new(
                search_points,
                max_dist,
                cells_per_radius,
            ))),
            METHOD_KDTREE => Ok(Backend::KdTree(KdTree::new(search_points))),
            METHOD_HYBRID => Ok(Backend::Hybrid(HybridGrid::new(
                search_points,
                max_dist,
                cells_per_radius,
                subtree_threshold,
            ))),
            METHOD_GRAPH => Ok(Backend::Graph(KnnGraph::new(
                search_points,
                max_dist,
                cells_per_radius,
                subtree_threshold,
                graph_degree,
            ))),
            #[cfg(feature = "kiddo-baseline")]
            METHOD_KIDDO => Ok(Backend::Kiddo(KiddoTree::new(search_points))),
            other => Err(PyValueError::new_err(format!(
                "unknown method {other:?}; expected one of {:?}",
                Self::available_methods()
            ))),
        }
    }

    /// Methods compiled into this build
    fn available_methods() -> Vec<&'static str> {
        vec![
            METHOD_VOXEL,
            METHOD_KDTREE,
            METHOD_HYBRID,
            METHOD_GRAPH,
            #[cfg(feature = "kiddo-baseline")]
            METHOD_KIDDO,
        ]
    }

    /// The voxel grid underneath grid-based methods, for statistics
    fn grid(&self) -> Option<&VoxelGrid> {
        match self {
            Backend::Voxel(grid) => Some(grid),
            Backend::Hybrid(hybrid) => Some(hybrid.grid()),
            Backend::Graph(graph) => Some(graph.grid()),
            _ => None,
        }
    }

    fn len(&self) -> usize {
        match self {
            Backend::Voxel(grid) => index::NeighbourIndex::len(grid),
            Backend::KdTree(tree) => index::NeighbourIndex::len(tree),
            Backend::Hybrid(hybrid) => index::NeighbourIndex::len(hybrid),
            Backend::Graph(graph) => index::NeighbourIndex::len(graph),
            #[cfg(feature = "kiddo-baseline")]
            Backend::Kiddo(tree) => index::NeighbourIndex::len(tree),
        }
    }

    fn find_neighbours(
        &self,
        query_points: ArrayView2<f32>,
        num_neighbours: usize,
        max_dist: f32,
        epsilon: f32,
        progress: bool,
    ) -> (Array2<i32>, Array2<f32>) {
        match self {
            Backend::Voxel(grid) => index::find_neighbours(
                grid,
                Some(grid.original_indices()),
                query_points,
                num_neighbours,
                max_dist,
                epsilon,
                progress,
            ),
            Backend::KdTree(tree) => index::find_neighbours(
                tree,
                Some(tree.original_indices()),
                query_points,
                num_neighbours,
                max_dist,
                epsilon,
                progress,
            ),
            Backend::Hybrid(hybrid) => index::find_neighbours(
                hybrid,
                Some(hybrid.grid().original_indices()),
                query_points,
                num_neighbours,
                max_dist,
                epsilon,
                progress,
            ),
            Backend::Graph(graph) => index::find_neighbours(
                graph,
                Some(graph.grid().original_indices()),
                query_points,
                num_neighbours,
                max_dist,
                epsilon,
                progress,
            ),
            #[cfg(feature = "kiddo-baseline")]
            Backend::Kiddo(tree) => index::find_neighbours(
                tree,
                None,
                query_points,
                num_neighbours,
                max_dist,
                epsilon,
                progress,
            ),
        }
    }

    fn count_neighbours(
        &self,
        query_points: ArrayView2<f32>,
        max_dist: f32,
        weight: Option<f32>,
        progress: bool,
    ) -> Array1<f32> {
        match self {
            Backend::Voxel(grid) => {
                index::count_neighbours(grid, query_points, max_dist, weight, progress)
            }
            Backend::KdTree(tree) => {
                index::count_neighbours(tree, query_points, max_dist, weight, progress)
            }
            Backend::Hybrid(hybrid) => {
                index::count_neighbours(hybrid, query_points, max_dist, weight, progress)
            }
            Backend::Graph(graph) => {
                index::count_neighbours(graph, query_points, max_dist, weight, progress)
            }
            #[cfg(feature = "kiddo-baseline")]
            Backend::Kiddo(tree) => {
                index::count_neighbours(tree, query_points, max_dist, weight, progress)
            }
        }
    }
}

/// Rust engine behind `oxvox.nns.OxVoxNNS`
#[derive(Serialize, Deserialize)]
#[pyclass(module = "oxvox")] // module = "oxvox" is required for pickling to find the class
struct OxVoxNNSEngine {
    backend: Backend,
    max_dist: f32,
    method: String,
    cells_per_radius: u32,
    subtree_threshold: usize,
    graph_degree: usize,
}

/// Build a rayon thread pool for a single query call
///
/// Args:
///     num_threads: Number of threads to use. `0` selects rayon's default (the number
///         of logical CPUs), matching the Python-facing `num_threads=0` convention
///
/// Returns:
///     A thread pool scoped to this call, so `num_threads` is honoured independently
///     each time rather than only on the first call
fn _build_thread_pool(num_threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("failed to build rayon thread pool")
}

/// Reject arrays that aren't (N, 3)
fn _check_three_columns(name: &str, shape: &[usize]) -> PyResult<()> {
    if shape.len() != 2 || shape[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape (N, 3), got {shape:?}"
        )));
    }
    Ok(())
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
        // Reject ids that don't index `counts`, and ids that appear more often than
        // their count claims, with a Python exception rather than a panic
        let id = usize::try_from(id)
            .ok()
            .filter(|&id| id < counts.len())
            .ok_or_else(|| PyValueError::new_err(format!("unique id {id} out of range")))?;
        let cursor = &mut write_cursors[id];
        if *cursor >= indices_by_id[id].len() {
            return Err(PyValueError::new_err(format!(
                "unique id {id} appears more times than its count"
            )));
        }
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
    /// Construct the engine, building the spatial index over the search points
    ///
    /// Args:
    ///     search_points: Points to search for neighbours amongst (N, 3)
    ///     max_dist: Search radius; neighbours at or beyond it are ignored
    ///     method: Search method: "voxel" (uniform grid), "kdtree", "hybrid" (grid with
    ///         KD subtrees in dense cells) or "graph" (approximate, kNN graph flood)
    ///     cells_per_radius: For grid-based methods, how many grid cells span one radius
    ///     subtree_threshold: For "hybrid"/"graph", cells with more points than this get
    ///         a KD subtree
    ///     graph_degree: For "graph", how many neighbours each point links to
    #[new]
    #[pyo3(signature = (search_points, max_dist, method = "voxel", cells_per_radius = 1, subtree_threshold = 64, graph_degree = 16))]
    fn new(
        py: Python<'_>,
        search_points: PyReadonlyArray2<f32>,
        max_dist: f32,
        method: &str,
        cells_per_radius: u32,
        subtree_threshold: usize,
        graph_degree: usize,
    ) -> PyResult<Self> {
        if max_dist <= 0.0 || !max_dist.is_finite() {
            return Err(PyValueError::new_err(format!(
                "max_dist must be a positive finite number, got {max_dist}"
            )));
        }
        _check_three_columns("search_points", search_points.shape())?;
        let search_points = search_points.as_array();

        // Building the index is a heavy, purely-Rust operation, so let other Python
        // threads run in the meantime
        let backend = py.detach(|| {
            Backend::build(
                method,
                search_points,
                max_dist,
                cells_per_radius,
                subtree_threshold,
                graph_degree,
            )
        })?;

        Ok(OxVoxNNSEngine {
            backend,
            max_dist,
            method: method.to_owned(),
            cells_per_radius,
            subtree_threshold,
            graph_degree,
        })
    }

    /// Search method this engine was built with
    #[getter]
    fn method(&self) -> &str {
        &self.method
    }

    /// Number of indexed search points
    fn __len__(&self) -> usize {
        self.backend.len()
    }

    /// Statistics about the voxel grid's occupancy, for choosing between methods
    ///
    /// Returns:
    ///     Dict with `num_cells`, `max_points_per_cell` and `mean_points_per_cell`
    ///     (plus `num_subtrees` for the hybrid method), or None for methods that
    ///     don't use a grid
    fn grid_stats<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(grid) = self.backend.grid() else {
            return Ok(None);
        };
        let stats = PyDict::new(py);
        stats.set_item("num_cells", grid.num_cells())?;
        stats.set_item("max_points_per_cell", grid.max_points_per_cell())?;
        stats.set_item("mean_points_per_cell", grid.mean_points_per_cell())?;
        if let Backend::Hybrid(hybrid) = &self.backend {
            stats.set_item("num_subtrees", hybrid.num_subtrees())?;
        }
        if let Backend::Graph(graph) = &self.backend {
            stats.set_item("graph_degree", graph.degree())?;
        }
        Ok(Some(stats))
    }

    /// Find neighbours of query points within search points
    ///
    /// Args:
    ///     query_points: Points to search for neighbours of (Q, 3)
    ///     num_neighbours: Maximum number of neighbours to search for
    ///     num_threads: Number of parallel threads to use for this call. `0` uses
    ///         rayon's default (all available CPUs)
    ///     epsilon: Once `num_neighbours` neighbours closer than this have been found,
    ///         the search for that query point stops early
    ///     progress: Show a progress bar over query points
    ///
    /// Returns:
    ///     Indices of neighbouring points for each query point (Q, num_neighbours), -1 padded
    ///     Distance from query point to each neighbour (Q, num_neighbours), -1 padded
    #[pyo3(signature = (query_points, num_neighbours, num_threads = 0, epsilon = 0.0, progress = false))]
    pub fn find_neighbours<'py>(
        &self,
        py: Python<'py>,
        query_points: PyReadonlyArray2<'py, f32>,
        num_neighbours: usize,
        num_threads: usize,
        epsilon: f32,
        progress: bool,
    ) -> PyResult<NeighbourArrays<'py>> {
        _check_three_columns("query_points", query_points.shape())?;
        let query_points = query_points.as_array();

        // Release the GIL for the duration of the parallel search, and scope this
        // call's rayon thread pool to just this call so num_threads is honoured
        let (indices, distances) = py.detach(|| {
            let pool = _build_thread_pool(num_threads);
            pool.install(|| {
                self.backend.find_neighbours(
                    query_points,
                    num_neighbours,
                    self.max_dist,
                    epsilon,
                    progress,
                )
            })
        });

        Ok((indices.into_pyarray(py), distances.into_pyarray(py)))
    }

    /// Count how many neighbours exist within the search radius of each query point,
    /// optionally weighting each neighbour's contribution by its distance
    ///
    /// Args:
    ///     query_points: Points to search for neighbours of (Q, 3)
    ///     num_threads: Number of parallel threads to use for this call. `0` uses
    ///         rayon's default (all available CPUs)
    ///     distance_weight_factor: If `None`, count neighbours exactly. If `Some(p)` (must
    ///         be non-negative), each neighbour within the search radius instead
    ///         contributes `(1 - distance / search_radius).powf(p)`, so contributions run
    ///         from 1 at zero distance down to 0 at the search radius
    ///     progress: Show a progress bar over query points
    ///
    /// Returns:
    ///     Number of neighbours (or distance-weighted sum) within radius for each query
    ///     point (Q,)
    #[pyo3(signature = (query_points, num_threads = 0, distance_weight_factor = None, progress = false))]
    pub fn count_neighbours<'py>(
        &self,
        py: Python<'py>,
        query_points: PyReadonlyArray2<'py, f32>,
        num_threads: usize,
        distance_weight_factor: Option<f32>,
        progress: bool,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        if let Some(p) = distance_weight_factor
            && p < 0.0
        {
            return Err(PyValueError::new_err(
                "distance_weight_factor must be non-negative",
            ));
        }
        _check_three_columns("query_points", query_points.shape())?;
        let query_points = query_points.as_array();

        let counts = py.detach(|| {
            let pool = _build_thread_pool(num_threads);
            pool.install(|| {
                self.backend.count_neighbours(
                    query_points,
                    self.max_dist,
                    distance_weight_factor,
                    progress,
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

    /// Arguments pickle passes to `__new__` before `__setstate__` restores the real
    /// state. An empty pointcloud keeps that placeholder construction trivial
    pub fn __getnewargs__<'py>(
        &self,
        py: Python<'py>,
    ) -> (Bound<'py, PyArray2<f32>>, f32, String, u32, usize, usize) {
        (
            Array2::<f32>::zeros((0, 3)).into_pyarray(py),
            self.max_dist,
            self.method.clone(),
            self.cells_per_radius,
            self.subtree_threshold,
            self.graph_degree,
        )
    }
}

/// Names of the search methods compiled into this build
#[pyfunction]
fn available_methods() -> Vec<&'static str> {
    Backend::available_methods()
}

#[pymodule]
#[pyo3(name = "_oxvox")]
fn oxvox(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OxVoxNNSEngine>()?;
    m.add_function(wrap_pyfunction!(indices_by_field, m)?)?;
    m.add_function(wrap_pyfunction!(available_methods, m)?)?;
    Ok(())
}
