
use std::collections::HashMap;

use std::vec;

use numpy::ndarray::{Array2, ArrayView1, ArrayView2, ArrayViewMut1, Axis};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::{
    pyclass, pymethods, pymodule,
    types::{PyModule}, PyResult, Python,
};

use rayon::prelude::*;

const MIN_LAYERS_FOR_SAFE_POINTS: usize = 6;

#[pyclass]
struct OxVoxNNS {
    search_points: Array2<f32>,
    points_by_voxel: HashMap<(i32, i32, i32), Vec<i32>>,
    voxel_offsets: Vec<Vec<Vec<i32>>>,
    voxel_size: f32,
}

#[pymethods]
impl OxVoxNNS {
    #[new]
    fn new(search_points: PyReadonlyArray2<f32>, max_dist: f32, voxel_size: f32) -> Self {
        // Convert search points to rust ndarray
        let search_points = search_points.as_array().to_owned();

        // Voxelise pointcloud - compute voxel coords for each point and construct
        // lookup of point indices by voxel coords
        let (_voxel_indices, points_by_voxel) = voxelise_points(search_points.view(), voxel_size);

        // Generate voxel coordinate offsets for each layer of concentric cubes
        // radiating outwards from a given voxel, within search radius
        let num_levels = (max_dist / voxel_size) as i32 + 2;
        let voxel_offsets = construct_voxel_offsets(num_levels);

        // Construct the NNS object with computed values required for querying
        OxVoxNNS{search_points, points_by_voxel, voxel_offsets, voxel_size}
    }

    /// Python wrapper for find_neighbours
    pub fn find_neighbours<'py>(
        &self,
        py: Python<'py>,
        query_points: PyReadonlyArray2<'py, f32>,
        num_neighbours: i32,
    ) -> (&'py PyArray2<i32>, &'py PyArray2<f32>) {
        // Convert query points to rust ndarray
        let query_points = query_points.as_array();

        // Run find_neighbours function
        let (indices, distances) = find_neighbours(
            query_points,
            self.search_points.view(),
            &self.points_by_voxel,
            &self.voxel_offsets,
            num_neighbours,
            self.voxel_size,
        );

        (indices.into_pyarray(py), distances.into_pyarray(py))
    }

    // /// Python wrapper for voxelise_points
    // #[pyfn(m)]
    // #[pyo3(name = "voxelise_points")]
    // fn voxelise_points_py<'py>(
    //     py: Python<'py>,
    //     search_points: PyReadonlyArray2<'py, f32>,
    //     voxel_size: f32,
    // ) -> (&'py PyArray2<i32>, PyObject) {
    //     let search_points = search_points.as_array();
    //     let (voxel_indices, points_by_voxel) = voxelise_points(search_points, voxel_size);
    //     (
    //         voxel_indices.into_pyarray(py),
    //         points_by_voxel.into_py_dict(py).into(),
    //     )
    // }
}

#[pymodule]
fn ox_vox_nns<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    // All our python interface is in the OxVoxNNS class
    m.add_class::<OxVoxNNS>()?;

    // Return a successful PyResult if the module compiled successfully
    Ok(())
}

/// Find the (up to) N nearest neighbours within a given radius for each query point
///
/// Args:
///     search_points: Pointcloud we are searching for neighbours within (S, 3)
///     query_points: Points we are searching for the neighbours of (Q, 3)
///     num_neighbours: Maximum number of neighbours to search for
///     max_dist: Furthest distance to neighbouring points before we don't care about them
///
/// Returns:
///     Indices of neighbouring points (Q, num_neighbours)
///     Distances of neighbouring points from query point (Q, num_neighbours)
fn find_neighbours(
    query_points: ArrayView2<f32>,
    search_points: ArrayView2<f32>,
    points_by_voxel: &HashMap<(i32, i32, i32), Vec<i32>>,
    voxel_offsets: &Vec<Vec<Vec<i32>>>,
    num_neighbours: i32,
    voxel_size: f32,
) -> (Array2<i32>, Array2<f32>) {
    // Compute useful metadata
    let num_query_points = query_points.shape()[0];

    // Construct output arrays, initialised with -1s
    let mut indices: Array2<i32> = Array2::zeros([num_query_points, num_neighbours as usize]) - 1;
    let mut distances: Array2<f32> =
        Array2::zeros([num_query_points, num_neighbours as usize]) - 1f32;

    // Map query point processing function across corresponding rows of query
    // points, indices, and distances arrays
    query_points
        .axis_iter(Axis(0))
        .zip(indices.axis_iter_mut(Axis(0)))
        .zip(distances.axis_iter_mut(Axis(0)))
        .for_each(|((query_point, indices_row), distances_row)| {
            find_query_point_neighbours(
                query_point,
                indices_row,
                distances_row,
                &search_points,
                &points_by_voxel,
                &voxel_offsets,
                num_neighbours,
                voxel_size,
            );
        });

    (indices, distances)
}

/// Generate voxel coordinates for each point (i.e. find which voxel each point
/// belongs to), and construct a hashmap of search point indices, indexed by voxel
/// coordinates
///
/// This is the first pass through the points we will make
fn voxelise_points(
    search_points: ArrayView2<f32>,
    voxel_size: f32,
) -> (Array2<i32>, HashMap<(i32, i32, i32), Vec<i32>>) {
    // Construct mapping from voxel coords to point indices
    let mut points_by_voxel = HashMap::new();

    // Construct an array to store each point's voxel coords
    let num_points = search_points.shape()[0];
    let mut voxel_indices: Array2<i32> = Array2::zeros([num_points, 3]);

    // Compute voxel index for each point and add to hashmap
    for i in 0..num_points {
        // Compute voxel coords
        for j in 0..3 {
            voxel_indices[[i, j]] = voxel_coord(search_points[[i, j]], voxel_size);
        }

        // Insert point index into hashmap
        let key: (i32, i32, i32) = (
            voxel_indices[[i, 0]],
            voxel_indices[[i, 1]],
            voxel_indices[[i, 2]],
        );
        let point_indices: &mut Vec<i32> = points_by_voxel.entry(key).or_insert(Vec::new());
        point_indices.push(i as i32);
    }

    // Return the voxel indices array and the hashmap
    (voxel_indices, points_by_voxel)
}

/// Run nearest neighbour search for a query point
///
/// This function is intended to be mapped (maybe in parallel) across rows of an
/// array of query points, zipped with rows from two mutable arrays for distances
/// and indices, which we will write to
///
/// Args:
///     query_point: The query point we are searching for neighbours of
///     indices_row: Mutable view of the row of the indices array corresponding to
///         this query point. We will write the point indices of our neighbouring
///         points here
///     distances_row: Mutable view of the row of the distances array corresponding
///         to this query point. We will write the point indices of our neighbouring
///         points here
///     search_points: Reference to search points array, for indexing and comparing
///         distances
///     points_by_voxel:
///     voxel_offsets:  
///     num_neighbours:
///     voxel_size:
fn find_query_point_neighbours(
    query_point: ArrayView1<f32>,
    mut indices_row: ArrayViewMut1<i32>,
    mut distances_row: ArrayViewMut1<f32>,
    search_points: &ArrayView2<f32>,
    points_by_voxel: &HashMap<(i32, i32, i32), Vec<i32>>,
    voxel_offsets: &Vec<Vec<Vec<i32>>>,
    num_neighbours: i32,
    voxel_size: f32,
) {
    // We need this a few times
    let root_3: f32 = 3f32.sqrt();

    // Compute voxel coords of query point
    let query_voxel = (
        voxel_coord(query_point[0], voxel_size),
        voxel_coord(query_point[1], voxel_size),
        voxel_coord(query_point[2], voxel_size),
    );

    // We store neighbouring points' indices and distances together in tuples, such
    // that we can sort them by distance.
    let mut neighbours: Vec<(i32, f32)> = Vec::new();

    // We track number of points we find per layer so we know when we can exit
    let mut points_per_layer: Vec<i32> = Vec::new();

    // The saturation layer is the layer at which we found >= num_neighbours. If we
    // find this at layer n, we can make the following assumptions:
    // - There is no need to search any further than layer: ceil(root_3 * (n + 1))
    // - All the points below layer: floor((n - 2 - root_3)/root_3) are safe
    let mut saturation_layer: Option<i32> = None;

    // Start from query voxel, working outwards in layers
    for (layer_num, layer_voxel_offsets) in voxel_offsets.iter().enumerate() {
        // Track points for this layer
        let mut this_layer_points = 0;

        // Find points from each voxel in this layer
        for voxel_offset in layer_voxel_offsets.iter() {
            // Construct voxel coords tuple for this voxel
            let this_voxel = (
                query_voxel.0 + voxel_offset[0],
                query_voxel.1 + voxel_offset[1],
                query_voxel.2 + voxel_offset[2],
            );

            // Only proceed if this voxel actually contains any points
            let this_voxel_point_indices = match points_by_voxel.get(&this_voxel) {
                Some(o) => o,
                None => continue,
            };

            // Update num points in this layer
            this_layer_points += this_voxel_point_indices.len();

            // Compute distance from each point to the query point and add to list
            // of neighbours
            for point_idx in this_voxel_point_indices.iter() {
                // Extract terms to simplify later expression
                let pt_idx_u = *point_idx as usize;
                let spx = search_points[[pt_idx_u, 0]];
                let spy = search_points[[pt_idx_u, 1]];
                let spz = search_points[[pt_idx_u, 2]];
                let qpx = query_point[0];
                let qpy = query_point[1];
                let qpz = query_point[2];
                let dx = spx - qpx;
                let dy = spy - qpy;
                let dz = spz - qpz;

                neighbours.push((*point_idx, (dx * dx + dy * dy + dz * dz).sqrt()));
            }
        }

        // Pull out cumulative number of points from last layer and compute for this layer
        let num_points_before_this_layer = match points_per_layer.last() {
            Some(previous_num_points) => *previous_num_points,
            None => 0i32,
        };
        let cumulative_num_points = num_points_before_this_layer + this_layer_points as i32;

        // Save this layer's cumulative number of points
        points_per_layer.push(cumulative_num_points);

        // Apply early break test if we have reached the saturation layer
        if let Some(saturation_layer) = saturation_layer {
            // If we found enough points a few layers ago we might be able to stop here
            if layer_num as f32 > root_3 * (saturation_layer + 1) as f32 + 1f32 {
                break;
            }
        }
        // Otherwise check if we have reached the saturation layer
        else if cumulative_num_points >= num_neighbours {
            saturation_layer = Some(layer_num as i32);
        }
    }

    // Apply test to see which layers actually need to be compared
    let safe_layer = match saturation_layer {
        Some(saturation_layer) => ((saturation_layer as f32 - 2f32 - root_3) / root_3) as usize,
        None => (voxel_offsets.len() - 1) as usize,
    };

    // Find the points which are guaranteed to be safe and insert into output arrays
    let num_safe_points = if voxel_offsets.len() > MIN_LAYERS_FOR_SAFE_POINTS {
        points_per_layer[safe_layer] as usize
    } else {
        0
    };
    for (j, (point_idx, distance)) in neighbours.iter().take(num_safe_points).enumerate() {
        indices_row[j] = *point_idx;
        distances_row[j] = *distance;
    }

    // Sort the remaining points and take as many of the closest as required
    let num_neighbours_required = num_neighbours as usize - num_safe_points;
    let mut remaining_points: Vec<(i32, f32)> =
        neighbours.iter().skip(num_safe_points).cloned().collect();
    remaining_points.par_sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    for (j, (point_idx, distance)) in remaining_points
        .iter()
        .take(num_neighbours_required)
        .enumerate()
    {
        indices_row[num_safe_points + j] = *point_idx;
        distances_row[num_safe_points + j] = *distance;
    }
}

/// Construct voxel offset arrays to find voxel coordinates of concentric cubes
/// radiating from a given voxel
///
/// Args:
///     num_levels: How many layers of concentric cubes' voxel coordinates we need
///
/// Returns:
///     Voxel coordinate offsets, per level
fn construct_voxel_offsets(num_levels: i32) -> Vec<Vec<Vec<i32>>> {
    // Construct outer vec, will be indexed by level number
    let mut offsets = Vec::new();

    for level in 0..num_levels {
        // Each element of the outer vec is another vec of offset arrays for that level
        let mut this_level_offsets = Vec::new();

        // Get voxel coords of all the voxels in this shell
        for vox_x in -level..=level {
            for vox_y in -level..=level {
                for vox_z in -level..=level {
                    if vox_x.abs() == level || vox_y.abs() == level || vox_z.abs() == level {
                        this_level_offsets.push(vec![vox_x, vox_y, vox_z]);
                    };
                }
            }
        }

        // Push our populated inner vec of offset arrays into the outer vec
        offsets.push(this_level_offsets);
    }

    offsets
}

/// Compute voxel coordinates from point coordinates
fn voxel_coord(point_coord: f32, voxel_size: f32) -> i32 {
    (point_coord / voxel_size) as i32
}

