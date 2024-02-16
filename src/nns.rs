use std::collections::HashMap;
use std::vec;

use indicatif::ProgressIterator;
use ndarray::array;
use numpy::ndarray::{Array2, ArrayView1, ArrayView2, ArrayViewMut1, Axis};

/// Perform initial passes over search points, preparing data structures for querying
///
/// Args:
///     search_points: Pointcloud we are searching for neighbours within (S, 3)
///     max_dist: Furthest distance to neighbouring points before we don't care about them
///
/// Returns:
///     Mapping from voxel coordinates to search point indices
///     Voxel coordinate offsets for the shell of voxels surrounding a given voxel
///     Triangulation point coordinates
///     Distance from each search point to triangulation points
pub fn initialise_nns(
    search_points: ArrayView2<f32>,
    max_dist: f32,
) -> (
    HashMap<(i32, i32, i32), Vec<i32>>,
    Array2<i32>,
) {
    // 2nd pass: Construct points_by_voxel mapping and compute distance from triagulation
    // points for each search point
    let points_by_voxel =
        _group_by_voxel(&search_points, max_dist);

    // Compute voxel offsets for local field of voxels
    let voxel_offsets = _compute_voxel_offsets();

    (
        points_by_voxel,
        voxel_offsets,
    )
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
pub fn find_neighbours(
    query_points: ArrayView2<f32>,
    search_points: &Array2<f32>,
    points_by_voxel: &HashMap<(i32, i32, i32), Vec<i32>>,
    voxel_offsets: &Array2<i32>,
    num_neighbours: i32,
    max_dist: f32,
    exact: bool,
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
        .progress_count(num_query_points as u64)
        .for_each(|((query_point, indices_row), distances_row)| {
            _find_query_point_neighbours(
                query_point,
                indices_row,
                distances_row,
                search_points,
                &points_by_voxel,
                voxel_offsets,
                num_neighbours,
                max_dist,
                exact,
            );
        });

    (indices, distances)
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
fn _find_query_point_neighbours(
    query_point: ArrayView1<f32>,
    mut indices_row: ArrayViewMut1<i32>,
    mut distances_row: ArrayViewMut1<f32>,
    search_points: &Array2<f32>,
    points_by_voxel: &HashMap<(i32, i32, i32), Vec<i32>>,
    voxel_offsets: &Array2<i32>,
    num_neighbours: i32,
    max_dist: f32,
    exact: bool,
) {
    // Define volumetric ratio of 3 unit sided cube to unit sphere, to predict how many
    // neighbouring points within 3x3 voxel cube lie within search radius
    let vox_to_sphere_ratio = 6.5f32;

    // Compute voxel coords of query point
    let query_voxel = (
        voxel_coord(query_point[0], max_dist),
        voxel_coord(query_point[1], max_dist),
        voxel_coord(query_point[2], max_dist),
    );

    // If not using EXACT mode algorithm, and enough points are present in local field,
    // find a suitable n such that only taking every n-th point will still very likely
    // find enough search points
    let mut relevant_neighbour_indices: Vec<i32> = Vec::new();
    let step_size = if exact {
        1
    } else {
        let mut num_points: usize = 0;
        for voxel_offset in voxel_offsets.rows() {
            let this_voxel = (
                query_voxel.0 + voxel_offset[0],
                query_voxel.1 + voxel_offset[1],
                query_voxel.2 + voxel_offset[2],
            );
            if let Some(voxel_point_indices) = points_by_voxel.get(&this_voxel) {
                relevant_neighbour_indices.extend(voxel_point_indices);
            }
        }
        (((relevant_neighbour_indices.len() as f32 / vox_to_sphere_ratio) / num_neighbours as f32)
            as usize)
            .max(1)
    };

    // When using exact algo, sort all neighbours
    if exact {
        let mut neighbours: Vec<(i32, f32)> = Vec::new();
        for search_point_idx in relevant_neighbour_indices.iter() {
            let distance =
                compute_l2_distance(query_point, search_points.row(*search_point_idx as usize));
            if distance < max_dist {
                neighbours.push((*search_point_idx, distance));
            }
        }
        // Sort the remaining points and take as many of the closest as required
        neighbours.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        for (j, (point_idx, distance)) in
            neighbours.iter().take(num_neighbours as usize).enumerate()
        {
            indices_row[j] = *point_idx;
            distances_row[j] = *distance;
        }
    } else {
        // If using sample/inexact algo, take points evenly distributed amongst relevant neighbours
        let step_size = (relevant_neighbour_indices.len() / num_neighbours as usize).max(1);
        let mut i = 0;
        for search_point_idx in relevant_neighbour_indices.iter().step_by(step_size) {
            let distance =
                compute_l2_distance(query_point, search_points.row(*search_point_idx as usize));

            // Add to output array if passes L2 criteria
            if distance < max_dist {
                indices_row[i] = *search_point_idx;
                distances_row[i] = distance;
                i += 1;
            }
            if i >= num_neighbours as usize {
                break;
            }
        }
    }
}

/// Compute L2 (euclidean) distance between two points
fn compute_l2_distance(point_a: ArrayView1<f32>, point_b: ArrayView1<f32>) -> f32 {
    let delta = point_a.to_owned() - point_b.to_owned();
    let dx = delta[0];
    let dy = delta[1];
    let dz = delta[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Find bounding box around pointcloud and compute triangulation point locations
fn _compute_triangulation_points(search_points: &ArrayView2<f32>) -> Array2<f32> {
    let mut min_x = search_points[[0, 0]];
    let mut max_x = search_points[[0, 0]];
    let mut min_y = search_points[[0, 1]];
    let mut max_y = search_points[[0, 1]];
    let mut min_z = search_points[[0, 2]];
    let mut max_z = search_points[[0, 2]];

    for point in search_points.axis_iter(Axis(0)) {
        if point[0] > max_x {
            max_x = point[0];
        } else if point[0] < min_x {
            min_x = point[0];
        }
        if point[1] > max_y {
            max_y = point[1];
        } else if point[1] < min_y {
            min_y = point[1];
        }
        if point[2] > max_z {
            max_z = point[2];
        } else if point[2] < min_z {
            min_z = point[2];
        }
    }

    let dx = max_x - min_x;
    let dy = max_y - min_y;
    let dz = max_z - min_z;

    Array2::<f32>::from(array![
        [dx / 2.0, dy / 2.0, -dz],
        [dx / 2.0, 2.0 * dy, dz / 2.0],
        [-dx, dy / 2.0, dz / 2.0],
    ])
}

/// Generate voxel coordinates for each point (i.e. find which voxel each point
/// belongs to), and construct a hashmap of search point indices, indexed by voxel
/// coordinates
///
/// While we're here, we compute distances to the triangulation points
///
/// This is the second pass through the points we will make
fn _group_by_voxel(
    search_points: &ArrayView2<f32>,
    voxel_size: f32,
) -> HashMap<(i32, i32, i32), Vec<i32>> {
    // Construct mapping from voxel coords to point indices
    let mut points_by_voxel = HashMap::new();

    // Construct an array to store each point's voxel coords
    let num_points = search_points.shape()[0];

    // Compute voxel index for each point and add to hashmap
    for i in 0..num_points {
        // Compute voxel coords
        let voxel_coords = (
            voxel_coord(search_points[[i, 0]], voxel_size),
            voxel_coord(search_points[[i, 1]], voxel_size),
            voxel_coord(search_points[[i, 2]], voxel_size),
        );
        let point_indices: &mut Vec<i32> =
            points_by_voxel.entry(voxel_coords).or_insert(Vec::new());
        point_indices.push(i as i32);
    }

    // Return the voxel indices array and the hashmap
    points_by_voxel
}

/// Construct array to generate relatie voxel coordinates (i.e. offsets) of neighbouring voxels
fn _compute_voxel_offsets() -> Array2<i32> {
    let mut voxel_offsets: Array2<i32> = Array2::zeros((27, 3));
    let mut idx = 0;
    for x in -1..=1 {
        for y in -1..=1 {
            for z in -1..=1 {
                voxel_offsets[[idx, 0]] = x;
                voxel_offsets[[idx, 1]] = y;
                voxel_offsets[[idx, 2]] = z;
                idx += 1;
            }
        }
    }
    voxel_offsets
}

// V1 functions

/// Compute voxel coordinates from point coordinates
fn voxel_coord(point_coord: f32, voxel_size: f32) -> i32 {
    (point_coord / voxel_size) as i32
}
