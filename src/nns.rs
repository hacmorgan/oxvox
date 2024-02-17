/// Nearest neighbour search logic
///
///
use std::collections::{HashMap, HashSet};
use std::vec;

use indicatif::ProgressIterator;
use ndarray::{array, Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, Axis};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// X, Y, and Z axis positions in (N, 3) Array2s
const X: usize = 0;
const Y: usize = 1;
const Z: usize = 2;

// Struct to store point indices sorted by coordinates in each axis
#[derive(Serialize, Deserialize)]
pub struct SortedIndices {
    x: Vec<usize>,
    y: Vec<usize>,
    z: Vec<usize>,
}

// Struct used to store which axes we have gone beyond the search radius in, to save redundant checks and terminate early
struct SearchableAxes {
    x_upper: bool,
    x_lower: bool,
    y_upper: bool,
    y_lower: bool,
    z_upper: bool,
    z_lower: bool,
}

// Struct used to store min and max values for each axis for a query
struct CoordinateLimits {
    x_upper: f32,
    x_lower: f32,
    y_upper: f32,
    y_lower: f32,
    z_upper: f32,
    z_lower: f32,
}

/// Perform initial passes over search points, preparing data structures for querying
///
/// Args:
///     search_points: Pointcloud we are searching for neighbours within (S, 3)
///     max_dist: Furthest distance to neighbouring points before we don't care about them
///
/// Returns:
///     Point indices sorted by x coordinate
///     Point indices sorted by y coordinate
///     Point indices sorted by z coordinate
pub fn initialise_nns(search_points: ArrayView2<f32>) -> SortedIndices {
    // Construct an array of point indices, and sort by x coordinate
    let mut indices_x: Vec<usize> = (0..search_points.nrows()).collect();
    indices_x.par_sort_by(|a, b| {
        search_points[[*a, X]]
            .partial_cmp(&search_points[[*b, X]])
            .unwrap()
    });

    // Make copies for y and z axes and sort them
    let mut indices_y = indices_x.clone();
    indices_y.par_sort_by(|a, b| {
        search_points[[*a, Y]]
            .partial_cmp(&search_points[[*b, Y]])
            .unwrap()
    });
    let mut indices_z = indices_x.clone();
    indices_z.par_sort_by(|a, b| {
        search_points[[*a, Z]]
            .partial_cmp(&search_points[[*b, Z]])
            .unwrap()
    });

    // Return sorted indices arrays in a struct for convenience
    SortedIndices {
        x: indices_x,
        y: indices_y,
        z: indices_z,
    }
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
    num_neighbours: i32,
    max_dist: f32,
    sorted_indices: &SortedIndices,
) -> (Array2<i32>, Array2<f32>) {
    // Compute useful metadata
    let num_query_points = query_points.nrows();

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
            find_query_point_neighbours(
                query_point,
                indices_row,
                distances_row,
                search_points,
                num_neighbours,
                max_dist,
                sorted_indices,
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
///     num_neighbours: Maximum number of neighbours desired
///     max_dist: Biggest permissible distance from query point for a search point to be
///         considered a neighbour
fn find_query_point_neighbours(
    query_point: ArrayView1<f32>,
    mut indices_row: ArrayViewMut1<i32>,
    mut distances_row: ArrayViewMut1<f32>,
    search_points: &Array2<f32>,
    num_neighbours: i32,
    max_dist: f32,
    sorted_indices: &SortedIndices,
) {
    // Define an accuracy for finding where our query point sits among search points
    let epsilon: f32 = max_dist / 100.0;

    // Find where query point would sit in sorted indices
    let query_indices = (
        find_index_in_sorted_indices(
            query_point[X],
            &sorted_indices.x,
            search_points.column(X),
            epsilon,
        ),
        find_index_in_sorted_indices(
            query_point[Y],
            &sorted_indices.y,
            search_points.column(Y),
            epsilon,
        ),
        find_index_in_sorted_indices(
            query_point[Z],
            &sorted_indices.z,
            search_points.column(Z),
            epsilon,
        ),
    );

    // Construct coordinate limits in each axis
    let coord_limits = CoordinateLimits {
        x_upper: query_point[X] + max_dist,
        x_lower: query_point[X] - max_dist,
        y_upper: query_point[Y] + max_dist,
        y_lower: query_point[Y] - max_dist,
        z_upper: query_point[Z] + max_dist,
        z_lower: query_point[Z] - max_dist,
    };

    // Construct a tracker for axes in which we have either gone beyond search radius or reached the end of the array
    let mut searchable_axes = SearchableAxes::new();

    // Construct a set to track which point indices we have already seen
    let mut seen_points: HashSet<usize> = HashSet::new();

    // We will insert values to the output array sequentially until we have enough
    let mut output_point_idx: usize = 0;

    // Kepp radiating out in each axis until we have found enough points or have exhausted our valid search points
    let mut search_level: usize = 0;
    loop {

        let this_level_neighbours = searchable_axes.valid_neighbours(search_level, query_indices, sorted_indices);
        // println!("First neighbour: {}", this_level_neighbours[0]);
        // At some point we will have run out of neighbours within range
        if this_level_neighbours.len() == 0 {
            return;
        }

        // Find valid neighbours at this
        for neighbour_idx in this_level_neighbours.iter() {
            check_point(
                *neighbour_idx,
                query_point,
                &mut indices_row,
                &mut distances_row,
                search_points,
                max_dist,
                &coord_limits,
                &mut seen_points,
                &mut output_point_idx,
            );

            // If we have found enough neighbours we can exit here
            if output_point_idx >= num_neighbours as usize {
                return;
            }
        }

        search_level+= 1;
    }
}

/// Check if a given search point is within range and update output columns if so
fn check_point(
    search_point_idx: usize,
    query_point: ArrayView1<f32>,
    mut indices_row:   &mut ArrayViewMut1<i32>,
    mut distances_row: &mut ArrayViewMut1<f32>,
    search_points: &Array2<f32>,
    max_dist: f32,
    coord_limits: &CoordinateLimits,
    mut seen_points: &mut HashSet<usize>,
    mut output_point_idx: &mut usize,
) {
    // Exit early if we have already seen this point
    if seen_points.contains(&search_point_idx) {
        return;
    }

    // Exit early if any coords lie outside range by coordinate value
    let search_point = search_points.row(search_point_idx);
    if search_point[X] < coord_limits.x_lower
        || search_point[X] > coord_limits.x_upper
        || search_point[Y] < coord_limits.y_lower
        || search_point[Y] > coord_limits.y_upper
        || search_point[Z] < coord_limits.z_lower
        || search_point[Z] > coord_limits.z_upper
    {
        return;
    }

    // Lastly compute L2 distance and append to output array if within range
    let distance = compute_l2_distance(query_point, search_points.row(search_point_idx));
    if distance < max_dist {
        // Save point idx so we can skip early if we see this point again in another axis
        seen_points.insert(search_point_idx);

        // Write to output array
        indices_row[*output_point_idx] = search_point_idx as i32;
        distances_row[*output_point_idx] = distance;
        *output_point_idx += 1;
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

/// Find where this point coordinate would sit within search point indices sorted by coordinate value
fn find_index_in_sorted_indices(
    point_coord: f32,
    sorted_indices_column: &Vec<usize>,
    search_coord_column: ArrayView1<f32>,
    epsilon: f32,
) -> usize {
    // Perform binary search until we get close enough to point
    let mut idx: usize = search_coord_column.len() / 2;
    let mut step_size = idx;
    loop {
        let delta = search_coord_column[sorted_indices_column[ idx]] - point_coord ;
        // If we are down to the smallest subdivision, just return here
        if step_size <= 1 {
            break;
        }

        // If we're within range of our query point, return current idx
        if delta < epsilon {
            break;
        }

        // Otherwise we'll take another step
        step_size = step_size / 2;
        if delta > 0.0 {
            idx -= step_size;
        } else {
            idx += step_size;
        }
    }

    idx
}

impl SearchableAxes {
    /// Construct fresh SearchableAxes object with all axes searchable
    pub fn new() -> Self {
        SearchableAxes {
            x_lower: true,
            x_upper: true,
            y_lower: true,
            y_upper: true,
            z_lower: true,
            z_upper: true,
        }
    }

    /// Find search point indices of neighbours at a given search level in all
    /// directions, updating state as we go
    pub fn valid_neighbours(
        &mut self,
        search_level: usize,
        query_indices: (usize, usize, usize),
        sorted_indices: &SortedIndices,
    ) -> Vec<usize> {
        let mut neighbour_indices: Vec<usize> = Vec::new();

        // For each axis in each direction, check points at this level
        if self.x_upper {
            let search_point_idx_idx = query_indices.0 + search_level;
            if search_point_idx_idx < 0 || search_point_idx_idx >= sorted_indices.x.len() {
                self.x_upper = false;
            } else {
                neighbour_indices.push(sorted_indices.x[search_point_idx_idx]);
            }
        }
        if self.x_lower && search_level > 0 {
            if search_level > query_indices.0 {
                self.x_lower = false;
            } else {
                let search_point_idx_idx = query_indices.0 - search_level;
                neighbour_indices.push(sorted_indices.x[search_point_idx_idx]);
            }
        }
        if self.y_upper {
            let search_point_idx_idx = query_indices.1 + search_level;
            if search_point_idx_idx >= sorted_indices.y.len() || search_point_idx_idx < 0 {
                self.y_upper = false;
            } else {
                neighbour_indices.push(sorted_indices.y[search_point_idx_idx]);
            }
        }
        if self.y_lower && search_level > 0 {
            if search_level > query_indices.1 {
                self.y_lower = false;
            } else {
                let search_point_idx_idx = query_indices.1 - search_level;
                neighbour_indices.push(sorted_indices.y[search_point_idx_idx]);
            }
        }
        if self.z_upper {
            let search_point_idx_idx = query_indices.2 + search_level;
            if search_point_idx_idx >= sorted_indices.z.len() ||search_point_idx_idx < 0 {
                self.z_upper = false;
            } else {
                neighbour_indices.push(sorted_indices.z[search_point_idx_idx]);
            }
        }
        if self.z_lower && search_level > 0 {
            if search_level > query_indices.2 {
                self.z_lower = false;
            } else {
            let search_point_idx_idx = query_indices.2 - search_level;
                neighbour_indices.push(sorted_indices.z[search_point_idx_idx]);
            }
        }

        neighbour_indices
    }
}

//     //  old stuff

//     // If not using EXACT mode algorithm, and enough points are present in local field,
//     // find a suitable n such that only taking every n-th point will still very likely
//     // find enough search points
//     let step_size = if !exact {
//         let mut num_points: usize = 0;
//         for voxel_offset in voxel_offsets.rows() {
//             let this_voxel = (
//                 query_voxel.0 + voxel_offset[0],
//                 query_voxel.1 + voxel_offset[1],
//                 query_voxel.2 + voxel_offset[2],
//             );
//             if let Some(voxel_point_indices) = points_by_voxel.get(&this_voxel) {
//                 num_points += voxel_point_indices.len()
//             }
//         }
//         (((num_points as f32 / vox_to_sphere_ratio) / num_neighbours as f32) as usize).max(1)
//     } else {
//         1
//     };

//     // Use triangulation points to find a subset of neighbours that are likely within range
//     let mut relevant_neighbour_indices: Vec<i32> = Vec::new();
//     for voxel_offset in voxel_offsets.rows() {
//         // Construct voxel coords tuple for this voxel
//         let this_voxel = (
//             query_voxel.0 + voxel_offset[0],
//             query_voxel.1 + voxel_offset[1],
//             query_voxel.2 + voxel_offset[2],
//         );

//         // Only proceed if this voxel actually contains any points
//         let this_voxel_point_indices = match points_by_voxel.get(&this_voxel) {
//             Some(o) => o,
//             None => continue,
//         };

//         // Filter out
//         for point_idx in this_voxel_point_indices
//             .iter()
//             .step_by(step_size)
//             .filter(|&i| {
//                 let idx = *i as usize;
//                 (triangulation_distances[[idx, 0]] - d_a).abs() < max_dist
//                     && (triangulation_distances[[idx, 1]] - d_b).abs() < max_dist
//                     && (triangulation_distances[[idx, 2]] - d_c).abs() < max_dist
//             })
//         {
//             relevant_neighbour_indices.push(*point_idx);
//         }
//     }

//     // When using exact algo, sort all neighbours
//     if exact {
//         let mut neighbours: Vec<(i32, f32)> = Vec::new();
//         for search_point_idx in relevant_neighbour_indices.iter() {
//             let distance =
//                 compute_l2_distance(query_point, search_points.row(*search_point_idx as usize));
//             if distance < max_dist {
//                 neighbours.push((*search_point_idx, distance));
//             }
//         }
//         // Sort the remaining points and take as many of the closest as required
//         neighbours.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
//         for (j, (point_idx, distance)) in
//             neighbours.iter().take(num_neighbours as usize).enumerate()
//         {
//             indices_row[j] = *point_idx;
//             distances_row[j] = *distance;
//         }
//     } else {
//         // If using sample/inexact algo, take points evenly distributed amongst relevant neighbours
//         let step_size = (relevant_neighbour_indices.len() / num_neighbours as usize).max(1);
//         let mut i = 0;
//         for search_point_idx in relevant_neighbour_indices.iter().step_by(step_size) {
//             let distance =
//                 compute_l2_distance(query_point, search_points.row(*search_point_idx as usize));

//             // Add to output array if passes L2 criteria
//             if distance < max_dist {
//                 indices_row[i] = *search_point_idx;
//                 distances_row[i] = distance;
//                 i += 1;
//             }
//             if i >= num_neighbours as usize {
//                 break;
//             }
//         }
//     }
// }

// /// Find bounding box around pointcloud and compute triangulation point locations
// fn _compute_triangulation_points(search_points: &ArrayView2<f32>) -> Array2<f32> {
//     let mut min_x = search_points[[0, 0]];
//     let mut max_x = search_points[[0, 0]];
//     let mut min_y = search_points[[0, 1]];
//     let mut max_y = search_points[[0, 1]];
//     let mut min_z = search_points[[0, 2]];
//     let mut max_z = search_points[[0, 2]];

//     for point in search_points.axis_iter(Axis(0)) {
//         if point[0] > max_x {
//             max_x = point[0];
//         } else if point[0] < min_x {
//             min_x = point[0];
//         }
//         if point[1] > max_y {
//             max_y = point[1];
//         } else if point[1] < min_y {
//             min_y = point[1];
//         }
//         if point[2] > max_z {
//             max_z = point[2];
//         } else if point[2] < min_z {
//             min_z = point[2];
//         }
//     }

//     let dx = max_x - min_x;
//     let dy = max_y - min_y;
//     let dz = max_z - min_z;

//     Array2::<f32>::from(array![
//         [dx / 2.0, dy / 2.0, -dz],
//         [dx / 2.0, 2.0 * dy, dz / 2.0],
//         [-dx, dy / 2.0, dz / 2.0],
//     ])
// }

// /// Generate voxel coordinates for each point (i.e. find which voxel each point
// /// belongs to), and construct a hashmap of search point indices, indexed by voxel
// /// coordinates
// ///
// /// While we're here, we compute distances to the triangulation points
// ///
// /// This is the second pass through the points we will make
// fn triangulate_points(
//     search_points: &ArrayView2<f32>,
//     triangulation_points: &Array2<f32>,
// ) -> (HashMap<(i32, i32, i32), Vec<i32>>, Array2<f32>) {
//     // Construct mapping from voxel coords to point indices
//     let mut points_by_voxel = HashMap::new();

//     // Construct an array to store each point's voxel coords
//     let num_points = search_points.shape()[0];
//     let mut triangulation_distances: Array2<f32> = Array2::zeros([num_points, 3]);

//     // Compute voxel index for each point and add to hashmap
//     for i in 0..num_points {
//         // Compute voxel coords
//         let voxel_coords = (
//             voxel_coord(search_points[[i, 0]], voxel_size),
//             voxel_coord(search_points[[i, 1]], voxel_size),
//             voxel_coord(search_points[[i, 2]], voxel_size),
//         );
//         let point_indices: &mut Vec<i32> =
//             points_by_voxel.entry(voxel_coords).or_insert(Vec::new());
//         point_indices.push(i as i32);

//         // Compute distances to triangulation points
//         for j in 0..3 {
//             triangulation_distances[[i, j]] = {
//                 let dx = triangulation_points[[j, 0]] - search_points[[i, 0]];
//                 let dy = triangulation_points[[j, 1]] - search_points[[i, 1]];
//                 let dz = triangulation_points[[j, 2]] - search_points[[i, 2]];
//                 (dx * dx + dy * dy + dz * dz).sqrt()
//             }
//         }
//     }

//     // Return the voxel indices array and the hashmap
//     (points_by_voxel, triangulation_distances)
// }

// /// Construct array to generate relatie voxel coordinates (i.e. offsets) of neighbouring voxels
// fn _compute_voxel_offsets() -> Array2<i32> {
//     let mut voxel_offsets: Array2<i32> = Array2::zeros((27, 3));
//     let mut idx = 0;
//     for x in -1..=1 {
//         for y in -1..=1 {
//             for z in -1..=1 {
//                 voxel_offsets[[idx, 0]] = x;
//                 voxel_offsets[[idx, 1]] = y;
//                 voxel_offsets[[idx, 2]] = z;
//                 idx += 1;
//             }
//         }
//     }
//     voxel_offsets
// }

// // V1 functions

// /// Compute voxel coordinates from point coordinates
// fn voxel_coord(point_coord: f32, voxel_size: f32) -> i32 {
//     (point_coord / voxel_size) as i32
// }
