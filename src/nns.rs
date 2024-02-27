use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};

use indicatif::{ParallelProgressIterator, ProgressIterator};

use ndarray::parallel::prelude::*;
use ndarray::{Array2, ArrayView1, ArrayView2, ArrayViewMut1, Axis};

const SPARSE_NUM_PASSES: usize = 3;

struct Neighbour {
    search_point_idx: i32,
    distance: f32,
}

impl Ord for Neighbour {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.distance < other.distance {
            Ordering::Less
        } else if self.distance > other.distance {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }
}

impl PartialOrd for Neighbour {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for Neighbour {}

impl PartialEq for Neighbour {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

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
    search_points: &Array2<f32>,
    max_dist: f32,
) -> (HashMap<(i32, i32, i32), Vec<i32>>, Array2<i32>) {
    // 2nd pass: Construct points_by_voxel mapping and compute distance from triagulation
    // points for each search point
    let points_by_voxel = _group_by_voxel(search_points, max_dist);

    // Compute voxel offsets for local field of voxels
    let voxel_offsets = _compute_voxel_offsets();

    (points_by_voxel, voxel_offsets)
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
pub fn find_neighbours_singlethread(
    query_points: ArrayView2<f32>,
    search_points: &Array2<f32>,
    points_by_voxel: &HashMap<(i32, i32, i32), Vec<i32>>,
    voxel_offsets: &Array2<i32>,
    num_neighbours: i32,
    max_dist: f32,
    epsilon: f32,
    sparse: bool,
    l2_distance: bool,
) -> (Array2<i32>, Array2<f32>) {
    // Compute useful metadata
    let num_query_points = query_points.shape()[0];

    // Precompute query voxels
    let query_voxels: Array2<i32> = query_points.map(|&x| (x / max_dist) as i32);

    // Construct output arrays, initialised with -1s
    let mut indices: Array2<i32> =
        Array2::from_elem([num_query_points, num_neighbours as usize], -1i32);
    let mut distances: Array2<f32> =
        Array2::from_elem([num_query_points, num_neighbours as usize], -1f32);

    // Map query point processing function across corresponding rows of query
    // points, indices, and distances arrays
    query_points
        .axis_iter(Axis(0))
        .zip(query_voxels.axis_iter(Axis(0)))
        .zip(indices.axis_iter_mut(Axis(0)))
        .zip(distances.axis_iter_mut(Axis(0)))
        .progress_count(num_query_points as u64)
        .for_each(
            |(((query_point, query_voxel), indices_row), distances_row)| {
                _find_query_point_neighbours(
                    query_point,
                    query_voxel,
                    indices_row,
                    distances_row,
                    &search_points,
                    &points_by_voxel,
                    voxel_offsets,
                    num_neighbours,
                    max_dist,
                    epsilon,
                    sparse,
                    l2_distance,
                );
            },
        );

    (indices, distances)
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
    epsilon: f32,
    sparse: bool,
    l2_distance: bool,
) -> (Array2<i32>, Array2<f32>) {
    // Precompute query voxels
    let query_voxels: Array2<i32> = query_points.map(|&x| (x / max_dist) as i32);

    // Construct output arrays, initialised with -1s
    let num_query_points = query_points.shape()[0];
    let mut indices: Array2<i32> =
        Array2::from_elem([num_query_points, num_neighbours as usize], -1i32);
    let mut distances: Array2<f32> =
        Array2::from_elem([num_query_points, num_neighbours as usize], -1f32);

    // Map query point processing function across corresponding rows of query
    // points, indices, and distances arrays
    query_points
        .axis_iter(Axis(0))
        .into_par_iter()
        .zip(query_voxels.axis_iter(Axis(0)))
        .zip(indices.axis_iter_mut(Axis(0)))
        .zip(distances.axis_iter_mut(Axis(0)))
        .progress_count(num_query_points as u64)
        .for_each(
            |(((query_point, query_voxel), indices_row), distances_row)| {
                _find_query_point_neighbours(
                    query_point,
                    query_voxel,
                    indices_row,
                    distances_row,
                    &search_points,
                    &points_by_voxel,
                    voxel_offsets,
                    num_neighbours,
                    max_dist,
                    epsilon,
                    sparse,
                    l2_distance,
                );
            },
        );

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
    query_voxel: ArrayView1<i32>,
    mut indices_row: ArrayViewMut1<i32>,
    mut distances_row: ArrayViewMut1<f32>,
    search_points: &Array2<f32>,
    points_by_voxel: &HashMap<(i32, i32, i32), Vec<i32>>,
    voxel_offsets: &Array2<i32>,
    num_neighbours: i32,
    max_dist: f32,
    epsilon: f32,
    sparse: bool,
    l2_distance: bool,
) {
    // Find indices of all neighbours in 3x3 local area
    let mut relevant_neighbour_indices: Vec<i32> = Vec::new();
    let voxels_iter = voxel_offsets.axis_iter(Axis(0)).map(|voxel_offset| {
        (
            query_voxel[0] + voxel_offset[0],
            query_voxel[1] + voxel_offset[1],
            query_voxel[2] + voxel_offset[2],
        )
    });

    // Find how many points are within our local field
    let mut num_points = 0;
    for voxel in voxels_iter.clone() {
        if let Some(voxel_point_indices) = points_by_voxel.get(&voxel) {
            num_points += voxel_point_indices.len();
        }
    }
    relevant_neighbour_indices.reserve_exact(num_points);
    for voxel in voxels_iter {
        if let Some(voxel_point_indices) = points_by_voxel.get(&voxel) {
            relevant_neighbour_indices.extend(voxel_point_indices);
        }
    }

    // If we could be stopping early, we want to make a few passes over the points to
    // help get a better distribution of the points
    let step_size = {
        (relevant_neighbour_indices.len() * SPARSE_NUM_PASSES / (num_neighbours as usize)).max(1)
    };

    // Construct an iterator of interleaved slices of points to help spread out search across multiple passes
    let neighbours_iter =
        (0..step_size).flat_map(|i| relevant_neighbour_indices.iter().skip(i).step_by(step_size));
    let neighbours_within_range = neighbours_iter
        .map(|&idx| Neighbour {
            search_point_idx: idx,
            distance: compute_distance(query_point, search_points.row(idx as usize), l2_distance),
        })
        .filter(|neighbour| neighbour.distance < max_dist);

    // When using exact algo, add all neighbours to BinaryHeap and pop off K elements
    if ! sparse {

        // Construct the binary heap and reserve enough memory for all neighbours to fit
        let mut neighbours = BinaryHeap::new();
        neighbours.reserve_exact(relevant_neighbour_indices.len());

        // Add all points within range to the binary heap
        let mut num_neighbours_within_eps = 0;
        for neighbour in neighbours_within_range {
            if neighbour.distance < epsilon {
                num_neighbours_within_eps += 1;
            }
            neighbours.push(Reverse(neighbour));

            // Stop early if we've found k neighbours within range eps of query point
            if num_neighbours_within_eps >= num_neighbours {
                break;
            }
        }

        // Pop as many elements as required off the heap
        for i in 0..neighbours.len().min(num_neighbours as usize) {
            if i == neighbours.len() - 1 {
                // `peek()` the last element to save a log(neighbours.len()) lookup for the next smallest element
                let neighbour = neighbours.peek().unwrap();
                indices_row[i] = neighbour.0.search_point_idx;
                distances_row[i] = neighbour.0.distance;
            } else {
                // `pop()` all other elements because we still need to know the next smallest element
                let neighbour = neighbours.pop().unwrap();
                indices_row[i] = neighbour.0.search_point_idx;
                distances_row[i] = neighbour.0.distance;
            };
        }
        
    } else {
        // If using sample/inexact algo, take points evenly distributed amongst relevant neighbours
        for (i, neighbour) in neighbours_within_range
            .take(num_neighbours as usize)
            .enumerate()
        {
            indices_row[i] = neighbour.search_point_idx;
            distances_row[i] = neighbour.distance;
        }
    }
}

/// Compute L2 (euclidean) distance between two points
fn compute_distance(point_a: ArrayView1<f32>, point_b: ArrayView1<f32>, l2_distance: bool) -> f32 {
    let dx = point_a[0] - point_b[0];
    let dy = point_a[1] - point_b[1];
    let dz = point_a[2] - point_b[2];
    if l2_distance {
        (dx * dx + dy * dy + dz * dz).sqrt()
    } else {
        dx + dy + dz
    }
}

/// Generate voxel coordinates for each point (i.e. find which voxel each point
/// belongs to), and construct a hashmap of search point indices, indexed by voxel
/// coordinates
///
/// While we're here, we compute distances to the triangulation points
///
/// This is the second pass through the points we will make
fn _group_by_voxel(
    search_points: &Array2<f32>,
    voxel_size: f32,
) -> HashMap<(i32, i32, i32), Vec<i32>> {
    // Construct mapping from voxel coords to point indices
    let mut points_by_voxel = HashMap::new();

    // Construct an array to store each point's voxel coords
    let _num_points = search_points.shape()[0];

    // Compute voxel index for each point and add to hashmap
    let voxel_indices: Array2<i32> = search_points.map(|&x| (x / voxel_size) as i32);

    // Construct point indices lookup
    for (i, voxel_row) in voxel_indices.axis_iter(Axis(0)).enumerate() {
        let voxel_coords = (voxel_row[0], voxel_row[1], voxel_row[2]);
        let point_indices: &mut Vec<i32> =
            points_by_voxel.entry(voxel_coords).or_insert(Vec::new());
        point_indices.push(i as i32);
    }

    // Return the point indices lookup
    points_by_voxel
}

/// Construct array to generate relative voxel coordinates (i.e. offsets) of neighbouring voxels
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

// /// Generate iterable of interleaved rows of array
// ///
// /// e.g. for step_size = 10, restart_jump = 2, and an array of shape (30, _) this would
// ///     yield rows: 0, 10, 20, 2, 12, 22, 4, 14, 24, ...
// fn smart_slice<T>(
//     array: ArrayView2<T>,
//     step_size: usize,
//     restart_jump: usize,
// ) -> impl Iterator<Item  = &T>{
//     (0..step_size).flat_map(move |i| {
//             let start = i * step_size;
//             array.slice(s![start..;step_size, ..]).into_iter()
//         })

// array
//     .axis_iter(Axis(0))
//     .take(step_size)
//     .enumerate()
//     .flat_map(move |(i, row)| {
//         let start = (i % restart_jump) * step_size;
//         row.slice(s![start..;step_size]).into_iter()
//     })
// }
