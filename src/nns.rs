use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};

use indicatif::{ParallelProgressIterator, ProgressIterator};

use ndarray::parallel::prelude::*;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, Axis};

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
    // Group point indices by voxel into a hashmap indexed by voxel coordinates
    let points_by_voxel = _group_by_voxel(search_points.view(), max_dist);

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
    search_points_by_voxel: &HashMap<(i32, i32, i32), Vec<i32>>,
    voxel_offsets: &Array2<i32>,
    num_neighbours: i32,
    max_dist: f32,
    epsilon: f32,
) -> (Array2<i32>, Array2<f32>) {
    // Group query point indices by voxel into a hashmap indexed by voxel coordinates
    let query_points_by_voxel = _group_by_voxel(query_points, max_dist);

    // Extract keys (unique voxel coords) into a vec to ensure we iterate over the
    // hashmap consistently
    let keys: Vec<(i32, i32, i32)> = query_points_by_voxel.clone().into_keys().collect();

    // Zip search points, query points, and output array chunks together, to be
    // processed in parallel
    let processed_chunks: Vec<((Array2<i32>, Array2<f32>), (i32, i32, i32))> = keys
        .clone()
        .into_iter()
        .progress_count(keys.len() as u64)
        .map(|voxel| {
            (
                _process_query_point_voxel(
                    &voxel,
                    &query_points,
                    &query_points_by_voxel,
                    search_points,
                    search_points_by_voxel,
                    voxel_offsets,
                    num_neighbours,
                    max_dist,
                    epsilon,
                ),
                voxel,
            )
        })
        .collect();

    // Construct output arrays, initialised with -1s
    let num_query_points = query_points.shape()[0];
    let mut indices: Array2<i32> =
        Array2::from_elem([num_query_points, num_neighbours as usize], -1i32);
    let mut distances: Array2<f32> =
        Array2::from_elem([num_query_points, num_neighbours as usize], -1f32);

    // Insert values from processed voxel chunks back into output array
    processed_chunks
        .iter()
        .for_each(|((chunk_indices, chunk_distances), voxel)| {
            let query_point_indices = query_points_by_voxel.get(&voxel).unwrap();
            query_point_indices
                .iter()
                .map(|&idx| idx as usize)
                .zip(chunk_indices.axis_iter(Axis(0)))
                .zip(chunk_distances.axis_iter(Axis(0)))
                .for_each(
                    |((query_point_idx, chunk_indices_row), chunk_distances_row)| {
                        chunk_indices_row.assign_to(indices.slice_mut(s![query_point_idx, ..]));
                        chunk_distances_row.assign_to(distances.slice_mut(s![query_point_idx, ..]));
                    },
                )
        });

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
    search_points_by_voxel: &HashMap<(i32, i32, i32), Vec<i32>>,
    voxel_offsets: &Array2<i32>,
    num_neighbours: i32,
    max_dist: f32,
    epsilon: f32,
) -> (Array2<i32>, Array2<f32>) {
    // Group query point indices by voxel into a hashmap indexed by voxel coordinates
    let query_points_by_voxel = _group_by_voxel(query_points, max_dist);

    // Extract keys (unique voxel coords) into a vec to ensure we iterate over the
    // hashmap consistently
    let keys: Vec<(i32, i32, i32)> = query_points_by_voxel.clone().into_keys().collect();

    // Zip search points, query points, and output array chunks together, to be
    // processed in parallel
    let processed_chunks: Vec<((Array2<i32>, Array2<f32>), (i32, i32, i32))> = keys
        .clone()
        .into_par_iter()
        .progress_count(keys.len() as u64)
        .map(|voxel| {
            (
                _process_query_point_voxel(
                    &voxel,
                    &query_points,
                    &query_points_by_voxel,
                    search_points,
                    search_points_by_voxel,
                    voxel_offsets,
                    num_neighbours,
                    max_dist,
                    epsilon,
                ),
                voxel,
            )
        })
        .collect();

    // Construct output arrays, initialised with -1s
    let num_query_points = query_points.shape()[0];
    let mut indices: Array2<i32> =
        Array2::from_elem([num_query_points, num_neighbours as usize], -1i32);
    let mut distances: Array2<f32> =
        Array2::from_elem([num_query_points, num_neighbours as usize], -1f32);

    // Insert values from processed voxel chunks back into output array
    processed_chunks
        .iter()
        .for_each(|((chunk_indices, chunk_distances), voxel)| {
            let query_point_indices = query_points_by_voxel.get(&voxel).unwrap();
            query_point_indices
                .iter()
                .map(|&idx| idx as usize)
                .zip(chunk_indices.axis_iter(Axis(0)))
                .zip(chunk_distances.axis_iter(Axis(0)))
                .for_each(
                    |((query_point_idx, chunk_indices_row), chunk_distances_row)| {
                        chunk_indices_row.assign_to(indices.slice_mut(s![query_point_idx, ..]));
                        chunk_distances_row.assign_to(distances.slice_mut(s![query_point_idx, ..]));
                    },
                )
        });

    (indices, distances)
}

/// Run kNN search for all query points in a given voxel
fn _process_query_point_voxel(
    voxel: &(i32, i32, i32),
    query_points: &ArrayView2<f32>,
    query_points_by_voxel: &HashMap<(i32, i32, i32), Vec<i32>>,
    search_points: &Array2<f32>,
    search_points_by_voxel: &HashMap<(i32, i32, i32), Vec<i32>>,
    voxel_offsets: &Array2<i32>,
    num_neighbours: i32,
    max_dist: f32,
    epsilon: f32,
) -> (Array2<i32>, Array2<f32>) {
    // Extract search point neighbours for this voxel
    let (neighbours, neighbour_indices) = _get_neighbouring_search_points(
        voxel,
        search_points,
        search_points_by_voxel,
        voxel_offsets,
    );

    // Construct contiguous array for query points
    let query_point_indices = query_points_by_voxel.get(voxel).unwrap();
    let mut this_voxel_query_points = Array2::zeros([query_point_indices.len(), 3]);
    this_voxel_query_points
        .axis_iter_mut(Axis(0))
        .zip(query_point_indices.iter())
        .for_each(|(row, &idx)| {
            query_points.row(idx as usize).assign_to(row);
        });

    // Construct output arrays for this chunk
    let mut indices_chunk: Array2<i32> =
        Array2::from_elem([query_point_indices.len(), num_neighbours as usize], -1i32);
    let mut distances_chunk: Array2<f32> =
        Array2::from_elem([query_point_indices.len(), num_neighbours as usize], -1f32);

    // Map query point processing function across voxels of query points
    this_voxel_query_points
        .axis_iter(Axis(0))
        .zip(indices_chunk.axis_iter_mut(Axis(0)))
        .zip(distances_chunk.axis_iter_mut(Axis(0)))
        .for_each(|((query_point, indices_row), distances_row)| {
            _find_query_point_neighbours(
                query_point,
                indices_row,
                distances_row,
                &neighbours,
                &neighbour_indices,
                num_neighbours,
                max_dist,
                epsilon,
            );
        });

    (indices_chunk, distances_chunk)
}

fn _get_neighbouring_search_points(
    voxel: &(i32, i32, i32),
    search_points: &Array2<f32>,
    search_points_by_voxel: &HashMap<(i32, i32, i32), Vec<i32>>,
    voxel_offsets: &Array2<i32>,
) -> (Array2<f32>, Vec<i32>) {
    // Construct an iterator of our neighbouring voxels
    let voxels_iter = voxel_offsets.axis_iter(Axis(0)).map(|voxel_offset| {
        (
            voxel.0 + voxel_offset[0],
            voxel.1 + voxel_offset[1],
            voxel.2 + voxel_offset[2],
        )
    });

    // Find how many points are within our local field
    let mut num_neighbours = 0;
    for voxel in voxels_iter.clone() {
        if let Some(voxel_point_indices) = search_points_by_voxel.get(&voxel) {
            num_neighbours += voxel_point_indices.len();
        }
    }

    // Find indices of all neighbours in local field
    let mut relevant_neighbour_indices: Vec<i32> = Vec::new();
    relevant_neighbour_indices.reserve_exact(num_neighbours);
    for voxel in voxels_iter {
        if let Some(voxel_point_indices) = search_points_by_voxel.get(&voxel) {
            relevant_neighbour_indices.extend(voxel_point_indices);
        }
    }

    // Construct a contiguous array with neighbours coordinates
    let mut neighbours = Array2::zeros([num_neighbours, 3]);
    neighbours
        .axis_iter_mut(Axis(0))
        .zip(relevant_neighbour_indices.iter())
        .for_each(|(row, &idx)| {
            search_points.row(idx as usize).assign_to(row);
        });

    (neighbours, relevant_neighbour_indices)
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
///     search_point_indices: Point indices (for indexing original source points array)
///         corresponding to search points
///     num_neighbours:
///     max_dist:
///     epsilon:
fn _find_query_point_neighbours(
    query_point: ArrayView1<f32>,
    mut indices_row: ArrayViewMut1<i32>,
    mut distances_row: ArrayViewMut1<f32>,
    search_points: &Array2<f32>,
    search_point_indices: &Vec<i32>,
    num_neighbours: i32,
    max_dist: f32,
    epsilon: f32,
) {
    // Construct an iterator of only the neighbours that are within range
    let neighbours_within_range = search_points
        .axis_iter(Axis(0))
        .zip(search_point_indices.iter())
        .map(|(search_point, &idx)| Neighbour {
            search_point_idx: idx,
            distance: {
                let dx = query_point[0] - search_point[0];
                let dy = query_point[1] - search_point[1];
                let dz = query_point[2] - search_point[2];
                (dx * dx + dy * dy + dz * dz).sqrt()
            },
        })
        .filter(|neighbour| neighbour.distance < max_dist);

    // Construct the binary heap and reserve enough memory for all neighbours to fit
    let mut neighbours = BinaryHeap::new();
    neighbours.reserve_exact(search_point_indices.len());

    // Add all points within range to the binary heap, stopping early if we've found k
    // neighbours within range eps of query point
    let mut num_neighbours_within_eps = 0;
    for neighbour in neighbours_within_range {
        if neighbour.distance < epsilon {
            num_neighbours_within_eps += 1;
        }
        neighbours.push(Reverse(neighbour));
        if num_neighbours_within_eps >= num_neighbours {
            break;
        }
    }

    // Pop as many elements as required off the heap
    for i in 0..neighbours.len().min(num_neighbours as usize) {
        let neighbour = neighbours.pop().unwrap();
        indices_row[i] = neighbour.0.search_point_idx;
        distances_row[i] = neighbour.0.distance;
    }
}

/// Count the nearest neighbours within a given radius for each query point
///
/// Args:
///     search_points: Pointcloud we are searching for neighbours within (S, 3)
///     query_points: Points we are searching for the neighbours of (Q, 3)
///     max_dist: Furthest distance to neighbouring points before we don't care about them
///
/// Returns:
///     Indices of neighbouring points (Q, num_neighbours)
///     Distances of neighbouring points from query point (Q, num_neighbours)
pub fn count_neighbours_singlethread(
    query_points: ArrayView2<f32>,
    search_points: &Array2<f32>,
    search_points_by_voxel: &HashMap<(i32, i32, i32), Vec<i32>>,
    voxel_offsets: &Array2<i32>,
    max_dist: f32,
) -> Array1<u32> {
    // Group query point indices by voxel into a hashmap indexed by voxel coordinates
    let query_points_by_voxel = _group_by_voxel(query_points, max_dist);

    // Extract keys (unique voxel coords) into a vec to ensure we iterate over the
    // hashmap consistently
    let keys: Vec<(i32, i32, i32)> = query_points_by_voxel.clone().into_keys().collect();

    // Construct output arrays, initialised with 0s
    let num_query_points = query_points.shape()[0];
    let mut counts: Array1<u32> = Array1::from_elem([num_query_points], 0u32);

    // Zip search points, query points, and output array chunks together, to be
    // processed in parallel
    let processed_chunks: Vec<(Array1<u32>, (i32, i32, i32))> = keys
        .clone()
        .into_iter()
        .progress_count(keys.len() as u64)
        .map(|voxel| {
            (
                _count_query_point_voxel(
                    &voxel,
                    &query_points,
                    &query_points_by_voxel,
                    search_points,
                    search_points_by_voxel,
                    voxel_offsets,
                    max_dist,
                ),
                voxel,
            )
        })
        .collect();

    // Insert values from processed voxel chunks back into output array
    processed_chunks.iter().for_each(|(chunk_counts, voxel)| {
        let query_point_indices = query_points_by_voxel.get(&voxel).unwrap();
        query_point_indices
            .iter()
            .map(|&idx| idx as usize)
            .zip(chunk_counts.axis_iter(Axis(0)))
            .for_each(|(query_point_idx, chunk_count)| {
                chunk_count.assign_to(counts.slice_mut(s![query_point_idx]));
            })
    });

    counts
}

/// Find the (up to) N nearest neighbours within a given radius for each query point
///
/// Args:
///     search_points: Pointcloud we are searching for neighbours within (S, 3)
///     query_points: Points we are searching for the neighbours of (Q, 3)
///     num_neighbours: Maximum number of neighbours to search for
///     max_dist: Furthest distance to neighbouring points before we don't care about them
///     epsilon:
///
/// Returns:
///     Indices of neighbouring points (Q, num_neighbours)
///     Distances of neighbouring points from query point (Q, num_neighbours)
pub fn count_neighbours(
    query_points: ArrayView2<f32>,
    search_points: &Array2<f32>,
    search_points_by_voxel: &HashMap<(i32, i32, i32), Vec<i32>>,
    voxel_offsets: &Array2<i32>,
    max_dist: f32,
) -> Array1<u32> {
    // Group query point indices by voxel into a hashmap indexed by voxel coordinates
    let query_points_by_voxel = _group_by_voxel(query_points, max_dist);

    // Extract keys (unique voxel coords) into a vec to ensure we iterate over the
    // hashmap consistently
    let keys: Vec<(i32, i32, i32)> = query_points_by_voxel.clone().into_keys().collect();

    // Construct output arrays, initialised with 0s
    let num_query_points = query_points.shape()[0];
    let mut counts: Array1<u32> = Array1::from_elem([num_query_points], 0u32);

    // Zip search points, query points, and output array chunks together, to be
    // processed in parallel
    let processed_chunks: Vec<(Array1<u32>, (i32, i32, i32))> = keys
        .clone()
        .into_par_iter()
        .progress_count(keys.len() as u64)
        .map(|voxel| {
            (
                _count_query_point_voxel(
                    &voxel,
                    &query_points,
                    &query_points_by_voxel,
                    search_points,
                    search_points_by_voxel,
                    voxel_offsets,
                    max_dist,
                ),
                voxel,
            )
        })
        .collect();

    // Insert values from processed voxel chunks back into output array
    processed_chunks.iter().for_each(|(chunk_counts, voxel)| {
        let query_point_indices = query_points_by_voxel.get(&voxel).unwrap();
        query_point_indices
            .iter()
            .map(|&idx| idx as usize)
            .zip(chunk_counts.axis_iter(Axis(0)))
            .for_each(|(query_point_idx, chunk_count)| {
                chunk_count.assign_to(counts.slice_mut(s![query_point_idx]));
            })
    });

    counts
}

/// Run rNN count for all query points in a given voxel
fn _count_query_point_voxel(
    voxel: &(i32, i32, i32),
    query_points: &ArrayView2<f32>,
    query_points_by_voxel: &HashMap<(i32, i32, i32), Vec<i32>>,
    search_points: &Array2<f32>,
    search_points_by_voxel: &HashMap<(i32, i32, i32), Vec<i32>>,
    voxel_offsets: &Array2<i32>,
    max_dist: f32,
) -> Array1<u32> {
    // Extract search point neighbours for this voxel
    let (neighbours, _) = _get_neighbouring_search_points(
        voxel,
        search_points,
        search_points_by_voxel,
        voxel_offsets,
    );

    // Construct contiguous array for query points
    let query_point_indices = query_points_by_voxel.get(voxel).unwrap();
    let mut this_voxel_query_points = Array2::zeros([query_point_indices.len(), 3]);
    this_voxel_query_points
        .axis_iter_mut(Axis(0))
        .zip(query_point_indices.iter())
        .for_each(|(row, &idx)| {
            query_points.row(idx as usize).assign_to(row);
        });

    // Construct output arrays for this chunk
    let mut counts_chunk: Array1<u32> = Array1::from_elem([query_point_indices.len()], 0u32);

    // Map query point processing function across voxels of query points
    this_voxel_query_points
        .axis_iter(Axis(0))
        .zip(counts_chunk.iter_mut())
        .for_each(|(query_point, count)| {
            *count = _count_query_point_neighbours(query_point, &neighbours, max_dist);
        });

    counts_chunk
}

/// Count how many neighbours are within a given radius of a query point
///
/// This function is intended to be mapped (maybe in parallel) across rows of an
/// array of query points, zipped with rows from two mutable arrays for distances
/// and indices, which we will write to
///
/// Args:
///     query_point: The query point we are searching for neighbours of
///     search_points: Reference to search points array, for indexing and comparing
///         distances
///     max_dist: Maximum distance to search for neighbours within
fn _count_query_point_neighbours(
    query_point: ArrayView1<f32>,
    search_points: &Array2<f32>,
    max_dist: f32,
) -> u32 {
    // Construct an iterator of only the neighbours that are within range, and count its
    // length
    search_points
        .axis_iter(Axis(0))
        .map(|search_point| {
            let dx = query_point[0] - search_point[0];
            let dy = query_point[1] - search_point[1];
            let dz = query_point[2] - search_point[2];
            (dx * dx + dy * dy + dz * dz).sqrt()
        })
        .filter(|distance| *distance < max_dist)
        .collect::<Vec<_>>()
        .len() as u32
}

/// Generate voxel coordinates for each point (i.e. find which voxel each point
/// belongs to), and construct a hashmap of search point indices, indexed by voxel
/// coordinates
///
/// While we're here, we compute distances to the triangulation points
///
/// This is the second pass through the points we will make
fn _group_by_voxel(
    search_points: ArrayView2<f32>,
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
