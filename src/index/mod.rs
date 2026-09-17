//! Spatial indices for radius-bounded nearest-neighbour search, and the query
//! machinery shared between them
//!
//! Each backend implements [`NeighbourIndex`], which only has to know how to visit the
//! search points near a single query point. Everything else (collecting the k best
//! candidates, counting, weighting, epsilon early-exit, parallelism over queries and
//! writing the output arrays) lives here, so that the backends stay small and every
//! backend gets identical semantics.

pub mod graph;
pub mod hybrid;
pub mod kdtree;
pub mod voxel;

#[cfg(feature = "kiddo-baseline")]
pub mod kiddo;

#[cfg(test)]
mod tests;

use indicatif::ProgressBar;
use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array2, ArrayView2, Axis};

/// A point in 3D
pub type Point = [f32; 3];

/// Number of query points each parallel task handles at a time. Large enough that the
/// per-task setup (allocating the candidate buffer) is amortised, small enough that
/// work stays balanced across threads even for a few thousand queries
const QUERY_CHUNK_SIZE: usize = 256;

/// Squared euclidean distance between two points
#[inline(always)]
pub fn distance_sq(a: &Point, b: &Point) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

/// Receives candidate search points from a backend's search around one query point
///
/// Backends call [`CandidateVisitor::bound_sq`] to learn how far away a candidate (or a
/// whole region of space) can be before it is irrelevant, so they can prune; they call
/// [`CandidateVisitor::done`] to find out whether the visitor already has everything
/// it needs and the search can stop early
pub trait CandidateVisitor {
    /// Squared distance beyond which candidates can no longer matter. Only candidates
    /// with `dist_sq < bound_sq()` are worth offering
    fn bound_sq(&self) -> f32;

    /// Offer a candidate at squared distance `dist_sq` with original index `idx`. The
    /// visitor is responsible for rejecting candidates at or beyond its bound
    fn visit(&mut self, dist_sq: f32, idx: u32);

    /// Whether the search may stop before all candidates have been offered
    fn done(&self) -> bool;

    /// How many nearest candidates the visitor wants, or None if it wants every
    /// candidate within its bound. Lets backends that run their own complete query
    /// (rather than streaming candidates) ask for the right thing; only the kiddo
    /// wrapper does today, hence the dead-code allowance in default builds
    #[allow(dead_code)]
    fn wanted(&self) -> Option<usize> {
        None
    }
}

/// A spatial index over a fixed set of search points
pub trait NeighbourIndex: Send + Sync {
    /// Per-thread working memory a search needs (visited marks, heaps, ...). `()` for
    /// backends that need none. The driver creates one per chunk of queries and reuses
    /// it, so backends never allocate per query
    type Scratch: Default + Send;

    /// Offer every search point that could lie within `visitor.bound_sq()` of `query`
    /// to the visitor (offering points beyond the bound is allowed, missing points
    /// within it is not, unless the backend is documented as approximate), stopping
    /// early once `visitor.done()`
    fn search<V: CandidateVisitor>(
        &self,
        query: &Point,
        visitor: &mut V,
        scratch: &mut Self::Scratch,
    );

    /// Number of search points in the index
    fn len(&self) -> usize;
}

/// Bounded collector of the k nearest candidates within a radius
///
/// Keeps the k smallest squared distances seen so far as a max-heap, so that the
/// current worst kept candidate is available in O(1) as a pruning bound and each new
/// candidate costs O(log k). Once `num_within_epsilon >= k` candidates closer than
/// `epsilon` have been kept, the search is declared done (this is the approximate
/// early-exit behaviour the `epsilon` argument has always had)
pub struct KnnCollector {
    k: usize,
    radius_sq: f32,
    epsilon_sq: f32,
    /// Max-heap on squared distance, stored as a plain Vec we sift manually
    heap: Vec<(f32, u32)>,
    num_within_epsilon: usize,
}

impl KnnCollector {
    pub fn new(k: usize, radius: f32, epsilon: f32) -> Self {
        KnnCollector {
            k,
            radius_sq: radius * radius,
            epsilon_sq: epsilon * epsilon,
            heap: Vec::with_capacity(k + 1),
            num_within_epsilon: 0,
        }
    }

    /// Reset for a new query point, keeping the allocation
    pub fn reset(&mut self) {
        self.heap.clear();
        self.num_within_epsilon = 0;
    }

    /// The kept candidates as `(squared distance, index)`, nearest first. Sorting
    /// destroys the heap order, so call `reset` before reusing the collector
    pub fn sorted(&mut self) -> &[(f32, u32)] {
        self.heap.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        &self.heap
    }

    /// Write the kept candidates, nearest first, into the output rows; pad with -1
    pub fn write_sorted(
        &mut self,
        indices_row: &mut [i32],
        distances_row: &mut [f32],
        original_indices: Option<&[u32]>,
    ) {
        let mut column = 0;
        for &(dist_sq, idx) in self.sorted() {
            let idx = match original_indices {
                Some(map) => map[idx as usize],
                None => idx,
            };
            indices_row[column] = idx as i32;
            distances_row[column] = dist_sq.sqrt();
            column += 1;
        }
        for column in column..indices_row.len() {
            indices_row[column] = -1;
            distances_row[column] = -1.0;
        }
    }

    #[inline(always)]
    fn sift_up(&mut self, mut pos: usize) {
        while pos > 0 {
            let parent = (pos - 1) / 2;
            if self.heap[parent].0 >= self.heap[pos].0 {
                break;
            }
            self.heap.swap(parent, pos);
            pos = parent;
        }
    }

    #[inline(always)]
    fn sift_down(&mut self, mut pos: usize) {
        let len = self.heap.len();
        loop {
            let left = 2 * pos + 1;
            let right = left + 1;
            let mut largest = pos;
            if left < len && self.heap[left].0 > self.heap[largest].0 {
                largest = left;
            }
            if right < len && self.heap[right].0 > self.heap[largest].0 {
                largest = right;
            }
            if largest == pos {
                break;
            }
            self.heap.swap(pos, largest);
            pos = largest;
        }
    }
}

impl CandidateVisitor for KnnCollector {
    #[inline(always)]
    fn bound_sq(&self) -> f32 {
        if self.heap.len() == self.k {
            // Once full, only candidates strictly closer than the current worst kept
            // one can displace it
            self.heap[0].0.min(self.radius_sq)
        } else {
            self.radius_sq
        }
    }

    #[inline(always)]
    fn visit(&mut self, dist_sq: f32, idx: u32) {
        if dist_sq >= self.bound_sq() || self.k == 0 {
            return;
        }
        if dist_sq < self.epsilon_sq {
            self.num_within_epsilon += 1;
        }
        if self.heap.len() < self.k {
            self.heap.push((dist_sq, idx));
            let last = self.heap.len() - 1;
            self.sift_up(last);
        } else {
            // Replace the current worst candidate (it is strictly farther than this one)
            self.heap[0] = (dist_sq, idx);
            self.sift_down(0);
        }
    }

    #[inline(always)]
    fn done(&self) -> bool {
        self.num_within_epsilon >= self.k
    }

    fn wanted(&self) -> Option<usize> {
        Some(self.k)
    }
}

/// Counts candidates within a radius, optionally weighting each by the normalised
/// kernel `(1 - distance / radius) ^ p`
pub struct CountCollector {
    radius_sq: f32,
    inv_radius: f32,
    weight: Option<f32>,
    total: f32,
}

impl CountCollector {
    pub fn new(radius: f32, weight: Option<f32>) -> Self {
        CountCollector {
            radius_sq: radius * radius,
            inv_radius: 1.0 / radius,
            weight,
            total: 0.0,
        }
    }

    pub fn reset(&mut self) {
        self.total = 0.0;
    }

    pub fn total(&self) -> f32 {
        self.total
    }
}

impl CandidateVisitor for CountCollector {
    #[inline(always)]
    fn bound_sq(&self) -> f32 {
        self.radius_sq
    }

    #[inline(always)]
    fn visit(&mut self, dist_sq: f32, _idx: u32) {
        if dist_sq >= self.radius_sq {
            return;
        }
        self.total += match self.weight {
            None => 1.0,
            Some(p) => {
                let falloff = 1.0 - dist_sq.sqrt() * self.inv_radius;
                if p == 1.0 { falloff } else { falloff.powf(p) }
            }
        };
    }

    #[inline(always)]
    fn done(&self) -> bool {
        false
    }
}

/// Convert an `(N, 3)` array view into a vector of points
pub fn to_points(points: ArrayView2<f32>) -> Vec<Point> {
    points
        .axis_iter(Axis(0))
        .map(|row| [row[0], row[1], row[2]])
        .collect()
}

/// Find the (up to) k nearest neighbours within `radius` of every query point
///
/// Runs on whatever rayon thread pool is installed by the caller
///
/// Args:
///     index: Spatial index over the search points
///     original_indices: If the index stores points in a permuted order, the mapping
///         from stored position to original search point index
///     query_points: Points to find the neighbours of (Q, 3)
///     num_neighbours: k
///     radius: Neighbours at or beyond this distance are ignored
///     epsilon: Once k neighbours closer than this have been found, stop searching
///     progress: Show a progress bar over query points
///
/// Returns:
///     Indices of neighbours (Q, k), -1 padded
///     Distances to neighbours (Q, k), -1 padded
pub fn find_neighbours<I: NeighbourIndex>(
    index: &I,
    original_indices: Option<&[u32]>,
    query_points: ArrayView2<f32>,
    num_neighbours: usize,
    radius: f32,
    epsilon: f32,
    progress: bool,
) -> (Array2<i32>, Array2<f32>) {
    let num_queries = query_points.shape()[0];
    let mut indices = Array2::<i32>::from_elem([num_queries, num_neighbours], -1);
    let mut distances = Array2::<f32>::from_elem([num_queries, num_neighbours], -1.0);
    let bar = _progress_bar(num_queries, progress);

    // Process queries in contiguous chunks so each task reuses one candidate buffer,
    // writing straight into its slice of the output arrays
    indices
        .axis_chunks_iter_mut(Axis(0), QUERY_CHUNK_SIZE)
        .into_par_iter()
        .zip(distances.axis_chunks_iter_mut(Axis(0), QUERY_CHUNK_SIZE))
        .zip(query_points.axis_chunks_iter(Axis(0), QUERY_CHUNK_SIZE))
        .for_each_init(
            // One collector and one scratch per rayon worker task, not per chunk: a
            // backend's scratch can be as large as the whole index (e.g. visited marks)
            || {
                (
                    KnnCollector::new(num_neighbours, radius, epsilon),
                    I::Scratch::default(),
                )
            },
            |(collector, scratch), ((mut indices_chunk, mut distances_chunk), queries_chunk)| {
                for ((mut indices_row, mut distances_row), query) in indices_chunk
                    .axis_iter_mut(Axis(0))
                    .zip(distances_chunk.axis_iter_mut(Axis(0)))
                    .zip(queries_chunk.axis_iter(Axis(0)))
                {
                    let query = [query[0], query[1], query[2]];
                    collector.reset();
                    index.search(&query, collector, scratch);
                    collector.write_sorted(
                        indices_row
                            .as_slice_mut()
                            .expect("output rows are contiguous"),
                        distances_row
                            .as_slice_mut()
                            .expect("output rows are contiguous"),
                        original_indices,
                    );
                }
                if let Some(bar) = &bar {
                    bar.inc(queries_chunk.shape()[0] as u64);
                }
            },
        );

    if let Some(bar) = bar {
        bar.finish_and_clear();
    }
    (indices, distances)
}

/// Count (optionally distance-weighted) neighbours within `radius` of every query point
///
/// Args:
///     index: Spatial index over the search points
///     query_points: Points to count the neighbours of (Q, 3)
///     radius: Neighbours at or beyond this distance are ignored
///     weight: If `Some(p)`, each neighbour contributes `(1 - d / radius) ^ p`
///         instead of 1
///     progress: Show a progress bar over query points
///
/// Returns:
///     Neighbour count or weighted sum per query point (Q,)
pub fn count_neighbours<I: NeighbourIndex>(
    index: &I,
    query_points: ArrayView2<f32>,
    radius: f32,
    weight: Option<f32>,
    progress: bool,
) -> Array1<f32> {
    let num_queries = query_points.shape()[0];
    let mut counts = Array1::<f32>::zeros(num_queries);
    let bar = _progress_bar(num_queries, progress);

    counts
        .axis_chunks_iter_mut(Axis(0), QUERY_CHUNK_SIZE)
        .into_par_iter()
        .zip(query_points.axis_chunks_iter(Axis(0), QUERY_CHUNK_SIZE))
        .for_each_init(
            || (CountCollector::new(radius, weight), I::Scratch::default()),
            |(collector, scratch), (mut counts_chunk, queries_chunk)| {
                for (count, query) in counts_chunk
                    .iter_mut()
                    .zip(queries_chunk.axis_iter(Axis(0)))
                {
                    let query = [query[0], query[1], query[2]];
                    collector.reset();
                    index.search(&query, collector, scratch);
                    *count = collector.total();
                }
                if let Some(bar) = &bar {
                    bar.inc(queries_chunk.shape()[0] as u64);
                }
            },
        );

    if let Some(bar) = bar {
        bar.finish_and_clear();
    }
    counts
}

/// Construct a progress bar over `total` items if requested
fn _progress_bar(total: usize, enabled: bool) -> Option<ProgressBar> {
    enabled.then(|| ProgressBar::new(total as u64))
}
