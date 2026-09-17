//! Approximate search over a k-nearest-neighbour graph
//!
//! Every search point is linked to its `degree` nearest neighbours (within the search
//! radius). A query first finds its single nearest search point exactly, through the
//! adaptive grid, then floods outwards along graph edges in best-first order, stopping
//! when the nearest unexpanded candidate is already beyond what the visitor still wants.
//!
//! This is the graph-based approximate nearest neighbour family (HNSW, NSG, ...) in its
//! simplest form. It is approximate: a point can only be found if a chain of graph edges
//! leads to it from the entry point through candidates that were still within bound
//! when they were expanded. Recall is measured in the benchmarks rather than promised.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::hybrid::HybridGrid;
use super::voxel::VoxelGrid;
use super::{CandidateVisitor, KnnCollector, NeighbourIndex, Point, distance_sq};

/// Padding in `adjacency` for points with fewer than `degree` neighbours in range
const NO_NEIGHBOUR: u32 = u32::MAX;

/// Points per parallel task when building the graph
const BUILD_CHUNK_SIZE: usize = 512;

#[derive(Serialize, Deserialize)]
pub struct KnnGraph {
    /// Exact index used to find each query's entry point (and to build the graph)
    entry_index: HybridGrid,
    degree: usize,
    /// `degree` neighbour positions per stored point, nearest first, padded with
    /// `NO_NEIGHBOUR`; positions index `entry_index.grid().points`
    adjacency: Vec<u32>,
}

/// A frontier entry ordered so that `BinaryHeap` pops the nearest candidate first
#[derive(PartialEq)]
struct Candidate {
    dist_sq: f32,
    position: u32,
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other.dist_sq.total_cmp(&self.dist_sq)
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Per-thread working memory for graph searches: generation-stamped visited marks (so
/// clearing between queries is O(1)) and the best-first frontier
#[derive(Default)]
pub struct GraphScratch {
    stamps: Vec<u32>,
    generation: u32,
    frontier: BinaryHeap<Candidate>,
}

impl GraphScratch {
    /// Start a new query over `num_points` points
    fn begin(&mut self, num_points: usize) {
        if self.stamps.len() != num_points {
            self.stamps = vec![0; num_points];
            self.generation = 0;
        }
        if self.generation == u32::MAX {
            self.stamps.fill(0);
            self.generation = 0;
        }
        self.generation += 1;
        self.frontier.clear();
    }

    /// Mark a point visited; returns false if it already was
    #[inline(always)]
    fn mark(&mut self, position: u32) -> bool {
        let stamp = &mut self.stamps[position as usize];
        if *stamp == self.generation {
            false
        } else {
            *stamp = self.generation;
            true
        }
    }
}

impl KnnGraph {
    /// Build the entry index, then link every point to its nearest neighbours
    ///
    /// Args:
    ///     search_points: Points to index (N, 3)
    ///     radius: Search radius; graph edges never span more than this
    ///     cells_per_radius, subtree_threshold: Passed to the adaptive grid
    ///     degree: Number of neighbours each point links to
    pub fn new(
        search_points: ndarray::ArrayView2<f32>,
        radius: f32,
        cells_per_radius: u32,
        subtree_threshold: usize,
        degree: usize,
    ) -> Self {
        let entry_index =
            HybridGrid::new(search_points, radius, cells_per_radius, subtree_threshold);
        let degree = degree.max(1);
        let points = &entry_index.grid().points;
        let mut adjacency = vec![NO_NEIGHBOUR; points.len() * degree];

        // Self-query every stored point for its degree nearest neighbours (plus one, to
        // make room for the point itself, which is then dropped)
        adjacency
            .par_chunks_mut(BUILD_CHUNK_SIZE * degree)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let mut collector = KnnCollector::new(degree + 1, radius, 0.0);
                for (row_idx, row) in chunk.chunks_mut(degree).enumerate() {
                    let position = (chunk_idx * BUILD_CHUNK_SIZE + row_idx) as u32;
                    collector.reset();
                    entry_index.search(&points[position as usize], &mut collector, &mut ());
                    let mut column = 0;
                    for &(_, neighbour) in collector.sorted() {
                        if neighbour != position && column < degree {
                            row[column] = neighbour;
                            column += 1;
                        }
                    }
                }
            });

        KnnGraph {
            entry_index,
            degree,
            adjacency,
        }
    }

    /// The underlying grid (for statistics and the original-index mapping)
    pub fn grid(&self) -> &VoxelGrid {
        self.entry_index.grid()
    }

    /// Number of neighbours each point links to
    pub fn degree(&self) -> usize {
        self.degree
    }
}

impl NeighbourIndex for KnnGraph {
    type Scratch = GraphScratch;

    fn search<V: CandidateVisitor>(
        &self,
        query: &Point,
        visitor: &mut V,
        scratch: &mut GraphScratch,
    ) {
        let points = &self.grid().points;
        if points.is_empty() {
            return;
        }

        // Exact entry point: the nearest search point within the visitor's bound. If
        // there is none, nothing within the bound exists at all
        let mut entry = KnnCollector::new(1, visitor.bound_sq().sqrt(), 0.0);
        self.entry_index.search(query, &mut entry, &mut ());
        let Some(&(entry_dist_sq, entry_position)) = entry.sorted().first() else {
            return;
        };

        // Best-first flood along graph edges
        scratch.begin(points.len());
        scratch.mark(entry_position);
        visitor.visit(entry_dist_sq, entry_position);
        scratch.frontier.push(Candidate {
            dist_sq: entry_dist_sq,
            position: entry_position,
        });

        while let Some(candidate) = scratch.frontier.pop() {
            if candidate.dist_sq >= visitor.bound_sq() {
                break;
            }
            let row = candidate.position as usize * self.degree;
            for &neighbour in &self.adjacency[row..row + self.degree] {
                if neighbour == NO_NEIGHBOUR {
                    break;
                }
                if !scratch.mark(neighbour) {
                    continue;
                }
                let dist_sq = distance_sq(query, &points[neighbour as usize]);
                visitor.visit(dist_sq, neighbour);
                if dist_sq < visitor.bound_sq() {
                    scratch.frontier.push(Candidate {
                        dist_sq,
                        position: neighbour,
                    });
                }
            }
            if visitor.done() {
                break;
            }
        }
    }

    fn len(&self) -> usize {
        self.grid().points.len()
    }
}
