//! Adaptive grid: the voxel grid, with a KD subtree inside every cell that holds more
//! than a threshold number of points
//!
//! The plain grid's cost per query is proportional to the number of points in the
//! neighbouring cells, which is what makes it slow on dense clusters. Here dense cells
//! are searched through a small KD-tree instead of linearly, while sparse cells keep
//! the cheap linear scan. Exact, like the two backends it combines.

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::kdtree::{Node, build_nodes, search_nodes, shift_node};
use super::voxel::VoxelGrid;
use super::{CandidateVisitor, NeighbourIndex, Point, distance_sq};

/// Marker in `subtree_roots` for a cell that is scanned linearly
const NO_SUBTREE: u32 = u32::MAX;

/// A dense cell's index, its points permuted into KD order, and its subtree's nodes
type BuiltSubtree = (usize, Vec<(Point, u32)>, Vec<Node>);

#[derive(Serialize, Deserialize)]
pub struct HybridGrid {
    grid: VoxelGrid,
    /// Per cell: root node index into `nodes`, or `NO_SUBTREE`
    subtree_roots: Vec<u32>,
    /// KD nodes of all cell subtrees, spliced end to end; leaves index `grid.points`
    nodes: Vec<Node>,
}

impl HybridGrid {
    /// Build the grid, then a KD subtree over every cell with more than
    /// `subtree_threshold` points
    ///
    /// Args:
    ///     search_points: Points to index (N, 3)
    ///     radius: Search radius the index will be queried with
    ///     cells_per_radius: How many grid cells span one radius (see `VoxelGrid`)
    ///     subtree_threshold: Cells with more points than this get a KD subtree
    pub fn new(
        search_points: ndarray::ArrayView2<f32>,
        radius: f32,
        cells_per_radius: u32,
        subtree_threshold: usize,
    ) -> Self {
        let mut grid = VoxelGrid::new(search_points, radius, cells_per_radius);
        let ranges: Vec<(usize, usize)> =
            (0..grid.num_cells()).map(|c| grid.cell_range(c)).collect();
        let dense_cells: Vec<usize> = (0..ranges.len())
            .filter(|&c| ranges[c].1 - ranges[c].0 > subtree_threshold.max(1))
            .collect();

        // Build each dense cell's subtree on a private copy of its points (in
        // parallel), then write the permuted copies back into the grid's storage
        let (points, original_indices) = grid.points_mut();
        let built: Vec<BuiltSubtree> = {
            let points: &[Point] = points;
            let original_indices: &[u32] = original_indices;
            dense_cells
                .par_iter()
                .map(|&cell| {
                    let (start, end) = ranges[cell];
                    let mut items: Vec<(Point, u32)> = points[start..end]
                        .iter()
                        .copied()
                        .zip(original_indices[start..end].iter().copied())
                        .collect();
                    let nodes = build_nodes(&mut items, start);
                    (cell, items, nodes)
                })
                .collect()
        };

        let mut subtree_roots = vec![NO_SUBTREE; ranges.len()];
        let mut nodes: Vec<Node> = Vec::new();
        for (cell, items, cell_nodes) in built {
            let (start, _) = ranges[cell];
            for (offset, (point, original)) in items.into_iter().enumerate() {
                points[start + offset] = point;
                original_indices[start + offset] = original;
            }
            let base = nodes.len() as u32;
            subtree_roots[cell] = base;
            nodes.extend(cell_nodes.into_iter().map(|node| shift_node(node, base)));
        }

        HybridGrid {
            grid,
            subtree_roots,
            nodes,
        }
    }

    /// The underlying grid (for statistics and the original-index mapping)
    pub fn grid(&self) -> &VoxelGrid {
        &self.grid
    }

    /// Number of cells that got a KD subtree
    pub fn num_subtrees(&self) -> usize {
        self.subtree_roots
            .iter()
            .filter(|&&r| r != NO_SUBTREE)
            .count()
    }
}

impl NeighbourIndex for HybridGrid {
    type Scratch = ();

    fn search<V: CandidateVisitor>(&self, query: &Point, visitor: &mut V, _scratch: &mut ()) {
        let grid = &self.grid;
        let query_cell = grid.cell_of(query);
        for offset in &grid.neighbour_offsets {
            let cell = [
                query_cell[0] + offset[0],
                query_cell[1] + offset[1],
                query_cell[2] + offset[2],
            ];
            if grid.cell_min_distance_sq(query, &cell) >= visitor.bound_sq() {
                continue;
            }
            let Some(&cell_idx) = grid.cell_lookup.get(&cell) else {
                continue;
            };

            let root = self.subtree_roots[cell_idx as usize];
            if root == NO_SUBTREE {
                let (start, end) = grid.cell_range(cell_idx as usize);
                for (position, point) in grid.points[start..end].iter().enumerate() {
                    visitor.visit(distance_sq(query, point), (start + position) as u32);
                }
            } else {
                search_nodes(&self.nodes, root, &grid.points, query, visitor);
            }
            if visitor.done() {
                return;
            }
        }
    }

    fn len(&self) -> usize {
        self.grid.points.len()
    }
}
