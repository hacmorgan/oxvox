//! Uniform voxel grid index in compressed sparse row (CSR) layout
//!
//! Search points are bucketed into cubic cells of side `radius / cells_per_radius`
//! and stored sorted by cell, so every cell's points are one contiguous slice. A query
//! visits the `(2c + 1)^3` cells around its own cell, skipping any whose bounding box
//! is already farther away than the visitor's current bound.

use std::collections::HashMap;

use ndarray::ArrayView2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::{CandidateVisitor, NeighbourIndex, Point, distance_sq, to_points};

/// Integer cell coordinates
type CellCoords = [i32; 3];

/// Fast non-cryptographic hasher for cell lookups
type CellHasher = ahash::RandomState;

#[derive(Serialize, Deserialize)]
pub struct VoxelGrid {
    cell_size: f32,
    inv_cell_size: f32,
    /// Search points, permuted so that each cell's points are contiguous
    points: Vec<Point>,
    /// Original index of each stored point
    original_indices: Vec<u32>,
    /// Coordinates of each non-empty cell, in storage order
    cell_coords: Vec<CellCoords>,
    /// Start offset into `points` of each cell, plus a final end sentinel
    cell_starts: Vec<u32>,
    /// Cell coordinates -> position in `cell_coords` / `cell_starts`
    cell_lookup: HashMap<CellCoords, u32, CellHasher>,
    /// Relative coordinates of the cells that can hold a point within the radius,
    /// nearest first so the pruning bound tightens as early as possible
    neighbour_offsets: Vec<CellCoords>,
}

impl VoxelGrid {
    /// Build the grid
    ///
    /// Args:
    ///     search_points: Points to index (N, 3)
    ///     radius: Search radius the grid will be queried with
    ///     cells_per_radius: How many cells span one radius. 1 gives the classic
    ///         27-cell neighbourhood; larger values give smaller cells (less
    ///         over-scan per query) at the cost of more cell lookups
    pub fn new(search_points: ArrayView2<f32>, radius: f32, cells_per_radius: u32) -> Self {
        let cells_per_radius = cells_per_radius.max(1) as i32;
        let cell_size = radius / cells_per_radius as f32;
        let inv_cell_size = 1.0 / cell_size;
        let unsorted_points = to_points(search_points);

        // Tag every point with its cell, then sort by cell so each cell is contiguous
        let mut keyed: Vec<(CellCoords, u32)> = unsorted_points
            .par_iter()
            .enumerate()
            .map(|(idx, point)| (_cell_of(point, inv_cell_size), idx as u32))
            .collect();
        keyed.par_sort_unstable_by_key(|(cell, _)| *cell);

        // Walk the sorted points once, recording where each new cell begins
        let mut points = Vec::with_capacity(keyed.len());
        let mut original_indices = Vec::with_capacity(keyed.len());
        let mut cell_coords: Vec<CellCoords> = Vec::new();
        let mut cell_starts: Vec<u32> = Vec::new();
        for (position, &(cell, idx)) in keyed.iter().enumerate() {
            if cell_coords.last() != Some(&cell) {
                cell_coords.push(cell);
                cell_starts.push(position as u32);
            }
            points.push(unsorted_points[idx as usize]);
            original_indices.push(idx);
        }
        cell_starts.push(points.len() as u32);

        let cell_lookup: HashMap<CellCoords, u32, CellHasher> = cell_coords
            .iter()
            .enumerate()
            .map(|(cell_idx, &cell)| (cell, cell_idx as u32))
            .collect();

        VoxelGrid {
            cell_size,
            inv_cell_size,
            points,
            original_indices,
            cell_coords,
            cell_starts,
            cell_lookup,
            neighbour_offsets: _neighbour_offsets(cells_per_radius),
        }
    }

    /// Mapping from stored point position to original search point index
    pub fn original_indices(&self) -> &[u32] {
        &self.original_indices
    }

    /// Number of non-empty cells
    pub fn num_cells(&self) -> usize {
        self.cell_coords.len()
    }

    /// Most points in any one cell
    pub fn max_points_per_cell(&self) -> usize {
        self.cell_starts
            .windows(2)
            .map(|pair| (pair[1] - pair[0]) as usize)
            .max()
            .unwrap_or(0)
    }

    /// Mean points per non-empty cell
    pub fn mean_points_per_cell(&self) -> f32 {
        if self.cell_coords.is_empty() {
            0.0
        } else {
            self.points.len() as f32 / self.cell_coords.len() as f32
        }
    }

    /// Squared distance from `query` to the nearest point of the cell's bounding box
    #[inline(always)]
    fn _cell_min_distance_sq(&self, query: &Point, cell: &CellCoords) -> f32 {
        let mut total = 0.0;
        for axis in 0..3 {
            let low = cell[axis] as f32 * self.cell_size;
            let high = low + self.cell_size;
            let gap = if query[axis] < low {
                low - query[axis]
            } else if query[axis] > high {
                query[axis] - high
            } else {
                0.0
            };
            total += gap * gap;
        }
        total
    }
}

impl NeighbourIndex for VoxelGrid {
    fn search<V: CandidateVisitor>(&self, query: &Point, visitor: &mut V) {
        let query_cell = _cell_of(query, self.inv_cell_size);
        for offset in &self.neighbour_offsets {
            let cell = [
                query_cell[0] + offset[0],
                query_cell[1] + offset[1],
                query_cell[2] + offset[2],
            ];

            // Skip cells whose whole bounding box is already beyond the bound
            if self._cell_min_distance_sq(query, &cell) >= visitor.bound_sq() {
                continue;
            }

            if let Some(&cell_idx) = self.cell_lookup.get(&cell) {
                let start = self.cell_starts[cell_idx as usize] as usize;
                let end = self.cell_starts[cell_idx as usize + 1] as usize;
                for (position, point) in self.points[start..end].iter().enumerate() {
                    visitor.visit(distance_sq(query, point), (start + position) as u32);
                }
                if visitor.done() {
                    return;
                }
            }
        }
    }

    fn len(&self) -> usize {
        self.points.len()
    }
}

/// Cell containing a point, using floor so cells are uniform on both sides of zero
#[inline(always)]
fn _cell_of(point: &Point, inv_cell_size: f32) -> CellCoords {
    [
        (point[0] * inv_cell_size).floor() as i32,
        (point[1] * inv_cell_size).floor() as i32,
        (point[2] * inv_cell_size).floor() as i32,
    ]
}

/// Relative coordinates of every cell within `cells_per_radius` cells of the origin
/// cell, ordered by the closest they could possibly be to a point in the origin cell
fn _neighbour_offsets(cells_per_radius: i32) -> Vec<CellCoords> {
    let mut offsets: Vec<CellCoords> = Vec::new();
    for x in -cells_per_radius..=cells_per_radius {
        for y in -cells_per_radius..=cells_per_radius {
            for z in -cells_per_radius..=cells_per_radius {
                offsets.push([x, y, z]);
            }
        }
    }
    let min_gap_sq = |offset: &CellCoords| -> i32 {
        offset
            .iter()
            .map(|o| (o.abs() - 1).max(0))
            .map(|gap| gap * gap)
            .sum()
    };
    offsets.sort_by_key(min_gap_sq);
    offsets
}
