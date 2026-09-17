//! Compact KD-tree with bucketed leaves
//!
//! Points are permuted in place during the build so that every leaf owns a contiguous
//! slice, and the tree is stored as a flat vector of nodes. Splits are at the median of
//! the widest axis of each node's bounding box, so the depth is always about
//! `log2(N / LEAF_SIZE)`.

use ndarray::ArrayView2;
use serde::{Deserialize, Serialize};

use super::{CandidateVisitor, NeighbourIndex, Point, distance_sq, to_points};

/// Maximum number of points in a leaf
pub(super) const LEAF_SIZE: usize = 16;

/// Subtrees with at least this many points are built on separate rayon tasks
const PARALLEL_BUILD_THRESHOLD: usize = 1 << 14;

/// Marker in `Node::split_dim` for a leaf
const LEAF: u8 = 3;

/// Upper bound on tree depth, and hence on the traversal stack. Median splits give a
/// depth of about log2(N / LEAF_SIZE), so this covers any N that fits in memory
const MAX_DEPTH: usize = 64;

#[derive(Serialize, Deserialize, Clone, Copy)]
pub(super) struct Node {
    /// Split coordinate (unused for leaves)
    split_value: f32,
    /// Axis split on, or `LEAF`
    split_dim: u8,
    /// Left child node index, or for a leaf the start of its point range
    first: u32,
    /// Right child node index, or for a leaf the end of its point range
    second: u32,
}

#[derive(Serialize, Deserialize)]
pub struct KdTree {
    /// Search points, permuted so each leaf's points are contiguous
    points: Vec<Point>,
    /// Original index of each stored point
    original_indices: Vec<u32>,
    /// Flat node storage; index 0 is the root
    nodes: Vec<Node>,
}

impl KdTree {
    /// Build the tree
    ///
    /// Args:
    ///     search_points: Points to index (N, 3)
    pub fn new(search_points: ArrayView2<f32>) -> Self {
        let mut items: Vec<(Point, u32)> = to_points(search_points)
            .into_iter()
            .enumerate()
            .map(|(idx, point)| (point, idx as u32))
            .collect();

        let nodes = if items.is_empty() {
            Vec::new()
        } else {
            build_nodes(&mut items, 0)
        };

        let (points, original_indices) = items.into_iter().unzip();
        KdTree {
            points,
            original_indices,
            nodes,
        }
    }

    /// Mapping from stored point position to original search point index
    pub fn original_indices(&self) -> &[u32] {
        &self.original_indices
    }

    /// Number of nodes (internal and leaf)
    #[cfg(test)]
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
}

impl NeighbourIndex for KdTree {
    type Scratch = ();

    fn search<V: CandidateVisitor>(&self, query: &Point, visitor: &mut V, _scratch: &mut ()) {
        if !self.nodes.is_empty() {
            search_nodes(&self.nodes, 0, &self.points, query, visitor);
        }
    }

    fn len(&self) -> usize {
        self.points.len()
    }
}

/// Search the subtree rooted at `nodes[root]`, whose leaves index into `points`
///
/// Depth-first traversal, nearest child first, with the far child parked on an explicit
/// stack along with a lower bound on its distance from the query. Shared with backends
/// that embed KD subtrees inside another structure
pub(super) fn search_nodes<V: CandidateVisitor>(
    nodes: &[Node],
    root: u32,
    points: &[Point],
    query: &Point,
    visitor: &mut V,
) {
    let mut stack = [(root, 0f32); MAX_DEPTH];
    let mut depth = 1;
    while depth > 0 {
        depth -= 1;
        let (node_idx, min_distance_sq) = stack[depth];
        if min_distance_sq >= visitor.bound_sq() {
            continue;
        }

        let node = nodes[node_idx as usize];
        if node.split_dim == LEAF {
            let start = node.first as usize;
            let end = node.second as usize;
            for (position, point) in points[start..end].iter().enumerate() {
                visitor.visit(distance_sq(query, point), (start + position) as u32);
            }
            if visitor.done() {
                return;
            }
        } else {
            let diff = query[node.split_dim as usize] - node.split_value;
            let (near, far) = if diff < 0.0 {
                (node.first, node.second)
            } else {
                (node.second, node.first)
            };
            stack[depth] = (far, min_distance_sq.max(diff * diff));
            stack[depth + 1] = (near, min_distance_sq);
            depth += 2;
        }
    }
}

/// Recursively build the subtree over `items`, which occupy positions
/// `offset..offset + items.len()` of the final point array
///
/// Returns the subtree's nodes with child indices relative to the returned vector
/// (the root at 0), so subtrees can be built independently and spliced together (see
/// `shift_node`). Leaves hold absolute point ranges, so `offset` must be where `items`
/// sits in the final point array
pub(super) fn build_nodes(items: &mut [(Point, u32)], offset: usize) -> Vec<Node> {
    if items.len() <= LEAF_SIZE {
        return vec![Node {
            split_value: 0.0,
            split_dim: LEAF,
            first: offset as u32,
            second: (offset + items.len()) as u32,
        }];
    }

    // Split the widest axis of the bounding box at its median
    let split_dim = _widest_axis(items);
    let mid = items.len() / 2;
    items.select_nth_unstable_by(mid, |a, b| {
        a.0[split_dim]
            .partial_cmp(&b.0[split_dim])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let split_value = items[mid].0[split_dim];

    let parallel = items.len() >= PARALLEL_BUILD_THRESHOLD;
    let (left_items, right_items) = items.split_at_mut(mid);
    let (left_nodes, right_nodes) = if parallel {
        rayon::join(
            || build_nodes(left_items, offset),
            || build_nodes(right_items, offset + mid),
        )
    } else {
        (
            build_nodes(left_items, offset),
            build_nodes(right_items, offset + mid),
        )
    };

    // Splice: root, then the left subtree, then the right subtree, shifting each
    // subtree's internal child indices by where it lands in this vector
    let left_base = 1u32;
    let right_base = left_base + left_nodes.len() as u32;
    let mut nodes = Vec::with_capacity(1 + left_nodes.len() + right_nodes.len());
    nodes.push(Node {
        split_value,
        split_dim: split_dim as u8,
        first: left_base,
        second: right_base,
    });
    nodes.extend(
        left_nodes
            .into_iter()
            .map(|node| shift_node(node, left_base)),
    );
    nodes.extend(
        right_nodes
            .into_iter()
            .map(|node| shift_node(node, right_base)),
    );
    nodes
}

/// Shift an internal node's child indices by `base`; leaves hold point ranges and are
/// left alone
#[inline(always)]
pub(super) fn shift_node(node: Node, base: u32) -> Node {
    if node.split_dim == LEAF {
        node
    } else {
        Node {
            first: node.first + base,
            second: node.second + base,
            ..node
        }
    }
}

/// Axis along which the points' bounding box is widest
fn _widest_axis(items: &[(Point, u32)]) -> usize {
    let mut low = [f32::INFINITY; 3];
    let mut high = [f32::NEG_INFINITY; 3];
    for (point, _) in items {
        for axis in 0..3 {
            low[axis] = low[axis].min(point[axis]);
            high[axis] = high[axis].max(point[axis]);
        }
    }
    let mut widest = 0;
    for axis in 1..3 {
        if high[axis] - low[axis] > high[widest] - low[widest] {
            widest = axis;
        }
    }
    widest
}
