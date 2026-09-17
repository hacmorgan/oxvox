//! Wrapper around the `kiddo` crate's KD-tree, compiled in only with the
//! `kiddo-baseline` feature, as an external reference point for benchmarking the
//! hand-rolled backends against

use std::num::NonZero;

use kiddo::{ImmutableKdTree, SquaredEuclidean};
use ndarray::ArrayView2;
use serde::{Deserialize, Serialize};

use super::{CandidateVisitor, NeighbourIndex, Point, to_points};

/// Kiddo's tree over f32 3D points, items are the original point indices
type Tree = ImmutableKdTree<f32, 3>;

#[derive(Serialize, Deserialize)]
pub struct KiddoTree {
    tree: Tree,
    num_points: usize,
}

impl KiddoTree {
    /// Build the tree
    ///
    /// Args:
    ///     search_points: Points to index (N, 3)
    pub fn new(search_points: ArrayView2<f32>) -> Self {
        let points = to_points(search_points);
        let tree = Tree::new_from_slice(&points).expect("kiddo tree construction failed");
        KiddoTree {
            tree,
            num_points: points.len(),
        }
    }
}

impl NeighbourIndex for KiddoTree {
    fn search<V: CandidateVisitor>(&self, query: &Point, visitor: &mut V) {
        if self.num_points == 0 {
            return;
        }
        // kiddo runs its own complete query, so we ask it for exactly what the visitor
        // wants and then replay the results through the visitor. Squared euclidean
        // distances keep kiddo's radius in the same units as the visitor's bound
        let radius_sq = visitor.bound_sq();
        match visitor.wanted() {
            Some(k) => {
                let Some(k) = NonZero::new(k) else {
                    return;
                };
                let results = self
                    .tree
                    .query(query)
                    .nearest_n::<SquaredEuclidean<f32>>(k)
                    .within(radius_sq)
                    .execute();
                for item in results {
                    visitor.visit(item.distance, item.item);
                }
            }
            None => {
                let results = self
                    .tree
                    .query(query)
                    .within::<SquaredEuclidean<f32>>(radius_sq)
                    .unsorted()
                    .execute();
                for item in results {
                    visitor.visit(item.distance, item.item);
                }
            }
        }
    }

    fn len(&self) -> usize {
        self.num_points
    }
}
