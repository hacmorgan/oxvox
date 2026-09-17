//! Exactness tests: every backend must agree with a brute-force reference

use ndarray::{Array2, ArrayView2};

use super::graph::KnnGraph;
use super::hybrid::HybridGrid;
use super::kdtree::KdTree;
use super::voxel::VoxelGrid;
use super::{NeighbourIndex, Point, count_neighbours, distance_sq, find_neighbours, to_points};

/// Tiny deterministic pseudo-random generator (xorshift), so tests need no extra crates
struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform in [low, high)
    fn uniform(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next_f32()
    }
}

fn uniform_cloud(rng: &mut Rng, n: usize, low: f32, high: f32) -> Array2<f32> {
    Array2::from_shape_fn((n, 3), |_| rng.uniform(low, high))
}

/// Several dense clusters straddling the origin planes, with negative coordinates
fn clustered_cloud(rng: &mut Rng, clusters: usize, per_cluster: usize) -> Array2<f32> {
    let mut points = Vec::with_capacity(clusters * per_cluster);
    for _ in 0..clusters {
        let centre = [
            rng.uniform(-1.0, 1.0),
            rng.uniform(-1.0, 1.0),
            rng.uniform(-1.0, 1.0),
        ];
        for _ in 0..per_cluster {
            for &coordinate in &centre {
                points.push(coordinate + rng.uniform(-0.05, 0.05));
            }
        }
    }
    Array2::from_shape_vec((clusters * per_cluster, 3), points).unwrap()
}

/// Brute-force neighbours within `radius`, nearest first, truncated to `k`
fn brute_force(search: &[Point], query: &Point, k: usize, radius: f32) -> Vec<(f32, u32)> {
    let mut found: Vec<(f32, u32)> = search
        .iter()
        .enumerate()
        .map(|(idx, point)| (distance_sq(query, point), idx as u32))
        .filter(|(dist_sq, _)| *dist_sq < radius * radius)
        .collect();
    found.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    found.truncate(k);
    found
}

/// Assert a backend reproduces the brute-force result for every query
fn assert_matches_brute_force<I: NeighbourIndex>(
    index: &I,
    original_indices: &[u32],
    search: ArrayView2<f32>,
    queries: ArrayView2<f32>,
    k: usize,
    radius: f32,
) {
    let search_points = to_points(search);
    let query_points = to_points(queries);
    let (indices, distances) = find_neighbours(
        index,
        Some(original_indices),
        queries,
        k,
        radius,
        0.0,
        false,
    );
    let counts = count_neighbours(index, queries, radius, None, false);
    let weighted = count_neighbours(index, queries, radius, Some(1.0), false);

    for (q, query) in query_points.iter().enumerate() {
        let expected = brute_force(&search_points, query, k, radius);
        let got: Vec<(i32, f32)> = (0..k)
            .map(|c| (indices[[q, c]], distances[[q, c]]))
            .take_while(|(idx, _)| *idx >= 0)
            .collect();
        assert_eq!(
            got.len(),
            expected.len(),
            "query {q}: found {} neighbours, expected {}",
            got.len(),
            expected.len()
        );
        // Padding after the found neighbours must be -1 all the way
        for c in got.len()..k {
            assert_eq!(indices[[q, c]], -1);
            assert_eq!(distances[[q, c]], -1.0);
        }
        for (position, ((idx, dist), (expected_dist_sq, _))) in
            got.iter().zip(expected.iter()).enumerate()
        {
            // Ties may be broken differently, so compare distances positionally and
            // check the reported index really is at the reported distance
            assert!(
                (dist - expected_dist_sq.sqrt()).abs() < 1e-5,
                "query {q} position {position}: distance {dist} vs expected {}",
                expected_dist_sq.sqrt()
            );
            let actual_dist = distance_sq(query, &search_points[*idx as usize]).sqrt();
            assert!((actual_dist - dist).abs() < 1e-5);
        }

        // Counts must match exactly, weighted sums closely
        let all_within = brute_force(&search_points, query, usize::MAX, radius);
        assert_eq!(counts[q], all_within.len() as f32, "query {q} count");
        let expected_weighted: f32 = all_within
            .iter()
            .map(|(dist_sq, _)| 1.0 - dist_sq.sqrt() / radius)
            .sum();
        assert!(
            (weighted[q] - expected_weighted).abs() < 1e-3 * (1.0 + expected_weighted),
            "query {q} weighted {} vs {expected_weighted}",
            weighted[q]
        );
    }
}

/// Run the brute-force comparison on every exact backend for a search/query pair
fn check_all_backends(search: ArrayView2<f32>, queries: ArrayView2<f32>, k: usize, radius: f32) {
    for cells_per_radius in [1, 2] {
        let grid = VoxelGrid::new(search, radius, cells_per_radius);
        assert_matches_brute_force(&grid, grid.original_indices(), search, queries, k, radius);
    }
    let tree = KdTree::new(search);
    assert_matches_brute_force(&tree, tree.original_indices(), search, queries, k, radius);
    // A low subtree threshold forces subtrees into most cells, so the KD path inside
    // the grid gets exercised even on small test clouds
    for subtree_threshold in [4, 64] {
        let hybrid = HybridGrid::new(search, radius, 1, subtree_threshold);
        assert_matches_brute_force(
            &hybrid,
            hybrid.grid().original_indices(),
            search,
            queries,
            k,
            radius,
        );
    }
}

#[test]
fn uniform_cloud_straddling_origin() {
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let search = uniform_cloud(&mut rng, 4000, -1.0, 1.0);
    let queries = uniform_cloud(&mut rng, 300, -1.1, 1.1);
    for k in [1, 8, 50] {
        check_all_backends(search.view(), queries.view(), k, 0.2);
    }
}

#[test]
fn clustered_cloud_with_self_queries() {
    let mut rng = Rng(42);
    let search = clustered_cloud(&mut rng, 6, 500);
    let queries = search.slice(ndarray::s![..;7, ..]);
    check_all_backends(search.view(), queries, 10, 0.03);
    check_all_backends(search.view(), queries, 3, 0.5);
}

#[test]
fn duplicate_points_and_k_larger_than_candidates() {
    let mut rng = Rng(7);
    let base = uniform_cloud(&mut rng, 200, 0.0, 1.0);
    let search = ndarray::concatenate(ndarray::Axis(0), &[base.view(), base.view()]).unwrap();
    let queries = uniform_cloud(&mut rng, 50, 0.0, 1.0);
    check_all_backends(search.view(), queries.view(), 500, 0.1);
}

#[test]
fn point_at_exactly_the_radius_is_excluded() {
    let search = ndarray::arr2(&[[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.5, 0.0]]);
    let queries = ndarray::arr2(&[[0.0f32, 0.0, 0.0]]);
    check_all_backends(search.view(), queries.view(), 5, 1.0);
}

#[test]
fn empty_search_points_find_nothing() {
    let search = Array2::<f32>::zeros((0, 3));
    let queries = ndarray::arr2(&[[0.0f32, 0.0, 0.0]]);
    check_all_backends(search.view(), queries.view(), 3, 1.0);
}

#[test]
fn epsilon_stops_early_with_k_valid_neighbours() {
    let mut rng = Rng(3);
    let search = uniform_cloud(&mut rng, 3000, 0.0, 1.0);
    let queries = uniform_cloud(&mut rng, 100, 0.0, 1.0);
    let (k, radius) = (5, 0.3);
    let search_points = to_points(search.view());

    let grid = VoxelGrid::new(search.view(), radius, 1);
    let tree = KdTree::new(search.view());
    let hybrid = HybridGrid::new(search.view(), radius, 1, 8);
    let results = [
        find_neighbours(
            &grid,
            Some(grid.original_indices()),
            queries.view(),
            k,
            radius,
            radius,
            false,
        ),
        find_neighbours(
            &tree,
            Some(tree.original_indices()),
            queries.view(),
            k,
            radius,
            radius,
            false,
        ),
        find_neighbours(
            &hybrid,
            Some(hybrid.grid().original_indices()),
            queries.view(),
            k,
            radius,
            radius,
            false,
        ),
    ];
    for (indices, distances) in results {
        for (q, query) in to_points(queries.view()).iter().enumerate() {
            let available = brute_force(&search_points, query, k, radius).len();
            for c in 0..k {
                let idx = indices[[q, c]];
                if c < available {
                    assert!(idx >= 0, "query {q}: expected {available} neighbours");
                    let dist = distance_sq(query, &search_points[idx as usize]).sqrt();
                    assert!(dist < radius);
                    assert!((dist - distances[[q, c]]).abs() < 1e-5);
                } else {
                    assert_eq!(idx, -1);
                }
            }
        }
    }
}

#[test]
fn voxel_grid_statistics() {
    let mut rng = Rng(11);
    let search = uniform_cloud(&mut rng, 1000, 0.0, 1.0);
    let grid = VoxelGrid::new(search.view(), 0.5, 1);
    assert_eq!(grid.len(), 1000);
    assert!(grid.num_cells() <= 8);
    assert!(grid.max_points_per_cell() >= grid.mean_points_per_cell() as usize);
}

#[test]
fn kdtree_has_expected_size() {
    let mut rng = Rng(5);
    let search = uniform_cloud(&mut rng, 10_000, 0.0, 1.0);
    let tree = KdTree::new(search.view());
    assert_eq!(tree.len(), 10_000);
    // Median splits with 16-point leaves: roughly 2 * N / 16 nodes
    assert!(tree.num_nodes() > 1000 && tree.num_nodes() < 3000);
}

#[test]
fn hybrid_builds_subtrees_only_in_dense_cells() {
    let mut rng = Rng(21);
    // Clusters ~0.1 wide in cells of 0.1: a few cells hold most of each cluster and
    // exceed the threshold, while the cells the clusters spill into stay sparse
    let search = clustered_cloud(&mut rng, 3, 400);
    let hybrid = HybridGrid::new(search.view(), 0.1, 1, 32);
    assert!(hybrid.num_subtrees() > 0);
    assert!(hybrid.num_subtrees() <= hybrid.grid().num_cells());
    assert_eq!(hybrid.len(), 1200);

    // A threshold nothing exceeds means a plain grid
    let plain = HybridGrid::new(search.view(), 0.1, 1, 10_000);
    assert_eq!(plain.num_subtrees(), 0);
}

/// Fraction of brute-force neighbours the graph backend recovers, while asserting the
/// hard guarantees it must satisfy regardless of recall
fn graph_recall(
    search: &Array2<f32>,
    queries: &Array2<f32>,
    k: usize,
    radius: f32,
    degree: usize,
) -> f64 {
    let graph = KnnGraph::new(search.view(), radius, 1, 64, degree);
    let search_points = to_points(search.view());
    let (indices, distances) = find_neighbours(
        &graph,
        Some(graph.grid().original_indices()),
        queries.view(),
        k,
        radius,
        0.0,
        false,
    );
    let mut hits = 0usize;
    let mut wanted = 0usize;
    for (q, query) in to_points(queries.view()).iter().enumerate() {
        let expected: std::collections::HashSet<u32> =
            brute_force(&search_points, query, k, radius)
                .into_iter()
                .map(|(_, i)| i)
                .collect();
        let mut seen = std::collections::HashSet::new();
        let mut padding_started = false;
        for c in 0..k {
            let idx = indices[[q, c]];
            if idx < 0 {
                padding_started = true;
                assert_eq!(distances[[q, c]], -1.0);
                continue;
            }
            assert!(!padding_started, "-1 padding must be trailing");
            assert!(seen.insert(idx), "duplicate neighbour returned");
            let dist = distance_sq(query, &search_points[idx as usize]).sqrt();
            assert!(dist < radius, "returned a point at or beyond the radius");
            assert!((dist - distances[[q, c]]).abs() < 1e-5);
            if c > 0 {
                assert!(
                    distances[[q, c]] >= distances[[q, c - 1]],
                    "results must be nearest-first"
                );
            }
            if expected.contains(&(idx as u32)) {
                hits += 1;
            }
        }
        wanted += expected.len();
    }
    hits as f64 / wanted.max(1) as f64
}

#[test]
fn graph_has_high_recall_on_uniform_data_for_small_k() {
    let mut rng = Rng(99);
    let search = uniform_cloud(&mut rng, 5000, 0.0, 1.0);
    let queries = uniform_cloud(&mut rng, 300, 0.0, 1.0);
    let recall = graph_recall(&search, &queries, 8, 0.15, 16);
    assert!(recall >= 0.99, "recall {recall}");
}

#[test]
fn graph_recall_degrades_gracefully_for_k_above_degree() {
    let mut rng = Rng(100);
    let search = uniform_cloud(&mut rng, 5000, 0.0, 1.0);
    let queries = uniform_cloud(&mut rng, 200, 0.0, 1.0);
    let recall = graph_recall(&search, &queries, 40, 0.3, 8);
    assert!(recall >= 0.8, "recall {recall}");
}

#[test]
fn graph_counts_never_exceed_the_exact_count() {
    let mut rng = Rng(101);
    let search = clustered_cloud(&mut rng, 4, 400);
    let queries = search.slice(ndarray::s![..;5, ..]);
    let radius = 0.04;
    let graph = KnnGraph::new(search.view(), radius, 1, 64, 12);
    let exact = KdTree::new(search.view());
    let approx_counts = count_neighbours(&graph, queries, radius, None, false);
    let exact_counts = count_neighbours(&exact, queries, radius, None, false);
    let mut total_approx = 0.0;
    let mut total_exact = 0.0;
    for (a, e) in approx_counts.iter().zip(exact_counts.iter()) {
        assert!(a <= e, "approximate count {a} exceeds exact {e}");
        total_approx += a;
        total_exact += e;
    }
    let recall = total_approx / total_exact;
    assert!(recall > 0.95, "count recall {recall}");
}
