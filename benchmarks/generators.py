"""
Synthetic pointcloud generators for the oxvox neighbour-search benchmark

Every generator takes a point count and returns a `GeneratedCloud`: a C-contiguous
`(N, 3)` float32 array plus a one-line description. All of them are seeded, so a
benchmark run is reproducible, and all of them fill roughly the same bounding box
(`extent` metres on a side) so that one search radius means a comparable density
target across the whole family.

The families are chosen to span the ways real pointclouds break a spatial index:
uniform (the easy case every structure handles), Gaussian clusters (dense pockets
separated by empty space, the voxel grid's classic weak spot), a cylinder shell with
axial density falloff (a laser scan of a pipe: a 2D manifold in 3D with a strong
density gradient), a planar sheet (a 2D manifold whose points are effectively coplanar,
which flattens KD-tree splits along one axis), and a mixed scene (all of the above at
once, so no single density describes the cloud).
"""

from typing import Callable, TypedDict

import numpy as np
import numpy.typing as npt

# Every generator fills a box of this side length, in metres, so a given search radius
# means the same density target regardless of which generator produced the cloud
DEFAULT_EXTENT = 100.0


class GeneratedCloud(TypedDict):
    """
    A generated pointcloud and a description of how it was made
    """

    points: npt.NDArray[np.float32]
    description: str


def uniform_box(num_points: int, extent: float = DEFAULT_EXTENT, seed: int = 0) -> GeneratedCloud:
    """
    Points drawn uniformly from a cube

    Args:
        num_points: How many points to generate
        extent: Side length of the cube
        seed: Seed for the random generator

    Returns:
        The generated cloud, with density `num_points / extent ** 3` everywhere
    """
    rng = np.random.default_rng(seed=seed)
    points = rng.random(size=(num_points, 3), dtype=np.float32) * np.float32(extent)
    return GeneratedCloud(
        points=np.ascontiguousarray(points),
        description=f"uniform cube of side {extent:g} m",
    )


def gaussian_clusters(
    num_points: int,
    extent: float = DEFAULT_EXTENT,
    num_clusters: int = 24,
    spread: float = 0.6,
    seed: int = 1,
) -> GeneratedCloud:
    """
    Isotropic Gaussian blobs with uniformly scattered centres

    The blobs are small compared with the box, so the cloud is mostly empty space with
    a handful of very dense pockets: locally the density is orders of magnitude above
    the box average, which is the regime a uniform grid has to survive

    Args:
        num_points: How many points to generate
        extent: Side length of the box the cluster centres are scattered in
        num_clusters: How many blobs to generate
        spread: Standard deviation of each blob, in metres
        seed: Seed for the random generator

    Returns:
        The generated cloud
    """
    rng = np.random.default_rng(seed=seed)
    centres = rng.random(size=(num_clusters, 3), dtype=np.float32) * np.float32(extent)

    # Spread the points unevenly over the clusters (a few fat blobs, many thin ones) so
    # that no single "points per cluster" number describes the cloud
    weights = rng.random(size=num_clusters) + 0.2
    counts = np.diff(
        np.round(np.concatenate([[0.0], np.cumsum(weights) / weights.sum()]) * num_points)
    ).astype(np.intp)

    points = np.repeat(centres, repeats=counts, axis=0)
    points += rng.normal(loc=0.0, scale=spread, size=points.shape).astype(np.float32)
    return GeneratedCloud(
        points=np.ascontiguousarray(points, dtype=np.float32),
        description=(
            f"{num_clusters} Gaussian clusters (sigma {spread:g} m) scattered in a "
            f"{extent:g} m box"
        ),
    )


def cylinder_shell(
    num_points: int,
    extent: float = DEFAULT_EXTENT,
    shell_radius: float = 6.0,
    noise: float = 0.02,
    falloff_length: float = 25.0,
    seed: int = 2,
) -> GeneratedCloud:
    """
    Points on the surface of a long cylinder, with density falling off along its axis

    This imitates a laser scan of a pipe or tank wall from one end: the points lie on a
    2D manifold embedded in 3D (so the local dimensionality is 2, not 3) and their
    along-axis spacing grows with range, giving a density gradient of one to two orders
    of magnitude across the same cloud

    Args:
        num_points: How many points to generate
        extent: Length of the cylinder along its axis
        shell_radius: Radius of the cylinder
        noise: Standard deviation of the radial measurement noise, in metres
        falloff_length: Distance over which the sampling density falls by 1/e
        seed: Seed for the random generator

    Returns:
        The generated cloud
    """
    rng = np.random.default_rng(seed=seed)

    # Exponentially decaying density along the axis, by inverse-CDF sampling of a
    # truncated exponential so the points stay inside [0, extent]
    uniform_samples = rng.random(size=num_points)
    axial = -falloff_length * np.log1p(-uniform_samples * (1.0 - np.exp(-extent / falloff_length)))

    angle = rng.random(size=num_points) * (2.0 * np.pi)
    radial = shell_radius + rng.normal(loc=0.0, scale=noise, size=num_points)
    points = np.stack(
        [radial * np.cos(angle), radial * np.sin(angle), axial],
        axis=1,
    )
    return GeneratedCloud(
        points=np.ascontiguousarray(points, dtype=np.float32),
        description=(
            f"cylinder shell of radius {shell_radius:g} m and length {extent:g} m, "
            f"density falling off with 1/e length {falloff_length:g} m"
        ),
    )


def planar_sheet(
    num_points: int,
    extent: float = DEFAULT_EXTENT,
    thickness: float = 0.03,
    undulation: float = 1.5,
    seed: int = 3,
) -> GeneratedCloud:
    """
    A gently undulating near-planar sheet with measurement noise

    All the points sit within a few centimetres of a smooth surface, so the cloud is
    effectively 2D: a search sphere sweeps an area rather than a volume, and splits
    along the surface normal separate almost nothing

    Args:
        num_points: How many points to generate
        extent: Side length of the sheet
        thickness: Standard deviation of the out-of-plane noise, in metres
        undulation: Amplitude of the smooth surface undulation, in metres
        seed: Seed for the random generator

    Returns:
        The generated cloud
    """
    rng = np.random.default_rng(seed=seed)
    horizontal = rng.random(size=(num_points, 2)) * extent
    height = (
        undulation
        * np.sin(horizontal[:, 0] * (2.0 * np.pi / extent) * 3.0)
        * np.cos(horizontal[:, 1] * (2.0 * np.pi / extent) * 2.0)
        + rng.normal(loc=0.0, scale=thickness, size=num_points)
    )
    points = np.stack([horizontal[:, 0], horizontal[:, 1], height], axis=1)
    return GeneratedCloud(
        points=np.ascontiguousarray(points, dtype=np.float32),
        description=(
            f"undulating sheet over a {extent:g} m square, {thickness * 100:g} cm of "
            "out-of-plane noise"
        ),
    )


def scene(num_points: int, extent: float = DEFAULT_EXTENT, seed: int = 4) -> GeneratedCloud:
    """
    A mixed scene: ground sheet, two pipes, one very dense blob and a sparse haze

    No single density describes this cloud, which is the point: an index that tunes
    itself to the average density has to cope with a region a thousand times denser
    than the average and a region a hundred times sparser, in the same query batch

    Args:
        num_points: How many points to generate
        extent: Side length of the scene
        seed: Seed for the random generator

    Returns:
        The generated cloud
    """
    rng = np.random.default_rng(seed=seed)

    # Fractions of the budget given to each primitive, in the order they are built
    ground_count = int(num_points * 0.4)
    pipe_count = int(num_points * 0.1)
    blob_count = int(num_points * 0.3)
    haze_count = num_points - ground_count - 2 * pipe_count - blob_count

    ground = planar_sheet(
        ground_count, extent=extent, thickness=0.02, undulation=0.4, seed=seed + 1
    )["points"]

    # Two pipes lying across the scene at different heights, rotated into place by
    # swapping axes rather than a full rotation matrix (the axis-aligned case is the
    # harder one for a KD-tree, so it is the honest choice here)
    first_pipe = cylinder_shell(
        pipe_count, extent=extent, shell_radius=0.6, falloff_length=extent, seed=seed + 2
    )["points"][:, [2, 0, 1]] + np.array([0.0, extent * 0.25, 3.0], dtype=np.float32)
    second_pipe = cylinder_shell(
        pipe_count, extent=extent, shell_radius=1.4, falloff_length=extent * 0.3, seed=seed + 3
    )["points"][:, [0, 2, 1]] + np.array([extent * 0.7, 0.0, 6.0], dtype=np.float32)

    # One pocket of extreme density: a third of the points inside a 2 m box
    blob_centre = np.array([extent * 0.35, extent * 0.6, 1.5], dtype=np.float32)
    blob = blob_centre + rng.normal(loc=0.0, scale=0.35, size=(blob_count, 3)).astype(np.float32)

    # A thin haze filling the whole volume, so queries in "empty" space still have work
    haze = rng.random(size=(haze_count, 3), dtype=np.float32) * np.float32(
        [extent, extent, extent * 0.2]
    )

    points = np.concatenate([ground, first_pipe, second_pipe, blob, haze], axis=0)
    return GeneratedCloud(
        points=np.ascontiguousarray(points, dtype=np.float32),
        description=(
            f"mixed scene in a {extent:g} m box: ground sheet (40%), two pipes (20%), "
            "one dense blob (30%) and a sparse volumetric haze (10%)"
        ),
    )


# Every synthetic dataset the benchmark can run, by the label used in the results JSON
GENERATORS: dict[str, Callable[[int], GeneratedCloud]] = {
    "uniform": uniform_box,
    "clusters": gaussian_clusters,
    "cylinder": cylinder_shell,
    "sheet": planar_sheet,
    "scene": scene,
}


def generate(name: str, num_points: int) -> GeneratedCloud:
    """
    Generate one of the named synthetic clouds

    Args:
        name: Key into `GENERATORS`
        num_points: How many points to generate

    Returns:
        The generated cloud
    """
    if name not in GENERATORS:
        raise KeyError(f"unknown generator {name!r}; expected one of {sorted(GENERATORS)}")
    return GENERATORS[name](num_points)


def uniform_radius_for_density(
    num_points: int, points_per_sphere: float, extent: float = DEFAULT_EXTENT
) -> float:
    """
    Search radius whose sphere holds `points_per_sphere` points on the uniform cloud

    The same radius is then used for every dataset at that point count, so the density
    target labels a column of the grid rather than a property of one cloud; the harness
    records each cloud's realised mean neighbour count alongside the timings

    Args:
        num_points: Number of points in the cube
        points_per_sphere: Desired expected number of points inside a search sphere
        extent: Side length of the cube

    Returns:
        The search radius, in metres
    """
    volume_per_point = extent**3 / num_points
    return float((points_per_sphere * volume_per_point * 3.0 / (4.0 * np.pi)) ** (1.0 / 3.0))
