import numpy as np
from scipy.spatial import cKDTree


def get_mesh_stats(mesh):
    """Gets basic stats from a trimesh object."""
    return {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces)
    }


def sample_points_from_mesh(mesh, num_points=10000):
    """Samples a consistent number of points from the mesh surface."""
    return mesh.sample(num_points)


def compute_chamfer_distance(points1, points2):
    """
    Computes the (one-way) Chamfer distance from points1 to points2.
    For symmetric distance, call this twice and average.
    """
    # Create a k-d tree for efficient nearest neighbor search
    tree = cKDTree(points2)
    # Find the nearest neighbor in points2 for each point in points1
    distances, _ = tree.query(points1, k=1)
    # Return the mean squared distance
    return np.mean(distances ** 2)


def compute_symmetric_chamfer(points1, points2):
    """Computes the symmetric Chamfer distance."""
    chamfer_1_to_2 = compute_chamfer_distance(points1, points2)
    chamfer_2_to_1 = compute_chamfer_distance(points2, points1)
    return (chamfer_1_to_2 + chamfer_2_to_1) / 2


def compute_hausdorff_distance(points1, points2):
    """Computes the symmetric Hausdorff distance using cKDTree."""

    try:
        # Create k-d trees for both point sets
        tree1 = cKDTree(points1)
        tree2 = cKDTree(points2)

        # 1. Find the nearest neighbor in points2 for each point in points1
        distances_1_to_2, _ = tree2.query(points1, k=1)

        # 2. Find the nearest neighbor in points1 for each point in points2
        distances_2_to_1, _ = tree1.query(points2, k=1)

        # 3. Find the maximum of these minimum distances (directed Hausdorff)
        hausdorff_1_to_2 = np.max(distances_1_to_2)
        hausdorff_2_to_1 = np.max(distances_2_to_1)

        # 4. The symmetric Hausdorff distance is the max of the two directed distances
        hausdorff_dist = max(hausdorff_1_to_2, hausdorff_2_to_1)

    except Exception as e:
        print(f"  [Hausdorff Error: {e}]")
        hausdorff_dist = np.nan

    return hausdorff_dist
    # ---------------


def compute_all_metrics(original_mesh, decompressed_mesh, num_points=10000):
    """Computes all distortion metrics."""
    try:
        # Sample points for a fair comparison
        points_orig = sample_points_from_mesh(original_mesh, num_points)
        points_decomp = sample_points_from_mesh(decompressed_mesh, num_points)

        chamfer = compute_symmetric_chamfer(points_orig, points_decomp)
        hausdorff = compute_hausdorff_distance(points_orig, points_decomp)

        return {"chamfer": chamfer, "hausdorff": hausdorff}
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return {"chamfer": np.nan, "hausdorff": np.nan}