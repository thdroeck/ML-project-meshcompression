# Centralize metric calculations for reuse in benchmarks

import trimesh
import numpy as np
from scipy.spatial import cKDTree

def get_mesh_stats(mesh):
    """Retrieves basic statistics from a trimesh object."""
    return {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces)
    }

def sample_points_from_mesh(mesh, num_points=10000):
    """Samples a consistent number of points from the mesh surface."""
    return mesh.sample(num_points)

def compute_chamfer_distance(points1, points2):
    """
    Computes the (one-sided) Chamfer distance from points1 to points2.
    For symmetric distance, call this function twice and take the average.
    """
    tree = cKDTree(points2)
    distances, _ = tree.query(points1, k=1)
    return np.mean(distances**2)

def compute_symmetric_chamfer(points1, points2):
    """Computes the symmetric Chamfer distance."""
    chamfer_1_to_2 = compute_chamfer_distance(points1, points2)
    chamfer_2_to_1 = compute_chamfer_distance(points2, points1)
    return (chamfer_1_to_2 + chamfer_2_to_1) / 2

def compute_hausdorff_distance(points1, points2):
    """Computes the Hausdorff distance with trimesh."""
    # trimesh.proximity.directed_hausdorff requires a pointcloud object
    pc1 = trimesh.PointCloud(points1)
    pc2 = trimesh.PointCloud(points2)
    
    hausdorff_1_to_2, _ = trimesh.proximity.directed_hausdorff(pc1, pc2)
    hausdorff_2_to_1, _ = trimesh.proximity.directed_hausdorff(pc2, pc1)
    
    return max(hausdorff_1_to_2, hausdorff_2_to_1)

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