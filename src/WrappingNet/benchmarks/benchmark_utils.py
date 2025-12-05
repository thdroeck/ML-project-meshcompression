import numpy as np
import trimesh
from scipy.spatial import cKDTree

def get_mesh_stats(mesh):
    """Gets basic stats from a trimesh object."""
    return {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces)
    }

def sample_points_and_normals(mesh, num_points=10000):
    """
    Samples points from the mesh and retrieves the face normals at those locations.
    """
    # trimesh.sample.sample_surface returns (points, face_indices)
    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
    
    # We use the face normal for the sampled point. 
    # (For smoother results on smooth meshes, one could interpolate vertex normals, 
    # but face normals are the standard robust metric for raw geometry).
    normals = mesh.face_normals[face_indices]
    
    return points, normals

def compute_chamfer_distance(points1, points2):
    """
    Computes the (one-way) Chamfer distance from points1 to points2.
    """
    tree = cKDTree(points2)
    distances, _ = tree.query(points1, k=1)
    return np.mean(distances ** 2)

def compute_symmetric_chamfer(points1, points2):
    """Computes the symmetric Chamfer distance."""
    chamfer_1_to_2 = compute_chamfer_distance(points1, points2)
    chamfer_2_to_1 = compute_chamfer_distance(points2, points1)
    return (chamfer_1_to_2 + chamfer_2_to_1) / 2

def compute_hausdorff_distance(points1, points2):
    """Computes the symmetric Hausdorff distance using cKDTree."""
    try:
        tree1 = cKDTree(points1)
        tree2 = cKDTree(points2)

        distances_1_to_2, _ = tree2.query(points1, k=1)
        distances_2_to_1, _ = tree1.query(points2, k=1)

        hausdorff_1_to_2 = np.max(distances_1_to_2)
        hausdorff_2_to_1 = np.max(distances_2_to_1)

        hausdorff_dist = max(hausdorff_1_to_2, hausdorff_2_to_1)
    except Exception as e:
        print(f"  [Hausdorff Error: {e}]")
        hausdorff_dist = np.nan

    return hausdorff_dist

def compute_normal_consistency(points1, normals1, points2, normals2):
    """
    Computes the average normal deviation (in degrees) between two point clouds.
    It finds the nearest neighbor in set 2 for every point in set 1, 
    and calculates the angle between their normals.
    """
    try:
        # Build tree on the second set
        tree = cKDTree(points2)
        
        # Find nearest indices in points2 for each point in points1
        _, indices = tree.query(points1, k=1)
        
        # Get the normals of the nearest neighbors
        nearest_normals = normals2[indices]
        
        # Compute dot product: (N1 . N2)
        # We assume normals are already normalized by trimesh
        dot_products = np.sum(normals1 * nearest_normals, axis=1)
        
        # Clip to valid range for arccos [-1, 1] to handle float errors
        dot_products = np.clip(dot_products, -1.0, 1.0)
        
        # Calculate angles in degrees
        angles = np.degrees(np.arccos(dot_products))
        
        # We perform this symmetrically? 
        # Usually Normal Consistency is computed One-Way (Original -> Reconstruction)
        # to see how well the reconstruction preserves the original normals.
        # However, for consistency with Chamfer, we can average both directions.
        return np.mean(angles)
        
    except Exception as e:
        print(f"Error computing normal consistency: {e}")
        return np.nan

def compute_symmetric_normal_consistency(points1, normals1, points2, normals2):
    """Computes symmetric average angular error in degrees."""
    # Orig -> Reconstructed
    nc_1_to_2 = compute_normal_consistency(points1, normals1, points2, normals2)
    
    # Reconstructed -> Orig (checks if the reconstruction has 'hallucinated' normals)
    nc_2_to_1 = compute_normal_consistency(points2, normals2, points1, normals1)
    
    return (nc_1_to_2 + nc_2_to_1) / 2

def compute_all_metrics(original_mesh, decompressed_mesh, num_points=10000):
    """Computes all distortion metrics including Normal Deviation."""
    try:
        # Sample points AND normals
        points_orig, normals_orig = sample_points_and_normals(original_mesh, num_points)
        points_decomp, normals_decomp = sample_points_and_normals(decompressed_mesh, num_points)

        # 1. Chamfer
        chamfer = compute_symmetric_chamfer(points_orig, points_decomp)
        
        # 2. Hausdorff
        hausdorff = compute_hausdorff_distance(points_orig, points_decomp)
        
        # 3. Normal Deviation (Symmetric)
        normal_dev = compute_symmetric_normal_consistency(
            points_orig, normals_orig, 
            points_decomp, normals_decomp
        )

        return {
            "chamfer": chamfer, 
            "hausdorff": hausdorff, 
            "normal_dev": normal_dev
        }
        
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return {
            "chamfer": np.nan, 
            "hausdorff": np.nan, 
            "normal_dev": np.nan
        }