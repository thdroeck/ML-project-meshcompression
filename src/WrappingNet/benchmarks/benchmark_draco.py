import os
import glob
import time
import DracoPy as draco
import trimesh
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import concurrent.futures
import warnings

from benchmark_utils import get_mesh_stats, compute_all_metrics

# Suppress trimesh runtime warnings (common with degenerate faces in compression)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="trimesh")

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).parent.resolve()
WRAPPINGNET_PATH = SCRIPT_DIR.parent

# Point to the root of the Manifold40 dataset
DATA_DIR = WRAPPINGNET_PATH / "datasets" / "shrec_16"

RESULTS_DIR = WRAPPINGNET_PATH / "results"
RESULT_FILE = RESULTS_DIR / "draco_benchmark_shrec_16.csv"

# quantization bits for position (geometry)
QUANTIZATION_LEVELS = [2, 4, 6, 8, 10, 12, 14]
# Number of points for metric calculations
N_METRIC_POINTS = 20000
# --------------------

def process_mesh(mesh_path):
    """
    Worker function to process a single mesh file for all quantization levels.
    Executed in parallel.
    """
    mesh_results = []
    
    # Try to determine a relative path for cleaner logging/CSV output
    try:
        # Check if mesh_path is relative to DATA_DIR
        rel_path = str(Path(mesh_path).relative_to(DATA_DIR))
    except ValueError:
        # Fallback if path is weird
        rel_path = Path(mesh_path).name

    try:
        # Get actual file size on disk for compression ratio
        original_file_size = os.path.getsize(mesh_path)

        original_mesh = trimesh.load_mesh(mesh_path, process=False)

        if original_mesh is None or not hasattr(original_mesh, 'vertices'):
            return []

        if not hasattr(original_mesh, 'faces') or len(original_mesh.faces) == 0:
            return []

        stats = get_mesh_stats(original_mesh)
        original_num_vertices = stats["vertices"]

        if original_num_vertices == 0:
            return []

        vertices = np.asarray(original_mesh.vertices, dtype=np.float32)
        faces = np.asarray(original_mesh.faces, dtype=np.int32)

    except Exception as e:
        # In multiprocessing, print can sometimes get messy, but it's okay for errors
        print(f"Error loading {rel_path}: {e}")
        return []

    # Loop through quantization levels
    for qp in QUANTIZATION_LEVELS:
        # --- 1. Rate & Compression Performance ---
        start_time = time.perf_counter()
        try:
            compressed_data = draco.encode_mesh_to_buffer(vertices, faces, quantization_bits=qp)
        except Exception as e:
            print(f"Draco encoding error: {e} for {rel_path} (QP={qp})")
            continue
        compression_time = time.perf_counter() - start_time

        if compressed_data is None or len(compressed_data) == 0:
            continue

        compressed_size_bytes = len(compressed_data)
        bpv = (compressed_size_bytes * 8) / original_num_vertices

        # Calculate Compression Ratio (File Size / Compressed Size)
        if compressed_size_bytes > 0:
            compression_ratio = original_file_size / float(compressed_size_bytes)
        else:
            compression_ratio = 0.0

        # --- 2. Decompression Performance ---
        start_time = time.perf_counter()
        try:
            decompressed_mesh_data = draco.decode_buffer_to_mesh(compressed_data)
            decompressed_mesh = trimesh.Trimesh(
                vertices=decompressed_mesh_data.points,
                faces=decompressed_mesh_data.faces,
                process=False
            )
        except Exception as e:
            print(f"Draco decoding error: {e} for {rel_path} (QP={qp})")
            continue
        decompression_time = time.perf_counter() - start_time

        # --- 3. Distortion (Geometric Error) ---
        try:
            distortion_metrics = compute_all_metrics(
                original_mesh,
                decompressed_mesh,
                N_METRIC_POINTS
            )
            
            # --- P2S (Point to Surface) CALCULATION ---
            # Sample points from both surfaces
            sample_size_p2s = 10000 
            p_orig, _ = trimesh.sample.sample_surface(original_mesh, sample_size_p2s)
            p_rec, _ = trimesh.sample.sample_surface(decompressed_mesh, sample_size_p2s)

            # Distance: Rec Points -> Orig Surface
            _, dist_r2o, _ = trimesh.proximity.closest_point(original_mesh, p_rec)
            
            # Distance: Orig Points -> Rec Surface
            _, dist_o2r, _ = trimesh.proximity.closest_point(decompressed_mesh, p_orig)
            
            # Symmetric Average
            p2s_dist = (dist_r2o.mean() + dist_o2r.mean()) / 2.0

        except Exception as e:
            # Often happens if rtree is missing or mesh is degenerate
            # print(f"Metric computation error: {e} for {rel_path} (QP={qp})")
            p2s_dist = np.nan
            distortion_metrics = {"chamfer": np.nan, "hausdorff": np.nan}

        # --- 4. Store Results ---
        result = {
            "mesh_path": rel_path,
            "q_level": qp,
            "original_vertices": original_num_vertices,
            "original_faces": stats["faces"],
            "original_file_size_bytes": int(original_file_size),
            
            # Compression Results
            "compressed_size_bytes": compressed_size_bytes,
            "compression_ratio": float(compression_ratio), # NEW
            "bpv": bpv,
            
            # Geometric Results
            "p2s_dist": float(p2s_dist), # NEW
            "chamfer_distance": distortion_metrics.get("chamfer", np.nan),
            "hausdorff_distance": distortion_metrics.get("hausdorff", np.nan),
            
            # Times
            "compression_time_sec": compression_time,
            "decompression_time_sec": decompression_time
        }
        mesh_results.append(result)

    return mesh_results


def run_benchmark():
    """
    Runs the Draco benchmark on all meshes in DATA_DIR/raw/*/test
    """
    print(f"Starting Draco benchmark...")
    print(f"Dataset root: {DATA_DIR}")
    print(f"Results will be saved in: {RESULT_FILE}")

    RESULTS_DIR.mkdir(exist_ok=True)

    # Search for all test meshes
    search_path_obj = os.path.join(DATA_DIR, "*", "test", "**", "*.obj")
    search_path_ply = os.path.join(DATA_DIR, "*", "test", "**", "*.ply")

    print("Scanning for files...")
    mesh_files = glob.glob(search_path_obj, recursive=True) + glob.glob(search_path_ply, recursive=True)

    if not mesh_files:
        print(f"No meshes found in {DATA_DIR}/*/test/")
        print("Please check directory structure.")
        return

    print(f"Found {len(mesh_files)} meshes. Starting parallel processing...")
    
    all_results = []
    
    # Determine max workers (default to CPU count if None)
    max_workers = os.cpu_count() 

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        # We use a list to force immediate submission so tqdm knows the total count
        results_iterator = list(tqdm(executor.map(process_mesh, mesh_files), total=len(mesh_files), desc="Processing"))

        # Aggregate results
        for mesh_result_list in results_iterator:
            all_results.extend(mesh_result_list)

    if not all_results:
        print("No results were generated. Check for errors.")
        return

    # Save to CSV
    df = pd.DataFrame(all_results)
    # Sort for better readability: by mesh path, then by QP
    df = df.sort_values(by=["mesh_path", "q_level"])
    df.to_csv(RESULT_FILE, index=False)
    
    print(f"\nBenchmark completed. Results saved in {RESULT_FILE}")
    print("\n--- Summary Statistics ---")
    print(f"Mean Compression Ratio: {df['compression_ratio'].mean():.2f}x")
    print(f"Mean P2S Distance:      {df['p2s_dist'].mean():.6f}")
    print(f"Mean Chamfer Distance:  {df['chamfer_distance'].mean():.6f}")

if __name__ == "__main__":
    run_benchmark()