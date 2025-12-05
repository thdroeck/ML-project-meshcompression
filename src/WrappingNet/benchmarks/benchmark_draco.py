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

from benchmark_utils import get_mesh_stats, compute_all_metrics

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).parent.resolve()
WRAPPINGNET_PATH = SCRIPT_DIR.parent

# Point to the root of the Manifold40 dataset
DATA_DIR = WRAPPINGNET_PATH / "datasets" / "Manifold40"

RESULTS_DIR = WRAPPINGNET_PATH / "results"
RESULT_FILE = RESULTS_DIR / "draco_benchmark_manifold40.csv"

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
        original_mesh = trimesh.load_mesh(mesh_path)

        if original_mesh is None or not hasattr(original_mesh, 'vertices'):
            return []

        if not hasattr(original_mesh, 'faces') or len(original_mesh.faces) == 0:
            return []

        if not original_mesh.is_watertight:
            pass

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

        # --- 2. Decompression Performance ---
        start_time = time.perf_counter()
        try:
            decompressed_mesh_data = draco.decode_buffer_to_mesh(compressed_data)
            decompressed_mesh = trimesh.Trimesh(
                vertices=decompressed_mesh_data.points,
                faces=decompressed_mesh_data.faces
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
        except Exception as e:
            print(f"Metric computation error: {e} for {rel_path} (QP={qp})")
            continue

        # --- 4. Store Results ---
        result = {
            "mesh_path": rel_path,
            "q_level": qp,
            "original_vertices": original_num_vertices,
            "original_faces": stats["faces"],
            "compressed_size_bytes": compressed_size_bytes,
            "bpv": bpv,
            "chamfer_distance": distortion_metrics["chamfer"],
            "hausdorff_distance": distortion_metrics["hausdorff"],
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

    # Pattern: ROOT / raw / <any_category> / test / <any_subfolder> / <file>
    # search_path_obj = os.path.join(DATA_DIR, "raw", "*", "test", "**", "*.obj")
    # search_path_ply = os.path.join(DATA_DIR, "raw", "*", "test", "**", "*.ply")
    search_path_obj = os.path.join(DATA_DIR, "raw", "airplane", "test", "**", "*.obj")
    search_path_ply = os.path.join(DATA_DIR, "raw", "airplane", "test", "**", "*.ply")

    print("Scanning for files...")
    mesh_files = glob.glob(search_path_obj, recursive=True) + glob.glob(search_path_ply, recursive=True)

    if not mesh_files:
        print(f"No meshes found in {DATA_DIR}/raw/*/test/")
        print("Please check directory structure.")
        return

    print(f"Found {len(mesh_files)} meshes. Starting parallel processing...")
    
    all_results = []
    
    # Determine max workers (default to CPU count if None)
    # You can lower this if memory usage is too high (e.g., max_workers=4)
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

if __name__ == "__main__":
    run_benchmark()