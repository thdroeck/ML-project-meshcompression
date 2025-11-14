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
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT.parent / "lib" / "Manifold40"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULT_FILE = RESULTS_DIR / "draco_benchmark_manifold40.csv"

# quantization bits for position (geometry)
QUANTIZATION_LEVELS = [8, 10, 12, 14, 16]
# Number of points for metric calculations
N_METRIC_POINTS = 20000


# --------------------

def process_mesh(mesh_path):
    """
    Worker function to process a single mesh file for all quantization levels.
    This function will be run in parallel by the ProcessPoolExecutor.
    """
    mesh_results = []
    try:
        original_mesh = trimesh.load_mesh(mesh_path)

        if original_mesh is None or not hasattr(original_mesh, 'vertices'):
            return []  # Return empty list if mesh is invalid

        if not hasattr(original_mesh, 'faces') or len(original_mesh.faces) == 0:
            return []  # Skip point clouds

        if not original_mesh.is_watertight:
            pass

        stats = get_mesh_stats(original_mesh)
        original_num_vertices = stats["vertices"]

        if original_num_vertices == 0:
            return []  # Skip empty meshes

        vertices = np.asarray(original_mesh.vertices, dtype=np.float32)
        faces = np.asarray(original_mesh.faces, dtype=np.int32)

    except Exception as e:
        print(f"Error loading {mesh_path}: {e}")
        return []  # Return empty list on failure

    # Loop through quantization levels for this single mesh
    for qp in QUANTIZATION_LEVELS:

        # --- 1. Rate & Compression Performance ---
        start_time = time.perf_counter()
        try:
            compressed_data = draco.encode_mesh_to_buffer(vertices, faces, quantization_bits=qp)
        except Exception as e:
            print(f"Draco encoding error: {e} for {mesh_path} with qp={qp}")
            continue  # Skip this qp level
        compression_time = time.perf_counter() - start_time

        if compressed_data is None or len(compressed_data) == 0:
            print(f"Draco encoding returned empty data for {mesh_path} with qp={qp}")
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
            print(f"Draco decoding error: {e} for {mesh_path} with qp={qp}")
            continue  # Skip this qp level
        decompression_time = time.perf_counter() - start_time

        # --- 3. Distortion (Geometric Error) ---
        distortion_metrics = compute_all_metrics(
            original_mesh,
            decompressed_mesh,
            N_METRIC_POINTS
        )

        # --- 4. Store Results ---
        result = {
            "mesh_path": str(Path(mesh_path).relative_to(DATA_DIR)),
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

    return mesh_results  # Return list of results for this mesh


def run_benchmark():
    """
    Runs the Draco benchmark on all meshes in DATA_DIR
    and saves the results in RESULT_FILE.
    """
    print(f"Starting Draco benchmark...")
    print(f"Test set source: {DATA_DIR}")
    print(f"Results will be saved in: {RESULT_FILE}")

    # Create the /results directory if it doesn't exist
    RESULTS_DIR.mkdir(exist_ok=True)

    # Recursive search for mesh files
    search_path_obj = os.path.join(DATA_DIR, "**", "*.obj")
    search_path_ply = os.path.join(DATA_DIR, "**", "*.ply")

    mesh_files = glob.glob(search_path_obj, recursive=True) + glob.glob(search_path_ply, recursive=True)

    if not mesh_files:
        print(f"No meshes (.obj or .ply) found in {DATA_DIR} and subdirectories.")
        return

    print(f"Found {len(mesh_files)} total meshes. Starting parallel processing...")
    all_results = []

    # --- MULTIPROCESSING ---
    with concurrent.futures.ProcessPoolExecutor() as executor:

        # Use executor.map to apply the 'process_mesh' function to every item in 'mesh_files'
        # This distributes the work across all your CPU cores.
        # We wrap this in tqdm to get a progress bar.

        results_iterator = executor.map(process_mesh, mesh_files)

        for mesh_results in tqdm(results_iterator, total=len(mesh_files), desc="Processing meshes"):
            # 'mesh_results' is the list of dicts returned by the 'process_mesh' function
            # We extend our main list with the results for this mesh
            all_results.extend(mesh_results)

    # -------------------------

    if not all_results:
        print("No results were generated. Check for errors above.")
        return

    # Save all results into a single CSV file
    df = pd.DataFrame(all_results)
    df.to_csv(RESULT_FILE, index=False)
    print(f"\nBenchmark completed. Results saved in {RESULT_FILE}")


if __name__ == "__main__":
    run_benchmark()