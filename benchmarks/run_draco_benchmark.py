import os
import glob
import time
import draco
import trimesh
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from benchmark_utils import get_mesh_stats, compute_all_metrics

# --- CONFIGURATION ---
DATA_DIR = "../data/test_set" 
RESULTS_DIR = "../results"
RESULT_FILE = os.path.join(RESULTS_DIR, "draco_benchmark.csv")

# quantization bits for position (geometry)
QUANTIZATION_LEVELS = [8, 10, 12, 14, 16] 
# Number of points for metric calculations
N_METRIC_POINTS = 20000
# --------------------

def run_benchmark():
    """
    Runs the Draco benchmark on all meshes in DATA_DIR
    and saves the results in RESULT_FILE.
    """
    print(f"Starting Draco benchmark...")
    print(f"Test set source: {DATA_DIR}")
    print(f"Results will be saved in: {RESULT_FILE}")
    
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    
    # Find all .obj or .ply files
    mesh_files = glob.glob(os.path.join(DATA_DIR, "*.obj")) + glob.glob(os.path.join(DATA_DIR, "*.ply"))
    
    if not mesh_files:
        print(f"No meshes found in {DATA_DIR}. Make sure your test set is located there.")
        return

    all_results = []

    for mesh_path in tqdm(mesh_files, desc="Processing meshes"):
        try:
            original_mesh = trimesh.load_mesh(mesh_path)
            if not original_mesh.is_watertight:
                print(f"Warning: {os.path.basename(mesh_path)} is not watertight.")
            
            stats = get_mesh_stats(original_mesh)
            original_num_vertices = stats["vertices"]
            
            # Convert to numpy arrays for the Draco encoder
            vertices = np.asarray(original_mesh.vertices)
            faces = np.asarray(original_mesh.faces)

        except Exception as e:
            print(f"Error loading {mesh_path}: {e}")
            continue

        for qp in tqdm(QUANTIZATION_LEVELS, desc=f"QP for {os.path.basename(mesh_path)}", leave=False):
            
            # --- 1. Rate & Compression Performance ---
            start_time = time.perf_counter()
            try:
                # Compress the mesh
                compressed_data = draco.encode_mesh(vertices, faces, quantization_bits=qp)
            except draco.EncodingError as e:
                print(f"Draco encoding error: {e}")
                continue
            compression_time = time.perf_counter() - start_time
            
            compressed_size_bytes = len(compressed_data)
            bpv = (compressed_size_bytes * 8) / original_num_vertices

            # --- 2. Decompression Performance ---
            start_time = time.perf_counter()
            try:
                # Decompress the mesh
                decompressed_mesh_data = draco.decode_buffer(compressed_data)
                decompressed_mesh = trimesh.Trimesh(
                    vertices=decompressed_mesh_data.vertices,
                    faces=decompressed_mesh_data.faces
                )
            except draco.DecodingError as e:
                print(f"Draco decoding error: {e}")
                continue
            decompression_time = time.perf_counter() - start_time

            # --- 3. Distortion (Geometric Error) ---
            distortion_metrics = compute_all_metrics(
                original_mesh, 
                decompressed_mesh, 
                N_METRIC_POINTS
            )

            # --- 4. Save Results ---
            result = {
                "mesh_name": os.path.basename(mesh_path),
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
            all_results.append(result)

    # Save all results into a single CSV file
    df = pd.DataFrame(all_results)
    df.to_csv(RESULT_FILE, index=False)
    print(f"\nBenchmark completed. Results saved in {RESULT_FILE}")

if __name__ == "__main__":
    run_benchmark()
