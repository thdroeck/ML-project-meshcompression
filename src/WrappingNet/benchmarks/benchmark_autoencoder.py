import glob
import os
import time
from pathlib import Path
import torch
import trimesh
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data
import concurrent.futures

from WrappingNet.wrappingnet.models import Autoencoder
from benchmark_utils import compute_all_metrics, get_mesh_stats  # same as Draco benchmark

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).parent.resolve()
WRAPPINGNET_PATH = SCRIPT_DIR.parent

DATA_DIR = WRAPPINGNET_PATH / "datasets" / "Manifold40"
RESULTS_DIR = WRAPPINGNET_PATH / "results"
RESULT_FILE = RESULTS_DIR / "autoencoder_benchmark.csv"

LATENT_DIM = 128  # default, can be overridden
MODEL_CHECKPOINT = WRAPPINGNET_PATH.parent.parent / "trained" / "MeshAE_MSL2_src_WrappingNet_datasets_Manifold40_raw_d128_e1.ckpt"
N_METRIC_POINTS = 20000
# --------------------

# Load model once
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Autoencoder(input_dim=7, feature_dim=LATENT_DIM, num_loop=3).to(device)
saved = torch.load(MODEL_CHECKPOINT, map_location=device)
model.load_state_dict(saved)
model.eval()

def process_mesh(mesh_path):
    mesh_results = []

    try:
        rel_path = str(Path(mesh_path).relative_to(DATA_DIR))
    except ValueError:
        rel_path = Path(mesh_path).name

    try:
        original_mesh = trimesh.load_mesh(mesh_path)
        if original_mesh is None or len(original_mesh.vertices) == 0:
            return []
        if not hasattr(original_mesh, 'faces') or len(original_mesh.faces) == 0:
            return []

        stats = get_mesh_stats(original_mesh)
        original_num_vertices = stats["vertices"]
        if original_num_vertices == 0:
            return []

        vertices = torch.tensor(np.asarray(original_mesh.vertices, dtype=np.float32), device=device)
        faces = torch.tensor(np.asarray(original_mesh.faces, dtype=np.int64), device=device)

        mesh_data = Data(pos=vertices, face=faces.T)

    except Exception as e:
        print(f"Error loading {rel_path}: {e}")
        return []

    try:
        # --- Compression (encoding) ---
        start_time = time.perf_counter()
        with torch.no_grad():
            latent, _, _ = model(mesh_data.pos, mesh_data.face.T)
        compression_time = time.perf_counter() - start_time

        # Compressed size in bytes
        compressed_size_bytes = LATENT_DIM * 4  # 32-bit float per latent
        bpv = (compressed_size_bytes * 8) / original_num_vertices

        # --- Decompression (decoding) ---
        start_time = time.perf_counter()
        with torch.no_grad():
            reconstructed_vertices, reconstructed_faces, _ = model(mesh_data.pos, mesh_data.face.T)
        decompression_time = time.perf_counter() - start_time

        decompressed_mesh = trimesh.Trimesh(
            vertices=reconstructed_vertices[-1].cpu().numpy(),
            faces=reconstructed_faces[-1].cpu().numpy()
        )

        # --- Distortion Metrics ---
        distortion_metrics = compute_all_metrics(
            original_mesh,
            decompressed_mesh,
            N_METRIC_POINTS
        )

        result = {
            "mesh_path": rel_path,
            "latent_dim": LATENT_DIM,
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

    except Exception as e:
        print(f"Error processing {rel_path}: {e}")
        return []

    return mesh_results

def run_benchmark():
    print(f"Starting Autoencoder benchmark...")
    print(f"Dataset root: {DATA_DIR}")
    print(f"Results will be saved in: {RESULT_FILE}")

    RESULTS_DIR.mkdir(exist_ok=True)
    search_path_obj = os.path.join(DATA_DIR, "raw", "airplane", "test", "**", "*.obj")
    search_path_ply = os.path.join(DATA_DIR, "raw", "airplane", "test", "**", "*.ply")

    print("Scanning for files...")
    mesh_files = [f for f in tqdm(sorted(glob.glob(search_path_obj, recursive=True) + 
                                         glob.glob(search_path_ply, recursive=True)))]

    if not mesh_files:
        print(f"No meshes found in {DATA_DIR}/raw/*/test/")
        return

    all_results = []

    max_workers = os.cpu_count()

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results_iterator = list(tqdm(executor.map(process_mesh, mesh_files), total=len(mesh_files)))
        for mesh_result_list in results_iterator:
            all_results.extend(mesh_result_list)

    if not all_results:
        print("No results generated. Check for errors.")
        return

    df = pd.DataFrame(all_results)
    df = df.sort_values(by=["mesh_path"])
    df.to_csv(RESULT_FILE, index=False)

    print(f"Benchmark completed. Results saved in {RESULT_FILE}")

if __name__ == "__main__":
    run_benchmark()
