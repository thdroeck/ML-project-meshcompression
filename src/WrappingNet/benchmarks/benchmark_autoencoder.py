import os
import glob
import time
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch
import trimesh
from torch_geometric.data import Data

# --- IMPORTS FROM YOUR PROJECT ---
# Ensure these paths are importable in your environment
from WrappingNet.wrappingnet.models import Autoencoder
from benchmark_utils import get_mesh_stats, compute_all_metrics

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULT_FILE = RESULTS_DIR / "autoencoder_benchmark.csv"

# Number of points for metric calculations (must match Draco script for fair comparison)
N_METRIC_POINTS = 20000
# --------------------

def run_benchmark(args):
    """
    Runs the WrappingNet benchmark on all meshes in the dataset
    and saves the results in a CSV similar to the Draco benchmark.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Benchmark on {device}")
    
    # 1. Load the Model
    print(f"Loading checkpoint: {args.model_checkpoint}")
    try:
        # Note: Input dim 7 and num_loop 3 based on your provided snippet
        model = Autoencoder(input_dim=7, feature_dim=args.latent_dim, num_loop=3)
        saved = torch.load(args.model_checkpoint, map_location="cpu") # Load to cpu first
        model.load_state_dict(saved)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 2. Setup Data Paths
    DATA_DIR = Path(args.dataset_path)
    print(f"Dataset root: {DATA_DIR}")
    
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    # Search specifically in raw/<category>/test
    search_path_obj = os.path.join(DATA_DIR, "raw", "*", "test", "**", "*.obj")
    mesh_files = glob.glob(search_path_obj, recursive=True)

    if not mesh_files:
        print(f"No .obj files found in {DATA_DIR}/raw/*/test/")
        return

    print(f"Found {len(mesh_files)} meshes. Starting benchmark...")

    all_results = []

    # 3. Processing Loop
    for mesh_path in tqdm(mesh_files, desc="Processing Meshes"):
        try:
            # --- Load Original Mesh ---
            original_mesh = trimesh.load_mesh(mesh_path)
            
            # Validation checks
            if original_mesh is None or not hasattr(original_mesh, 'vertices'): continue
            if not hasattr(original_mesh, 'faces') or len(original_mesh.faces) == 0: continue
            
            stats = get_mesh_stats(original_mesh)
            original_num_vertices = stats["vertices"]
            if original_num_vertices == 0: continue

            # Prepare Data for WrappingNet
            pos = torch.tensor(original_mesh.vertices, dtype=torch.float32).to(device)
            face = torch.tensor(original_mesh.faces, dtype=torch.long).to(device)
            
            # --- Run Inference (Measure Time) ---
            # Note: For Deep Learning methods, 'compression' is encoding to latent, 
            # 'decompression' is decoding latent. This model usually does both in forward().
            start_time = time.perf_counter()
            
            with torch.no_grad():
                # model inputs: pos, face_indices (transposed)
                pos_list, face_list, _ = model(pos, face.T)
            
            inference_time = time.perf_counter() - start_time
            
            # --- Extract Reconstructed Mesh ---
            # The last element in the list is the final refinement
            out_verts = pos_list[-1].detach().cpu().numpy()
            out_faces = face_list[-1].detach().cpu().numpy()
            
            reconstructed_mesh = trimesh.Trimesh(vertices=out_verts, faces=out_faces)

            # --- Calculate "Compressed" Size ---
            # For AE based compression, the "size" is the size of the latent vector.
            # Assuming float32 (4 bytes) precision for the latent vector.
            latent_size_bytes = args.latent_dim * 4 
            bpv = (latent_size_bytes * 8) / original_num_vertices

            # --- Compute Distortion Metrics ---
            # Using the same utils as Draco script for fairness
            distortion_metrics = compute_all_metrics(
                original_mesh,
                reconstructed_mesh,
                N_METRIC_POINTS
            )

            # --- Store Results ---
            try:
                rel_path = str(Path(mesh_path).relative_to(DATA_DIR))
            except ValueError:
                rel_path = Path(mesh_path).name

            result = {
                "mesh_path": rel_path,
                "model_latent_dim": args.latent_dim,
                "original_vertices": original_num_vertices,
                "original_faces": stats["faces"],
                
                # Compression Stats
                "compressed_size_bytes": latent_size_bytes,
                "bpv": bpv,
                
                # Distortion
                "chamfer_distance": distortion_metrics["chamfer"],
                "hausdorff_distance": distortion_metrics["hausdorff"],
                
                # Timing
                # Since the model runs end-to-end, we log total inference. 
                # You can split this by 2 if you want to estimate enc/dec split.
                "inference_time_sec": inference_time, 
            }
            all_results.append(result)

        except Exception as e:
            print(f"Error processing {mesh_path}: {e}")
            continue

    # 4. Save to CSV
    df = pd.DataFrame(all_results)
    df.to_csv(RESULT_FILE, index=False)
    print(f"\nBenchmark completed. Results saved in {RESULT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--latent_dim",
        type=int,
        required=True,
        help="Latent dimension used during training (e.g., 512, 1024)",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        required=True,
        help="Path to the .pt or .pth model checkpoint",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Root path to Manifold40 dataset",
    )

    args = parser.parse_args()
    run_benchmark(args)