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
import argparse
import warnings

from WrappingNet.wrappingnet.models import Autoencoder
from benchmark_utils import get_mesh_stats

# Suppress runtime warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).parent.resolve()
WRAPPINGNET_PATH = SCRIPT_DIR.parent

DATA_DIR_DEFAULT = WRAPPINGNET_PATH / "datasets" / "Manifold40"
CHECKPOINT_DEFAULT = WRAPPINGNET_PATH.parent.parent / "trained" / \
                      "MeshAE_MSL2_src_WrappingNet_datasets_Manifold40_raw_d128_e10.ckpt"

RESULTS_DIR = WRAPPINGNET_PATH / "results"
RESULT_FILE = RESULTS_DIR / "ae_compression_ratios.csv"

LATENT_DIM_DEFAULT = 512
P2S_SAMPLE_SIZE = 10000 # Enough for accurate average
# --------------------

# --- GLOBAL VARIABLES ---
GLOBAL_ARGS = None
GLOBAL_MODEL = None
GLOBAL_DEVICE = torch.device("cpu")

def load_model_instance(args, device):
    """Loads the Autoencoder model once per worker process."""
    model = Autoencoder(input_dim=7, feature_dim=args.latent_dim, num_loop=3).to(device)
    saved = torch.load(args.checkpoint, map_location=device)
    
    state = saved['state_dict'] if 'state_dict' in saved else saved
    new_state = {}
    for k, v in state.items():
        new_key = k.replace('module.', '').replace('model.', '')
        new_state[new_key] = v
    
    model.load_state_dict(new_state)
    model.eval()
    torch.set_num_threads(1)
    return model

def worker_init(args, gpu_id=None):
    global GLOBAL_MODEL, GLOBAL_ARGS, GLOBAL_DEVICE
    GLOBAL_ARGS = args 
    GLOBAL_DEVICE = torch.device(f"cuda:{gpu_id}") if (gpu_id is not None and torch.cuda.is_available()) else torch.device("cpu")
    GLOBAL_MODEL = load_model_instance(args, GLOBAL_DEVICE)

def process_mesh(mesh_path):
    global GLOBAL_MODEL, GLOBAL_ARGS, GLOBAL_DEVICE
    model = GLOBAL_MODEL
    device = GLOBAL_DEVICE
    
    if model is None: return []

    try:
        # Relative path for CSV
        try: rel_path = str(Path(mesh_path).relative_to(GLOBAL_ARGS.dataset))
        except: rel_path = Path(mesh_path).name

        # 1. Load Original
        original_mesh = trimesh.load_mesh(mesh_path, process=False)
        if original_mesh is None or not hasattr(original_mesh, 'vertices'): return []
        
        num_vertices = len(original_mesh.vertices)
        if num_vertices == 0: return []

        original_file_size = os.path.getsize(mesh_path)

        # Prepare for Model
        pos = torch.tensor(original_mesh.vertices, dtype=torch.float32, device=device)
        face = torch.tensor(original_mesh.faces, dtype=torch.long, device=device)
        data = Data(pos=pos, face=face.T)

        # 2. Encode
        with torch.no_grad():
            t0 = time.perf_counter()
            face_base, features = model.encoder(data.pos, data.face.T)
            latent_code = torch.max(features, dim=0)[0].unsqueeze(0)
            latent_code = model.mlp2(latent_code).squeeze(0)
            encode_time = time.perf_counter() - t0

        if latent_code is None: return []

        # 3. Calculate Sizes & Ratios
        latent_bytes = latent_code.numel() * 4
        
        # Base Mesh overhead (for Full Ratio)
        try: num_nodes = int(face_base.max().item()) + 1
        except: num_nodes = pos.shape[0]
        pos_base = data.pos[0:num_nodes]
        base_bytes = (pos_base.numel() * 4) + (face_base.numel() * 4)
        
        full_compressed_bytes = latent_bytes + base_bytes

        # Ratios
        ratio_vertex = float(num_vertices) / float(GLOBAL_ARGS.latent_dim)
        ratio_file_full = original_file_size / float(full_compressed_bytes) if full_compressed_bytes > 0 else 0
        ratio_latent_only = original_file_size / float(latent_bytes) if latent_bytes > 0 else 0

        # 4. Decode
        with torch.no_grad():
            t1 = time.perf_counter()
            features_repeat = latent_code.repeat(features.shape[0], 1)
            pos_list, face_list = model.decoder(pos_base, face_base, features_repeat)
            decode_time = time.perf_counter() - t1

        # 5. Calculate P2S (Point-to-Surface) ONLY
        rec_mesh = trimesh.Trimesh(vertices=pos_list[-1].cpu().numpy(), faces=face_list[-1].cpu().numpy(), process=False)
        
        try:
            # Sample points
            p_orig, _ = trimesh.sample.sample_surface(original_mesh, P2S_SAMPLE_SIZE)
            p_rec, _ = trimesh.sample.sample_surface(rec_mesh, P2S_SAMPLE_SIZE)

            # Distance Rec -> Orig
            _, dist_r2o, _ = trimesh.proximity.closest_point(original_mesh, p_rec)
            # Distance Orig -> Rec
            _, dist_o2r, _ = trimesh.proximity.closest_point(rec_mesh, p_orig)
            
            p2s_dist = (dist_r2o.mean() + dist_o2r.mean()) / 2.0
        except Exception:
            p2s_dist = np.nan

        return [{
            "mesh_path": rel_path,
            "original_vertices": num_vertices,
            "original_file_size": int(original_file_size),
            
            # The requested comparison metrics
            "ratio_vertex": float(ratio_vertex),
            "ratio_file_full": float(ratio_file_full),
            "ratio_latent_only": float(ratio_latent_only),
            "p2s_dist": float(p2s_dist),
            
            "enc_time": float(encode_time),
            "dec_time": float(decode_time)
        }]

    except Exception as e:
        return []

def run_benchmark():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATA_DIR_DEFAULT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DEFAULT)
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM_DEFAULT)
    parser.add_argument("--results", type=Path, default=RESULT_FILE)
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    print(f"--- AE Ratio Benchmark ---")
    print(f"Latent Dim: {args.latent_dim}")
    print(f"Saving to: {args.results}")

    # Worker setup
    gpu_ids = [int(x) for x in args.gpus.split(',')] if args.gpus else list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    total_workers = args.workers if args.workers else (max(1, len(gpu_ids)) if gpu_ids else os.cpu_count())
    total_workers = 4
    print(total_workers, "workers will be used.")

    # File scan
    files = sorted(glob.glob(os.path.join(args.dataset, "*", "test", "**", "*.obj"), recursive=True))
    if not files: return print("No files found.")

    init_gpu = gpu_ids[0] if gpu_ids else None
    
    all_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=total_workers, initializer=worker_init, initargs=(args, init_gpu)) as pool:
        for res in tqdm(pool.map(process_mesh, files), total=len(files)):
            all_results.extend(res)

    df = pd.DataFrame(all_results)
    if not df.empty:
        df = df.sort_values("mesh_path")
        df.to_csv(args.results, index=False)
        print("\n--- Averages ---")
        print(f"Ratio (Latent Only):  {df['ratio_latent_only'].mean():.2f}x")
        print(f"Ratio (Vertex):       {df['ratio_vertex'].mean():.2f}x")
        print(f"P2S Distance:         {df['p2s_dist'].mean():.6f}")
    else:
        print("No results.")

if __name__ == "__main__":
    if os.name != 'nt':
        try: torch.multiprocessing.set_start_method('spawn', force=True)
        except: pass
    run_benchmark()