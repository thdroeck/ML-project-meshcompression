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

from WrappingNet.wrappingnet.models import Autoencoder
from benchmark_utils import compute_all_metrics, get_mesh_stats

import warnings

# Suppress runtime warnings from trimesh regarding zero division
warnings.filterwarnings("ignore", category=RuntimeWarning, module="trimesh")

# --- CONFIGURATION (Default values if arguments aren't provided) ---
SCRIPT_DIR = Path(__file__).parent.resolve()
WRAPPINGNET_PATH = SCRIPT_DIR.parent

DATA_DIR_DEFAULT = WRAPPINGNET_PATH / "datasets" / "Manifold40"
CHECKPOINT_DEFAULT = WRAPPINGNET_PATH.parent.parent / "trained" / \
                     "MeshAE_MSL2_src_WrappingNet_datasets_Manifold40_raw_d128_e10.ckpt"

RESULTS_DIR = WRAPPINGNET_PATH / "results"
RESULT_FILE = RESULTS_DIR / "autoencoder_benchmark.csv"

LATENT_DIM_DEFAULT = 512
N_METRIC_POINTS = 50000

# --- GLOBAL VARIABLES FOR WORKER PROCESSES ---
GLOBAL_ARGS = None
GLOBAL_MODEL = None
GLOBAL_DEVICE = torch.device("cpu")


def load_model_instance(args, device):
    """Loads the Autoencoder model once per worker process."""
    
    # Initialize the model structure
    model = Autoencoder(input_dim=7, feature_dim=args.latent_dim, num_loop=3).to(device)

    # Load the state dictionary
    saved = torch.load(args.checkpoint, map_location=device)
    if isinstance(saved, dict) and 'state_dict' in saved:
        state = saved['state_dict']
    else:
        state = saved

    # Load state dict with prefix stripping fallback
    try:
        model.load_state_dict(state)
    except RuntimeError:
        new_state = {}
        for k, v in state.items():
            new_key = k
            if k.startswith('module.'):
                new_key = k[len('module.'):]
            elif k.startswith('model.'):
                new_key = k[len('model.'):]
            new_state[new_key] = v
        model.load_state_dict(new_state)

    model.eval()
    # In worker processes, limit intraop/omp threads to avoid oversubscription
    torch.set_num_threads(1)
    return model


def worker_init(args, gpu_id=None):
    """
    Initializer function for each worker process.
    Loads the model and sets the global device/model for the process.
    """
    global GLOBAL_MODEL
    global GLOBAL_ARGS
    global GLOBAL_DEVICE
    
    # Set the global arguments
    GLOBAL_ARGS = args 
    
    # Determine the device for this worker
    if gpu_id is not None and torch.cuda.is_available():
        GLOBAL_DEVICE = torch.device(f"cuda:{gpu_id}")
    else:
        GLOBAL_DEVICE = torch.device("cpu")
        
    print(f"Worker process {os.getpid()} initialized on {GLOBAL_DEVICE}")
        
    # Load the model only once per worker
    GLOBAL_MODEL = load_model_instance(args, GLOBAL_DEVICE)


def process_mesh(mesh_path):
    """Executed inside worker process for a single mesh file, using GLOBAL_MODEL."""
    
    global GLOBAL_MODEL
    global GLOBAL_ARGS
    global GLOBAL_DEVICE

    model = GLOBAL_MODEL
    device = GLOBAL_DEVICE
    
    if model is None:
        return []

    results = []
    
    # Get relative path
    try:
        rel_path = str(Path(mesh_path).relative_to(GLOBAL_ARGS.dataset))
    except Exception:
        rel_path = Path(mesh_path).name

    try:
        # --- MESH LOADING AND DATA PREPARATION ---
        original_mesh = trimesh.load_mesh(mesh_path, process=False)
        if original_mesh is None or not hasattr(original_mesh, 'vertices') or len(original_mesh.vertices) == 0:
            print(f"Warning: skipping empty mesh {rel_path}")
            return []

        stats = get_mesh_stats(original_mesh)
        num_vertices = stats.get('vertices', len(original_mesh.vertices))
        if num_vertices == 0:
            return []

        # Prepare tensors and move to the worker's device
        pos = torch.tensor(original_mesh.vertices, dtype=torch.float32, device=device)
        face = torch.tensor(original_mesh.faces, dtype=torch.long, device=device)
        data = Data(pos=pos, face=face.T)

    except Exception as e:
        print(f"Error loading {rel_path}: {e}")
        return []

    try:
        # --- ENCODING ---
        with torch.no_grad():
            start = time.perf_counter()
            face_base, features = model.encoder(data.pos, data.face.T)
            # Max pooling to get the latent code
            latent_code = torch.max(features, dim=0)[0].unsqueeze(0)
            latent_code = model.mlp2(latent_code).squeeze(0)
            encode_time = time.perf_counter() - start

        if latent_code is None or latent_code.numel() == 0:
            return []

        # Compute auxiliary data
        try:
            num_nodes = int(face_base.max().item()) + 1
        except Exception:
            num_nodes = pos.shape[0]

        pos_base = data.pos[0:num_nodes]
        if not isinstance(face_base, torch.Tensor):
            face_base = torch.tensor(face_base, dtype=torch.long, device=device)

        # --- COMPRESSION METRICS ---
        
        # 1. Latent Size
        latent_bytes = latent_code.numel() * 4
        latent_bits = latent_bytes * 8.0
        bpv_latent_only = latent_bits / float(num_vertices)

        # 2. File Compression Ratio (Disk Size)
        pos_base_bytes = pos_base.numel() * 4
        face_base_bytes = face_base.numel() * 4 
        compressed_bytes_full = int(latent_bytes + pos_base_bytes + face_base_bytes)
        original_file_size = os.path.getsize(mesh_path)
        
        if compressed_bytes_full > 0:
            compression_ratio_file = original_file_size / float(compressed_bytes_full)
        else:
            compression_ratio_file = 0.0

        # 3. Vertex Compression Ratio
        vertex_compression_ratio = float(num_vertices) / float(GLOBAL_ARGS.latent_dim)

        # --- DECODING ---
        with torch.no_grad():
            start = time.perf_counter()
            repeat_count = features.shape[0]
            features_repeat = latent_code.repeat(repeat_count, 1)
            pos_list, face_list = model.decoder(pos_base, face_base, features_repeat)
            decode_time = time.perf_counter() - start

        rec_vertices = pos_list[-1].cpu().numpy()
        rec_faces = face_list[-1].cpu().numpy()
        rec_mesh = trimesh.Trimesh(vertices=rec_vertices, faces=rec_faces, process=False)

        # --- GEOMETRIC & ROBUSTNESS METRICS ---
        metrics = compute_all_metrics(original_mesh, rec_mesh, N_METRIC_POINTS)

        # --- P2S (Point to Surface) CALCULATION ---
        # Note: Chamfer is usually P2P (Point to Point). P2S is more accurate.
        # We sample points from Original and find distance to Surface of Rec, and vice versa.
        try:
            # 1. Sample points
            sample_size_p2s = 10000
            p_orig, _ = trimesh.sample.sample_surface(original_mesh, sample_size_p2s)
            p_rec, _ = trimesh.sample.sample_surface(rec_mesh, sample_size_p2s)

            # 2. Distance: Rec Points -> Orig Surface
            # closest_point returns (closest_points, distances, triangle_id)
            _, dist_r2o, _ = trimesh.proximity.closest_point(original_mesh, p_rec)
            p2s_r2o = dist_r2o.mean()

            # 3. Distance: Orig Points -> Rec Surface
            _, dist_o2r, _ = trimesh.proximity.closest_point(rec_mesh, p_orig)
            p2s_o2r = dist_o2r.mean()

            # Average symmetric P2S
            p2s_dist = (p2s_r2o + p2s_o2r) / 2.0
        except Exception as e_p2s:
            print(f"P2S Calc Error {rel_path}: {e_p2s}")
            p2s_dist = np.nan

        results.append({
            "mesh_path": rel_path,
            "latent_dim": int(latent_code.numel()),
            "original_vertices": int(num_vertices),
            
            # SIZE & RATIOS
            "original_file_size_bytes": int(original_file_size),
            "compression_ratio_file": float(compression_ratio_file),
            "compression_ratio_vertex": float(vertex_compression_ratio), # NEW
            "bpv": float(bpv_latent_only),
            
            # GEOMETRY
            "p2s_dist": float(p2s_dist), # NEW
            "chamfer_distance": float(metrics.get('chamfer', np.nan)),
            "hausdorff_distance": float(metrics.get('hausdorff', np.nan)),
            "normal_deviation": float(metrics.get('normal_dev', np.nan)),
            
            # EFFICIENCY
            "compression_time_sec": float(encode_time),
            "decompression_time_sec": float(decode_time),
        })

    except Exception as e:
        print(f"Error processing {rel_path}: {e}")
        return []

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Autoencoder compression on 3D meshes.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATA_DIR_DEFAULT,
        help="Path to the Manifold40 dataset directory"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_DEFAULT,
        help="Path to the model checkpoint file (.ckpt)"
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=LATENT_DIM_DEFAULT,
        help="Latent dimension size used in the Autoencoder model"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=RESULT_FILE,
        help="Path to output CSV results file"
    )
    parser.add_argument(
        "--gpus",
        type=lambda s: [int(item) for item in s.split(',')],
        default=None,
        help="Comma-separated list of GPU IDs"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes."
    )
    return parser.parse_args()


def run_benchmark():
    args = parse_args()

    print("Starting Autoencoder benchmark...")
    print(f"Dataset directory: {args.dataset}")
    print(f"Model checkpoint: {args.checkpoint}")
    print(f"Latent dimension: {args.latent_dim}")
    
    RESULTS_DIR.mkdir(exist_ok=True)

    # --- Worker and GPU Setup ---
    total_cpus = os.cpu_count()
    gpu_ids = []

    if torch.cuda.is_available():
        if args.gpus is not None:
            gpu_ids = args.gpus
        else:
            gpu_ids = list(range(torch.cuda.device_count()))
        
        if gpu_ids:
            print(f"CUDA available. Using {len(gpu_ids)} GPU(s): {gpu_ids}")
        
    if args.workers is not None:
        num_workers = args.workers
    elif gpu_ids:
        num_workers = min(total_cpus, len(gpu_ids) * 2) 
    else:
        num_workers = total_cpus

    num_workers = max(1, num_workers)
    print(f"Using {num_workers} worker processes.")
    
    # --- File Scanning ---
    search_path_obj = os.path.join(args.dataset, "*", "test", "**", "*.obj")
    search_path_ply = os.path.join(args.dataset, "*", "test", "**", "*.ply")

    print("Scanning files...")
    mesh_files = sorted(
        glob.glob(search_path_obj, recursive=True)
        + glob.glob(search_path_ply, recursive=True)
    )

    if not mesh_files:
        print(f"No meshes found in {args.dataset}. Check paths and file types.")
        return

    all_results = []

    # Note: Using gpu_ids[0] for initialization here simplistically. 
    # For advanced multi-GPU round-robin with ProcessPool, a Queue is usually required.
    # This setup assumes one main GPU or that the OS handles CUDA context switching reasonably well.
    init_gpu = gpu_ids[0] if gpu_ids else None

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers, 
        initializer=worker_init, 
        initargs=(args, init_gpu)
    ) as pool:
        for output in tqdm(pool.map(process_mesh, mesh_files), total=len(mesh_files)):
            all_results.extend(output)

    df = pd.DataFrame(all_results)
    if not df.empty:
        df = df.sort_values("mesh_path")
        df.to_csv(args.results, index=False)
        print(f"\nBenchmark done and results saved to → {args.results}")
        
        # Updated Summary Statistics
        print("\n--- Summary Statistics ---")
        print(f"Mean Vertex Comp Ratio:   {df['compression_ratio_vertex'].mean():.2f}x (Verts/Latent)")
        print(f"Mean File Comp Ratio:     {df['compression_ratio_file'].mean():.2f}x")
        print(f"Mean P2S Distance:        {df['p2s_dist'].mean():.6f}")
        print(f"Mean Chamfer Distance:    {df['chamfer_distance'].mean():.6f}")
        print(f"Mean Comp Time:           {df['compression_time_sec'].mean()*1000:.2f} ms")
        
    else:
        print("\nNo results to write.")


if __name__ == "__main__":
    if os.name == 'nt' or os.uname().sysname == 'Darwin':
        torch.multiprocessing.set_start_method('spawn', force=True)
        
    run_benchmark()