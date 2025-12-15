#!/usr/bin/env python3
"""
Saves results to CSV with many additional metrics (geometry, topology, runtime, robustness).
"""

import os
import glob
import time
import math
import warnings
from pathlib import Path
import concurrent.futures

import numpy as np
import pandas as pd
import trimesh
from tqdm import tqdm

# Optional faster dependencies (best-effort)
try:
    from scipy.spatial import cKDTree as KDTree
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except Exception:
    PSUTIL_AVAILABLE = False

import DracoPy as draco

# Local utils (you had benchmark_utils previously). If some functions exist there,
# keep using them; but we compute many additional metrics here.
# from benchmark_utils import get_mesh_stats, compute_all_metrics
# We'll still call get_mesh_stats if available; otherwise implement fallback.
try:
    from benchmark_utils import get_mesh_stats, compute_all_metrics
    HAVE_BENCH_UTILS = True
except Exception:
    HAVE_BENCH_UTILS = False

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).parent.resolve()
WRAPPINGNET_PATH = SCRIPT_DIR.parent

DATA_DIR = WRAPPINGNET_PATH / "datasets" / "Manifold40"
RESULTS_DIR = WRAPPINGNET_PATH / "results"
RESULT_FILE = RESULTS_DIR / "draco_benchmark_manifold40_extended.csv"

QUANTIZATION_LEVELS = [2, 4, 6, 8, 10, 12, 14, 16]
N_METRIC_POINTS = 20000  # points sampled for distance-based metrics
# ---------------------

# ---------------------
# Helper functions
# ---------------------
def safe_get_mesh_stats(mesh):
    """Fallback if benchmark_utils.get_mesh_stats not available."""
    if HAVE_BENCH_UTILS:
        try:
            return get_mesh_stats(mesh)
        except Exception:
            pass
    stats = {
        "vertices": int(len(mesh.vertices)) if hasattr(mesh, "vertices") else 0,
        "faces": int(len(mesh.faces)) if hasattr(mesh, "faces") else 0,
    }
    return stats

def sample_points_on_mesh(mesh, n):
    """
    Uniform random sampling of points on the mesh surface using trimesh.sample.sample_surface.
    Returns (points, face_indices)
    """
    # trimesh.sample.sample_surface returns (points, face_index)
    try:
        pts, face_idx = trimesh.sample.sample_surface(mesh, n)
        return np.asarray(pts, dtype=np.float64), np.asarray(face_idx, dtype=np.int64)
    except Exception:
        # fallback: sample vertices (biased but safe)
        verts = np.asarray(mesh.vertices)
        if len(verts) == 0:
            return np.zeros((0, 3)), np.zeros((0,), dtype=np.int64)
        idx = np.random.choice(len(verts), size=n, replace=True)
        return verts[idx], np.zeros((len(idx),), dtype=np.int64)

def closest_point_distances(target_mesh, query_points):
    """
    For each query point, compute distance to target_mesh surface using trimesh.proximity.closest_point.
    Returns distances (float array) and closest_points array.
    """
    try:
        closest, distances, tri_id = trimesh.proximity.closest_point(target_mesh, query_points)
        return np.asarray(distances, dtype=np.float64), np.asarray(closest, dtype=np.float64)
    except Exception as e:
        # fallback: brute-force to vertices (slow)
        verts = np.asarray(target_mesh.vertices)
        if verts.size == 0:
            return np.full(len(query_points), np.nan), np.zeros((len(query_points), 3))
        # use KDTree if available
        if SCIPY_AVAILABLE:
            tree = KDTree(verts)
            d, idx = tree.query(query_points, k=1)
            return d, verts[idx]
        else:
            # Numpy brute force
            dists = []
            closest = []
            for p in query_points:
                dif = verts - p.reshape(1, 3)
                dist2 = np.sum(dif * dif, axis=1)
                i = np.argmin(dist2)
                dists.append(math.sqrt(dist2[i]))
                closest.append(verts[i])
            return np.array(dists, dtype=np.float64), np.array(closest, dtype=np.float64)

def signed_distances(target_mesh, query_points):
    """
    Estimate signed distance: distance * sign where sign is negative if point is inside mesh.
    Uses mesh.contains (ray casting) to test inside/outside.
    """
    distances, _ = closest_point_distances(target_mesh, query_points)
    try:
        inside_mask = target_mesh.contains(query_points)
    except Exception:
        # If contains fails, conservatively assume outside
        inside_mask = np.zeros(len(query_points), dtype=bool)
    signed = distances.copy()
    signed[inside_mask] *= -1.0
    return signed

def compute_basic_geometric_metrics(original_mesh, reconstructed_mesh, n_points=N_METRIC_POINTS):
    """
    Compute point-sampling based metrics:
    - Chamfer (mean of distances both ways)
    - Hausdorff (max of directed distances)
    - RMS (directed)
    - Quantiles
    - Signed distance stats (original->reconstructed)
    """
    # Sample points on original surface
    pts_orig, _ = sample_points_on_mesh(original_mesh, n_points)
    pts_rec, _ = sample_points_on_mesh(reconstructed_mesh, n_points)

    # Directed distances: original -> reconstructed (for distortion measures)
    d_orig_to_rec, _ = closest_point_distances(reconstructed_mesh, pts_orig)
    d_rec_to_orig, _ = closest_point_distances(original_mesh, pts_rec)

    chamfer = 0.5 * (np.mean(d_orig_to_rec) + np.mean(d_rec_to_orig))
    hausdorff = max(np.nanmax(d_orig_to_rec), np.nanmax(d_rec_to_orig))

    rms_orig_to_rec = math.sqrt(np.nanmean(np.square(d_orig_to_rec)))
    rms_rec_to_orig = math.sqrt(np.nanmean(np.square(d_rec_to_orig)))

    # quantiles for original->reconstructed
    q50 = float(np.nanpercentile(d_orig_to_rec, 50))
    q90 = float(np.nanpercentile(d_orig_to_rec, 90))
    q95 = float(np.nanpercentile(d_orig_to_rec, 95))
    q99 = float(np.nanpercentile(d_orig_to_rec, 99))

    # signed distances (original surface points to reconstructed)
    signed = signed_distances(reconstructed_mesh, pts_orig)
    signed_stats = {
        "signed_mean": float(np.nanmean(signed)),
        "signed_std": float(np.nanstd(signed)),
        "signed_min": float(np.nanmin(signed)),
        "signed_max": float(np.nanmax(signed)),
    }

    return {
        "chamfer": float(chamfer),
        "hausdorff": float(hausdorff),
        "rms_orig_to_rec": float(rms_orig_to_rec),
        "rms_rec_to_orig": float(rms_rec_to_orig),
        "quantile_p50": q50,
        "quantile_p90": q90,
        "quantile_p95": q95,
        "quantile_p99": q99,
        **signed_stats
    }

def vertex_normal_deviation_stats(original_mesh, reconstructed_mesh):
    """
    For each reconstructed vertex, find nearest original vertex and compute angle between normals.
    Returns mean/median/max in degrees.
    """
    rec_verts = np.asarray(reconstructed_mesh.vertices)
    rec_vn = np.asarray(reconstructed_mesh.vertex_normals)
    orig_verts = np.asarray(original_mesh.vertices)
    orig_vn = np.asarray(original_mesh.vertex_normals)

    if len(rec_verts) == 0 or len(orig_verts) == 0:
        return {"normal_mean_deg": np.nan, "normal_median_deg": np.nan, "normal_max_deg": np.nan}

    # find nearest original vertex for each reconstructed vertex
    if SCIPY_AVAILABLE:
        tree = KDTree(orig_verts)
        _, idx = tree.query(rec_verts, k=1)
        matched_normals = orig_vn[idx]
    else:
        # naive nearest
        idx = []
        for v in rec_verts:
            d = np.sum((orig_verts - v.reshape(1, 3))**2, axis=1)
            idx.append(int(np.argmin(d)))
        matched_normals = orig_vn[idx]

    # compute angle
    dot = np.sum(matched_normals * rec_vn, axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    angles = np.degrees(np.arccos(dot))  # degrees
    return {
        "normal_mean_deg": float(np.nanmean(angles)),
        "normal_median_deg": float(np.nanmedian(angles)),
        "normal_max_deg": float(np.nanmax(angles))
    }

def edge_length_statistics(original_mesh, reconstructed_mesh):
    """
    Compute edge-length arrays and statistics, then return ratios of means/max/rms.
    """
    def edge_lengths(mesh):
        # mesh.edges_unique_length exists in trimesh
        try:
            return np.asarray(mesh.edges_unique_length, dtype=np.float64)
        except Exception:
            # fallback: compute from unique edges manually
            edges = mesh.edges_unique
            v = np.asarray(mesh.vertices)
            lengths = np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)
            return lengths

    orig_lengths = edge_lengths(original_mesh)
    rec_lengths = edge_lengths(reconstructed_mesh)

    if len(orig_lengths) == 0 or len(rec_lengths) == 0:
        return {
            "edge_mean_ratio": np.nan,
            "edge_max_ratio": np.nan,
            "edge_rms_ratio": np.nan
        }

    # Compare central tendencies
    orig_mean = float(np.mean(orig_lengths))
    rec_mean = float(np.mean(rec_lengths))
    orig_max = float(np.max(orig_lengths))
    rec_max = float(np.max(rec_lengths))
    orig_rms = float(math.sqrt(np.mean(orig_lengths**2)))
    rec_rms = float(math.sqrt(np.mean(rec_lengths**2)))

    return {
        "edge_mean_ratio": rec_mean / orig_mean if orig_mean != 0 else np.nan,
        "edge_max_ratio": rec_max / orig_max if orig_max != 0 else np.nan,
        "edge_rms_ratio": rec_rms / orig_rms if orig_rms != 0 else np.nan
    }

def psnr_geometry(original_mesh, reconstructed_mesh):
    """
    Compute a geometry PSNR: use RMS error and bbox diagonal as reference 'peak'.
    PSNR = 20 * log10(peak / RMS)
    peak = bbox_diagonal
    """
    bbox = np.asarray(original_mesh.bounding_box.extents)
    peak = float(np.linalg.norm(bbox))
    # Use chamfer RMS (sqrt of mean squared directed distances original->reconstructed)
    basic = compute_basic_geometric_metrics(original_mesh, reconstructed_mesh, n_points=min(5000, N_METRIC_POINTS))
    rms = basic.get("rms_orig_to_rec", np.nan)
    if rms == 0 or np.isnan(rms) or peak == 0:
        return {"psnr_db": np.nan}
    psnr = 20.0 * math.log10(peak / rms)
    return {"psnr_db": float(psnr)}

def compute_topology_and_validity(original_mesh, reconstructed_mesh):
    """
    Compute Euler characteristic, connected components, boundary loops,
    degenerate faces, and a heuristic for flipped faces / self-intersection.
    """
    def euler_char(mesh):
        V = int(len(mesh.vertices))
        F = int(len(mesh.faces))
        # compute unique edges count
        try:
            E = int(len(mesh.edges_unique))
        except Exception:
            # fallback
            edges = mesh.edges_unique
            E = int(len(edges)) if hasattr(edges, "__len__") else 0
        return V - E + F

    # original
    orig_euler = euler_char(original_mesh)
    rec_euler = euler_char(reconstructed_mesh)

    # connected components
    try:
        orig_components = len(original_mesh.split(only_watertight=False))
    except Exception:
        # fallback: attempt graph-based components
        try:
            orig_components = original_mesh.split().shape[0]
        except Exception:
            orig_components = 1
    try:
        rec_components = len(reconstructed_mesh.split(only_watertight=False))
    except Exception:
        rec_components = 1

    # degenerate faces (zero area)
    try:
        face_areas_rec = reconstructed_mesh.area_faces
        degenerate_faces = int(np.sum(face_areas_rec <= 1e-12))
    except Exception:
        degenerate_faces = 0

    # boundary loop count (using boundary edges)
    try:
        b_edges = reconstructed_mesh.edges_boundary
        if b_edges is None or len(b_edges) == 0:
            boundary_loop_count = 0
        else:
            # build adjacency to count connected components on boundary edges
            from collections import defaultdict, deque
            g = defaultdict(list)
            for e in b_edges:
                a, b = int(e[0]), int(e[1])
                g[a].append(b)
                g[b].append(a)
            visited = set()
            loops = 0
            for v in g.keys():
                if v in visited:
                    continue
                # BFS
                queue = deque([v])
                visited.add(v)
                while queue:
                    u = queue.popleft()
                    for w in g[u]:
                        if w not in visited:
                            visited.add(w)
                            queue.append(w)
                loops += 1
            boundary_loop_count = loops
    except Exception:
        boundary_loop_count = 0

    # flipped-face heuristic: compare face normal sign via nearest face centroid matching
    try:
        # compute centroids and normals
        rec_face_centroids = reconstructed_mesh.triangles_center
        rec_face_normals = reconstructed_mesh.face_normals
        # find nearest original face centroid for each reconstructed face centroid
        orig_face_centroids = original_mesh.triangles_center
        if len(orig_face_centroids) == 0:
            flipped_estimate = 0
        else:
            if SCIPY_AVAILABLE:
                tree = KDTree(orig_face_centroids)
                _, idx = tree.query(rec_face_centroids, k=1)
                matched_orig_normals = original_mesh.face_normals[idx]
            else:
                matched_orig_normals = []
                for c in rec_face_centroids:
                    d = np.sum((orig_face_centroids - c.reshape(1, 3))**2, axis=1)
                    matched_orig_normals.append(original_mesh.face_normals[int(np.argmin(d))])
                matched_orig_normals = np.array(matched_orig_normals)
            dot = np.sum(rec_face_normals * matched_orig_normals, axis=1)
            flipped = np.sum(dot < 0)
            flipped_estimate = int(flipped)
    except Exception:
        flipped_estimate = 0

    # self-intersection heuristic: use winding consistency & watertightness
    try:
        is_watertight = bool(reconstructed_mesh.is_watertight)
    except Exception:
        is_watertight = False
    try:
        winding_ok = bool(reconstructed_mesh.is_winding_consistent)
    except Exception:
        winding_ok = False
    # if not watertight or inconsistent winding, flag possible self-intersection
    self_intersection_flag = (not is_watertight) or (not winding_ok)

    return {
        "orig_euler": int(orig_euler),
        "rec_euler": int(rec_euler),
        "euler_change": int(rec_euler - orig_euler),
        "orig_components": int(orig_components),
        "rec_components": int(rec_components),
        "degenerate_faces": int(degenerate_faces),
        "boundary_loop_count": int(boundary_loop_count),
        "flipped_face_estimate": int(flipped_estimate),
        "self_intersection_flag": bool(self_intersection_flag),
        "is_watertight": bool(is_watertight),
        "winding_consistent": bool(winding_ok)
    }

def memory_rss():
    """Return current process RSS in bytes if psutil available, else np.nan"""
    if PSUTIL_AVAILABLE:
        p = psutil.Process(os.getpid())
        return int(p.memory_info().rss)
    return np.nan

def compute_reencode_drift(decompressed_mesh, qp):
    """
    Re-encode the decompressed mesh and decode to obtain a second decompressed mesh.
    Compute chamfer (and a few other metrics) between first and second decompressed meshes.
    Returns dict with chamfer_drift, hausdorff_drift, bytes_second, bpv_second
    """
    try:
        verts = np.asarray(decompressed_mesh.vertices, dtype=np.float32)
        faces = np.asarray(decompressed_mesh.faces, dtype=np.int32)
        buf2 = draco.encode_mesh_to_buffer(verts, faces, quantization_bits=qp)
        if buf2 is None or len(buf2) == 0:
            return {"reencode_success": False}
        decompressed2 = draco.decode_buffer_to_mesh(buf2)
        mesh2 = trimesh.Trimesh(vertices=decompressed2.points, faces=decompressed2.faces)
        # compute chamfer between mesh1 and mesh2 (using smaller sample to save time)
        metrics = compute_basic_geometric_metrics(decompressed_mesh, mesh2, n_points=min(5000, N_METRIC_POINTS))
        return {
            "reencode_success": True,
            "reencoded_size_bytes": len(buf2),
            "chamfer_drift": float(metrics["chamfer"]),
            "hausdorff_drift": float(metrics["hausdorff"]),
            "rms_drift": float(metrics["rms_orig_to_rec"])
        }
    except Exception:
        return {"reencode_success": False}

# ---------------------
# Main per-mesh worker
# ---------------------
def process_mesh(mesh_path):
    mesh_results = []

    try:
        rel_path = str(Path(mesh_path).relative_to(DATA_DIR))
    except Exception:
        rel_path = Path(mesh_path).name

    try:
        original_mesh = trimesh.load_mesh(mesh_path, process=False)
        if original_mesh is None or not hasattr(original_mesh, "vertices"):
            return []
        if not hasattr(original_mesh, "faces") or len(original_mesh.faces) == 0:
            return []

        stats = safe_get_mesh_stats(original_mesh)
        original_num_vertices = int(stats["vertices"])
        if original_num_vertices == 0:
            return []

        vertices = np.asarray(original_mesh.vertices, dtype=np.float32)
        faces = np.asarray(original_mesh.faces, dtype=np.int32)

    except Exception as e:
        print(f"Error loading {rel_path}: {e}")
        return []

    # Precompute a small sample for normals if needed
    # Main loop over quantization levels
    for qp in QUANTIZATION_LEVELS:
        row_base = {
            "mesh_path": rel_path,
            "q_level": int(qp),
            "original_vertices": original_num_vertices,
            "original_faces": int(stats.get("faces", len(original_mesh.faces))),
        }

        # -- ENCODE --
        encode_start = time.perf_counter()
        rss_before_encode = memory_rss()
        encode_failed = False
        encoded_buf = None
        try:
            encoded_buf = draco.encode_mesh_to_buffer(vertices, faces, quantization_bits=qp)
        except Exception as e:
            encode_failed = True
            print(f"Draco encoding error for {rel_path} QP={qp}: {e}")
        encode_time = time.perf_counter() - encode_start
        rss_after_encode = memory_rss()
        encode_rss_delta = (rss_after_encode - rss_before_encode) if (PSUTIL_AVAILABLE and not math.isnan(rss_before_encode)) else np.nan

        if encode_failed or encoded_buf is None or len(encoded_buf) == 0:
            # record failure row (no decode / metrics)
            mesh_results.append({
                **row_base,
                "compressed_size_bytes": int(len(encoded_buf)) if encoded_buf else 0,
                "bpv": float((len(encoded_buf) * 8) / original_num_vertices) if (encoded_buf and original_num_vertices > 0) else np.nan,
                "encode_time_sec": float(encode_time),
                "encode_rss_delta_bytes": int(encode_rss_delta) if not np.isnan(encode_rss_delta) else np.nan,
                "encode_failed": True,
                "decode_failed": None
            })
            continue

        compressed_size_bytes = int(len(encoded_buf))
        bpv = (compressed_size_bytes * 8) / original_num_vertices if original_num_vertices > 0 else np.nan

        # -- DECODE --
        decode_start = time.perf_counter()
        rss_before_decode = memory_rss()
        decode_failed = False
        decompressed_mesh = None
        try:
            decompressed_mesh_data = draco.decode_buffer_to_mesh(encoded_buf)
            decompressed_mesh = trimesh.Trimesh(vertices=decompressed_mesh_data.points, faces=decompressed_mesh_data.faces, process=False)
        except Exception as e:
            decode_failed = True
            print(f"Draco decoding error for {rel_path} QP={qp}: {e}")
        decode_time = time.perf_counter() - decode_start
        rss_after_decode = memory_rss()
        decode_rss_delta = (rss_after_decode - rss_before_decode) if (PSUTIL_AVAILABLE and not math.isnan(rss_before_decode)) else np.nan

        if decode_failed or decompressed_mesh is None or len(decompressed_mesh.vertices) == 0:
            mesh_results.append({
                **row_base,
                "compressed_size_bytes": compressed_size_bytes,
                "bpv": float(bpv),
                "encode_time_sec": float(encode_time),
                "encode_rss_delta_bytes": int(encode_rss_delta) if not np.isnan(encode_rss_delta) else np.nan,
                "decode_time_sec": float(decode_time),
                "decode_rss_delta_bytes": int(decode_rss_delta) if not np.isnan(decode_rss_delta) else np.nan,
                "encode_failed": False,
                "decode_failed": True
            })
            continue

        # Ensure meshes have consistent winding/normals for comparisons
        try:
            original_mesh.remove_duplicate_faces()
            original_mesh.remove_degenerate_faces()
        except Exception:
            pass
        try:
            decompressed_mesh.remove_duplicate_faces()
            decompressed_mesh.remove_degenerate_faces()
        except Exception:
            pass

        # -- METRICS --
        # 1) Basic geometric / distortion metrics
        try:
            basic_metrics = compute_basic_geometric_metrics(original_mesh, decompressed_mesh, n_points=N_METRIC_POINTS)
        except Exception as e:
            print(f"Metric computation error (basic) for {rel_path} QP={qp}: {e}")
            basic_metrics = {}

        # 2) Vertex normal deviation
        try:
            normal_stats = vertex_normal_deviation_stats(original_mesh, decompressed_mesh)
        except Exception as e:
            print(f"Normal deviation error for {rel_path} QP={qp}: {e}")
            normal_stats = {"normal_mean_deg": np.nan, "normal_median_deg": np.nan, "normal_max_deg": np.nan}

        # 3) Edge length distortion
        try:
            edge_stats = edge_length_statistics(original_mesh, decompressed_mesh)
        except Exception as e:
            print(f"Edge length stats error for {rel_path} QP={qp}: {e}")
            edge_stats = {"edge_mean_ratio": np.nan, "edge_max_ratio": np.nan, "edge_rms_ratio": np.nan}

        # 4) PSNR geometry
        try:
            psnr = psnr_geometry(original_mesh, decompressed_mesh)
        except Exception:
            psnr = {"psnr_db": np.nan}

        # 5) Topology & validity
        try:
            topo = compute_topology_and_validity(original_mesh, decompressed_mesh)
        except Exception as e:
            print(f"Topology computation error for {rel_path} QP={qp}: {e}")
            topo = {}

        # 6) Throughput
        encode_throughput = original_num_vertices / encode_time if encode_time > 0 else np.nan
        decode_throughput = original_num_vertices / decode_time if decode_time > 0 else np.nan

        # 7) Re-encode drift (re-encode decompressed and compare)
        try:
            reencode = compute_reencode_drift(decompressed_mesh, qp)
        except Exception as e:
            print(f"Reencode drift error for {rel_path} QP={qp}: {e}")
            reencode = {"reencode_success": False}

        # 8) Additional flags
        decode_valid_mesh = True
        try:
            if len(decompressed_mesh.faces) == 0 or len(decompressed_mesh.vertices) == 0:
                decode_valid_mesh = False
        except Exception:
            decode_valid_mesh = False

        # Build result row
        result = {
            **row_base,
            "compressed_size_bytes": compressed_size_bytes,
            "bpv": float(bpv),
            "encode_time_sec": float(encode_time),
            "encode_rss_delta_bytes": int(encode_rss_delta) if not np.isnan(encode_rss_delta) else np.nan,
            "decode_time_sec": float(decode_time),
            "decode_rss_delta_bytes": int(decode_rss_delta) if not np.isnan(decode_rss_delta) else np.nan,
            "encode_throughput_vtx_per_sec": float(encode_throughput) if not np.isnan(encode_throughput) else np.nan,
            "decode_throughput_vtx_per_sec": float(decode_throughput) if not np.isnan(decode_throughput) else np.nan,
            "encode_failed": False,
            "decode_failed": False,
            "decode_valid_mesh": bool(decode_valid_mesh),
            # basic metrics
            "chamfer_distance": float(basic_metrics.get("chamfer", np.nan)),
            "hausdorff_distance": float(basic_metrics.get("hausdorff", np.nan)),
            "rms_orig_to_rec": float(basic_metrics.get("rms_orig_to_rec", np.nan)),
            "rms_rec_to_orig": float(basic_metrics.get("rms_rec_to_orig", np.nan)),
            "quantile_p50": float(basic_metrics.get("quantile_p50", np.nan)),
            "quantile_p90": float(basic_metrics.get("quantile_p90", np.nan)),
            "quantile_p95": float(basic_metrics.get("quantile_p95", np.nan)),
            "quantile_p99": float(basic_metrics.get("quantile_p99", np.nan)),
            "signed_mean": float(basic_metrics.get("signed_mean", np.nan)),
            "signed_std": float(basic_metrics.get("signed_std", np.nan)),
            "signed_min": float(basic_metrics.get("signed_min", np.nan)),
            "signed_max": float(basic_metrics.get("signed_max", np.nan)),
            # normals
            "normal_mean_deg": float(normal_stats.get("normal_mean_deg", np.nan)),
            "normal_median_deg": float(normal_stats.get("normal_median_deg", np.nan)),
            "normal_max_deg": float(normal_stats.get("normal_max_deg", np.nan)),
            # edges
            "edge_mean_ratio": float(edge_stats.get("edge_mean_ratio", np.nan)),
            "edge_max_ratio": float(edge_stats.get("edge_max_ratio", np.nan)),
            "edge_rms_ratio": float(edge_stats.get("edge_rms_ratio", np.nan)),
            # psnr
            "psnr_db": float(psnr.get("psnr_db", np.nan)),
            # topology
            "orig_euler": int(topo.get("orig_euler", np.nan)) if topo.get("orig_euler", None) is not None else np.nan,
            "rec_euler": int(topo.get("rec_euler", np.nan)) if topo.get("rec_euler", None) is not None else np.nan,
            "euler_change": int(topo.get("euler_change", np.nan)) if topo.get("euler_change", None) is not None else np.nan,
            "orig_components": int(topo.get("orig_components", np.nan)) if topo.get("orig_components", None) is not None else np.nan,
            "rec_components": int(topo.get("rec_components", np.nan)) if topo.get("rec_components", None) is not None else np.nan,
            "degenerate_faces": int(topo.get("degenerate_faces", 0)),
            "boundary_loop_count": int(topo.get("boundary_loop_count", 0)),
            "flipped_face_estimate": int(topo.get("flipped_face_estimate", 0)),
            "self_intersection_flag": bool(topo.get("self_intersection_flag", False)),
            "is_watertight": bool(topo.get("is_watertight", False)),
            "winding_consistent": bool(topo.get("winding_consistent", False)),
            # reencode drift
            "reencode_success": bool(reencode.get("reencode_success", False)),
            "reencoded_size_bytes": int(reencode.get("reencoded_size_bytes", 0)) if reencode.get("reencode_success", False) else np.nan,
            "chamfer_drift": float(reencode.get("chamfer_drift", np.nan)),
            "hausdorff_drift": float(reencode.get("hausdorff_drift", np.nan)),
            "rms_drift": float(reencode.get("rms_drift", np.nan)),
        }

        mesh_results.append(result)

    return mesh_results

# ---------------------
# Main runner
# ---------------------
def run_benchmark():
    print(f"Starting Draco benchmark (extended)...")
    print(f"Dataset root: {DATA_DIR}")
    print(f"Results will be saved in: {RESULT_FILE}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    search_path_obj = os.path.join(DATA_DIR, "raw", "*", "test", "**", "*.obj")
    search_path_ply = os.path.join(DATA_DIR, "raw", "*", "test", "**", "*.ply")

    print("Scanning for files...")
    mesh_files = glob.glob(search_path_obj, recursive=True) + glob.glob(search_path_ply, recursive=True)
    mesh_files = sorted(mesh_files)

    if not mesh_files:
        print(f"No meshes found in {DATA_DIR}/raw/*/test/")
        print("Please check directory structure.")
        return

    print(f"Found {len(mesh_files)} meshes. Starting parallel processing...")
    all_results = []

    max_workers = max(1, min(os.cpu_count() or 1, 8))  # sensible default, don't spawn too many processes

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results_iterator = list(tqdm(executor.map(process_mesh, mesh_files), total=len(mesh_files), desc="Processing"))
        for mesh_result_list in results_iterator:
            if mesh_result_list:
                all_results.extend(mesh_result_list)

    if not all_results:
        print("No results were generated. Check for errors.")
        return

    # Save to CSV
    df = pd.DataFrame(all_results)
    # Sort for readability
    df = df.sort_values(by=["mesh_path", "q_level"])
    df.to_csv(RESULT_FILE, index=False)
    print(f"\nBenchmark completed. Results saved in {RESULT_FILE}")

if __name__ == "__main__":
    # Warn if optional libraries missing
    if not SCIPY_AVAILABLE:
        warnings.warn("scipy.spatial.cKDTree not available — nearest-neighbor lookups will be slower.")
    if not PSUTIL_AVAILABLE:
        warnings.warn("psutil not available — memory metrics will be NaN.")
    run_benchmark()
