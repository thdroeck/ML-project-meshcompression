from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from WrappingNet.wrappingnet import utils
from WrappingNet.wrappingnet.losses import chamfer  
from WrappingNet.wrappingnet.dataloaders import manifold40, manifold40_dset, preprocess_mesh
from WrappingNet.wrappingnet.models import (
    ExtendedAutoencoder,
    ExtendedDecoder,
    SimpleAutoencoder,
    WrappingNet_sphere_LC,
    WrappingNet_global_basesup3,
    Autoencoder,
)

import torch
import trimesh
import numpy as np

from torch_geometric.data import Data


def get_model(args):
    if args.model == "basic":
        model = Autoencoder(input_dim=7, feature_dim=args.latent_dim, num_loop=3)
    elif args.model == "simple":
        model = SimpleAutoencoder(input_dim=7, feature_dim=args.latent_dim, num_loop=3)
    elif args.model == "extended":
        model = ExtendedAutoencoder(input_dim=7, feature_dim=args.latent_dim, num_loop=3)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    return model

def get_method(args):
    if args.method == "visualize":
        return visualize
    elif args.method == "performance":
        return performance
    elif args.method == "view_manifold_dataloader":
        return view_mandold_dataloader
    else:
        raise ValueError(f"Unknown method: {args.method}")

def render_2_meshes(mesh1, mesh2):
    # load mesh in trimesh before we change it with the model
    mesh_trimesh1 = trimesh.Trimesh(
        vertices=mesh1.pos.cpu().numpy(), faces=mesh1.face.T.cpu().numpy()
    )

    mesh_trimesh2 = trimesh.Trimesh(
        vertices=mesh2.pos.cpu().numpy(), faces=mesh2.face.T.cpu().numpy()
    )

    bbox_size = mesh_trimesh1.bounds[1] - mesh_trimesh1.bounds[0]
    offset = np.array([bbox_size[0] * 2, 0, 0])  # 20% spacing

    # Apply translation to mesh2
    mesh2_shifted = mesh_trimesh2.copy()
    mesh2_shifted.apply_translation(offset)

    # Create scene with both
    scene = trimesh.Scene([mesh_trimesh1, mesh2_shifted])
    scene.show()


def view_mandold_dataloader(args):
    dataset = manifold40_dset(root=f"{args.dataset_path}")
    average_vertices = np.mean([data.pos.shape[0] for data in dataset])
    average_faces = np.mean([data.face.shape[1] for data in dataset])
    print(f"Average number of vertices: {average_vertices}")
    print(f"Average number of faces: {average_faces}")

    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    model_checkpoint = args.model_checkpoint
    dataset_path = args.dataset_path
    # Load data for evaluation
    model = get_model(args)
    saved = torch.load(
        model_checkpoint,
        map_location="cpu",
    )
    model.load_state_dict(saved)
    model.eval()

    # Load mesh data
    dset_test = manifold40_dset(root=dataset_path, train=False)

    iterator = iter(dset_test)
    # for _ in range(200):
    #     next(iterator)
    mesh1 = next(iterator)
    mesh1 = mesh1.to(device)
    print(mesh1)
    print(mesh1.pos)
    print(mesh1.face)
    pos_base, faces_base = utils.get_base_mesh(mesh1.pos, mesh1.face.T)
    pos_list, face_list, _ = model(mesh1.pos, mesh1.face.T)

    mesh2 = Data(pos=pos_list[-1].detach(), face=face_list[-1].detach().T)

    print("Evaluation completed.")
    print(f"Final output vertices: {pos_list[-1].shape}")
    print(f"Final output faces: {face_list[-1].shape}")
    print(pos_list[-1])
    print(face_list[-1])
    print(mesh2)

    render_2_meshes(mesh1, mesh2)

def visualize(args):
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    model_checkpoint = args.model_checkpoint
    dataset_path = args.dataset_path
    # Load data for evaluation
    model = get_model(args)
    saved = torch.load(
        model_checkpoint,
        map_location="cpu",
    )
    model.load_state_dict(saved)
    model.eval()


    # choose random folder from dataset_path
    # folder = np.random.choice(os.listdir(dataset_path))
    # test_path = os.path.join(dataset_path, folder, "test")  # path to test folder
    # choose random .obj file from test_path
    # mesh_file = np.random.choice(os.listdir(test_path))
    # mesh = trimesh.load(os.path.join(test_path, mesh_file))
    mesh1 = trimesh.load(dataset_path)
    new_faces = mesh1.faces
    new_vert = mesh1.vertices
    if (len (new_vert) % 2 == 1):
        new_vert, new_faces = trimesh.remesh.subdivide(new_vert, new_faces, face_index=None, vertex_attributes=None, return_index=False)
    pos = torch.tensor(new_vert, dtype=torch.float32)
    face = torch.tensor(new_faces, dtype=torch.long)
    mesh1 = preprocess_mesh(Data(pos=pos, face=face.T))
    mesh1 = Data(pos=pos, face=face.T)

    pos_list, face_list, _ = model(mesh1.pos, mesh1.face.T)
    mesh2 = Data(pos=pos_list[-1].detach(), face=face_list[-1].detach().T)

    print("Evaluation completed.")
    print(f"Final output vertices: {pos_list[-1].shape}")
    print(f"Final output faces: {face_list[-1].shape}")
    print(pos_list[-1])
    print(face_list[-1])

    render_2_meshes(mesh1, mesh2)


def evaluate_folder(model, dataset_path, folder):
    test_path = os.path.join(dataset_path, folder, "test")

    if not os.path.exists(test_path):
        return folder, None  # no test folder

    total_chamfer = 0.0
    total_meshes = 0

    for mesh_file in os.listdir(test_path):
        mesh_full_path = os.path.join(test_path, mesh_file)
        mesh = trimesh.load(mesh_full_path)

        pos = torch.tensor(mesh.vertices, dtype=torch.float32)
        face = torch.tensor(mesh.faces, dtype=torch.long)
        mesh_data = Data(pos=pos, face=face.T)

        # run model (ensure it's CPU-safe)
        pos_list, face_list, _ = model(mesh_data.pos, mesh_data.face.T)

        chamfer_loss = chamfer(pos_list[-1], mesh_data.pos)
        total_chamfer += chamfer_loss.item()
        total_meshes += 1

    if total_meshes == 0:
        return folder, None
    
    return folder, total_chamfer / total_meshes

def performance(args):
    model_checkpoint = args.model_checkpoint
    dataset_path = args.dataset_path
    # Load data for evaluation
    model = Autoencoder(input_dim=7, feature_dim=args.latent_dim, num_loop=3)
    saved = torch.load(
        model_checkpoint,
        map_location="cpu",
    )
    model.load_state_dict(saved)
    model.eval()



    # # loop over all folders in dataset_path
    # for folder in os.listdir(dataset_path):
    #     test_path = os.path.join(dataset_path, folder, "test")  # path to test folder
    #     if os.path.exists(test_path):  # check if test folder exists
    #         print(f"Evaluating data from {test_path}")  # log
    #         total_chamfer = 0.0
    #         total_meshes = 0
    #         for mesh_file in os.listdir(test_path):
    #             mesh = trimesh.load(os.path.join(test_path, mesh_file))
    #             pos = torch.tensor(mesh.vertices, dtype=torch.float32)
    #             face = torch.tensor(mesh.faces, dtype=torch.long)
    #             mesh = Data(pos=pos, face=face.T)
    #             pos_list, face_list, _ = model(mesh.pos, mesh.face.T)
    #             chamfer_loss = chamfer(pos_list[-1], mesh.pos)
    #             total_chamfer += chamfer_loss.item()
    #             total_meshes += 1
    #         avg_chamfer = total_chamfer / total_meshes
    #         print(f"Average Chamfer Distance for {folder}: {avg_chamfer:.6f}")

    folders = os.listdir(dataset_path)

    print(f"Starting parallel evaluation with {os.cpu_count()} workers...\n")

    tasks = [(model, dataset_path, folder) for folder in folders]

    results = {}
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(evaluate_folder, *t): t[1] for t in tasks}

        for future in as_completed(futures):
            folder = futures[future]
            folder, avg_chamfer = future.result()

            if avg_chamfer is None:
                print(f"[{folder}] No test folder found.")
            else:
                print(f"[{folder}] Average Chamfer Distance: {avg_chamfer:.6f}")

            results[folder] = avg_chamfer

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--method",
        dest="method",
        type=str,
        required=True,
        help="method to run: visualize or performance",
    )
    parser.add_argument(
        "--latent_dim",
        dest="latent_dim",
        type=int,
        required=True,
        help="latent dimension for the model",
    )
    parser.add_argument(
        "--model_checkpoint",
        dest="model_checkpoint",
        type=str,
        required=True,
        help="path to the model checkpoint",
    )

    parser.add_argument(
        "--dataset_path",
        dest="dataset_path",
        type=str,
        required=True,
        help="path to the dataset",
    )

    parser.add_argument(
        "--model",
        dest="model",
        type=str,
        default="basic",
        help="model type: simple, basic or extended",
    )

    args = parser.parse_args()

    method = get_method(args)
    method(args)