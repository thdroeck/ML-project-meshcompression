from argparse import ArgumentParser
import os
from WrappingNet.wrappingnet import utils
from WrappingNet.wrappingnet.losses import chamfer  
from WrappingNet.wrappingnet.dataloaders import manifold40_dset
from WrappingNet.wrappingnet.models import (
    WrappingNet_sphere_LC,
    WrappingNet_global_basesup3,
    Autoencoder,
)

import torch
import trimesh
import numpy as np

from torch_geometric.data import Data


def get_method(args):
    if args.method == "visualize":
        return visualize
    elif args.method == "performance":
        return performance
    else:
        raise ValueError(f"Unknown method: {args.method}")

def visualize(args):
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

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

    # choose random folder from dataset_path
    # folder = np.random.choice(os.listdir(dataset_path))
    # test_path = os.path.join(dataset_path, folder, "test")  # path to test folder
    # choose random .obj file from test_path
    # mesh_file = np.random.choice(os.listdir(test_path))
    # mesh = trimesh.load(os.path.join(test_path, mesh_file))
    mesh = trimesh.load(args.dataset_path)
    pos = torch.tensor(mesh.vertices, dtype=torch.float32)
    face = torch.tensor(mesh.faces, dtype=torch.long)
    mesh = Data(pos=pos, face=face.T)

    # load mesh in trimesh before we change it with the model
    mesh_trimesh1 = trimesh.Trimesh(
        vertices=mesh.pos.cpu().numpy(), faces=mesh.face.T.cpu().numpy()
    )

    pos_list, face_list, _ = model(mesh.pos, mesh.face.T)

    print("Evaluation completed.")
    print(f"Final output vertices: {pos_list[-1].shape}")
    print(f"Final output faces: {face_list[-1].shape}")

    mesh_trimesh2 = trimesh.Trimesh(
        vertices=pos_list[-1].detach().numpy(), faces=face_list[-1].detach().numpy()
    )

    bbox_size = mesh_trimesh1.bounds[1] - mesh_trimesh1.bounds[0]
    offset = np.array([bbox_size[0] * 2, 0, 0])  # 20% spacing

    # Apply translation to mesh2
    mesh2_shifted = mesh_trimesh2.copy()
    mesh2_shifted.apply_translation(offset)

    # Create scene with both
    scene = trimesh.Scene([mesh_trimesh1, mesh2_shifted])
    scene.export("output_visualization.glb")
    print("Scene saved to output_visualization.glb")

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

    # loop over all folders in dataset_path
    for folder in os.listdir(dataset_path):
        test_path = os.path.join(dataset_path, folder, "test")  # path to test folder
        if os.path.exists(test_path):  # check if test folder exists
            print(f"Evaluating data from {test_path}")  # log
            total_chamfer = 0.0
            total_meshes = 0
            for mesh_file in os.listdir(test_path):
                mesh = trimesh.load(os.path.join(test_path, mesh_file))
                pos = torch.tensor(mesh.vertices, dtype=torch.float32)
                face = torch.tensor(mesh.faces, dtype=torch.long)
                mesh = Data(pos=pos, face=face.T)
                pos_list, face_list, _ = model(mesh.pos, mesh.face.T)
                chamfer_loss = chamfer(pos_list[-1], mesh.pos)
                total_chamfer += chamfer_loss.item()
                total_meshes += 1
            avg_chamfer = total_chamfer / total_meshes
            print(f"Average Chamfer Distance for {folder}: {avg_chamfer:.6f}")

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

    args = parser.parse_args()

    method = get_method(args)
    method(args)