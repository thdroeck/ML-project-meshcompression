from argparse import ArgumentParser
from WrappingNet.wrappingnet import utils
from WrappingNet.wrappingnet.dataloaders import manifold40_dset
from WrappingNet.wrappingnet.models import (
    WrappingNet_sphere_LC,
    WrappingNet_global_basesup3,
    Autoencoder,
)

import torch
import trimesh
import numpy as np


def main(args):
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    model_checkpoint = args.model_checkpoint
    dataset_path = args.dataset_path
    # Load data for evaluation
    model = Autoencoder(input_dim=7, feature_dim=128, num_loop=3)
    saved = torch.load(
        model_checkpoint,
        map_location="cpu",
    )
    model.load_state_dict(saved)
    model.eval()

    # Load mesh data
    dset_test = manifold40_dset(root=dataset_path, train=False)

    iterator = iter(dset_test)
    next(iterator)
    next(iterator)
    mesh = next(iterator)
    mesh = mesh.to(device)

    print(mesh.pos)
    print(mesh.face)

    # load mesh in trimesh before we change it with the model
    mesh_trimesh1 = trimesh.Trimesh(
        vertices=mesh.pos.cpu().numpy(), faces=mesh.face.T.cpu().numpy()
    )

    pos_base, faces_base = utils.get_base_mesh(mesh.pos, mesh.face.T)
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
    scene.show()


if __name__ == "__main__":
    parser = ArgumentParser()

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

    main(args)
