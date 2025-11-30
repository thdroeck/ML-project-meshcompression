from argparse import ArgumentParser

import os
from WrappingNet.wrappingnet.dataloaders import preprocess_mesh
import trimesh
import torch
from tqdm import tqdm
from torch_geometric.data import Data
from WrappingNet.wrappingnet.models import Autoencoder
from WrappingNet.wrappingnet import losses


def train(args):
    """Training function to loop over all .obj files in the dataset path and train the model."""

    print(
        f"Training with batch size {args.batch_size}, lr {args.lr}, epochs {args.epochs}, latent dim {args.latent_dim}, data root {args.data_root}"
    )
    # Training code would go here

    model = Autoencoder(input_dim=7, feature_dim=args.latent_dim, num_loop=3, print_debug=False)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    data = []
    dataset_counter = 0
    for folder in os.listdir(args.data_root): # each folder is a catagory
        train_path = os.path.join(args.data_root, folder, "train") # path to train folder
        if os.path.exists(train_path) and dataset_counter < args.max_datasets: # check if train folder exists
            print(f"Loading data from {train_path}") # log
            for file in os.listdir(train_path):
                if file.endswith(".obj"):
                    mesh = trimesh.load(os.path.join(train_path, file))
                    pos = torch.tensor(mesh.vertices, dtype=torch.float32)
                    face = torch.tensor(mesh.faces, dtype=torch.long)
                    data.append(preprocess_mesh(Data(pos=pos, face=face.T)))
            dataset_counter += 1

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for i in tqdm(range(len(data))):
            pos = data[i].pos
            face = data[i].face
            pos_list, face_list, _ = model(
                pos, face.T
            )  # self.model(data.pos, data.face.T, pos_base)
            rate = torch.tensor(0.0)

            distortion_loss = losses.get_distortion_loss(args.loss_function)(
                pos_list, face_list, pos, face.T
            )
            chamfer_loss = losses.chamfer(pos_list[-1], pos)  # unused for now
            loss = rate + args.lmbda * distortion_loss
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            model.zero_grad()

        epoch_loss /= len(data)
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {epoch_loss:.6f}")

    if not os.path.exists("trained/"):
        os.makedirs("trained/")
    torch.save(
        model.state_dict(),
        f"trained/TRAIN_MeshAE_{args.loss_function}_{args.data_root}_d{args.latent_dim}_e{args.epochs}.ckpt",
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument(
        "--epochs", type=int, required=True, help="Number of training epochs"
    )
    parser.add_argument(
        "--latent_dim", type=int, required=True, help="bottleneck dimension"
    )
    parser.add_argument("--data_root", type=str, required=True, help="data root")
    parser.add_argument(
        "--loss_function", type=str, default="MSL2", help="Loss function"
    )
    parser.add_argument(
        "--lmbda", type=float, default=1.0, help="Weight for distortion loss"
    )
    parser.add_argument(
        "--max_datasets", type=int, default=float('inf'), help="Maximum number of datasets to use"
    )
    args = parser.parse_args()

    train(args)
