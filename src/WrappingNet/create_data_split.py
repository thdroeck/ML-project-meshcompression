"""Script to create data splits for training, validation and testing."""

from pathlib import Path

import torch
import trimesh
from argparse import ArgumentParser


def main(args):
    """Main function to create splits, we first check if the dataset is sorted in folders with .obj files.
    after this we load them using trimesh and than we create a large tensor
    to save to 3 .pt files for train, val and test splits.
    """

    # check if dataset path exists
    dataset_path = Path(args.dataset_path)
    if (not dataset_path.exists()) or (not dataset_path.is_dir()):
        raise ValueError(
            f"Dataset path {dataset_path} does not exist or is not a directory."
        )

    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # check if dataset is sorted in folders
    categories = [d for d in dataset_path.iterdir() if d.is_dir()]
    if len(categories) == 0:
        raise ValueError(
            f"Dataset path {dataset_path} does not contain any category folders."
        )

    categories = sorted(categories)
    train_meshes = []
    test_meshes = []
    for category in categories:
        for split in ["train", "test"]:
            split_folder = category / split
            if not split_folder.exists() or not split_folder.is_dir():
                raise ValueError(
                    f"Split folder {split_folder} does not exist or is not a directory."
                )
            mesh_files = list(split_folder.glob("*.obj"))
            for mesh_file in mesh_files:
                mesh = trimesh.load(mesh_file, process=False)
                if split == "train":
                    train_meshes.append(mesh)
                elif split == "test":
                    test_meshes.append(mesh)

    # Save the splits
    torch.save(train_meshes, output_directory / "training.pt")
    torch.save(test_meshes, output_directory / "testing.pt")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--dataset_path",
        dest="dataset_path",
        type=str,
        required=True,
        help="path to the dataset",
    )

    parser.add_argument(
        "--output_directory",
        dest="output_directory",
        type=str,
        required=True,
        help="path to the output directory",
    )

    args = parser.parse_args()

    main(args)
