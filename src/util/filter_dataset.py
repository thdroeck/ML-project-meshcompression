# this file creates executable script that takes a directory as input and filters dataset to only contain watertight meshes

import fire
import os
import trimesh
from pathlib import Path
import shutil


def filter(input_dir: str, output_dir: str) -> None:
    """Filter dataset to only contain watertight meshes.

    Args:
        input_dir (str): Path to input dataset directory.
        output_dir (str): Path to output filtered dataset directory.
    """

    ### first loop over all .obj files recursively in input_dir

    if input_dir is None or output_dir is None:
        raise ValueError("Both input_dir and output_dir must be provided.")

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True)
    for root, _, files in os.walk(input_path):
        for file in files:
            if file.endswith(".obj"):
                file_path = Path(root) / file
                try:
                    mesh = trimesh.load_mesh(file_path)
                    if mesh.is_watertight:
                        # create corresponding output directory
                        relative_path = file_path.relative_to(input_path)
                        output_file_path = output_path / relative_path
                        output_file_path.parent.mkdir(parents=True, exist_ok=True)
                        # copy file to output directory
                        shutil.copy(file_path, output_file_path)
                        print(
                            f"Copied watertight mesh: {file_path} to {output_file_path}"
                        )
                    else:
                        print(f"Skipped non-watertight mesh: {file_path}")
                except Exception as e:
                    print(f"Error loading mesh {file_path}: {e}")


if __name__ == "__main__":
    fire.Fire(filter)
