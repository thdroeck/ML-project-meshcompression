############################################################################
### NOT IMPLEMENTED YET - WORK IN PROGRESS
############################################################################


import numpy as np
import os
import random
import trimesh

from meshcompression.constants import MODELNET_DIR


def load_random_model():
    """Load a random model from the ModelNet dataset."""

    # choose random directory in modelnet
    random_dir = random.choice(os.listdir(MODELNET_DIR))

    model_files = list((MODELNET_DIR / random_dir).glob("*.npy"))
    if not model_files:
        raise FileNotFoundError(f"No .npy model files found in {MODELNET_DIR}")

    random_file = random.choice(model_files)
    print(f"Loading random model from file: {random_file}")

    model_data = trimesh.load(random_file)
    # print(f"Model data shape: {model_data.shape}")

    # # Extract vertex pairs
    # V1 = model_data[:, :3]
    # V2 = model_data[:, 3:]

    # vertices = np.vstack((V1, V2))
    # vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)

    # # Build edges (each row connects 2 vertices)
    # edges = np.column_stack((inverse[: len(model_data)], inverse[len(model_data) :]))

    # # Create a Path3D (wireframe)
    # path = trimesh.load_path(vertices[edges])
    model_data.show()
