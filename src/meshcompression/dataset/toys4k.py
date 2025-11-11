import numpy as np
import os
import random
import trimesh
from pathlib import Path

from meshcompression.constants import TOYS4K_DIR


def load_random_model():
    """Load a random model from the Toys4k dataset."""
    # choose random directory in toys4k
    random_dir = random.choice(os.listdir(TOYS4K_DIR))
    random_dir_deeper = random.choice(os.listdir(os.path.join(TOYS4K_DIR, random_dir)))
    random_dir_path = Path(os.path.join(TOYS4K_DIR, random_dir, random_dir_deeper))

    # load random object file from that directory using trimesh
    model_files = list(random_dir_path.glob("*.obj"))
    if not model_files:
        raise FileNotFoundError(f"No .obj model files found in {random_dir_path}")

    mesh = trimesh.load(random.choice(model_files))
    mesh.show()
