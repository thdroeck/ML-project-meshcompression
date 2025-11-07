from pathlib import Path
import trimesh

from meshcompression.render import render_from_file, render_random_objaverse_model
from meshcompression.constants import ASSET_DIR


def main() -> None:
    bunny_example()
    # render_random_objaverse_model()


def bunny_example() -> None:
    mesh_file = ASSET_DIR / "bunny.obj"
    render_from_file(mesh_file)


if __name__ == "__main__":
    main()
