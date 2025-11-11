# from meshcompression.dataset.modelnet import load_random_model

from meshcompression.dataset.toys4k import load_random_model
from meshcompression.render import (
    render_from_file,
    render_n_random_watertight_objaverse_models,
    render_random_objaverse_model,
    render_random_watertight_objaverse_model,
)
from meshcompression.constants import ASSET_DIR


def main() -> None:
    # bunny_example()
    # render_random_objaverse_model()
    # render_random_watertight_objaverse_model()
    # render_n_random_watertight_objaverse_models(3)
    # load_random_model()
    load_random_model()


def bunny_example() -> None:
    mesh_file = ASSET_DIR / "bunny.obj"
    render_from_file(mesh_file)


if __name__ == "__main__":
    main()
