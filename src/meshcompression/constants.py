from pathlib import Path

ASSET_DIR: Path = Path(__file__).parent.parent.parent / "assets"
LIB_DIR: Path = Path(__file__).parent.parent.parent.parent / "lib"
SHAPENET_DIR: Path = (
    LIB_DIR / "lowshot-shapebias" / "data" / "ShapeNet55-LS" / "pointclouds"
)
MODELNET_DIR: Path = (
    LIB_DIR / "lowshot-shapebias" / "data" / "ModelNet40-LS" / "pointclouds"
)
TOYS4K_DIR: Path = LIB_DIR / "toys4k" / "toys4k_obj_files"
