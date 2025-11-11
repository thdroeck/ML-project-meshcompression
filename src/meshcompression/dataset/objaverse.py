import objaverse.xl as oxl
import pandas as pd
from meshcompression.constants import ASSET_DIR
from typing import Any, Callable, Dict, Hashable, Optional


def get_annotations() -> pd.DataFrame:
    """List available model IDs in the Objaverse dataset."""
    model_ids: pd.DataFrame = oxl.get_alignment_annotations()
    m = model_ids[model_ids["fileType"] == "obj"]
    print(f"Found {len(m)} .obj models in Objaverse dataset.")
    print("Sample model IDs:")
    print(m.sample(n=5))
    return m


def download_model_by_id(model_id: int) -> None:
    """Download a specific model from the Objaverse dataset by its ID."""
    model_ids = get_annotations()
    if model_id > model_ids.index.max() or model_id < model_ids.index.min():
        raise ValueError(f"Model ID {model_id} not found in Objaverse dataset.")

    oxl.download_objects(
        objects=model_ids.iloc[[model_id]],
        download_dir=str(ASSET_DIR / "objaverse_models"),
        handle_found_object=DEBUG_handle_found_object,
    )


def DEBUG_handle_found_object(
    local_path: str,
    file_identifier: str,
    sha256: str,
    metadata: Dict[Hashable, Any],
) -> None:
    """Debug callback to handle found objects during download."""
    print(
        "\n\n\n---DEBUG_HANDLE_FOUND_OBJECT CALLED---\n",
        f"  {local_path=}\n  {file_identifier=}\n  {sha256=}\n  {metadata=}\n\n\n",
    )


def process_random_model(
    optional_callback: Optional[Callable] = DEBUG_handle_found_object,
) -> None:
    """Fetch a random model from the Objaverse dataset."""
    model_ids = get_annotations()
    random_id = model_ids.sample(n=1)

    oxl.download_objects(
        objects=random_id,
        download_dir=str(ASSET_DIR / "objaverse_models"),
        handle_found_object=optional_callback,
    )


def download_n_random_models(
    n: int,
    optional_callback: Optional[Callable] = DEBUG_handle_found_object,
) -> None:
    """Fetch n random models from the Objaverse dataset."""
    model_ids = get_annotations()
    random_ids = model_ids.sample(n=n)

    oxl.download_objects(
        objects=random_ids,
        download_dir=str(ASSET_DIR / "objaverse_models"),
        handle_found_object=optional_callback,
    )
