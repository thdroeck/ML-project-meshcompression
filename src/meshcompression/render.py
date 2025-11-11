import trimesh


def render(mesh) -> None:
    """Render a mesh using trimesh viewer."""
    mesh.show(viewer="gl", smooth=True)


def render_from_file(mesh_file) -> None:
    """Load a mesh from file and render it."""
    mesh = trimesh.load_mesh(mesh_file)
    mesh.is_watertight  # noqa: F841
    if not mesh.is_watertight:
        print("Warning: Mesh is not watertight.")
    render(mesh)


def render_watertight_from_file(mesh_file) -> None:
    """Load a mesh from file and render it if watertight. Otherwise load new mesh."""
    mesh = trimesh.load_mesh(mesh_file)
    mesh.is_watertight  # noqa: F841
    if mesh.is_watertight:
        render(mesh)


def _objaverse_callback_render_from_file(
    local_path: str,
    file_identifier: str,
    sha256: str,
    metadata: dict,
) -> None:
    """Callback to render a downloaded Objaverse model."""
    render_from_file(local_path)


def render_random_objaverse_model() -> None:
    """Download and render a random model from the Objaverse dataset."""
    from meshcompression.dataset.objaverse import process_random_model

    process_random_model(optional_callback=_objaverse_callback_render_from_file)


def _objaverse_callback_render_watertight_from_file(
    local_path: str,
    file_identifier: str,
    sha256: str,
    metadata: dict,
) -> None:
    """Callback to render a downloaded watertight Objaverse model."""
    render_watertight_from_file(local_path)


def render_random_watertight_objaverse_model() -> None:
    """Download and render a random watertight model from the Objaverse dataset."""
    from meshcompression.dataset.objaverse import process_random_model

    process_random_model(
        optional_callback=_objaverse_callback_render_watertight_from_file
    )


def render_n_random_watertight_objaverse_models(n: int) -> None:
    """Download and render n random watertight models from the Objaverse dataset."""
    from meshcompression.dataset.objaverse import download_n_random_models

    download_n_random_models(
        n=n,
        optional_callback=_objaverse_callback_render_watertight_from_file,
    )
