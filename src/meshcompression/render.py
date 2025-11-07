import trimesh


def render(mesh) -> None:
    """Render a mesh using trimesh viewer."""
    mesh.show(viewer="gl", smooth=True)


def render_from_file(mesh_file) -> None:
    """Load a mesh from file and render it."""
    mesh = trimesh.load_mesh(mesh_file)
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
