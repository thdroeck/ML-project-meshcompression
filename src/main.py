import trimesh


mesh = trimesh.load_mesh("data/bunny.obj")

mesh.show(viewer="gl", smooth=True)
