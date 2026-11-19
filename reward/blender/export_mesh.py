"""Export scene meshes, including modifiers and instances, to world coordinates.

Run after the generated script:
    blender --background --python response.py --python export_mesh.py -- mesh.npz

Output NPZ: vertices (Nx3 coordinates), faces (Mx3 triangle vertex indices).
"""
import sys
from collections.abc import Sequence
from typing import cast

# Blender provides the runtime module; Pixi only contains its type stubs.
import bpy  # pyright: ignore[reportMissingModuleSource]
import numpy as np

vertices: list[tuple[float, float, float]] = []
faces: list[tuple[int, int, int]] = []
depsgraph = bpy.context.evaluated_depsgraph_get()

for instance in depsgraph.object_instances:
    obj = instance.object
    if obj is None or obj.type != "MESH":
        continue
    mesh = obj.to_mesh()
    mesh.calc_loop_triangles()
    offset = len(vertices)
    for vertex in mesh.vertices:
        point = instance.matrix_world @ vertex.co
        vertices.append((point.x, point.y, point.z))
    for triangle in mesh.loop_triangles:
        # bpy_prop_array supports indexing, which is missing from the stubs.
        indices = cast(Sequence[int], triangle.vertices)
        faces.append((offset + indices[0], offset + indices[1], offset + indices[2]))
    obj.to_mesh_clear()

np.savez(sys.argv[sys.argv.index("--") + 1], vertices=vertices, faces=faces)
