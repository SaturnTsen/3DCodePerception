"""Run inside Blender after the generated script; export evaluated world-space meshes."""
import sys
import bpy
import numpy as np

vertices, faces = [], []
depsgraph = bpy.context.evaluated_depsgraph_get()
for instance in depsgraph.object_instances:
    obj = instance.object
    if obj.type != "MESH":
        continue
    mesh = obj.to_mesh()
    mesh.calc_loop_triangles()
    offset = len(vertices)
    vertices.extend(tuple(instance.matrix_world @ v.co) for v in mesh.vertices)
    faces.extend(tuple(offset + i for i in triangle.vertices) for triangle in mesh.loop_triangles)
    obj.to_mesh_clear()
np.savez(sys.argv[sys.argv.index("--") + 1], vertices=vertices, faces=faces)
