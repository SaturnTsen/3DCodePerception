"""Import a GLB before running rl/export_mesh.py in the same Blender process.

Usage: blender ... --python import_glb.py --python export_mesh.py -- OUTPUT_NPZ INPUT_GLB
"""

import sys

import bpy  # pyright: ignore[reportMissingModuleSource]


bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
bpy.ops.import_scene.gltf(filepath=sys.argv[sys.argv.index('--') + 2])
