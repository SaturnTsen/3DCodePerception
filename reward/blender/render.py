"""Render the current scene from the reference camera azimuths.

Run after the generated script and export_mesh.py in the same Blender process:
    blender --background --python response.py --python export_mesh.py --python render.py \
        -- mesh.npz --render-dir DIR --engine CYCLES --device GPU --samples 16 --resolution 512 \
        --azimuths 45 135 225 315

Writes DIR/view_0.png, view_1.png, ... (RGBA, transparent background), one per azimuth
in degrees. The reward service config pairs each azimuth with a reference image.

The camera, lights, world, and default material follow 3DHarnessBench
core/blender_scene_setup.py (rig version benchmark_extent_squared_v1). The scene
is recentered for rendering, so export the mesh before running this script.
"""
import argparse
import math
from pathlib import Path
import sys

# Blender provides the runtime modules; Pixi only contains their type stubs.
import bpy  # pyright: ignore[reportMissingModuleSource]
from mathutils import Vector  # pyright: ignore[reportMissingModuleSource]

GEOMETRY_TYPES = ('MESH', 'CURVE', 'SURFACE', 'FONT', 'META')
LIGHT_REFERENCE_EXTENT = 2.5
# name: (position in units of camera radius/height, energy, size relative to extent)
LIGHTS = {
    'Key': ((0.9, -0.7, 1.7), 1200.0, 0.9),
    'Fill': ((-0.6, -0.4, 1.0), 400.0, 1.2),
    'Rim': ((0.0, 0.9, 1.3), 600.0, 0.7),
}


def bounds(objects: list) -> tuple[Vector, float]:
    """Bounding-box center and largest side, ignoring objects far larger than the median."""
    boxes = []
    for obj in objects:
        corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        low = Vector(tuple(min(c[i] for c in corners) for i in range(3)))
        high = Vector(tuple(max(c[i] for c in corners) for i in range(3)))
        boxes.append((max((high - low)[:]), low, high))
    sizes = sorted(size for size, _, _ in boxes)
    median = sizes[len(sizes) // 2]
    limit = max(median * 30.0, 30.0) if median > 0 else 30.0
    kept = [box for box in boxes if box[0] <= limit] or boxes
    low = Vector(tuple(min(box[1][i] for box in kept) for i in range(3)))
    high = Vector(tuple(max(box[2][i] for box in kept) for i in range(3)))
    return (low + high) / 2, max((high - low)[:]) or 1.0


def look_at(obj, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat('-Z', 'Y').to_euler()


def enable_cycles_gpu() -> None:
    preferences = bpy.context.preferences.addons['cycles'].preferences
    for backend in ('OPTIX', 'CUDA'):
        preferences.compute_device_type = backend  # pyright: ignore[reportAttributeAccessIssue]
        preferences.refresh_devices()  # pyright: ignore[reportAttributeAccessIssue]
        devices = [d for d in preferences.devices if d.type == backend]  # pyright: ignore[reportAttributeAccessIssue]
        if devices:
            for device in preferences.devices:  # pyright: ignore[reportAttributeAccessIssue]
                device.use = device.type == backend
            bpy.context.scene.cycles.device = 'GPU'  # pyright: ignore[reportAttributeAccessIssue]
            return
    raise RuntimeError('render device GPU requested, but Cycles found no OptiX or CUDA device')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh')
    parser.add_argument('--render-dir', type=Path, required=True)
    parser.add_argument('--engine', choices=('CYCLES', 'BLENDER_EEVEE'), required=True)
    parser.add_argument('--device', choices=('GPU', 'CPU'), required=True)
    parser.add_argument('--samples', type=int, required=True)
    parser.add_argument('--resolution', type=int, required=True)
    parser.add_argument('--azimuths', type=float, nargs='+', required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])

    scene = bpy.context.scene
    assert scene is not None
    for obj in list(scene.objects):
        if obj.type in ('CAMERA', 'LIGHT'):
            bpy.data.objects.remove(obj, do_unlink=True)
    objects = [obj for obj in scene.objects if obj.type in GEOMETRY_TYPES]
    if not objects:
        raise RuntimeError('scene has no renderable geometry')

    bpy.context.view_layer.update()  # pyright: ignore[reportOptionalMemberAccess]
    center, extent = bounds(objects)
    roots = set()
    for obj in objects:
        while obj.parent is not None:
            obj = obj.parent
        roots.add(obj)
    for root in roots:
        root.location = root.location - center
    origin = Vector((0.0, 0.0, 0.0))
    radius, height = extent * 1.8, extent * 0.6

    world = scene.world or bpy.data.worlds.new('World')
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes  # pyright: ignore[reportOptionalMemberAccess]
    nodes.clear()
    background = nodes.new('ShaderNodeBackground')
    background.inputs[0].default_value = (0.012, 0.013, 0.021, 1)  # pyright: ignore[reportAttributeAccessIssue]
    background.inputs[1].default_value = 1.0  # pyright: ignore[reportAttributeAccessIssue]
    world.node_tree.links.new(  # pyright: ignore[reportOptionalMemberAccess]
        background.outputs[0], nodes.new('ShaderNodeOutputWorld').inputs[0])

    energy_scale = (extent / LIGHT_REFERENCE_EXTENT) ** 2
    for name, ((x, y, z), energy, size) in LIGHTS.items():
        data = bpy.data.lights.new(name, 'AREA')
        data.energy = energy * energy_scale  # pyright: ignore[reportAttributeAccessIssue]
        data.size = extent * size  # pyright: ignore[reportAttributeAccessIssue]
        light = bpy.data.objects.new(name, data)
        light.location = (radius * x, radius * y, height * z)
        look_at(light, origin)
        scene.collection.objects.link(light)

    material = bpy.data.materials.new('eval_default')
    material.use_nodes = True
    shader = material.node_tree.nodes['Principled BSDF']  # pyright: ignore[reportOptionalMemberAccess]
    shader.inputs['Base Color'].default_value = (0.72, 0.72, 0.75, 1)  # pyright: ignore[reportAttributeAccessIssue]
    shader.inputs['Roughness'].default_value = 0.65  # pyright: ignore[reportAttributeAccessIssue]
    for obj in objects:
        if hasattr(obj.data, 'materials') and not obj.data.materials:  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
            obj.data.materials.append(material)  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]

    camera_data = bpy.data.cameras.new('EvalCam')
    camera_data.lens = 50
    camera_data.clip_end = extent * 25
    camera = bpy.data.objects.new('EvalCam', camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera

    render = scene.render
    render.resolution_x = render.resolution_y = args.resolution
    render.resolution_percentage = 100
    render.image_settings.file_format = 'PNG'
    render.image_settings.color_mode = 'RGBA'
    render.film_transparent = True
    render.engine = args.engine
    if args.engine == 'CYCLES':
        scene.cycles.samples = args.samples  # pyright: ignore[reportAttributeAccessIssue]
        scene.cycles.use_denoising = True  # pyright: ignore[reportAttributeAccessIssue]
        if args.device == 'GPU':
            enable_cycles_gpu()
        else:
            scene.cycles.device = 'CPU'  # pyright: ignore[reportAttributeAccessIssue]
    else:
        scene.eevee.taa_render_samples = args.samples  # pyright: ignore[reportAttributeAccessIssue]

    args.render_dir.mkdir(parents=True, exist_ok=True)
    for index, azimuth in enumerate(args.azimuths):
        angle = math.radians(azimuth)
        camera.location = (radius * math.cos(angle), radius * math.sin(angle), height)
        look_at(camera, origin)
        render.filepath = str(args.render_dir / f'view_{index}.png')
        bpy.ops.render.render(write_still=True)


main()
