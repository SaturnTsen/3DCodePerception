"""第 2 步：场景、对象与坐标变换。在 Blender 中运行：

blender --background --factory-startup --python docs/learn/step2_transform.py \
    --python rl/export_mesh.py -- temp/learn/step2.npz
"""

import math

import bpy  # pyright: ignore[reportMissingModuleSource]

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# 对象 A：一个边长 2 的立方体，平移 + 旋转 + 拉伸
bpy.ops.mesh.primitive_cube_add(size=2)
a = bpy.context.active_object
assert a is not None
a.name = 'A'
a.location = (3.0, 0.0, 0.0)
a.rotation_euler = (0.0, 0.0, math.radians(45))
a.scale = (2.0, 1.0, 1.0)

# 对象 B：和 A 共用同一份网格数据（linked duplicate），只是变换不同
b = a.copy()
b.name = 'B'
b.location = (-3.0, 0.0, 0.0)
b.rotation_euler = (0.0, 0.0, 0.0)
b.scale = (1.0, 1.0, 1.0)
bpy.context.scene.collection.objects.link(b)

# 对象 B 加一个阵列修改器：沿 Y 方向复制 3 份
modifier = b.modifiers.new('array', 'ARRAY')
modifier.count = 3  # pyright: ignore[reportAttributeAccessIssue]
modifier.relative_offset_displace = (0.0, 1.5, 0.0)  # pyright: ignore[reportAttributeAccessIssue]

bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()
for obj in (a, b):
    mesh = obj.data
    print(f'\n=== 对象 {obj.name} ===')
    print('A 和 B 共用同一份网格数据:', a.data == b.data)
    print('原始网格顶点数:', len(mesh.vertices))  # pyright: ignore[reportAttributeAccessIssue]
    v0 = mesh.vertices[0].co  # pyright: ignore[reportAttributeAccessIssue]
    world = obj.matrix_world @ v0
    print('第 0 个顶点 局部坐标:', tuple(round(c, 3) for c in v0))
    print('第 0 个顶点 世界坐标:', tuple(round(c, 3) for c in world))
    print('matrix_world:')
    for row in obj.matrix_world:
        print('   ', tuple(round(c, 3) for c in row))
    evaluated = obj.evaluated_get(depsgraph)
    print('应用修改器后的顶点数:', len(evaluated.to_mesh().vertices))
    evaluated.to_mesh_clear()
