# 3DCodePerception

本项目使用 verl 训练视觉语言模型，让模型根据图片和文字指令生成完整的 Blender Python 脚本。训练时，程序执行生成的脚本，将生成的三维几何与目标几何进行比较，并把比较结果作为奖励。

训练采用 GRPO 算法：模型针对同一条输入生成多个回答，verl 根据这些回答的相对奖励更新模型。当前配置为每条输入生成四个独立回答，每个回答只进行一次生成，不包含工具交互或多轮对话。

本项目负责数据转换、奖励和实验配置。模型生成、策略更新、训练过程中的验证及模型保存均由 verl 负责。当前尚未完成整个分布式训练流程的验证。

## 环境

项目使用 Pixi 管理依赖，并使用项目中的 `.pixi/envs/default` 环境。在项目根目录执行：

```bash
pixi install --locked
pixi lock --check
```

依赖版本及覆盖规则记录在 `pixi.toml` 中，具体安装版本由 `pixi.lock` 锁定。项目通过 Pixi 配置禁用用户目录下的 Python 包，避免环境外的依赖影响运行。

## 数据准备

数据接口不依赖特定数据集。训练集和验证集分别使用一个 JSONL 文件，每行是一条 JSON 样本：

```json
{"object_id":"object-001","images":["view.png"],"instruction":"Reconstruct this object.","target":"target.npz"}
```

各字段的含义如下：

| 字段 | 含义 |
| --- | --- |
| `object_id` | 对象标识，用于检查同一对象是否同时出现在训练集和验证集中。 |
| `images` | 输入图片的路径列表，至少包含一张图片。 |
| `instruction` | 输入给模型的任务描述。 |
| `target` | 目标几何文件的路径，仅用于计算奖励，不作为模型输入。 |

图片和目标几何的路径均相对于 JSONL 文件所在目录解析。目标几何使用 NumPy 的 NPZ 文件格式，包含两个数组：`vertices` 保存形状为 N×3 的顶点世界坐标，`faces` 保存形状为 M×3 的三角面顶点索引。数据提供者需要保证图片与目标几何对应，并统一几何的坐标系和单位。

在 `configs/data.yaml` 中填写训练集和验证集的 JSONL 路径、输出路径，以及指导模型回答的系统提示词。数据路径目前为 `null`，需要填写后才能执行转换：

```bash
pixi run prepare-rl
```

转换程序检查图片和目标几何，并拒绝空数据集，以及训练集与验证集包含相同对象的情况。程序将样本保存为 verl 使用的 Parquet 表格文件，其中包含对话消息、图片和奖励所需的目标路径。

## 训练

在 `configs/grpo.yaml` 中填写 `data.train_files` 和 `data.val_files`，使其指向转换后的文件，并确认模型路径、Blender 可执行文件路径和训练参数适合当前机器。目标几何文件必须能被计算奖励的进程访问；当前配置面向单机运行。

```bash
pixi run train-rl
```

启动程序在数据路径为空或文件不存在时直接报错，然后调用 verl 的训练入口。训练配置继承 verl 官方的 `ppo_trainer` 配置，并在项目 YAML 中覆盖实验参数。模型输入的对话格式使用模型自带的聊天模板。

## 几何奖励

模型应输出一个标记为 `python` 的 Markdown 代码块，或直接输出完整的 Blender Python 脚本。程序在独立 Blender 进程中执行脚本，再提取场景中的三角网格。提取结果包含修改器计算后的几何，并将对象实例转换到世界坐标。

程序根据三角面的面积分布，在生成几何和目标几何的表面采样点。随后，程序分别计算两个方向的平均最近邻距离，并取两者的平均值，得到 Chamfer 距离。该距离除以目标包围盒的对角线长度后记为 `d`，最终奖励为 `exp(-d)`。距离越小，奖励越接近 1。

比较时不会分别对齐或缩放两份几何，因此位置和尺寸误差也会影响奖励。当前奖励只比较几何形状，不比较纹理或渲染图片。

脚本执行失败、超时或未生成有效网格时，奖励为零。目标几何无效时，程序报错。Blender 子进程具有运行用户的权限，独立进程和超时机制不能限制脚本访问文件或网络；运行生成代码需要使用受限的执行环境。

## 验证范围

以下命令分别运行几何评分测试，以及数据转换与 Blender 执行测试：

```bash
pixi run python -m unittest discover -s tests -p test_rl_geometry.py
pixi run python -m unittest discover -s tests -p test_rl_pipeline.py
```

这些测试检查几何评分、图片和目标数据转换，以及脚本成功或失败时的奖励结果。它们不验证完整的分布式训练，也不能证明真实模型的训练效果。

官方接口说明：[verl 多模态训练](https://verl.readthedocs.io/en/latest/examples/multi_modal_example.html)、[verl 奖励扩展](https://verl.readthedocs.io/en/latest/advance/reward_loop.html)。
