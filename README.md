# 3DCodePerception

本项目使用 verl 训练视觉语言模型，让模型根据图片和文字指令生成完整的 Blender Python 脚本。训练时，程序执行生成的脚本，将生成的三维几何体与目标几何体进行比较，并把比较结果作为奖励。

训练采用 GRPO 算法：模型针对同一条输入生成多个回答，verl 根据这些回答的相对奖励更新模型。

当前配置：每条输入生成四个独立回答，每个回答只进行一次生成，不包含工具交互或多轮对话。

本项目负责数据转换、奖励和实验配置。模型生成、策略更新、训练过程中的验证及模型保存均由 verl 负责。当前尚未完成整个分布式训练流程的验证。

## 环境

项目使用 Pixi 管理依赖，并使用项目中的 `.pixi/envs/default` 环境。在项目根目录执行：

```bash
pixi install --locked
pixi lock --check
```

依赖版本及覆盖规则记录在 `pixi.toml` 中，具体安装版本由 `pixi.lock` 锁定。项目通过 Pixi 配置禁用用户目录下的 Python 包，避免环境外的依赖影响运行。

## 数据准备

数据准备由 `rl.data_prep.base.DataPrepProcessor` 的子类实现。子类接收完整 YAML 配置，在 `process()` 中读取原始数据、验证图片与标签、划分训练集与验证集并构造提示词，返回包含 `train` 和 `val` 两项的记录字典。子类负责避免训练集与验证集之间的对象泄漏；通用入口只调用一次 `process()`。

在 `rl/prepare.py` 中直接导入处理器类，并在 `PROCESSORS` 字典中填写名称到类的映射。配置中的 `data_preparation` 填写字典中的名称，`outputs.train` 和 `outputs.val` 填写训练集与验证集的输出路径，`system_prompt` 和 `user_prompt` 必须显式提供。原始数据路径等专属配置由具体子类定义。所有配置路径均相对于运行命令时的工作目录。命令行入口先通过 `parse_data_config` 读取 YAML 并使用 Pydantic 模型校验，再将 `DataConfig` 传给 `prepare` 和处理器。

```bash
pixi run prepare-rl --config configs/data-3d_code_data_factory.yaml
```

通用入口将处理器返回的 `train` 和 `val` 记录分别写入配置指定的 Parquet 文件，并拒绝未知处理器、空数据集和重复的输出路径。每条记录应包含 `data_source`、`prompt`、`images`、`reward_model` 和 `extra_info` 等 verl 所需字段，具体内容由处理器构造。写出按 split 进行；后续 split 失败时，之前已写出的文件会保留。

`3DCodeDataFactoryPreprocessor` 读取 `source_root` 下每个对象目录中的同名 GLB，并按 `image_paths` 的顺序加载参考图。当前配置使用四张彩色图组成一条记录。处理器先对对象目录排序，再使用 `seed` 打乱，以 `ceil(对象数 × val_fraction)` 个对象作为验证集，其余作为训练集；同一对象的所有视图始终属于同一个集合。当前 205 个对象在 `val_fraction: 0.1` 时划分为 184 个训练对象和 21 个验证对象。

数据集专属参数由 `ThreeDCodeDataFactoryConfig` 校验，全部必须在 YAML 中显式填写。入口通过处理器的 `config_model` 选择配置模型，不提供参数默认值。Blender 使用 `blender` 和 `timeout` 配置，将 GLB 导入空场景，再通过 `rl/export_mesh.py` 导出包含世界坐标顶点和三角面的 NPZ。目标文件保存到 `target_dir`，绝对路径写入 `reward_model.ground_truth`；每次准备都会重新转换。程序检查输入图片和目标网格，不执行数据集中的 Python 脚本。

准备过程只转换训练数据，不启动模型训练。临时 Blender 日志放在 `temp/` 下；导入失败时会报告对象路径和日志，超时则报错停止。

## 训练

在 `configs/grpo.yaml` 中填写 `data.train_files` 和 `data.val_files`，使其指向转换后的文件，并确认模型路径、Blender 可执行文件路径和训练参数适合当前机器。目标几何文件必须能被计算奖励的进程访问；当前配置面向单机运行。

```bash
pixi run train-rl
```

可以在命令行覆盖配置，例如 `pixi run train-rl trainer.total_epochs=2`。通过 `--config-name` 可以选择其他实验配置，通过 `--cfg job` 可以查看组合后的配置而不启动训练。

Hydra 负责加载和组合 YAML 配置。项目入口检查数据路径、调用 verl 的配置校验，再通过 verl 的 `run_ppo` 启动官方训练任务。数据路径为空或文件不存在时，程序直接报错。训练配置继承 verl 官方的 `ppo_trainer` 配置，并在项目 YAML 中覆盖实验参数。模型输入的对话格式使用模型自带的聊天模板。

## 几何奖励

模型应输出一个标记为 `python` 的 Markdown 代码块，或直接输出完整的 Blender Python 脚本。程序在独立 Blender 进程中执行脚本，再提取场景中的三角网格。提取结果包含修改器计算后的几何，并将对象实例转换到世界坐标。

程序根据三角面的面积分布，在生成几何和目标几何的表面采样点。随后，程序分别计算两个方向的平均最近邻距离，并取两者的平均值，得到 Chamfer 距离。该距离除以目标包围盒的对角线长度后记为 `d`，最终奖励为 `exp(-d)`。距离越小，奖励越接近 1。

比较时不会分别对齐或缩放两份几何，因此位置和尺寸误差也会影响奖励。当前奖励只比较几何形状，不比较纹理或渲染图片。

脚本执行失败、超时或未生成有效网格时，奖励为零。目标几何无效时，程序报错。Blender 子进程具有运行用户的权限，独立进程和超时机制不能限制脚本访问文件或网络；运行生成代码需要使用受限的执行环境。

## 验证范围

项目代码和测试使用 Pylance 的 `standard` 类型检查级别，配置在 `pyproject.toml` 中。命令行使用 Pyright 检查：

```bash
pixi run typecheck
```

以下命令分别运行几何评分、数据准备调度、数据集转换与 Blender 执行测试：

```bash
pixi run python -m unittest discover -s tests -p test_rl_geometry.py
pixi run python -m unittest discover -s tests -p test_data_prep.py
pixi run python -m unittest discover -s tests -p test_data_factory.py
pixi run python -m unittest discover -s tests -p test_rl_pipeline.py
```

这些测试检查几何评分、图片和目标数据转换，以及脚本成功或失败时的奖励结果。它们不验证完整的分布式训练，也不能证明真实模型的训练效果。

官方接口说明：[verl 多模态训练](https://verl.readthedocs.io/en/latest/examples/multi_modal_example.html)、[verl 奖励扩展](https://verl.readthedocs.io/en/latest/advance/reward_loop.html)。
