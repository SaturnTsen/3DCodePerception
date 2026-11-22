# 3DCodePerception

本项目使用 verl 训练视觉语言模型，让模型根据图片和文字指令生成完整的 Blender Python 脚本。训练时，程序执行生成的脚本，从几何形状和渲染图两方面与目标比较，并把比较结果作为奖励。

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

`3DCodeDataFactoryPreprocessor` 把 `source_root` 下的每个目录当作一个对象，目录名就是对象编号，并按 `image_paths` 的顺序加载参考图。当前配置使用四张彩色图组成一条记录。处理器先对对象目录排序，再使用 `seed` 打乱，以 `ceil(对象数 × val_fraction)` 个对象作为验证集，其余作为训练集；同一对象的所有视图始终属于同一个集合。当前 205 个对象在 `val_fraction: 0.1` 时划分为 184 个训练对象和 21 个验证对象。

数据集专属参数由 `ThreeDCodeDataFactoryConfig` 校验，全部必须在 YAML 中显式填写。入口通过处理器的 `config_model` 选择配置模型，不提供参数默认值。`reward_model.ground_truth` 写入对象编号。数据准备不读取目标模型，也不调用 Blender；目标一侧的数据全部由奖励服务从同一个 `source_root` 自行读取，见"奖励服务"一节。`image_paths` 只决定模型看到哪些参考图，与奖励服务比较时使用的视角无关。程序不执行数据集中的 Python 脚本。

准备过程只写出训练数据，不启动模型训练。

## 训练

`configs/train-3d_code_data_factory-grpo.yaml` 中的 `data.train_files` 和 `data.val_files` 指向数据准备配置 `outputs` 写出的 Parquet 文件；修改输出路径时两处需要同步。训练前请确认模型路径和训练参数适合训练机器。训练机器不需要 Blender 和目标文件：奖励函数把代码发给 gbc1 上的奖励服务，地址由 `reward_kwargs.service_url` 指定，通过 TCP 端口转发访问。

```bash
pixi run train-rl
```

可以在命令行覆盖配置，例如 `pixi run train-rl trainer.total_epochs=2`。通过 `--config-name` 可以选择其他实验配置，通过 `--cfg job` 可以查看组合后的配置而不启动训练。

Hydra 负责加载和组合 YAML 配置。项目入口检查数据路径、调用 verl 的配置校验，再通过 verl 的 `run_ppo` 启动官方训练任务。数据路径为空或文件不存在时，程序直接报错。训练配置继承 verl 官方的 `ppo_trainer` 配置，并在项目 YAML 中覆盖实验参数。模型输入的对话格式使用模型自带的聊天模板。

## 奖励

`rl/reward.py` 是 verl 的自定义奖励函数，以异步方式运行，verl 会同时为一批回答发出请求。模型应输出一个标记为 `python` 的 Markdown 代码块；回答中恰好有一个这样的代码块时取其内容，否则把整个回答当作脚本。奖励函数把脚本、对象编号和本次训练的 `run_id` 发给奖励服务，再把返回的原始数值合成为四个分量，每个分量都由 `reward_kwargs` 中的必填参数控制。`run_id` 在 YAML 中写作 `???`，由 `rl/train.py` 在启动时填为 `<trainer.experiment_name>-<启动时间>`，例如 `single_response_grpo-20261122-065834`，用来区分不同的训练：

| 分量 | 计算方式 |
|---|---|
| `chamfer` | exp(−d/τ)。d 是 `chamfer_normalization` 归一化方式下的 Chamfer 距离，τ 是 `chamfer_temperature` |
| `dinov3` | 四个渲染视角与四张参考图在最优一一匹配下的平均余弦相似度 |
| `uni3d` | 生成点云与目标点云的 Uni3D 余弦相似度 |
| `uni3d_image` | 生成点云的 Uni3D 向量与参考图 `image_0` 的 CLIP 向量的余弦相似度。Uni3D 的点云编码器训练时对齐到这个 CLIP 的向量空间，所以两者可以直接比较 |

总奖励是四个分量按 `weight_chamfer`、`weight_dinov3`、`weight_uni3d`、`weight_uni3d_image` 加权求和。执行出错、超时或几何无效时，总奖励和各分量都为 0，`chamfer_distance` 为 −1。返回结果中的 `success`、`chamfer_distance` 和各分量会由 verl 一起记录。奖励服务返回 HTTP 错误时，奖励函数直接报错，训练停止。

Chamfer 距离的定义：在生成几何和目标几何的表面按面积撒点，两份点云都在 Blender 世界坐标系中，Z 轴向上。比较前分别减去各自的质心，再除以一个半径：`rms_radius` 用所有点到质心距离的均方根，`max_radius` 用离质心最远的点的距离。然后把生成点云绕 Z 轴旋转 0°、90°、180° 和 270°，对每个角度计算两个方向平均最近邻距离的平均值，取四个角度中的最小值。因此，位置、整体尺度和 90° 整数倍的水平朝向差异不影响这一项；上下颠倒、侧放、比例错误以及缺失或多余的部件会让距离变大。

当前配置中的温度和权重是占位值，需要先统计初始模型回答的各分量分布再确定，见 `docs/TODOs.md`。

## 奖励服务

`reward/` 是运行在 gbc1 上的 HTTP 服务，负责运行模型生成的代码并测量结果。训练集群通过 TCP 端口转发访问它，这是训练端和 gbc1 之间唯一的接口：请求和返回的数据结构定义在 `reward/types.py` 中，`rl/` 只引用这一个文件，不引用 `reward/` 的其他代码，也不读取对方的配置或输出，两边只共享同一个原始数据集目录。配置写在 `configs/service-reward.yaml` 中，所有字段都必须显式填写。启动命令：

```bash
pixi run reward-service --config configs/service-reward.yaml
```

| 文件 | 作用 |
|---|---|
| `reward/app.py` | 服务入口：启动时加载编码器、准备目标特征，然后处理 `POST /run` |
| `reward/types.py` | 服务和 `rl/reward.py` 共用的数据结构，例如测量结果 `MeasureResult`；只依赖 Pydantic |
| `reward/script_runner.py` | 启动 Blender 子进程运行生成的代码，并对结果分类 |
| `reward/measure.py` | 撒点、编码，并把一次运行的网格和渲染图与目标特征比较，返回各指标的原始数值 |
| `reward/target_feats.py` | 启动时为缺少目标特征的对象撒点、编码，写出目标特征 |
| `reward/sampling.py` | 在三角网格表面按面积撒点，供 Chamfer 和 Uni3D 使用 |
| `reward/encoders/` | DINOv3、Uni3D、CLIP 三个编码器，以及 Uni3D 的点云网络；只负责把图片或点云变成向量 |
| `reward/metrics/` | Chamfer 距离和视角相似度；只负责比较两组数据 |
| `reward/blender/` | 在 Blender 中运行的脚本：导入 GLB、导出网格、渲染视角 |

### 视角

配置项 `views` 列出比较时使用的视角，每一项包含渲染相机的方位角 `azimuth` 和对应的参考图 `reference`（相对于对象目录的路径）。当前四个视角的方位角是 45°、135°、225° 和 315°，对应 `color_renders/Image_005.png`、`Image_015.png`、`Image_025.png` 和 `Image_035.png`。生成结果按这些方位角渲染，渲染图和参考图按视角比较。这一设置只属于奖励服务，与数据准备的 `image_paths` 无关。

### 启动时准备目标特征

服务启动时先加载 DINOv3 和 Uni3D 编码器，常驻 `device` 指定的 GPU，约占 6 GB 显存。然后检查 `source_root` 下的每个对象目录在 `features_dir` 中是否有特征文件 `<对象编号>.npz`，缺少的由 `reward/target_feats.py` 当场计算：

1. 用 Blender 运行 `reward/blender/import_glb.py` 导入对象目录中的 `<对象编号>.glb`，再用 `reward/blender/export_mesh.py` 导出世界坐标系下的三角网格。导出模型回答的网格用的也是同一个 `export_mesh.py`。单个对象的转换时间限制为 `glb_timeout`。
2. 按 `views` 读取参考图。
3. 写出四项特征：Chamfer 的表面采样点、各视角参考图的 DINOv3 向量、目标网格的 Uni3D 向量，以及第一张参考图的 CLIP 向量。

CLIP 图像编码器约 9.4 GB，只在有特征缺失时加载，算完立即释放，不占用服务期间的显存。目标一侧和模型回答一侧用的是同一个 DINOv3 编码器对象和同一个 Uni3D 编码器对象，Chamfer 撒点读同一个配置项，所以两边的处理方式完全一致。已有的特征文件不会重算；修改编码器、撒点设置或视角后，需要删除 `features_dir`，再重启服务。

### 处理一个请求

`POST /run` 接收 JSON `{"run_id": ..., "object_id": ..., "code": ...}`。`run_id` 只能包含字母、数字、`.`、`_` 和 `-`，且不能以 `.` 开头，否则返回 422。每个请求的文件保存在 `runs_dir/<run_id>/<时间>_<对象编号>_<8 位随机串>/` 中，处理完不删除：生成的代码 `response.py`、导出的网格 `mesh.npz`、渲染图 `renders/`、Blender 日志 `blender.log`，以及与返回内容相同的 `result.json`。渲染失败返回 500 时目录同样保留，但没有 `result.json`；`object_id` 未知时在创建目录之前返回 404。`work_dir` 只在启动时计算目标特征时存放临时文件。请求依次完成两件事：

1. **运行**：启动一个 Blender 子进程（`reward/script_runner.py`），依次执行代码、用 `export_mesh.py` 导出网格、用 `reward/blender/render.py` 按 `views` 的方位角渲染。相机和灯光设置与 3DHarnessBench 相同。Blender 只使用 `render.gpus` 指定的 GPU（通过 `CUDA_VISIBLE_DEVICES`），应与编码器使用的 `device` 错开。同时运行的 Blender 进程数由 `max_parallel` 限制，超出的请求排队等待。
2. **测量**：在服务进程内调用 `reward/measure.py`，把网格和渲染图与 `features_dir/<object_id>.npz` 中的目标特征比较。每个编码器自带一把锁，并发请求依次使用 GPU。

| 指标 | 返回的数值 | 实现依据 |
|---|---|---|
| `chamfer` | 两种归一化方式（`rms_radius` 和 `max_radius`）下的 Chamfer 距离 | 与"奖励"一节的定义相同 |
| `dinov3` | 按视角编号逐一配对的平均余弦相似度，以及最优一一匹配下的平均余弦相似度 | 3DHarnessBench `image_similarity.py` |
| `uni3d` | 生成点云与目标点云的 Uni3D 向量之间（`cos_3d_3d`），以及生成点云的 Uni3D 向量与第一张参考图的 CLIP 向量之间（`cos_image_3d`）的余弦相似度 | 3DHarnessBench `shape_uni3d.py` |

Uni3D 的点云网络改编自 Uni3D 官方仓库，见 `reward/encoders/point_encoder.py`。最远点采样改成了 PyTorch 实现，从第 0 个点开始，与官方的 CUDA 实现一致，结果是确定的。服务只返回原始数值，怎样合成训练奖励由 `rl/reward.py` 决定。

### 返回结果

`script_timeout` 只限制生成脚本本身的运行时间，从 Blender 启动时开始计算，不包括排队时间。网格导出之后，渲染另有 `render_timeout` 的时间限制。返回的 `run_status` 表示生成代码的运行结果（不是 HTTP 状态码），有四种取值：

| 取值 | 含义 |
|---|---|
| `ok` | 运行和测量都成功，`metrics` 中是各指标的原始数值 |
| `execution_error` | 脚本报错，或者在导出网格之前退出 |
| `timeout` | 脚本或渲染超过了时间限制 |
| `invalid_geometry` | 导出的网格没有三角面（例如场景是空的），或者三角面的总面积为 0、坐标不是有限值，无法在表面上撒点 |

其他情况都按 HTTP 错误返回：`object_id` 未知时返回 404；服务自身出错时返回 5xx。调用方遇到 HTTP 错误应当停止训练，而不是按 0 分处理。状态不是 `ok` 时，`log` 字段包含 Blender 日志的末尾部分。返回结果中的 `seconds` 记录排队、Blender 和测量各自的耗时。

渲染失败单独处理：如果导出的网格有三角面，但 Blender 没有正常渲染出四个视角（例如显存不足、找不到 GPU、Blender 崩溃），说明模型已经生成了几何体，失败应归因于渲染环境，不能给模型 0 分。这时服务返回 HTTP 500，响应中附带 Blender 日志的末尾部分，同时把日志写入服务自己的输出；奖励函数遇到 500 会抛出异常，训练随之停止。服务进程第一次遇到渲染失败时，还会按配置项 `alert` 发送一条 Bark 推送（`level: critical`，即使手机静音也会响铃）。同一进程内之后的渲染失败只返回 500，不再重复推送；排查完成后重启服务即可恢复推送。渲染超时仍然按 `timeout` 计入模型的结果。

服务不隔离生成的代码：Blender 以运行服务的用户身份执行，能够访问该用户的文件和网络。本项目的服务运行在容器中，这一点是有意的取舍。

## 验证范围

项目代码和测试使用 Pylance 的 `standard` 类型检查级别，配置在 `pyproject.toml` 中。命令行使用 Pyright 检查：

```bash
pixi run typecheck
```

以下命令分别运行奖励服务、数据准备调度、数据集转换和奖励合成的测试：

```bash
pixi run python -m unittest discover -s tests -p "test_reward_*.py"
pixi run python -m unittest discover -s tests -p test_data_prep.py
pixi run python -m unittest discover -s tests -p test_data_factory.py
pixi run python -m unittest discover -s tests -p test_rl_reward.py
```

奖励服务的测试在 `cuda:1` 上加载真实的 DINOv3、Uni3D 和 CLIP 编码器，并启动真实的 Blender 转换目标、运行代码和渲染，需要 GPU。测试只检查数值范围和结果分类，不与 3DHarnessBench 逐项对比数值，也不验证完整的分布式训练和真实模型的训练效果。

官方接口说明：[verl 多模态训练](https://verl.readthedocs.io/en/latest/examples/multi_modal_example.html)、[verl 奖励扩展](https://verl.readthedocs.io/en/latest/advance/reward_loop.html)。
