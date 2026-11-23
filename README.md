# T800 Isaac Lab 训练工程

把 T800 人形机器人接进 Isaac Lab，用 rsl_rl 的 PPO 训练速度跟踪（走/跑）。
工程结构参考 Isaac Lab 官方扩展格式，可以直接 `pip install -e`。

## 目录结构

```
t800_isaaclab/
├── assets/t800/                 # 从 EngineAI SDK 复制
│   ├── urdf/serial_t800.urdf
│   ├── meshes/                  # .dae / .obj
│   ├── texture/
│   └── xml/                     # MuJoCo MJCF，留作参考
├── scripts/
│   ├── convert_t800.py          # URDF -> USD
│   ├── train.py                 # 训练
│   └── play.py                  # 回放
├── source/t800_lab/
│   ├── config/extension.toml
│   ├── setup.py
│   └── t800_lab/
│       ├── assets/t800.py       # ArticulationCfg
│       └── tasks/locomotion/velocity/
│           ├── t800_env_cfg.py  # 环境/奖励/命令
│           └── agents/rsl_rl_ppo_cfg.py
└── README.md
```

## 环境要求

- Ubuntu 22.04 / 24.04
- NVIDIA 驱动 >= 570（RTX 5070 Ti 是 Blackwell，需要新驱动）
- Isaac Sim 5.0 + Isaac Lab 2.x
- Python 3.10 / 3.11（跟随 Isaac Lab 要求）
- 显存 16 GB：建议 `--num_envs 1024` 或 `2048`，4096 可能 OOM

## 0. 检查 conda 环境

先确认 Isaac Lab 的环境能正常导入：

```bash
conda activate Isaaclab
cd t800_isaaclab
python scripts/check_env.py
```

会检查 isaaclab / isaaclab_rl / isaaclab_tasks / rsl_rl / torch / gymnasium，
以及 CUDA、驱动、T800 资产和 USD 是否存在。缺什么按提示补。

## 安装

先按 Isaac Lab 官方文档装好 Isaac Sim 和 Isaac Lab，然后在本项目里：

```bash
conda activate Isaaclab
cd t800_isaaclab
pip install -r requirements.txt
pip install -e source/t800_lab
```

`requirements.txt` 只装本项目额外需要的 `psutil / toml / tensorboard`，
不会碰 Isaac Lab、Isaac Sim、torch、rsl-rl-lib，避免把环境版本搞乱。

如果不想 `pip install -e`，`scripts/train.py` 和 `play.py` 也会自动把
`source/t800_lab` 加进 `sys.path`，可以直接运行。

## 1. 生成 USD

Isaac Lab 不能直接加载 MuJoCo MJCF，需要先把 URDF 转成 USD：

```bash
python scripts/convert_t800.py --headless
```

脚本会：

1. 把 `package://resource/robot/t800/` 改写成相对路径 `../meshes/`
2. 调用 Isaac Lab 的 `UrdfConverter`
3. 生成 `source/t800_lab/t800_lab/assets/t800.usd`

确认文件存在：

```bash
ls -lh source/t800_lab/t800_lab/assets/t800.usd
```

> USD 是生成文件，已在 `.gitignore` 里排除，换机器要重新转一次。

## 2. 训练

```bash
python scripts/train.py \
    --task Isaac-T800-Locomotion-v0 \
    --num_envs 1024 \
    --headless
```

- 任务 id：`Isaac-T800-Locomotion-v0`
- 日志：`logs/rsl_rl/t800_locomotion/<时间戳>/`
- 模型：`model_*.pt`
- 参数快照：`params/env.yaml`、`params/agent.yaml`

看曲线：

```bash
tensorboard --logdir logs/rsl_rl
```

## 代码导读

第一次看这套代码，建议先读：

- [docs/CODE_GUIDE.md](docs/CODE_GUIDE.md)：包/类/术语速查、训练数据流、阅读顺序
- `source/t800_lab/t800_lab/tasks/locomotion/velocity/t800_env_cfg.py`：环境与奖励
- `source/t800_lab/t800_lab/assets/t800.py`：机器人配置
- `scripts/train.py`：训练入口

## 3. 回放

```bash
python scripts/play.py \
    --task Isaac-T800-Locomotion-v0 \
    --num_envs 16 \
    --checkpoint logs/rsl_rl/t800_locomotion/<时间戳>/model_1000.pt
```

不传 `--checkpoint` 会自动找最新的 `.pt`。去掉 `--headless` 可以看到 Isaac Sim 窗口。

## 任务设计

- 机器人：T800，25 个驱动关节
- 动作：25 个关节位置增量，`scale=0.25`
- 观测：基座线速度、角速度、投影重力、速度指令、关节角/角速度、上一动作
- 速度指令：`vx ∈ [-0.5, 2.5] m/s`，`vy ∈ [-0.5, 0.5]`，`wz ∈ [-1, 1]` rad/s
- 奖励：
  - 线速度跟踪 +3.0
  - 角速度跟踪 +1.5
  - 基座竖直速度、横滚俯仰角速度、姿态不水平：惩罚
  - 足端腾空时间 +5.0（鼓励迈步/腾空）
  - 足端滑动 -0.05
  - 关节力矩、关节加速度、动作变化率：惩罚
- 终止：基座着地、身体倾斜超过 0.8 rad、超时
- 地形：平地（后续可切 rough）

## 常见问题

**1. `t800.usd` 不存在**

先跑 `python scripts/convert_t800.py --headless`。

**2. URDF 转换时找不到 mesh**

确认 `assets/t800/meshes/` 存在，且 `serial_t800_local.urdf` 里是 `../meshes/...`。

**3. 显存不够**

把 `--num_envs` 降到 1024 或 512；Isaac Sim 本身也占显存。

**4. 训练不收敛**

- 先确认机器人在环境里能站住（`play.py` 看初始姿态）
- 检查 `LINK_FOOT_L/R` 是否被接触传感器识别
- 速度范围可以从 `(-0.5, 1.0)` 开始，稳定后再放开到 2.5

## 资产来源

`assets/t800/` 来自 EngineAI 的 `engineai_robotics_native_sdk-main`，仅用于本地训练；
对外分发前请确认 SDK 的许可证。


## 参考步态模仿（Mimic）

用 MNN 走路策略录一段参考步态，再让 Isaac Lab 的策略跟踪它。

### 1. 录制参考轨迹（MuJoCo 环境）

```bash
conda activate mujoco
cd ~/t800_isaaclab

python scripts/record_mnn_reference.py \
  --mujoco-dir ~/mujoco_t800 \
  --out assets/reference/walk_ref.npz \
  --command 1.0 --warmup 2 --duration 10
```

生成 `assets/reference/walk_ref.npz`，并同步到
`source/t800_lab/t800_lab/assets/reference/walk_ref.npz`。

### 2. 训练 Mimic 任务

```bash
conda activate isaaclab
cd ~/t800_isaaclab

python -u scripts/train.py \
  --task Isaac-T800-Mimic-v0 \
  --num_envs 4096 \
  --max_iterations 3000 \
  --headless
```

动作是 `参考关节角 + action * 0.25`，策略只需要微调平衡。
日志在 `logs/rsl_rl/t800_mimic/`。
