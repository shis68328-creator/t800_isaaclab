# T800 Isaac Lab 代码导读（小白版）

这份文档解释：每个包是干什么的、数据怎么流动、每个文件负责什么。
配合源码里的中文批注一起看。

---

## 1. 先分清三个东西

| 名字 | 是什么 | 在本项目里的角色 |
|---|---|---|
| **Isaac Sim** | NVIDIA 的机器人仿真器（基于 Omniverse） | 提供物理引擎、USD 场景、GPU 渲染 |
| **Isaac Lab** | 建立在 Isaac Sim 上的机器人学习框架 | 提供 `ManagerBasedRLEnv`、场景/奖励/终止管理、机器人配置 |
| **rsl_rl** | ETH 的强化学习库，实现 PPO | 真正执行“采样 → 更新网络”的训练器 |

一句话：

```text
Isaac Sim 负责“仿真”，Isaac Lab 负责“把仿真组织成 RL 环境”，rsl_rl 负责“训练”。
```

---

## 2. 训练时数据怎么流动

```text
命令行参数 --task / --num_envs
        ↓
gym.make("Isaac-T800-Locomotion-v0")
        ↓
ManagerBasedRLEnv（Isaac Lab 环境）
   ├─ 场景：地面 + T800 + 接触传感器
   ├─ 观测：基座速度/姿态 + 指令 + 关节角/速度 + 上一动作
   ├─ 动作：25 个关节的位置增量
   ├─ 奖励：速度跟踪、腾空、防滑、省力……
   └─ 终止：基座着地 / 倾斜过大 / 超时
        ↓
RslRlVecEnvWrapper（包装成 rsl_rl 能吃的向量环境）
        ↓
OnPolicyRunner（PPO）
   ├─ 采样：num_envs 个环境各跑 24 步
   ├─ 用奖励算 advantage / return
   ├─ 更新 actor（策略）和 critic（价值）
   └─ 定期保存 model_*.pt
        ↓
TensorBoard 日志 / .pt 模型
```

---

## 3. 包和类速查

### 3.1 标准库

| 包 | 作用 |
|---|---|
| `argparse` | 解析 `--task`、`--num_envs` 等命令行参数，返回 `Namespace` |
| `pathlib.Path` | 拼路径，`Path(__file__).resolve().parents[1]` 表示项目根目录 |
| `os` / `sys` | 环境变量、退出码、`sys.path` |
| `datetime` | 生成训练日志的时间戳 |
| `glob` | 找最新的 `.pt` |
| `importlib.metadata` | 读 pip 包的版本号，不 import 模块 |
| `subprocess` | 调用 `nvidia-smi` |

### 3.2 强化学习接口

| 包 / 类 | 作用 |
|---|---|
| `gymnasium`（`gym`） | RL 环境标准接口：`reset()`、`step()`、`spaces.Box` |
| `gym.make(task_id, cfg=...)` | 根据注册表创建 Isaac Lab 环境 |
| `gym.register(...)` | 把 `Isaac-T800-Locomotion-v0` 注册到 Gym |
| `spaces.Box` | 描述观测/动作的形状和范围 |

Gymnasium 的 `step()` 返回 5 个值：

```python
obs, reward, terminated, truncated, info = env.step(action)
```

- `terminated`：真失败（摔倒）
- `truncated`：到时间上限（超时）
- `info`：额外信息

### 3.3 Isaac Lab 核心

| 包 / 类 | 作用 |
|---|---|
| `isaaclab.app.AppLauncher` | 启动 Isaac Sim；必须在 import isaaclab 之前调用 |
| `isaaclab.envs.ManagerBasedRLEnv` | 基于“管理器”的 RL 环境：场景、观测、奖励、终止都拆成配置 |
| `isaaclab.utils.configclass` | 配置类装饰器，类似 dataclass；支持嵌套、`to_dict()`、`replace()` |
| `isaaclab.assets.ArticulationCfg` | 机器人整体配置：USD、初始姿态、执行器 |
| `isaaclab.actuators.ImplicitActuatorCfg` | 隐式 PD 执行器：`stiffness`=kp，`damping`=kd |
| `isaaclab.sim.UsdFileCfg` | 从 USD 文件生成机器人 |
| `isaaclab.sim.RigidBodyPropertiesCfg` | 刚体属性：重力、阻尼、最大速度 |
| `isaaclab.sim.ArticulationRootPropertiesCfg` | 关节体属性：自碰撞、求解器迭代 |
| `isaaclab.terrains.TerrainImporterCfg` | 地形配置；`terrain_type="plane"` 是平地 |
| `isaaclab.managers.SceneEntityCfg` | 引用场景里的实体（机器人、传感器、body） |
| `isaaclab.managers.RewTerm` | 一个奖励项：函数 + 权重 + 参数 |
| `isaaclab.managers.ObsTerm` | 一个观测项 |
| `isaaclab.managers.DoneTerm` | 一个终止条件 |
| `isaaclab.managers.EventTerm` | 随机化 / 重置事件 |
| `isaaclab_tasks.utils.parse_env_cfg` | 根据任务 id 取出环境配置，并设置 `num_envs`、`device` |
| `isaaclab_tasks.utils.parse_cfg.load_cfg_from_registry` | 根据任务 id 取出 PPO 配置 |

### 3.4 Isaac Lab 的 RL 适配

| 包 / 类 | 作用 |
|---|---|
| `isaaclab_rl.rsl_rl.RslRlVecEnvWrapper` | 把 Isaac Lab 环境包装成 rsl_rl 的向量环境 |
| `isaaclab_rl.rsl_rl.RslRlOnPolicyRunnerCfg` | 训练器总配置 |
| `isaaclab_rl.rsl_rl.RslRlPpoActorCriticCfg` | actor/critic 网络结构 |
| `isaaclab_rl.rsl_rl.RslRlPpoAlgorithmCfg` | PPO 超参数 |
| `rsl_rl.runners.OnPolicyRunner` | 真正执行 PPO 训练 / 加载 checkpoint |

### 3.5 资产转换

| 包 / 类 | 作用 |
|---|---|
| `isaaclab.sim.converters.UrdfConverter` | URDF → USD |
| `UrdfConverterCfg` | 转换参数：输入、输出、是否固定基座、是否合并 fixed joint |

### 3.6 深度学习

| 包 | 作用 |
|---|---|
| `torch` | 神经网络、GPU 张量、`torch.inference_mode()` |
| `torch.backends.cuda.matmul.allow_tf32` | 允许 TF32，提高 GPU 矩阵乘法速度 |

---

## 4. 关键术语

| 术语 | 解释 |
|---|---|
| **USD** | Omniverse 的场景文件格式，Isaac Lab 只能加载 USD |
| **Articulation** | 带关节的机器人（多个 link + joint） |
| **DOF** | 自由度，T800 是 25 个驱动关节 + 1 个浮动基座 |
| **action** | 策略输出的 25 个数，单位是“归一化增量” |
| **observation** | 喂给网络的向量：速度、姿态、关节角、指令、上一动作 |
| **command** | 期望速度 `[vx, vy, wz]`，每个 episode 随机采样 |
| **reward** | 每一步的分数，PPO 靠它调动作概率 |
| **terminated** | 摔倒 / 基座着地，回合提前结束 |
| **truncated** | 到 20 秒上限，回合正常结束 |
| **curriculum** | 课程学习，从简单到难（本项目已关掉地形课程） |
| **domain randomization** | 随机化摩擦、质量、推力，让策略更鲁棒 |
| **PPO** | 一种 on-policy 强化学习算法，限制每次策略更新幅度 |
| **actor / critic** | actor 输出动作，critic 估计状态价值 |
| **GAE** | 广义优势估计，用 critic 降低方差 |
| **checkpoint** | 保存的模型权重 `.pt` |

---

## 5. 每个文件负责什么

| 文件 | 职责 |
|---|---|
| `scripts/convert_t800.py` | URDF → USD；改写 `package://` 网格路径 |
| `scripts/check_env.py` | 检查 Isaac Lab / rsl_rl / CUDA / 资产是否齐全 |
| `scripts/train.py` | 启动 Isaac Sim、建环境、跑 PPO、存日志 |
| `scripts/play.py` | 加载 `.pt`，用策略在环境里走 |
| `source/.../assets/t800.py` | 机器人配置：USD、初始姿态、PD 增益 |
| `source/.../t800_env_cfg.py` | 环境配置：地形、指令、动作、奖励、终止 |
| `source/.../agents/rsl_rl_ppo_cfg.py` | PPO 超参数和网络结构 |
| `source/.../velocity/__init__.py` | 用 `gym.register` 注册任务 |
| `source/t800_lab/setup.py` | 把 `t800_lab` 装进当前 Python 环境 |

---

## 6. 建议的阅读顺序

1. `scripts/check_env.py` —— 先知道环境里有什么
2. `scripts/convert_t800.py` —— 知道 USD 怎么来的
3. `assets/t800.py` —— 机器人怎么加载
4. `t800_env_cfg.py` —— 环境、奖励怎么定义
5. `rsl_rl_ppo_cfg.py` —— PPO 参数
6. `train.py` / `play.py` —— 整体流程
7. 最后再看 `tasks/.../__init__.py` 的注册

---

## 7. 常见报错

| 报错 | 原因 | 解决 |
|---|---|---|
| `t800.usd` 不存在 | 没跑转换脚本 | `python scripts/convert_t800.py --headless` |
| `No module named isaaclab` | 没装 Isaac Lab 或没激活环境 | `conda activate isaaclab`，按官方装 |
| `ModuleNotFoundError: omni` | import 顺序错了 | 必须在 `AppLauncher` 之后再 import isaaclab |
| `Invalid prim path` / 找不到 `LINK_FOOT` | USD 里 body 名字不对 | 用 Isaac Sim 打开 USD 看层级，改 `feet` 列表 |
| `CUDA out of memory` | `num_envs` 太大 | 降到 1024 / 512 |
| pip 依赖冲突 | Isaac Sim pin 的版本和 Isaac Lab 不一致 | 先忽略；运行报错再按提示降版本 |
