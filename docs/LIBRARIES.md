# 代码里的库怎么用（小例子版）

这份文档配合 `docs/GLOSSARY.md` 看：GLOSSARY 解释名词，这里解释“代码怎么写”。

---

## 1. argparse：读命令行参数

```python
import argparse

parser = argparse.ArgumentParser(description="demo")
parser.add_argument("--task", type=str, default="Isaac-T800-Mimic-v0")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--headless", action="store_true")

args = parser.parse_args()
print(args.task)       # 字符串
print(args.num_envs)   # 整数
print(args.headless)   # True / False
```

常用：

- `add_argument("--x", type=int, default=1)`：带值参数
- `action="store_true"`：开关，不带值
- `parse_args()`：读取终端输入

---

## 2. pathlib：路径

```python
from pathlib import Path

p = Path.home() / "t800_isaaclab" / "README.md"
print(p.exists())
print(p.parent)
print(p.name)
print(p.read_text()[:20])

out = Path("logs") / "a.txt"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("hello")
```

常用：

- `/` 拼路径
- `.resolve()` 绝对路径
- `.parents[1]` 上一级目录
- `.read_text()` / `.write_text()`
- `.mkdir(parents=True, exist_ok=True)`

---

## 3. os / sys / subprocess / shutil / glob

```python
import os, sys, shutil, subprocess, glob

print(os.environ.get("CONDA_DEFAULT_ENV"))
print(sys.executable)
print(shutil.which("nvidia-smi"))

out = subprocess.check_output(["nvidia-smi"], text=True)
print(out[:80])

files = glob.glob("logs/**/*.pt", recursive=True)
```

用途：

- `os.environ`：读环境变量
- `sys.executable`：当前 Python 解释器
- `subprocess`：调用外部命令
- `shutil.which`：找可执行文件
- `glob`：按通配符找文件

---

## 4. numpy

```python
import numpy as np

a = np.array([1, 2, 3], dtype=np.float32)
print(a.shape, a.mean(), a.max())
np.savez_compressed("data.npz", q=a)
d = np.load("data.npz")
print(d["q"])
```

用途：

- 存参考轨迹、读 npz
- 数组运算
- `np.clip`、`np.linalg.norm`

---

## 5. torch

```python
import torch

x = torch.zeros((2, 3), device="cuda:0")
y = torch.ones_like(x)
print((x + y).shape)

with torch.inference_mode():
    z = x @ y.T
```

用途：

- 神经网络输入输出
- GPU 张量
- `torch.inference_mode()` 推理时关梯度

---

## 6. gymnasium（gym）

```python
import gymnasium as gym

env = gym.make("Isaac-T800-Mimic-v0", cfg=env_cfg)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

用途：

- 强化学习环境标准接口
- `gym.register` 注册任务
- `gym.make` 创建环境

---

## 7. isaaclab.app.AppLauncher

```python
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# 这之后才能 import isaaclab.sim / converters / envs
simulation_app.close()
```

用途：

- 启动 Isaac Sim
- 自动加 `--headless`、`--device` 参数
- 必须在其它 isaaclab 模块之前调用

---

## 8. isaaclab.utils.configclass

```python
from isaaclab.utils import configclass

@configclass
class MyCfg:
    num_envs: int = 4096
    dt: float = 0.005

cfg = MyCfg()
print(cfg.to_dict())
cfg2 = cfg.replace(num_envs=8192)
```

用途：

- 写配置类，类似 dataclass
- 支持嵌套、`to_dict()`、`replace()`

---

## 9. isaaclab.managers

```python
from isaaclab.managers import (
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    EventTermCfg as EventTerm,
    SceneEntityCfg,
)
```

用途：

- `ObsTerm`：一个观测项
- `RewTerm`：一个奖励项
- `DoneTerm`：一个终止条件
- `EventTerm`：reset/随机化事件
- `SceneEntityCfg`：引用场景里的机器人、传感器、body

`SceneEntityCfg("contact_forces", body_names=["LINK_FOOT_L"])` 的意思是：
“从 contact_forces 传感器里，找名字叫 LINK_FOOT_L 的 body”。

---

## 10. isaaclab.assets.ArticulationCfg / actuators

```python
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

robot_cfg = ArticulationCfg(
    spawn=...,
    init_state=...,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*KNEE.*"],
            stiffness=180.0,
            damping=5.0,
        )
    },
)
```

用途：

- `ArticulationCfg`：机器人整体配置
- `ImplicitActuatorCfg`：隐式 PD，`stiffness=kp`、`damping=kd`

---

## 11. isaaclab.sim

```python
import isaaclab.sim as sim_utils

sim_utils.UsdFileCfg(usd_path="robot.usd")
sim_utils.RigidBodyPropertiesCfg(disable_gravity=False)
sim_utils.DomeLightCfg(intensity=750.0)
```

用途：

- 生成 USD 资产
- 刚体属性
- 灯光、材质

---

## 12. isaaclab.sim.converters

```python
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

cfg = UrdfConverterCfg(
    asset_path="robot.urdf",
    usd_dir="out",
    usd_file_name="robot.usd",
    fix_base=False,
    merge_fixed_joints=False,
    force_usd_conversion=True,
)
converter = UrdfConverter(cfg)
print(converter.usd_path)
```

用途：

- URDF → USD
- 必须在 AppLauncher 启动后 import

---

## 13. isaaclab_tasks.utils

```python
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

env_cfg = parse_env_cfg("Isaac-T800-Mimic-v0", device="cuda:0", num_envs=4096)
agent_cfg = load_cfg_from_registry("Isaac-T800-Mimic-v0", "rsl_rl_cfg_entry_point")
```

用途：

- `parse_env_cfg`：按任务 id 取环境配置
- `load_cfg_from_registry`：按任务 id 取 PPO 配置

---

## 14. isaaclab_rl.rsl_rl

```python
from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

env = RslRlVecEnvWrapper(env, clip_actions=100.0)
```

用途：

- `RslRlVecEnvWrapper`：把 Isaac Lab 环境包装成 rsl_rl 能用的向量环境
- 三个 `Cfg`：训练器、网络结构、PPO 超参数

---

## 15. rsl_rl.runners.OnPolicyRunner

```python
from rsl_rl.runners import OnPolicyRunner

runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir="logs", device="cuda:0")
runner.load("model_2999.pt")
policy = runner.get_inference_policy(device="cuda:0")
actions = policy(obs)
```

用途：

- PPO 训练
- 加载 checkpoint
- 取推理策略

---

## 16. mujoco / MNN

```python
import mujoco
import MNN

model = mujoco.MjModel.from_xml_path("serial_t800.xml")
data = mujoco.MjData(model)

net = MNN.Interpreter("policy.mnn")
session = net.createSession()
```

用途：

- MuJoCo 加载 MJCF、推进物理
- MNN 加载 `.mnn` 策略推理

---

## 17. setuptools / toml

```python
import toml
from setuptools import setup, find_packages

meta = toml.load("config/extension.toml")
setup(name="t800_lab", packages=find_packages(), version=meta["package"]["version"])
```

用途：

- `pip install -e source/t800_lab` 时执行
- `find_packages()` 自动找子包
- `toml.load` 读扩展元数据

---

## 18. pytest / unittest（本项目没用到）

```python
def test_add():
    assert 1 + 1 == 2
```

如果以后要写测试，可以用 pytest。
