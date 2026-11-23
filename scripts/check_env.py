"""检查 Isaac Lab 的 conda 环境是否满足 T800 训练要求。

======================================================================
为什么需要这个脚本
======================================================================
Isaac Lab 不是一个 pip 包就能装完的东西，它由很多部分组成：
    - Isaac Sim（NVIDIA 的仿真器，一堆 isaacsim-* 包）
    - isaaclab（核心库）
    - isaaclab_assets（官方机器人资产）
    - isaaclab_tasks（官方任务）
    - isaaclab_rl（RL 框架适配）
    - rsl_rl（PPO 训练库）
    - gymnasium（环境接口）
    - torch（GPU 上的神经网络）
只要有一个没装好，训练就会在 import 阶段报错。

这个脚本不启动 Isaac Sim，只检查：
    1. Python 包是否安装（用 importlib.metadata 查 pip 元数据）
    2. Python 模块路径是否存在
    3. CUDA / 显卡 / 驱动
    4. T800 的 URDF / meshes / USD 是否存在
    5. t800_lab 是否已安装

为什么不用 import 来检查？
    直接 import isaaclab 会触发 Omniverse 初始化，没有启动 AppLauncher
    时会报错。用 importlib.metadata 查“包有没有装”更安全，也不会启动仿真。

======================================================================
用到的包
======================================================================
importlib.metadata：
    读取已安装 pip 包的版本号（比如 isaaclab==0.47.2）。
importlib.util.find_spec：
    查找模块的文件路径，不执行模块代码。
importlib.util.find_spec("isaaclab") 只查路径，不 import。
torch：
    查 CUDA 是否可用、显卡型号。
subprocess：
    调用 nvidia-smi 查驱动。
pathlib / os / sys：
    路径、环境变量、退出码。
"""

import importlib.metadata as md
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (显示名, pip 包名, import 名, 是否必需)
# pip 包名和 import 名不一定一样，比如 rsl-rl-lib 的 import 名是 rsl_rl。
PACKAGES = [
    ("Isaac Lab", "isaaclab", "isaaclab", True),
    ("Isaac Lab Assets", "isaaclab_assets", "isaaclab_assets", True),
    ("Isaac Lab RL", "isaaclab_rl", "isaaclab_rl", True),
    ("Isaac Lab Tasks", "isaaclab_tasks", "isaaclab_tasks", True),
    ("rsl_rl", "rsl-rl-lib", "rsl_rl", True),
    ("PyTorch", "torch", "torch", True),
    ("Gymnasium", "gymnasium", "gymnasium", True),
    ("NumPy", "numpy", "numpy", True),
    ("TensorBoard", "tensorboard", "tensorboard", False),
    ("psutil", "psutil", "psutil", True),
    ("toml", "toml", "toml", True),
]


def dist_version(name: str):
    """返回 pip 包版本；没装返回 None。"""
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def module_path(name: str):
    """返回模块文件路径；找不到返回 None。不会执行模块代码。"""
    try:
        spec = importlib.util.find_spec(name)
        return spec.origin if spec else None
    except Exception:
        return None


def main():
    print("=" * 64)
    print(" T800 Isaac Lab 环境检查")
    print("=" * 64)
    print(f" Python      : {sys.version.split()[0]}")
    print(f" 解释器      : {sys.executable}")
    print(f" conda 环境  : {os.environ.get('CONDA_DEFAULT_ENV', '(未激活)')}")
    print(f" 项目目录    : {ROOT}")
    print()

    missing = []
    print(f" {'组件':<18} {'版本':<16} {'状态'}")
    print("-" * 64)
    for label, dist, module, required in PACKAGES:
        version = dist_version(dist)
        path = module_path(module)
        ok = version is not None or path is not None
        if not ok and required:
            missing.append(label)
        status = "OK" if ok else ("WARN" if not required else "MISSING")
        print(f" {label:<18} {(version or '-'):<16} {status}")
        if path:
            print(f"   path: {path}")

    print()
    # ---- CUDA ----
    try:
        import torch

        print(f" torch.version.cuda : {torch.version.cuda}")
        print(f" CUDA available    : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f" GPU               : {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f" [警告] 导入 torch 失败：{exc}")
        missing.append("PyTorch")

    # ---- 驱动 ----
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                 "--format=csv,noheader"],
                text=True,
                timeout=10,
            ).strip()
            print(f" nvidia-smi        : {out}")
        except Exception as exc:
            print(f" [警告] nvidia-smi 执行失败：{exc}")
    else:
        print(" [警告] 找不到 nvidia-smi")

    # ---- 资产 ----
    print()
    usd = ROOT / "source/t800_lab/t800_lab/assets/t800.usd"
    urdf = ROOT / "assets/t800/urdf/serial_t800.urdf"
    meshes = ROOT / "assets/t800/meshes"
    print(f" T800 URDF         : {'OK' if urdf.exists() else 'MISSING'}")
    print(f" T800 meshes       : {'OK' if meshes.exists() else 'MISSING'}")
    print(f" T800 USD          : {'OK' if usd.exists() else '未生成（先跑 convert_t800.py）'}")

    # ---- 本项目包 ----
    spec = importlib.util.find_spec("t800_lab")
    print(f" t800_lab          : {'OK' if spec else '未安装（pip install -e source/t800_lab）'}")

    print()
    if missing:
        print(f"[结果] 缺少组件：{', '.join(missing)}")
        print("       请先按 Isaac Lab 官方文档安装对应环境。")
        return 1

    print("[结果] 基础依赖齐全。")
    if not usd.exists():
        print("       下一步：python scripts/convert_t800.py --headless")
    else:
        print("       下一步：python scripts/train.py --task Isaac-T800-Locomotion-v0 "
              "--num_envs 1024 --headless")
    return 0


if __name__ == "__main__":
    sys.exit(main())
