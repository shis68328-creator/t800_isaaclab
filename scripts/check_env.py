"""检查 Ubuntu 上的 Isaac Lab conda 环境是否满足 T800 训练要求。

这个脚本不启动 Isaac Sim，只查 Python 包、路径、CUDA 和资产文件，
所以可以在任何终端里直接跑。

用法：
    conda activate Isaaclab
    cd t800_isaaclab
    python scripts/check_env.py
"""

import importlib.metadata as md
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (显示名, pip 包名, import 名)
PACKAGES = [
    ("Isaac Lab", "isaaclab", "isaaclab"),
    ("Isaac Lab Assets", "isaaclab_assets", "isaaclab_assets"),
    ("Isaac Lab RL", "isaaclab_rl", "isaaclab_rl"),
    ("Isaac Lab Tasks", "isaaclab_tasks", "isaaclab_tasks"),
    ("rsl_rl", "rsl-rl-lib", "rsl_rl"),
    ("PyTorch", "torch", "torch"),
    ("Gymnasium", "gymnasium", "gymnasium"),
    ("NumPy", "numpy", "numpy"),
    ("TensorBoard", "tensorboard", "tensorboard"),
    ("psutil", "psutil", "psutil"),
    ("toml", "toml", "toml"),
]


def dist_version(name: str):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def module_path(name: str):
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
    optional = {"TensorBoard"}
    print(f" {'组件':<18} {'版本':<16} {'状态'}")
    print("-" * 64)
    for label, dist, module in PACKAGES:
        version = dist_version(dist)
        path = module_path(module)
        ok = version is not None or path is not None
        if not ok and label not in optional:
            missing.append(label)
        status = "OK" if ok else ("WARN" if label in optional else "MISSING")
        print(f" {label:<18} {(version or '-'):<16} {status}")
        if path:
            print(f"   path: {path}")

    print()
    # CUDA
    try:
        import torch

        print(f" torch.version.cuda : {torch.version.cuda}")
        print(f" CUDA available    : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f" GPU               : {torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f" [警告] 导入 torch 失败：{exc}")
        missing.append("PyTorch")

    # 驱动
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

    # 资产和生成文件
    print()
    usd = ROOT / "source/t800_lab/t800_lab/assets/t800.usd"
    urdf = ROOT / "assets/t800/urdf/serial_t800.urdf"
    meshes = ROOT / "assets/t800/meshes"
    print(f" T800 URDF         : {'OK' if urdf.exists() else 'MISSING'}")
    print(f" T800 meshes       : {'OK' if meshes.exists() else 'MISSING'}")
    print(f" T800 USD          : {'OK' if usd.exists() else '未生成（先跑 convert_t800.py）'}")

    # 本项目包是否可导入
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
        print("       下一步：python scripts/train.py --task Isaac-T800-Locomotion-v0 --num_envs 1024 --headless")
    return 0


if __name__ == "__main__":
    sys.exit(main())
