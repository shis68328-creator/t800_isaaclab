"""把 T800 的 URDF 转换成 Isaac Lab 使用的 USD。

Isaac Lab / Isaac Sim 不能直接加载 MuJoCo 的 MJCF，需要先转成 USD。
这里用 Isaac Lab 自带的 UrdfConverter。

运行（必须能启动 Isaac Sim，Linux + NVIDIA）：
    cd t800_isaaclab
    python scripts/convert_t800.py --headless

生成：
    source/t800_lab/t800_lab/assets/t800.usd

如果 URDF 里的 mesh 路径是 package://，脚本会先复制一份并把路径改成
相对路径 ../meshes/...，避免 importer 找不到网格。
"""

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
URDF_SRC = ROOT / "assets/t800/urdf/serial_t800.urdf"
URDF_LOCAL = ROOT / "assets/t800/urdf/serial_t800_local.urdf"
USD_DIR = ROOT / "source/t800_lab/t800_lab/assets"
USD_NAME = "t800.usd"


def rewrite_urdf():
    """把 package:// 路径改成相对于 URDF 的 ../meshes/。"""
    text = URDF_SRC.read_text()
    text = text.replace("package://resource/robot/t800/", "../")
    URDF_LOCAL.write_text(text)
    print(f"[信息] 已生成 {URDF_LOCAL}")


def main():
    parser = argparse.ArgumentParser(description="T800 URDF -> USD")
    parser.add_argument("--force", action="store_true", help="强制重新转换")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # 启动 Isaac Sim（转换 USD 需要 Omniverse）
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # 必须在 AppLauncher 之后再导入 converter
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    rewrite_urdf()
    USD_DIR.mkdir(parents=True, exist_ok=True)

    cfg = UrdfConverterCfg(
        asset_path=str(URDF_LOCAL),
        usd_dir=str(USD_DIR),
        usd_file_name=USD_NAME,
        fix_base=False,
        merge_fixed_joints=False,
        force_usd_conversion=True,
    )
    converter = UrdfConverter(cfg)
    print(f"[完成] USD 已生成：{converter.usd_path}")
    print("[下一步] python scripts/train.py --task Isaac-T800-Locomotion-v0 --headless")

    simulation_app.close()


if __name__ == "__main__":
    main()
