"""把 T800 的 URDF 转换成 Isaac Lab 使用的 USD。

======================================================================
为什么需要转换
======================================================================
Isaac Sim 的场景格式是 USD（Universal Scene Description），
而 SDK 提供的是 URDF（XML）和 MuJoCo MJCF（XML）。
Isaac Lab 不能直接 spawn URDF/MJCF，所以要用 Isaac Lab 自带的
UrdfConverter 把 URDF 转成 .usd。

转换会做：
    - 解析 URDF 的 link / joint / 惯性 / 碰撞
    - 导入 .dae / .obj 网格
    - 生成 USD 层级结构
    - 给关节创建 PhysX drive

URDF 里的网格路径是：
    package://resource/robot/t800/meshes/LINK_BASE.dae
这是 ROS 的 package:// 写法，Isaac Sim 不认识。
所以脚本先把它改写成相对路径：
    ../meshes/LINK_BASE.dae
这样从 urdf/ 目录出发就能找到 meshes/。

======================================================================
用到的包
======================================================================
AppLauncher：
    启动 Isaac Sim。USD 转换需要 Omniverse 的 USD / PhysX 组件，
    所以必须先启动 App，再 import UrdfConverter。
UrdfConverterCfg：
    转换参数：输入 URDF、输出目录、文件名、是否固定基座、
    是否合并 fixed joint、是否强制重新转换。
UrdfConverter：
    真正执行转换的类，converter.usd_path 是生成的 USD 路径。
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
URDF_SRC = ROOT / "assets/t800/urdf/serial_t800.urdf"
URDF_LOCAL = ROOT / "assets/t800/urdf/serial_t800_local.urdf"
USD_DIR = ROOT / "source/t800_lab/t800_lab/assets"
USD_NAME = "t800.usd"


def rewrite_urdf():
    """把 package:// 路径改成相对路径，写到 serial_t800_local.urdf。"""
    text = URDF_SRC.read_text()
    # package://resource/robot/t800/ -> ../，相对 urdf/ 目录
    text = text.replace("package://resource/robot/t800/", "../")
    URDF_LOCAL.write_text(text)
    print(f"[信息] 已生成 {URDF_LOCAL}")


def main():
    parser = argparse.ArgumentParser(description="T800 URDF -> USD")
    parser.add_argument("--force", action="store_true", help="强制重新转换")
    # AppLauncher 会添加 --headless、--device 等参数
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # 先启动 Isaac Sim，再 import converter
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    rewrite_urdf()
    USD_DIR.mkdir(parents=True, exist_ok=True)

    cfg = UrdfConverterCfg(
        asset_path=str(URDF_LOCAL),       # 输入 URDF
        usd_dir=str(USD_DIR),             # 输出目录
        usd_file_name=USD_NAME,           # 输出文件名
        fix_base=False,                   # 机器人是浮动基座，不固定
        merge_fixed_joints=False,         # 保留 fixed link（脚、手腕）
        force_usd_conversion=True,        # 每次都重新转
        # URDF 转 USD 时必须给一组默认的关节 PD 增益，
        # 否则 UrdfConverterCfg.validate() 会报
        # "Missing values ... joint_drive.gains.stiffness"。
        # 这只是 USD 里的初始 drive；真正训练时会被
        # assets/t800.py 里的 ImplicitActuatorCfg 覆盖。
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=100.0,
                damping=1.0,
            ),
        ),
    )
    converter = UrdfConverter(cfg)

    print(f"[完成] USD 已生成：{converter.usd_path}")
    print("[下一步] python scripts/train.py --task Isaac-T800-Locomotion-v0 --headless")

    simulation_app.close()


if __name__ == "__main__":
    main()
