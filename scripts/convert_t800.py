"""把 T800 的 URDF 转换成 Isaac Lab 使用的 USD。

==============================================================================
一、先用大白话讲：URDF 和 USD 是什么
==============================================================================
URDF：
    一种 XML 文件，描述机器人有哪些连杆、关节、质量、碰撞形状。
    SDK 给的就是 URDF。

USD：
    Isaac Sim / Omniverse 使用的场景文件格式。
    Isaac Sim 不能直接加载 URDF，必须先转成 .usd。

所以这个脚本 = “文件格式转换器”。

==============================================================================
二、代码里用到的 4 个库（每个都给小例子）
==============================================================================

------------------------------------------------------------------------------
1) argparse：解析命令行参数
------------------------------------------------------------------------------
作用：
    让你在终端里写 `python convert_t800.py --force --headless`，
    Python 能读到 `--force`、`--headless` 这些开关。

小例子：
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="t800")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    print(args.name)    # 没传就是 "t800"，传了 --name abc 就是 "abc"
    print(args.force)   # 没传是 False，传了 --force 是 True

关键方法：
    ArgumentParser(description=...)  : 创建解析器
    add_argument("--xxx", ...)       : 声明一个参数
    parse_args()                     : 真正读取 sys.argv，返回 Namespace

`action="store_true"` 的意思是：
    这个参数不带值，出现就是 True，不出现就是 False。

------------------------------------------------------------------------------
2) pathlib.Path：比字符串更好用的路径
------------------------------------------------------------------------------
作用：
    拼路径、读文件、写文件、建目录。

小例子：
    from pathlib import Path

    p = Path.home() / "mujoco" / "10.py"
    print(p)            # /home/akihisa/mujoco/10.py
    print(p.parent)     # /home/akihisa/mujoco
    print(p.name)       # 10.py
    print(p.exists())   # True / False

    p2 = Path("out") / "a.txt"
    p2.parent.mkdir(parents=True, exist_ok=True)
    p2.write_text("hello")
    print(p2.read_text())   # hello

关键方法：
    Path(__file__)        : 当前 Python 文件的路径
    .resolve()            : 转成绝对路径
    .parents[1]           : 往上一级（0 是当前文件所在目录）
    /                     : 拼路径，等价于字符串拼接，但更安全
    .read_text()          : 读文本
    .write_text()         : 写文本
    .mkdir(parents=True, exist_ok=True) : 建目录，父目录不存在也一起建
    .exists()             : 判断文件/目录是否存在

------------------------------------------------------------------------------
3) AppLauncher：启动 Isaac Sim
------------------------------------------------------------------------------
作用：
    Isaac Lab 的很多功能依赖 Omniverse / Isaac Sim 的底层组件。
    必须先启动 Isaac Sim，才能 import UrdfConverter。

小例子：
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # 这之后才能 import isaaclab 的仿真/转换模块
    from isaaclab.sim.converters import UrdfConverter

    # 用完关闭
    simulation_app.close()

`AppLauncher.add_app_launcher_args(parser)` 会自动加这些参数：
    --headless        不打开 3D 窗口
    --device          cuda:0 / cpu
    --enable_cameras  是否启用相机

所以脚本里没有手写 `--headless`，但你可以传。

------------------------------------------------------------------------------
4) UrdfConverter / UrdfConverterCfg：真正做转换
------------------------------------------------------------------------------
作用：
    UrdfConverterCfg 是“转换参数”；
    UrdfConverter 是“执行转换”的对象。

小例子：
    cfg = UrdfConverterCfg(
        asset_path="robot.urdf",        # 输入
        usd_dir="out",                 # 输出目录
        usd_file_name="robot.usd",     # 输出文件名
        fix_base=False,                # 机器人自由移动
        merge_fixed_joints=False,      # 保留 fixed link
        force_usd_conversion=True,     # 每次都重新转
    )
    converter = UrdfConverter(cfg)
    print(converter.usd_path)          # 生成后的 USD 路径

字段解释：
    fix_base=False：
        人形机器人是浮动基座，不能钉在地上；True 会把它固定住。
    merge_fixed_joints=False：
        URDF 里有些 link 是 fixed joint（脚、手腕）。
        True 会把它们合并到父 link，可能导致 LINK_FOOT 消失，
        接触传感器和奖励就找不到脚了。
    force_usd_conversion=True：
        即使 USD 已存在也重新转换，避免用到旧文件。
    joint_drive：
        URDF 转 USD 时必须给一组默认关节 PD 增益，
        否则会报 "Missing values ... stiffness"。
        这只是 USD 里的初始 drive，训练时会被 ImplicitActuatorCfg 覆盖。

==============================================================================
三、这个脚本的执行流程
==============================================================================
1. argparse 读命令行参数
2. AppLauncher 启动 Isaac Sim
3. rewrite_urdf：把 package:// 路径改成相对路径
4. 创建输出目录
5. 配置 UrdfConverterCfg
6. UrdfConverter 执行转换
7. 打印 USD 路径，关闭 Isaac Sim

==============================================================================
四、为什么要把 package:// 改掉
==============================================================================
URDF 里的网格路径是：
    package://resource/robot/t800/meshes/LINK_BASE.dae

这是 ROS 的写法，Isaac Sim 不认识 `package://`。
所以脚本把它替换成：
    ../meshes/LINK_BASE.dae

因为 `serial_t800_local.urdf` 在 `assets/t800/urdf/` 下，
`../meshes/` 正好指向 `assets/t800/meshes/`。
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

# __file__ 是当前文件：t800_isaaclab/scripts/convert_t800.py
# .resolve() 转绝对路径
# .parents[0] 是 scripts/，.parents[1] 是项目根目录 t800_isaaclab/
ROOT = Path(__file__).resolve().parents[1]

URDF_SRC = ROOT / "assets/t800/urdf/serial_t800.urdf"
URDF_LOCAL = ROOT / "assets/t800/urdf/serial_t800_local.urdf"
USD_DIR = ROOT / "source/t800_lab/t800_lab/assets"
USD_NAME = "t800.usd"


def rewrite_urdf():
    """把 package:// 路径改成相对路径，写到 serial_t800_local.urdf。

    参数：无
    返回：无（会生成 URDF_LOCAL 文件）

    为什么不在原文件上改？
        原文件是从 SDK 复制来的，保留它方便对比/重新生成。
    """
    # read_text()：一次读出整个文本文件，返回 str
    text = URDF_SRC.read_text()

    # str.replace(旧, 新)：把所有旧字符串替换成新字符串
    text = text.replace("package://resource/robot/t800/", "../")

    # write_text()：把字符串写回文件
    URDF_LOCAL.write_text(text)
    print(f"[信息] 已生成 {URDF_LOCAL}")


def main():
    # ---- 1. 解析命令行参数 ----
    parser = argparse.ArgumentParser(description="T800 URDF -> USD")
    # 这个参数目前没有真正用到（下面 force_usd_conversion 写死 True），
    # 留着是为了以后可以改成“只有 --force 才重新转换”。
    parser.add_argument("--force", action="store_true", help="强制重新转换")
    # AppLauncher 会自动添加 --headless、--device 等参数
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # ---- 2. 启动 Isaac Sim ----
    # 必须在使用 UrdfConverter 之前启动，否则会缺少 Omniverse 组件
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # ---- 3. 启动之后再 import 转换器 ----
    # 注意：UrdfConverter 不能放到文件顶部 import，
    # 因为它在 Isaac Sim 启动前不可用。
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    # ---- 4. 改写 URDF 里的网格路径 ----
    rewrite_urdf()

    # ---- 5. 创建输出目录 ----
    # parents=True：父目录不存在也一起创建
    # exist_ok=True：目录已存在也不报错
    USD_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 6. 配置转换参数 ----
    cfg = UrdfConverterCfg(
        asset_path=str(URDF_LOCAL),       # 输入 URDF（改写后的版本）
        usd_dir=str(USD_DIR),             # 输出目录
        usd_file_name=USD_NAME,           # 输出文件名
        fix_base=False,                   # 浮动基座，不固定
        merge_fixed_joints=False,         # 保留脚、手腕等 fixed link
        force_usd_conversion=True,        # 每次都重新转，避免旧 USD
        # URDF 转 USD 必须给一组默认 PD 增益，否则校验会报错。
        # 这里只是占位，训练时会被 ImplicitActuatorCfg 覆盖。
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",           # 用“力矩”驱动
            target_type="position",       # 目标是“位置”
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=100.0,          # 默认 kp
                damping=1.0,              # 默认 kd
            ),
        ),
    )

    # ---- 7. 执行转换 ----
    converter = UrdfConverter(cfg)
    print(f"[完成] USD 已生成：{converter.usd_path}")
    print("[下一步] python scripts/train.py --task Isaac-T800-Locomotion-v0 --headless")

    # ---- 8. 关闭 Isaac Sim ----
    simulation_app.close()


# 只有直接运行这个文件时才执行 main()
# 如果被别的文件 import，就不会自动执行
if __name__ == "__main__":
    main()
