"""用 MuJoCo + MNN 走路策略录制参考步态。

==============================================================================
先用大白话解释：为什么要录这个
==============================================================================
强化学习如果从零开始，让机器人自己“随机试”出走路，非常难。
所以我们先找一个已经会走路的老师（SDK 的 MNN 策略），
让它走一遍，把 25 个关节角每秒 100 次记下来。

这份记录就是“参考动作”，后面 Isaac Lab 里的学生只需要跟着它走。

比喻：
    MNN 策略 = 会走路的老师
    这个脚本 = 录像机
    walk_ref.npz = 录下来的教学视频（数字版）
    Isaac Lab 的策略 = 看视频学习的学生

==============================================================================
关键名词
==============================================================================
MuJoCo：
    物理仿真引擎，负责算“施加这些力矩后机器人会怎么动”。

MNN：
    一个轻量级神经网络推理库。SDK 的走路策略是 .mnn 文件，
    用 MNN 才能在 Python 里加载和推理。

qpos：
    MuJoCo 里的“位置数组”。前 7 个是基座（x,y,z + 四元数），
    后面 25 个是关节角。

qvel：
    MuJoCo 里的“速度数组”。前 6 个是基座线速度/角速度，
    后面 25 个是关节角速度。

控制周期：
    每隔多久让策略输出一次动作。这里是 0.01 秒 = 100 Hz。

==============================================================================
输出文件里有什么
==============================================================================
    joint_names   : 25 个关节名（MuJoCo 顺序）
    joint_pos     : (T, 25) 关节角，弧度
    joint_vel     : (T, 25) 关节速度
    base_pos      : (T, 3)  基座位置
    base_quat     : (T, 4)  基座四元数 (w, x, y, z)
    base_lin_vel  : (T, 3)
    base_ang_vel  : (T, 3)
    dt            : 0.01 秒
    command       : 录制时的目标速度

注意：
    joint_names 很重要。Isaac Lab 的关节顺序和 MuJoCo 不一定一样，
    后面必须按名字重新排序，不能直接按列号用。

==============================================================================
用法
==============================================================================
    conda activate mujoco
    python t800_isaaclab/scripts/record_mnn_reference.py \
        --mujoco-dir ~/mujoco_t800 \
        --out ~/t800_isaaclab/assets/reference/walk_ref.npz
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np


def load_walker_module(mujoco_dir: Path):
    """从 mujoco 目录加载 10.py 里的 Walker 类。

    参数：
        mujoco_dir：包含 10.py、SDK、MNN 策略的目录

    返回：
        Python 模块对象，里面有 Walker、POLICY_PATH、CONTROL_DT 等。
    """
    script = mujoco_dir / "10.py"
    if not script.exists():
        raise FileNotFoundError(f"找不到 {script}")

    # importlib 可以按文件路径加载模块，即使文件名以数字开头也可以。
    spec = importlib.util.spec_from_file_location("t800_mnn_10", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules["t800_mnn_10"] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mujoco-dir", type=str, default=str(Path.home() / "mujoco_t800"))
    parser.add_argument("--out", type=str, default="assets/reference/walk_ref.npz")
    parser.add_argument("--command", type=float, default=0.5, help="参考步态的目标 vx")
    parser.add_argument("--warmup", type=float, default=2.0, help="先走几秒再录")
    parser.add_argument("--duration", type=float, default=2.0, help="录几秒")
    args = parser.parse_args()

    mujoco_dir = Path(args.mujoco_dir).expanduser()
    out = Path(args.out)
    if not out.is_absolute():
        out = Path.cwd() / out
    out.parent.mkdir(parents=True, exist_ok=True)

    # ---- 1. 加载会走路的老师 ----
    mod = load_walker_module(mujoco_dir)
    walker = mod.Walker(mod.POLICY_PATH)

    # 设置目标速度。command=1.0 时 MNN 走得比较明显；
    # command=0.5 时它经常原地踏步，不适合当参考。
    walker.command[:] = [args.command, 0.0, 0.0]
    walker.reset()

    warmup_steps = int(round(args.warmup / mod.CONTROL_DT))
    record_steps = int(round(args.duration / mod.CONTROL_DT))

    # ---- 2. 预热 ----
    # 刚 reset 时机器人还在调整姿态，前几秒的步态不稳定，不能直接录。
    print(f"预热 {warmup_steps} 步（{args.warmup:.1f}s）...")
    for _ in range(warmup_steps):
        walker.step()

    # ---- 3. 建立“关节名 -> qpos/qvel 下标”的映射 ----
    # MuJoCo 里关节类型有很多种，这里只取 hinge（旋转关节）。
    joint_names = []
    qpos_idx = []
    dof_idx = []
    for j in range(walker.model.njnt):
        if walker.model.jnt_type[j] == mod.mujoco.mjtJoint.mjJNT_HINGE:
            joint_names.append(walker.model.joint(j).name)
            qpos_idx.append(int(walker.model.jnt_qposadr[j]))
            dof_idx.append(int(walker.model.jnt_dofadr[j]))

    print(f"关节数：{len(joint_names)}")

    # ---- 4. 开始录 ----
    joint_pos = []
    joint_vel = []
    base_pos = []
    base_quat = []
    base_lin_vel = []
    base_ang_vel = []

    for _ in range(record_steps):
        # walker.step() 会：
        #   拼观测 -> MNN 推理 -> 算 PD 力矩 -> 推进 5 个物理步
        walker.step()

        joint_pos.append(walker.data.qpos[qpos_idx].copy())
        joint_vel.append(walker.data.qvel[dof_idx].copy())
        base_pos.append(walker.data.qpos[0:3].copy())
        base_quat.append(walker.data.qpos[3:7].copy())
        base_lin_vel.append(walker.data.qvel[0:3].copy())
        base_ang_vel.append(walker.data.qvel[3:6].copy())

    joint_pos = np.asarray(joint_pos, dtype=np.float32)
    joint_vel = np.asarray(joint_vel, dtype=np.float32)

    # ---- 5. 保存 ----
    np.savez_compressed(
        out,
        joint_names=np.array(joint_names),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        base_pos=np.asarray(base_pos, dtype=np.float32),
        base_quat=np.asarray(base_quat, dtype=np.float32),
        base_lin_vel=np.asarray(base_lin_vel, dtype=np.float32),
        base_ang_vel=np.asarray(base_ang_vel, dtype=np.float32),
        dt=np.float32(mod.CONTROL_DT),
        command=np.float32(args.command),
    )

    print(f"已保存：{out}")
    print(f"  joint_pos: {joint_pos.shape}, joint_vel: {joint_vel.shape}")
    print(f"  base_pos : {np.asarray(base_pos).shape}")
    print(f"  关节名   : {joint_names}")


if __name__ == "__main__":
    main()
