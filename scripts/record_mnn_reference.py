"""用 MuJoCo + MNN 走路策略录制参考步态。

输出：
    assets/reference/walk_ref.npz

里面包含：
    joint_names   : 25 个关节名（MuJoCo 顺序）
    joint_pos     : (T, 25) 关节角，弧度
    joint_vel     : (T, 25) 关节速度
    base_pos      : (T, 3)  基座位置
    base_quat     : (T, 4)  基座四元数 (w, x, y, z)
    base_lin_vel  : (T, 3)
    base_ang_vel  : (T, 3)
    dt            : 0.01 秒
    command       : 录制时的 vx

用法（在装了 mujoco + MNN 的环境里）：
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
    """从 mujoco 目录加载 10.py 里的 Walker 类。"""
    script = mujoco_dir / "10.py"
    if not script.exists():
        raise FileNotFoundError(f"找不到 {script}")

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

    mod = load_walker_module(mujoco_dir)
    walker = mod.Walker(mod.POLICY_PATH)
    walker.command[:] = [args.command, 0.0, 0.0]
    walker.reset()

    warmup_steps = int(round(args.warmup / mod.CONTROL_DT))
    record_steps = int(round(args.duration / mod.CONTROL_DT))

    print(f"预热 {warmup_steps} 步（{args.warmup:.1f}s）...")
    for _ in range(warmup_steps):
        walker.step()

    # MuJoCo 的 25 个 hinge 关节名和 qpos 下标
    joint_names = []
    qpos_idx = []
    dof_idx = []
    for j in range(walker.model.njnt):
        if walker.model.jnt_type[j] == mod.mujoco.mjtJoint.mjJNT_HINGE:
            joint_names.append(walker.model.joint(j).name)
            qpos_idx.append(int(walker.model.jnt_qposadr[j]))
            dof_idx.append(int(walker.model.jnt_dofadr[j]))

    print(f"关节数：{len(joint_names)}")

    joint_pos = []
    joint_vel = []
    base_pos = []
    base_quat = []
    base_lin_vel = []
    base_ang_vel = []

    for _ in range(record_steps):
        walker.step()
        joint_pos.append(walker.data.qpos[qpos_idx].copy())
        joint_vel.append(walker.data.qvel[dof_idx].copy())
        base_pos.append(walker.data.qpos[0:3].copy())
        base_quat.append(walker.data.qpos[3:7].copy())
        base_lin_vel.append(walker.data.qvel[0:3].copy())
        base_ang_vel.append(walker.data.qvel[3:6].copy())

    joint_pos = np.asarray(joint_pos, dtype=np.float32)
    joint_vel = np.asarray(joint_vel, dtype=np.float32)

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
