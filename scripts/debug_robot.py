"""调试 T800 初始姿态 / PD 是否生效。

用法（Ubuntu，headless）：
    python -u scripts/debug_robot.py --headless

它会：
  1. 创建 1 个环境，重置
  2. 打印关节名、默认关节角、PD 增益、足端高度
  3. 用零动作（目标=默认站立角）走 150 步
  4. 每 10 步打印基座高度、足端高度、关节误差、基座接触力

如果零动作时机器人也快速倒下，说明问题在初始姿态/PD/碰撞，
不是 PPO 的锅。
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Debug T800 reset pose")
parser.add_argument("--task", type=str, default="Isaac-T800-Locomotion-v0")
parser.add_argument("--steps", type=int, default=150)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg

import t800_lab  # noqa: F401


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    obs, _ = env.reset()

    robot = env.unwrapped.scene["robot"]
    sensor = env.unwrapped.scene["contact_forces"]

    print("=" * 70)
    print("joint names:", robot.joint_names)
    print("body names:", robot.body_names)
    print("default_joint_pos:", robot.data.default_joint_pos[0].cpu().numpy())
    print("joint_pos after reset:", robot.data.joint_pos[0].cpu().numpy())
    if hasattr(robot.data, "joint_stiffness"):
        print("joint_stiffness:", robot.data.joint_stiffness[0].cpu().numpy())
    if hasattr(robot.data, "joint_damping"):
        print("joint_damping:", robot.data.joint_damping[0].cpu().numpy())
    if hasattr(robot.data, "joint_effort_limits"):
        print("joint_effort_limits:", robot.data.joint_effort_limits[0].cpu().numpy())
    if hasattr(robot.data, "joint_pos_target"):
        print("joint_pos_target after reset:", robot.data.joint_pos_target[0].cpu().numpy())
    if hasattr(robot.data, "applied_torque"):
        print("applied_torque after reset:", robot.data.applied_torque[0].cpu().numpy())

    foot_ids = [robot.body_names.index("LINK_FOOT_L"), robot.body_names.index("LINK_FOOT_R")]
    # air_time 要按 contact sensor 的 body 顺序取，不能混用 robot.body_names
    sensor_foot_ids = [
        sensor.body_names.index("LINK_FOOT_L"),
        sensor.body_names.index("LINK_FOOT_R"),
    ]
    print("sensor foot ids:", sensor_foot_ids)
    base_id = robot.body_names.index("LINK_BASE")
    print("foot body ids:", foot_ids)
    print("sensor body names:", sensor.body_names)

    # 零动作 = 目标关节角等于默认站立角。
    # 不要用 env.action_space.shape[0]：这个版本它是 (1, 25)，
    # 取 [0] 会得到 1。用 action_manager.total_action_dim 最稳。
    action_dim = env.unwrapped.action_manager.total_action_dim
    print("action_dim:", action_dim)
    actions = torch.zeros((1, action_dim), device=env.unwrapped.device)

    for i in range(args_cli.steps):
        obs, rew, terminated, truncated, info = env.step(actions)

        if i in (0, 20, 40, 60):
            forces = sensor.data.net_forces_w
            print("net_forces_w shape:", tuple(forces.shape))
            if forces.dim() == 4:
                force_now = forces[0, -1]
            else:
                force_now = forces[0]
            norms = force_now.norm(dim=-1)
            top = norms.topk(min(5, norms.numel()))
            # 必须用 sensor.body_names 的顺序，不能拿 robot.body_names 索引
            print(
                f"[contact step {i}] "
                + ", ".join(
                    f"{sensor.body_names[j]}={norms[j].item():.1f}N"
                    for j in top.indices.tolist()
                )
            )

        if i in (0, 10, 20, 30):
            if hasattr(robot.data, "joint_pos_target"):
                target = robot.data.joint_pos_target[0].cpu().numpy()
                print(f"[target step {i}]", target.round(3))
            err = (robot.data.joint_pos[0] - robot.data.default_joint_pos[0]).abs()
            top = err.topk(5)
            print(
                f"[top joint err step {i}] "
                + ", ".join(
                    f"{robot.joint_names[j]}={err[j].item():.3f}"
                    for j in top.indices.tolist()
                )
            )

        if i % 10 == 0 or bool(terminated.any()) or bool(truncated.any()):
            root_z = robot.data.root_pos_w[0, 2].item()
            max_torque = (
                robot.data.applied_torque[0].abs().max().item()
                if hasattr(robot.data, "applied_torque")
                else float("nan")
            )
            feet_z = robot.data.body_pos_w[0, foot_ids, 2].cpu().numpy()
            joint_err = (robot.data.joint_pos[0] - robot.data.default_joint_pos[0]).abs()
            max_err = joint_err.max().item()
            base_force = sensor.data.net_forces_w[0, base_id].norm().item()
            air = sensor.data.current_air_time[0, sensor_foot_ids].cpu().numpy()
            print(
                f"step {i:3d}  root_z={root_z:.3f}  feet_z={feet_z.round(3)}  "
                f"max_joint_err={max_err:.3f}  max_torque={max_torque:.1f}  base_force={base_force:.1f}  "
                f"foot_air={air.round(3)}  rew={rew.item():.3f}  "
                f"term={bool(terminated.any())}"
            )

        if bool(terminated.any()) or bool(truncated.any()):
            print(f"[结束] 在第 {i} 步终止/超时")
            break

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        simulation_app.close()
        raise
