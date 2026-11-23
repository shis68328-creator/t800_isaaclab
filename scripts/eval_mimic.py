"""Headless 评估 Mimic 策略：打印实际位移、速度、高度。

==============================================================================
为什么要单独评估
==============================================================================
训练日志里的 reward 只能说明“策略觉得自己做得好”，
但不一定真的在往前走。比如它可能站着不动、原地跟踪关节角。

这个脚本让训练好的策略在 1 个环境里跑 500 步（10 秒），
直接打印：
    - 基座实际走了多远
    - 平均前进速度
    - 平均高度
这样就能判断它是真的在走，还是原地摆动作。

==============================================================================
关键名词
==============================================================================
policy（策略）：
    神经网络，输入观测，输出动作。

checkpoint：
    训练保存的模型文件（.pt）。

RslRlVecEnvWrapper：
    把 Isaac Lab 环境包装成 rsl_rl 能用的“向量环境”。
    即使只有 1 个环境，也需要这层包装。

inference_mode：
    推理模式，不计算梯度，省显存、速度快。
"""

import argparse

from isaaclab.app import AppLauncher

# 注意：必须先启动 Isaac Sim，再 import isaaclab / rsl_rl。
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-T800-Mimic-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--steps", type=int, default=500)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import t800_lab  # noqa: F401  导入后注册任务


def main():
    # ---- 1. 创建 1 个环境 ----
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    # ---- 2. 读取 PPO 配置 ----
    # 这里只用它来构建网络结构，不训练。
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args_cli.device

    # ---- 3. 加载模型 ----
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # ---- 4. 跑 500 步 ----
    obs, _ = env.reset()
    robot = env.unwrapped.scene["robot"]
    start = robot.data.root_pos_w[0, :2].clone()
    speeds = []
    heights = []

    for i in range(args_cli.steps):
        with torch.inference_mode():
            actions = policy(obs)

        # 环境推进一步：动作 -> PD -> 物理仿真
        obs, rew, dones, infos = env.step(actions)

        # 基座 x 方向速度（机器人自己的坐标系）
        vx = robot.data.root_lin_vel_b[0, 0].item()
        z = robot.data.root_pos_w[0, 2].item()
        speeds.append(vx)
        heights.append(z)

        if i % 50 == 0:
            pos = robot.data.root_pos_w[0].cpu().numpy()
            print(
                f"step {i:3d}  x={pos[0]:+.3f}  y={pos[1]:+.3f}  z={z:.3f}  "
                f"vx={vx:+.3f}  rew={rew.item():.3f}",
                flush=True,
            )

    # ---- 5. 汇总 ----
    end = robot.data.root_pos_w[0, :2]
    distance = torch.linalg.norm(end - start).item()
    print("=" * 60)
    print(f"总位移: {distance:.3f} m")
    print(f"平均 vx: {sum(speeds)/len(speeds):+.3f} m/s")
    print(f"平均高度: {sum(heights)/len(heights):.3f} m")
    print(f"最终位置: {end.cpu().numpy().round(3)}")

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
