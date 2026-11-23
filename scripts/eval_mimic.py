"""Headless 评估 Mimic 策略：打印实际位移、速度、高度。

用法：
    python -u scripts/eval_mimic.py \
        --task Isaac-T800-Mimic-v0 \
        --checkpoint logs/rsl_rl/t800_mimic/<时间戳>/model_2999.pt \
        --steps 500 --headless
"""

import argparse

from isaaclab.app import AppLauncher

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

import t800_lab  # noqa: F401


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args_cli.device

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs, _ = env.reset()
    robot = env.unwrapped.scene["robot"]
    start = robot.data.root_pos_w[0, :2].clone()
    speeds = []
    heights = []

    for i in range(args_cli.steps):
        with torch.inference_mode():
            actions = policy(obs)
        obs, rew, dones, infos = env.step(actions)

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
