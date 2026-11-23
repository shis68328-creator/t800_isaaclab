"""T800 Isaac Lab 回放脚本。

运行：
    python scripts/play.py --task Isaac-T800-Locomotion-v0 \
        --num_envs 32 --checkpoint logs/rsl_rl/t800_locomotion/xxx/model_1000.pt

不传 --checkpoint 时自动找最新的 .pt。
"""

import argparse
import glob
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play T800 policy")
parser.add_argument("--task", type=str, default="Isaac-T800-Locomotion-v0")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--checkpoint", type=str, default="")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 非 headless 时打开相机，才能看到画面
if not args_cli.headless:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import t800_lab  # noqa: F401


def find_checkpoint():
    if args_cli.checkpoint:
        return args_cli.checkpoint
    pattern = os.path.join("logs", "rsl_rl", "t800_locomotion", "**", "*.pt")
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError("没有找到 .pt，请先训练或传 --checkpoint")
    return max(files, key=os.path.getmtime)


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    env_cfg.seed = args_cli.seed

    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args_cli.device

    checkpoint = find_checkpoint()
    print(f"[INFO] checkpoint: {checkpoint}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs, _ = env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        simulation_app.close()
        raise
