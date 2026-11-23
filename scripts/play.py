# ==============================================================================
# 本文件用到的库（小例子见 docs/LIBRARIES.md）
#   argparse: 读 --task / --checkpoint / --num_envs
#   glob: 找最新的 .pt
#       glob.glob("logs/**/*.pt", recursive=True)
#   AppLauncher: 启动 Isaac Sim
#   gymnasium: 创建环境
#   rsl_rl.OnPolicyRunner: 加载 checkpoint 并取推理策略
#       runner.load(checkpoint); policy = runner.get_inference_policy()
# ==============================================================================

"""T800 Isaac Lab 回放脚本。

======================================================================
和 train.py 的区别
======================================================================
- 不训练，只加载 .pt 模型，用 policy(obs) 输出动作
- 环境数量少（默认 16/32），方便看画面
- 需要模型 checkpoint

======================================================================
用到的包
======================================================================
AppLauncher：
    启动 Isaac Sim。必须在 import isaaclab 之前。
gymnasium：
    创建已注册的 Isaac Lab 任务。
RslRlVecEnvWrapper：
    把环境包装成 rsl_rl 的向量环境，和训练时一致。
OnPolicyRunner：
    用来加载 checkpoint，并取出推理策略。
torch.inference_mode()：
    推理时关闭梯度，省显存。


看不懂名词时，先看 docs/GLOSSARY.md（高中生版术语表）。
"""

import argparse
import glob
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# ----------------------------------------------------------------------
# 1. 命令行参数
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Play T800 policy")
parser.add_argument("--task", type=str, default="Isaac-T800-Locomotion-v0")
parser.add_argument("--num_envs", type=int, default=32)
# 不传就从 logs/rsl_rl/t800_locomotion 下找最新的 .pt
parser.add_argument("--checkpoint", type=str, default="")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 非 headless 时打开相机，Isaac Sim 才能渲染画面
if not args_cli.headless:
    args_cli.enable_cameras = True

# ----------------------------------------------------------------------
# 2. 启动 Isaac Sim
# ----------------------------------------------------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ----------------------------------------------------------------------
# 3. 启动后再导入
# ----------------------------------------------------------------------
import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/t800_lab"))

import t800_lab  # noqa: F401  注册任务


def find_checkpoint():
    """找 checkpoint：优先用 --checkpoint，否则取最新的 .pt。"""
    if args_cli.checkpoint:
        return args_cli.checkpoint
    pattern = os.path.join("logs", "rsl_rl", "t800_locomotion", "**", "*.pt")
    files = glob.glob(pattern, recursive=True)
    if not files:
        raise FileNotFoundError("没有找到 .pt，请先训练或传 --checkpoint")
    return max(files, key=os.path.getmtime)


def main():
    # --------------------------------------------------------------
    # 4. 环境配置
    # --------------------------------------------------------------
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    env_cfg.seed = args_cli.seed

    # --------------------------------------------------------------
    # 5. PPO 配置（只用来构建网络结构，和训练时一致）
    # --------------------------------------------------------------
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args_cli.device

    checkpoint = find_checkpoint()
    print(f"[INFO] checkpoint: {checkpoint}")

    # --------------------------------------------------------------
    # 6. 创建环境
    # --------------------------------------------------------------
    # render_mode="rgb_array"：让 Isaac Lab 创建可渲染的视口
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # --------------------------------------------------------------
    # 7. 加载模型
    # --------------------------------------------------------------
    # OnPolicyRunner 需要一个环境来构建网络结构；
    # 然后 load(checkpoint) 把训练好的权重读进来。
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)

    # get_inference_policy() 返回一个函数：输入 obs，输出动作
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # --------------------------------------------------------------
    # 8. 推理循环
    # --------------------------------------------------------------
    obs, _ = env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
        # 环境 step：动作送进仿真，返回新的观测
        obs, _, _, _ = env.step(actions)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        simulation_app.close()
        raise
