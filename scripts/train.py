"""T800 Isaac Lab 训练脚本（rsl_rl PPO）。

运行：
    cd t800_isaaclab
    python scripts/train.py --task Isaac-T800-Locomotion-v0 \
        --num_envs 2048 --headless

训练日志和模型保存在 logs/rsl_rl/t800_locomotion/<时间戳>/。
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# 先启动 Isaac Sim，再导入其它 Isaac Lab 模块
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train T800 with Isaac Lab + rsl_rl")
parser.add_argument("--task", type=str, default="Isaac-T800-Locomotion-v0")
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_iterations", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

# 没有 pip install 时，也能直接从 source/ 导入并注册任务
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/t800_lab"))

# 导入包会注册 Isaac-T800-Locomotion-v0
import t800_lab  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    # 环境配置（从 Gym 注册表拿 T800LocomotionEnvCfg）
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    env_cfg.seed = args_cli.seed

    # PPO 配置
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args_cli.device
    agent_cfg.seed = args_cli.seed
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    # 日志目录
    log_root = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root, log_dir)
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)

    print(f"[INFO] 任务: {args_cli.task}")
    print(f"[INFO] 环境数: {args_cli.num_envs}")
    print(f"[INFO] 日志目录: {log_dir}")
    print_dict(env_cfg.to_dict(), "env_cfg")
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # 创建环境
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # 训练器
    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=log_dir,
        device=agent_cfg.device,
    )
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=True,
    )

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        simulation_app.close()
        raise
