# ==============================================================================
# 本文件用到的库（小例子见 docs/LIBRARIES.md）
#   argparse: 读 --task / --num_envs / --resume
#       parser.add_argument("--num_envs", type=int, default=4096)
#   AppLauncher: 启动 Isaac Sim
#       app = AppLauncher(args).app
#   gymnasium: 创建环境
#       env = gym.make("Isaac-T800-Mimic-v0", cfg=env_cfg)
#   rsl_rl.runners.OnPolicyRunner: PPO 训练器
#       runner = OnPolicyRunner(env, cfg, log_dir, device)
#   pathlib.Path: 拼日志/模型路径
# ==============================================================================

"""T800 Isaac Lab 训练入口。

======================================================================
这个脚本的执行顺序（很重要）
======================================================================
1. argparse 解析命令行
2. AppLauncher 启动 Isaac Sim（这一步必须最先做）
3. 启动之后再 import isaaclab / isaaclab_tasks / rsl_rl
4. 读取环境配置 + PPO 配置
5. gym.make 创建环境，套上 RslRlVecEnvWrapper
6. OnPolicyRunner 开始训练
7. 保存日志和模型

为什么 AppLauncher 要最先？
    Isaac Lab 依赖 Omniverse / Isaac Sim 的 C++ 扩展和 Python 绑定，
    这些只有在 AppLauncher 启动 Isaac Sim 之后才可用。
    如果先 import isaaclab，会报 “omni 模块找不到” 之类的错误。

======================================================================
用到的包
======================================================================
argparse：
    标准库，解析 --task、--num_envs 等命令行参数。
AppLauncher：
    Isaac Lab 提供的启动器，负责启动 Isaac Sim 应用。
    它会添加 --headless、--device、--enable_cameras 等参数。
gymnasium（gym）：
    强化学习环境标准接口。Isaac Lab 把环境注册成 Gym 任务，
    所以用 gym.make("Isaac-T800-Locomotion-v0") 创建。
rsl_rl.runners.OnPolicyRunner：
    PPO 训练器，负责采样、更新、保存。
RslRlVecEnvWrapper：
    把 Isaac Lab 的 ManagerBasedRLEnv 包装成 rsl_rl 需要的向量环境。
parse_env_cfg：
    根据任务 id 从 Gym 注册表取出环境配置，并按 num_envs/device 复制。
load_cfg_from_registry：
    根据任务 id 取出 PPO 配置。


看不懂名词时，先看 docs/GLOSSARY.md（高中生版术语表）。
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# 注意：AppLauncher 必须在其它 isaaclab 模块之前导入/使用
from isaaclab.app import AppLauncher

# ----------------------------------------------------------------------
# 1. 命令行参数
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train T800 with Isaac Lab + rsl_rl")
# 任务 id：在 tasks/locomotion/velocity/__init__.py 里注册
parser.add_argument("--task", type=str, default="Isaac-T800-Locomotion-v0")
# 并行环境数量：Isaac Lab 在 GPU 上同时跑多个环境
parser.add_argument("--num_envs", type=int, default=2048)
# 随机种子
parser.add_argument("--seed", type=int, default=0)
# 覆盖 PPO 配置里的 max_iterations
parser.add_argument("--max_iterations", type=int, default=None)
# 从已有 checkpoint 继续训练（rsl_rl 的 .pt）
parser.add_argument("--resume", type=str, default="", help="从 model_xxx.pt 继续训练")
# 添加 Isaac Sim 自己的参数（--headless、--device 等）
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ----------------------------------------------------------------------
# 2. 启动 Isaac Sim
# ----------------------------------------------------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ----------------------------------------------------------------------
# 3. 启动后再导入 Isaac Lab / rsl_rl
# ----------------------------------------------------------------------
import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

# 如果没 pip install -e，也能从 source 目录导入 t800_lab
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/t800_lab"))

# 导入 t800_lab 会执行 tasks/__init__.py，从而向 Gym 注册
# "Isaac-T800-Locomotion-v0"
import t800_lab  # noqa: F401

# PyTorch 在 NVIDIA 显卡上的性能选项
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    print("[DEBUG] enter main", flush=True)
    # --------------------------------------------------------------
    # 4. 读取环境配置
    # --------------------------------------------------------------
    # parse_env_cfg 会：
    #   - 从 Gym 注册表拿到 T800LocomotionEnvCfg
    #   - 设置 num_envs、device
    #   - 给每个环境分配空间
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    env_cfg.seed = args_cli.seed
    print("[DEBUG] after parse_env_cfg", flush=True)

    # --------------------------------------------------------------
    # 5. 读取 PPO 配置
    # --------------------------------------------------------------
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    print("[DEBUG] after load_cfg_from_registry", flush=True)
    agent_cfg.device = args_cli.device
    agent_cfg.seed = args_cli.seed
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    # --------------------------------------------------------------
    # 6. 日志目录
    # --------------------------------------------------------------
    log_root = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root, log_dir)
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)

    print(f"[INFO] 任务: {args_cli.task}")
    print(f"[INFO] 环境数: {args_cli.num_envs}")
    print(f"[INFO] 日志目录: {log_dir}")
    print_dict(env_cfg.to_dict())
    # 把这次训练的环境/PPO 配置存下来，方便复现
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # --------------------------------------------------------------
    # 7. 创建环境
    # --------------------------------------------------------------
    # gym.make 会调用注册时指定的 entry_point（ManagerBasedRLEnv），
    # 并把 env_cfg 传进去。
    print("[DEBUG] before gym.make", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    print("[DEBUG] after gym.make", flush=True)

    # rsl_rl 需要“向量环境”接口：一次 step 返回 num_envs 个结果。
    # clip_actions 会把策略输出裁剪到 [-100, 100]。
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print("[DEBUG] after RslRlVecEnvWrapper", flush=True)

    # --------------------------------------------------------------
    # 8. 训练器
    # --------------------------------------------------------------
    # OnPolicyRunner 内部就是 PPO：
    #   for iteration:
    #       采样 num_steps_per_env 步 × num_envs
    #       计算 GAE / 优势
    #       更新 actor / critic
    #       定期保存
    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=log_dir,
        device=agent_cfg.device,
    )

    # 从 checkpoint 继续：会恢复网络权重、优化器和当前 iteration
    if args_cli.resume:
        print(f"[INFO] 从 checkpoint 继续训练: {args_cli.resume}", flush=True)
        runner.load(args_cli.resume)

    print("[DEBUG] before runner.learn", flush=True)
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=True,  # 每个环境随机初始化 episode 长度
    )

    # --------------------------------------------------------------
    # 9. 收尾
    # --------------------------------------------------------------
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        # 有些库会调 sys.exit(0)，这里把退出码打出来
        print(f"[DEBUG] SystemExit code={exc.code}", flush=True)
        simulation_app.close()
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        simulation_app.close()
        raise
