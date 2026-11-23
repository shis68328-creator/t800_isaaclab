"""T800 的 PPO / rsl_rl 配置。

======================================================================
用到的包
======================================================================
rsl_rl：
    Robotic Systems Lab 出的强化学习库（ETH Zurich），Isaac Lab 官方推荐
    的 PPO 实现之一。它负责：
      - 收集环境数据
      - 用 PPO 更新 actor / critic 网络
      - 保存 / 加载模型
isaaclab_rl.rsl_rl：
    Isaac Lab 对 rsl_rl 的适配层，包含配置类和向量环境包装器。
RslRlOnPolicyRunnerCfg：
    训练器总配置（跑多少轮、多久存一次、实验名）。
RslRlPpoActorCriticCfg：
    网络结构：actor（策略）和 critic（价值）各几层、多宽、什么激活。
RslRlPpoAlgorithmCfg：
    PPO 算法超参数：学习率、clip、熵、epoch 数等。

======================================================================
PPO 是什么（一句话）
======================================================================
PPO 让策略网络在“旧策略附近”小步更新：
    新策略和旧策略的概率比不能偏离太远（clip），
    同时用 critic 估计状态价值，降低梯度方差。
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class T800PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # 每个环境每次 rollout 收集多少步。
    # 总样本 = num_steps_per_env × num_envs。
    num_steps_per_env = 24

    # 总共训练多少轮（iteration）。一轮 ≈ 收集一次数据 + 更新一次网络。
    max_iterations = 3000

    # 每多少轮保存一次模型
    save_interval = 50

    # 日志 / 模型保存的子目录名
    experiment_name = "t800_g1style"

    # 是否对观测做 running mean/std 归一化。
    # 这里交给 Isaac Lab / rsl_rl 的默认处理，关闭额外的经验归一化。
    empirical_normalization = False

    # 动作裁剪范围（策略输出超过就裁掉）
    clip_actions = 100.0

    # ==============================================================
    # 网络结构
    # ==============================================================
    policy = RslRlPpoActorCriticCfg(
        # 初始探索噪声标准差。1.0 表示动作一开始比较随机，鼓励探索。
        init_noise_std=1.0,
        # 噪声用 log 形式存储（更稳定）
        noise_std_type="log",
        # actor：输入观测，输出动作均值；三层 MLP
        actor_hidden_dims=[512, 256, 128],
        # critic：输入观测，输出状态价值；三层 MLP
        critic_hidden_dims=[512, 256, 128],
        # 激活函数，elu 在机器人控制里比较稳
        activation="elu",
    )

    # ==============================================================
    # PPO 超参数
    # ==============================================================
    algorithm = RslRlPpoAlgorithmCfg(
        # 价值函数损失权重
        value_loss_coef=1.0,
        # 使用 clipped value loss，限制 critic 每次更新幅度
        use_clipped_value_loss=True,
        # PPO 的 clip 范围：新策略概率 / 旧策略概率限制在 [0.8, 1.2]
        clip_param=0.2,
        # 熵奖励系数：越大越鼓励探索
        entropy_coef=0.008,
        # 同一批数据重复训练几轮
        num_learning_epochs=5,
        # 每轮把数据分成几个 mini-batch
        num_mini_batches=4,
        # 学习率
        learning_rate=1.0e-3,
        # 学习率调度：adaptive 表示根据 KL 自动调
        schedule="adaptive",
        # 折扣因子：0.99 表示关注未来约 100 步
        gamma=0.99,
        # GAE 的 lambda，平衡偏差和方差
        lam=0.95,
        # adaptive 调度的目标 KL
        desired_kl=0.01,
        # 梯度裁剪
        max_grad_norm=1.0,
    )
