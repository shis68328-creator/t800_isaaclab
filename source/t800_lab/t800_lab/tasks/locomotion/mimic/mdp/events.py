# ==============================================================================
# 本文件用到的库（小例子见 docs/LIBRARIES.md）
#   torch: zeros_like
#   SceneEntityCfg: 引用机器人
#   robot.write_joint_state_to_sim: 直接写关节状态
# ==============================================================================

"""Mimic 任务的 reset 事件。

==============================================================================
reset 是什么
==============================================================================
每个回合开始时，仿真要把机器人“摆回初始状态”，这个过程叫 reset。

普通任务 reset 到默认站姿就够了，但 Mimic 任务必须 reset 到
参考轨迹的第 0 帧，否则：
    当前关节角 = 默认站姿
    目标关节角 = 参考步态第一帧
两者差很多，PD 会猛地一拽，机器人一开始就摔。
"""

import torch
from isaaclab.managers import SceneEntityCfg


def reset_joints_to_reference(env, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """把指定环境的关节直接写成参考轨迹第 0 帧。

    参数：
        env：环境
        env_ids：这次要 reset 哪些环境（并行环境可能只 reset 一部分）
        asset_cfg：机器人配置，默认名字叫 "robot"

    返回：
        无。会直接修改仿真状态。
    """
    # 回合计数归零，保证参考索引从第 0 帧开始
    env.episode_length_buf[env_ids] = 0

    # 从动作项里拿参考第 0 帧
    term = env.action_manager.get_term("joint_pos")
    ref = term.get_reference()[env_ids]

    # 速度设成 0，避免 reset 时机器人带着速度冲出去
    zeros = torch.zeros_like(ref)

    robot = env.scene[asset_cfg.name]
    robot.write_joint_state_to_sim(
        ref, zeros, joint_ids=term.joint_ids, env_ids=env_ids
    )
