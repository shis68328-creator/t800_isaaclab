"""Mimic 任务的 reset 事件。"""

import torch
from isaaclab.managers import SceneEntityCfg


def reset_joints_to_reference(env, env_ids, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """把机器人关节直接写到参考轨迹第 0 帧。"""
    # 确保参考相位从 0 开始
    env.episode_length_buf[env_ids] = 0
    term = env.action_manager.get_term("joint_pos")
    ref = term.get_reference()[env_ids]
    zeros = torch.zeros_like(ref)
    robot = env.scene[asset_cfg.name]
    robot.write_joint_state_to_sim(ref, zeros, joint_ids=term.joint_ids, env_ids=env_ids)
