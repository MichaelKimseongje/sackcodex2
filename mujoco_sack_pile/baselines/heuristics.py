from __future__ import annotations

import numpy as np

from ..environment import SackPileEnv


def _gripper_down_quat() -> np.ndarray:
    return SackPileEnv.euler_to_quat(np.array([0.0, -np.pi / 2.0, 0.0], dtype=np.float64))


def _scoop_forward_quat() -> np.ndarray:
    return SackPileEnv.euler_to_quat(np.array([0.0, 0.0, 0.0], dtype=np.float64))


def top_grasp_flat_scoop_drag(env: SackPileEnv, viewer=None):
    """top grasp + flat scoop insertion + drag baseline."""

    target_top = env.target_site("top_site")
    target_center, _ = env.target_state()
    # 상단 노출부를 먼저 집어 올려 얕은 gap을 만든 뒤 scoop를 밀어 넣는다.
    env.set_gripper_width(env.open_width)
    env.move_mocap_linear("gripper", target_top + np.array([0.0, -0.030, 0.110]), _gripper_down_quat(), 150, viewer=viewer)
    env.move_mocap_linear("gripper", target_top + np.array([0.0, -0.010, 0.030]), _gripper_down_quat(), 130, viewer=viewer)
    env.set_gripper_width(env.closed_width)
    env.step(140, viewer=viewer)

    scoop_entry = target_center + np.array([-0.180, 0.0, -0.010])
    scoop_mid = target_center + np.array([-0.040, 0.0, -0.005])
    scoop_drag = target_center + np.array([0.025, 0.0, 0.010])
    env.move_mocap_linear("scoop", scoop_entry, _scoop_forward_quat(), 160, viewer=viewer)
    env.move_mocap_linear("scoop", scoop_mid, _scoop_forward_quat(), 130, viewer=viewer)
    env.move_mocap_linear("scoop", scoop_drag, _scoop_forward_quat(), 140, viewer=viewer)

    env.move_mocap_linear("gripper", target_top + np.array([0.030, 0.0, 0.120]), _gripper_down_quat(), 120, viewer=viewer)
    env.mark_pre_lift_state()
    env.move_mocap_linear("scoop", scoop_drag + np.array([0.0, 0.0, 0.055]), _scoop_forward_quat(), 160, viewer=viewer)
    env.step(180, viewer=viewer)


def fixed_pose_dual_support(env: SackPileEnv, viewer=None):
    """fixed 2F+scoop pose baseline."""

    target_center, _ = env.target_state()
    grip_pose = target_center + np.array([0.015, -0.015, 0.105])
    scoop_pose = target_center + np.array([-0.015, 0.0, 0.000])

    # 별도 탐색 없이 고정된 상대 배치로 바로 지지 상태를 만든다.
    env.set_gripper_width(0.022)
    env.move_mocap_linear("gripper", grip_pose, _gripper_down_quat(), 220, viewer=viewer)
    env.move_mocap_linear("scoop", scoop_pose, _scoop_forward_quat(), 220, viewer=viewer)
    env.set_gripper_width(env.closed_width)
    env.step(180, viewer=viewer)
    env.mark_pre_lift_state()
    env.move_mocap_linear("gripper", grip_pose + np.array([0.020, 0.0, 0.040]), _gripper_down_quat(), 120, viewer=viewer)
    env.move_mocap_linear("scoop", scoop_pose + np.array([0.030, 0.0, 0.040]), _scoop_forward_quat(), 120, viewer=viewer)
    env.step(180, viewer=viewer)


def scoop_first_gap_creation_regrasp(env: SackPileEnv, viewer=None):
    """scoop-first gap creation + regrasp baseline."""

    target_center, _ = env.target_state()
    target_side = env.target_site("side_site")

    # 먼저 scoop로 하부 공간을 만들고, 이후 side grip으로 재파지한다.
    env.move_mocap_linear("scoop", target_center + np.array([-0.190, 0.0, -0.008]), _scoop_forward_quat(), 150, viewer=viewer)
    env.move_mocap_linear("scoop", target_center + np.array([-0.055, 0.0, -0.002]), _scoop_forward_quat(), 100, viewer=viewer)
    env.move_mocap_linear("scoop", target_center + np.array([-0.010, 0.0, 0.000]), _scoop_forward_quat(), 120, viewer=viewer)

    env.set_gripper_width(env.open_width)
    env.move_mocap_linear("gripper", target_side + np.array([0.000, -0.040, 0.065]), _gripper_down_quat(), 160, viewer=viewer)
    env.move_mocap_linear("gripper", target_side + np.array([0.000, -0.012, 0.020]), _gripper_down_quat(), 110, viewer=viewer)
    env.set_gripper_width(env.closed_width)
    env.step(150, viewer=viewer)

    env.mark_pre_lift_state()
    env.move_mocap_linear("gripper", target_side + np.array([0.020, 0.0, 0.095]), _gripper_down_quat(), 120, viewer=viewer)
    env.move_mocap_linear("scoop", target_center + np.array([0.030, 0.0, 0.015]), _scoop_forward_quat(), 120, viewer=viewer)
    env.move_mocap_linear("scoop", target_center + np.array([0.040, 0.0, 0.055]), _scoop_forward_quat(), 140, viewer=viewer)
    env.step(180, viewer=viewer)


BASELINES = {
    "top_grasp_flat_scoop_drag": top_grasp_flat_scoop_drag,
    "fixed_2f_scoop_pose": fixed_pose_dual_support,
    "scoop_first_gap_creation_regrasp": scoop_first_gap_creation_regrasp,
}
