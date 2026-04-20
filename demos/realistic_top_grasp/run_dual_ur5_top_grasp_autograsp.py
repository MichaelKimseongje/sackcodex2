from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
LOW_FILL_DIR = ROOT_DIR.parent / "01_low_fill"
if str(LOW_FILL_DIR) not in sys.path:
    sys.path.insert(0, str(LOW_FILL_DIR))

from dual_ur5_low_fill_env import collect_shell_body_ids  # noqa: E402
from dual_ur5_top_grasp_env import DualUR5TopGraspEnv  # noqa: E402
from generate_sack_mesh import RING_COUNT, available_content_cases, available_scenarios  # noqa: E402


STAGE_SETTLE_SECONDS = 1.2
STAGE_MOVE_SECONDS = 1.2
STAGE_CLOSE_SECONDS = 0.7
STAGE_LIFT_SECONDS = 1.4
STAGE_HOLD_SECONDS = 5.0
STAGE_INITIAL_HOLD_SECONDS = 0.8
OPEN_GRIPPER_GAP_M = 0.120
GRASP_POINT_OFFSET_Z = 0.006
GRASP_MIN_Z = 0.030
PREGRASP_OFFSET_Z = 0.120
APPROACH_OFFSET_Z = 0.030
LIFT_OFFSET_Z = 0.135
HOLD_SUCCESS_MIN_LIFT_M = 0.035
HOLD_SUCCESS_MAX_SLIP_M = 0.075
HOLD_SUCCESS_MAX_FINAL_DIST_M = 0.090
ASSIST_ACTIVE_MAX_GRIPPER_GAP_M = 0.018
PRECAPTURE_GRIPPER_GAP_M = 0.035
ASSIST_PATCH_RADIUS_M = 0.130
ASSIST_PATCH_MAX_POINTS = 3
ASSIST_KP = 130.0
ASSIST_KD = 4.5
ASSIST_MAX_FORCE_PER_POINT_N = 8.0
VISUAL_PINCH_KP = 75.0
VISUAL_PINCH_KD = 4.0
VISUAL_PINCH_MAX_FORCE_PER_POINT_N = 3.0
FOLLOW_MOVE_DELTA = np.array([0.045, 0.0, 0.085], dtype=np.float64)
JAW_CAPTURE_HALF_X = 0.052
JAW_CAPTURE_HALF_Z = 0.115
JAW_CAPTURE_Y_MARGIN = 0.030
SURFACE_CAPTURE_AREA_PER_POINT_M2 = 0.00065
SURFACE_CAPTURE_MIN_AREA_M2 = 0.00120
SURFACE_CAPTURE_MIN_SPAN_Y_M = 0.004
SURFACE_CAPTURE_FRAME_FOLLOW_GAIN_XY = 0.34
SURFACE_CAPTURE_FRAME_FOLLOW_GAIN_Z_UP = 0.0
SURFACE_CAPTURE_FRAME_FOLLOW_GAIN_Z_DOWN = 0.18
SURFACE_CAPTURE_MAX_FRAME_STEP_XY_M = 0.0030
SURFACE_CAPTURE_MAX_FRAME_STEP_Z_UP_M = 0.0
SURFACE_CAPTURE_MAX_FRAME_STEP_Z_DOWN_M = 0.0015
SURFACE_CAPTURE_PIN_BLEND = 0.28
SURFACE_CAPTURE_MAX_PATCH_STRETCH_M = 0.16
SURFACE_CAPTURE_MAX_TARGET_STEP_M = 0.0040
SURFACE_CAPTURE_RELEASE_EE_JUMP_M = 0.065
SURFACE_CAPTURE_MAX_TOP_OFFSET_M = 0.145
SURFACE_CAPTURE_MAX_MID_OFFSET_M = 0.115
SURFACE_CAPTURE_MAX_LOWER_OFFSET_M = 0.085
SURFACE_CAPTURE_EXTRA_SHELL_SAG_FORCE_N = 0.025
SURFACE_CAPTURE_EXTRA_CONTENT_SAG_FORCE_N = 0.18
GUARD_MIN_CAPTURED_POINTS = 1
GUARD_MIN_THICKNESS_M = 0.006
GUARD_MAX_TANGENTIAL_SLIP_MPS = 0.45
GUARD_MAX_COUPLED_MOTION = 1.50
RELEASE_MAX_SAG_M = 0.090
RELEASE_MAX_SLIP_MPS = 0.75
RELEASE_MAX_COUPLED_MOTION = 2.00


@dataclass
class HoldPatch:
    body_ids: list[int]
    relative_positions: np.ndarray
    captured_body_ids: list[int]
    capture_quality: dict[str, object]
    jaw_local_positions: np.ndarray | None = None
    target_world_positions: np.ndarray | None = None


def label_from_candidate_name(name: str, fallback: str) -> str:
    for label in ("plain_top", "seam", "fold"):
        if name.endswith(f"_{label}"):
            return label
    return fallback


def label_for_shell_body_id(env: DualUR5TopGraspEnv, body_id: int, fallback: str) -> str:
    body_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, int(body_id)) or ""
    if body_name.startswith("bag_shell_"):
        try:
            point_index = int(body_name.split("_")[2])
        except (IndexError, ValueError):
            return fallback
        if 0 <= point_index < len(env._scenario_labels):
            return env._scenario_labels[point_index]
    return fallback


def top_candidate_positions(env: DualUR5TopGraspEnv) -> list[tuple[str, int, np.ndarray]]:
    shell_ids = collect_shell_body_ids(env.model)
    candidates: list[tuple[str, int, np.ndarray]] = []
    for point_index in range(0, min(1 + RING_COUNT, len(shell_ids))):
        label = env._scenario_labels[point_index] if point_index < len(env._scenario_labels) else "other"
        if label not in {"seam", "fold", "plain_top"}:
            continue
        body_id = env.shell_point_body_id(point_index)
        candidates.append((label, body_id, env.data.xpos[body_id].copy()))
    return candidates


def choose_grasp_target(env: DualUR5TopGraspEnv, preferred_label: str | None = None) -> tuple[str, int, np.ndarray]:
    if hasattr(env, "target_for_label"):
        requested_label = preferred_label or "auto"
        name, body_id, pos = env.target_for_label(requested_label)
        label = label_from_candidate_name(name, requested_label)
        return label, body_id, pos.copy()

    candidates = top_candidate_positions(env)
    if not candidates:
        name, pos = env.nearest_grasp_target(env.end_effector_pos("left"))
        shell_ids = collect_shell_body_ids(env.model)
        return name, shell_ids[0], pos

    if preferred_label is None:
        preferred_label = "seam" if env.scenario_name == "exposed_seam" else "fold"

    preferred = [item for item in candidates if item[0] == preferred_label]
    pool = preferred if preferred else candidates
    # 너무 낮은 접힌 점보다, 실제 gripper가 접근 가능한 상단 쪽 후보를 우선한다.
    label, body_id, pos = max(pool, key=lambda item: (item[2][2], -abs(item[2][1])))
    return label, body_id, pos.copy()


def current_grasp_center(env: DualUR5TopGraspEnv, target_body_id: int) -> np.ndarray:
    target = env.data.xpos[target_body_id].copy()
    center = target + np.array([0.0, 0.0, GRASP_POINT_OFFSET_Z], dtype=np.float64)
    # 접힌 top patch가 바닥 근처까지 내려가도 fingertip이 바닥을 파고들지 않게 한다.
    center[2] = max(float(center[2]), GRASP_MIN_Z)
    return center


def set_target_and_step(
    env: DualUR5TopGraspEnv,
    target_xyz: np.ndarray,
    *,
    seconds: float,
    viewer=None,
    speed: float = 1.0,
    hold_patch: HoldPatch | None = None,
) -> dict[str, float | bool | int]:
    result = env.solve_ee_position_ik(
        "left",
        target_xyz,
        iterations=120,
        tolerance=0.004,
        damping=0.08,
        max_step_deg=5.0,
    )
    step_for_seconds(env, seconds, viewer=viewer, speed=speed, hold_patch=hold_patch)
    return result


def snap_arm_to_current_targets(env: DualUR5TopGraspEnv, arm: str = "left") -> None:
    # 파지 데모는 home에서 자루를 가로지르는 경로가 아니라 pregrasp 이후를 보여준다.
    for joint_name, actuator_name in zip(env.arm_joint_names(arm), env.arm_actuator_names(arm)):
        env.data.qpos[env._joint_qpos_address(joint_name)] = env.actuator_ctrl(actuator_name)
    mujoco.mj_forward(env.model, env.data)


def set_gripper_gap_gradual(
    env: DualUR5TopGraspEnv,
    start_gap: float,
    end_gap: float,
    *,
    seconds: float,
    viewer=None,
    speed: float = 1.0,
) -> None:
    steps = max(1, int(seconds / env.model.opt.timestep))
    sleep_dt = env.model.opt.timestep / max(speed, 1e-6)
    for step_index in range(steps):
        alpha = (step_index + 1) / steps
        gap = (1.0 - alpha) * start_gap + alpha * end_gap
        env.set_left_gripper_gap(gap, immediate=False)
        step_start = time.perf_counter()
        env.step()
        if viewer is not None:
            viewer.sync()
        remaining = sleep_dt - (time.perf_counter() - step_start)
        if viewer is not None and remaining > 0:
            time.sleep(remaining)


def _geom_name(env: DualUR5TopGraspEnv, geom_id: int) -> str:
    if geom_id < 0:
        return "flex"
    return mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)) or f"geom_{geom_id}"


def _is_target_sack_contact_name(name: str) -> bool:
    if name == "flex":
        return True
    if name.startswith("bag_shell"):
        return True
    return False


def finger_sack_contact_flags(env: DualUR5TopGraspEnv) -> tuple[bool, bool]:
    left_contact = False
    right_contact = False
    for contact_index in range(env.data.ncon):
        contact = env.data.contact[contact_index]
        geom_names = (_geom_name(env, contact.geom1), _geom_name(env, contact.geom2))
        for finger_name, other_name in ((geom_names[0], geom_names[1]), (geom_names[1], geom_names[0])):
            if finger_name == "left_finger_l_pad" and _is_target_sack_contact_name(other_name):
                left_contact = True
            if finger_name == "left_finger_r_pad" and _is_target_sack_contact_name(other_name):
                right_contact = True
    return left_contact, right_contact


def _patch_motion_stats(
    env: DualUR5TopGraspEnv,
    selected_body_ids: list[int],
) -> tuple[float, float, float]:
    selected = set(selected_body_ids)
    captured_velocities = [
        float(np.linalg.norm(env.data.cvel[body_id, 3:6]))
        for body_id in selected_body_ids
    ]
    captured_speed = float(np.mean(captured_velocities)) if captured_velocities else 0.0

    neighbor_velocities = []
    for _label, body_id, _pos in top_candidate_positions(env):
        if body_id in selected:
            continue
        neighbor_velocities.append(float(np.linalg.norm(env.data.cvel[body_id, 3:6])))
    neighbor_speed = float(np.mean(neighbor_velocities)) if neighbor_velocities else 0.0
    coupled_motion_score = neighbor_speed / max(captured_speed, 0.05)
    return captured_speed, neighbor_speed, float(coupled_motion_score)


def make_hold_patch(
    env: DualUR5TopGraspEnv,
    center_xyz: np.ndarray,
    *,
    required_body_id: int | None = None,
    allow_visual_pinch_latch: bool = False,
) -> HoldPatch:
    candidates = top_candidate_positions(env)
    if not candidates:
        return HoldPatch([], np.zeros((0, 3), dtype=np.float64), [], {"captured_shell_points": 0})
    candidate_count = len(candidates)
    site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "left_gripper_pinch")
    if site_id < 0:
        return HoldPatch([], np.zeros((0, 3), dtype=np.float64), [], {"captured_shell_points": 0})

    jaw_center = env.data.site_xpos[site_id].copy()
    jaw_frame = env.data.site_xmat[site_id].reshape(3, 3).copy()
    gap_half = max(0.5 * env.left_gripper_gap(), 0.0)

    jaw_captured: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    for label, body_id, pos in candidates:
        local = jaw_frame.T @ (pos - jaw_center)
        inside_pad_face = (
            abs(float(local[0])) <= JAW_CAPTURE_HALF_X
            and abs(float(local[2])) <= JAW_CAPTURE_HALF_Z
            and abs(float(local[1])) <= gap_half + JAW_CAPTURE_Y_MARGIN
        )
        close_to_pinch = float(np.linalg.norm(pos - center_xyz)) <= ASSIST_PATCH_RADIUS_M
        if inside_pad_face and close_to_pinch:
            jaw_captured.append((label, body_id, pos, local))
    if not jaw_captured:
        return HoldPatch(
            [],
            np.zeros((0, 3), dtype=np.float64),
            [],
            {
                "captured_shell_points": 0,
                "captured_required_target": False,
                "candidate_shell_points": candidate_count,
                "jaw_gap_m": float(env.left_gripper_gap()),
                "left_finger_contact": False,
                "right_finger_contact": False,
                "thickness_proxy_m": 0.0,
                "tangential_slip_proxy_mps": 0.0,
                "neighbor_speed_mps": 0.0,
                "coupled_motion_score": 0.0,
                "guarded_latch_approved": False,
                "latch_method": "none",
            },
        )
    candidates = jaw_captured
    # 닫힌 gripper 근처의 아주 작은 local patch만 assist 대상으로 삼는다.
    # 범위가 넓거나 힘이 크면 flex shell이 늘어나 발산처럼 보일 수 있다.
    ranked = sorted(candidates, key=lambda item: float(np.linalg.norm(item[2] - center_xyz)))
    selected = [item for item in ranked if np.linalg.norm(item[2] - center_xyz) <= ASSIST_PATCH_RADIUS_M][
        :ASSIST_PATCH_MAX_POINTS
    ]
    if len(selected) < 2:
        selected = ranked[: min(2, len(ranked))]
    if required_body_id is not None and all(int(body_id) != int(required_body_id) for _label, body_id, _pos, _local in selected):
        required_item = next((item for item in candidates if int(item[1]) == int(required_body_id)), None)
        if required_item is not None:
            selected = [required_item] + [item for item in selected if int(item[1]) != int(required_body_id)]
            selected = selected[:ASSIST_PATCH_MAX_POINTS]
    body_ids = [int(body_id) for _label, body_id, _pos, _local in selected]
    positions = np.asarray([env.data.xpos[body_id].copy() for body_id in body_ids], dtype=np.float64)
    local_positions = np.asarray([local for _label, _body_id, _pos, local in selected], dtype=np.float64)
    if local_positions.size:
        thickness_proxy = float(np.ptp(local_positions[:, 1]) + 2.0 * 0.0045)
        jaw_span_y = float(np.ptp(local_positions[:, 1]))
        side_negative = int(np.count_nonzero(local_positions[:, 1] < -0.001))
        side_positive = int(np.count_nonzero(local_positions[:, 1] > 0.001))
    else:
        thickness_proxy = 0.0
        jaw_span_y = 0.0
        side_negative = 0
        side_positive = 0
    left_contact, right_contact = finger_sack_contact_flags(env)
    captured_speed, neighbor_speed, coupled_motion_score = _patch_motion_stats(env, body_ids)
    tangential_slip_proxy = captured_speed
    min_captured_points = 2 if allow_visual_pinch_latch else GUARD_MIN_CAPTURED_POINTS
    pad_contact_area_proxy = float(len(body_ids) * SURFACE_CAPTURE_AREA_PER_POINT_M2)
    side_balance_proxy = float(min(side_negative, side_positive) / max(max(side_negative, side_positive), 1))
    area_capture_approved = bool(
        len(body_ids) >= 2
        and pad_contact_area_proxy >= SURFACE_CAPTURE_MIN_AREA_M2
        and thickness_proxy >= GUARD_MIN_THICKNESS_M
        and (jaw_span_y >= SURFACE_CAPTURE_MIN_SPAN_Y_M or left_contact or right_contact)
    )
    base_quality_ok = (
        len(body_ids) >= min_captured_points
        and thickness_proxy >= GUARD_MIN_THICKNESS_M
        and tangential_slip_proxy <= GUARD_MAX_TANGENTIAL_SLIP_MPS
        and coupled_motion_score <= GUARD_MAX_COUPLED_MOTION
    )
    guarded_latch_approved = bool(base_quality_ok and area_capture_approved) if allow_visual_pinch_latch else bool(
        base_quality_ok
        and left_contact
        and right_contact
    )
    latch_method = "none"
    if guarded_latch_approved:
        latch_method = "surface_capture_latch" if allow_visual_pinch_latch else "adhesion_force_guarded"
    capture_quality = {
        "captured_shell_points": len(selected),
        "captured_required_target": bool(
            required_body_id is not None and any(int(body_id) == int(required_body_id) for body_id in body_ids)
        ),
        "candidate_shell_points": candidate_count,
        "jaw_gap_m": float(env.left_gripper_gap()),
        "left_finger_contact": bool(left_contact),
        "right_finger_contact": bool(right_contact),
        "pad_contact_area_proxy_m2": pad_contact_area_proxy,
        "area_capture_threshold_m2": float(SURFACE_CAPTURE_MIN_AREA_M2),
        "area_capture_approved": bool(area_capture_approved),
        "jaw_span_y_m": jaw_span_y,
        "side_negative_points": side_negative,
        "side_positive_points": side_positive,
        "side_balance_proxy": side_balance_proxy,
        "thickness_proxy_m": thickness_proxy,
        "tangential_slip_proxy_mps": tangential_slip_proxy,
        "neighbor_speed_mps": neighbor_speed,
        "coupled_motion_score": coupled_motion_score,
        "guarded_latch_approved": bool(guarded_latch_approved),
        "latch_method": latch_method,
        "requires_bilateral_contact": not bool(allow_visual_pinch_latch),
        "active_max_gap_m": float(PRECAPTURE_GRIPPER_GAP_M + 0.012 if allow_visual_pinch_latch else ASSIST_ACTIVE_MAX_GRIPPER_GAP_M),
    }
    return HoldPatch(
        body_ids=body_ids,
        relative_positions=positions - center_xyz[None, :],
        captured_body_ids=body_ids,
        capture_quality=capture_quality,
        jaw_local_positions=local_positions.copy(),
        target_world_positions=positions.copy(),
    )


def _shell_slide_joint_info(env: DualUR5TopGraspEnv, body_id: int) -> list[tuple[int, int, np.ndarray]]:
    joint_info: list[tuple[int, int, np.ndarray]] = []
    for joint_id in range(env.model.njnt):
        if int(env.model.jnt_bodyid[joint_id]) != int(body_id):
            continue
        if int(env.model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        qpos_adr = int(env.model.jnt_qposadr[joint_id])
        qvel_adr = int(env.model.jnt_dofadr[joint_id])
        axis = env.model.jnt_axis[joint_id].copy()
        joint_info.append((qpos_adr, qvel_adr, axis))
    return joint_info


def _pin_shell_body_to_world(env: DualUR5TopGraspEnv, body_id: int, target_world: np.ndarray) -> None:
    # flexcomp의 각 shell point는 bag_frame 아래 3개 slide joint로 표현된다.
    # 시연용 capture latch에서는 선택된 2~3개 point만 gripper 좌표계에 맞춰 직접 위치시킨다.
    joint_info = _shell_slide_joint_info(env, body_id)
    if len(joint_info) < 3:
        return
    bag_frame_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "bag_frame")
    if bag_frame_id < 0:
        return
    bag_pos = env.data.xpos[bag_frame_id].copy()
    bag_frame = env.data.xmat[bag_frame_id].reshape(3, 3).copy()
    target_local = bag_frame.T @ (np.asarray(target_world, dtype=np.float64) - bag_pos)
    slide_offset = target_local - env.model.body_pos[body_id]
    for qpos_adr, qvel_adr, axis in joint_info:
        env.data.qpos[qpos_adr] = float(np.dot(slide_offset, axis))
        env.data.qvel[qvel_adr] = 0.0


def _bag_frame_freejoint_addresses(env: DualUR5TopGraspEnv) -> tuple[int, int] | None:
    bag_frame_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "bag_frame")
    if bag_frame_id < 0:
        return None
    for joint_id in range(env.model.njnt):
        if int(env.model.jnt_bodyid[joint_id]) != int(bag_frame_id):
            continue
        if int(env.model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
            return int(env.model.jnt_qposadr[joint_id]), int(env.model.jnt_dofadr[joint_id])
    return None


def _translate_bag_frame(env: DualUR5TopGraspEnv, delta_world: np.ndarray) -> None:
    addresses = _bag_frame_freejoint_addresses(env)
    if addresses is None:
        return
    qpos_adr, qvel_adr = addresses
    env.data.qpos[qpos_adr : qpos_adr + 3] += np.asarray(delta_world, dtype=np.float64)
    env.data.qvel[qvel_adr : qvel_adr + 3] = 0.0


def _surface_capture_frame_delta(mean_patch_delta: np.ndarray) -> np.ndarray:
    # top patch는 gripper를 따르되, 자루 전체 frame은 특히 z축에서 늦게 따라오게 해서 하부 처짐을 남긴다.
    raw = np.asarray(mean_patch_delta, dtype=np.float64)
    delta = np.zeros(3, dtype=np.float64)
    delta[:2] = raw[:2] * SURFACE_CAPTURE_FRAME_FOLLOW_GAIN_XY
    xy_norm = float(np.linalg.norm(delta[:2]))
    if xy_norm > SURFACE_CAPTURE_MAX_FRAME_STEP_XY_M:
        delta[:2] *= SURFACE_CAPTURE_MAX_FRAME_STEP_XY_M / xy_norm

    if raw[2] >= 0.0:
        delta[2] = min(
            raw[2] * SURFACE_CAPTURE_FRAME_FOLLOW_GAIN_Z_UP,
            SURFACE_CAPTURE_MAX_FRAME_STEP_Z_UP_M,
        )
    else:
        delta[2] = max(
            raw[2] * SURFACE_CAPTURE_FRAME_FOLLOW_GAIN_Z_DOWN,
            -SURFACE_CAPTURE_MAX_FRAME_STEP_Z_DOWN_M,
        )
    return delta


def _clamp_internal_content_joints(env: DualUR5TopGraspEnv) -> None:
    # 내부 clump는 DEM이 아니라 제한된 slide surrogate이므로, 수치 충격 후 range 밖으로 밀리면 즉시 되돌린다.
    for joint_id in range(env.model.njnt):
        joint_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""
        if not (joint_name.startswith("bag_content_clump_") or joint_name.startswith("bag_content_support_")):
            continue
        if int(env.model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_SLIDE):
            continue
        qpos_adr = int(env.model.jnt_qposadr[joint_id])
        qvel_adr = int(env.model.jnt_dofadr[joint_id])
        low, high = env.model.jnt_range[joint_id]
        value = float(env.data.qpos[qpos_adr])
        clipped = float(np.clip(value, low, high))
        if not np.isclose(value, clipped):
            env.data.qpos[qpos_adr] = clipped
            env.data.qvel[qvel_adr] = 0.0


def _is_internal_content_body_name(body_name: str) -> bool:
    return body_name.startswith("bag_content_clump_") or body_name == "bag_content_support"


def _shell_point_index_from_body_name(env: DualUR5TopGraspEnv, body_id: int) -> int | None:
    body_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, int(body_id)) or ""
    if not body_name.startswith("bag_shell_"):
        return None
    try:
        return int(body_name.split("_")[2])
    except (IndexError, ValueError):
        return None


def _offset_limit_for_shell_point(point_index: int) -> float:
    if point_index <= RING_COUNT:
        return SURFACE_CAPTURE_MAX_TOP_OFFSET_M
    if point_index <= 2 * RING_COUNT:
        return SURFACE_CAPTURE_MAX_MID_OFFSET_M
    return SURFACE_CAPTURE_MAX_LOWER_OFFSET_M


def _clamp_shell_deformation(env: DualUR5TopGraspEnv, protected_body_ids: set[int] | None = None) -> None:
    protected = protected_body_ids or set()
    for body_id in collect_shell_body_ids(env.model):
        if int(body_id) in protected:
            continue
        point_index = _shell_point_index_from_body_name(env, body_id)
        if point_index is None:
            continue
        joint_info = _shell_slide_joint_info(env, body_id)
        if len(joint_info) < 3:
            continue
        axes = np.asarray([axis for _qpos, _qvel, axis in joint_info], dtype=np.float64)
        values = np.asarray([env.data.qpos[qpos_adr] for qpos_adr, _qvel, _axis in joint_info], dtype=np.float64)
        offset = axes.T @ values
        limit = _offset_limit_for_shell_point(point_index)
        offset_norm = float(np.linalg.norm(offset))
        if offset_norm <= limit:
            continue
        scaled_offset = offset * (limit / max(offset_norm, 1e-9))
        for qpos_adr, qvel_adr, axis in joint_info:
            env.data.qpos[qpos_adr] = float(np.dot(scaled_offset, axis))
            env.data.qvel[qvel_adr] = 0.0


def _apply_surface_capture_sag_load(env: DualUR5TopGraspEnv, protected_body_ids: set[int]) -> None:
    # 실제 자루는 상단 patch가 잡혀도 하부 shell과 내용물이 중력 방향으로 늦게 따라온다.
    # MuJoCo flex surrogate에서는 이 시각적 sag를 보존하기 위해 약한 추가 하중만 준다.
    for body_id in collect_shell_body_ids(env.model):
        if int(body_id) in protected_body_ids:
            continue
        point_index = _shell_point_index_from_body_name(env, body_id)
        if point_index is None or point_index <= RING_COUNT:
            continue
        env.data.xfrc_applied[body_id, 2] -= SURFACE_CAPTURE_EXTRA_SHELL_SAG_FORCE_N

    for body_id in range(env.model.nbody):
        body_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if _is_internal_content_body_name(body_name):
            env.data.xfrc_applied[body_id, 2] -= SURFACE_CAPTURE_EXTRA_CONTENT_SAG_FORCE_N


def _filtered_surface_targets(patch: HoldPatch, desired_worlds: np.ndarray) -> np.ndarray | None:
    if patch.target_world_positions is None or patch.target_world_positions.shape != desired_worlds.shape:
        patch.target_world_positions = desired_worlds.copy()
        return patch.target_world_positions

    requested_delta = desired_worlds - patch.target_world_positions
    requested_norms = np.linalg.norm(requested_delta, axis=1)
    max_requested = float(np.max(requested_norms)) if len(requested_norms) else 0.0
    patch.capture_quality["requested_ee_jump_m"] = max_requested
    if max_requested > SURFACE_CAPTURE_RELEASE_EE_JUMP_M:
        patch.capture_quality["release_reason"] = "manual_motion_too_fast"
        return None

    step_scale = np.ones_like(requested_norms)
    moving = requested_norms > SURFACE_CAPTURE_MAX_TARGET_STEP_M
    step_scale[moving] = SURFACE_CAPTURE_MAX_TARGET_STEP_M / np.maximum(requested_norms[moving], 1e-9)
    patch.target_world_positions = patch.target_world_positions + requested_delta * step_scale[:, None]
    patch.capture_quality["filtered_target_lag_m"] = float(
        np.max(np.linalg.norm(desired_worlds - patch.target_world_positions, axis=1))
    )
    return patch.target_world_positions


def _apply_surface_capture_latch(env: DualUR5TopGraspEnv, patch: HoldPatch) -> bool:
    if patch.jaw_local_positions is None or len(patch.jaw_local_positions) != len(patch.body_ids):
        return False
    site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "left_gripper_pinch")
    if site_id < 0:
        return False
    jaw_center = env.data.site_xpos[site_id].copy()
    jaw_frame = env.data.site_xmat[site_id].reshape(3, 3).copy()
    desired_worlds = np.asarray([jaw_center + jaw_frame @ local for local in patch.jaw_local_positions], dtype=np.float64)
    target_worlds = _filtered_surface_targets(patch, desired_worlds)
    if target_worlds is None:
        return False
    current_worlds = np.asarray([env.data.xpos[body_id].copy() for body_id in patch.body_ids], dtype=np.float64)
    stretch = float(np.max(np.linalg.norm(target_worlds - current_worlds, axis=1))) if len(current_worlds) else 0.0
    if stretch > SURFACE_CAPTURE_MAX_PATCH_STRETCH_M:
        patch.capture_quality["release_reason"] = "excessive_patch_stretch"
        return False

    protected_body_ids = set(patch.body_ids)
    frame_delta = _surface_capture_frame_delta(np.mean(target_worlds - current_worlds, axis=0))
    _translate_bag_frame(env, frame_delta)
    _clamp_internal_content_joints(env)
    _clamp_shell_deformation(env, protected_body_ids=protected_body_ids)
    mujoco.mj_forward(env.model, env.data)

    for body_id, target_world in zip(patch.body_ids, target_worlds):
        current_world = env.data.xpos[body_id].copy()
        blended_target = current_world + SURFACE_CAPTURE_PIN_BLEND * (target_world - current_world)
        _pin_shell_body_to_world(env, body_id, blended_target)
    _clamp_internal_content_joints(env)
    _clamp_shell_deformation(env, protected_body_ids=protected_body_ids)
    env.data.xfrc_applied[:, :] = 0.0
    _apply_surface_capture_sag_load(env, protected_body_ids)
    mujoco.mj_forward(env.model, env.data)
    return True


def hold_patch_is_active(env: DualUR5TopGraspEnv, patch: HoldPatch | None, center_xyz: np.ndarray) -> bool:
    if patch is None or not patch.body_ids:
        return False
    if not bool(patch.capture_quality.get("guarded_latch_approved", False)):
        return False
    # 보조 파지는 "닫힌 gripper가 실제 patch를 잡고 있다"는 surrogate이므로,
    # gripper를 다시 열면 즉시 꺼져야 한다.
    active_max_gap = float(patch.capture_quality.get("active_max_gap_m", ASSIST_ACTIVE_MAX_GRIPPER_GAP_M))
    if env.left_gripper_gap() > active_max_gap:
        patch.capture_quality["release_reason"] = "gripper_opened"
        return False
    if patch.capture_quality.get("latch_method") == "surface_capture_latch":
        return True
    left_contact, right_contact = finger_sack_contact_flags(env)
    requires_bilateral_contact = bool(patch.capture_quality.get("requires_bilateral_contact", True))
    if requires_bilateral_contact and (not left_contact or not right_contact):
        patch.capture_quality["release_reason"] = "one_side_contact_loss"
        return False
    captured_speed, _neighbor_speed, coupled_motion_score = _patch_motion_stats(env, patch.body_ids)
    if captured_speed > RELEASE_MAX_SLIP_MPS:
        patch.capture_quality["release_reason"] = "slip_spike"
        return False
    if coupled_motion_score > RELEASE_MAX_COUPLED_MOTION:
        patch.capture_quality["release_reason"] = "neighbor_co_motion"
        return False
    site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "left_gripper_pinch")
    if site_id < 0:
        return False
    jaw_center = env.data.site_xpos[site_id].copy()
    jaw_frame = env.data.site_xmat[site_id].reshape(3, 3).copy()
    gap_half = max(0.5 * env.left_gripper_gap(), 0.0)
    for body_id, relative in zip(patch.body_ids, patch.relative_positions):
        local = jaw_frame.T @ (env.data.xpos[body_id].copy() - jaw_center)
        sag = float((center_xyz + relative)[2] - env.data.xpos[body_id, 2])
        if sag > RELEASE_MAX_SAG_M:
            patch.capture_quality["release_reason"] = "excessive_sag"
            return False
        if (
            abs(float(local[0])) <= JAW_CAPTURE_HALF_X * 1.8
            and abs(float(local[2])) <= JAW_CAPTURE_HALF_Z * 1.8
            and abs(float(local[1])) <= gap_half + JAW_CAPTURE_Y_MARGIN * 2.5
        ):
            return True
    # 잡힌 patch가 jaw 주변을 벗어나면 assist도 풀어서 염력처럼 끌려오지 않게 한다.
    patch.capture_quality["release_reason"] = "jaw_escape"
    return False


def apply_hold_patch(env: DualUR5TopGraspEnv, patch: HoldPatch | None, center_xyz: np.ndarray) -> bool:
    env.data.xfrc_applied[:, :] = 0.0
    if not hold_patch_is_active(env, patch, center_xyz):
        return False
    if patch.capture_quality.get("latch_method") == "surface_capture_latch":
        return _apply_surface_capture_latch(env, patch)
    if patch.capture_quality.get("latch_method") == "visual_pinch_latch":
        kp = VISUAL_PINCH_KP
        kd = VISUAL_PINCH_KD
        max_force = VISUAL_PINCH_MAX_FORCE_PER_POINT_N
    else:
        kp = ASSIST_KP
        kd = ASSIST_KD
        max_force = ASSIST_MAX_FORCE_PER_POINT_N
    for body_id, relative in zip(patch.body_ids, patch.relative_positions):
        target = center_xyz + relative
        pos = env.data.xpos[body_id].copy()
        vel = env.data.cvel[body_id, 3:6].copy()
        force = kp * (target - pos) - kd * vel
        norm = float(np.linalg.norm(force))
        if norm > max_force:
            force *= max_force / norm
        env.data.xfrc_applied[body_id, :3] += force
    return True


def step_for_seconds(
    env: DualUR5TopGraspEnv,
    seconds: float,
    *,
    viewer=None,
    speed: float = 1.0,
    hold_patch: HoldPatch | None = None,
) -> None:
    steps = max(1, int(seconds / env.model.opt.timestep))
    sleep_dt = env.model.opt.timestep / max(speed, 1e-6)
    for _ in range(steps):
        step_start = time.perf_counter()
        if hold_patch is not None:
            apply_hold_patch(env, hold_patch, env.end_effector_pos("left"))
        else:
            env.data.xfrc_applied[:, :] = 0.0
        _clamp_internal_content_joints(env)
        if hold_patch is not None:
            _clamp_shell_deformation(env, protected_body_ids=set(hold_patch.body_ids))
        env.step()
        _clamp_internal_content_joints(env)
        if hold_patch is not None:
            _clamp_shell_deformation(env, protected_body_ids=set(hold_patch.body_ids))
        if viewer is not None:
            viewer.sync()
            remaining = sleep_dt - (time.perf_counter() - step_start)
            if remaining > 0:
                time.sleep(remaining)


def report(env: DualUR5TopGraspEnv, stage: str, target_label: str, target_body_id: int) -> None:
    ee = env.end_effector_pos("left")
    target = env.data.xpos[target_body_id].copy()
    dist = float(np.linalg.norm(ee - target))
    gap = env.left_gripper_gap()
    print(
        f"stage={stage} label={target_label} ee={np.round(ee, 4).tolist()} "
        f"target={np.round(target, 4).tolist()} dist_m={dist:.4f} gap_m={gap:.4f}"
    )


def internal_content_escape_metrics(env: DualUR5TopGraspEnv) -> dict[str, float | bool | int]:
    shell_ids = collect_shell_body_ids(env.model)
    shell_positions = np.asarray([env.data.xpos[body_id].copy() for body_id in shell_ids], dtype=np.float64)
    content_positions = []
    for body_id in range(env.model.nbody):
        body_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if body_name.startswith("bag_content_clump_") or body_name == "bag_content_support":
            content_positions.append(env.data.xpos[body_id].copy())
    if shell_positions.size == 0 or not content_positions:
        return {
            "escaped_internal_bodies": False,
            "escaped_internal_body_count": 0,
            "max_content_shell_overrun_m": 0.0,
        }
    content_array = np.asarray(content_positions, dtype=np.float64)
    shell_min = shell_positions.min(axis=0) - np.array([0.040, 0.040, 0.065], dtype=np.float64)
    shell_max = shell_positions.max(axis=0) + np.array([0.040, 0.040, 0.035], dtype=np.float64)
    over_low = np.maximum(shell_min[None, :] - content_array, 0.0)
    over_high = np.maximum(content_array - shell_max[None, :], 0.0)
    overrun = np.linalg.norm(over_low + over_high, axis=1)
    escaped_count = int(np.count_nonzero(overrun > 1e-5))
    return {
        "escaped_internal_bodies": bool(escaped_count > 0),
        "escaped_internal_body_count": escaped_count,
        "max_content_shell_overrun_m": float(np.max(overrun)) if len(overrun) else 0.0,
    }


def sack_sag_metrics(env: DualUR5TopGraspEnv, target_body_id: int) -> dict[str, float]:
    target_z = float(env.data.xpos[target_body_id, 2])
    lower_positions = []
    for body_id in collect_shell_body_ids(env.model):
        point_index = _shell_point_index_from_body_name(env, body_id)
        if point_index is not None and point_index > 2 * RING_COUNT:
            lower_positions.append(env.data.xpos[body_id].copy())
    content_positions = []
    for body_id in range(env.model.nbody):
        body_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if _is_internal_content_body_name(body_name):
            content_positions.append(env.data.xpos[body_id].copy())

    lower_mean_z = float(np.mean(np.asarray(lower_positions)[:, 2])) if lower_positions else target_z
    content_mean_z = float(np.mean(np.asarray(content_positions)[:, 2])) if content_positions else lower_mean_z
    return {
        "lower_shell_mean_z": lower_mean_z,
        "content_mean_z": content_mean_z,
        "target_to_lower_sag_m": float(target_z - lower_mean_z),
        "target_to_content_lag_m": float(target_z - content_mean_z),
    }


def compute_hold_metrics(
    env: DualUR5TopGraspEnv,
    target_body_id: int,
    closed_target_xyz: np.ndarray,
    hold_start_target_xyz: np.ndarray,
    hold_start_ee_xyz: np.ndarray,
    follow_command_xyz: np.ndarray | None = None,
) -> dict[str, float | bool]:
    final_target = env.data.xpos[target_body_id].copy()
    final_ee = env.end_effector_pos("left")
    lift_delta = float(final_target[2] - closed_target_xyz[2])
    relative_start = hold_start_target_xyz - hold_start_ee_xyz
    relative_final = final_target - final_ee
    hold_slip = float(np.linalg.norm(relative_final - relative_start))
    final_dist = float(np.linalg.norm(final_target - final_ee))
    ee_motion = final_ee - hold_start_ee_xyz
    target_motion = final_target - hold_start_target_xyz
    ee_motion_m = float(np.linalg.norm(ee_motion))
    target_motion_m = float(np.linalg.norm(target_motion))
    if ee_motion_m > 1e-6:
        target_follow_along_m = float(np.dot(target_motion, ee_motion / ee_motion_m))
    else:
        target_follow_along_m = 0.0
    follow_ratio = target_follow_along_m / max(ee_motion_m, 1e-6)
    commanded_motion_m = (
        float(np.linalg.norm(np.asarray(follow_command_xyz, dtype=np.float64) - hold_start_ee_xyz))
        if follow_command_xyz is not None
        else 0.0
    )
    pass_fail = (
        ee_motion_m >= 0.040
        and target_follow_along_m >= 0.030
        and follow_ratio >= 0.45
        and hold_slip <= HOLD_SUCCESS_MAX_SLIP_M
        and final_dist <= HOLD_SUCCESS_MAX_FINAL_DIST_M
    )
    metrics = {
        "target_lift_delta_m": lift_delta,
        "hold_slip_m": hold_slip,
        "final_ee_target_dist_m": final_dist,
        "commanded_follow_motion_m": commanded_motion_m,
        "actual_ee_motion_m": ee_motion_m,
        "target_motion_m": target_motion_m,
        "target_follow_along_m": target_follow_along_m,
        "follow_ratio": float(follow_ratio),
        "pass_fail": bool(pass_fail),
    }
    metrics.update(internal_content_escape_metrics(env))
    metrics.update(sack_sag_metrics(env, target_body_id))
    return metrics


def run_autograsp(
    *,
    scenario: str,
    content_case: str,
    preferred_label: str | None,
    with_content_support: bool,
    assist: bool,
    viewer_enabled: bool,
    speed: float,
) -> int:
    env = DualUR5TopGraspEnv(scenario_name=scenario, content_case=content_case, with_content_support=with_content_support)
    print("mode=visual_demo_assisted")
    print(f"requested_target_label={preferred_label or 'auto'}")
    env.print_summary()
    env.set_left_gripper_gap(OPEN_GRIPPER_GAP_M, immediate=True)

    def _run_sequence(viewer=None) -> int:
        print("stage=settle")
        step_for_seconds(env, STAGE_SETTLE_SECONDS, viewer=viewer, speed=speed)

        target_label, target_body_id, target_pos = choose_grasp_target(env, preferred_label=preferred_label)
        if hasattr(env, "set_content_bias_from_grasp"):
            bias_info = env.set_content_bias_from_grasp(target_pos)
            print(f"eccentric_fill_bias={bias_info}")
            target_pos = env.data.xpos[target_body_id].copy()
        # pinch site가 target보다 아주 살짝 위에 오도록 잡는다. 너무 낮으면 shell을 밀고 지나간다.
        grasp_center = current_grasp_center(env, target_body_id)
        pregrasp = grasp_center + np.array([0.0, 0.0, PREGRASP_OFFSET_Z], dtype=np.float64)

        print(f"chosen_target_label={target_label}")
        print(f"chosen_target_body={target_body_id}")
        print(f"chosen_target_xyz={target_pos.tolist()}")

        env.set_left_gripper_gap(OPEN_GRIPPER_GAP_M, immediate=False)
        ik_pre = env.solve_ee_position_ik(
            "left",
            pregrasp,
            iterations=160,
            tolerance=0.004,
            damping=0.08,
            max_step_deg=5.0,
        )
        step_for_seconds(env, STAGE_MOVE_SECONDS, viewer=viewer, speed=speed)
        print(f"ik_pregrasp={ik_pre}")
        report(env, "pregrasp", target_label, target_body_id)

        # 이동 중 자루가 더 눕거나 내부 support bias 때문에 target patch가 바뀔 수 있으므로
        # 접근 직전에 현재 shell 상태 기준으로 한 번 더 고른다.
        target_label, target_body_id, target_pos = choose_grasp_target(env, preferred_label=preferred_label)
        print(f"reacquired_target_label={target_label}")
        print(f"reacquired_target_body={target_body_id}")
        print(f"reacquired_target_xyz={target_pos.tolist()}")

        grasp_center = current_grasp_center(env, target_body_id)
        approach = grasp_center + np.array([0.0, 0.0, APPROACH_OFFSET_Z], dtype=np.float64)
        ik_approach = set_target_and_step(env, approach, seconds=STAGE_MOVE_SECONDS, viewer=viewer, speed=speed)
        print(f"ik_approach={ik_approach}")
        report(env, "approach", target_label, target_body_id)

        set_gripper_gap_gradual(
            env,
            env.left_gripper_gap(),
            PRECAPTURE_GRIPPER_GAP_M,
            seconds=0.45,
            viewer=viewer,
            speed=speed,
        )
        report(env, "precapture_close", target_label, target_body_id)

        grasp_center = current_grasp_center(env, target_body_id)
        ik_grasp = set_target_and_step(env, grasp_center, seconds=STAGE_MOVE_SECONDS, viewer=viewer, speed=speed)
        print(f"ik_grasp={ik_grasp}")
        report(env, "grasp_pose", target_label, target_body_id)

        grasp_center = current_grasp_center(env, target_body_id)
        ik_final_grasp = set_target_and_step(
            env,
            grasp_center,
            seconds=0.5 * STAGE_MOVE_SECONDS,
            viewer=viewer,
            speed=speed,
        )
        print(f"ik_final_grasp={ik_final_grasp}")
        report(env, "final_grasp_pose", target_label, target_body_id)

        set_gripper_gap_gradual(env, env.left_gripper_gap(), 0.0, seconds=STAGE_CLOSE_SECONDS, viewer=viewer, speed=speed)
        report(env, "closed", target_label, target_body_id)
        closed_target_xyz = env.data.xpos[target_body_id].copy()

        hold_patch = (
            make_hold_patch(
                env,
                env.end_effector_pos("left"),
                required_body_id=target_body_id,
                allow_visual_pinch_latch=True,
            )
            if assist
            else None
        )
        print(f"assist_hold={assist}")
        print(f"hold_patch_body_count={0 if hold_patch is None else len(hold_patch.body_ids)}")
        print(f"hold_patch_capture_quality={None if hold_patch is None else hold_patch.capture_quality}")
        if hold_patch is not None and not bool(hold_patch.capture_quality.get("guarded_latch_approved", False)):
            print("guarded_latch_activated=False")
            hold_patch = None
        elif hold_patch is not None:
            print("guarded_latch_activated=True")
        if hold_patch is not None and hold_patch.body_ids and target_body_id not in hold_patch.body_ids:
            target_body_id = hold_patch.body_ids[0]
            target_label = label_for_shell_body_id(env, target_body_id, target_label)
            closed_target_xyz = env.data.xpos[target_body_id].copy()
            print(f"actual_captured_body={target_body_id}")
            print(f"actual_captured_label={target_label}")

        print("stage=hold_only")
        step_for_seconds(env, STAGE_INITIAL_HOLD_SECONDS, viewer=viewer, speed=speed, hold_patch=hold_patch)
        report(env, "hold_only", target_label, target_body_id)

        hold_start_target_xyz = env.data.xpos[target_body_id].copy()
        hold_start_ee_xyz = env.end_effector_pos("left")
        follow_target = hold_start_ee_xyz + FOLLOW_MOVE_DELTA
        ik_follow = set_target_and_step(
            env,
            follow_target,
            seconds=STAGE_HOLD_SECONDS,
            viewer=viewer,
            speed=speed,
            hold_patch=hold_patch,
        )
        print(f"ik_follow_validation={ik_follow}")
        report(env, "follow_validation", target_label, target_body_id)
        hold_metrics = compute_hold_metrics(
            env,
            target_body_id,
            closed_target_xyz,
            hold_start_target_xyz,
            hold_start_ee_xyz,
            follow_command_xyz=follow_target,
        )
        print(f"hold_validation_seconds={STAGE_HOLD_SECONDS:.1f}")
        print(f"hold_validation_metrics={hold_metrics}")
        print("autograsp_demo_done=True")
        return 0

    if not viewer_enabled:
        return _run_sequence(None)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = np.array([0.45, 0.0, 0.25])
        viewer.cam.distance = 1.25
        viewer.cam.azimuth = 132.0
        viewer.cam.elevation = -18.0
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXFACE] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXEDGE] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXVERT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        return _run_sequence(viewer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="visual_demo_assisted scripted Dual UR5 top-grasp demonstration")
    parser.add_argument("--scenario", choices=available_scenarios(), default="exposed_seam")
    parser.add_argument("--content-case", choices=available_content_cases(), default="underfilled")
    parser.add_argument("--preferred-label", choices=("seam", "fold", "plain_top"), default=None)
    parser.add_argument("--no-content-support", action="store_true")
    parser.add_argument("--no-assist", action="store_true", help="disable quality-gated local hold assist")
    parser.add_argument("--headless", action="store_true", help="run without opening the MuJoCo viewer")
    parser.add_argument("--speed", type=float, default=0.7, help="viewer playback speed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_autograsp(
        scenario=args.scenario,
        content_case=args.content_case,
        preferred_label=args.preferred_label,
        with_content_support=not args.no_content_support,
        assist=not args.no_assist,
        viewer_enabled=not args.headless,
        speed=args.speed,
    )


if __name__ == "__main__":
    raise SystemExit(main())
