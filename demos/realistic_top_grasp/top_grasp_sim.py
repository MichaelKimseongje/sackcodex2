from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import mujoco
import numpy as np

from generate_sack_mesh import (
    BAG_FRAME_Z,
    BAG_SHELL_BODY_PREFIX,
    CONTENT_CLUMP_NAMES,
    JAW_CLOSED_GAP,
    JAW_OPEN_GAP,
    JAW_PAD_HALF_X,
    JAW_PAD_HALF_Y,
    JAW_PAD_HALF_Z,
    LEFT_JAW_BODY,
    OUTPUT_DIR,
    CCD_MODES,
    NATIVECCD_MODES,
    PAD_CONDIM_OPTIONS,
    PAD_PROFILES,
    RIGHT_JAW_BODY,
    RING_COUNT,
    SCENARIOS,
    SELF_COLLISION_MODE,
    SELF_COLLISION_MODES,
    VERTCOLLIDE_MODES,
    available_content_cases,
    available_scenarios,
    make_sack_points,
    nominal_grasp_center,
    write_scene_xml,
)


# 동일한 quality rule이 모든 scenario/trial에 적용된다.
QUALITY_THRESHOLD = 0.57
CAPTURE_MARGIN = 0.011
CONTACT_MARGIN = 0.016
PULL_TEST_LIFT_M = 0.040
FULL_LIFT_M = 0.135
MICRO_LIFT_M = 0.075
CONTACT_ONLY_HOLD_SECONDS = 0.80
FINAL_SEARCH_LIFT_M = 0.045
FINAL_SEARCH_HOLD_SECONDS = 0.75
SEARCH_APPROACH_Z = 0.115
SEARCH_PROBE_Z = 0.006
SEARCH_TUG_LIFT_M = 0.022
SEARCH_MICRO_LIFT_M = 0.035
SEARCH_APPROACH_SECONDS = 0.20
SEARCH_CLOSE_SECONDS = 0.24
PRECOMPRESSION_DWELL_SECONDS = 0.06
SEARCH_TUG_SECONDS = 0.20
SEARCH_LIFT_SECONDS = 0.28
RECTIFICATION_SHIFT_M = 0.018
MAX_CANDIDATES_TO_PROBE = 5
MODE_PATH_TOKEN = {
    "contact_only_eval": "c",
    "qualification_gated_latch_eval": "l",
    "qualification_gated_capture": "cap",
}
LATCH_MODES = {"qualification_gated_latch_eval", "qualification_gated_capture"}
AUTO_SEARCH_MODES = {"contact_only_eval", *LATCH_MODES}
CAPTURE_MIN_THICKNESS_M = 0.006
# 38-point shell surrogate에서는 정상적으로 뭉쳐 잡힌 fold/root patch도 10 cm 이상 두껍게 측정될 수 있다.
CAPTURE_MAX_THICKNESS_M = 0.180
CAPTURE_MAX_TANGENTIAL_SLIP_MPS = 0.70
CAPTURE_MIN_LOAD_FOLLOWING_RATIO = -0.15
CAPTURE_MIN_MICRO_LIFT_M = 0.003
CLOSE_TIMESTEP_OPTIONS = (0.001, 0.0005, 0.0002)
ROLLBACK_MAX_SHELL_SPEED_MPS = 1.20
ROLLBACK_MAX_QVEL_NORM = 55.0
ROLLBACK_MIN_CONTACT_DIST_M = -0.012
SHAKE_AMPLITUDE_M = 0.016
SETTLE_SECONDS = 0.45
APPROACH_SECONDS = 0.50
CLOSE_SECONDS = 0.55
PULL_SECONDS = 0.48
LIFT_SECONDS = 0.72
SHAKE_SECONDS = 0.55
FRAME_TARGET_COUNT = 90


# trial은 파지 접근 위치의 작은 차이만 바꾼다. 성공 여부 자체는 label을 보지 않는다.
TRIAL_OFFSETS: dict[str, tuple[tuple[float, float, float], ...]] = {
    "exposed_seam": (
        (0.000, 0.000, 0.000),
        (0.045, -0.004, 0.000),
        (-0.030, -0.046, -0.004),
        (0.018, 0.010, 0.004),
    ),
    "simple_fold": (
        (0.000, 0.000, 0.000),
        (0.034, 0.004, 0.000),
        (-0.042, -0.010, -0.006),
        (-0.014, 0.018, 0.004),
    ),
    "severe_fold": (
        (0.000, 0.000, 0.000),
        (0.038, 0.002, 0.000),
        (-0.038, -0.014, -0.008),
        (0.012, 0.020, 0.006),
    ),
}


@dataclass
class GraspPatch:
    body_ids: list[int]
    point_indices: list[int]
    relative_positions: np.ndarray


@dataclass
class TrialRuntime:
    model: mujoco.MjModel
    data: mujoco.MjData
    scenario_name: str
    content_case: str
    selfcollide_mode: str
    noslip_iterations: int
    multiccd_mode: str
    nativeccd_mode: str
    pad_profile: str
    pad_condim: int
    vertcollide_mode: str
    shell_thickness_scale: float
    close_timestep: float
    gripper_kv: float
    gripper_dampratio: float
    labels: list[str]
    shell_point_to_body: dict[int, int]
    left_mocap_id: int
    right_mocap_id: int
    center_mocap_id: int


@dataclass
class GraspCandidate:
    rank: int
    root_ring_index: int
    point_indices: list[int]
    center: np.ndarray
    requested_target_label: str
    policy_tag: str
    score: float
    local_thickness_proxy: float
    backing_score: float
    bilateral_clearance: float
    top_center_thin_flap_penalty: float
    rectification_used: str = "none"


@dataclass
class ProbeOutcome:
    candidate: GraspCandidate
    captured_at_close: list[int]
    requested_target_label: str
    actual_region_label_at_close: str
    left_contact_present: bool
    right_contact_present: bool
    trapped_shell_points: int
    trapped_patch_arc_length: float
    captured_shell_points: int
    bundle_thickness_proxy: float
    bilateral_contact_balance: float
    tangential_slip_proxy: float
    jaw_escape: bool
    rollback_used: bool
    contact_persistence_ms: float
    pull_test_slip_mm: float
    micro_lift_survival: bool
    load_following_ratio: float
    lift_height: float
    hold_time: float
    final_slip_distance: float
    drop_or_not: bool
    pass_fail: bool
    latch_activated: bool
    latch_activation_time: float


def make_render_option() -> mujoco.MjvOption:
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    option.flags[mujoco.mjtVisFlag.mjVIS_FLEXFACE] = True
    option.flags[mujoco.mjtVisFlag.mjVIS_FLEXEDGE] = True
    option.flags[mujoco.mjtVisFlag.mjVIS_FLEXVERT] = True
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    return option


def _steps(model: mujoco.MjModel, seconds: float) -> int:
    return max(1, int(math.ceil(seconds / model.opt.timestep)))


def _body_id(model: mujoco.MjModel, name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise RuntimeError(f"body not found: {name}")
    return int(body_id)


def _mocap_id(model: mujoco.MjModel, body_name: str) -> int:
    body_id = _body_id(model, body_name)
    mocap_id = int(model.body_mocapid[body_id])
    if mocap_id < 0:
        raise RuntimeError(f"body is not mocap: {body_name}")
    return mocap_id


def collect_shell_point_body_map(model: mujoco.MjModel) -> dict[int, int]:
    point_to_body: dict[int, int] = {}
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not name or not name.startswith(BAG_SHELL_BODY_PREFIX):
            continue
        try:
            point_index = int(name.removeprefix(BAG_SHELL_BODY_PREFIX))
        except ValueError:
            continue
        point_to_body[point_index] = body_id
    return point_to_body


def load_runtime(
    scenario_name: str,
    xml_path: Path | None = None,
    content_case: str = "underfilled",
    selfcollide_mode: str = SELF_COLLISION_MODE,
    noslip_iterations: int = 0,
    multiccd_mode: str = "off",
    nativeccd_mode: str = "off",
    pad_profile: str = "lip",
    pad_condim: int = 4,
    vertcollide_mode: str = "false",
    shell_thickness_scale: float = 1.0,
    close_timestep: float = 0.001,
    gripper_kv: float = 320.0,
    gripper_dampratio: float = 0.20,
) -> tuple[Path, TrialRuntime]:
    if scenario_name not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario_name}")
    if content_case not in available_content_cases():
        raise ValueError(f"unknown content case: {content_case}")
    if selfcollide_mode not in SELF_COLLISION_MODES:
        raise ValueError(f"unknown selfcollide_mode: {selfcollide_mode}")
    if multiccd_mode not in CCD_MODES:
        raise ValueError(f"unknown multiccd_mode: {multiccd_mode}")
    if nativeccd_mode not in NATIVECCD_MODES:
        raise ValueError(f"unknown nativeccd_mode: {nativeccd_mode}")
    if pad_profile not in PAD_PROFILES:
        raise ValueError(f"unknown pad_profile: {pad_profile}")
    if int(pad_condim) not in PAD_CONDIM_OPTIONS:
        raise ValueError(f"unknown pad_condim: {pad_condim}")
    if vertcollide_mode not in VERTCOLLIDE_MODES:
        raise ValueError(f"unknown vertcollide_mode: {vertcollide_mode}")
    if not any(math.isclose(float(close_timestep), option, rel_tol=0.0, abs_tol=1e-12) for option in CLOSE_TIMESTEP_OPTIONS):
        raise ValueError(f"unknown close_timestep: {close_timestep}")
    if xml_path is None:
        xml_path = write_scene_xml(
            scenario_name,
            content_case=content_case,
            selfcollide_mode=selfcollide_mode,
            noslip_iterations=noslip_iterations,
            multiccd_mode=multiccd_mode,
            nativeccd_mode=nativeccd_mode,
            pad_profile=pad_profile,
            pad_condim=pad_condim,
            vertcollide_mode=vertcollide_mode,
            shell_thickness_scale=shell_thickness_scale,
        )
    else:
        write_scene_xml(
            scenario_name,
            output_path=xml_path,
            content_case=content_case,
            selfcollide_mode=selfcollide_mode,
            noslip_iterations=noslip_iterations,
            multiccd_mode=multiccd_mode,
            nativeccd_mode=nativeccd_mode,
            pad_profile=pad_profile,
            pad_condim=pad_condim,
            vertcollide_mode=vertcollide_mode,
            shell_thickness_scale=shell_thickness_scale,
        )

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    _points, labels = make_sack_points(scenario_name)
    runtime = TrialRuntime(
        model=model,
        data=data,
        scenario_name=scenario_name,
        content_case=content_case,
        selfcollide_mode=selfcollide_mode,
        noslip_iterations=int(noslip_iterations),
        multiccd_mode=multiccd_mode,
        nativeccd_mode=nativeccd_mode,
        pad_profile=pad_profile,
        pad_condim=int(pad_condim),
        vertcollide_mode=vertcollide_mode,
        shell_thickness_scale=float(shell_thickness_scale),
        close_timestep=float(close_timestep),
        gripper_kv=float(gripper_kv),
        gripper_dampratio=float(gripper_dampratio),
        labels=labels,
        shell_point_to_body=collect_shell_point_body_map(model),
        left_mocap_id=_mocap_id(model, LEFT_JAW_BODY),
        right_mocap_id=_mocap_id(model, RIGHT_JAW_BODY),
        center_mocap_id=_mocap_id(model, "center_capture_mocap"),
    )
    return xml_path, runtime


def set_gripper(runtime: TrialRuntime, center: np.ndarray, gap: float) -> None:
    # mocap jaw 두 개를 직접 움직여서 파지 후보 위치만 평가한다.
    offset = gap * 0.5 + JAW_PAD_HALF_X
    runtime.data.mocap_pos[runtime.left_mocap_id, :] = center + np.array([-offset, 0.0, 0.0])
    runtime.data.mocap_pos[runtime.right_mocap_id, :] = center + np.array([offset, 0.0, 0.0])
    runtime.data.mocap_pos[runtime.center_mocap_id, :] = center
    runtime.data.mocap_quat[runtime.left_mocap_id, :] = np.array([1.0, 0.0, 0.0, 0.0])
    runtime.data.mocap_quat[runtime.right_mocap_id, :] = np.array([1.0, 0.0, 0.0, 0.0])
    runtime.data.mocap_quat[runtime.center_mocap_id, :] = np.array([1.0, 0.0, 0.0, 0.0])


def shell_positions(runtime: TrialRuntime) -> tuple[list[int], np.ndarray]:
    point_indices = sorted(runtime.shell_point_to_body.keys())
    body_ids = [runtime.shell_point_to_body[index] for index in point_indices]
    return point_indices, np.asarray(runtime.data.xpos[body_ids], dtype=np.float64)


def _capture_point_indices(runtime: TrialRuntime, center: np.ndarray, gap: float) -> list[int]:
    point_indices, positions = shell_positions(runtime)
    local = positions - center[None, :]
    in_y = np.abs(local[:, 1]) <= (JAW_PAD_HALF_Y + CAPTURE_MARGIN)
    in_z = np.abs(local[:, 2]) <= (JAW_PAD_HALF_Z + CAPTURE_MARGIN)
    in_x = np.abs(local[:, 0]) <= (gap * 0.5 + CAPTURE_MARGIN)
    captured = np.logical_and.reduce((in_x, in_y, in_z))
    return [point_index for point_index, flag in zip(point_indices, captured) if bool(flag)]


def _points_to_body_ids(runtime: TrialRuntime, point_indices: Iterable[int]) -> list[int]:
    return [runtime.shell_point_to_body[index] for index in point_indices if index in runtime.shell_point_to_body]


def _positions_for_points(runtime: TrialRuntime, point_indices: Iterable[int]) -> np.ndarray:
    body_ids = _points_to_body_ids(runtime, point_indices)
    if not body_ids:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(runtime.data.xpos[body_ids], dtype=np.float64)


def _top_index(ring_index: int) -> int:
    return 1 + (ring_index % RING_COUNT)


def _middle_index(ring_index: int) -> int:
    return 1 + RING_COUNT + (ring_index % RING_COUNT)


def _lower_index(ring_index: int) -> int:
    return 1 + 2 * RING_COUNT + (ring_index % RING_COUNT)


def _patch_indices_for_ring(ring_index: int) -> list[int]:
    # 한 점이 아니라 top/middle ring의 작은 local bundle을 후보로 삼는다.
    indices: list[int] = []
    for offset in (-1, 0, 1):
        indices.append(_top_index(ring_index + offset))
    for offset in (-1, 0, 1):
        indices.append(_middle_index(ring_index + offset))
    return sorted(set(indices))


def _content_clump_body_ids(runtime: TrialRuntime) -> list[int]:
    body_ids: list[int] = []
    for body_name in CONTENT_CLUMP_NAMES:
        body_id = mujoco.mj_name2id(runtime.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id >= 0:
            body_ids.append(int(body_id))
    return body_ids


def _content_clump_positions(runtime: TrialRuntime) -> np.ndarray:
    body_ids = _content_clump_body_ids(runtime)
    if not body_ids:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray([runtime.data.xpos[body_id].copy() for body_id in body_ids], dtype=np.float64)


def _content_clump_com(runtime: TrialRuntime) -> np.ndarray | None:
    body_ids = _content_clump_body_ids(runtime)
    if not body_ids:
        return None
    masses = np.asarray([runtime.model.body_mass[body_id] for body_id in body_ids], dtype=np.float64)
    positions = np.asarray([runtime.data.xpos[body_id].copy() for body_id in body_ids], dtype=np.float64)
    total = float(np.sum(masses))
    if total <= 1e-9:
        return np.mean(positions, axis=0)
    return (positions * masses[:, None]).sum(axis=0) / total


def _label_counts(labels: list[str], point_indices: Iterable[int]) -> dict[str, int]:
    counts = {"seam": 0, "fold": 0, "plain_top": 0, "other": 0}
    for point_index in point_indices:
        label = labels[point_index] if 0 <= point_index < len(labels) else "other"
        counts[label if label in counts else "other"] += 1
    return counts


def _patch_label(runtime: TrialRuntime, point_indices: list[int]) -> str:
    counts = _label_counts(runtime.labels, point_indices)
    return max(counts.items(), key=lambda item: item[1])[0]


def _patch_center(runtime: TrialRuntime, point_indices: list[int]) -> np.ndarray:
    positions = _positions_for_points(runtime, point_indices)
    if positions.size == 0:
        return np.zeros(3, dtype=np.float64)
    center = np.mean(positions, axis=0)
    center[2] += SEARCH_PROBE_Z
    return center


def _patch_backing_score(runtime: TrialRuntime, ring_index: int, center: np.ndarray) -> float:
    lower_indices = [_lower_index(ring_index + offset) for offset in (-1, 0, 1)]
    lower_positions = _positions_for_points(runtime, lower_indices)
    if lower_positions.size:
        lower_xy = np.mean(lower_positions[:, :2], axis=0)
        lower_distance = float(np.linalg.norm(lower_xy - center[:2]))
        lower_backing = float(np.clip(1.0 - lower_distance / 0.20, 0.0, 1.0))
    else:
        lower_backing = 0.0

    clump_positions = _content_clump_positions(runtime)
    if clump_positions.size:
        distances = np.linalg.norm(clump_positions[:, :2] - center[None, :2], axis=1)
        clump_backing = float(np.clip(1.0 - float(np.min(distances)) / 0.16, 0.0, 1.0))
    else:
        clump_backing = 0.0
    return float(0.55 * lower_backing + 0.45 * clump_backing)


def _bilateral_clearance_score(runtime: TrialRuntime, center: np.ndarray) -> float:
    # 여기서는 주변 물체 회피가 아니라 2F pad가 좌우로 걸칠 공간이 있는지를 보는 대용값이다.
    _point_indices, positions = shell_positions(runtime)
    if positions.size == 0:
        return 0.0
    xy_distances = np.linalg.norm(positions[:, :2] - center[None, :2], axis=1)
    near_count = int(np.count_nonzero(xy_distances < 0.035))
    local_density_score = float(np.clip(near_count / 5.0, 0.0, 1.0))
    floor_clearance = float(np.clip((center[2] - 0.025) / 0.12, 0.0, 1.0))
    return float(0.55 * local_density_score + 0.45 * floor_clearance)


def _fold_root_score(runtime: TrialRuntime, ring_index: int) -> float:
    labels = runtime.labels
    center_label = labels[_top_index(ring_index)] if _top_index(ring_index) < len(labels) else "other"
    neighbor_labels = [
        labels[_top_index(ring_index + offset)] if _top_index(ring_index + offset) < len(labels) else "other"
        for offset in (-2, -1, 1, 2)
    ]
    if center_label == "fold" and any(label != "fold" for label in neighbor_labels):
        return 1.0
    if center_label == "fold":
        return 0.55
    if any(label == "fold" for label in neighbor_labels):
        return 0.35
    return 0.0


def _underfilled_shoulder_score(runtime: TrialRuntime, center: np.ndarray) -> float:
    clump_positions = _content_clump_positions(runtime)
    if clump_positions.shape[0] <= 1:
        return 0.0
    # central bulk보다 좌/우 support clump 위쪽 shoulder를 우선 탐색한다.
    lateral_positions = clump_positions[1:]
    distances = np.linalg.norm(lateral_positions[:, :2] - center[None, :2], axis=1)
    xy_score = float(np.clip(1.0 - float(np.min(distances)) / 0.16, 0.0, 1.0))
    dz = center[2] - float(np.max(lateral_positions[:, 2]))
    above_score = float(np.clip((dz + 0.02) / 0.18, 0.0, 1.0))
    return float(0.65 * xy_score + 0.35 * above_score)


def make_grasp_candidates(
    runtime: TrialRuntime,
    requested_target_label: str = "auto",
    *,
    include_top_center: bool = True,
) -> list[GraspCandidate]:
    if requested_target_label not in {"auto", "seam", "fold", "plain_top"}:
        raise ValueError(f"unknown requested_target_label: {requested_target_label}")

    raw_candidates: list[GraspCandidate] = []
    top_fold = runtime.scenario_name in {"simple_fold", "severe_fold"}
    for ring_index in range(RING_COUNT):
        point_indices = _patch_indices_for_ring(ring_index)
        counts = _label_counts(runtime.labels, point_indices)
        if requested_target_label != "auto" and counts.get(requested_target_label, 0) == 0:
            continue
        center = _patch_center(runtime, point_indices)
        thickness = bundle_thickness_proxy(runtime, point_indices)
        backing = _patch_backing_score(runtime, ring_index, center)
        clearance = _bilateral_clearance_score(runtime, center)
        top_center_penalty = 0.0
        label_score = 0.0
        policy_tag = "ring_patch"

        if runtime.content_case == "underfilled":
            shoulder = _underfilled_shoulder_score(runtime, center)
            label_score += 1.05 * shoulder
            policy_tag = "underfilled_shoulder"

        if top_fold:
            fold_root = _fold_root_score(runtime, ring_index)
            rolled_body = min(counts.get("fold", 0) / 4.0, 1.0)
            adjacent = 0.35 if counts.get("fold", 0) > 0 and counts.get("plain_top", 0) > 0 else 0.0
            label_score += 1.15 * fold_root + 0.55 * rolled_body + adjacent
            policy_tag = "fold_root_or_rolled_body"
        elif runtime.scenario_name == "exposed_seam":
            label_score += 0.45 * min(counts.get("seam", 0) / 3.0, 1.0)

        thickness_score = float(np.clip(thickness / 0.055, 0.0, 1.0))
        score = (
            0.95 * label_score
            + 0.70 * thickness_score
            + 0.60 * backing
            + 0.40 * clearance
            - 0.85 * top_center_penalty
        )
        raw_candidates.append(
            GraspCandidate(
                rank=0,
                root_ring_index=ring_index,
                point_indices=point_indices,
                center=center,
                requested_target_label=requested_target_label,
                policy_tag=policy_tag,
                score=float(score),
                local_thickness_proxy=float(thickness),
                backing_score=float(backing),
                bilateral_clearance=float(clearance),
                top_center_thin_flap_penalty=float(top_center_penalty),
            )
        )

    if include_top_center and requested_target_label in {"auto", "plain_top"}:
        center_indices = [0, *_patch_indices_for_ring(0)]
        center = _patch_center(runtime, center_indices)
        thickness = bundle_thickness_proxy(runtime, center_indices)
        backing = _patch_backing_score(runtime, 0, center)
        clearance = _bilateral_clearance_score(runtime, center)
        penalty = 1.0
        score = 0.30 * float(np.clip(thickness / 0.055, 0.0, 1.0)) + 0.25 * backing + 0.20 * clearance - penalty
        raw_candidates.append(
            GraspCandidate(
                rank=0,
                root_ring_index=0,
                point_indices=sorted(set(center_indices)),
                center=center,
                requested_target_label=requested_target_label,
                policy_tag="top_center_last_resort",
                score=float(score),
                local_thickness_proxy=float(thickness),
                backing_score=float(backing),
                bilateral_clearance=float(clearance),
                top_center_thin_flap_penalty=penalty,
            )
        )

    raw_candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    for rank, candidate in enumerate(raw_candidates):
        candidate.rank = rank
    return raw_candidates


def bilateral_contact_balance(runtime: TrialRuntime, point_indices: list[int], center: np.ndarray, gap: float) -> float:
    positions = _positions_for_points(runtime, point_indices)
    if positions.size == 0:
        return 0.0
    local = positions - center[None, :]
    near_yz = np.logical_and(
        np.abs(local[:, 1]) <= JAW_PAD_HALF_Y + CONTACT_MARGIN,
        np.abs(local[:, 2]) <= JAW_PAD_HALF_Z + CONTACT_MARGIN,
    )
    if not np.any(near_yz):
        return 0.0
    x_values = local[near_yz, 0]
    left_count = int(np.count_nonzero(x_values <= 0.0))
    right_count = int(np.count_nonzero(x_values > 0.0))
    total = max(left_count + right_count, 1)
    return float(min(left_count, right_count) / max(max(left_count, right_count), 1) * min(total / 4.0, 1.0))


def bundle_thickness_proxy(runtime: TrialRuntime, point_indices: list[int]) -> float:
    positions = _positions_for_points(runtime, point_indices)
    if positions.shape[0] <= 1:
        return 0.0
    span_x = float(np.ptp(positions[:, 0]))
    span_z = float(np.ptp(positions[:, 2]))
    return max(span_x, 0.35 * span_z)


def trapped_patch_arc_length(runtime: TrialRuntime, point_indices: list[int]) -> float:
    positions = _positions_for_points(runtime, point_indices)
    if positions.shape[0] <= 1:
        return 0.0
    deltas = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    return float(np.max(distances))


def capture_contact_flags(runtime: TrialRuntime, point_indices: list[int], center: np.ndarray, gap: float) -> tuple[bool, bool]:
    positions = _positions_for_points(runtime, point_indices)
    if positions.size == 0:
        return False, False
    local = positions - center[None, :]
    near_yz = np.logical_and(
        np.abs(local[:, 1]) <= JAW_PAD_HALF_Y + CONTACT_MARGIN,
        np.abs(local[:, 2]) <= JAW_PAD_HALF_Z + CONTACT_MARGIN,
    )
    if not np.any(near_yz):
        return False, False
    # 실제 접촉 API만 쓰면 flex point ID를 안정적으로 역추적하기 어려워서,
    # jaw 양쪽 면에 걸쳐 들어왔는지를 contact proxy로 기록한다.
    contact_band = max(CONTACT_MARGIN, 0.45 * gap)
    left_face_x = -0.5 * gap
    right_face_x = 0.5 * gap
    x_values = local[near_yz, 0]
    left_contact = bool(np.any(np.abs(x_values - left_face_x) <= contact_band) or np.any(x_values < 0.0))
    right_contact = bool(np.any(np.abs(x_values - right_face_x) <= contact_band) or np.any(x_values > 0.0))
    return left_contact, right_contact


def jaw_escape_proxy(runtime: TrialRuntime, point_indices: list[int], center: np.ndarray, gap: float) -> bool:
    if not point_indices:
        return True
    still_inside = set(_capture_point_indices(runtime, center, gap))
    retained = sum(1 for point_index in point_indices if point_index in still_inside)
    minimum_retained = max(1, min(2, math.ceil(len(point_indices) * 0.5)))
    return bool(retained < minimum_retained)


def overlap_ratios(labels: list[str], point_indices: list[int]) -> dict[str, float | str]:
    if not point_indices:
        return {
            "region_label_at_close": "none",
            "seam_overlap_ratio": 0.0,
            "fold_overlap_ratio": 0.0,
            "plain_top_overlap_ratio": 0.0,
        }

    counts = {"seam": 0, "fold": 0, "plain_top": 0, "other": 0}
    for point_index in point_indices:
        label = labels[point_index] if point_index < len(labels) else "other"
        counts[label if label in counts else "other"] += 1

    total = max(len(point_indices), 1)
    dominant = max(counts.items(), key=lambda item: item[1])[0]
    return {
        "region_label_at_close": dominant,
        "seam_overlap_ratio": counts["seam"] / total,
        "fold_overlap_ratio": counts["fold"] / total,
        "plain_top_overlap_ratio": counts["plain_top"] / total,
    }


def compute_load_margin(
    *,
    captured_shell_points: int,
    bundle_thickness: float,
    balance: float,
    contact_persistence_ms: float,
    pull_test_slip_mm: float,
) -> float:
    # 라벨은 이 식에 들어가지 않는다. local geometry와 pull-test 결과만 사용한다.
    captured_score = min(captured_shell_points / 4.0, 1.0)
    thickness_min = min(bundle_thickness / 0.018, 1.0)
    thickness_too_large_penalty = max(0.0, (bundle_thickness - 0.058) / 0.040)
    thickness_score = max(0.0, thickness_min * (1.0 - thickness_too_large_penalty))
    persistence_score = min(contact_persistence_ms / 160.0, 1.0)
    slip_score = max(0.0, 1.0 - pull_test_slip_mm / 48.0)
    load_margin = (
        0.30 * captured_score
        + 0.17 * thickness_score
        + 0.20 * balance
        + 0.18 * persistence_score
        + 0.15 * slip_score
    )
    return float(load_margin)


def _make_hold_patch(runtime: TrialRuntime, point_indices: list[int], center: np.ndarray) -> GraspPatch:
    body_ids = _points_to_body_ids(runtime, point_indices)
    positions = np.asarray(runtime.data.xpos[body_ids], dtype=np.float64)
    return GraspPatch(
        body_ids=body_ids,
        point_indices=list(point_indices),
        relative_positions=positions - center[None, :],
    )


def _apply_hold_surrogate(runtime: TrialRuntime, patch: GraspPatch | None, center: np.ndarray) -> None:
    runtime.data.xfrc_applied[:, :] = 0.0
    if patch is None or not patch.body_ids:
        return

    # pure contact가 수치적으로 불안정할 수 있어, pull-test를 통과한 local patch에만 약한 hold force를 건다.
    kp = float(runtime.gripper_kv)
    kd = float(2.0 * math.sqrt(max(kp, 1e-6)) * runtime.gripper_dampratio)
    max_force = 20.0
    for body_id, relative_position in zip(patch.body_ids, patch.relative_positions):
        target = center + relative_position
        position = np.asarray(runtime.data.xpos[body_id], dtype=np.float64)
        velocity = np.asarray(runtime.data.cvel[body_id, 3:6], dtype=np.float64)
        force = kp * (target - position) - kd * velocity
        norm = float(np.linalg.norm(force))
        if norm > max_force:
            force *= max_force / norm
        runtime.data.xfrc_applied[body_id, :3] += force


def _nonfinite(runtime: TrialRuntime) -> bool:
    return not (np.all(np.isfinite(runtime.data.qpos)) and np.all(np.isfinite(runtime.data.qvel)))


def _state_snapshot(runtime: TrialRuntime) -> dict[str, np.ndarray]:
    return {
        "qpos": runtime.data.qpos.copy(),
        "qvel": runtime.data.qvel.copy(),
        "mocap_pos": runtime.data.mocap_pos.copy(),
        "mocap_quat": runtime.data.mocap_quat.copy(),
    }


def _restore_state_snapshot(runtime: TrialRuntime, snapshot: dict[str, np.ndarray]) -> None:
    runtime.data.qpos[:] = snapshot["qpos"]
    runtime.data.qvel[:] = snapshot["qvel"]
    runtime.data.mocap_pos[:] = snapshot["mocap_pos"]
    runtime.data.mocap_quat[:] = snapshot["mocap_quat"]
    runtime.data.xfrc_applied[:, :] = 0.0
    mujoco.mj_forward(runtime.model, runtime.data)


def _max_shell_speed(runtime: TrialRuntime) -> float:
    body_ids = list(runtime.shell_point_to_body.values())
    if not body_ids:
        return 0.0
    speeds = np.linalg.norm(runtime.data.cvel[body_ids, 3:6], axis=1)
    return float(np.max(speeds))


def _min_contact_distance(runtime: TrialRuntime) -> float:
    if runtime.data.ncon <= 0:
        return 0.0
    distances = [float(runtime.data.contact[index].dist) for index in range(runtime.data.ncon)]
    return min(distances) if distances else 0.0


def _nan_risk_or_spike(runtime: TrialRuntime) -> tuple[bool, dict[str, float | bool]]:
    shell_speed = _max_shell_speed(runtime)
    qvel_norm = float(np.linalg.norm(runtime.data.qvel))
    min_contact_dist = _min_contact_distance(runtime)
    nonfinite = _nonfinite(runtime)
    unstable = bool(
        nonfinite
        or shell_speed > ROLLBACK_MAX_SHELL_SPEED_MPS
        or qvel_norm > ROLLBACK_MAX_QVEL_NORM
        or min_contact_dist < ROLLBACK_MIN_CONTACT_DIST_M
    )
    return unstable, {
        "nonfinite": nonfinite,
        "max_shell_speed_mps": shell_speed,
        "qvel_norm": qvel_norm,
        "min_contact_distance_m": min_contact_dist,
    }


class TrialRenderer:
    def __init__(self, model: mujoco.MjModel, output_dir: Path, enabled: bool, width: int = 1280, height: int = 720):
        self.output_dir = output_dir
        self.enabled = enabled
        self.frames: list[np.ndarray] = []
        self.renderer: mujoco.Renderer | None = None
        self.scene_option: mujoco.MjvOption | None = None
        if not enabled:
            return
        try:
            self.renderer = mujoco.Renderer(model, height=height, width=width)
            self.scene_option = make_render_option()
        except Exception as exc:
            print(f"render_disabled={exc}")
            self.enabled = False

    def capture(self, data: mujoco.MjData, name: str | None = None, keep_frame: bool = True) -> None:
        if not self.enabled or self.renderer is None or self.scene_option is None:
            return
        self.renderer.update_scene(data, camera="overview", scene_option=self.scene_option)
        image = np.asarray(self.renderer.render(), dtype=np.uint8)
        if name is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(self.output_dir / f"{name}.png", image)
        if keep_frame:
            self.frames.append(image.copy())

    def close(self, save_mp4: bool = True) -> None:
        if not self.enabled:
            return
        frames_dir = self.output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        writer = None
        mp4_path = self.output_dir / "sequence.mp4"
        if save_mp4:
            try:
                writer = imageio.get_writer(mp4_path, fps=24)
            except Exception as exc:
                print(f"mp4_disabled={exc}")
                writer = None
        for index, image in enumerate(self.frames):
            imageio.imwrite(frames_dir / f"frame_{index:03d}.png", image)
            if writer is not None:
                writer.append_data(image)
        if writer is not None:
            writer.close()
        if self.renderer is not None:
            self.renderer.close()


def _advance_stage(
    runtime: TrialRuntime,
    renderer: TrialRenderer,
    *,
    steps: int,
    center_start: np.ndarray,
    center_end: np.ndarray,
    gap_start: float,
    gap_end: float,
    hold_patch: GraspPatch | None = None,
    track_persistence: bool = False,
) -> float:
    persistence_ms = 0.0
    frame_stride = max(1, steps // max(1, FRAME_TARGET_COUNT // 7))
    for step_index in range(steps):
        alpha = (step_index + 1) / steps
        center = (1.0 - alpha) * center_start + alpha * center_end
        gap = (1.0 - alpha) * gap_start + alpha * gap_end
        set_gripper(runtime, center, gap)
        _apply_hold_surrogate(runtime, hold_patch, center)

        if track_persistence:
            captured = _capture_point_indices(runtime, center, gap)
            balance = bilateral_contact_balance(runtime, captured, center, gap)
            if len(captured) >= 2 and balance > 0.15:
                persistence_ms += runtime.model.opt.timestep * 1000.0

        mujoco.mj_step(runtime.model, runtime.data)
        if step_index % frame_stride == 0:
            renderer.capture(runtime.data, keep_frame=True)
    return float(persistence_ms)


def _advance_close_with_rollback(
    runtime: TrialRuntime,
    renderer: TrialRenderer,
    *,
    steps: int,
    center_start: np.ndarray,
    center_end: np.ndarray,
    gap_start: float,
    gap_end: float,
) -> tuple[float, bool, float, dict[str, float | bool]]:
    persistence_ms = 0.0
    frame_stride = max(1, steps // max(1, FRAME_TARGET_COUNT // 7))
    last_stable_snapshot = _state_snapshot(runtime)
    last_stable_gap = float(gap_start)
    last_diagnostics: dict[str, float | bool] = {
        "nonfinite": False,
        "max_shell_speed_mps": 0.0,
        "qvel_norm": 0.0,
        "min_contact_distance_m": 0.0,
    }

    for step_index in range(steps):
        alpha = (step_index + 1) / steps
        center = (1.0 - alpha) * center_start + alpha * center_end
        gap = float((1.0 - alpha) * gap_start + alpha * gap_end)
        set_gripper(runtime, center, gap)

        captured = _capture_point_indices(runtime, center, gap)
        balance = bilateral_contact_balance(runtime, captured, center, gap)
        if len(captured) >= 2 and balance > 0.15:
            persistence_ms += runtime.model.opt.timestep * 1000.0

        mujoco.mj_step(runtime.model, runtime.data)
        unstable, diagnostics = _nan_risk_or_spike(runtime)
        last_diagnostics = diagnostics
        if unstable:
            _restore_state_snapshot(runtime, last_stable_snapshot)
            renderer.capture(runtime.data, name="rollback", keep_frame=True)
            return float(persistence_ms), True, float(last_stable_gap), diagnostics

        last_stable_snapshot = _state_snapshot(runtime)
        last_stable_gap = gap
        if step_index % frame_stride == 0:
            renderer.capture(runtime.data, keep_frame=True)

    renderer.capture(runtime.data, name="rollback", keep_frame=False)
    return float(persistence_ms), False, float(last_stable_gap), last_diagnostics


def _target_for_trial(scenario_name: str, trial_index: int) -> np.ndarray:
    base = nominal_grasp_center(scenario_name)
    offsets = TRIAL_OFFSETS.get(scenario_name, ((0.0, 0.0, 0.0),))
    dx, dy, dz = offsets[trial_index % len(offsets)]
    return base + np.array([dx, dy, dz], dtype=np.float64)


def _refine_target_z_from_settled_shell(runtime: TrialRuntime, planned_center: np.ndarray) -> np.ndarray:
    # 접근 ROI의 x/y는 유지하고, settle 뒤 실제 상단 shell 높이에 맞춰 z만 보정한다.
    candidate_indices = list(range(0, 1 + RING_COUNT))
    positions = _positions_for_points(runtime, candidate_indices)
    if positions.size == 0:
        return planned_center

    distances_xy = np.linalg.norm(positions[:, :2] - planned_center[None, :2], axis=1)
    nearest_count = min(5, positions.shape[0])
    nearest = positions[np.argsort(distances_xy)[:nearest_count]]
    refined = planned_center.copy()
    refined[2] = float(np.percentile(nearest[:, 2], 68))
    return refined


def _selector_label_for_request(scenario_name: str, requested_target_label: str) -> str:
    if requested_target_label == "auto":
        return "seam" if scenario_name == "exposed_seam" else "fold"
    return requested_target_label


def _label_path_token(requested_target_label: str) -> str:
    return {
        "auto": "auto",
        "seam": "seam",
        "fold": "fold",
        "plain_top": "plain",
    }.get(requested_target_label, requested_target_label)


def _target_for_requested_label_from_settled_shell(
    runtime: TrialRuntime,
    planned_center: np.ndarray,
    requested_target_label: str,
) -> tuple[np.ndarray, str, int]:
    # target label은 후보를 고르는 데만 쓴다. 성공/실패 판정에는 절대 쓰지 않는다.
    selector_label = _selector_label_for_request(runtime.scenario_name, requested_target_label)
    candidate_indices = list(range(0, 1 + RING_COUNT))
    label_candidates = [
        point_index
        for point_index in candidate_indices
        if point_index < len(runtime.labels) and runtime.labels[point_index] == selector_label
    ]
    if not label_candidates:
        label_candidates = candidate_indices

    positions = _positions_for_points(runtime, label_candidates)
    if positions.size == 0:
        return _refine_target_z_from_settled_shell(runtime, planned_center), selector_label, -1

    distances_xy = np.linalg.norm(positions[:, :2] - planned_center[None, :2], axis=1)
    # 높은 후보를 우선하되, planned ROI에서 너무 벗어난 후보는 피한다.
    scores = positions[:, 2] - 0.35 * distances_xy
    local_index = int(np.argmax(scores))
    point_index = int(label_candidates[local_index])
    target = positions[local_index].copy()
    return target, selector_label, point_index


def _mean_position_for_points(runtime: TrialRuntime, point_indices: list[int]) -> np.ndarray | None:
    positions = _positions_for_points(runtime, point_indices)
    if positions.size == 0:
        return None
    return np.mean(positions, axis=0)


def _support_reference_z(runtime: TrialRuntime, candidate: GraspCandidate) -> float:
    lower_indices = [_lower_index(candidate.root_ring_index + offset) for offset in (-1, 0, 1)]
    lower_positions = _positions_for_points(runtime, lower_indices)
    values: list[float] = []
    if lower_positions.size:
        values.append(float(np.mean(lower_positions[:, 2])))
    clump_com = _content_clump_com(runtime)
    if clump_com is not None:
        values.append(float(clump_com[2]))
    if not values:
        return float(candidate.center[2])
    return float(np.mean(values))


def _candidate_result_stub(
    *,
    mode: str,
    runtime: TrialRuntime,
    requested_target_label: str,
    output_dir: Path,
    xml_path: Path,
    trial_index: int,
    no_graspable_patch_found: bool,
) -> dict[str, float | int | bool | str]:
    return {
        "mode": mode,
        "scenario_name": runtime.scenario_name,
        "content_case": runtime.content_case,
        "selfcollide_mode": runtime.selfcollide_mode,
        "noslip_iterations": int(runtime.noslip_iterations),
        "multiccd_mode": runtime.multiccd_mode,
        "nativeccd_mode": runtime.nativeccd_mode,
        "pad_profile": runtime.pad_profile,
        "pad_condim": int(runtime.pad_condim),
        "vertcollide_mode": runtime.vertcollide_mode,
        "shell_thickness_scale": float(runtime.shell_thickness_scale),
        "close_timestep": float(runtime.close_timestep),
        "gripper_kv": float(runtime.gripper_kv),
        "gripper_dampratio": float(runtime.gripper_dampratio),
        "trial_index": int(trial_index),
        "requested_target_label": requested_target_label,
        "actual_region_label_at_close": "none",
        "accepted_candidate_rank": -1,
        "seam_overlap_ratio": 0.0,
        "fold_overlap_ratio": 0.0,
        "plain_top_overlap_ratio": 0.0,
        "left_contact_present": False,
        "right_contact_present": False,
        "trapped_shell_points": 0,
        "trapped_patch_arc_length": 0.0,
        "captured_shell_points": 0,
        "bundle_thickness_proxy": 0.0,
        "bilateral_contact_balance": 0.0,
        "tangential_slip_proxy": 999.0,
        "jaw_escape": True,
        "rollback_used": False,
        "contact_persistence_ms": 0.0,
        "pull_test_slip_mm": 999.0,
        "micro_lift_survival": False,
        "load_following_ratio": 0.0,
        "latch_activated": False,
        "latch_activation_time": -1.0,
        "hold_surrogate_activated": False,
        "lift_height": 0.0,
        "lift_height_contact_only": 0.0,
        "hold_time": 0.0,
        "hold_time_contact_only": 0.0,
        "final_slip_distance": 999.0,
        "drop_or_not": True,
        "no_graspable_patch_found": bool(no_graspable_patch_found),
        "pass_fail": False,
        "rectification_used": "none",
        "candidate_policy_tag": "none",
        "candidate_score": 0.0,
        "nonfinite": bool(_nonfinite(runtime)),
        "xml": str(xml_path),
        "output_dir": str(output_dir),
    }


def _probe_candidate(
    runtime: TrialRuntime,
    renderer: TrialRenderer,
    candidate: GraspCandidate,
    *,
    mode: str,
    close_seconds: float = SEARCH_CLOSE_SECONDS,
    precompression_dwell_seconds: float = PRECOMPRESSION_DWELL_SECONDS,
    close_timestep: float | None = None,
) -> ProbeOutcome:
    latch_allowed = mode in LATCH_MODES
    high_center = candidate.center + np.array([0.0, 0.0, SEARCH_APPROACH_Z], dtype=np.float64)
    close_center = candidate.center.copy()
    tug_center = close_center + np.array([0.0, 0.0, SEARCH_TUG_LIFT_M], dtype=np.float64)
    micro_center = close_center + np.array([0.0, 0.0, SEARCH_MICRO_LIFT_M], dtype=np.float64)
    final_center = close_center + np.array([0.0, 0.0, FINAL_SEARCH_LIFT_M], dtype=np.float64)

    set_gripper(runtime, high_center, JAW_OPEN_GAP)
    mujoco.mj_forward(runtime.model, runtime.data)

    _advance_stage(
        runtime,
        renderer,
        steps=_steps(runtime.model, SEARCH_APPROACH_SECONDS),
        center_start=high_center,
        center_end=close_center,
        gap_start=JAW_OPEN_GAP,
        gap_end=JAW_OPEN_GAP,
        hold_patch=None,
    )
    original_timestep = float(runtime.model.opt.timestep)
    requested_close_timestep = float(close_timestep if close_timestep is not None else runtime.close_timestep)
    runtime.model.opt.timestep = requested_close_timestep
    close_persistence, rollback_used, actual_close_gap, rollback_diagnostics = _advance_close_with_rollback(
        runtime,
        renderer,
        steps=_steps(runtime.model, max(close_seconds, runtime.model.opt.timestep)),
        center_start=close_center,
        center_end=close_center,
        gap_start=JAW_OPEN_GAP,
        gap_end=JAW_CLOSED_GAP,
    )
    if precompression_dwell_seconds > 0.0 and not rollback_used:
        close_persistence += _advance_stage(
            runtime,
            renderer,
            steps=_steps(runtime.model, precompression_dwell_seconds),
            center_start=close_center,
            center_end=close_center,
            gap_start=actual_close_gap,
            gap_end=actual_close_gap,
            hold_patch=None,
            track_persistence=True,
        )
    runtime.model.opt.timestep = original_timestep
    captured_at_close = _capture_point_indices(runtime, close_center, actual_close_gap)
    captured_shell_points = len(captured_at_close)
    close_mean = _mean_position_for_points(runtime, captured_at_close)
    close_rel = close_mean - close_center if close_mean is not None else np.zeros(3, dtype=np.float64)
    close_z = float(close_mean[2]) if close_mean is not None else 0.0
    renderer.capture(runtime.data, name="close", keep_frame=True)

    support_start_z = _support_reference_z(runtime, candidate)
    tug_persistence = _advance_stage(
        runtime,
        renderer,
        steps=_steps(runtime.model, SEARCH_TUG_SECONDS),
        center_start=close_center,
        center_end=tug_center,
        gap_start=actual_close_gap,
        gap_end=actual_close_gap,
        hold_patch=None,
        track_persistence=True,
    )
    tug_mean = _mean_position_for_points(runtime, captured_at_close)
    if tug_mean is not None:
        tug_rel = tug_mean - tug_center
        pull_test_slip_mm = float(np.linalg.norm(tug_rel - close_rel) * 1000.0)
    else:
        pull_test_slip_mm = 999.0
    renderer.capture(runtime.data, name="tug_test", keep_frame=True)

    micro_persistence = _advance_stage(
        runtime,
        renderer,
        steps=_steps(runtime.model, SEARCH_LIFT_SECONDS),
        center_start=tug_center,
        center_end=micro_center,
        gap_start=actual_close_gap,
        gap_end=actual_close_gap,
        hold_patch=None,
        track_persistence=True,
    )
    micro_mean = _mean_position_for_points(runtime, captured_at_close)
    support_after_micro_z = _support_reference_z(runtime, candidate)
    gripper_dz = max(float(micro_center[2] - close_center[2]), 1e-6)
    load_following_ratio = float(np.clip((support_after_micro_z - support_start_z) / gripper_dz, -2.0, 2.0))
    renderer.capture(runtime.data, name="micro_lift", keep_frame=True)

    thickness = bundle_thickness_proxy(runtime, captured_at_close)
    balance = bilateral_contact_balance(runtime, captured_at_close, close_center, actual_close_gap)
    left_contact_present, right_contact_present = capture_contact_flags(runtime, captured_at_close, close_center, actual_close_gap)
    trapped_shell_points = captured_shell_points
    trapped_arc_length = trapped_patch_arc_length(runtime, captured_at_close)
    tangential_slip_proxy = float((pull_test_slip_mm / 1000.0) / max(SEARCH_TUG_SECONDS, runtime.model.opt.timestep))
    jaw_escape = jaw_escape_proxy(runtime, captured_at_close, micro_center, actual_close_gap)
    contact_persistence_ms = float(close_persistence + tug_persistence + micro_persistence)
    close_ok = bool(
        left_contact_present
        and right_contact_present
        and trapped_shell_points >= 2
        and CAPTURE_MIN_THICKNESS_M <= thickness <= CAPTURE_MAX_THICKNESS_M
        and balance >= 0.08
    )
    micro_lift_survival = bool(
        micro_mean is not None
        and float(micro_mean[2] - close_z) >= CAPTURE_MIN_MICRO_LIFT_M
        and pull_test_slip_mm <= 150.0
        and contact_persistence_ms >= 45.0
    )
    qualification_ok = bool(
        close_ok
        and micro_lift_survival
        and pull_test_slip_mm <= 130.0
        and tangential_slip_proxy <= CAPTURE_MAX_TANGENTIAL_SLIP_MPS
        and load_following_ratio >= CAPTURE_MIN_LOAD_FOLLOWING_RATIO
        and not jaw_escape
    )

    hold_patch = _make_hold_patch(runtime, captured_at_close, micro_center) if latch_allowed and qualification_ok else None
    latch_activated = bool(hold_patch is not None and hold_patch.body_ids)
    latch_activation_time = float(runtime.data.time) if latch_activated else -1.0
    renderer.capture(runtime.data, name="latch_on", keep_frame=True)
    activation_rel = (
        np.mean(hold_patch.relative_positions, axis=0)
        if hold_patch is not None and len(hold_patch.relative_positions)
        else close_rel
    )
    _advance_stage(
        runtime,
        renderer,
        steps=_steps(runtime.model, SEARCH_LIFT_SECONDS),
        center_start=micro_center,
        center_end=final_center,
        gap_start=actual_close_gap,
        gap_end=actual_close_gap,
        hold_patch=hold_patch,
        track_persistence=False,
    )
    _advance_stage(
        runtime,
        renderer,
        steps=_steps(runtime.model, FINAL_SEARCH_HOLD_SECONDS),
        center_start=final_center,
        center_end=final_center,
        gap_start=actual_close_gap,
        gap_end=actual_close_gap,
        hold_patch=hold_patch,
        track_persistence=False,
    )
    renderer.capture(runtime.data, name="final_lift", keep_frame=True)
    renderer.capture(runtime.data, name="lift", keep_frame=False)

    final_mean = _mean_position_for_points(runtime, captured_at_close)
    if final_mean is not None:
        final_rel = final_mean - final_center
        final_slip_distance = float(np.linalg.norm(final_rel - activation_rel))
        lift_height = float(final_mean[2] - close_z)
    else:
        final_slip_distance = 999.0
        lift_height = 0.0
    drop_or_not = bool(
        final_mean is None
        or lift_height < 0.010
        or final_slip_distance > 0.120
        or _nonfinite(runtime)
    )
    pass_fail = bool(
        qualification_ok
        and lift_height >= 0.030
        and FINAL_SEARCH_HOLD_SECONDS >= 0.50
        and final_slip_distance <= 0.095
        and not drop_or_not
    )
    ratios = overlap_ratios(runtime.labels, captured_at_close)
    return ProbeOutcome(
        candidate=candidate,
        captured_at_close=list(captured_at_close),
        requested_target_label=candidate.requested_target_label,
        actual_region_label_at_close=str(ratios["region_label_at_close"]),
        left_contact_present=bool(left_contact_present),
        right_contact_present=bool(right_contact_present),
        trapped_shell_points=int(trapped_shell_points),
        trapped_patch_arc_length=float(trapped_arc_length),
        captured_shell_points=int(captured_shell_points),
        bundle_thickness_proxy=float(thickness),
        bilateral_contact_balance=float(balance),
        tangential_slip_proxy=float(tangential_slip_proxy),
        jaw_escape=bool(jaw_escape),
        rollback_used=bool(rollback_used),
        contact_persistence_ms=float(contact_persistence_ms),
        pull_test_slip_mm=float(pull_test_slip_mm),
        micro_lift_survival=bool(micro_lift_survival),
        load_following_ratio=float(load_following_ratio),
        lift_height=float(lift_height),
        hold_time=float(FINAL_SEARCH_HOLD_SECONDS),
        final_slip_distance=float(final_slip_distance),
        drop_or_not=bool(drop_or_not),
        pass_fail=bool(pass_fail),
        latch_activated=bool(latch_activated),
        latch_activation_time=float(latch_activation_time),
    )


def _outcome_to_result(
    *,
    mode: str,
    runtime: TrialRuntime,
    outcome: ProbeOutcome,
    output_dir: Path,
    xml_path: Path,
    trial_index: int,
) -> dict[str, float | int | bool | str]:
    return {
        "mode": mode,
        "scenario_name": runtime.scenario_name,
        "content_case": runtime.content_case,
        "selfcollide_mode": runtime.selfcollide_mode,
        "noslip_iterations": int(runtime.noslip_iterations),
        "multiccd_mode": runtime.multiccd_mode,
        "nativeccd_mode": runtime.nativeccd_mode,
        "pad_profile": runtime.pad_profile,
        "pad_condim": int(runtime.pad_condim),
        "vertcollide_mode": runtime.vertcollide_mode,
        "shell_thickness_scale": float(runtime.shell_thickness_scale),
        "close_timestep": float(runtime.close_timestep),
        "gripper_kv": float(runtime.gripper_kv),
        "gripper_dampratio": float(runtime.gripper_dampratio),
        "trial_index": int(trial_index),
        "requested_target_label": outcome.requested_target_label,
        "actual_region_label_at_close": outcome.actual_region_label_at_close,
        "accepted_candidate_rank": int(outcome.candidate.rank),
        "seam_overlap_ratio": float(overlap_ratios(runtime.labels, outcome.captured_at_close)["seam_overlap_ratio"]),
        "fold_overlap_ratio": float(overlap_ratios(runtime.labels, outcome.captured_at_close)["fold_overlap_ratio"]),
        "plain_top_overlap_ratio": float(overlap_ratios(runtime.labels, outcome.captured_at_close)["plain_top_overlap_ratio"]),
        "left_contact_present": bool(outcome.left_contact_present),
        "right_contact_present": bool(outcome.right_contact_present),
        "trapped_shell_points": int(outcome.trapped_shell_points),
        "trapped_patch_arc_length": float(outcome.trapped_patch_arc_length),
        "captured_shell_points": int(outcome.captured_shell_points),
        "bundle_thickness_proxy": float(outcome.bundle_thickness_proxy),
        "bilateral_contact_balance": float(outcome.bilateral_contact_balance),
        "tangential_slip_proxy": float(outcome.tangential_slip_proxy),
        "jaw_escape": bool(outcome.jaw_escape),
        "rollback_used": bool(outcome.rollback_used),
        "contact_persistence_ms": float(outcome.contact_persistence_ms),
        "pull_test_slip_mm": float(outcome.pull_test_slip_mm),
        "micro_lift_survival": bool(outcome.micro_lift_survival),
        "load_following_ratio": float(outcome.load_following_ratio),
        "latch_activated": bool(outcome.latch_activated),
        "latch_activation_time": float(outcome.latch_activation_time),
        "hold_surrogate_activated": bool(outcome.latch_activated),
        "lift_height": float(outcome.lift_height),
        "lift_height_contact_only": float(outcome.lift_height),
        "hold_time": float(outcome.hold_time),
        "hold_time_contact_only": float(outcome.hold_time),
        "final_slip_distance": float(outcome.final_slip_distance),
        "drop_or_not": bool(outcome.drop_or_not),
        "no_graspable_patch_found": False,
        "pass_fail": bool(outcome.pass_fail),
        "rectification_used": outcome.candidate.rectification_used,
        "candidate_policy_tag": outcome.candidate.policy_tag,
        "candidate_score": float(outcome.candidate.score),
        "nonfinite": bool(_nonfinite(runtime)),
        "xml": str(xml_path),
        "output_dir": str(output_dir),
    }


def _rectified_candidate(candidate: GraspCandidate, primitive_name: str, side_sign: float = 1.0) -> GraspCandidate:
    shifted = candidate.center.copy()
    shifted[1] += side_sign * RECTIFICATION_SHIFT_M
    return GraspCandidate(
        rank=candidate.rank,
        root_ring_index=candidate.root_ring_index,
        point_indices=list(candidate.point_indices),
        center=shifted,
        requested_target_label=candidate.requested_target_label,
        policy_tag=f"{candidate.policy_tag}+{primitive_name}",
        score=float(candidate.score - 0.05),
        local_thickness_proxy=candidate.local_thickness_proxy,
        backing_score=candidate.backing_score,
        bilateral_clearance=candidate.bilateral_clearance,
        top_center_thin_flap_penalty=candidate.top_center_thin_flap_penalty,
        rectification_used=primitive_name,
    )


def run_auto_search_and_probe_trial(
    scenario_name: str,
    *,
    mode: str = "contact_only_eval",
    content_case: str = "underfilled",
    requested_target_label: str = "auto",
    trial_index: int = 0,
    output_dir: Path | None = None,
    render: bool = True,
    save_mp4: bool = True,
    width: int = 1280,
    height: int = 720,
    selfcollide_mode: str = SELF_COLLISION_MODE,
    noslip_iterations: int = 0,
    multiccd_mode: str = "off",
    nativeccd_mode: str = "off",
    pad_profile: str = "lip",
    pad_condim: int = 4,
    vertcollide_mode: str = "false",
    shell_thickness_scale: float = 1.0,
    close_seconds: float = SEARCH_CLOSE_SECONDS,
    precompression_dwell_seconds: float = PRECOMPRESSION_DWELL_SECONDS,
    close_timestep: float = 0.001,
    gripper_kv: float = 320.0,
    gripper_dampratio: float = 0.20,
) -> dict[str, float | int | bool | str]:
    if mode not in AUTO_SEARCH_MODES:
        raise ValueError(f"unknown mode: {mode}")
    if scenario_name not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario_name}")
    if requested_target_label not in {"auto", "seam", "fold", "plain_top"}:
        raise ValueError(f"unknown requested_target_label: {requested_target_label}")
    if output_dir is None:
        label_token = _label_path_token(requested_target_label)
        output_dir = (
            OUTPUT_DIR
            / "as"
            / MODE_PATH_TOKEN[mode]
            / (
                f"{content_case[:3]}_{selfcollide_mode}_n{int(noslip_iterations)}_"
                f"ccd{1 if multiccd_mode == 'on' else 0}_{pad_profile[0]}c{int(pad_condim)}_"
                f"t{int(round(shell_thickness_scale * 100))}_dt{int(round(float(close_timestep) * 1_000_000))}"
            )
            / f"{scenario_name}_{label_token}_t{trial_index:02d}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    xml_path = output_dir / f"scene_{MODE_PATH_TOKEN[mode]}_t{trial_index:02d}.xml"
    _xml_path, runtime = load_runtime(
        scenario_name,
        xml_path=xml_path,
        content_case=content_case,
        selfcollide_mode=selfcollide_mode,
        noslip_iterations=noslip_iterations,
        multiccd_mode=multiccd_mode,
        nativeccd_mode=nativeccd_mode,
        pad_profile=pad_profile,
        pad_condim=pad_condim,
        vertcollide_mode=vertcollide_mode,
        shell_thickness_scale=shell_thickness_scale,
        close_timestep=close_timestep,
        gripper_kv=gripper_kv,
        gripper_dampratio=gripper_dampratio,
    )
    renderer = TrialRenderer(runtime.model, output_dir, enabled=render, width=width, height=height)

    planned_target_center = _target_for_trial(scenario_name, trial_index)
    high_center = planned_target_center + np.array([0.0, 0.0, 0.145], dtype=np.float64)
    set_gripper(runtime, high_center, JAW_OPEN_GAP)
    mujoco.mj_forward(runtime.model, runtime.data)
    renderer.capture(runtime.data, name="initial", keep_frame=True)
    _advance_stage(
        runtime,
        renderer,
        steps=_steps(runtime.model, SETTLE_SECONDS),
        center_start=high_center,
        center_end=high_center,
        gap_start=JAW_OPEN_GAP,
        gap_end=JAW_OPEN_GAP,
        hold_patch=None,
    )
    candidates = make_grasp_candidates(runtime, requested_target_label)
    if candidates:
        set_gripper(runtime, candidates[0].center + np.array([0.0, 0.0, SEARCH_APPROACH_Z], dtype=np.float64), JAW_OPEN_GAP)
        mujoco.mj_forward(runtime.model, runtime.data)
    renderer.capture(runtime.data, name="candidate_overlay", keep_frame=True)
    settled_qpos = runtime.data.qpos.copy()
    settled_qvel = runtime.data.qvel.copy()
    settled_mocap_pos = runtime.data.mocap_pos.copy()
    settled_mocap_quat = runtime.data.mocap_quat.copy()

    best_failure: ProbeOutcome | None = None
    for candidate in candidates[:MAX_CANDIDATES_TO_PROBE]:
        runtime.data.qpos[:] = settled_qpos
        runtime.data.qvel[:] = settled_qvel
        runtime.data.mocap_pos[:] = settled_mocap_pos
        runtime.data.mocap_quat[:] = settled_mocap_quat
        runtime.data.xfrc_applied[:, :] = 0.0
        mujoco.mj_forward(runtime.model, runtime.data)
        outcome = _probe_candidate(
            runtime,
            renderer,
            candidate,
            mode=mode,
            close_seconds=close_seconds,
            precompression_dwell_seconds=precompression_dwell_seconds,
            close_timestep=close_timestep,
        )
        if best_failure is None or outcome.final_slip_distance < best_failure.final_slip_distance:
            best_failure = outcome
        if outcome.pass_fail:
            renderer.close(save_mp4=save_mp4)
            return _outcome_to_result(
                mode=mode,
                runtime=runtime,
                outcome=outcome,
                output_dir=output_dir,
                xml_path=xml_path,
                trial_index=trial_index,
            )

        top_fold = scenario_name in {"simple_fold", "severe_fold"}
        if top_fold and candidate.rank < 2 and outcome.candidate.rectification_used == "none":
            # fold를 완전히 펴지 않고, 실패한 경우에만 작은 sideways brush / realign을 시도한다.
            primitive = "brush_fold_sideways" if outcome.actual_region_label_at_close == "fold" else "reopen_and_realign"
            rectified = _rectified_candidate(candidate, primitive_name=primitive, side_sign=1.0 if candidate.center[1] <= 0 else -1.0)
            runtime.data.qpos[:] = settled_qpos
            runtime.data.qvel[:] = settled_qvel
            runtime.data.mocap_pos[:] = settled_mocap_pos
            runtime.data.mocap_quat[:] = settled_mocap_quat
            runtime.data.xfrc_applied[:, :] = 0.0
            mujoco.mj_forward(runtime.model, runtime.data)
            outcome = _probe_candidate(
                runtime,
                renderer,
                rectified,
                mode=mode,
                close_seconds=close_seconds,
                precompression_dwell_seconds=precompression_dwell_seconds,
                close_timestep=close_timestep,
            )
            if best_failure is None or outcome.final_slip_distance < best_failure.final_slip_distance:
                best_failure = outcome
            if outcome.pass_fail:
                renderer.close(save_mp4=save_mp4)
                return _outcome_to_result(
                    mode=mode,
                    runtime=runtime,
                    outcome=outcome,
                    output_dir=output_dir,
                    xml_path=xml_path,
                    trial_index=trial_index,
                )

    renderer.close(save_mp4=save_mp4)
    if best_failure is None:
        return _candidate_result_stub(
            mode=mode,
            runtime=runtime,
            requested_target_label=requested_target_label,
            output_dir=output_dir,
            xml_path=xml_path,
            trial_index=trial_index,
            no_graspable_patch_found=True,
        )
    result = _outcome_to_result(
        mode=mode,
        runtime=runtime,
        outcome=best_failure,
        output_dir=output_dir,
        xml_path=xml_path,
        trial_index=trial_index,
    )
    accepted_candidate_found = bool(
        best_failure.latch_activated
        or (
            best_failure.micro_lift_survival
            and best_failure.left_contact_present
            and best_failure.right_contact_present
            and not best_failure.jaw_escape
        )
    )
    result["no_graspable_patch_found"] = not accepted_candidate_found
    result["pass_fail"] = False
    return result


def run_contact_only_trial(
    scenario_name: str,
    *,
    content_case: str = "underfilled",
    requested_target_label: str = "auto",
    trial_index: int = 0,
    output_dir: Path | None = None,
    render: bool = True,
    save_mp4: bool = True,
    width: int = 1280,
    height: int = 720,
) -> dict[str, float | int | bool | str]:
    if scenario_name not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario_name}")
    if content_case not in available_content_cases():
        raise ValueError(f"unknown content case: {content_case}")
    if requested_target_label not in {"auto", "seam", "fold", "plain_top"}:
        raise ValueError(f"unknown requested_target_label: {requested_target_label}")
    label_token = _label_path_token(requested_target_label)
    if output_dir is None:
        output_dir = OUTPUT_DIR / "contact_only_eval" / content_case / scenario_name / label_token / f"t{trial_index:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    xml_path = output_dir / f"scene_{label_token}_t{trial_index:02d}.xml"
    _xml_path, runtime = load_runtime(scenario_name, xml_path=xml_path, content_case=content_case)
    renderer = TrialRenderer(runtime.model, output_dir, enabled=render, width=width, height=height)

    planned_target_center = _target_for_trial(scenario_name, trial_index)
    high_center = planned_target_center + np.array([0.0, 0.0, 0.145], dtype=np.float64)

    set_gripper(runtime, high_center, JAW_OPEN_GAP)
    mujoco.mj_forward(runtime.model, runtime.data)
    renderer.capture(runtime.data, name="initial", keep_frame=True)

    _advance_stage(
        runtime,
        renderer,
        steps=_steps(runtime.model, SETTLE_SECONDS),
        center_start=high_center,
        center_end=high_center,
        gap_start=JAW_OPEN_GAP,
        gap_end=JAW_OPEN_GAP,
        hold_patch=None,
    )
    nonfinite = _nonfinite(runtime)

    target_center, selector_label, selected_point_index = _target_for_requested_label_from_settled_shell(
        runtime,
        planned_target_center,
        requested_target_label,
    )
    high_center = target_center + np.array([0.0, 0.0, 0.145], dtype=np.float64)
    tug_center = target_center + np.array([0.0, 0.0, PULL_TEST_LIFT_M], dtype=np.float64)
    micro_lift_center = target_center + np.array([0.0, 0.0, MICRO_LIFT_M], dtype=np.float64)

    _advance_stage(
        runtime,
        renderer,
        steps=_steps(runtime.model, APPROACH_SECONDS),
        center_start=high_center,
        center_end=target_center,
        gap_start=JAW_OPEN_GAP,
        gap_end=JAW_OPEN_GAP,
        hold_patch=None,
    )
    close_persistence = _advance_stage(
        runtime,
        renderer,
        steps=_steps(runtime.model, CLOSE_SECONDS),
        center_start=target_center,
        center_end=target_center,
        gap_start=JAW_OPEN_GAP,
        gap_end=JAW_CLOSED_GAP,
        hold_patch=None,
        track_persistence=True,
    )
    captured_at_close = _capture_point_indices(runtime, target_center, JAW_CLOSED_GAP)
    captured_shell_points = len(captured_at_close)
    close_mean = _mean_position_for_points(runtime, captured_at_close)
    close_rel = close_mean - target_center if close_mean is not None else np.zeros(3, dtype=np.float64)
    close_z = float(close_mean[2]) if close_mean is not None else 0.0
    renderer.capture(runtime.data, name="close", keep_frame=True)

    tug_persistence = _advance_stage(
        runtime,
        renderer,
        steps=_steps(runtime.model, PULL_SECONDS),
        center_start=target_center,
        center_end=tug_center,
        gap_start=JAW_CLOSED_GAP,
        gap_end=JAW_CLOSED_GAP,
        hold_patch=None,
        track_persistence=True,
    )
    tug_mean = _mean_position_for_points(runtime, captured_at_close)
    if tug_mean is not None:
        tug_rel = tug_mean - tug_center
        pull_test_slip_mm = float(np.linalg.norm(tug_rel - close_rel) * 1000.0)
    else:
        pull_test_slip_mm = 999.0
    renderer.capture(runtime.data, name="tug_test", keep_frame=True)

    _advance_stage(
        runtime,
        renderer,
        steps=_steps(runtime.model, LIFT_SECONDS),
        center_start=tug_center,
        center_end=micro_lift_center,
        gap_start=JAW_CLOSED_GAP,
        gap_end=JAW_CLOSED_GAP,
        hold_patch=None,
        track_persistence=True,
    )
    micro_mean = _mean_position_for_points(runtime, captured_at_close)
    renderer.capture(runtime.data, name="micro_lift", keep_frame=True)

    hold_steps = _steps(runtime.model, CONTACT_ONLY_HOLD_SECONDS)
    hold_persistence = _advance_stage(
        runtime,
        renderer,
        steps=hold_steps,
        center_start=micro_lift_center,
        center_end=micro_lift_center,
        gap_start=JAW_CLOSED_GAP,
        gap_end=JAW_CLOSED_GAP,
        hold_patch=None,
        track_persistence=True,
    )
    final_mean = _mean_position_for_points(runtime, captured_at_close)
    if final_mean is not None:
        final_rel = final_mean - micro_lift_center
        final_slip_distance = float(np.linalg.norm(final_rel - close_rel))
        lift_height_contact_only = float(final_mean[2] - close_z)
    else:
        final_slip_distance = 999.0
        lift_height_contact_only = 0.0

    contact_persistence_ms = float(close_persistence + tug_persistence + hold_persistence)
    hold_time_contact_only = float(hold_persistence)
    thickness = bundle_thickness_proxy(runtime, captured_at_close)
    balance = bilateral_contact_balance(runtime, captured_at_close, target_center, JAW_CLOSED_GAP)
    ratios = overlap_ratios(runtime.labels, captured_at_close)
    micro_lift_survival = bool(
        final_mean is not None
        and lift_height_contact_only >= 0.020
        and final_slip_distance <= 0.110
        and contact_persistence_ms >= 80.0
    )
    drop_or_not = bool(
        final_mean is None
        or lift_height_contact_only < 0.010
        or final_slip_distance > 0.140
        or _nonfinite(runtime)
    )
    pass_fail = bool(
        captured_shell_points >= 2
        and balance > 0.10
        and pull_test_slip_mm <= 120.0
        and micro_lift_survival
        and not drop_or_not
    )
    nonfinite = bool(nonfinite or _nonfinite(runtime))

    result: dict[str, float | int | bool | str] = {
        "mode": "contact_only_eval",
        "scenario_name": scenario_name,
        "content_case": content_case,
        "trial_index": int(trial_index),
        "requested_target_label": requested_target_label,
        "candidate_selector_label": selector_label,
        "selected_point_index": int(selected_point_index),
        "actual_region_label_at_close": str(ratios["region_label_at_close"]),
        "seam_overlap_ratio": float(ratios["seam_overlap_ratio"]),
        "fold_overlap_ratio": float(ratios["fold_overlap_ratio"]),
        "plain_top_overlap_ratio": float(ratios["plain_top_overlap_ratio"]),
        "captured_shell_points": int(captured_shell_points),
        "bundle_thickness_proxy": float(thickness),
        "bilateral_contact_balance": float(balance),
        "contact_persistence_ms": float(contact_persistence_ms),
        "pull_test_slip_mm": float(pull_test_slip_mm),
        "micro_lift_survival": bool(micro_lift_survival),
        "lift_height_contact_only": float(lift_height_contact_only),
        "hold_time_contact_only": float(hold_time_contact_only),
        "final_slip_distance": float(final_slip_distance),
        "drop_or_not": bool(drop_or_not),
        "pass_fail": bool(pass_fail),
        "hold_surrogate_activated": False,
        "nonfinite": bool(nonfinite),
        "xml": str(xml_path),
        "output_dir": str(output_dir),
    }
    renderer.close(save_mp4=save_mp4)
    return result


def run_trial(
    scenario_name: str,
    *,
    content_case: str = "underfilled",
    trial_index: int = 0,
    output_dir: Path | None = None,
    render: bool = True,
    save_mp4: bool = True,
    width: int = 1280,
    height: int = 720,
) -> dict[str, float | int | bool | str]:
    if scenario_name not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario_name}")
    if content_case not in available_content_cases():
        raise ValueError(f"unknown content case: {content_case}")
    if output_dir is None:
        output_dir = OUTPUT_DIR / content_case / scenario_name / f"trial_{trial_index:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    xml_path = output_dir / f"scene_{scenario_name}_trial_{trial_index:02d}.xml"
    _xml_path, runtime = load_runtime(scenario_name, xml_path=xml_path, content_case=content_case)
    renderer = TrialRenderer(runtime.model, output_dir, enabled=render, width=width, height=height)

    planned_target_center = _target_for_trial(scenario_name, trial_index)
    high_center = planned_target_center + np.array([0.0, 0.0, 0.145], dtype=np.float64)

    set_gripper(runtime, high_center, JAW_OPEN_GAP)
    mujoco.mj_forward(runtime.model, runtime.data)
    renderer.capture(runtime.data, name="initial", keep_frame=True)

    _advance_stage(
        runtime,
        renderer,
        steps=_steps(runtime.model, SETTLE_SECONDS),
        center_start=high_center,
        center_end=high_center,
        gap_start=JAW_OPEN_GAP,
        gap_end=JAW_OPEN_GAP,
    )
    nonfinite = _nonfinite(runtime)

    target_center = _refine_target_z_from_settled_shell(runtime, planned_target_center)
    high_center = target_center + np.array([0.0, 0.0, 0.145], dtype=np.float64)
    pull_center = target_center + np.array([0.0, 0.0, PULL_TEST_LIFT_M], dtype=np.float64)
    lift_center = target_center + np.array([0.0, 0.0, FULL_LIFT_M], dtype=np.float64)

    _advance_stage(
        runtime,
        renderer,
        steps=_steps(runtime.model, APPROACH_SECONDS),
        center_start=high_center,
        center_end=target_center,
        gap_start=JAW_OPEN_GAP,
        gap_end=JAW_OPEN_GAP,
    )
    close_persistence = _advance_stage(
        runtime,
        renderer,
        steps=_steps(runtime.model, CLOSE_SECONDS),
        center_start=target_center,
        center_end=target_center,
        gap_start=JAW_OPEN_GAP,
        gap_end=JAW_CLOSED_GAP,
        track_persistence=True,
    )
    captured_at_close = _capture_point_indices(runtime, target_center, JAW_CLOSED_GAP)
    close_rel = np.zeros(3, dtype=np.float64)
    close_positions = _positions_for_points(runtime, captured_at_close)
    if close_positions.size:
        close_rel = np.mean(close_positions, axis=0) - target_center
    renderer.capture(runtime.data, name="close", keep_frame=True)

    pull_persistence = _advance_stage(
        runtime,
        renderer,
        steps=_steps(runtime.model, PULL_SECONDS),
        center_start=target_center,
        center_end=pull_center,
        gap_start=JAW_CLOSED_GAP,
        gap_end=JAW_CLOSED_GAP,
        track_persistence=True,
    )
    pull_positions = _positions_for_points(runtime, captured_at_close)
    if pull_positions.size:
        pull_rel = np.mean(pull_positions, axis=0) - pull_center
        pull_test_slip_mm = float(np.linalg.norm(pull_rel - close_rel) * 1000.0)
    else:
        pull_test_slip_mm = 999.0
    renderer.capture(runtime.data, name="pull-test", keep_frame=True)

    captured_shell_points = len(captured_at_close)
    thickness = bundle_thickness_proxy(runtime, captured_at_close)
    balance = bilateral_contact_balance(runtime, captured_at_close, target_center, JAW_CLOSED_GAP)
    contact_persistence_ms = close_persistence + pull_persistence
    load_margin_proxy = compute_load_margin(
        captured_shell_points=captured_shell_points,
        bundle_thickness=thickness,
        balance=balance,
        contact_persistence_ms=contact_persistence_ms,
        pull_test_slip_mm=pull_test_slip_mm,
    )
    hold_surrogate_activated = bool(load_margin_proxy >= QUALITY_THRESHOLD and captured_shell_points >= 2)
    hold_patch = _make_hold_patch(runtime, captured_at_close, pull_center) if hold_surrogate_activated else None
    activation_rel = (
        np.mean(hold_patch.relative_positions, axis=0)
        if hold_patch is not None and len(hold_patch.relative_positions)
        else close_rel
    )

    _advance_stage(
        runtime,
        renderer,
        steps=_steps(runtime.model, LIFT_SECONDS),
        center_start=pull_center,
        center_end=lift_center,
        gap_start=JAW_CLOSED_GAP,
        gap_end=JAW_CLOSED_GAP,
        hold_patch=hold_patch,
    )
    renderer.capture(runtime.data, name="lift", keep_frame=True)

    # shake test는 local hold가 실제로 lift 후에도 유지되는지 보는 짧은 흔들림이다.
    shake_start = lift_center.copy()
    shake_steps = _steps(runtime.model, SHAKE_SECONDS)
    frame_stride = max(1, shake_steps // 12)
    for step_index in range(shake_steps):
        phase = 2.0 * math.pi * step_index / max(1, shake_steps - 1)
        center = shake_start + np.array([SHAKE_AMPLITUDE_M * math.sin(phase), 0.0, 0.0])
        set_gripper(runtime, center, JAW_CLOSED_GAP)
        _apply_hold_surrogate(runtime, hold_patch, center)
        mujoco.mj_step(runtime.model, runtime.data)
        if step_index % frame_stride == 0:
            renderer.capture(runtime.data, keep_frame=True)
    final_center = shake_start
    renderer.capture(runtime.data, name="shake", keep_frame=True)

    final_positions = _positions_for_points(runtime, captured_at_close)
    if final_positions.size:
        final_rel = np.mean(final_positions, axis=0) - final_center
        final_slip_distance = float(np.linalg.norm(final_rel - activation_rel))
        # full lift 성과는 hold surrogate가 켜진 pull-test 이후 기준으로 본다.
        # pull-test에서 발생한 slip은 pull_test_slip_mm으로 별도 기록한다.
        lift_height = float(np.mean(final_positions[:, 2]) - (pull_center[2] + activation_rel[2]))
    else:
        final_slip_distance = 999.0
        lift_height = 0.0

    nonfinite = bool(nonfinite or _nonfinite(runtime))
    shake_survival = bool(hold_surrogate_activated and final_slip_distance <= 0.070 and not nonfinite)
    pass_fail = bool(shake_survival and lift_height >= 0.035)

    ratios = overlap_ratios(runtime.labels, captured_at_close)
    result: dict[str, float | int | bool | str] = {
        "scenario_name": scenario_name,
        "content_case": content_case,
        "trial_index": int(trial_index),
        "xml": str(xml_path),
        "region_label_at_close": str(ratios["region_label_at_close"]),
        "seam_overlap_ratio": float(ratios["seam_overlap_ratio"]),
        "fold_overlap_ratio": float(ratios["fold_overlap_ratio"]),
        "plain_top_overlap_ratio": float(ratios["plain_top_overlap_ratio"]),
        "captured_shell_points": int(captured_shell_points),
        "bundle_thickness_proxy": float(thickness),
        "bilateral_contact_balance": float(balance),
        "contact_persistence_ms": float(contact_persistence_ms),
        "pull_test_slip_mm": float(pull_test_slip_mm),
        "load_margin_proxy": float(load_margin_proxy),
        "hold_surrogate_activated": bool(hold_surrogate_activated),
        "lift_height": float(lift_height),
        "shake_survival": bool(shake_survival),
        "final_slip_distance": float(final_slip_distance),
        "pass_fail": bool(pass_fail),
        "nonfinite": bool(nonfinite),
        "output_dir": str(output_dir),
    }
    renderer.close(save_mp4=save_mp4)
    return result


def result_fieldnames() -> list[str]:
    return [
        "scenario_name",
        "content_case",
        "trial_index",
        "region_label_at_close",
        "seam_overlap_ratio",
        "fold_overlap_ratio",
        "plain_top_overlap_ratio",
        "captured_shell_points",
        "bundle_thickness_proxy",
        "bilateral_contact_balance",
        "contact_persistence_ms",
        "pull_test_slip_mm",
        "load_margin_proxy",
        "hold_surrogate_activated",
        "lift_height",
        "shake_survival",
        "final_slip_distance",
        "pass_fail",
        "nonfinite",
        "xml",
        "output_dir",
    ]


def write_summary_csv(results: list[dict[str, float | int | bool | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=result_fieldnames())
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key, "") for key in result_fieldnames()})


def contact_only_fieldnames() -> list[str]:
    return [
        "mode",
        "scenario_name",
        "content_case",
        "selfcollide_mode",
        "noslip_iterations",
        "multiccd_mode",
        "nativeccd_mode",
        "pad_profile",
        "pad_condim",
        "vertcollide_mode",
        "shell_thickness_scale",
        "close_timestep",
        "gripper_kv",
        "gripper_dampratio",
        "trial_index",
        "requested_target_label",
        "actual_region_label_at_close",
        "accepted_candidate_rank",
        "seam_overlap_ratio",
        "fold_overlap_ratio",
        "plain_top_overlap_ratio",
        "left_contact_present",
        "right_contact_present",
        "trapped_shell_points",
        "trapped_patch_arc_length",
        "captured_shell_points",
        "bundle_thickness_proxy",
        "bilateral_contact_balance",
        "tangential_slip_proxy",
        "jaw_escape",
        "rollback_used",
        "contact_persistence_ms",
        "pull_test_slip_mm",
        "micro_lift_survival",
        "load_following_ratio",
        "latch_activated",
        "latch_activation_time",
        "lift_height",
        "hold_time",
        "lift_height_contact_only",
        "hold_time_contact_only",
        "final_slip_distance",
        "drop_or_not",
        "no_graspable_patch_found",
        "pass_fail",
        "hold_surrogate_activated",
        "rectification_used",
        "candidate_policy_tag",
        "candidate_score",
        "nonfinite",
        "xml",
        "output_dir",
    ]


def write_contact_only_summary_csv(results: list[dict[str, float | int | bool | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=contact_only_fieldnames())
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key, "") for key in contact_only_fieldnames()})


def write_contact_only_summary_markdown(results: list[dict[str, float | int | bool | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = len(results)
    pass_count = sum(1 for result in results if bool(result.get("pass_fail", False)))
    drop_count = sum(1 for result in results if bool(result.get("drop_or_not", False)))
    lines = [
        "# Auto Search And Probe Graspability Summary",
        "",
        "This report is generated from auto search + probing.",
        "`contact_only_eval` keeps latch disabled. `qualification_gated_capture` enables a task-driven soft latch only after capture qualification; `qualification_gated_latch_eval` is kept as a backward-compatible alias.",
        "",
        f"- total_trials: {total}",
        f"- pass_count: {pass_count}",
        f"- fail_count: {total - pass_count}",
        f"- pass_rate: {pass_count / max(total, 1):.3f}",
        f"- drop_count: {drop_count}",
        f"- no_graspable_patch_found_count: {sum(1 for result in results if bool(result.get('no_graspable_patch_found', False)))}",
        "",
        "| mode | scenario | content_case | requested_label | actual_label | rank | trapped | L/R | escape | slip_mm | follow | latch | no_patch | pass |",
        "|---|---|---|---|---|---:|---:|---|---|---:|---:|---|---|---|",
    ]
    for result in results:
        lines.append(
            "| {mode} | {scenario} | {content_case} | {requested} | {actual} | {rank} | {captured} | {lr} | {escape} | {slip:.1f} | {follow:.2f} | {latch} | {no_patch} | {passed} |".format(
                mode=result.get("mode", ""),
                scenario=result.get("scenario_name", ""),
                content_case=result.get("content_case", ""),
                requested=result.get("requested_target_label", ""),
                actual=result.get("actual_region_label_at_close", ""),
                rank=int(result.get("accepted_candidate_rank", -1)),
                captured=int(result.get("trapped_shell_points", result.get("captured_shell_points", 0))),
                lr=f"{bool(result.get('left_contact_present', False))}/{bool(result.get('right_contact_present', False))}",
                escape=result.get("jaw_escape", ""),
                slip=float(result.get("pull_test_slip_mm", 0.0)),
                follow=float(result.get("load_following_ratio", 0.0)),
                latch=result.get("latch_activated", ""),
                no_patch=result.get("no_graspable_patch_found", ""),
                passed=result.get("pass_fail", ""),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_contact_only_suite(
    *,
    scenarios: Iterable[str] | None = None,
    mode: str = "contact_only_eval",
    content_case: str = "underfilled",
    requested_target_labels: Iterable[str] | None = None,
    trials: int = 4,
    output_dir: Path | None = None,
    render: bool = True,
    save_mp4: bool = True,
    width: int = 1280,
    height: int = 720,
    selfcollide_mode: str = SELF_COLLISION_MODE,
    noslip_iterations: int = 0,
    multiccd_mode: str = "off",
    nativeccd_mode: str = "off",
    pad_profile: str = "lip",
    pad_condim: int = 4,
    vertcollide_mode: str = "false",
    shell_thickness_scale: float = 1.0,
    close_seconds: float = SEARCH_CLOSE_SECONDS,
    precompression_dwell_seconds: float = PRECOMPRESSION_DWELL_SECONDS,
    close_timestep: float = 0.001,
    gripper_kv: float = 320.0,
    gripper_dampratio: float = 0.20,
) -> tuple[list[dict[str, float | int | bool | str]], Path, Path]:
    if mode not in AUTO_SEARCH_MODES:
        raise ValueError(f"unknown mode: {mode}")
    if content_case not in available_content_cases():
        raise ValueError(f"unknown content case: {content_case}")
    if output_dir is None:
        output_dir = (
            OUTPUT_DIR
            / "as"
            / MODE_PATH_TOKEN[mode]
            / (
                f"{content_case[:3]}_{selfcollide_mode}_n{int(noslip_iterations)}_"
                f"ccd{1 if multiccd_mode == 'on' else 0}_{pad_profile[0]}c{int(pad_condim)}_"
                f"t{int(round(shell_thickness_scale * 100))}_dt{int(round(float(close_timestep) * 1_000_000))}"
            )
        )
    scenario_names = list(scenarios or available_scenarios())
    target_labels = list(requested_target_labels or ("auto",))
    results: list[dict[str, float | int | bool | str]] = []
    for scenario_name in scenario_names:
        for requested_target_label in target_labels:
            for trial_index in range(trials):
                trial_output = output_dir / f"{scenario_name}_{_label_path_token(requested_target_label)}_t{trial_index:02d}"
                result = run_auto_search_and_probe_trial(
                    scenario_name,
                    mode=mode,
                    content_case=content_case,
                    requested_target_label=requested_target_label,
                    trial_index=trial_index,
                    output_dir=trial_output,
                    render=render,
                    save_mp4=save_mp4,
                    width=width,
                    height=height,
                    selfcollide_mode=selfcollide_mode,
                    noslip_iterations=noslip_iterations,
                    multiccd_mode=multiccd_mode,
                    nativeccd_mode=nativeccd_mode,
                    pad_profile=pad_profile,
                    pad_condim=pad_condim,
                    vertcollide_mode=vertcollide_mode,
                    shell_thickness_scale=shell_thickness_scale,
                    close_seconds=close_seconds,
                    precompression_dwell_seconds=precompression_dwell_seconds,
                    close_timestep=close_timestep,
                    gripper_kv=gripper_kv,
                    gripper_dampratio=gripper_dampratio,
                )
                results.append(result)
                print(
                    " ".join(
                        [
                            f"mode={result['mode']}",
                            f"scenario_name={result['scenario_name']}",
                            f"content_case={result['content_case']}",
                            f"selfcollide_mode={result['selfcollide_mode']}",
                            f"noslip_iterations={result['noslip_iterations']}",
                            f"multiccd_mode={result['multiccd_mode']}",
                            f"pad_profile={result['pad_profile']}",
                            f"pad_condim={result['pad_condim']}",
                            f"close_timestep={result['close_timestep']}",
                            f"requested_target_label={result['requested_target_label']}",
                            f"actual_region_label_at_close={result['actual_region_label_at_close']}",
                            f"accepted_candidate_rank={result['accepted_candidate_rank']}",
                            f"trapped_shell_points={result['trapped_shell_points']}",
                            f"left_contact_present={result['left_contact_present']}",
                            f"right_contact_present={result['right_contact_present']}",
                            f"jaw_escape={result['jaw_escape']}",
                            f"rollback_used={result['rollback_used']}",
                            f"pull_test_slip_mm={float(result['pull_test_slip_mm']):.1f}",
                            f"load_following_ratio={float(result['load_following_ratio']):.2f}",
                            f"latch_activated={result['latch_activated']}",
                            f"no_graspable_patch_found={result['no_graspable_patch_found']}",
                            f"micro_lift_survival={result['micro_lift_survival']}",
                            f"drop_or_not={result['drop_or_not']}",
                            f"pass_fail={result['pass_fail']}",
                        ]
                    )
                )
    summary_csv = output_dir / "summary.csv"
    summary_md = output_dir / "summary.md"
    write_contact_only_summary_csv(results, summary_csv)
    write_contact_only_summary_markdown(results, summary_md)
    return results, summary_csv, summary_md


def run_suite(
    *,
    scenarios: Iterable[str] | None = None,
    content_case: str = "underfilled",
    trials: int = 4,
    output_dir: Path | None = None,
    render: bool = True,
    save_mp4: bool = True,
    width: int = 1280,
    height: int = 720,
) -> tuple[list[dict[str, float | int | bool | str]], Path]:
    if content_case not in available_content_cases():
        raise ValueError(f"unknown content case: {content_case}")
    if output_dir is None:
        output_dir = OUTPUT_DIR / "validation" / content_case
    scenario_names = list(scenarios or available_scenarios())
    results: list[dict[str, float | int | bool | str]] = []
    for scenario_name in scenario_names:
        for trial_index in range(trials):
            trial_output = output_dir / scenario_name / f"trial_{trial_index:02d}"
            result = run_trial(
                scenario_name,
                content_case=content_case,
                trial_index=trial_index,
                output_dir=trial_output,
                render=render,
                save_mp4=save_mp4,
                width=width,
                height=height,
            )
            results.append(result)
            print(
                " ".join(
                    [
                        f"scenario_name={result['scenario_name']}",
                        f"content_case={result['content_case']}",
                        f"trial_index={result['trial_index']}",
                        f"region_label_at_close={result['region_label_at_close']}",
                        f"captured_shell_points={result['captured_shell_points']}",
                        f"load_margin_proxy={float(result['load_margin_proxy']):.3f}",
                        f"hold_surrogate_activated={result['hold_surrogate_activated']}",
                        f"pass_fail={result['pass_fail']}",
                    ]
                )
            )
    summary_path = output_dir / "summary.csv"
    write_summary_csv(results, summary_path)
    return results, summary_path


def launch_viewer(
    scenario_name: str,
    trial_index: int = 0,
    speed: float = 1.0,
    content_case: str = "underfilled",
) -> None:
    import mujoco.viewer

    xml_path, runtime = load_runtime(scenario_name, content_case=content_case)
    print(f"xml={xml_path}")
    print(f"content_case={content_case}")
    print("viewer=true left-drag=rotate right-drag=pan wheel=zoom")

    sleep_dt = runtime.model.opt.timestep / max(speed, 1e-6)

    with mujoco.viewer.launch_passive(runtime.model, runtime.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = np.array([0.0, 0.0, BAG_FRAME_Z])
        viewer.cam.distance = 0.78
        viewer.cam.azimuth = 138.0
        viewer.cam.elevation = -16.0
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXFACE] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXEDGE] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXVERT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

        def play_stage(
            *,
            seconds: float,
            start: np.ndarray,
            end: np.ndarray,
            gap_start: float,
            gap_end: float,
            hold_patch: GraspPatch | None = None,
            track_persistence: bool = False,
        ) -> float:
            persistence_ms = 0.0
            steps = _steps(runtime.model, seconds)
            for step_index in range(steps):
                if not viewer.is_running():
                    return persistence_ms
                step_start = time.perf_counter()
                alpha = (step_index + 1) / steps
                center = (1.0 - alpha) * start + alpha * end
                gap = (1.0 - alpha) * gap_start + alpha * gap_end
                set_gripper(runtime, center, gap)
                _apply_hold_surrogate(runtime, hold_patch, center)
                if track_persistence:
                    captured = _capture_point_indices(runtime, center, gap)
                    balance = bilateral_contact_balance(runtime, captured, center, gap)
                    if len(captured) >= 2 and balance > 0.15:
                        persistence_ms += runtime.model.opt.timestep * 1000.0
                mujoco.mj_step(runtime.model, runtime.data)
                viewer.sync()
                elapsed = time.perf_counter() - step_start
                remaining = sleep_dt - elapsed
                if remaining > 0:
                    time.sleep(remaining)
            return float(persistence_ms)

        while viewer.is_running():
            mujoco.mj_resetData(runtime.model, runtime.data)
            planned_target = _target_for_trial(scenario_name, trial_index)
            high_center = planned_target + np.array([0.0, 0.0, 0.145], dtype=np.float64)
            set_gripper(runtime, high_center, JAW_OPEN_GAP)
            mujoco.mj_forward(runtime.model, runtime.data)

            play_stage(
                seconds=SETTLE_SECONDS,
                start=high_center,
                end=high_center,
                gap_start=JAW_OPEN_GAP,
                gap_end=JAW_OPEN_GAP,
            )
            target_center = _refine_target_z_from_settled_shell(runtime, planned_target)
            high_center = target_center + np.array([0.0, 0.0, 0.145], dtype=np.float64)
            pull_center = target_center + np.array([0.0, 0.0, PULL_TEST_LIFT_M], dtype=np.float64)
            lift_center = target_center + np.array([0.0, 0.0, FULL_LIFT_M], dtype=np.float64)

            play_stage(
                seconds=APPROACH_SECONDS,
                start=high_center,
                end=target_center,
                gap_start=JAW_OPEN_GAP,
                gap_end=JAW_OPEN_GAP,
            )
            close_persistence = play_stage(
                seconds=CLOSE_SECONDS,
                start=target_center,
                end=target_center,
                gap_start=JAW_OPEN_GAP,
                gap_end=JAW_CLOSED_GAP,
                track_persistence=True,
            )
            captured_at_close = _capture_point_indices(runtime, target_center, JAW_CLOSED_GAP)
            close_positions = _positions_for_points(runtime, captured_at_close)
            close_rel = (
                np.mean(close_positions, axis=0) - target_center
                if close_positions.size
                else np.zeros(3, dtype=np.float64)
            )
            pull_persistence = play_stage(
                seconds=PULL_SECONDS,
                start=target_center,
                end=pull_center,
                gap_start=JAW_CLOSED_GAP,
                gap_end=JAW_CLOSED_GAP,
                track_persistence=True,
            )
            pull_positions = _positions_for_points(runtime, captured_at_close)
            if pull_positions.size:
                pull_rel = np.mean(pull_positions, axis=0) - pull_center
                pull_test_slip_mm = float(np.linalg.norm(pull_rel - close_rel) * 1000.0)
            else:
                pull_test_slip_mm = 999.0

            load_margin_proxy = compute_load_margin(
                captured_shell_points=len(captured_at_close),
                bundle_thickness=bundle_thickness_proxy(runtime, captured_at_close),
                balance=bilateral_contact_balance(runtime, captured_at_close, target_center, JAW_CLOSED_GAP),
                contact_persistence_ms=close_persistence + pull_persistence,
                pull_test_slip_mm=pull_test_slip_mm,
            )
            hold_active = bool(load_margin_proxy >= QUALITY_THRESHOLD and len(captured_at_close) >= 2)
            hold_patch = _make_hold_patch(runtime, captured_at_close, pull_center) if hold_active else None
            print(
                f"captured_shell_points={len(captured_at_close)} "
                f"load_margin_proxy={load_margin_proxy:.3f} hold_surrogate_activated={hold_active}"
            )

            play_stage(
                seconds=LIFT_SECONDS,
                start=pull_center,
                end=lift_center,
                gap_start=JAW_CLOSED_GAP,
                gap_end=JAW_CLOSED_GAP,
                hold_patch=hold_patch,
            )
            time.sleep(0.35 / max(speed, 1e-6))
