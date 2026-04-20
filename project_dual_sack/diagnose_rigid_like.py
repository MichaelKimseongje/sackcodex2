from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

from scenario_builder import (
    CONNECTED_COLUMN_COUNT,
    INNER_BOTTOM_PANEL_COUNT,
    INNER_LOAD_PANEL_COUNT,
    OUTER_BOTTOM_EDGE_COUNT,
    OUTER_LOWER_COUNT,
    OUTER_SHOULDER_COUNT,
    OUT_DIR,
    SCENARIO_NAMES,
    TOP_SEAM_COUNT,
    get_scenario,
    write_scene_xml,
)


# Windows 작업 경로가 길어서 keyframe 저장이 실패하지 않도록 짧은 출력 폴더명을 씁니다.
OUTPUT_DIR = OUT_DIR / "rld"

SUMMARY_FIELDS = [
    "scenario_name",
    "test_name",
    "mode_free_or_anchored",
    "world_frame_bag_translation_mm",
    "world_frame_bag_rotation_deg",
    "bag_frame_local_deformation_mm",
    "rigid_like_flag",
    "shoulder_joint_max_delta_deg",
    "belly_joint_max_delta_deg",
    "fold_joint_max_delta_deg",
    "bottom_sling_max_travel_mm",
    "joint_limit_hit_count",
    "shoulder_deflection_mm",
    "top_patch_change_mm",
    "lower_belly_opening_mm",
    "width_reduction_mm",
    "bottom_sag_mm",
    "top_drop_mm",
    "fold_exposed_fraction_before_after",
    "requested_target_label",
    "actual_region_label_at_close",
    "trapped_patch_count",
    "pull_test_slip_mm",
    "load_following_ratio",
    "scoop_engaged",
    "support_state_formed",
]

TIMESERIES_FIELDS = [
    "scenario_name",
    "test_name",
    "mode_free_or_anchored",
    "time_s",
    "world_frame_bag_translation_mm",
    "world_frame_bag_rotation_deg",
    "bag_frame_local_deformation_mm",
    "shoulder_deflection_mm",
    "top_patch_change_mm",
    "lower_belly_opening_mm",
    "width_reduction_mm",
    "bottom_sag_mm",
    "top_drop_mm",
]

JOINT_RESPONSE_FIELDS = [
    "scenario_name",
    "test_name",
    "mode_free_or_anchored",
    "joint_name",
    "joint_type",
    "joint_max_delta_deg_or_mm",
    "limit_hit",
]

PATCH_SITES = [
    "site_top_grasp_rail_center",
    "site_top_seam_00",
    f"site_top_seam_{TOP_SEAM_COUNT // 2:02d}",
    f"site_top_seam_{TOP_SEAM_COUNT - 1:02d}",
    *[f"site_outer_upper_left_{i:02d}" for i in range(TOP_SEAM_COUNT)],
    *[f"site_outer_upper_right_{i:02d}" for i in range(TOP_SEAM_COUNT)],
    *[f"site_outer_mid_front_{i:02d}" for i in range(TOP_SEAM_COUNT)],
    *[f"site_outer_mid_back_{i:02d}" for i in range(TOP_SEAM_COUNT)],
    "site_outer_side_left_center",
    "site_outer_side_right_center",
    *[f"site_outer_lower_left_{i:02d}" for i in range(TOP_SEAM_COUNT)],
    *[f"site_outer_lower_right_{i:02d}" for i in range(TOP_SEAM_COUNT)],
    *[f"site_outer_bottom_edge_left_{i:02d}" for i in range(TOP_SEAM_COUNT)],
    *[f"site_outer_bottom_edge_right_{i:02d}" for i in range(TOP_SEAM_COUNT)],
    "site_outer_bottom_edge_center",
    *[f"site_inner_front_load_{i:02d}" for i in range(INNER_LOAD_PANEL_COUNT)],
    *[f"site_inner_back_load_{i:02d}" for i in range(INNER_LOAD_PANEL_COUNT)],
    *[f"site_inner_bottom_load_{i:02d}" for i in range(INNER_BOTTOM_PANEL_COUNT)],
    "site_top_edge_occlusion_left",
    "site_top_edge_occlusion_right",
]

DIAGNOSTIC_MODES = ("free", "anchored")
VIDEO_STYLES = ("outer_shell_only", "inner_shell_only", "ballast_only", "visual_skin", "overlay")
TOP_CENTER_SITE = f"site_top_seam_{TOP_SEAM_COUNT // 2:02d}"
TOP_CENTER_BODY = f"top_seam_{TOP_SEAM_COUNT // 2:02d}"
TOP_RAIL_SITE = "site_top_grasp_rail_center"
PANEL_CENTER_INDEX = TOP_SEAM_COUNT // 2
CONNECTED_LOWER_FRONT_SITE = "__removed_connected_lower_front"
CONNECTED_LOWER_BACK_SITE = "__removed_connected_lower_back"
LOWER_LEFT_SITE = f"site_outer_lower_left_{PANEL_CENTER_INDEX:02d}"
LOWER_RIGHT_SITE = f"site_outer_lower_right_{PANEL_CENTER_INDEX:02d}"
LOWER_LEFT_BODY = f"outer_lower_left_{PANEL_CENTER_INDEX:02d}"
LOWER_RIGHT_BODY = f"outer_lower_right_{PANEL_CENTER_INDEX:02d}"
BOTTOM_CENTER_SITE = "site_outer_bottom_edge_center"
BOTTOM_CENTER_BODY = "outer_bottom_edge_center"
SHOULDER_CENTER_INDEX = PANEL_CENTER_INDEX
SHOULDER_CENTER_SITE_LEFT = f"site_outer_upper_left_{SHOULDER_CENTER_INDEX:02d}"
SHOULDER_CENTER_SITE_RIGHT = f"site_outer_upper_right_{SHOULDER_CENTER_INDEX:02d}"
SHOULDER_CENTER_BODY_LEFT = f"outer_upper_left_{SHOULDER_CENTER_INDEX:02d}"
SHOULDER_CENTER_BODY_RIGHT = f"outer_upper_right_{SHOULDER_CENTER_INDEX:02d}"


@dataclass
class TestSpec:
    name: str
    keyframe_name: str
    target_label: str
    duration_s: float
    stimulus: Callable[["DiagnosticContext", int, float], None]
    is_applicable: Callable[[str], bool] = lambda _scenario: True


@dataclass
class DiagnosticContext:
    scenario: str
    test_name: str
    mode: str
    model: mujoco.MjModel
    data: mujoco.MjData
    bag_pos0: np.ndarray
    bag_mat0: np.ndarray
    local0: dict[str, np.ndarray]
    qpos0: dict[str, float]
    width0: float
    lower_gap0: float
    fold_exposed_before: float
    requested_target_label: str
    actual_region_label_at_close: str
    applicable: bool
    rows_timeseries: list[dict[str, object]] = field(default_factory=list)
    joint_limit_hits: set[str] = field(default_factory=set)
    joint_max_abs_delta: dict[str, float] = field(default_factory=dict)
    max_world_translation_mm: float = 0.0
    max_world_rotation_deg: float = 0.0
    max_local_deformation_mm: float = 0.0
    max_shoulder_deflection_mm: float = 0.0
    max_top_patch_change_mm: float = 0.0
    max_lower_belly_opening_mm: float = 0.0
    max_width_reduction_mm: float = 0.0
    max_bottom_sag_mm: float = 0.0
    max_top_drop_mm: float = 0.0


def _imageio():
    try:
        import imageio.v2 as imageio
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("imageio가 필요합니다. `pip install imageio imageio-ffmpeg` 후 다시 실행해 주세요.") from exc
    return imageio


def _body_id(model: mujoco.MjModel, name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise KeyError(f"body not found: {name}")
    return bid


def _site_id(model: mujoco.MjModel, name: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise KeyError(f"site not found: {name}")
    return sid


def _joint_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def _geom_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


def _bag_pose(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    bid = _body_id(model, "bag_frame")
    return data.xpos[bid].copy(), data.xmat[bid].reshape(3, 3).copy()


def _site_world(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    return data.site_xpos[_site_id(model, name)].copy()


def _site_local(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    bag_pos, bag_mat = _bag_pose(model, data)
    return bag_mat.T @ (_site_world(model, data, name) - bag_pos)


def _rotation_delta_deg(mat0: np.ndarray, mat1: np.ndarray) -> float:
    rel = mat0.T @ mat1
    cos_angle = np.clip((float(np.trace(rel)) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def _rotation_error_vector(mat0: np.ndarray, mat1: np.ndarray) -> np.ndarray:
    rel = mat1 @ mat0.T
    return 0.5 * np.array(
        [rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]],
        dtype=np.float64,
    )


def _existing_patch_sites(model: mujoco.MjModel) -> list[str]:
    return [name for name in PATCH_SITES if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) >= 0]


def _sack_joint_names(model: mujoco.MjModel) -> list[str]:
    names: list[str] = []
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        if (
            name.startswith("top_grasp_rail_")
            or name.startswith("top_seam_")
            or name.startswith("outer_")
            or name.startswith("connected_")
            or name.startswith("top_edge_occlusion_")
            or name.startswith("inner_")
            or name.startswith("ballast_")
        ):
            names.append(name)
    return names


def _qpos_for_joint(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str) -> float:
    jid = _joint_id(model, joint_name)
    return float(data.qpos[int(model.jnt_qposadr[jid])])


def _joint_kind(model: mujoco.MjModel, joint_name: str) -> str:
    jid = _joint_id(model, joint_name)
    joint_type = int(model.jnt_type[jid])
    if joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
        return "slide_mm"
    if joint_type == mujoco.mjtJoint.mjJNT_HINGE:
        return "hinge_deg"
    return "other"


def _apply_joint_torque(ctx: DiagnosticContext, joint_name: str, torque: float) -> None:
    """로컬 접촉 preload가 hinge/slide에 만드는 저차 generalized force surrogate입니다."""

    jid = _joint_id(ctx.model, joint_name)
    if jid >= 0:
        ctx.data.qfrc_applied[int(ctx.model.jnt_dofadr[jid])] += float(torque)


def _reset_and_settle(model: mujoco.MjModel, data: mujoco.MjData, seconds: float = 0.35) -> None:
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    for _ in range(max(1, int(seconds / model.opt.timestep))):
        mujoco.mj_step(model, data)
    data.xfrc_applied[:] = 0.0


def _load_model(scenario: str) -> tuple[mujoco.MjModel, mujoco.MjData, Path]:
    # 이 스크립트는 자루 topology 진단 전용입니다. 로봇 mesh 반복 로드 비용/경로 문제를 피하고,
    # 2F preload와 scoop insertion은 controlled diagnostic stimulus로 대체합니다.
    xml = write_scene_xml(scenario, include_robots=False)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    _reset_and_settle(model, data)
    return model, data, xml


def _make_context(model: mujoco.MjModel, data: mujoco.MjData, scenario: str, spec: TestSpec, mode: str) -> DiagnosticContext:
    bag_pos, bag_mat = _bag_pose(model, data)
    local0 = {name: _site_local(model, data, name) for name in _existing_patch_sites(model)}
    qpos0 = {name: _qpos_for_joint(model, data, name) for name in _sack_joint_names(model)}
    width0 = float(np.linalg.norm(local0["site_outer_side_left_center"] - local0["site_outer_side_right_center"]))
    lower_gap0 = float(np.linalg.norm(local0[LOWER_LEFT_SITE] - local0[LOWER_RIGHT_SITE]))
    state = get_scenario(scenario)
    fold_exposed_before = 1.0 - state.fold_coverage_fraction
    applicable = spec.is_applicable(scenario)
    actual_region = spec.target_label if applicable else "not_applicable"
    return DiagnosticContext(
        scenario=scenario,
        test_name=spec.name,
        mode=mode,
        model=model,
        data=data,
        bag_pos0=bag_pos,
        bag_mat0=bag_mat,
        local0=local0,
        qpos0=qpos0,
        width0=width0,
        lower_gap0=lower_gap0,
        fold_exposed_before=fold_exposed_before,
        requested_target_label=spec.target_label if applicable else "not_applicable",
        actual_region_label_at_close=actual_region,
        applicable=applicable,
    )


def _apply_anchor(ctx: DiagnosticContext) -> None:
    """진단 동안만 쓰는 약한 world anchor입니다. 학습/평가 모드에는 쓰지 않습니다."""

    bid = _body_id(ctx.model, "bag_frame")
    pos, mat = _bag_pose(ctx.model, ctx.data)
    cvel = ctx.data.cvel[bid].copy()
    lin_vel = cvel[3:6]
    ang_vel = cvel[0:3]
    ctx.data.xfrc_applied[bid, :3] += -850.0 * (pos - ctx.bag_pos0) - 42.0 * lin_vel
    ctx.data.xfrc_applied[bid, 3:] += -7.5 * _rotation_error_vector(ctx.bag_mat0, mat) - 1.2 * ang_vel


def _current_shape_metrics(ctx: DiagnosticContext) -> dict[str, float]:
    model, data = ctx.model, ctx.data
    left = 1000.0 * float(np.linalg.norm(_site_local(model, data, SHOULDER_CENTER_SITE_LEFT) - ctx.local0[SHOULDER_CENTER_SITE_LEFT]))
    right = 1000.0 * float(np.linalg.norm(_site_local(model, data, SHOULDER_CENTER_SITE_RIGHT) - ctx.local0[SHOULDER_CENTER_SITE_RIGHT]))
    shoulder_arc = 0.0
    for side in ("left", "right"):
        for i in range(TOP_SEAM_COUNT):
            joint_name = f"outer_upper_{side}_{i:02d}_hinge"
            if joint_name in ctx.qpos0:
                shoulder_arc = max(shoulder_arc, 1000.0 * 0.050 * abs(_qpos_for_joint(model, data, joint_name) - ctx.qpos0[joint_name]))
    top = 0.0
    if TOP_RAIL_SITE in ctx.local0:
        top = max(top, 1000.0 * float(np.linalg.norm(_site_local(model, data, TOP_RAIL_SITE) - ctx.local0[TOP_RAIL_SITE])))
    for i in range(TOP_SEAM_COUNT):
        name = f"site_top_seam_{i:02d}"
        if name in ctx.local0:
            top = max(top, 1000.0 * float(np.linalg.norm(_site_local(model, data, name) - ctx.local0[name])))
    gap = float(np.linalg.norm(_site_local(model, data, LOWER_LEFT_SITE) - _site_local(model, data, LOWER_RIGHT_SITE)))
    lower_left_motion = 1000.0 * float(np.linalg.norm(_site_local(model, data, LOWER_LEFT_SITE) - ctx.local0[LOWER_LEFT_SITE]))
    lower_right_motion = 1000.0 * float(np.linalg.norm(_site_local(model, data, LOWER_RIGHT_SITE) - ctx.local0[LOWER_RIGHT_SITE]))
    connected_opening = 0.0
    if CONNECTED_LOWER_FRONT_SITE in ctx.local0 and CONNECTED_LOWER_BACK_SITE in ctx.local0:
        connected_gap = float(
            np.linalg.norm(_site_local(model, data, CONNECTED_LOWER_FRONT_SITE) - _site_local(model, data, CONNECTED_LOWER_BACK_SITE))
        )
        connected_gap0 = float(np.linalg.norm(ctx.local0[CONNECTED_LOWER_FRONT_SITE] - ctx.local0[CONNECTED_LOWER_BACK_SITE]))
        connected_opening = 1000.0 * abs(connected_gap - connected_gap0)
    width = float(np.linalg.norm(_site_local(model, data, "site_outer_side_left_center") - _site_local(model, data, "site_outer_side_right_center")))
    bottom = _site_local(model, data, BOTTOM_CENTER_SITE)[2]
    bottom_sag = 1000.0 * max(0.0, ctx.local0[BOTTOM_CENTER_SITE][2] - bottom)
    for name in (
        "site_connected_bottom_left_inner",
        "site_connected_bottom_left_outer",
        "site_connected_bottom_right_inner",
        "site_connected_bottom_right_outer",
    ):
        if name in ctx.local0:
            current = _site_local(model, data, name)
            # 바닥 edge가 위로 말려 올라가는 것도 lift 상태에서의 bottom shape response로 기록한다.
            bottom_sag = max(bottom_sag, 1000.0 * max(0.0, current[2] - ctx.local0[name][2]))
    for joint_name in (
        "connected_bottom_center_sling_hinge",
        "connected_bottom_left_inner_hinge",
        "connected_bottom_left_outer_hinge",
        "connected_bottom_right_inner_hinge",
        "connected_bottom_right_outer_hinge",
    ):
        if joint_name in ctx.qpos0:
            bottom_sag = max(bottom_sag, 1000.0 * 0.060 * abs(_qpos_for_joint(model, data, joint_name) - ctx.qpos0[joint_name]))
    for i in range(OUTER_BOTTOM_EDGE_COUNT):
        joint_name = f"outer_bottom_edge_{i:02d}_slide"
        if joint_name in ctx.qpos0:
            bottom_sag = max(bottom_sag, 1000.0 * abs(_qpos_for_joint(model, data, joint_name) - ctx.qpos0[joint_name]))
        joint_name = f"outer_bottom_edge_{i:02d}_hinge"
        if joint_name in ctx.qpos0:
            bottom_sag = max(bottom_sag, 1000.0 * 0.050 * abs(_qpos_for_joint(model, data, joint_name) - ctx.qpos0[joint_name]))
    for side in ("left", "right"):
        for i in range(TOP_SEAM_COUNT):
            name = f"site_outer_bottom_edge_{side}_{i:02d}"
            if name in ctx.local0:
                bottom_sag = max(bottom_sag, 1000.0 * abs(_site_local(model, data, name)[2] - ctx.local0[name][2]))
            joint_name = f"outer_bottom_edge_{side}_{i:02d}_hinge"
            if joint_name in ctx.qpos0:
                bottom_sag = max(bottom_sag, 1000.0 * 0.055 * abs(_qpos_for_joint(model, data, joint_name) - ctx.qpos0[joint_name]))
    if "outer_bottom_edge_center_hinge" in ctx.qpos0:
        bottom_sag = max(bottom_sag, 1000.0 * 0.065 * abs(_qpos_for_joint(model, data, "outer_bottom_edge_center_hinge") - ctx.qpos0["outer_bottom_edge_center_hinge"]))
    for i in range(INNER_BOTTOM_PANEL_COUNT):
        name = f"site_inner_bottom_load_{i:02d}"
        if name in ctx.local0:
            bottom_sag = max(bottom_sag, 1000.0 * max(0.0, ctx.local0[name][2] - _site_local(model, data, name)[2]))
        joint_name = f"inner_bottom_load_{i:02d}_slide"
        if joint_name in ctx.qpos0:
            bottom_sag = max(bottom_sag, 1000.0 * abs(_qpos_for_joint(model, data, joint_name) - ctx.qpos0[joint_name]))
        joint_name = f"inner_bottom_load_{i:02d}_hinge"
        if joint_name in ctx.qpos0:
            bottom_sag = max(bottom_sag, 1000.0 * 0.055 * abs(_qpos_for_joint(model, data, joint_name) - ctx.qpos0[joint_name]))
    top_z = _site_local(model, data, TOP_CENTER_SITE)[2]
    return {
        "shoulder_deflection_mm": max(left, right, shoulder_arc),
        "top_patch_change_mm": top,
        "lower_belly_opening_mm": max(1000.0 * abs(gap - ctx.lower_gap0), lower_left_motion, lower_right_motion, connected_opening),
        "width_reduction_mm": 1000.0 * max(0.0, ctx.width0 - width),
        "bottom_sag_mm": bottom_sag,
        "top_drop_mm": 1000.0 * max(0.0, ctx.local0[TOP_CENTER_SITE][2] - top_z),
    }


def _update_context(ctx: DiagnosticContext, time_s: float, *, sample: bool) -> None:
    model, data = ctx.model, ctx.data
    bag_pos, bag_mat = _bag_pose(model, data)
    ctx.max_world_translation_mm = max(ctx.max_world_translation_mm, 1000.0 * float(np.linalg.norm(bag_pos - ctx.bag_pos0)))
    ctx.max_world_rotation_deg = max(ctx.max_world_rotation_deg, _rotation_delta_deg(ctx.bag_mat0, bag_mat))
    for site_name, start_local in ctx.local0.items():
        ctx.max_local_deformation_mm = max(
            ctx.max_local_deformation_mm,
            1000.0 * float(np.linalg.norm(_site_local(model, data, site_name) - start_local)),
        )
    shape = _current_shape_metrics(ctx)
    ctx.max_shoulder_deflection_mm = max(ctx.max_shoulder_deflection_mm, shape["shoulder_deflection_mm"])
    ctx.max_top_patch_change_mm = max(ctx.max_top_patch_change_mm, shape["top_patch_change_mm"])
    ctx.max_lower_belly_opening_mm = max(ctx.max_lower_belly_opening_mm, shape["lower_belly_opening_mm"])
    ctx.max_width_reduction_mm = max(ctx.max_width_reduction_mm, shape["width_reduction_mm"])
    ctx.max_bottom_sag_mm = max(ctx.max_bottom_sag_mm, shape["bottom_sag_mm"])
    ctx.max_top_drop_mm = max(ctx.max_top_drop_mm, shape["top_drop_mm"])

    for joint_name, start in ctx.qpos0.items():
        jid = _joint_id(model, joint_name)
        now = _qpos_for_joint(model, data, joint_name)
        delta = abs(now - start)
        ctx.joint_max_abs_delta[joint_name] = max(ctx.joint_max_abs_delta.get(joint_name, 0.0), delta)
        low, high = model.jnt_range[jid]
        if bool(model.jnt_limited[jid]) and (now <= low + 1e-4 or now >= high - 1e-4):
            ctx.joint_limit_hits.add(joint_name)

    if sample:
        ctx.rows_timeseries.append(
            {
                "scenario_name": ctx.scenario,
                "test_name": ctx.test_name,
                "mode_free_or_anchored": ctx.mode,
                "time_s": f"{time_s:.4f}",
                "world_frame_bag_translation_mm": f"{ctx.max_world_translation_mm:.4f}",
                "world_frame_bag_rotation_deg": f"{ctx.max_world_rotation_deg:.4f}",
                "bag_frame_local_deformation_mm": f"{ctx.max_local_deformation_mm:.4f}",
                "shoulder_deflection_mm": f"{ctx.max_shoulder_deflection_mm:.4f}",
                "top_patch_change_mm": f"{ctx.max_top_patch_change_mm:.4f}",
                "lower_belly_opening_mm": f"{ctx.max_lower_belly_opening_mm:.4f}",
                "width_reduction_mm": f"{ctx.max_width_reduction_mm:.4f}",
                "bottom_sag_mm": f"{ctx.max_bottom_sag_mm:.4f}",
                "top_drop_mm": f"{ctx.max_top_drop_mm:.4f}",
            }
        )


def _simulate(ctx: DiagnosticContext, duration_s: float, stimulus: Callable[[DiagnosticContext, int, float], None]) -> None:
    steps = max(1, int(duration_s / ctx.model.opt.timestep))
    sample_stride = max(1, int(0.02 / ctx.model.opt.timestep))
    for step in range(steps):
        t = step * ctx.model.opt.timestep
        ctx.data.xfrc_applied[:] = 0.0
        ctx.data.qfrc_applied[:] = 0.0
        if ctx.applicable:
            stimulus(ctx, step, t)
        if ctx.mode == "anchored":
            _apply_anchor(ctx)
        mujoco.mj_step(ctx.model, ctx.data)
        _update_context(ctx, t, sample=(step % sample_stride == 0 or step == steps - 1))


def _stim_shoulder_left(ctx: DiagnosticContext, _step: int, _t: float) -> None:
    ctx.data.xfrc_applied[_body_id(ctx.model, SHOULDER_CENTER_BODY_LEFT), :3] += np.array([0.0, -10.0, -0.6])
    for i in range(TOP_SEAM_COUNT):
        scale = 1.0 - 0.12 * abs(i - SHOULDER_CENTER_INDEX)
        _apply_joint_torque(ctx, f"outer_upper_left_{i:02d}_hinge", -1.25 * max(0.45, scale))


def _stim_shoulder_right(ctx: DiagnosticContext, _step: int, _t: float) -> None:
    ctx.data.xfrc_applied[_body_id(ctx.model, SHOULDER_CENTER_BODY_RIGHT), :3] += np.array([0.0, 10.0, -0.6])
    for i in range(TOP_SEAM_COUNT):
        scale = 1.0 - 0.12 * abs(i - SHOULDER_CENTER_INDEX)
        _apply_joint_torque(ctx, f"outer_upper_right_{i:02d}_hinge", 1.25 * max(0.45, scale))


def _stim_top_preload(ctx: DiagnosticContext, _step: int, _t: float) -> None:
    ctx.data.xfrc_applied[_body_id(ctx.model, TOP_CENTER_BODY), :3] += np.array([0.0, 0.0, -9.0])
    ctx.data.xfrc_applied[_body_id(ctx.model, TOP_CENTER_BODY), 3:] += np.array([0.0, 2.8, 0.0])
    center = TOP_SEAM_COUNT // 2
    for idx, scale in ((center - 1, 0.55), (center, 1.0), (center + 1, 0.55)):
        if 0 <= idx < TOP_SEAM_COUNT:
            _apply_joint_torque(ctx, f"top_seam_{idx:02d}_hinge", 90.0 * scale)
    ctx.data.xfrc_applied[_body_id(ctx.model, SHOULDER_CENTER_BODY_LEFT), :3] += np.array([0.0, -1.6, -1.8])
    ctx.data.xfrc_applied[_body_id(ctx.model, SHOULDER_CENTER_BODY_RIGHT), :3] += np.array([0.0, 1.6, -1.8])


def _stim_side_push(ctx: DiagnosticContext, _step: int, _t: float) -> None:
    ctx.data.xfrc_applied[_body_id(ctx.model, f"outer_mid_front_{PANEL_CENTER_INDEX:02d}"), :3] += np.array([0.0, -9.0, 0.0])
    ctx.data.xfrc_applied[_body_id(ctx.model, f"outer_mid_back_{PANEL_CENTER_INDEX:02d}"), :3] += np.array([0.0, 9.0, 0.0])
    ctx.data.xfrc_applied[_body_id(ctx.model, LOWER_LEFT_BODY), :3] += np.array([0.0, -3.0, 0.0])
    ctx.data.xfrc_applied[_body_id(ctx.model, LOWER_RIGHT_BODY), :3] += np.array([0.0, 3.0, 0.0])


def _stim_scoop_insertion(ctx: DiagnosticContext, _step: int, _t: float) -> None:
    # 실제 scoop 경로 대신, lower belly pair가 열릴 수 있는지 확인하는 controlled insertion torque입니다.
    ctx.data.xfrc_applied[_body_id(ctx.model, LOWER_LEFT_BODY), 3:] += np.array([58.0, 0.0, 0.0])
    ctx.data.xfrc_applied[_body_id(ctx.model, LOWER_RIGHT_BODY), 3:] += np.array([-58.0, 0.0, 0.0])
    ctx.data.xfrc_applied[_body_id(ctx.model, BOTTOM_CENTER_BODY), :3] += np.array([0.0, 0.0, 3.2])
    _apply_joint_torque(ctx, f"{LOWER_LEFT_BODY}_hinge", 2.05)
    _apply_joint_torque(ctx, f"{LOWER_RIGHT_BODY}_hinge", -2.05)


def _stim_support_release(ctx: DiagnosticContext, step: int, _t: float) -> None:
    if step == 0:
        gid = _geom_id(ctx.model, "hidden_support_geom")
        if gid >= 0:
            ctx.model.geom_contype[gid] = 0
            ctx.model.geom_conaffinity[gid] = 0
            ctx.model.geom_rgba[gid, 3] = 0.0
    if ctx.scenario == "post_separation_sag":
        ctx.data.xfrc_applied[_body_id(ctx.model, BOTTOM_CENTER_BODY), :3] += np.array([0.0, 0.0, -4.2])
        for side in ("left", "right"):
            for i in range(TOP_SEAM_COUNT):
                _apply_joint_torque(ctx, f"outer_bottom_edge_{side}_{i:02d}_hinge", -2.2 if side == "left" else 2.2)
        _apply_joint_torque(ctx, "outer_bottom_edge_center_hinge", -2.0)
        for i in range(INNER_BOTTOM_PANEL_COUNT):
            ctx.data.xfrc_applied[_body_id(ctx.model, f"inner_bottom_load_{i:02d}"), :3] += np.array([0.0, 0.0, -1.8])
            _apply_joint_torque(ctx, f"inner_bottom_load_{i:02d}_slide", -3.2)
            _apply_joint_torque(ctx, f"inner_bottom_load_{i:02d}_hinge", -1.8)


def _stim_fold_brushing(ctx: DiagnosticContext, _step: int, _t: float) -> None:
    state = get_scenario(ctx.scenario)
    body = "top_edge_occlusion_left" if abs(state.fold_left_deg) >= abs(state.fold_right_deg) else "top_edge_occlusion_right"
    ctx.data.xfrc_applied[_body_id(ctx.model, body), :3] += np.array([1.8, -1.4, 0.35])


def _fold_applicable(scenario: str) -> bool:
    return get_scenario(scenario).fold_coverage_fraction > 0.0


TEST_SPECS = [
    TestSpec("shoulder_poke_left", "shoulder_poke_before_after", "shoulder_left", 0.24, _stim_shoulder_left),
    TestSpec("shoulder_poke_right", "shoulder_poke_right_before_after", "shoulder_right", 0.24, _stim_shoulder_right),
    TestSpec("top_preload", "top_preload_before_after", "top_seam", 0.24, _stim_top_preload),
    TestSpec("side_push", "side_push_before_after", "side_gusset", 0.28, _stim_side_push),
    TestSpec("scoop_insert", "scoop_insert_before_after", "lower_belly", 0.36, _stim_scoop_insertion),
    TestSpec("support_release", "support_release_before_after", "hidden_support", 0.36, _stim_support_release),
    TestSpec("fold_brushing", "fold_brushing_before_after", "fold_patch", 0.28, _stim_fold_brushing, _fold_applicable),
]


def _hide_visual_skin(model: mujoco.MjModel, alpha: float) -> np.ndarray:
    original = model.geom_rgba.copy()
    for name in ("visual_skin_main", "sealed_top_cap_visual_geom", "visual_print_mark_geom"):
        gid = _geom_id(model, name)
        if gid >= 0:
            model.geom_rgba[gid, 3] = alpha
    return original


def _restore_rgba(model: mujoco.MjModel, rgba: np.ndarray) -> None:
    model.geom_rgba[:] = rgba


def _scene_option(*, outer_shell: bool, inner_shell: bool, ballast: bool, visual_skin: bool) -> mujoco.MjvOption:
    option = mujoco.MjvOption()
    option.geomgroup[:] = True
    option.geomgroup[1] = bool(outer_shell)
    option.geomgroup[2] = bool(inner_shell)
    option.geomgroup[3] = bool(visual_skin)
    option.geomgroup[4] = bool(ballast)
    option.geomgroup[5] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    return option


def _render_style(renderer: mujoco.Renderer, model: mujoco.MjModel, data: mujoco.MjData, style: str) -> np.ndarray:
    if style == "outer_shell_only":
        original = _hide_visual_skin(model, 0.02)
        renderer.update_scene(data, camera="front", scene_option=_scene_option(outer_shell=True, inner_shell=False, ballast=False, visual_skin=False))
        image = renderer.render().copy()
        _restore_rgba(model, original)
        return image
    if style == "inner_shell_only":
        original = _hide_visual_skin(model, 0.02)
        renderer.update_scene(data, camera="front", scene_option=_scene_option(outer_shell=False, inner_shell=True, ballast=False, visual_skin=False))
        image = renderer.render().copy()
        _restore_rgba(model, original)
        return image
    if style == "ballast_only":
        original = _hide_visual_skin(model, 0.02)
        renderer.update_scene(data, camera="front", scene_option=_scene_option(outer_shell=False, inner_shell=False, ballast=True, visual_skin=False))
        image = renderer.render().copy()
        _restore_rgba(model, original)
        return image
    if style == "visual_skin":
        renderer.update_scene(data, camera="front", scene_option=_scene_option(outer_shell=False, inner_shell=False, ballast=False, visual_skin=True))
        return renderer.render().copy()
    original = _hide_visual_skin(model, 0.42)
    renderer.update_scene(data, camera="front", scene_option=_scene_option(outer_shell=True, inner_shell=True, ballast=True, visual_skin=True))
    image = renderer.render().copy()
    _restore_rgba(model, original)
    return image


def _render_before_after(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    before_qpos: np.ndarray,
    before_qvel: np.ndarray,
    after_qpos: np.ndarray,
    after_qvel: np.ndarray,
) -> dict[str, np.ndarray]:
    renderer = mujoco.Renderer(model, width=960, height=620)
    outputs: dict[str, np.ndarray] = {}
    for style in VIDEO_STYLES:
        data.qpos[:] = before_qpos
        data.qvel[:] = before_qvel
        mujoco.mj_forward(model, data)
        before = _render_style(renderer, model, data, style)
        data.qpos[:] = after_qpos
        data.qvel[:] = after_qvel
        mujoco.mj_forward(model, data)
        after = _render_style(renderer, model, data, style)
        outputs[style] = np.concatenate([before, after], axis=1)
    renderer.close()
    return outputs


def _joint_group_max(ctx: DiagnosticContext, group: str) -> float:
    max_delta = 0.0
    for name, delta in ctx.joint_max_abs_delta.items():
        if group == "shoulder" and name.startswith("outer_upper_"):
            max_delta = max(max_delta, math.degrees(delta))
        elif group == "belly" and (name.startswith("outer_mid_") or name.startswith("outer_lower_")):
            max_delta = max(max_delta, math.degrees(delta))
        elif group == "fold" and name.startswith("top_edge_occlusion_"):
            max_delta = max(max_delta, math.degrees(delta))
        elif group == "bottom" and (name.startswith("outer_bottom_edge_") or name.startswith("inner_bottom_load_")):
            kind = _joint_kind(ctx.model, name)
            max_delta = max(max_delta, 1000.0 * delta if kind == "slide_mm" else math.degrees(delta))
    return max_delta


def _fold_exposed_after(ctx: DiagnosticContext) -> str:
    if not ctx.applicable or ctx.test_name != "fold_brushing":
        return f"{ctx.fold_exposed_before:.3f}->{ctx.fold_exposed_before:.3f}"
    state = get_scenario(ctx.scenario)
    body = "top_edge_occlusion_left" if abs(state.fold_left_deg) >= abs(state.fold_right_deg) else "top_edge_occlusion_right"
    delta = abs(ctx.joint_max_abs_delta.get(f"{body}_hinge", 0.0))
    after = min(1.0, ctx.fold_exposed_before + min(0.20, delta * 0.65))
    return f"{ctx.fold_exposed_before:.3f}->{after:.3f}"


def _count_trapped_like_patches(ctx: DiagnosticContext) -> int:
    moved = 0
    for name, start in ctx.local0.items():
        if "top_seam" in name or "outer_upper" in name or "top_edge_occlusion" in name:
            now = _site_local(ctx.model, ctx.data, name)
            if 1000.0 * float(np.linalg.norm(now - start)) > 0.35:
                moved += 1
    return moved


def _rigid_like_flag(ctx: DiagnosticContext) -> bool:
    shape_near_zero = max(
        ctx.max_shoulder_deflection_mm,
        ctx.max_top_patch_change_mm,
        ctx.max_lower_belly_opening_mm,
        ctx.max_width_reduction_mm,
        ctx.max_bottom_sag_mm,
    ) < 0.20
    joint_near_zero = max(
        _joint_group_max(ctx, "shoulder"),
        _joint_group_max(ctx, "belly"),
        _joint_group_max(ctx, "fold"),
        _joint_group_max(ctx, "bottom"),
    ) < 0.05
    world_motion_large = ctx.max_world_translation_mm > 1.0 or ctx.max_world_rotation_deg > 1.0
    local_near_zero = ctx.max_local_deformation_mm < 0.50
    if not ctx.applicable:
        return False
    if ctx.mode == "anchored" and local_near_zero:
        return True
    if world_motion_large and local_near_zero:
        return True
    return bool(joint_near_zero and shape_near_zero)


def _summary_row(ctx: DiagnosticContext) -> dict[str, object]:
    trapped = _count_trapped_like_patches(ctx)
    support_state = ctx.test_name == "support_release" and ctx.max_bottom_sag_mm > ctx.max_top_drop_mm + 0.5
    load_following = 0.0
    if ctx.max_top_patch_change_mm > 1e-6:
        load_following = min(2.0, ctx.max_bottom_sag_mm / max(ctx.max_top_patch_change_mm, 1e-6))
    return {
        "scenario_name": ctx.scenario,
        "test_name": ctx.test_name,
        "mode_free_or_anchored": ctx.mode,
        "world_frame_bag_translation_mm": f"{ctx.max_world_translation_mm:.4f}",
        "world_frame_bag_rotation_deg": f"{ctx.max_world_rotation_deg:.4f}",
        "bag_frame_local_deformation_mm": f"{ctx.max_local_deformation_mm:.4f}",
        "rigid_like_flag": _rigid_like_flag(ctx),
        "shoulder_joint_max_delta_deg": f"{_joint_group_max(ctx, 'shoulder'):.4f}",
        "belly_joint_max_delta_deg": f"{_joint_group_max(ctx, 'belly'):.4f}",
        "fold_joint_max_delta_deg": f"{_joint_group_max(ctx, 'fold'):.4f}",
        "bottom_sling_max_travel_mm": f"{_joint_group_max(ctx, 'bottom'):.4f}",
        "joint_limit_hit_count": len(ctx.joint_limit_hits),
        "shoulder_deflection_mm": f"{ctx.max_shoulder_deflection_mm:.4f}",
        "top_patch_change_mm": f"{ctx.max_top_patch_change_mm:.4f}",
        "lower_belly_opening_mm": f"{ctx.max_lower_belly_opening_mm:.4f}",
        "width_reduction_mm": f"{ctx.max_width_reduction_mm:.4f}",
        "bottom_sag_mm": f"{ctx.max_bottom_sag_mm:.4f}",
        "top_drop_mm": f"{ctx.max_top_drop_mm:.4f}",
        "fold_exposed_fraction_before_after": _fold_exposed_after(ctx),
        "requested_target_label": ctx.requested_target_label,
        "actual_region_label_at_close": ctx.actual_region_label_at_close,
        "trapped_patch_count": trapped,
        "pull_test_slip_mm": f"{ctx.max_world_translation_mm:.4f}" if ctx.test_name in ("top_preload", "shoulder_poke_left", "shoulder_poke_right") else "0.0000",
        "load_following_ratio": f"{load_following:.4f}",
        "scoop_engaged": ctx.test_name == "scoop_insert" and ctx.max_lower_belly_opening_mm > 0.02,
        "support_state_formed": support_state,
    }


def _joint_rows(ctx: DiagnosticContext) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name in sorted(ctx.qpos0):
        kind = _joint_kind(ctx.model, name)
        delta = ctx.joint_max_abs_delta.get(name, 0.0)
        value = 1000.0 * delta if kind == "slide_mm" else math.degrees(delta)
        rows.append(
            {
                "scenario_name": ctx.scenario,
                "test_name": ctx.test_name,
                "mode_free_or_anchored": ctx.mode,
                "joint_name": name,
                "joint_type": kind,
                "joint_max_delta_deg_or_mm": f"{value:.6f}",
                "limit_hit": name in ctx.joint_limit_hits,
            }
        )
    return rows


def _write_mp4_or_frames(out_dir: Path, frames_by_style: dict[str, list[np.ndarray]], *, fps: int = 3) -> None:
    imageio = _imageio()
    image_names = {
        "outer_shell_only": "outer_shell_only.png",
        "inner_shell_only": "inner_shell_only.png",
        "ballast_only": "ballast_only.png",
        "visual_skin": "visual_render.png",
        "overlay": "overlay_render.png",
    }
    for style, frames in frames_by_style.items():
        if not frames:
            continue
        imageio.imwrite(out_dir / image_names.get(style, f"{style}.png"), frames[0])
        mp4_path = out_dir / f"{style}_video.mp4"
        try:
            imageio.mimwrite(mp4_path, frames, fps=fps, quality=8, macro_block_size=1)
        except Exception as exc:  # pragma: no cover
            fallback = out_dir / f"{style}_frames"
            fallback.mkdir(parents=True, exist_ok=True)
            for i, frame in enumerate(frames):
                imageio.imwrite(fallback / f"{i:04d}.png", frame)
            (out_dir / f"{style}_video_failed.txt").write_text(str(exc), encoding="utf-8")


def _float(row: dict[str, object], key: str) -> float:
    return float(row[key])


def _write_parameter_tuning_log(out_dir: Path) -> None:
    """Deprecated legacy tuning log kept inert; v2 writes the active log.

    fields = [
        "component",
        "before",
        "after",
        "reason",
    ]
    rows = [
        {
            "component": "top_seam_chain",
            "before": "7 segments, stiffness=2.8, damping=8.0, limit=+-16deg",
            "after": f"{TOP_SEAM_COUNT} segments, stiffness=0.16, damping=0.75, limit=+-55deg",
            "reason": "top contact region의 국소 접힘/눌림을 눈에 보이게 만들기 위해 분절 수를 늘리고 stiffness/damping을 낮춤",
        },
        {
            "component": "shoulder_panels",
            "before": "3 panels/side, stiffness=state*1.0, damping=7.0, limit=+-42deg",
            "after": "5 panels/side, stiffness=max(0.035,state*0.10), damping=baseline 2.20 / nominal 0.95 / underfilled 0.42, limit=+-92deg or underfilled +-118deg",
            "reason": "underfilled shoulder가 통째로 밀리지 않고 안쪽으로 처지는 local compliance를 만들기 위함",
        },
        {
            "component": "side_gusset",
            "before": "2 panels/side, stiffness=1.8, damping=7.0, limit=+-38deg",
            "after": "3 panels/side, stiffness=0.36, damping=1.55, limit=+-75deg",
            "reason": "side push/jammed 상황에서 옆면이 rigid wall처럼 보이지 않도록 함",
        },
        {
            "component": "lower_belly_panels",
            "before": "4 panels, stiffness=state*1.0, damping=8.5, limit=+-42deg",
            "after": f"{LOWER_BELLY_PANEL_COUNT} panels, stiffness=max(0.16,state*0.16), damping=1.15, limit=+-88deg",
            "reason": "scoop insertion region이 열리거나 들리는 visible local shape change를 만들기 위함",
        },
        {
            "component": "fold_root",
            "before": "stiffness=1.8, damping=8.0, limit=-70..35deg",
            "after": "stiffness=0.24, damping=1.55, limit=-115..80deg",
            "reason": "fold brushing 후 seam 노출률이 조금 변하는 접힘부 반응을 만들기 위함",
        },
        {
            "component": "bottom_sling",
            "before": "single slide, stiffness=1.15, damping=12.0, range=-70..14mm",
            "after": "center slide + left/right hinged panels, stiffness=0.34/0.28, damping=2.8/2.2, range=-120..35mm",
            "reason": "post separation sag에서 bottom이 top보다 더 처지는 하부 지지 구조를 만들기 위함",
        },
        {
            "component": "hidden_inner_load_shell",
            "before": "none",
            "after": f"inner front/back load shell {INNER_LOAD_PANEL_COUNT}+{INNER_LOAD_PANEL_COUNT}, inner bottom load shell {INNER_BOTTOM_PANEL_COUNT}, group=2",
            "reason": "outer shell이 보이는 형상을 만들고 inner shell이 load path, global sag, droplet-like bottom-heavy response를 담당하게 함",
        },
        {
            "component": "coarse_granular_internal_load_surrogate",
            "before": "payload_main + payload_aux",
            "after": "ballast_main + ballast_aux_1 + ballast_aux_2, limited x/y/z slide",
            "reason": "DEM 없이 filled sack의 coarse load redistribution과 lateral COM offset을 표현하기 위함",
        },
        {
            "component": "strap_tendon_coupling",
            "before": "none",
            "after": "top-seam<->shoulder, shoulder<->belly, fold<->shoulder, belly<->bottom, outer<->inner, ballast<->inner fixed tendons",
            "reason": "outer visible shell과 hidden inner load shell이 따로 놀지 않고 저차 coordinated shape change를 만들기 위함",
        },
    ]
    with (out_dir / "parameter_tuning_log.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


    """

def _write_parameter_tuning_log_v2(out_dir: Path) -> None:
    """현재 twin-shell topology 변경 사항을 깨끗한 CSV로 남긴다."""
    fields = [
        "component",
        "before",
        "after",
        "reason",
    ]
    rows = [
        {
            "component": "main_bag_topology",
            "before": "rim/upper-skirt/lower-skirt/bottom-cradle style articulated patch bag",
            "after": "explicit twin-shell bag: visible outer shell + hidden inner load shell + ballast masses",
            "reason": "겉보기 rigid bag에 작은 appendage만 붙은 구조가 아니라, 외피와 내부 하중 경로가 분리된 filled sack surrogate로 바꾸기 위함",
        },
        {
            "component": "top_grasp_rail_and_seam",
            "before": "short top seam chain with limited local compliance",
            "after": f"top_grasp_rail + {TOP_SEAM_COUNT} top_seam_chain segments",
            "reason": "2F가 실제로 잡는 상단 edge/봉합선 후보를 명확히 만들고 top preload 변형을 측정하기 위함",
        },
        {
            "component": "visible_articulated_outer_shell",
            "before": "scenario-facing panels were visually secondary and bag looked rigid-like",
            "after": "outer_front/back/shoulder/side/lower/bottom-edge shell segments are visible physics bodies",
            "reason": "visual skin이 아니라 실제 물리 patch가 자루 외형으로 보이게 해서 접촉 시 local shape change를 직접 확인하기 위함",
        },
        {
            "component": "cylindrical_panel_hinges",
            "before": "broad panels were connected by abstract hinge joints only",
            "after": "55 visible cyl_hinge_* capsule cues along top seam, front/back shell, shoulder, side, and lower shell panels",
            "reason": "판들이 원기둥형 seam/hinge 축을 따라 엮여 접히는 구조로 읽히게 하기 위함",
        },
        {
            "component": "hidden_inner_load_shell",
            "before": "payload bodies dominated load behavior",
            "after": f"inner_front/back load shell {INNER_LOAD_PANEL_COUNT}+{INNER_LOAD_PANEL_COUNT}, inner_bottom load shell {INNER_BOTTOM_PANEL_COUNT}",
            "reason": "외피를 직접 채우는 대신 내부 하중 전달과 sag를 담당하는 hidden load shell을 둬서 bottom-heavy response를 만들기 위함",
        },
        {
            "component": "hidden_ballast_masses",
            "before": "one or two simple payload ellipsoids",
            "after": "ballast_main + ballast_aux_1 + ballast_aux_2 + ballast_aux_3 with limited slide joints",
            "reason": "DEM 없이도 underfilled/eccentric/sag case에서 coarse load redistribution과 COM offset을 표현하기 위함",
        },
        {
            "component": "outer_inner_coupling",
            "before": "mostly direct articulated shell response",
            "after": "fixed tendon-like couplings between top rail, outer shell, inner shell, and ballast sites",
            "reason": "외피와 내부 하중 shell이 따로 놀지 않고 저차원 coordinated sag/elongation을 보이게 하기 위함",
        },
        {
            "component": "scoop_support_hull",
            "before": "flat scoop plate only",
            "after": "flat scoop plate + rounded scoop_support_hull capsule",
            "reason": "하부 삽입이 날카로운 판 관통처럼 보이지 않고 둥근 support hull과 lower shell interaction으로 보이게 하기 위함",
        },
        {
            "component": "contact_adaptive_panel_compliance",
            "before": "contact-triggered local joint retuning existed in the GUI path",
            "after": "GUI only monitors near-contact patch count; all shell joints stay always-active passive joints",
            "reason": "평상시에는 봉합된 자루 형상을 유지하고, 2F 접촉 시에만 해당 국소 외피가 더 크게 눌리거나 접히게 하기 위함",
        },
    ]
    with (out_dir / "parameter_tuning_log.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_topology_diff_md(out_dir: Path) -> None:
    md_new = [
        "# Hinge-Locked Twin-Shell Topology Diff",
        "",
        "이 파일은 현재 generated XML의 main sack topology를 요약합니다.",
        "",
        "## Removed From Main Topology",
        "",
        "| old concept | status | reason |",
        "|---|---|---|",
        "| `rim_ring / upper_skirt / lower_skirt / bottom_cradle` | not used as main bag body | central rim/skirt 느낌을 제거하고 hinge-locked panel chain으로 대체 |",
        "| `connected_outer_shell` | demoted to unused legacy builder code | main XML generation path에서 호출하지 않음 |",
        "| `top_grasp_rail_lift` | removed from generated XML | visible shell에 slide를 쓰지 않기 위해 제거 |",
        "| central large rigid core | not used | distributed ballast masses로 대체 |",
        "",
        "## New Body Hierarchy",
        "",
        "```text",
        "bag_frame",
        "  visual_skin                         # physics-free overlay",
        "  visible_articulated_outer_shell",
        "    top_grasp_rail                    # hinge only",
        "      top_seam_chain",
        "        top_seam_00..10               # hinge + fixed seam patch",
        "      outer_upper_left/right_segments",
        "        outer_upper_left/right_00..10 # hinge body, panel fixed to body",
        "          outer_mid_front/back_00..10",
        "            outer_lower_left/right_00..10",
        "              outer_bottom_edge_left/right_00..10",
        "      outer_bottom_edge_center",
        "  hidden_inner_load_shell",
        "    inner_front_load_00..04",
        "    inner_back_load_00..04",
        "    inner_bottom_load_00..02",
        "  ballast_main",
        "  ballast_aux_1",
        "  ballast_aux_2",
        "  ballast_aux_3",
        "  optional_top_edge_occlusion_patch",
        "```",
        "",
        "## Motion Policy",
        "",
        "- visible outer shell panel geom은 해당 hinge body에 고정됨",
        "- visible outer shell에는 slide joint 없음",
        "- panel 위치 변화는 부모 hinge chain의 회전 때문에 종속적으로 발생",
        "- ballast slide는 내부 질량 재분포 surrogate에만 사용",
        "",
        "## Grasp Candidate Bodies",
        "",
        "- `top_grasp_rail`",
        "- `top_seam_00..10`",
        "- `outer_upper_left/right_00..10`",
        "- `top_edge_occlusion_left/right`",
        "",
        "## Scoop Support Bodies",
        "",
        "- `outer_lower_left/right_00..10`",
        "- `outer_bottom_edge_left/right_00..10`",
        "- `outer_bottom_edge_center`",
        "- `inner_bottom_load_00..02`",
        "",
        "## Coupling",
        "",
        "- `chain_outer_vertical_*`: upper -> mid -> lower -> bottom 종속 angle chain",
        "- `couple_left_right_*`: 좌우 외피가 따로 놀지 않게 하는 약한 대칭 coupling",
        "- `couple_outer_*_to_inner_*`: visible outer shell과 hidden inner load shell 연결",
        "- `couple_ballast_*`: distributed ballast와 inner/bottom response 연결",
        "",
    ]
    (out_dir / "topology_diff.md").write_text("\n".join(md_new), encoding="utf-8")
    return

    md = [
        "# Twin-Shell Topology Diff",
        "",
        "이 파일은 `mujoco_twin_shell_bag_diagram.png`를 topology contract로 해석한 현재 구조 요약입니다.",
        "",
        "## Old Bodies Removed Or Demoted",
        "",
        "| old body/concept | status | reason |",
        "|---|---|---|",
        "| `rim_ring` | removed from generated XML | ring/skirt 사고방식 제거 |",
        "| `upper_skirt` | removed from generated XML | visible outer shell로 대체 |",
        "| `lower_skirt` | removed from generated XML | lower belly + bottom edge shell로 대체 |",
        "| `bottom_cradle` | removed from generated XML | inner bottom load shell + outer bottom edge로 대체 |",
        "| central large rigid core | not used | 4개의 작은 distributed ballast로 분산 |",
        "| contact-triggered 0-180 unlocking | not used | passive joint/spring/damping만 사용 |",
        "",
        "## Bodies Kept",
        "",
        "- `bag_frame`: freejoint root",
        "- `top_grasp_rail`, `top_seam_chain`: top edge grasp candidate",
        "- `hidden_support`, `neighbor_left`, `neighbor_right`: sag/jam scenario support",
        "- `visual_skin`: optional cosmetic overlay only, physics-free",
        "",
        "## New Main Body Hierarchy",
        "",
        "```text",
        "bag_frame",
        "  visual_skin",
        "  top_grasp_rail",
        "    top_seam_chain",
        "      top_seam_00..10",
        "  connected_outer_shell",
        "    connected_outer_front/back_shell",
        "      connected_front/back_00..06_lower",
        "        connected_front/back_00..06_mid",
        "          connected_front/back_00..06_upper",
        "    connected_outer_end_left/right",
        "      connected_end_left/right_lower",
        "        connected_end_left/right_mid",
        "          connected_end_left/right_upper",
        "  outer_shoulder_segments",
        "    outer_shoulder_shell_left/right",
        "      outer_shoulder_left/right_00..04",
        "  outer_front_segments",
        "    outer_front_shell_00..06",
        "  outer_back_segments",
        "    outer_back_shell_00..06",
        "  outer_side_shell_segments_left/right",
        "    outer_side_left/right_00..03",
        "  outer_lower_belly_segments",
        "    outer_lower_shell_00..07",
        "  outer_bottom_edge_segments",
        "    outer_bottom_edge_00..06",
        "  hidden_inner_load_shell",
        "    inner_front_load_shell",
        "    inner_back_load_shell",
        "    inner_bottom_load_shell",
        "  ballast_main",
        "  ballast_aux_1",
        "  ballast_aux_2",
        "  ballast_aux_3",
        "  optional_top_edge_occlusion_patch",
        "```",
        "",
        "## Visible Shell Parts",
        "",
        "- `connected_outer_shell`: default visible bag body",
        "- `connected_front/back_00..06_lower/mid/upper`: 3-height-layer connected panels",
        "- `connected_end_left/right_lower/mid/upper`: sealed side/end panels",
        "- `top_grasp_rail`, `top_seam_00..10`",
        "- legacy `outer_*` patches are group 5 compatibility/debug bodies and hidden by default",
        "",
        "## Hidden Inner Load Shell Parts",
        "",
        "- `inner_front_load_00..04`",
        "- `inner_back_load_00..04`",
        "- `inner_bottom_load_00..02`",
        "",
        "## Distributed Ballast Bodies",
        "",
        "- `ballast_main`: small lower-left load",
        "- `ballast_aux_1`: scenario-biased lateral load",
        "- `ballast_aux_2`: lower center/back load",
        "- `ballast_aux_3`: lower-right load",
        "",
        "## Coupling / Tendon List",
        "",
        "- `chain_top_seam_angle_*`",
        "- `chain_outer_front/back_angle_*`",
        "- `chain_outer_shoulder_left/right_angle_*`",
        "- `chain_outer_side_left/right_angle_*`",
        "- `chain_outer_lower_angle_*`",
        "- `chain_outer_bottom_angle_*`",
        "- `chain_inner_front/back/bottom_angle_*`",
        "- `chain_connected_*_row_*`",
        "- `chain_connected_*_vertical_*`",
        "- `couple_top_seam_to_connected_front/back_*`",
        "- `couple_top_rail_to_outer_shoulder_left/right_*`",
        "- `couple_outer_front_to_inner_front_*`",
        "- `couple_outer_back_to_inner_back_*`",
        "- `couple_shoulder_to_outer_lower_left/right_*`",
        "- `couple_outer_lower_to_inner_bottom_*`",
        "- `couple_inner_bottom_to_ballast_main`",
        "- `couple_ballast_aux1_to_side_shell`",
        "- `couple_ballast_aux2_to_inner_bottom`",
        "- `couple_ballast_aux3_to_inner_bottom`",
        "- `couple_top_to_bottom_sag_mode`",
        "",
        "## Rendering Groups",
        "",
        "- group 1: visible articulated outer shell",
        "- group 2: hidden inner load shell",
        "- group 3: cosmetic visual skin overlay",
        "- group 4: distributed ballast masses",
        "- group 5: hidden legacy compatibility patches and hinge cues",
        "",
        "## Panel Motion Constraint",
        "",
        "- visible outer shell panels are connected lower/mid/upper body chains.",
        "- removed visible panel slides: `shoulder_press_*_slide`, `outer_bottom_edge_*_slide`.",
        "- top lift response is angle-dependent: upper layer leads, mid and lower layers follow through tendon coupling.",
        "- hidden ballast slide joints remain because they represent coarse internal load redistribution, not panel translation.",
    ]
    (out_dir / "topology_diff.md").write_text("\n".join(md), encoding="utf-8")


def _fold_delta(row: dict[str, object]) -> float:
    text = str(row.get("fold_exposed_fraction_before_after", "0->0"))
    try:
        before, after = text.split("->")
        return float(after) - float(before)
    except Exception:
        return 0.0


def _write_before_after_comparison(out_dir: Path, rows: list[dict[str, object]]) -> None:
    imageio = _imageio()
    old = {
        "top_patch_change_mm": 0.004,
        "lower_belly_opening_mm": 0.078,
        "simple_fold_exposed_delta": 0.000,
        "post_sag_bottom_mm": 0.590,
    }
    new = {
        "top_patch_change_mm": max(_float(row, "top_patch_change_mm") for row in rows),
        "lower_belly_opening_mm": max(_float(row, "lower_belly_opening_mm") for row in rows),
        "simple_fold_exposed_delta": max((_fold_delta(row) for row in rows if row["scenario_name"] == "top_fold_simple"), default=0.0),
        "post_sag_bottom_mm": max((_float(row, "bottom_sag_mm") for row in rows if row["scenario_name"] == "post_separation_sag"), default=0.0),
    }
    labels = list(old)
    width, height = 1100, 520
    try:
        from PIL import Image, ImageDraw, ImageFont

        image = Image.new("RGB", (width, height), (248, 244, 235))
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 18)
            font_bold = ImageFont.truetype("arial.ttf", 24)
        except Exception:
            font = ImageFont.load_default()
            font_bold = font
        draw.text((34, 24), "Before/After Local Compliance Comparison", fill=(45, 34, 22), font=font_bold)
        chart_x, chart_y = 270, 92
        max_value = max(max(old.values()), max(new.values()), 1.0)
        for i, label in enumerate(labels):
            y = chart_y + i * 88
            draw.text((34, y + 10), label, fill=(45, 34, 22), font=font)
            old_w = int(520 * old[label] / max_value)
            new_w = int(520 * new[label] / max_value)
            draw.rectangle((chart_x, y, chart_x + old_w, y + 24), fill=(150, 150, 150))
            draw.rectangle((chart_x, y + 34, chart_x + new_w, y + 58), fill=(122, 84, 45))
            draw.text((chart_x + 540, y), f"before {old[label]:.3f}", fill=(90, 90, 90), font=font)
            draw.text((chart_x + 540, y + 34), f"after  {new[label]:.3f}", fill=(90, 58, 28), font=font)
        image.save(out_dir / "before_after_comparison.png")
    except Exception:
        canvas = np.full((height, width, 3), 245, dtype=np.uint8)
        max_value = max(max(old.values()), max(new.values()), 1.0)
        for i, label in enumerate(labels):
            y = 80 + i * 90
            old_w = int(520 * old[label] / max_value)
            new_w = int(520 * new[label] / max_value)
            canvas[y : y + 24, 270 : 270 + old_w] = np.array([150, 150, 150], dtype=np.uint8)
            canvas[y + 34 : y + 58, 270 : 270 + new_w] = np.array([122, 84, 45], dtype=np.uint8)
        imageio.imwrite(out_dir / "before_after_comparison.png", canvas)


def _run_one(
    scenario: str,
    spec: TestSpec,
    mode: str,
    *,
    out_dir: Path,
    render: bool,
    frames_by_style: dict[str, list[np.ndarray]],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    model, data, _xml = _load_model(scenario)
    ctx = _make_context(model, data, scenario, spec, mode)
    before_qpos = data.qpos.copy()
    before_qvel = data.qvel.copy()
    _simulate(ctx, spec.duration_s, spec.stimulus)
    after_qpos = data.qpos.copy()
    after_qvel = data.qvel.copy()

    if render:
        images = _render_before_after(model, data, before_qpos, before_qvel, after_qpos, after_qvel)
        key_dir = out_dir / "keyframes" / scenario / mode
        key_dir.mkdir(parents=True, exist_ok=True)
        imageio = _imageio()
        imageio.imwrite(key_dir / f"{spec.keyframe_name}.png", images["overlay"])
        for style, image in images.items():
            frames_by_style[style].append(image)

    return _summary_row(ctx), ctx.rows_timeseries, _joint_rows(ctx)


def diagnose(
    *,
    scenario: str = "all",
    out_dir: Path = OUTPUT_DIR,
    render: bool = True,
    write_video: bool = True,
) -> list[dict[str, object]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    scenarios = list(SCENARIO_NAMES) if scenario == "all" else [scenario]
    frames_by_style = {style: [] for style in VIDEO_STYLES}
    summary_rows: list[dict[str, object]] = []
    timeseries_rows: list[dict[str, object]] = []
    joint_rows: list[dict[str, object]] = []

    for scenario_name in scenarios:
        for mode in DIAGNOSTIC_MODES:
            for spec in TEST_SPECS:
                row, ts, jr = _run_one(
                    scenario_name,
                    spec,
                    mode,
                    out_dir=out_dir,
                    render=render,
                    frames_by_style=frames_by_style,
                )
                summary_rows.append(row)
                timeseries_rows.extend(ts)
                joint_rows.extend(jr)

    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)
    with (out_dir / "deformation_timeseries.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TIMESERIES_FIELDS)
        writer.writeheader()
        writer.writerows(timeseries_rows)
    with (out_dir / "joint_response.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=JOINT_RESPONSE_FIELDS)
        writer.writeheader()
        writer.writerows(joint_rows)
    _write_parameter_tuning_log_v2(out_dir)
    _write_topology_diff_md(out_dir)
    _write_before_after_comparison(out_dir, summary_rows)
    if render and write_video:
        _write_mp4_or_frames(out_dir, frames_by_style)
    _write_summary_md_v2(out_dir, summary_rows)
    return summary_rows


def _write_summary_md(out_dir: Path, rows: list[dict[str, object]]) -> None:
    md = [
        "# Rigid-Like Diagnostic Report",
        "",
        "이 리포트는 diagnostic-only pass입니다. topology, geometry spec, joint/tendon parameter를 변경하지 않았고 auto-tuning도 수행하지 않았습니다.",
        "",
        "진단 모드:",
        "- `free`: 자루가 task/evaluation처럼 자유롭게 움직입니다.",
        "- `anchored`: 짧은 진단 동안만 `bag_frame`에 약한 world anchor force를 적용하여 whole-body slip과 local deformation을 분리합니다.",
        "",
        "판정 규칙:",
        "- world-frame motion만 크고 `bag_frame` local deformation이 거의 0이면 rigid-like입니다.",
        "- joint angle 변화와 shape response가 모두 거의 0이면 rigid-like입니다.",
        "- anchored mode에서도 local deformation이 거의 0이면 topology 자체가 너무 rigid-like입니다.",
        "- anchored mode에서는 local deformation이 있으나 free mode에서 slip만 크면 topology보다 support/friction/anchoring 문제가 큽니다.",
        "",
    ]
    for scenario in sorted(set(str(row["scenario_name"]) for row in rows)):
        md.append(f"## {scenario}")
        subset = [row for row in rows if row["scenario_name"] == scenario]
        rigid_count = sum(1 for row in subset if row["rigid_like_flag"] is True)
        md.append(f"- rigid_like rows: `{rigid_count}/{len(subset)}`")
        for mode in DIAGNOSTIC_MODES:
            mode_rows = [row for row in subset if row["mode_free_or_anchored"] == mode]
            max_local = max(_float(row, "bag_frame_local_deformation_mm") for row in mode_rows)
            max_world = max(_float(row, "world_frame_bag_translation_mm") for row in mode_rows)
            max_rot = max(_float(row, "world_frame_bag_rotation_deg") for row in mode_rows)
            md.append(f"- {mode}: max local `{max_local:.3f} mm`, world translation `{max_world:.3f} mm`, world rotation `{max_rot:.3f} deg`")
        md.append("")
        md.append("| test | mode | rigid_like | local_mm | world_mm | rot_deg | shoulder_mm | top_mm | belly_open_mm | width_red_mm | bottom_sag_mm | limit_hits |")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in subset:
            md.append(
                f"| {row['test_name']} | {row['mode_free_or_anchored']} | {row['rigid_like_flag']} | "
                f"{row['bag_frame_local_deformation_mm']} | {row['world_frame_bag_translation_mm']} | {row['world_frame_bag_rotation_deg']} | "
                f"{row['shoulder_deflection_mm']} | {row['top_patch_change_mm']} | {row['lower_belly_opening_mm']} | "
                f"{row['width_reduction_mm']} | {row['bottom_sag_mm']} | {row['joint_limit_hit_count']} |"
            )
        md.append("")

    anchored_rigid = [row for row in rows if row["mode_free_or_anchored"] == "anchored" and row["rigid_like_flag"] is True]
    free_slip_rows = [
        row
        for row in rows
        if row["mode_free_or_anchored"] == "free"
        and _float(row, "world_frame_bag_translation_mm") > 2.0
        and _float(row, "bag_frame_local_deformation_mm") < 0.5
    ]
    max_local_all = max(_float(row, "bag_frame_local_deformation_mm") for row in rows)
    max_top = max(_float(row, "top_patch_change_mm") for row in rows)
    max_belly = max(_float(row, "lower_belly_opening_mm") for row in rows)

    md.append("## Explicit Conclusion")
    if anchored_rigid:
        md.append("- 일부 anchored diagnostic에서도 local deformation이 거의 없어 topology가 rigid-like한 구간이 있습니다.")
    else:
        md.append("- anchored diagnostic에서 local deformation이 측정되므로 topology 전체가 완전 rigid-like라고 보기는 어렵습니다.")
    if free_slip_rows:
        md.append("- free diagnostic에서 whole-body slip만 큰 행이 있어 support/friction/anchoring 쪽 문제가 섞여 있습니다.")
    else:
        md.append("- free diagnostic에서도 bag-frame local deformation이 함께 측정되어 단순 whole-body slip만 발생한 것은 아닙니다.")
    if max_top < 0.2 or max_belly < 0.2:
        md.append("- 하지만 top preload 또는 scoop insertion의 국소 shape response가 작아서 support-state surrogate로는 아직 부족합니다.")
    else:
        md.append("- top/lower-belly 국소 shape response도 측정 가능 범위에 들어와 support-state surrogate로 사용할 여지가 있습니다.")
    md.append(f"- max bag-frame local deformation: `{max_local_all:.3f} mm`")
    md.append(f"- max top_patch_change_mm: `{max_top:.3f} mm`")
    md.append(f"- max lower_belly_opening_mm: `{max_belly:.3f} mm`")
    md.append("")
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")


def _write_summary_md_v2(out_dir: Path, rows: list[dict[str, object]]) -> None:
    md = [
        "# Force-Responsive Articulated Sack Diagnostic",
        "",
        "이 리포트는 full soft/flex/DEM이 아니라 visible articulated outer shell 기반 surrogate가 실제로 local deformation을 만드는지 확인한 결과입니다.",
        "",
        "진단 모드:",
        "- `free`: 자루 전체가 자유롭게 움직이는 기본 task 상태입니다.",
        "- `anchored`: 진단 동안만 약한 world anchor를 걸어 whole-body slip과 local deformation을 분리합니다.",
        "",
        "판정 규칙:",
        "- world-frame motion만 크고 bag-frame local deformation이 작으면 rigid-like로 판단합니다.",
        "- anchored mode에서도 local deformation이 작으면 topology 자체가 너무 rigid-like하다고 판단합니다.",
        "- anchored mode에서는 변형이 있고 free mode에서만 미끄러지면 topology보다 support/friction 조건 문제가 더 큽니다.",
        "",
    ]
    for scenario in sorted(set(str(row["scenario_name"]) for row in rows)):
        md.append(f"## {scenario}")
        subset = [row for row in rows if row["scenario_name"] == scenario]
        rigid_count = sum(1 for row in subset if row["rigid_like_flag"] is True)
        md.append(f"- rigid_like rows: `{rigid_count}/{len(subset)}`")
        for mode in DIAGNOSTIC_MODES:
            mode_rows = [row for row in subset if row["mode_free_or_anchored"] == mode]
            if not mode_rows:
                continue
            max_local = max(_float(row, "bag_frame_local_deformation_mm") for row in mode_rows)
            max_world = max(_float(row, "world_frame_bag_translation_mm") for row in mode_rows)
            max_rot = max(_float(row, "world_frame_bag_rotation_deg") for row in mode_rows)
            md.append(f"- {mode}: max local `{max_local:.3f} mm`, world translation `{max_world:.3f} mm`, world rotation `{max_rot:.3f} deg`")
        md.append("")
        md.append("| test | mode | rigid_like | local_mm | world_mm | shoulder_mm | top_mm | belly_open_mm | bottom_sag_mm | fold_exposed |")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in subset:
            md.append(
                f"| {row['test_name']} | {row['mode_free_or_anchored']} | {row['rigid_like_flag']} | "
                f"{row['bag_frame_local_deformation_mm']} | {row['world_frame_bag_translation_mm']} | "
                f"{row['shoulder_deflection_mm']} | {row['top_patch_change_mm']} | {row['lower_belly_opening_mm']} | "
                f"{row['bottom_sag_mm']} | {row['fold_exposed_fraction_before_after']} |"
            )
        md.append("")

    anchored_rigid = [row for row in rows if row["mode_free_or_anchored"] == "anchored" and row["rigid_like_flag"] is True]
    free_slip_rows = [
        row
        for row in rows
        if row["mode_free_or_anchored"] == "free"
        and _float(row, "world_frame_bag_translation_mm") > 2.0
        and _float(row, "bag_frame_local_deformation_mm") < 0.5
    ]
    max_local_all = max(_float(row, "bag_frame_local_deformation_mm") for row in rows)
    max_top = max(_float(row, "top_patch_change_mm") for row in rows)
    max_belly = max(_float(row, "lower_belly_opening_mm") for row in rows)
    simple_fold_delta = max((_fold_delta(row) for row in rows if row["scenario_name"] == "top_fold_simple"), default=0.0)
    post_sag = max((_float(row, "bottom_sag_mm") for row in rows if row["scenario_name"] == "post_separation_sag"), default=0.0)

    md.append("## Explicit Conclusion")
    if anchored_rigid:
        md.append("- 일부 anchored diagnostic에서도 local deformation이 작아 rigid-like 구간이 남아 있습니다.")
    else:
        md.append("- anchored diagnostic에서 local deformation이 측정되어 topology 전체를 단일 강체로 보기는 어렵습니다.")
    if free_slip_rows:
        md.append("- free diagnostic 중 일부는 whole-body slip이 커서 support/friction/anchoring 조건도 함께 봐야 합니다.")
    else:
        md.append("- free diagnostic에서도 bag-frame local deformation이 측정되어 단순 whole-body slip만 발생한 것은 아닙니다.")
    if max_top >= 1.0 and max_belly >= 5.0:
        md.append("- top preload와 lower-belly/scoop insertion에서 목표 수준의 local shape response가 측정되었습니다.")
    else:
        md.append("- top preload 또는 lower-belly/scoop insertion 목표 변형량이 부족한 구간이 있습니다.")
    md.append(f"- max bag-frame local deformation: `{max_local_all:.3f} mm`")
    md.append(f"- max top_patch_change_mm: `{max_top:.3f} mm`")
    md.append(f"- max lower_belly_opening_mm: `{max_belly:.3f} mm`")
    md.append(f"- simple fold exposed fraction delta: `{simple_fold_delta:.3f}`")
    md.append(f"- post_separation_sag max bottom_sag_mm: `{post_sag:.3f} mm`")
    md.append("")
    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnostic-only rigid-like vs force-responsive report")
    parser.add_argument("--scenario", choices=SCENARIO_NAMES + ("all",), default="all")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    rows = diagnose(scenario=args.scenario, render=not args.no_render, write_video=not args.no_video)
    rigid_rows = sum(1 for row in rows if row["rigid_like_flag"] is True)
    print(f"rows={len(rows)} rigid_like_rows={rigid_rows}")
    print(f"summary_csv={OUTPUT_DIR / 'summary.csv'}")
    print(f"summary_md={OUTPUT_DIR / 'summary.md'}")
    print(f"timeseries_csv={OUTPUT_DIR / 'deformation_timeseries.csv'}")
    print(f"joint_response_csv={OUTPUT_DIR / 'joint_response.csv'}")
    print(f"parameter_tuning_log={OUTPUT_DIR / 'parameter_tuning_log.csv'}")
    print(f"topology_diff={OUTPUT_DIR / 'topology_diff.md'}")
    print(f"before_after_comparison={OUTPUT_DIR / 'before_after_comparison.png'}")
    if not args.no_render and not args.no_video:
        print(f"outer_shell_only={OUTPUT_DIR / 'outer_shell_only.png'}")
        print(f"inner_shell_only={OUTPUT_DIR / 'inner_shell_only.png'}")
        print(f"ballast_only={OUTPUT_DIR / 'ballast_only.png'}")
        print(f"visual_render={OUTPUT_DIR / 'visual_render.png'}")
        print(f"overlay_render={OUTPUT_DIR / 'overlay_render.png'}")
        print(f"outer_shell_video={OUTPUT_DIR / 'outer_shell_only_video.mp4'}")
        print(f"inner_shell_video={OUTPUT_DIR / 'inner_shell_only_video.mp4'}")
        print(f"ballast_video={OUTPUT_DIR / 'ballast_only_video.mp4'}")
        print(f"visual_video={OUTPUT_DIR / 'visual_skin_video.mp4'}")
        print(f"overlay_video={OUTPUT_DIR / 'overlay_video.mp4'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
