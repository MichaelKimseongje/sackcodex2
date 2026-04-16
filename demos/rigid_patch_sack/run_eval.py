"""2F gripper로 rigid patch sack의 top-region graspability를 평가한다."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np

from build_sack_surrogate import SEGMENT_COUNT, write_scene_xml
from scenario_builder import SCENARIO_NAMES, get_scenario

ROOT_DIR = Path(__file__).resolve().parent
OUT_DIR = ROOT_DIR / "out"

REQUESTED_TARGETS = ("auto", "all", "seam", "rim", "fold", "shoulder")
MODES = ("contact_only_eval", "qualification_gated_connect", "visual_demo")
LOAD_FOLLOW_THRESHOLD = 0.0


@dataclass(frozen=True)
class CandidatePatch:
    """파지 후보는 한 점이 아니라 작은 rigid/articulated patch 단위로 다룬다."""

    label: str
    site_name: str
    body_names: tuple[str, ...]
    rank_bias: float


@dataclass
class ConnectState:
    """qualification 통과 후 선택 patch에만 약한 spring-damper force를 건다."""

    active: bool
    body_ids: list[int]
    offsets: list[np.ndarray]
    activation_time: float


def _ensure_imageio():
    try:
        import imageio.v2 as imageio
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("이미지 저장에는 imageio가 필요합니다. `pip install imageio`를 확인해 주세요.") from exc
    return imageio


def _mocap_id(model: mujoco.MjModel, body_name: str) -> int:
    body_id = model.body(body_name).id
    mocap_id = int(model.body_mocapid[body_id])
    if mocap_id < 0:
        raise ValueError(f"{body_name}은 mocap body가 아닙니다.")
    return mocap_id


def _geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""


def _body_pos(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    return data.xpos[model.body(body_name).id].copy()


def _site_pos(model: mujoco.MjModel, data: mujoco.MjData, site_name: str) -> np.ndarray:
    return data.site_xpos[model.site(site_name).id].copy()


def _set_gripper(model: mujoco.MjModel, data: mujoco.MjData, center: np.ndarray, gap: float) -> None:
    """두 finger pad가 y축 방향으로 열리고 닫히도록 mocap 위치를 직접 지정한다."""
    left_id = _mocap_id(model, "gripper_left_mocap")
    right_id = _mocap_id(model, "gripper_right_mocap")
    center_id = _mocap_id(model, "gripper_center_mocap")
    data.mocap_pos[left_id] = center + np.array([0.0, -0.5 * gap, 0.0])
    data.mocap_pos[right_id] = center + np.array([0.0, 0.5 * gap, 0.0])
    data.mocap_pos[center_id] = center
    data.mocap_quat[left_id] = [1.0, 0.0, 0.0, 0.0]
    data.mocap_quat[right_id] = [1.0, 0.0, 0.0, 0.0]
    data.mocap_quat[center_id] = [1.0, 0.0, 0.0, 0.0]


def _current_gripper(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, float]:
    left = data.mocap_pos[_mocap_id(model, "gripper_left_mocap")]
    right = data.mocap_pos[_mocap_id(model, "gripper_right_mocap")]
    return 0.5 * (left + right), float(np.linalg.norm(right - left))


def _contact_flags(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[bool, bool]:
    """좌/우 pad가 sack patch와 실제 contact를 만들었는지 확인한다."""
    left_pad = "gripper_left_mocap_pad"
    right_pad = "gripper_right_mocap_pad"
    target_prefixes = ("seam_band", "shoulder_panel", "belly_panel", "fold_root_flap", "bottom_sling", "payload")
    left_contact = False
    right_contact = False
    for idx in range(data.ncon):
        contact = data.contact[idx]
        name1 = _geom_name(model, int(contact.geom1))
        name2 = _geom_name(model, int(contact.geom2))
        pair = {name1, name2}
        other_names = (name2 if name1 in (left_pad, right_pad) else name1,)
        if left_pad in pair and any(name.startswith(target_prefixes) for name in other_names):
            left_contact = True
        if right_pad in pair and any(name.startswith(target_prefixes) for name in other_names):
            right_contact = True
    return left_contact, right_contact


def _candidate_patches(scenario_name: str, requested_target: str) -> list[CandidatePatch]:
    """scenario와 target selector에 맞는 후보 patch를 만든다."""
    config = get_scenario(scenario_name)
    candidates: list[CandidatePatch] = []
    if requested_target in ("auto", "all", "seam", "rim"):
        rim_bias = 1.0
        if scenario_name.startswith("top_fold"):
            rim_bias = 2.6
        for idx in range(SEGMENT_COUNT):
            candidates.append(CandidatePatch("seam", f"grasp_seam_{idx:02d}", (f"seam_band_{idx:02d}",), rim_bias))
    if requested_target in ("auto", "all", "shoulder"):
        shoulder_bias = 0.4 if scenario_name == "underfilled" else 1.4
        for idx in range(SEGMENT_COUNT):
            candidates.append(
                CandidatePatch(
                    "shoulder",
                    f"grasp_shoulder_{idx:02d}",
                    (f"shoulder_panel_{idx:02d}", f"belly_panel_{idx:02d}"),
                    shoulder_bias,
                )
            )
    if requested_target in ("auto", "all", "fold"):
        fold_bias = 0.25 if scenario_name.startswith("top_fold") else 4.0
        if config.fold1_enabled:
            candidates.append(CandidatePatch("fold", "grasp_fold_1", ("fold_root_flap_1",), fold_bias))
        if config.fold2_enabled:
            candidates.append(CandidatePatch("fold", "grasp_fold_2", ("fold_root_flap_2",), fold_bias + 0.15))
    return candidates


def _rank_candidates(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    candidates: Iterable[CandidatePatch],
    scenario_name: str,
) -> list[CandidatePatch]:
    """후보를 label rule이 아니라 접근성/높이/현상별 prior로 정렬한다."""
    scored: list[tuple[float, CandidatePatch]] = []
    for patch in candidates:
        pos = _site_pos(model, data, patch.site_name)
        y_clearance = 0.25 * abs(float(pos[1]))
        high_patch_bonus = -0.15 * float(pos[2])
        shoulder_side_bonus = 0.0
        if scenario_name == "underfilled" and patch.label == "shoulder":
            shoulder_side_bonus = -0.35 * min(abs(float(pos[0])) / 0.10, 1.0)
        fold_root_bonus = 0.0
        if scenario_name.startswith("top_fold") and patch.label == "fold":
            fold_root_bonus = -0.25
        score = patch.rank_bias + y_clearance + high_patch_bonus + shoulder_side_bonus + fold_root_bonus
        scored.append((score, patch))
    scored.sort(key=lambda item: item[0])
    return [patch for _, patch in scored]


def _all_patch_body_labels(scenario_name: str) -> dict[str, str]:
    config = get_scenario(scenario_name)
    labels: dict[str, str] = {}
    for idx in range(SEGMENT_COUNT):
        labels[f"seam_band_{idx:02d}"] = "seam"
        labels[f"shoulder_panel_{idx:02d}"] = "shoulder"
    if config.fold1_enabled:
        labels["fold_root_flap_1"] = "fold"
    if config.fold2_enabled:
        labels["fold_root_flap_2"] = "fold"
    return labels


def _capture_metrics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scenario_name: str,
    center: np.ndarray,
    gap: float,
) -> dict[str, object]:
    """jaw capture zone 안에 실제로 들어온 top-region patch를 계산한다."""
    body_labels = _all_patch_body_labels(scenario_name)
    trapped: list[tuple[str, str, np.ndarray]] = []
    for body_name, label in body_labels.items():
        pos = _body_pos(model, data, body_name)
        inside_x = abs(float(pos[0] - center[0])) <= 0.070
        inside_y = abs(float(pos[1] - center[1])) <= 0.5 * gap + 0.026
        inside_z = abs(float(pos[2] - center[2])) <= 0.080
        if inside_x and inside_y and inside_z:
            trapped.append((body_name, label, pos))

    counts: dict[str, int] = {}
    for _, label, _ in trapped:
        counts[label] = counts.get(label, 0) + 1
    actual = "none"
    if counts:
        actual = max(counts.items(), key=lambda item: item[1])[0]

    left_contact, right_contact = _contact_flags(model, data)
    if left_contact and right_contact:
        balance = 1.0
    elif left_contact or right_contact:
        balance = 0.50
    elif trapped:
        ys = np.array([pos[1] - center[1] for _, _, pos in trapped])
        neg = int(np.sum(ys < 0.0))
        pos_count = int(np.sum(ys >= 0.0))
        balance = 1.0 - abs(neg - pos_count) / max(len(trapped), 1)
    else:
        balance = 0.0

    return {
        "left_contact_present": left_contact,
        "right_contact_present": right_contact,
        "trapped_patch_count": len(trapped),
        "trapped_body_names": [body for body, _, _ in trapped],
        "actual_region_label_at_close": actual,
        "bilateral_contact_balance": float(balance),
        "jaw_escape": len(trapped) == 0,
    }


def _patch_mean_pos(model: mujoco.MjModel, data: mujoco.MjData, body_names: Iterable[str]) -> np.ndarray:
    positions = []
    for body_name in body_names:
        if model.body(body_name).id >= 0:
            positions.append(_body_pos(model, data, body_name))
    if not positions:
        return np.zeros(3)
    return np.mean(np.vstack(positions), axis=0)


def _support_reference_z(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """하중이 따라오는지 보기 위해 cradle과 payload의 평균 z를 쓴다."""
    names = ("bottom_sling", "payload_main")
    return float(np.mean([_body_pos(model, data, name)[2] for name in names]))


def _apply_soft_connect(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    connect: ConnectState,
    center: np.ndarray,
    *,
    kp: float = 2600.0,
    kd: float = 34.0,
    max_force: float = 180.0,
) -> None:
    """선택된 patch 1~2개에만 약한 connect surrogate force를 적용한다."""
    data.xfrc_applied[:, :] = 0.0
    if not connect.active:
        return
    n = max(len(connect.body_ids), 1)
    for body_id, offset in zip(connect.body_ids, connect.offsets):
        target = center + offset
        pos = data.xpos[body_id]
        vel = data.cvel[body_id, 3:6]
        force = kp * (target - pos) - kd * vel
        norm = float(np.linalg.norm(force))
        if norm > max_force:
            force *= max_force / norm
        data.xfrc_applied[body_id, 0:3] += force / n


def _step_controlled(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    center: np.ndarray,
    gap: float,
    steps: int,
    connect: ConnectState | None = None,
) -> None:
    for _ in range(max(1, steps)):
        _set_gripper(model, data, center, gap)
        if connect is not None:
            _apply_soft_connect(model, data, connect, center)
        else:
            data.xfrc_applied[:, :] = 0.0
        mujoco.mj_step(model, data)


def _move_gripper(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_center: np.ndarray,
    target_gap: float,
    seconds: float,
    connect: ConnectState | None = None,
    frames: list[np.ndarray] | None = None,
    renderer: mujoco.Renderer | None = None,
) -> None:
    start_center, start_gap = _current_gripper(model, data)
    steps = max(1, int(seconds / model.opt.timestep))
    frame_stride = max(1, steps // 12)
    for step in range(steps):
        alpha = (step + 1) / steps
        center = (1.0 - alpha) * start_center + alpha * target_center
        gap = (1.0 - alpha) * start_gap + alpha * target_gap
        _set_gripper(model, data, center, float(gap))
        if connect is not None:
            _apply_soft_connect(model, data, connect, center)
        else:
            data.xfrc_applied[:, :] = 0.0
        mujoco.mj_step(model, data)
        if frames is not None and renderer is not None and len(frames) < 60 and step % frame_stride == 0:
            renderer.update_scene(data, camera="overview")
            frames.append(renderer.render().copy())


def _render_stage(renderer: mujoco.Renderer, data: mujoco.MjData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    renderer.update_scene(data, camera="overview")
    _ensure_imageio().imwrite(path, renderer.render())


def _write_frame_sequence(frames: list[np.ndarray], out_dir: Path) -> None:
    if not frames:
        return
    imageio = _ensure_imageio()
    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames[:60]):
        imageio.imwrite(frame_dir / f"frame_{idx:03d}.png", frame)
    try:
        imageio.mimsave(out_dir / "sequence.mp4", frames[:60], fps=30)
    except Exception:
        # mp4 codec이 없으면 frame sequence만 남긴다.
        pass


def _write_summary_md(row: dict[str, object], out_dir: Path) -> None:
    lines = [
        "# Rigid Patch Sack Grasp Summary",
        "",
        f"- mode: `{row['mode']}`",
        f"- scenario: `{row['scenario_name']}`",
        f"- requested_target_label: `{row['requested_target_label']}`",
        f"- actual_region_label_at_close: `{row['actual_region_label_at_close']}`",
        f"- trapped_patch_count: `{row['trapped_patch_count']}`",
        f"- connect_activated: `{row['connect_activated']}`",
        f"- drop_or_not: `{row['drop_or_not']}`",
        f"- no_graspable_patch_found: `{row['no_graspable_patch_found']}`",
        "",
        "이 결과는 full soft sack 마찰 증명이 아니라, task-driven local patch qualification 결과입니다.",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _write_summary_csv(row: dict[str, object], out_dir: Path) -> None:
    fields = [
        "scenario_name",
        "requested_target_label",
        "actual_region_label_at_close",
        "trapped_patch_count",
        "bilateral_contact_balance",
        "pull_test_slip_mm",
        "load_following_ratio",
        "connect_activated",
        "lift_height",
        "hold_time",
        "final_slip_distance",
        "drop_or_not",
        "no_graspable_patch_found",
    ]
    extra_fields = ["mode", "accepted_candidate_rank", "jaw_escape"]
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields + extra_fields)
        writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fields + extra_fields})


def run_trial(mode: str, scenario_name: str, requested_target: str, out_dir: Path) -> dict[str, object]:
    """한 scenario에서 하나의 자동 탐색 + close + tug + lift 평가를 수행한다."""
    xml_path = write_scene_xml(scenario_name)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=720, width=960)
    frames: list[np.ndarray] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    _set_gripper(model, data, np.array([0.0, 0.0, 0.315]), 0.18)
    mujoco.mj_forward(model, data)
    _step_controlled(model, data, np.array([0.0, 0.0, 0.315]), 0.18, int(0.45 / model.opt.timestep))
    _render_stage(renderer, data, out_dir / "initial.png")

    candidates = _rank_candidates(model, data, _candidate_patches(scenario_name, requested_target), scenario_name)
    if not candidates:
        row = {
            "mode": mode,
            "scenario_name": scenario_name,
            "requested_target_label": requested_target,
            "actual_region_label_at_close": "none",
            "trapped_patch_count": 0,
            "bilateral_contact_balance": 0.0,
            "pull_test_slip_mm": math.inf,
            "load_following_ratio": 0.0,
            "connect_activated": False,
            "lift_height": 0.0,
            "hold_time": 0.0,
            "final_slip_distance": math.inf,
            "drop_or_not": True,
            "no_graspable_patch_found": True,
            "accepted_candidate_rank": -1,
            "jaw_escape": True,
        }
        _write_summary_csv(row, out_dir)
        _write_summary_md(row, out_dir)
        renderer.close()
        return row

    selected = candidates[0]
    selected_rank = 0
    selected_pos = _site_pos(model, data, selected.site_name)
    approach_center = selected_pos + np.array([0.0, 0.0, 0.010])
    approach_center[2] = max(approach_center[2], 0.145)

    _move_gripper(model, data, approach_center, 0.145, 0.45, frames=frames, renderer=renderer)
    _move_gripper(model, data, approach_center, 0.030, 0.55, frames=frames, renderer=renderer)
    _step_controlled(model, data, approach_center, 0.030, int(0.20 / model.opt.timestep))
    _render_stage(renderer, data, out_dir / "grasp_close.png")
    _render_stage(renderer, data, out_dir / "close.png")

    close_center, close_gap = _current_gripper(model, data)
    close_metrics = _capture_metrics(model, data, scenario_name, close_center, close_gap)
    trapped_names = list(close_metrics["trapped_body_names"])
    # latch는 실제 jaw 안에 포획된 local patch에만 건다. 선택 후보가 빠져나갔으면 captured patch로 대체한다.
    selected_trapped = [name for name in selected.body_names if name in trapped_names]
    patch_names = selected_trapped[:2] if selected_trapped else trapped_names[:2]
    patch_close_pos = _patch_mean_pos(model, data, patch_names)
    support_start_z = _support_reference_z(model, data)

    tug_center = close_center + np.array([0.0, 0.0, 0.032])
    _move_gripper(model, data, tug_center, close_gap, 0.35, frames=frames, renderer=renderer)
    _step_controlled(model, data, tug_center, close_gap, int(0.10 / model.opt.timestep))
    _render_stage(renderer, data, out_dir / "grasp_tug_test.png")
    _render_stage(renderer, data, out_dir / "tug_test.png")

    tug_patch_pos = _patch_mean_pos(model, data, patch_names)
    support_after_z = _support_reference_z(model, data)
    tug_center_now, tug_gap_now = _current_gripper(model, data)
    pull_slip_mm = float(np.linalg.norm((tug_patch_pos - tug_center_now) - (patch_close_pos - close_center)) * 1000.0)
    gripper_dz = max(float(tug_center_now[2] - close_center[2]), 1e-6)
    load_following_ratio = float(np.clip((support_after_z - support_start_z) / gripper_dz, -1.0, 1.5))
    tug_metrics = _capture_metrics(model, data, scenario_name, tug_center_now, tug_gap_now)

    left_contact = bool(close_metrics["left_contact_present"] or tug_metrics["left_contact_present"])
    right_contact = bool(close_metrics["right_contact_present"] or tug_metrics["right_contact_present"])
    trapped_count = int(close_metrics["trapped_patch_count"])
    jaw_escape = bool(tug_metrics["jaw_escape"])
    relative_pose_ok = pull_slip_mm <= 45.0
    qualification_pass = (
        left_contact
        and right_contact
        and trapped_count >= 1
        and float(close_metrics["bilateral_contact_balance"]) >= 0.45
        and pull_slip_mm <= 55.0
        and load_following_ratio >= LOAD_FOLLOW_THRESHOLD
        and not jaw_escape
        and relative_pose_ok
    )

    connect_active = False
    connect = ConnectState(False, [], [], data.time)
    if mode == "visual_demo":
        # 시연용은 조건을 완화하지만, 선택 patch 1~2개만 약하게 따른다.
        connect_active = len(patch_names) > 0 and not jaw_escape
    elif mode == "qualification_gated_connect":
        connect_active = qualification_pass

    if connect_active:
        connect_body_ids = [model.body(name).id for name in patch_names[:2]]
        connect_offsets = [_body_pos(model, data, name) - tug_center_now for name in patch_names[:2]]
        connect = ConnectState(True, connect_body_ids, connect_offsets, float(data.time))
    _render_stage(renderer, data, out_dir / "latch_on.png")

    lift_start_patch = _patch_mean_pos(model, data, patch_names)
    lift_center = tug_center_now + np.array([0.0, 0.0, 0.055])
    _move_gripper(model, data, lift_center, tug_gap_now, 0.45, connect=connect, frames=frames, renderer=renderer)
    hold_time = 0.65
    _step_controlled(model, data, lift_center, tug_gap_now, int(hold_time / model.opt.timestep), connect=connect)
    _render_stage(renderer, data, out_dir / "grasp_lift.png")
    _render_stage(renderer, data, out_dir / "lift.png")

    final_patch_pos = _patch_mean_pos(model, data, patch_names)
    final_center, _ = _current_gripper(model, data)
    lift_height = float(final_patch_pos[2] - lift_start_patch[2])
    final_slip_distance = float(np.linalg.norm((final_patch_pos - final_center) - (patch_close_pos - close_center)))
    drop_or_not = bool(lift_height < 0.018 or final_slip_distance > 0.070)
    no_graspable = bool(not qualification_pass and mode != "visual_demo")

    row: dict[str, object] = {
        "mode": mode,
        "scenario_name": scenario_name,
        "requested_target_label": requested_target,
        "actual_region_label_at_close": close_metrics["actual_region_label_at_close"],
        "trapped_patch_count": trapped_count,
        "bilateral_contact_balance": float(close_metrics["bilateral_contact_balance"]),
        "pull_test_slip_mm": pull_slip_mm,
        "load_following_ratio": load_following_ratio,
        "connect_activated": connect_active,
        "lift_height": lift_height,
        "hold_time": hold_time,
        "final_slip_distance": final_slip_distance,
        "drop_or_not": drop_or_not,
        "no_graspable_patch_found": no_graspable,
        "accepted_candidate_rank": selected_rank,
        "jaw_escape": jaw_escape,
    }

    _write_frame_sequence(frames, out_dir)
    _write_summary_csv(row, out_dir)
    _write_summary_md(row, out_dir)
    renderer.close()
    return row


def _print_row(row: dict[str, object]) -> None:
    print(
        f"{row['mode']} | {row['scenario_name']} | "
        f"target={row['requested_target_label']} actual={row['actual_region_label_at_close']} "
        f"trapped={row['trapped_patch_count']} slip_mm={float(row['pull_test_slip_mm']):.1f} "
        f"load_follow={float(row['load_following_ratio']):.2f} "
        f"connect={row['connect_activated']} drop={row['drop_or_not']} "
        f"no_patch={row['no_graspable_patch_found']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default="contact_only_eval")
    parser.add_argument("--scenario", choices=[*SCENARIO_NAMES, "all"], default="underfilled")
    parser.add_argument("--target-label", choices=REQUESTED_TARGETS, default="auto")
    args = parser.parse_args()

    scenarios = list(SCENARIO_NAMES) if args.scenario == "all" else [args.scenario]
    rows = []
    for scenario_name in scenarios:
        out_dir = OUT_DIR / f"{args.mode}_{scenario_name}_{args.target_label}"
        row = run_trial(args.mode, scenario_name, args.target_label, out_dir)
        rows.append(row)
        _print_row(row)
        print(f"output_dir={out_dir}")

    aggregate = OUT_DIR / f"{args.mode}_summary.csv"
    fields = [
        "scenario_name",
        "requested_target_label",
        "actual_region_label_at_close",
        "trapped_patch_count",
        "bilateral_contact_balance",
        "pull_test_slip_mm",
        "load_following_ratio",
        "connect_activated",
        "lift_height",
        "hold_time",
        "final_slip_distance",
        "drop_or_not",
        "no_graspable_patch_found",
        "mode",
        "accepted_candidate_rank",
        "jaw_escape",
    ]
    with aggregate.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fields})
    print(f"aggregate_summary={aggregate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
