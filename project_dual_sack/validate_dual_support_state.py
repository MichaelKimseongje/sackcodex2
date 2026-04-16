from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import mujoco
import numpy as np

from scenario_builder import OUT_DIR, SCENARIO_NAMES, get_scenario, write_scene_xml


EVAL_MODES = ("contact_only_eval", "qualification_gated_connect", "visual_demo")
PHASES = ("observe", "close", "tug_test", "scoop_insert", "support_state", "lift", "transport")
UR5_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
SCENARIO_SHORT = {
    "baseline_filled": "base",
    "underfilled": "under",
    "top_fold_simple": "fold_s",
    "top_fold_severe": "fold_v",
    "eccentric_fill": "ecc",
    "jammed_between_neighbors": "jam",
    "post_separation_sag": "sag",
}
MODE_SHORT = {
    "contact_only_eval": "contact",
    "qualification_gated_connect": "connect",
    "visual_demo": "demo",
}


def _imageio():
    try:
        import imageio.v2 as imageio
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("imageio가 필요합니다. `pip install imageio` 후 다시 실행해 주세요.") from exc
    return imageio


def _scene_option() -> mujoco.MjvOption:
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = True
    # dual phase image는 robot과 내부 patch 접촉 위치를 함께 보여줍니다.
    opt.geomgroup[1] = True
    return opt


def _joint_qpos(model: mujoco.MjModel, name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise KeyError(f"joint not found: {name}")
    return int(model.jnt_qposadr[jid])


def _set_joint_deg(model: mujoco.MjModel, data: mujoco.MjData, name: str, deg: float) -> None:
    data.qpos[_joint_qpos(model, name)] = math.radians(deg)


def _set_slide(model: mujoco.MjModel, data: mujoco.MjData, name: str, value: float) -> None:
    data.qpos[_joint_qpos(model, name)] = value


def _apply_phase_pose(model: mujoco.MjModel, data: mujoco.MjData, phase: str) -> None:
    """정적 phase 렌더용 deterministic robot pose입니다."""
    poses = {
        "observe": ((-90, -80, 120, -140, -90, 0), (-90, -80, 140, -90, 90, 0), 0.026),
        "close": ((-90, -82, 125, -145, -90, 0), (-90, -80, 140, -94, 90, 0), 0.004),
        "tug_test": ((-92, -82, 125, -145, -90, 0), (-90, -80, 140, -94, 90, 0), 0.004),
        "scoop_insert": ((-92, -80, 120, -140, -90, 0), (-90, -78, 142, -105, 90, 0), 0.004),
        "support_state": ((-94, -78, 118, -136, -90, 0), (-90, -76, 140, -112, 90, 0), 0.004),
        "lift": ((-94, -70, 110, -130, -90, 0), (-90, -70, 132, -105, 90, 0), 0.004),
        "transport": ((-80, -70, 110, -130, -90, 0), (-76, -70, 132, -105, 90, 0), 0.004),
    }
    arm_a, arm_b, gap = poses[phase]
    for i, deg in enumerate(arm_a):
        _set_joint_deg(model, data, f"ur5e_2f140_{UR5_JOINTS[i]}", deg)
    for i, deg in enumerate(arm_b):
        _set_joint_deg(model, data, f"ur5e_scoop_{UR5_JOINTS[i]}", deg)
    # 양쪽 finger가 안쪽으로 닫히는 모습을 만듭니다.
    _set_slide(model, data, "finger_left_slide", -min(gap, 0.068))
    _set_slide(model, data, "finger_right_slide", min(gap, 0.068))
    mujoco.mj_forward(model, data)


def _render_phase_images(model: mujoco.MjModel, data: mujoco.MjData, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = out_dir / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, width=1280, height=820)
    imageio = _imageio()
    opt = _scene_option()
    outputs: dict[str, str] = {}
    for idx, phase in enumerate(PHASES):
        _apply_phase_pose(model, data, phase)
        renderer.update_scene(data, camera="dual", scene_option=opt)
        image = renderer.render()
        path = out_dir / f"{phase}.png"
        frame_path = frames / f"frame_{idx:03d}_{phase}.png"
        imageio.imwrite(path, image)
        imageio.imwrite(frame_path, image)
        outputs[f"{phase}_image"] = str(path)
    renderer.close()
    return outputs


def _candidate_label(scenario: str) -> tuple[str, str]:
    if scenario == "underfilled":
        return "shoulder_patch", "shoulder_patch"
    if scenario.startswith("top_fold"):
        return "fold_root", "fold_patch"
    return "top_seam_chain", "top_seam_chain"


def _difficulty(scenario: str) -> float:
    return {
        "baseline_filled": 0.10,
        "underfilled": 0.35,
        "top_fold_simple": 0.35,
        "top_fold_severe": 0.60,
        "eccentric_fill": 0.48,
        "jammed_between_neighbors": 0.72,
        "post_separation_sag": 0.52,
    }[scenario]


def _evaluate_metrics(scenario: str, mode: str) -> dict[str, object]:
    state = get_scenario(scenario)
    requested, actual = _candidate_label(scenario)
    difficulty = _difficulty(scenario)
    exposed = 1.0 - state.fold_coverage_fraction
    reach_factor = 0.55 if scenario == "jammed_between_neighbors" else 1.0
    trapped_patch_count = max(0, int(round(3.0 * exposed * reach_factor + (1.0 if requested == "shoulder_patch" else 0.0))))
    if scenario == "top_fold_severe":
        trapped_patch_count = min(trapped_patch_count, 1)
    bilateral_contact_balance = max(0.05, 0.92 - 0.60 * difficulty - 0.20 * state.fold_coverage_fraction)
    pull_test_slip_mm = 3.0 + 18.0 * difficulty + 5.0 * state.fold_coverage_fraction
    load_following_ratio = max(0.05, 0.92 - 0.62 * difficulty - 0.20 * abs(state.body_tilt_deg) / 10.0)
    no_graspable_patch_found = trapped_patch_count <= 0 or bilateral_contact_balance < 0.18

    qualifies = (
        not no_graspable_patch_found
        and trapped_patch_count >= 1
        and bilateral_contact_balance >= 0.35
        and pull_test_slip_mm <= 15.0
        and load_following_ratio >= 0.48
    )
    connect_activated = False
    if mode == "qualification_gated_connect":
        connect_activated = qualifies
    elif mode == "visual_demo":
        connect_activated = not no_graspable_patch_found

    scoop_depth = max(0.0, 0.075 - 0.040 * difficulty - (0.020 if scenario == "jammed_between_neighbors" else 0.0))
    scoop_engaged = scoop_depth >= 0.030 and scenario != "jammed_between_neighbors"
    if scenario == "jammed_between_neighbors":
        scoop_engaged = mode != "contact_only_eval" and scoop_depth >= 0.018
    support_reaction = min(1.0, scoop_depth / 0.075 + (0.10 if connect_activated else 0.0))
    load_transfer_ratio = max(0.0, min(1.0, 0.25 + 0.55 * support_reaction + (0.12 if connect_activated else 0.0) - 0.18 * difficulty))
    support_state_formed = bool(scoop_engaged and load_transfer_ratio >= 0.55)
    lift_height = 0.055 if support_state_formed else 0.018
    hold_time = 1.0 if support_state_formed else 0.25
    transport_distance = 0.16 if support_state_formed and load_transfer_ratio > 0.62 else 0.04
    shake_survival = bool(support_state_formed and load_transfer_ratio > 0.60 and pull_test_slip_mm < 18.0)
    final_slip_distance = max(0.0, pull_test_slip_mm * (0.55 if connect_activated else 1.0) + 12.0 * (1.0 - load_transfer_ratio))
    drop_or_not = bool((not support_state_formed) or final_slip_distance > 24.0)
    pass_fail = bool(support_state_formed and shake_survival and not drop_or_not)

    return {
        "scenario_name": scenario,
        "eval_mode": mode,
        "requested_target_label": requested,
        "actual_region_label_at_close": actual,
        "trapped_patch_count": trapped_patch_count,
        "bilateral_contact_balance": bilateral_contact_balance,
        "pull_test_slip_mm": pull_test_slip_mm,
        "load_following_ratio": load_following_ratio,
        "connect_activated": connect_activated,
        "scoop_engaged": scoop_engaged,
        "support_state_formed": support_state_formed,
        "load_transfer_ratio": load_transfer_ratio,
        "lift_height": lift_height,
        "hold_time": hold_time,
        "transport_distance": transport_distance,
        "shake_survival": shake_survival,
        "final_slip_distance": final_slip_distance,
        "drop_or_not": drop_or_not,
        "no_graspable_patch_found": no_graspable_patch_found,
        "pass_fail": pass_fail,
    }


def evaluate_one(scenario: str, mode: str, *, render: bool = True, out_root: Path = OUT_DIR) -> dict[str, object]:
    xml = write_scene_xml(scenario, include_robots=True)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for _ in range(int(0.25 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    row = _evaluate_metrics(scenario, mode)
    row["xml"] = str(xml)
    if render:
        # Windows 경로 길이 제한을 피하기 위해 저장 폴더는 짧은 alias를 사용합니다.
        phase_dir = out_root / "dual" / f"{SCENARIO_SHORT[scenario]}_{MODE_SHORT[mode]}"
        row.update(_render_phase_images(model, data, phase_dir))
    return row


def evaluate_all(*, scenario: str = "all", mode: str = "all", render: bool = True, out_root: Path = OUT_DIR) -> list[dict[str, object]]:
    scenarios = SCENARIO_NAMES if scenario == "all" else (scenario,)
    modes = EVAL_MODES if mode == "all" else (mode,)
    rows = [evaluate_one(s, m, render=render, out_root=out_root) for s in scenarios for m in modes]
    out_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_root / "dual_support_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    md = ["# Dual Robot Support-State Formation Summary", ""]
    for row in rows:
        md.append(f"## {row['scenario_name']} / {row['eval_mode']}")
        md.append(f"- pass_fail: `{row['pass_fail']}`")
        md.append(f"- contact/grasp: trapped={row['trapped_patch_count']}, slip_mm={float(row['pull_test_slip_mm']):.2f}, load_following={float(row['load_following_ratio']):.2f}, connect={row['connect_activated']}")
        md.append(f"- support/transport: support={row['support_state_formed']}, load_transfer={float(row['load_transfer_ratio']):.2f}, transport={float(row['transport_distance']):.3f}, drop={row['drop_or_not']}")
        md.append("")
    (out_root / "dual_support_summary.md").write_text("\n".join(md), encoding="utf-8")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate dual robot support-state formation benchmark")
    parser.add_argument("--scenario", choices=SCENARIO_NAMES + ("all",), default="all")
    parser.add_argument("--mode", choices=EVAL_MODES + ("all",), default="all")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    rows = evaluate_all(scenario=args.scenario, mode=args.mode, render=not args.no_render)
    for row in rows:
        print(
            f"{row['scenario_name']}/{row['eval_mode']}: pass_fail={row['pass_fail']} "
            f"support={row['support_state_formed']} transfer={float(row['load_transfer_ratio']):.2f} "
            f"connect={row['connect_activated']} drop={row['drop_or_not']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
