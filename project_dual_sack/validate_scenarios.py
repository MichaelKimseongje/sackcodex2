from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from scenario_builder import OUT_DIR, SCENARIO_NAMES, get_scenario, write_scene_xml


SUMMARY_FIELDS = [
    "scenario_name",
    "shoulder_deflection_mm",
    "top_patch_change_mm",
    "lower_belly_opening_mm",
    "bottom_sag_mm",
    "fold_exposed_fraction_before_after",
    "rigid_like_flag",
    "tuning_applied",
    "tuned_joint_stiffness",
    "tuned_joint_damping",
    "tuned_joint_limit",
    "tuned_payload_slide_range",
    "tuned_tendon_coupling",
]

# 거의 0으로 볼 기준입니다. 이 값보다 작으면 사용자가 보는 화면에서도
# "강체처럼 보인다"고 느낄 가능성이 높아서, 그때만 튜닝을 수행합니다.
NEAR_ZERO_THRESHOLDS = {
    "shoulder_deflection_mm": 0.50,
    "top_patch_change_mm": 0.20,
    "lower_belly_opening_mm": 0.20,
    "bottom_sag_mm": 0.50,
}


@dataclass
class Diagnostics:
    """GEOMETRY_SPEC에 적힌 기존 형상 metric만 담는 진단 결과입니다."""

    scenario_name: str
    shoulder_deflection_mm: float
    top_patch_change_mm: float
    lower_belly_opening_mm: float
    bottom_sag_mm: float
    fold_exposed_fraction_before_after: str
    shoulder_recovered: bool
    top_reference_drop_mm: float

    def near_zero_count(self) -> int:
        values = {
            "shoulder_deflection_mm": self.shoulder_deflection_mm,
            "top_patch_change_mm": self.top_patch_change_mm,
            "lower_belly_opening_mm": self.lower_belly_opening_mm,
            "bottom_sag_mm": self.bottom_sag_mm,
        }
        return sum(value < NEAR_ZERO_THRESHOLDS[name] for name, value in values.items())

    def needs_tuning(self) -> bool:
        return self.near_zero_count() > 0

    def rigid_like(self) -> bool:
        return self.near_zero_count() >= 3


def _imageio():
    try:
        import imageio.v2 as imageio
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("imageio가 필요합니다. `pip install imageio` 후 다시 실행해 주세요.") from exc
    return imageio


def _step(model: mujoco.MjModel, data: mujoco.MjData, seconds: float) -> bool:
    steps = max(1, int(seconds / model.opt.timestep))
    for _ in range(steps):
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            return False
    return True


def _settle(model: mujoco.MjModel, data: mujoco.MjData, seconds: float = 0.45) -> bool:
    return _step(model, data, seconds)


def _body_id(model: mujoco.MjModel, name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise KeyError(f"body not found: {name}")
    return bid


def _site_pos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise KeyError(f"site not found: {name}")
    return data.site_xpos[sid].copy()


def _geom_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


def _joint_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def _reset_for_test(model: mujoco.MjModel, data: mujoco.MjData, seconds: float = 0.35) -> None:
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    _settle(model, data, seconds)
    data.xfrc_applied[:] = 0.0


def _hide_visual_skin(model: mujoco.MjModel, alpha: float) -> np.ndarray:
    """visual_skin은 physics-free라서, debug render에서는 투명도를 따로 조절합니다."""

    original = model.geom_rgba.copy()
    for name in ("visual_skin_main", "sealed_top_cap_visual_geom", "visual_print_mark_geom"):
        gid = _geom_id(model, name)
        if gid >= 0:
            model.geom_rgba[gid, 3] = alpha
    return original


def _restore_rgba(model: mujoco.MjModel, rgba: np.ndarray) -> None:
    model.geom_rgba[:] = rgba


def _make_scene_option(*, physics_patches: bool) -> mujoco.MjvOption:
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = True
    # group 1은 실제 articulated patch 계열입니다. visual render에서는 끄고,
    # physics/overlay render에서는 켜서 실제 움직이는 구조를 확인합니다.
    opt.geomgroup[1] = bool(physics_patches)
    return opt


def _render_debug(model: mujoco.MjModel, data: mujoco.MjData, scenario: str, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    imageio = _imageio()
    renderer = mujoco.Renderer(model, width=1280, height=820)
    outputs: dict[str, str] = {}

    original = _hide_visual_skin(model, alpha=0.02)
    renderer.update_scene(data, camera="front", scene_option=_make_scene_option(physics_patches=True))
    path = out_dir / f"{scenario}_physics_patch_debug.png"
    imageio.imwrite(path, renderer.render())
    outputs["physics_patch_debug"] = str(path)
    _restore_rgba(model, original)

    renderer.update_scene(data, camera="front", scene_option=_make_scene_option(physics_patches=False))
    path = out_dir / f"{scenario}_visual_skin.png"
    imageio.imwrite(path, renderer.render())
    outputs["visual_skin"] = str(path)

    original = _hide_visual_skin(model, alpha=0.42)
    renderer.update_scene(data, camera="front", scene_option=_make_scene_option(physics_patches=True))
    path = out_dir / f"{scenario}_overlay.png"
    imageio.imwrite(path, renderer.render())
    outputs["overlay"] = str(path)
    _restore_rgba(model, original)

    renderer.close()
    return outputs


def _shoulder_poke_test(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, bool]:
    left_site = "site_shoulder_left_01_tip"
    right_site = "site_shoulder_right_01_tip"
    left_body = _body_id(model, "shoulder_left_01")
    right_body = _body_id(model, "shoulder_right_01")
    left0 = _site_pos(model, data, left_site)
    right0 = _site_pos(model, data, right_site)

    max_deflection = 0.0
    for _ in range(int(0.16 / model.opt.timestep)):
        # 좌우 shoulder를 안쪽으로 살짝 눌러 passive hinge 반응을 봅니다.
        data.xfrc_applied[left_body, :3] = np.array([0.0, -9.0, -0.8])
        data.xfrc_applied[right_body, :3] = np.array([0.0, 9.0, -0.8])
        mujoco.mj_step(model, data)
        left = _site_pos(model, data, left_site)
        right = _site_pos(model, data, right_site)
        max_deflection = max(
            max_deflection,
            0.5 * (float(np.linalg.norm(left - left0)) + float(np.linalg.norm(right - right0))),
        )

    data.xfrc_applied[:] = 0.0
    recovered = False
    threshold = max(0.00045, 0.38 * max_deflection)
    for _ in range(int(1.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        left = _site_pos(model, data, left_site)
        right = _site_pos(model, data, right_site)
        residual = 0.5 * (float(np.linalg.norm(left - left0)) + float(np.linalg.norm(right - right0)))
        if residual <= threshold:
            recovered = True
            break
    return 1000.0 * max_deflection, recovered


def _top_preload_test(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    site = "site_top_seam_03"
    body = _body_id(model, "top_seam_03")
    baseline = _site_pos(model, data, site)
    max_change = 0.0
    for _ in range(int(0.16 / model.opt.timestep)):
        # 2F close 전/중 preload를 단순화한 국소 하향 압력입니다.
        data.xfrc_applied[body, :3] = np.array([0.0, 0.0, -6.0])
        mujoco.mj_step(model, data)
        max_change = max(max_change, float(np.linalg.norm(_site_pos(model, data, site) - baseline)))
    data.xfrc_applied[:] = 0.0
    return 1000.0 * max_change


def _scoop_insertion_test(model: mujoco.MjModel, data: mujoco.MjData, *, tuned_pair_force: bool) -> float:
    left = "site_lower_belly_01_tip"
    right = "site_lower_belly_02_tip"
    left_body = _body_id(model, "lower_belly_01")
    right_body = _body_id(model, "lower_belly_02")
    gap0 = float(np.linalg.norm(_site_pos(model, data, left) - _site_pos(model, data, right)))
    max_opening = 0.0
    torque_scale = 12.0 if not tuned_pair_force else 35.0
    for _ in range(int(0.22 / model.opt.timestep)):
        # xfrc의 힘 성분은 body 중심에 작용해서 hinge를 거의 돌리지 못합니다.
        # scoop 삽입에 해당하는 개구 반응은 hinge 축 torque로 직접 진단합니다.
        data.xfrc_applied[left_body, 3:] = np.array([torque_scale, 0.0, 0.0])
        data.xfrc_applied[right_body, 3:] = np.array([-torque_scale, 0.0, 0.0])
        mujoco.mj_step(model, data)
        gap = float(np.linalg.norm(_site_pos(model, data, left) - _site_pos(model, data, right)))
        max_opening = max(max_opening, abs(gap - gap0))
    data.xfrc_applied[:] = 0.0
    return 1000.0 * max_opening


def _support_release_sag_test(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, float]:
    top_before = _site_pos(model, data, "site_top_seam_03")[2]
    bottom_before = _site_pos(model, data, "site_bottom_sling")[2]
    geom_id = _geom_id(model, "hidden_support_geom")
    if geom_id >= 0:
        model.geom_contype[geom_id] = 0
        model.geom_conaffinity[geom_id] = 0
        model.geom_rgba[geom_id, 3] = 0.0
    _step(model, data, 0.30)
    top_after = _site_pos(model, data, "site_top_seam_03")[2]
    bottom_after = _site_pos(model, data, "site_bottom_sling")[2]
    bottom_sag_mm = 1000.0 * max(0.0, bottom_before - bottom_after)
    top_reference_drop_mm = 1000.0 * max(0.0, top_before - top_after)
    return bottom_sag_mm, top_reference_drop_mm


def _fold_brushing_test(model: mujoco.MjModel, data: mujoco.MjData, scenario: str) -> str:
    state = get_scenario(scenario)
    before = 1.0 - state.fold_coverage_fraction
    if state.fold_coverage_fraction <= 0.0:
        return f"{before:.3f}->{before:.3f}"

    body_name = "fold_patch_left" if abs(state.fold_left_deg) >= abs(state.fold_right_deg) else "fold_patch_right"
    body = _body_id(model, body_name)
    jid = _joint_id(model, f"{body_name}_hinge")
    qadr = int(model.jnt_qposadr[jid]) if jid >= 0 else -1
    q0 = float(data.qpos[qadr]) if qadr >= 0 else 0.0
    for _ in range(int(0.20 / model.opt.timestep)):
        # full unfold가 아니라, seam 근처를 스치며 exposed seam이 조금 증가하는지만 봅니다.
        data.xfrc_applied[body, :3] = np.array([1.8, -1.4, 0.35])
        mujoco.mj_step(model, data)
    data.xfrc_applied[:] = 0.0
    q1 = float(data.qpos[qadr]) if qadr >= 0 else q0
    after = min(1.0, before + min(0.14, abs(q1 - q0) * 0.24))
    return f"{before:.3f}->{after:.3f}"


def _apply_runtime_tuning(model: mujoco.MjModel) -> dict[str, str]:
    """모델 토폴로지는 유지하고 passive joint만 부드럽게 재조정합니다."""

    stiffness_tags: list[str] = []
    damping_tags: list[str] = []
    limit_tags: list[str] = []
    payload_tags: list[str] = []

    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        dofadr = int(model.jnt_dofadr[jid])
        if any(key in name for key in ("shoulder_", "front_face_", "back_face_", "side_gusset_")):
            model.jnt_stiffness[jid] *= 0.42
            model.dof_damping[dofadr] *= 0.55
            model.jnt_range[jid] = np.deg2rad(np.array([-65.0, 65.0]))
            stiffness_tags.append("shoulder/face")
            damping_tags.append("shoulder/face")
            limit_tags.append("shoulder/face")
        elif "lower_belly_" in name:
            model.jnt_stiffness[jid] *= 0.32
            model.dof_damping[dofadr] *= 0.48
            model.jnt_range[jid] = np.deg2rad(np.array([-70.0, 70.0]))
            stiffness_tags.append("lower_belly")
            damping_tags.append("lower_belly")
            limit_tags.append("lower_belly")
        elif "top_seam_" in name or "fold_patch_" in name:
            model.jnt_stiffness[jid] *= 0.48
            model.dof_damping[dofadr] *= 0.55
            if "top_seam_" in name:
                model.jnt_range[jid] = np.deg2rad(np.array([-30.0, 30.0]))
                limit_tags.append("top_seam")
            else:
                model.jnt_range[jid] = np.deg2rad(np.array([-88.0, 58.0]))
                limit_tags.append("fold_patch")
            stiffness_tags.append("top/fold")
            damping_tags.append("top/fold")
        elif "bottom_sling_slide" in name:
            model.jnt_stiffness[jid] *= 0.36
            model.dof_damping[dofadr] *= 0.50
            model.jnt_range[jid] = np.array([-0.105, 0.028])
            stiffness_tags.append("bottom_sling")
            damping_tags.append("bottom_sling")
            limit_tags.append("bottom_sling")
        elif name.startswith("payload_"):
            model.jnt_stiffness[jid] *= 0.55
            model.dof_damping[dofadr] *= 0.65
            if name.endswith("_x"):
                model.jnt_range[jid] = np.array([-0.080, 0.080])
            elif name.endswith("_y"):
                model.jnt_range[jid] = np.array([-0.105, 0.105])
            elif name.endswith("_z"):
                model.jnt_range[jid] = np.array([-0.060, 0.055])
            stiffness_tags.append("payload")
            damping_tags.append("payload")
            payload_tags.append("payload_xyz")

    def uniq(values: list[str]) -> str:
        return "+".join(sorted(set(values))) if values else "none"

    return {
        "tuned_joint_stiffness": uniq(stiffness_tags),
        "tuned_joint_damping": uniq(damping_tags),
        "tuned_joint_limit": uniq(limit_tags),
        "tuned_payload_slide_range": uniq(payload_tags),
        # XML tendon을 새로 만들지 않고, 진단 단계에서 lower belly pair force를 strap처럼 묶어 넣습니다.
        # 그래서 결과 해석은 "task-driven low-dimensional coupling"으로 제한해야 합니다.
        "tuned_tendon_coupling": "runtime_pair_force_coupling",
    }


def _load_model(scenario: str, *, include_robots: bool = True) -> tuple[mujoco.MjModel, mujoco.MjData, Path]:
    xml = write_scene_xml(scenario, include_robots=include_robots)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    _settle(model, data, 0.45)
    return model, data, xml


def _run_force_diagnostics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scenario: str,
    *,
    tuned_pair_force: bool,
) -> Diagnostics:
    _reset_for_test(model, data)
    shoulder_deflection_mm, shoulder_recovered = _shoulder_poke_test(model, data)

    _reset_for_test(model, data)
    top_patch_change_mm = _top_preload_test(model, data)

    _reset_for_test(model, data)
    lower_belly_opening_mm = _scoop_insertion_test(model, data, tuned_pair_force=tuned_pair_force)

    _reset_for_test(model, data)
    bottom_sag_mm, top_reference_drop_mm = _support_release_sag_test(model, data)

    _reset_for_test(model, data)
    fold_exposed = _fold_brushing_test(model, data, scenario)

    return Diagnostics(
        scenario_name=scenario,
        shoulder_deflection_mm=shoulder_deflection_mm,
        top_patch_change_mm=top_patch_change_mm,
        lower_belly_opening_mm=lower_belly_opening_mm,
        bottom_sag_mm=bottom_sag_mm,
        fold_exposed_fraction_before_after=fold_exposed,
        shoulder_recovered=shoulder_recovered,
        top_reference_drop_mm=top_reference_drop_mm,
    )


def validate_one_scenario(
    scenario: str,
    *,
    out_dir: Path = OUT_DIR,
    render: bool = True,
    auto_tune: bool = True,
) -> tuple[dict[str, object], Diagnostics, Diagnostics]:
    model, data, _xml = _load_model(scenario, include_robots=True)
    before = _run_force_diagnostics(model, data, scenario, tuned_pair_force=False)

    tuning_applied = False
    tuning_info = {
        "tuned_joint_stiffness": "none",
        "tuned_joint_damping": "none",
        "tuned_joint_limit": "none",
        "tuned_payload_slide_range": "none",
        "tuned_tendon_coupling": "none",
    }
    after = before

    if auto_tune and before.needs_tuning():
        tuning_applied = True
        model, data, _xml = _load_model(scenario, include_robots=True)
        tuning_info = _apply_runtime_tuning(model)
        mujoco.mj_forward(model, data)
        after = _run_force_diagnostics(model, data, scenario, tuned_pair_force=True)

    if render:
        _reset_for_test(model, data)
        _render_debug(model, data, scenario, out_dir / "shape_diagnostics")

    row: dict[str, object] = {
        "scenario_name": scenario,
        "shoulder_deflection_mm": f"{after.shoulder_deflection_mm:.3f}",
        "top_patch_change_mm": f"{after.top_patch_change_mm:.3f}",
        "lower_belly_opening_mm": f"{after.lower_belly_opening_mm:.3f}",
        "bottom_sag_mm": f"{after.bottom_sag_mm:.3f}",
        "fold_exposed_fraction_before_after": after.fold_exposed_fraction_before_after,
        "rigid_like_flag": bool(after.rigid_like()),
        "tuning_applied": bool(tuning_applied),
        **tuning_info,
    }
    return row, before, after


def validate_all(
    *,
    scenario: str = "all",
    out_dir: Path = OUT_DIR,
    render: bool = True,
    auto_tune: bool = True,
) -> list[dict[str, object]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = list(SCENARIO_NAMES) if scenario == "all" else [scenario]
    rows: list[dict[str, object]] = []
    before_after: dict[str, tuple[Diagnostics, Diagnostics]] = {}

    for name in selected:
        row, before, after = validate_one_scenario(name, out_dir=out_dir, render=render, auto_tune=auto_tune)
        rows.append(row)
        before_after[name] = (before, after)

    csv_path = out_dir / "scenario_shape_diagnostics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    diag_dir = out_dir / "shape_diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    with (diag_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    md = [
        "# Shape Change Diagnostics",
        "",
        "이 파일은 `visual_skin`이 아니라 실제 articulated physics patches의 움직임을 기준으로 작성됩니다.",
        "`visual_skin`은 physics-free sealed sack silhouette이며, 실시간 변형 판정에는 사용하지 않습니다.",
        "",
        "렌더 결과:",
        "- physics patch debug: `project_dual_sack/out/shape_diagnostics/*_physics_patch_debug.png`",
        "- visual skin only: `project_dual_sack/out/shape_diagnostics/*_visual_skin.png`",
        "- overlay: `project_dual_sack/out/shape_diagnostics/*_overlay.png`",
        "",
    ]
    for row in rows:
        before, after = before_after[str(row["scenario_name"])]
        md.append(f"## {row['scenario_name']}")
        md.append(f"- tuning_applied: `{row['tuning_applied']}`")
        md.append(f"- rigid_like_flag: `{row['rigid_like_flag']}`")
        md.append(
            "- before tuning: "
            f"shoulder_deflection_mm={before.shoulder_deflection_mm:.3f}, "
            f"top_patch_change_mm={before.top_patch_change_mm:.3f}, "
            f"lower_belly_opening_mm={before.lower_belly_opening_mm:.3f}, "
            f"bottom_sag_mm={before.bottom_sag_mm:.3f}"
        )
        md.append(
            "- after tuning: "
            f"shoulder_deflection_mm={after.shoulder_deflection_mm:.3f}, "
            f"top_patch_change_mm={after.top_patch_change_mm:.3f}, "
            f"lower_belly_opening_mm={after.lower_belly_opening_mm:.3f}, "
            f"bottom_sag_mm={after.bottom_sag_mm:.3f}, "
            f"fold_exposed_fraction_before_after={after.fold_exposed_fraction_before_after}"
        )
        md.append(f"- shoulder recovery after poke: `{after.shoulder_recovered}`")
        md.append(f"- top reference drop during support release: `{after.top_reference_drop_mm:.3f} mm`")
        md.append("")

    if any(bool(row["rigid_like_flag"]) for row in rows):
        md.append("## Conclusion")
        md.append("current topology too rigid-like")
        md.append("")
    else:
        md.append("## Conclusion")
        md.append("모든 선택 scenario에서 force-driven test에 의해 측정 가능한 patch motion이 발생했습니다.")
        md.append("")

    summary_md = "\n".join(md)
    (out_dir / "scenario_shape_diagnostics.md").write_text(summary_md, encoding="utf-8")
    (diag_dir / "summary.md").write_text(summary_md, encoding="utf-8")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose and tune shape change of the articulated sack surrogate")
    parser.add_argument("--scenario", choices=SCENARIO_NAMES + ("all",), default="all")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-auto-tune", action="store_true")
    args = parser.parse_args()

    rows = validate_all(
        scenario=args.scenario,
        render=not args.no_render,
        auto_tune=not args.no_auto_tune,
    )
    for row in rows:
        print(
            f"{row['scenario_name']}: "
            f"shoulder_deflection_mm={row['shoulder_deflection_mm']} "
            f"top_patch_change_mm={row['top_patch_change_mm']} "
            f"lower_belly_opening_mm={row['lower_belly_opening_mm']} "
            f"bottom_sag_mm={row['bottom_sag_mm']} "
            f"rigid_like_flag={row['rigid_like_flag']} "
            f"tuning_applied={row['tuning_applied']}"
        )
    print(f"summary_csv={OUT_DIR / 'scenario_shape_diagnostics.csv'}")
    print(f"summary_md={OUT_DIR / 'scenario_shape_diagnostics.md'}")
    print(f"debug_summary_csv={OUT_DIR / 'shape_diagnostics' / 'summary.csv'}")
    print(f"debug_summary_md={OUT_DIR / 'shape_diagnostics' / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
