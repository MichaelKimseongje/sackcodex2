from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from scenario_builder import OUT_DIR, SCENARIO_NAMES, get_scenario, write_scene_xml


OUTPUT_DIR = OUT_DIR / "force_responsiveness"

SUMMARY_FIELDS = [
    "scenario_name",
    "shoulder_deflection_mm",
    "top_patch_change_mm",
    "lower_belly_opening_mm",
    "bottom_sag_mm",
    "fold_exposed_fraction_before_after",
    "world_frame_bag_translation_mm",
    "world_frame_bag_rotation_deg",
    "bag_frame_local_deformation_mm",
    "rigid_like_flag",
]

LOCAL_DEFORMATION_THRESHOLD_MM = 0.50
WORLD_TRANSLATION_THRESHOLD_MM = 1.00
WORLD_ROTATION_THRESHOLD_DEG = 1.00

PATCH_SITES = [
    "site_top_seam_00",
    "site_top_seam_03",
    "site_top_seam_06",
    "site_shoulder_left_00_tip",
    "site_shoulder_left_01_tip",
    "site_shoulder_left_02_tip",
    "site_shoulder_right_00_tip",
    "site_shoulder_right_01_tip",
    "site_shoulder_right_02_tip",
    "site_side_gusset_left_00_tip",
    "site_side_gusset_left_01_tip",
    "site_side_gusset_right_00_tip",
    "site_side_gusset_right_01_tip",
    "site_lower_belly_00_tip",
    "site_lower_belly_01_tip",
    "site_lower_belly_02_tip",
    "site_lower_belly_03_tip",
    "site_bottom_sling",
    "site_fold_patch_left",
    "site_fold_patch_right",
]


@dataclass
class MotionTracker:
    """bag_frame rigid motion과 bag-frame 기준 local deformation을 분리해 추적합니다."""

    bag_pos0: np.ndarray
    bag_mat0: np.ndarray
    local0: dict[str, np.ndarray]
    max_world_translation_mm: float = 0.0
    max_world_rotation_deg: float = 0.0
    max_local_deformation_mm: float = 0.0

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        pos, mat = _bag_pose(model, data)
        self.max_world_translation_mm = max(
            self.max_world_translation_mm,
            1000.0 * float(np.linalg.norm(pos - self.bag_pos0)),
        )
        self.max_world_rotation_deg = max(
            self.max_world_rotation_deg,
            _rotation_delta_deg(self.bag_mat0, mat),
        )
        for site_name, local_start in self.local0.items():
            local_now = _site_pos_local(model, data, site_name)
            self.max_local_deformation_mm = max(
                self.max_local_deformation_mm,
                1000.0 * float(np.linalg.norm(local_now - local_start)),
            )


@dataclass
class TestResult:
    name: str
    metric_mm: float
    world_translation_mm: float
    world_rotation_deg: float
    local_deformation_mm: float
    extra: str = ""


def _imageio():
    try:
        import imageio.v2 as imageio
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("imageio가 필요합니다. `pip install imageio` 후 다시 실행해 주세요.") from exc
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


def _geom_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


def _joint_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def _bag_pose(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    bid = _body_id(model, "bag_frame")
    pos = data.xpos[bid].copy()
    mat = data.xmat[bid].reshape(3, 3).copy()
    return pos, mat


def _site_pos_world(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    return data.site_xpos[_site_id(model, name)].copy()


def _site_pos_local(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    bag_pos, bag_mat = _bag_pose(model, data)
    return bag_mat.T @ (_site_pos_world(model, data, name) - bag_pos)


def _rotation_delta_deg(mat0: np.ndarray, mat1: np.ndarray) -> float:
    rel = mat0.T @ mat1
    cos_angle = np.clip((float(np.trace(rel)) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(cos_angle))


def _step(model: mujoco.MjModel, data: mujoco.MjData, seconds: float, tracker: MotionTracker | None = None) -> bool:
    for _ in range(max(1, int(seconds / model.opt.timestep))):
        mujoco.mj_step(model, data)
        if tracker is not None:
            tracker.update(model, data)
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            return False
    return True


def _reset_and_settle(model: mujoco.MjModel, data: mujoco.MjData, seconds: float = 0.40) -> None:
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    _step(model, data, seconds)
    data.xfrc_applied[:] = 0.0


def _make_tracker(model: mujoco.MjModel, data: mujoco.MjData) -> MotionTracker:
    bag_pos, bag_mat = _bag_pose(model, data)
    local0: dict[str, np.ndarray] = {}
    for name in PATCH_SITES:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) >= 0:
            local0[name] = _site_pos_local(model, data, name)
    return MotionTracker(bag_pos0=bag_pos, bag_mat0=bag_mat, local0=local0)


def _hide_visual_skin(model: mujoco.MjModel, alpha: float) -> np.ndarray:
    original = model.geom_rgba.copy()
    for name in ("visual_skin_main", "sealed_top_cap_visual_geom", "visual_print_mark_geom"):
        gid = _geom_id(model, name)
        if gid >= 0:
            model.geom_rgba[gid, 3] = alpha
    return original


def _restore_rgba(model: mujoco.MjModel, rgba: np.ndarray) -> None:
    model.geom_rgba[:] = rgba


def _scene_option(*, physics_patches: bool) -> mujoco.MjvOption:
    option = mujoco.MjvOption()
    option.geomgroup[:] = True
    option.geomgroup[1] = bool(physics_patches)
    return option


def _render_views(model: mujoco.MjModel, data: mujoco.MjData, scenario: str, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, width=1280, height=820)
    imageio = _imageio()
    outputs: dict[str, str] = {}

    original = _hide_visual_skin(model, alpha=0.02)
    renderer.update_scene(data, camera="front", scene_option=_scene_option(physics_patches=True))
    path = out_dir / f"{scenario}_physics_patch_debug.png"
    imageio.imwrite(path, renderer.render())
    outputs["physics_patch_debug"] = str(path)
    _restore_rgba(model, original)

    renderer.update_scene(data, camera="front", scene_option=_scene_option(physics_patches=False))
    path = out_dir / f"{scenario}_visual_skin.png"
    imageio.imwrite(path, renderer.render())
    outputs["visual_skin"] = str(path)

    original = _hide_visual_skin(model, alpha=0.42)
    renderer.update_scene(data, camera="front", scene_option=_scene_option(physics_patches=True))
    path = out_dir / f"{scenario}_overlay.png"
    imageio.imwrite(path, renderer.render())
    outputs["overlay"] = str(path)
    _restore_rgba(model, original)

    renderer.close()
    return outputs


def _make_result(name: str, metric_mm: float, tracker: MotionTracker, extra: str = "") -> TestResult:
    return TestResult(
        name=name,
        metric_mm=metric_mm,
        world_translation_mm=tracker.max_world_translation_mm,
        world_rotation_deg=tracker.max_world_rotation_deg,
        local_deformation_mm=tracker.max_local_deformation_mm,
        extra=extra,
    )


def _shoulder_poke_test(model: mujoco.MjModel, data: mujoco.MjData) -> TestResult:
    _reset_and_settle(model, data)
    tracker = _make_tracker(model, data)
    left_body = _body_id(model, "shoulder_left_01")
    right_body = _body_id(model, "shoulder_right_01")
    left0 = _site_pos_local(model, data, "site_shoulder_left_01_tip")
    right0 = _site_pos_local(model, data, "site_shoulder_right_01_tip")
    max_deflection = 0.0
    for _ in range(int(0.18 / model.opt.timestep)):
        data.xfrc_applied[left_body, :3] = np.array([0.0, -9.0, -0.8])
        data.xfrc_applied[right_body, :3] = np.array([0.0, 9.0, -0.8])
        mujoco.mj_step(model, data)
        tracker.update(model, data)
        left = _site_pos_local(model, data, "site_shoulder_left_01_tip")
        right = _site_pos_local(model, data, "site_shoulder_right_01_tip")
        max_deflection = max(
            max_deflection,
            0.5 * (float(np.linalg.norm(left - left0)) + float(np.linalg.norm(right - right0))),
        )
    data.xfrc_applied[:] = 0.0
    _step(model, data, 0.25, tracker)
    return _make_result("shoulder_poke", 1000.0 * max_deflection, tracker)


def _top_preload_test(model: mujoco.MjModel, data: mujoco.MjData) -> TestResult:
    _reset_and_settle(model, data)
    tracker = _make_tracker(model, data)
    body = _body_id(model, "top_seam_03")
    baseline = _site_pos_local(model, data, "site_top_seam_03")
    max_change = 0.0
    for _ in range(int(0.18 / model.opt.timestep)):
        data.xfrc_applied[body, :3] = np.array([0.0, 0.0, -6.0])
        mujoco.mj_step(model, data)
        tracker.update(model, data)
        max_change = max(max_change, float(np.linalg.norm(_site_pos_local(model, data, "site_top_seam_03") - baseline)))
    data.xfrc_applied[:] = 0.0
    _step(model, data, 0.20, tracker)
    return _make_result("top_preload", 1000.0 * max_change, tracker)


def _lateral_squeeze_test(model: mujoco.MjModel, data: mujoco.MjData) -> TestResult:
    _reset_and_settle(model, data)
    tracker = _make_tracker(model, data)
    left_body = _body_id(model, "side_gusset_left_00")
    right_body = _body_id(model, "side_gusset_right_00")
    left0 = _site_pos_local(model, data, "site_side_gusset_left_00_tip")
    right0 = _site_pos_local(model, data, "site_side_gusset_right_00_tip")
    width0 = float(np.linalg.norm(left0 - right0))
    max_change = 0.0
    for _ in range(int(0.20 / model.opt.timestep)):
        data.xfrc_applied[left_body, :3] = np.array([0.0, -8.0, 0.0])
        data.xfrc_applied[right_body, :3] = np.array([0.0, 8.0, 0.0])
        mujoco.mj_step(model, data)
        tracker.update(model, data)
        left = _site_pos_local(model, data, "site_side_gusset_left_00_tip")
        right = _site_pos_local(model, data, "site_side_gusset_right_00_tip")
        max_change = max(max_change, abs(float(np.linalg.norm(left - right)) - width0))
    data.xfrc_applied[:] = 0.0
    _step(model, data, 0.20, tracker)
    return _make_result("lateral_squeeze", 1000.0 * max_change, tracker)


def _scoop_insertion_test(model: mujoco.MjModel, data: mujoco.MjData) -> TestResult:
    _reset_and_settle(model, data)
    tracker = _make_tracker(model, data)
    left_body = _body_id(model, "lower_belly_01")
    right_body = _body_id(model, "lower_belly_02")
    left0 = _site_pos_local(model, data, "site_lower_belly_01_tip")
    right0 = _site_pos_local(model, data, "site_lower_belly_02_tip")
    gap0 = float(np.linalg.norm(left0 - right0))
    max_opening = 0.0
    for _ in range(int(0.24 / model.opt.timestep)):
        # scoop 삽입을 직접적인 hinge torque로 표현해서 local opening을 진단합니다.
        data.xfrc_applied[left_body, 3:] = np.array([12.0, 0.0, 0.0])
        data.xfrc_applied[right_body, 3:] = np.array([-12.0, 0.0, 0.0])
        mujoco.mj_step(model, data)
        tracker.update(model, data)
        left = _site_pos_local(model, data, "site_lower_belly_01_tip")
        right = _site_pos_local(model, data, "site_lower_belly_02_tip")
        max_opening = max(max_opening, abs(float(np.linalg.norm(left - right)) - gap0))
    data.xfrc_applied[:] = 0.0
    _step(model, data, 0.20, tracker)
    return _make_result("scoop_insertion", 1000.0 * max_opening, tracker)


def _support_release_sag_test(model: mujoco.MjModel, data: mujoco.MjData) -> TestResult:
    _reset_and_settle(model, data)
    tracker = _make_tracker(model, data)
    bottom0 = _site_pos_local(model, data, "site_bottom_sling")[2]
    geom_id = _geom_id(model, "hidden_support_geom")
    if geom_id >= 0:
        model.geom_contype[geom_id] = 0
        model.geom_conaffinity[geom_id] = 0
        model.geom_rgba[geom_id, 3] = 0.0
    _step(model, data, 0.30, tracker)
    bottom1 = _site_pos_local(model, data, "site_bottom_sling")[2]
    bottom_sag_mm = 1000.0 * max(0.0, bottom0 - bottom1)
    return _make_result("support_release_sag", bottom_sag_mm, tracker)


def _fold_brushing_test(model: mujoco.MjModel, data: mujoco.MjData, scenario: str) -> tuple[TestResult, str]:
    _reset_and_settle(model, data)
    tracker = _make_tracker(model, data)
    state = get_scenario(scenario)
    before = 1.0 - state.fold_coverage_fraction
    if state.fold_coverage_fraction <= 0.0:
        return _make_result("fold_brushing", 0.0, tracker), f"{before:.3f}->{before:.3f}"

    body_name = "fold_patch_left" if abs(state.fold_left_deg) >= abs(state.fold_right_deg) else "fold_patch_right"
    site_name = f"site_{body_name}"
    body = _body_id(model, body_name)
    jid = _joint_id(model, f"{body_name}_hinge")
    qadr = int(model.jnt_qposadr[jid]) if jid >= 0 else -1
    q0 = float(data.qpos[qadr]) if qadr >= 0 else 0.0
    local0 = _site_pos_local(model, data, site_name)
    max_change = 0.0
    for _ in range(int(0.22 / model.opt.timestep)):
        data.xfrc_applied[body, :3] = np.array([1.8, -1.4, 0.35])
        mujoco.mj_step(model, data)
        tracker.update(model, data)
        max_change = max(max_change, float(np.linalg.norm(_site_pos_local(model, data, site_name) - local0)))
    data.xfrc_applied[:] = 0.0
    q1 = float(data.qpos[qadr]) if qadr >= 0 else q0
    after = min(1.0, before + min(0.14, abs(q1 - q0) * 0.24))
    return _make_result("fold_brushing", 1000.0 * max_change, tracker), f"{before:.3f}->{after:.3f}"


def _load_model(scenario: str) -> tuple[mujoco.MjModel, mujoco.MjData, Path]:
    xml = write_scene_xml(scenario, include_robots=True)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    _reset_and_settle(model, data)
    return model, data, xml


def diagnose_one_scenario(scenario: str, *, out_dir: Path = OUTPUT_DIR, render: bool = True) -> tuple[dict[str, object], list[TestResult]]:
    model, data, _xml = _load_model(scenario)

    shoulder = _shoulder_poke_test(model, data)
    top = _top_preload_test(model, data)
    squeeze = _lateral_squeeze_test(model, data)
    scoop = _scoop_insertion_test(model, data)
    sag = _support_release_sag_test(model, data)
    fold, fold_fraction = _fold_brushing_test(model, data, scenario)
    tests = [shoulder, top, squeeze, scoop, sag, fold]

    world_translation_mm = max(test.world_translation_mm for test in tests)
    world_rotation_deg = max(test.world_rotation_deg for test in tests)
    local_deformation_mm = max(test.local_deformation_mm for test in tests)
    rigid_like = (
        local_deformation_mm < LOCAL_DEFORMATION_THRESHOLD_MM
        and (
            world_translation_mm > WORLD_TRANSLATION_THRESHOLD_MM
            or world_rotation_deg > WORLD_ROTATION_THRESHOLD_DEG
        )
    ) or local_deformation_mm < 0.25

    if render:
        _reset_and_settle(model, data)
        _render_views(model, data, scenario, out_dir)

    row: dict[str, object] = {
        "scenario_name": scenario,
        "shoulder_deflection_mm": f"{shoulder.metric_mm:.3f}",
        "top_patch_change_mm": f"{top.metric_mm:.3f}",
        "lower_belly_opening_mm": f"{scoop.metric_mm:.3f}",
        "bottom_sag_mm": f"{sag.metric_mm:.3f}",
        "fold_exposed_fraction_before_after": fold_fraction,
        "world_frame_bag_translation_mm": f"{world_translation_mm:.3f}",
        "world_frame_bag_rotation_deg": f"{world_rotation_deg:.3f}",
        "bag_frame_local_deformation_mm": f"{local_deformation_mm:.3f}",
        "rigid_like_flag": bool(rigid_like),
    }
    return row, tests


def diagnose_all(*, scenario: str = "all", out_dir: Path = OUTPUT_DIR, render: bool = True) -> list[dict[str, object]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = list(SCENARIO_NAMES) if scenario == "all" else [scenario]
    rows: list[dict[str, object]] = []
    detail: dict[str, list[TestResult]] = {}

    for name in selected:
        row, tests = diagnose_one_scenario(name, out_dir=out_dir, render=render)
        rows.append(row)
        detail[name] = tests

    summary_csv = out_dir / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    md = [
        "# Force-Responsive Semi-Deformable Diagnostics",
        "",
        "이 진단은 `visual_skin`이 아니라 physics patches 기준으로 local deformation을 계산합니다.",
        "`bag_frame`이 freejoint이므로 world-frame motion과 bag-frame aligned local deformation을 분리했습니다.",
        "",
        "판정 기준:",
        "- world-frame motion만 크고 bag-frame local deformation이 거의 0이면 `rigid_like_flag=True`입니다.",
        "- bag-frame local deformation이 유의미하면 force-responsive surrogate로 해석합니다.",
        "",
        "렌더:",
        "- `*_physics_patch_debug.png`: visual skin을 거의 숨기고 실제 패치를 표시",
        "- `*_visual_skin.png`: physics-free 외피만 표시",
        "- `*_overlay.png`: 외피와 실제 패치를 함께 표시",
        "",
    ]
    for row in rows:
        scenario_name = str(row["scenario_name"])
        md.append(f"## {scenario_name}")
        md.append(f"- rigid_like_flag: `{row['rigid_like_flag']}`")
        md.append(
            "- summary: "
            f"shoulder_deflection_mm={row['shoulder_deflection_mm']}, "
            f"top_patch_change_mm={row['top_patch_change_mm']}, "
            f"lower_belly_opening_mm={row['lower_belly_opening_mm']}, "
            f"bottom_sag_mm={row['bottom_sag_mm']}, "
            f"bag_frame_local_deformation_mm={row['bag_frame_local_deformation_mm']}"
        )
        md.append(
            "- rigid body motion: "
            f"world_frame_bag_translation_mm={row['world_frame_bag_translation_mm']}, "
            f"world_frame_bag_rotation_deg={row['world_frame_bag_rotation_deg']}"
        )
        md.append(f"- fold_exposed_fraction_before_after: `{row['fold_exposed_fraction_before_after']}`")
        md.append("- per-test separation:")
        for test in detail[scenario_name]:
            md.append(
                f"  - {test.name}: metric_mm={test.metric_mm:.3f}, "
                f"world_translation_mm={test.world_translation_mm:.3f}, "
                f"world_rotation_deg={test.world_rotation_deg:.3f}, "
                f"local_deformation_mm={test.local_deformation_mm:.3f}"
            )
        md.append("")

    if any(bool(row["rigid_like_flag"]) for row in rows):
        md.append("## Conclusion")
        md.append("일부 scenario는 current topology too rigid-like 입니다.")
    else:
        md.append("## Conclusion")
        md.append("선택한 scenario는 world-frame rigid motion만이 아니라 bag-frame local deformation도 측정되어 force-responsive surrogate로 볼 수 있습니다.")

    (out_dir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose force-responsiveness of physics patches separate from bag-frame rigid motion")
    parser.add_argument("--scenario", choices=SCENARIO_NAMES + ("all",), default="all")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    rows = diagnose_all(scenario=args.scenario, render=not args.no_render)
    for row in rows:
        verdict = "rigid-like" if row["rigid_like_flag"] else "force-responsive"
        print(
            f"{row['scenario_name']}: {verdict} "
            f"local={row['bag_frame_local_deformation_mm']}mm "
            f"world_translation={row['world_frame_bag_translation_mm']}mm "
            f"world_rotation={row['world_frame_bag_rotation_deg']}deg"
        )
    print(f"summary_csv={OUTPUT_DIR / 'summary.csv'}")
    print(f"summary_md={OUTPUT_DIR / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
