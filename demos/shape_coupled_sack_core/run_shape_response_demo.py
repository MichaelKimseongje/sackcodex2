from __future__ import annotations

import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from build_shape_coupled_sack import OUT_DIR, build_scene_tree
from scenario_builder import available_scenarios, get_scenario
from shape_response_controller import ReducedOrderShapeController, ShapeMetrics, measure_shape_metrics


ROOT_DIR = Path(__file__).resolve().parent
GENERATED_DIR = ROOT_DIR / "generated"
RESPONSE_OUT_DIR = OUT_DIR / "shape_response"


def _imageio():
    try:
        import imageio.v2 as imageio
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("imageio가 필요합니다. `pip install imageio` 후 다시 실행해 주세요.") from exc
    return imageio


def _smooth(alpha: float) -> float:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _obj_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    return mujoco.mj_name2id(model, obj_type, name)


def _site_pos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    site_id = _obj_id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site_id < 0:
        raise ValueError(f"site not found: {name}")
    return data.site_xpos[site_id].copy()


def _body_pos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    body_id = _obj_id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"body not found: {name}")
    return data.xpos[body_id].copy()


def _set_mocap(model: mujoco.MjModel, data: mujoco.MjData, body_name: str, pos: np.ndarray) -> None:
    body_id = _obj_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"mocap body not found: {body_name}")
    mocap_id = int(model.body_mocapid[body_id])
    if mocap_id < 0:
        raise ValueError(f"body is not mocap: {body_name}")
    data.mocap_pos[mocap_id] = np.asarray(pos, dtype=np.float64)
    data.mocap_quat[mocap_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _set_gripper(model: mujoco.MjModel, data: mujoco.MjData, center: np.ndarray, gap: float) -> None:
    center = np.asarray(center, dtype=np.float64)
    _set_mocap(model, data, "gripper_left_mocap", center + np.array([0.0, -0.5 * gap, 0.0]))
    _set_mocap(model, data, "gripper_right_mocap", center + np.array([0.0, 0.5 * gap, 0.0]))
    _set_mocap(model, data, "gripper_center_mocap", center)


def _set_scoop(model: mujoco.MjModel, data: mujoco.MjData, pos: np.ndarray) -> None:
    _set_mocap(model, data, "scoop_mocap", np.asarray(pos, dtype=np.float64))


def _render(model: mujoco.MjModel, data: mujoco.MjData, renderer: mujoco.Renderer, path: Path, camera: str) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    renderer.update_scene(data, camera=camera)
    image = renderer.render()
    _imageio().imwrite(path, image)
    return image


def _choose_target(scenario: str) -> tuple[str, int, float]:
    if scenario.startswith("top_fold"):
        return "grasp_fold_1", 2, 1.0
    if scenario == "underfilled":
        return "grasp_shoulder_02", 2, 1.0
    if scenario == "post_separation_sag":
        return "grasp_shoulder_02", 2, 1.0
    return "grasp_seam_02", 2, 1.0


def _write_xml_for_demo(scenario: str, *, post_release: bool) -> Path:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    root = build_scene_tree(get_scenario(scenario, post_release=post_release), include_eval_gripper=True, include_eval_scoop=True)
    ET.indent(root, space="  ")
    path = GENERATED_DIR / f"scene_shape_response_{scenario}{'_after' if post_release else ''}.xml"
    path.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    return path


def _step_phase(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: ReducedOrderShapeController,
    *,
    phase: str,
    seconds: float,
    gripper_start: np.ndarray,
    gripper_end: np.ndarray,
    gap_start: float,
    gap_end: float,
    scoop_start: np.ndarray,
    scoop_end: np.ndarray,
    renderer: mujoco.Renderer,
    video_frames: list[np.ndarray],
    camera: str,
    frame_stride: int = 18,
) -> None:
    steps = max(1, int(seconds / model.opt.timestep))
    for step in range(steps):
        alpha = _smooth(step / max(1, steps - 1))
        gripper_pos = (1.0 - alpha) * gripper_start + alpha * gripper_end
        scoop_pos = (1.0 - alpha) * scoop_start + alpha * scoop_end
        gap = float((1.0 - alpha) * gap_start + alpha * gap_end)
        _set_gripper(model, data, gripper_pos, gap)
        _set_scoop(model, data, scoop_pos)
        controller.apply(model, data, phase=phase)
        mujoco.mj_step(model, data)
        if step % frame_stride == 0:
            renderer.update_scene(data, camera=camera)
            video_frames.append(renderer.render().copy())


def _write_summary(rows: list[ShapeMetrics], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].row().keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.row())


def _write_markdown(rows: list[ShapeMetrics], path: Path, *, scenario: str, target_site: str, video_path: Path | None) -> None:
    initial = rows[0]
    final = rows[-1]
    lines = [
        "# Shape Response Demo Summary",
        "",
        f"- scenario: `{scenario}`",
        f"- target_site: `{target_site}`",
        f"- video: `{video_path}`" if video_path else "- video: frame sequence only",
        "",
        "## 핵심 변화",
        f"- upper_half_width delta: `{final.upper_half_width - initial.upper_half_width:.5f} m`",
        f"- lower_half_width delta: `{final.lower_half_width - initial.lower_half_width:.5f} m`",
        f"- local shoulder angle delta: `{final.shoulder_angle_local_deg - initial.shoulder_angle_local_deg:.2f} deg`",
        f"- bottom_sag delta: `{final.bottom_sag_m - initial.bottom_sag_m:.5f} m`",
        f"- payload_y delta: `{final.payload_slide_y_m - initial.payload_slide_y_m:.5f} m`",
        f"- bag_com_z delta: `{final.bag_com_z_m - initial.bag_com_z_m:.5f} m`",
        "",
        "## 해석",
        "이 데모는 full soft cloth가 아니라, 접촉과 지지가 들어오면 패널 힌지, 하부 sling, 내부 payload가 함께 반응하는 reduced-order shape coupling입니다.",
        "따라서 pure material simulator라고 주장하지 않고, support-state formation 평가에 필요한 형상 변화와 하중 재분배를 안정적으로 보여주는 task-driven surrogate로 사용합니다.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_demo(scenario: str, *, post_release: bool = False, save_video: bool = True, camera: str = "overview") -> int:
    out_dir = RESPONSE_OUT_DIR / scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    xml_path = _write_xml_for_demo(scenario, post_release=post_release)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    target_site, target_index, lateral_bias = _choose_target(scenario)
    target = _site_pos(model, data, target_site)
    controller = ReducedOrderShapeController(target_index=target_index, lateral_bias=lateral_bias)
    renderer = mujoco.Renderer(model, width=1280, height=820)
    video_frames: list[np.ndarray] = []
    rows: list[ShapeMetrics] = []

    open_center = target + np.array([0.0, 0.0, 0.052])
    pinch_center = target + np.array([0.0, 0.0, 0.012])
    lift_center = pinch_center + np.array([0.0, 0.0, 0.050])
    scoop_far = np.array([-0.245, 0.0, 0.064], dtype=np.float64)
    scoop_under = np.array([-0.020, 0.0, 0.064], dtype=np.float64)
    scoop_lift = scoop_under + np.array([0.0, 0.0, 0.026])

    _set_gripper(model, data, open_center, 0.145)
    _set_scoop(model, data, scoop_far)
    for _ in range(int(0.35 / model.opt.timestep)):
        controller.apply(model, data, phase="observe")
        mujoco.mj_step(model, data)

    initial = measure_shape_metrics(model, data, phase="observe", target_index=target_index)
    rows.append(initial)
    _render(model, data, renderer, out_dir / "observe.png", camera)

    _step_phase(
        model,
        data,
        controller,
        phase="approach",
        seconds=0.45,
        gripper_start=open_center,
        gripper_end=pinch_center,
        gap_start=0.145,
        gap_end=0.105,
        scoop_start=scoop_far,
        scoop_end=scoop_far,
        renderer=renderer,
        video_frames=video_frames,
        camera=camera,
    )
    rows.append(measure_shape_metrics(model, data, phase="approach", target_index=target_index, initial=initial))
    _render(model, data, renderer, out_dir / "approach.png", camera)

    _step_phase(
        model,
        data,
        controller,
        phase="pinch",
        seconds=0.75,
        gripper_start=pinch_center,
        gripper_end=pinch_center,
        gap_start=0.105,
        gap_end=0.036,
        scoop_start=scoop_far,
        scoop_end=scoop_far,
        renderer=renderer,
        video_frames=video_frames,
        camera=camera,
    )
    rows.append(measure_shape_metrics(model, data, phase="pinch", target_index=target_index, initial=initial))
    _render(model, data, renderer, out_dir / "pinch_shape_change.png", camera)

    _step_phase(
        model,
        data,
        controller,
        phase="micro_lift",
        seconds=0.85,
        gripper_start=pinch_center,
        gripper_end=lift_center,
        gap_start=0.036,
        gap_end=0.036,
        scoop_start=scoop_far,
        scoop_end=scoop_far,
        renderer=renderer,
        video_frames=video_frames,
        camera=camera,
    )
    rows.append(measure_shape_metrics(model, data, phase="micro_lift", target_index=target_index, initial=initial))
    _render(model, data, renderer, out_dir / "micro_lift_sag.png", camera)

    _step_phase(
        model,
        data,
        controller,
        phase="scoop_insert",
        seconds=0.95,
        gripper_start=lift_center,
        gripper_end=lift_center,
        gap_start=0.036,
        gap_end=0.036,
        scoop_start=scoop_far,
        scoop_end=scoop_under,
        renderer=renderer,
        video_frames=video_frames,
        camera=camera,
    )
    rows.append(measure_shape_metrics(model, data, phase="scoop_insert", target_index=target_index, initial=initial))
    _render(model, data, renderer, out_dir / "scoop_support.png", camera)

    _step_phase(
        model,
        data,
        controller,
        phase="support_lift",
        seconds=0.75,
        gripper_start=lift_center,
        gripper_end=lift_center + np.array([0.020, 0.0, 0.018]),
        gap_start=0.036,
        gap_end=0.038,
        scoop_start=scoop_under,
        scoop_end=scoop_lift,
        renderer=renderer,
        video_frames=video_frames,
        camera=camera,
    )
    rows.append(measure_shape_metrics(model, data, phase="support_lift", target_index=target_index, initial=initial))
    _render(model, data, renderer, out_dir / "support_lift_recovered_shape.png", camera)

    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    imageio = _imageio()
    for idx, frame in enumerate(video_frames):
        imageio.imwrite(frame_dir / f"frame_{idx:04d}.png", frame)
    video_path: Path | None = None
    if save_video and video_frames:
        video_path = out_dir / "shape_response_demo.mp4"
        try:
            imageio.mimsave(video_path, video_frames, fps=30)
        except Exception:
            video_path = None

    _write_summary(rows, out_dir / "shape_response_summary.csv")
    _write_markdown(rows, out_dir / "shape_response_summary.md", scenario=scenario, target_site=target_site, video_path=video_path)
    renderer.close()

    final = rows[-1]
    print(f"scene_xml={xml_path}")
    print(f"out_dir={out_dir}")
    print(f"target_site={target_site}")
    print(f"upper_width_delta_m={final.upper_half_width - initial.upper_half_width:.5f}")
    print(f"lower_width_delta_m={final.lower_half_width - initial.lower_half_width:.5f}")
    print(f"local_shoulder_angle_delta_deg={final.shoulder_angle_local_deg - initial.shoulder_angle_local_deg:.2f}")
    print(f"bottom_sag_delta_m={final.bottom_sag_m - initial.bottom_sag_m:.5f}")
    print(f"payload_y_delta_m={final.payload_slide_y_m - initial.payload_slide_y_m:.5f}")
    print(f"summary_csv={out_dir / 'shape_response_summary.csv'}")
    print(f"summary_md={out_dir / 'shape_response_summary.md'}")
    if video_path:
        print(f"video={video_path}")
    else:
        print(f"frames={frame_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Show robot-contact-driven shape response for the sack surrogate")
    parser.add_argument("--scenario", choices=available_scenarios(), default="underfilled")
    parser.add_argument("--post-release", action="store_true")
    parser.add_argument("--camera", choices=("overview", "front", "side"), default="overview")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    return run_demo(args.scenario, post_release=args.post_release, save_video=not args.no_video, camera=args.camera)


if __name__ == "__main__":
    raise SystemExit(main())
