from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import mujoco
import numpy as np

from .environment import SackPileEnv
from .phenomenon_presets import PHENOMENON_PRESETS, select_scene_for_phenomenon
from .scene_generator import EpisodeScene


@dataclass
class DemoConfig:
    phenomenon: str
    fps: int = 24
    width: int = 1280
    height: int = 720


class VideoRecorder:
    """MuJoCo offscreen frame를 mp4로 저장한다."""

    def __init__(self, model: mujoco.MjModel, out_path: Path, fps: int, width: int, height: int):
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.renderer = mujoco.Renderer(model, width=width, height=height)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.out_path), fourcc, fps, (width, height))
        self.width = width
        self.height = height

    def write(self, model: mujoco.MjModel, data: mujoco.MjData, overlay_lines: list[str]):
        self.renderer.update_scene(data, camera="overview")
        frame = self.renderer.render()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = _draw_overlay(frame, overlay_lines)
        self.writer.write(frame)

    def close(self):
        self.writer.release()
        self.renderer.close()


def _draw_overlay(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    if not lines:
        return frame
    out = frame.copy()
    line_h = 30
    pad = 14
    box_h = pad * 2 + line_h * len(lines)
    box_w = 560
    cv2.rectangle(out, (16, 16), (16 + box_w, 16 + box_h), (20, 24, 28), thickness=-1)
    cv2.rectangle(out, (16, 16), (16 + box_w, 16 + box_h), (140, 170, 190), thickness=2)
    for idx, line in enumerate(lines):
        y = 16 + pad + 22 + idx * line_h
        scale = 0.72 if idx == 0 else 0.62
        color = (235, 245, 250) if idx == 0 else (220, 232, 238)
        cv2.putText(out, line, (28, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
    return out


class DemoRuntime:
    """viewer 또는 offscreen recorder를 통해 자동 재생 데모를 실행한다."""

    def __init__(self, env: SackPileEnv, phenomenon: str, recorder: VideoRecorder | None = None, viewer=None, fps: int = 24):
        self.env = env
        self.phenomenon = phenomenon
        self.recorder = recorder
        self.viewer = viewer
        self.fps = fps
        self.steps_per_frame = max(1, int(round(1.0 / (fps * self.env.model.opt.timestep))))
        self.phase_label = "before"
        self.overlay_fn = None
        self.note_lines: list[str] = []
        self.initial_top_height = float(env.target_site("top_site")[2])
        self.initial_center_height = float(env.target_site("center_site")[2])
        self.initial_target_pos, self.initial_target_xmat = env.target_state()
        self.target_name = env.scene.target_name
        self.target_neighbor_name = self._find_nearest_neighbor()
        self.reference_neighbor_xy = self._neighbor_xy()
        self.reference_target_xy = self.env.target_state()[0][:2].copy()
        self.pre_response_top_height = self.initial_top_height
        self.pre_response_center_height = self.initial_center_height
        self.pre_response_target_pos = self.initial_target_pos.copy()
        self.pre_response_neighbor_xy = self.reference_neighbor_xy.copy() if self.reference_neighbor_xy is not None else None

    def emphasize_visuals(self):
        self.env.set_sack_visual_emphasis(self.target_name, static_alpha=0.22, shell_alpha=0.08, dynamic_alpha=0.64)
        self.env.set_neighbor_visual_emphasis(alpha=0.74)

    def save_reference_state(self):
        target_pos, _ = self.env.target_state()
        self.pre_response_top_height = float(self.env.target_site("top_site")[2])
        self.pre_response_center_height = float(self.env.target_site("center_site")[2])
        self.pre_response_target_pos = target_pos.copy()
        self.pre_response_neighbor_xy = self._neighbor_xy()

    def hold(self, seconds: float, phase: str):
        frames = max(1, int(round(seconds * self.fps)))
        self.phase_label = phase
        for _ in range(frames):
            self.env.step(self.steps_per_frame, viewer=self.viewer, sleep=self.viewer is not None)
            self._capture()

    def play_frames(self, seconds: float, phase: str, updater):
        frames = max(1, int(round(seconds * self.fps)))
        self.phase_label = phase
        for frame_idx in range(frames):
            alpha = (frame_idx + 1) / frames
            updater(alpha)
            self.env.step(self.steps_per_frame, viewer=self.viewer, sleep=self.viewer is not None)
            self._capture()

    def move_tool(self, tool: str, target_pos: np.ndarray, target_quat: np.ndarray, seconds: float, phase: str):
        mocap_id = self.env.gripper_mocap_id if tool == "gripper" else self.env.scoop_mocap_id
        start_pos = self.env.data.mocap_pos[mocap_id].copy()
        start_quat = self.env.data.mocap_quat[mocap_id].copy()
        target_quat = target_quat / np.linalg.norm(target_quat)

        def updater(alpha: float):
            pos = (1.0 - alpha) * start_pos + alpha * target_pos
            quat = start_quat + alpha * (target_quat - start_quat)
            quat = quat / np.linalg.norm(quat)
            if tool == "gripper":
                self.env.set_gripper_pose(pos, quat)
            else:
                self.env.set_scoop_pose(pos, quat)

        self.play_frames(seconds, phase, updater)

    def animate_joints(self, joint_targets: dict[str, float], seconds: float, phase: str):
        start_values = {name: self.env.get_joint_qpos(name) for name in joint_targets}

        def updater(alpha: float):
            for name, target in joint_targets.items():
                self.env.set_joint_qpos(name, (1.0 - alpha) * start_values[name] + alpha * target)

        self.play_frames(seconds, phase, updater)

    def animate_body_pose(self, body_name: str, target_pos: np.ndarray, target_quat: np.ndarray, seconds: float, phase: str):
        start_pos, start_quat = self.env.get_free_body_pose(body_name)
        target_quat = target_quat / np.linalg.norm(target_quat)

        def updater(alpha: float):
            quat = start_quat + alpha * (target_quat - start_quat)
            quat = quat / np.linalg.norm(quat)
            pos = (1.0 - alpha) * start_pos + alpha * target_pos
            self.env.set_free_body_pose(body_name, pos, quat)

        self.play_frames(seconds, phase, updater)

    def _capture(self):
        if self.recorder is None:
            return
        self.recorder.write(self.env.model, self.env.data, self.overlay_lines())

    def overlay_lines(self) -> list[str]:
        metrics = phenomenon_metrics(self)
        phenomenon_title = PHENOMENON_PRESETS.get(self.phenomenon)
        title = phenomenon_title.label_ko if phenomenon_title is not None else self.phenomenon
        phase_text = self.phase_label.replace("_", " ")
        lines = [f"{title} | phase={phase_text}"]
        for key, value in metrics:
            lines.append(f"{key}: {value}")
        return lines

    def _find_nearest_neighbor(self) -> str | None:
        target_pos, _ = self.env.target_state()
        best_name = None
        best_dist = float("inf")
        for sack in self.env.scene.sacks:
            if sack.is_target:
                continue
            body_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, sack.name)
            pos = self.env.data.xpos[body_id].copy()
            dist = float(np.linalg.norm(pos[:2] - target_pos[:2]))
            if dist < best_dist:
                best_dist = dist
                best_name = sack.name
        return best_name

    def _neighbor_xy(self) -> np.ndarray | None:
        if self.target_neighbor_name is None:
            return None
        body_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, self.target_neighbor_name)
        return self.env.data.xpos[body_id][:2].copy()


def _proxy_metric_value(value: float, scale: float = 1.0, lower: float = 0.0) -> float:
    return max(lower, float(value * scale))


def phenomenon_metrics(rt: DemoRuntime) -> list[tuple[str, str]]:
    env = rt.env
    target_pos, target_xmat = env.target_state()
    top_height = float(env.target_site("top_site")[2])
    center_height = float(env.target_site("center_site")[2])
    top_z = env.get_joint_qpos(f"{rt.target_name}_top_z")
    top_y = env.get_joint_qpos(f"{rt.target_name}_top_y")
    left_side = env.get_joint_qpos(f"{rt.target_name}_left_side_slide")
    right_side = env.get_joint_qpos(f"{rt.target_name}_right_side_slide")
    bulge_z = env.get_joint_qpos(f"{rt.target_name}_bulge_z")
    bulge_y = env.get_joint_qpos(f"{rt.target_name}_bulge_y")
    bottom_z = env.get_joint_qpos(f"{rt.target_name}_bottom_z")
    metrics = env.finalize_metrics()

    if rt.phenomenon == "underfilled_slack":
        center_sag = max(0.0, rt.initial_top_height - top_height) + max(0.0, rt.initial_center_height - center_height) + max(0.0, -bulge_z)
        support_margin = max(0.0, target_pos[2] - (center_height - 0.06 + bottom_z))
        return [
            ("top_height", f"{top_height:.3f} m"),
            ("center_sag", f"{center_sag:.3f}"),
            ("support_margin", f"{support_margin:.3f}"),
        ]

    if rt.phenomenon == "top_fold_occluded":
        fold_depth = max(0.0, rt.initial_top_height - top_height) + abs(top_y) + max(0.0, -top_z)
        visible_top_area = max(0.0, min(1.0, 0.82 - 7.5 * fold_depth + 8.0 * max(0.0, top_z)))
        return [
            ("visible_top_area", f"{visible_top_area:.2f}"),
            ("fold_depth", f"{fold_depth:.3f}"),
        ]

    if rt.phenomenon == "eccentric_fill":
        roll = float(np.degrees(np.arctan2(target_xmat[2, 1], target_xmat[2, 2])))
        imbalance = np.clip(0.50 + 0.50 * math.tanh(6.0 * (left_side + right_side) + 0.04 * roll + 20.0 * bulge_y), 0.0, 1.0)
        return [
            ("left/right load share", f"{imbalance:.2f} / {1.0 - imbalance:.2f}"),
            ("tilt_angle", f"{metrics.tilt_deg:.2f} deg"),
        ]

    if rt.phenomenon == "neighbor_contact_wedge":
        neighbor_xy = rt._neighbor_xy()
        if neighbor_xy is None:
            gap_width = 0.0
            coupled = 0.0
        else:
            gap_width = max(0.0, float(np.linalg.norm(target_pos[:2] - neighbor_xy) - 0.12))
            neighbor_move = float(np.linalg.norm(neighbor_xy - rt.pre_response_neighbor_xy)) if rt.pre_response_neighbor_xy is not None else 0.0
            target_move = float(np.linalg.norm(target_pos[:2] - rt.pre_response_target_pos[:2]))
            coupled = neighbor_move / max(0.01, target_move + 0.01)
        return [
            ("gap_width", f"{gap_width:.3f} m"),
            ("coupled_motion_score", f"{coupled:.2f}"),
        ]

    if rt.phenomenon == "partial_support_sag":
        sag_index = max(0.0, rt.pre_response_top_height - top_height) + max(0.0, rt.pre_response_center_height - center_height)
        scoop_support = min(1.0, metrics.scoop_insertion_depth / 0.18)
        grip_support = max(0.05, 1.0 - metrics.slip_distance / 0.06)
        load_share_ratio = scoop_support / (scoop_support + grip_support)
        slip_risk = np.clip(0.45 * (metrics.slip_distance / 0.10) + 0.55 * (metrics.tilt_deg / 24.0), 0.0, 1.0)
        return [
            ("sag_index", f"{sag_index:.3f}"),
            ("load_share_ratio", f"{load_share_ratio:.2f}"),
            ("slip_risk", f"{slip_risk:.2f}"),
        ]

    return [
        ("support_state_score", f"{metrics.support_state_score:.2f}"),
        ("tilt_angle", f"{metrics.tilt_deg:.2f} deg"),
    ]


def _gripper_down_quat() -> np.ndarray:
    return SackPileEnv.euler_to_quat(np.array([0.0, -np.pi / 2.0, 0.0], dtype=np.float64))


def _scoop_forward_quat() -> np.ndarray:
    return SackPileEnv.euler_to_quat(np.array([0.0, 0.0, 0.0], dtype=np.float64))


def _compose_roll_quat(base_quat: np.ndarray, extra_roll: float) -> np.ndarray:
    base_euler = SackPileEnv.quat_to_euler(base_quat)
    return SackPileEnv.euler_to_quat(base_euler + np.array([extra_roll, 0.0, 0.0], dtype=np.float64))


def run_phenomenon_demo(
    *,
    base_dir: Path,
    phenomenon: str,
    seed: int,
    save_path: Path | None = None,
    viewer=None,
) -> dict:
    chosen = None
    for seed_offset in range(12):
        scene, preset, summary = select_scene_for_phenomenon(base_dir=base_dir, phenomenon=phenomenon, base_seed=seed + seed_offset)
        env = SackPileEnv(scene=scene, log_dir=base_dir / "mujoco_sack_pile" / "logs")
        settle_report = env.reset(settle_seconds=5.0, verify_stability=True, viewer=viewer, sleep=viewer is not None)
        if settle_report.stable:
            chosen = (scene, preset, summary, env, settle_report)
            break
    if chosen is None:
        scene, preset, summary = select_scene_for_phenomenon(base_dir=base_dir, phenomenon=phenomenon, base_seed=seed)
        env = SackPileEnv(scene=scene, log_dir=base_dir / "mujoco_sack_pile" / "logs")
        settle_report = env.reset(settle_seconds=5.0, verify_stability=True, viewer=viewer, sleep=viewer is not None)
    else:
        scene, preset, summary, env, settle_report = chosen
    recorder = VideoRecorder(env.model, save_path, fps=24, width=1280, height=720) if save_path is not None else None
    rt = DemoRuntime(env, phenomenon=phenomenon, recorder=recorder, viewer=viewer, fps=24)
    rt.emphasize_visuals()

    try:
        if phenomenon == "underfilled_slack":
            _demo_underfilled_slack(rt)
        elif phenomenon == "top_fold_occluded":
            _demo_top_fold_occluded(rt)
        elif phenomenon == "eccentric_fill":
            _demo_eccentric_fill(rt)
        elif phenomenon == "neighbor_contact_wedge":
            _demo_neighbor_contact_wedge(rt)
        elif phenomenon == "partial_support_sag":
            _demo_partial_support_sag(rt)
        else:
            raise ValueError(f"지원하지 않는 phenomenon: {phenomenon}")
    finally:
        if recorder is not None:
            recorder.close()

    result = {
        "phenomenon": phenomenon,
        "label_ko": preset.label_ko,
        "description": preset.explanation_ko,
        "scene_xml": str(scene.xml_path),
        "seed": summary["seed"],
        "settle_stable": settle_report.stable,
        "settle_failure_tags": settle_report.failure_tags,
        "final_metrics": env.finalize_metrics().to_dict(),
        "save_path": str(save_path) if save_path is not None else None,
    }
    return result


def _demo_underfilled_slack(rt: DemoRuntime):
    env = rt.env
    top_site = env.target_site("top_site")
    rt.hold(1.5, "before")
    env.set_gripper_width(0.030)
    rt.move_tool("gripper", top_site + np.array([0.000, -0.020, 0.090]), _gripper_down_quat(), 1.2, "stimulus")
    rt.animate_joints(
        {
            f"{rt.target_name}_top_z": -0.028,
            f"{rt.target_name}_bulge_z": -0.010,
            f"{rt.target_name}_bottom_z": -0.004,
        },
        0.8,
        "stimulus",
    )
    rt.save_reference_state()
    env.set_gripper_width(0.020)
    rt.move_tool("gripper", top_site + np.array([0.012, -0.006, 0.118]), _gripper_down_quat(), 1.2, "response")
    rt.animate_joints(
        {
            f"{rt.target_name}_top_z": 0.008,
            f"{rt.target_name}_top_y": 0.006,
            f"{rt.target_name}_bulge_z": -0.008,
            f"{rt.target_name}_bottom_z": -0.002,
        },
        1.3,
        "response",
    )
    env.set_gripper_width(env.open_width)
    rt.move_tool("gripper", top_site + np.array([0.040, -0.060, 0.160]), _gripper_down_quat(), 0.8, "after")
    rt.hold(1.4, "after")


def _demo_top_fold_occluded(rt: DemoRuntime):
    env = rt.env
    top_site = env.target_site("top_site")
    rt.hold(1.5, "before")
    env.set_gripper_width(0.034)
    rt.move_tool("gripper", top_site + np.array([-0.055, 0.000, 0.082]), _gripper_down_quat(), 1.0, "stimulus")
    rt.animate_joints(
        {
            f"{rt.target_name}_top_y": -0.016,
            f"{rt.target_name}_top_z": -0.020,
            f"{rt.target_name}_left_side_slide": -0.010,
        },
        1.0,
        "stimulus",
    )
    rt.save_reference_state()
    rt.move_tool("gripper", top_site + np.array([0.055, 0.000, 0.094]), _gripper_down_quat(), 1.2, "response")
    rt.animate_joints(
        {
            f"{rt.target_name}_top_y": 0.010,
            f"{rt.target_name}_top_z": 0.006,
            f"{rt.target_name}_left_side_slide": -0.002,
            f"{rt.target_name}_right_side_slide": -0.008,
        },
        1.3,
        "response",
    )
    rt.move_tool("gripper", top_site + np.array([0.080, 0.000, 0.150]), _gripper_down_quat(), 0.7, "after")
    rt.hold(1.3, "after")


def _demo_eccentric_fill(rt: DemoRuntime):
    env = rt.env
    target_center, _ = env.target_state()
    grip_pose = target_center + np.array([0.010, -0.012, 0.100])
    scoop_pose = target_center + np.array([-0.020, 0.000, -0.004])
    rt.hold(1.5, "before")
    env.set_gripper_width(0.024)
    rt.move_tool("gripper", grip_pose, _gripper_down_quat(), 1.1, "stimulus")
    rt.move_tool("scoop", scoop_pose, _scoop_forward_quat(), 1.1, "stimulus")
    env.set_gripper_width(env.closed_width)
    rt.hold(0.4, "stimulus")
    rt.save_reference_state()
    rt.move_tool("gripper", grip_pose + np.array([0.000, 0.000, 0.040]), _gripper_down_quat(), 1.2, "response")
    rt.move_tool("scoop", scoop_pose + np.array([0.020, 0.000, 0.038]), _scoop_forward_quat(), 1.2, "response")
    pos, quat = env.get_free_body_pose(rt.target_name)
    tilted_quat = _compose_roll_quat(quat, 0.26)
    rt.animate_body_pose(rt.target_name, pos + np.array([0.008, 0.015, 0.000]), tilted_quat, 1.0, "response")
    rt.animate_joints(
        {
            f"{rt.target_name}_left_side_slide": -0.016,
            f"{rt.target_name}_right_side_slide": 0.010,
            f"{rt.target_name}_bulge_y": 0.016,
            f"{rt.target_name}_bulge_z": 0.010,
        },
        1.0,
        "response",
    )
    rt.hold(1.5, "after")


def _demo_neighbor_contact_wedge(rt: DemoRuntime):
    env = rt.env
    target_center, _ = env.target_state()
    rt.hold(1.5, "before")
    rt.save_reference_state()
    rt.move_tool("scoop", target_center + np.array([-0.185, 0.000, -0.008]), _scoop_forward_quat(), 1.0, "stimulus")
    rt.move_tool("scoop", target_center + np.array([-0.040, 0.000, -0.004]), _scoop_forward_quat(), 1.2, "stimulus")

    if rt.target_neighbor_name is not None:
        npos, nquat = env.get_free_body_pose(rt.target_neighbor_name)
        rt.animate_body_pose(
            rt.target_neighbor_name,
            npos + np.array([0.015, -0.010, 0.000]),
            nquat,
            1.0,
            "response",
        )
    rt.move_tool("scoop", target_center + np.array([0.012, 0.018, 0.000]), _scoop_forward_quat(), 1.0, "response")
    rt.hold(1.8, "after")


def _demo_partial_support_sag(rt: DemoRuntime):
    env = rt.env
    target_center, _ = env.target_state()
    target_side = env.target_site("side_site")
    rt.hold(1.4, "before")
    env.set_gripper_width(env.open_width)
    rt.move_tool("scoop", target_center + np.array([-0.165, 0.0, -0.010]), _scoop_forward_quat(), 1.0, "stimulus")
    rt.move_tool("scoop", target_center + np.array([-0.025, 0.0, -0.004]), _scoop_forward_quat(), 1.0, "stimulus")
    rt.move_tool("gripper", target_side + np.array([0.005, -0.030, 0.070]), _gripper_down_quat(), 1.0, "stimulus")
    rt.move_tool("gripper", target_side + np.array([0.005, -0.010, 0.022]), _gripper_down_quat(), 0.8, "stimulus")
    env.set_gripper_width(env.closed_width)
    rt.hold(0.3, "stimulus")

    rt.save_reference_state()
    env.mark_pre_lift_state()
    rt.move_tool("scoop", target_center + np.array([0.020, 0.0, 0.025]), _scoop_forward_quat(), 1.2, "response")
    rt.move_tool("gripper", target_side + np.array([0.025, 0.0, 0.090]), _gripper_down_quat(), 1.2, "response")
    rt.animate_joints(
        {
            f"{rt.target_name}_bottom_z": -0.004,
            f"{rt.target_name}_left_side_slide": -0.012,
            f"{rt.target_name}_right_side_slide": 0.008,
            f"{rt.target_name}_bulge_z": -0.004,
        },
        1.0,
        "response",
    )
    rt.hold(1.5, "after")


def select_integrated_scene(base_dir: Path, seed: int, attempt_limit: int = 64) -> tuple[EpisodeScene, dict]:
    """underfilled + top-fold-occluded + neighbor-contact를 함께 볼 수 있는 장면을 고른다."""

    for attempt_idx in range(attempt_limit):
        scene, _, summary = select_scene_for_phenomenon(
            base_dir=base_dir,
            phenomenon="top_fold_occluded",
            base_seed=seed + attempt_idx,
            attempt_limit=1,
        )
        target = next(sack for sack in scene.sacks if sack.is_target)
        near_side_bulged = None
        for sack in scene.sacks:
            if sack.is_target:
                continue
            dist = float(np.linalg.norm(np.array(sack.pos[:2]) - np.array(target.pos[:2])))
            if sack.variant.name == "side_bulged_unstable" and dist < 0.18:
                near_side_bulged = sack
                break
        if near_side_bulged is not None:
            return scene, {
                "seed": summary["seed"],
                "target_variant": target.variant.name,
                "target_pile_difficulty": target.pile_difficulty,
                "neighbor_name": near_side_bulged.name,
                "neighbor_variant": near_side_bulged.variant.name,
            }
    raise RuntimeError("통합 demo용 장면을 찾지 못했습니다.")


def run_integrated_demo(
    *,
    base_dir: Path,
    seed: int,
    save_path: Path | None = None,
    viewer=None,
) -> dict:
    chosen = None
    for seed_offset in range(12):
        scene, summary = select_integrated_scene(base_dir=base_dir, seed=seed + seed_offset)
        env = SackPileEnv(scene=scene, log_dir=base_dir / "mujoco_sack_pile" / "logs")
        settle_report = env.reset(settle_seconds=5.0, verify_stability=True, viewer=viewer, sleep=viewer is not None)
        if settle_report.stable:
            chosen = (scene, summary, env, settle_report)
            break
    if chosen is None:
        scene, summary = select_integrated_scene(base_dir=base_dir, seed=seed)
        env = SackPileEnv(scene=scene, log_dir=base_dir / "mujoco_sack_pile" / "logs")
        settle_report = env.reset(settle_seconds=5.0, verify_stability=True, viewer=viewer, sleep=viewer is not None)
    else:
        scene, summary, env, settle_report = chosen
    recorder = VideoRecorder(env.model, save_path, fps=24, width=1280, height=720) if save_path is not None else None
    rt = DemoRuntime(env, phenomenon="integrated_case", recorder=recorder, viewer=viewer, fps=24)
    rt.emphasize_visuals()
    rt.target_neighbor_name = summary["neighbor_name"]
    try:
        _demo_integrated_case(rt)
    finally:
        if recorder is not None:
            recorder.close()

    return {
        "scene_xml": str(scene.xml_path),
        "seed": summary["seed"],
        "settle_stable": settle_report.stable,
        "settle_failure_tags": settle_report.failure_tags,
        "save_path": str(save_path) if save_path is not None else None,
        "final_metrics": env.finalize_metrics().to_dict(),
    }


def _demo_integrated_case(rt: DemoRuntime):
    env = rt.env
    target_center, _ = env.target_state()
    target_side = env.target_site("side_site")
    rt.hold(1.4, "before")

    # 형상 정리
    env.set_gripper_width(0.034)
    rt.move_tool("gripper", env.target_site("top_site") + np.array([-0.045, 0.000, 0.085]), _gripper_down_quat(), 0.9, "shape_tidy")
    rt.animate_joints(
        {
            f"{rt.target_name}_top_y": 0.010,
            f"{rt.target_name}_top_z": 0.004,
        },
        0.8,
        "shape_tidy",
    )

    # 틈 생성
    rt.save_reference_state()
    rt.move_tool("scoop", target_center + np.array([-0.180, 0.0, -0.008]), _scoop_forward_quat(), 0.9, "gap_creation")
    rt.move_tool("scoop", target_center + np.array([-0.030, 0.0, -0.004]), _scoop_forward_quat(), 1.0, "gap_creation")
    if rt.target_neighbor_name is not None:
        npos, nquat = env.get_free_body_pose(rt.target_neighbor_name)
        rt.animate_body_pose(rt.target_neighbor_name, npos + np.array([0.018, -0.012, 0.000]), nquat, 0.8, "gap_creation")

    # 부분 지지 형성
    rt.move_tool("gripper", target_side + np.array([0.010, -0.012, 0.022]), _gripper_down_quat(), 0.8, "support_formation")
    env.set_gripper_width(env.closed_width)
    env.mark_pre_lift_state()
    rt.move_tool("gripper", target_side + np.array([0.022, 0.0, 0.088]), _gripper_down_quat(), 1.0, "support_formation")
    rt.move_tool("scoop", target_center + np.array([0.025, 0.0, 0.028]), _scoop_forward_quat(), 1.0, "support_formation")
    rt.animate_joints(
        {
            f"{rt.target_name}_bottom_z": -0.004,
            f"{rt.target_name}_left_side_slide": -0.010,
            f"{rt.target_name}_right_side_slide": 0.006,
            f"{rt.target_name}_bulge_z": -0.003,
        },
        0.9,
        "stabilize",
    )
    rt.hold(1.4, "stabilize")
