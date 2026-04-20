from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

from .evaluation import Evaluator, EpisodeMetrics
from .scene_generator import EpisodeScene
from .visualization import Visualizer


@dataclass
class SettleReport:
    """5초 settle 이후 장면 안정성을 요약한다."""

    settle_seconds: float
    settle_steps: int
    stable: bool
    max_linear_speed: float
    max_angular_speed: float
    max_position_drift: float
    min_body_height: float
    failure_tags: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


class SackPileEnv:
    """benchmark case 실행, baseline trajectory 적용, 로그 저장을 담당한다."""

    def __init__(self, scene: EpisodeScene, log_dir: Path):
        self.scene = scene
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.model = mujoco.MjModel.from_xml_path(str(scene.xml_path))
        self.data = mujoco.MjData(self.model)
        self.evaluator = Evaluator(self.model)
        self.visualizer = Visualizer(self.model)

        self.gripper_mocap_id = self.model.body_mocapid[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper_mocap")]
        self.scoop_mocap_id = self.model.body_mocapid[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "scoop_mocap")]
        self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, scene.target_name)
        self.sack_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, sack.name)
            for sack in self.scene.sacks
        ]
        self.left_finger_act = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_finger_act")
        self.right_finger_act = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_finger_act")
        self.gripper_ctrl_joint_names = [
            "gripper_ctrl_x",
            "gripper_ctrl_y",
            "gripper_ctrl_z",
            "gripper_ctrl_roll",
            "gripper_ctrl_pitch",
            "gripper_ctrl_yaw",
        ]
        self.scoop_ctrl_joint_names = [
            "scoop_ctrl_x",
            "scoop_ctrl_y",
            "scoop_ctrl_z",
            "scoop_ctrl_roll",
            "scoop_ctrl_pitch",
            "scoop_ctrl_yaw",
        ]
        self.gripper_ctrl_act_names = [
            "gripper_ctrl_x_act",
            "gripper_ctrl_y_act",
            "gripper_ctrl_z_act",
            "gripper_ctrl_roll_act",
            "gripper_ctrl_pitch_act",
            "gripper_ctrl_yaw_act",
        ]
        self.scoop_ctrl_act_names = [
            "scoop_ctrl_x_act",
            "scoop_ctrl_y_act",
            "scoop_ctrl_z_act",
            "scoop_ctrl_roll_act",
            "scoop_ctrl_pitch_act",
            "scoop_ctrl_yaw_act",
        ]
        self.gripper_ctrl_qpos_adr = [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in self.gripper_ctrl_joint_names]
        self.scoop_ctrl_qpos_adr = [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in self.scoop_ctrl_joint_names]
        self.gripper_ctrl_act_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.gripper_ctrl_act_names]
        self.scoop_ctrl_act_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in self.scoop_ctrl_act_names]

        self.open_width = 0.040
        self.closed_width = 0.012
        self.pre_lift_target_pos: np.ndarray | None = None
        self.pre_lift_scoop_pos: np.ndarray | None = None
        self.target_origin_xy = np.zeros(2, dtype=np.float64)
        self.settle_report: SettleReport | None = None

    def reset(self, settle_seconds: float = 5.0, verify_stability: bool = True, viewer=None, sleep: bool = False):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.set_gripper_pose(np.array([0.36, -0.33, 0.30]), self.euler_to_quat(np.array([0.0, -np.pi / 2.0, 0.0])))
        self.set_scoop_pose(np.array([0.42, 0.28, 0.20]), self.euler_to_quat(np.array([0.0, 0.0, 0.0])))
        self.set_gripper_width(self.open_width)
        # scene generator가 만든 pile이 5초 안에 안정화되는지 자동으로 검사한다.
        if verify_stability:
            self.settle_report = self.settle_and_check(settle_seconds=settle_seconds, viewer=viewer, sleep=sleep)
        else:
            self.step(max(1, int(round(settle_seconds / self.model.opt.timestep))), viewer=viewer, sleep=sleep)
            self.settle_report = None
        self.target_origin_xy = self.data.xpos[self.target_body_id][:2].copy()
        self.pre_lift_target_pos = None
        self.pre_lift_scoop_pos = None
        return self.settle_report

    def set_gripper_pose(self, pos: np.ndarray, quat: np.ndarray):
        self._set_ctrl_state(self.gripper_ctrl_qpos_adr, self.gripper_ctrl_act_ids, pos, self.quat_to_euler(quat))
        self.data.mocap_pos[self.gripper_mocap_id] = pos
        self.data.mocap_quat[self.gripper_mocap_id] = quat

    def set_scoop_pose(self, pos: np.ndarray, quat: np.ndarray):
        self._set_ctrl_state(self.scoop_ctrl_qpos_adr, self.scoop_ctrl_act_ids, pos, self.quat_to_euler(quat))
        self.data.mocap_pos[self.scoop_mocap_id] = pos
        self.data.mocap_quat[self.scoop_mocap_id] = quat

    def set_gripper_width(self, width: float):
        self.data.ctrl[self.left_finger_act] = width
        self.data.ctrl[self.right_finger_act] = width

    def step(self, steps: int = 1, viewer=None, sleep: bool = False):
        for _ in range(steps):
            self._sync_mocap_from_ctrl_joints()
            mujoco.mj_step(self.model, self.data)
            if viewer is not None:
                self.render_viewer(viewer)
                if sleep:
                    time.sleep(self.model.opt.timestep)

    def render_viewer(self, viewer):
        """passive viewer와 상태를 동기화해 마우스 perturb 입력을 반영한다."""

        metrics = self.peek_metrics()
        if hasattr(viewer, "lock"):
            with viewer.lock():
                self.visualizer.update(viewer, self.data, metrics, self.scene.target_name)
        else:
            self.visualizer.update(viewer, self.data, metrics, self.scene.target_name)
        if hasattr(viewer, "sync"):
            viewer.sync()

    def move_mocap_linear(self, tool: str, target_pos: np.ndarray, target_quat: np.ndarray, steps: int, viewer=None):
        mocap_id = self.gripper_mocap_id if tool == "gripper" else self.scoop_mocap_id
        start_pos = self.data.mocap_pos[mocap_id].copy()
        start_quat = self.data.mocap_quat[mocap_id].copy()
        for alpha in np.linspace(0.0, 1.0, steps):
            self.data.mocap_pos[mocap_id] = (1.0 - alpha) * start_pos + alpha * target_pos
            quat = start_quat + alpha * (target_quat - start_quat)
            self.data.mocap_quat[mocap_id] = quat / np.linalg.norm(quat)
            self.step(1, viewer=viewer)

    def target_state(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.xpos[self.target_body_id].copy(), self.data.xmat[self.target_body_id].reshape(3, 3).copy()

    def target_site(self, suffix: str) -> np.ndarray:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"{self.scene.target_name}_{suffix}")
        return self.data.site_xpos[site_id].copy()

    def get_joint_qpos(self, joint_name: str) -> float:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        adr = self.model.jnt_qposadr[joint_id]
        return float(self.data.qpos[adr])

    def set_joint_qpos(self, joint_name: str, value: float):
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        adr = self.model.jnt_qposadr[joint_id]
        self.data.qpos[adr] = value
        mujoco.mj_forward(self.model, self.data)

    def animate_joint_targets(self, joint_targets: dict[str, float], steps: int, viewer=None, sleep: bool = False):
        start_values = {name: self.get_joint_qpos(name) for name in joint_targets}
        for alpha in np.linspace(0.0, 1.0, max(2, steps)):
            for name, target in joint_targets.items():
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                adr = self.model.jnt_qposadr[joint_id]
                self.data.qpos[adr] = (1.0 - alpha) * start_values[name] + alpha * target
            mujoco.mj_forward(self.model, self.data)
            self.step(1, viewer=viewer, sleep=sleep)

    def get_free_body_pose(self, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        joint_name = f"{body_name}_free"
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        adr = self.model.jnt_qposadr[joint_id]
        pos = self.data.qpos[adr : adr + 3].copy()
        quat = self.data.qpos[adr + 3 : adr + 7].copy()
        return pos, quat

    def set_free_body_pose(self, body_name: str, pos: np.ndarray, quat: np.ndarray):
        joint_name = f"{body_name}_free"
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        adr = self.model.jnt_qposadr[joint_id]
        self.data.qpos[adr : adr + 3] = pos
        self.data.qpos[adr + 3 : adr + 7] = quat / np.linalg.norm(quat)
        mujoco.mj_forward(self.model, self.data)

    def animate_free_body_pose(
        self,
        body_name: str,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        steps: int,
        viewer=None,
        sleep: bool = False,
    ):
        start_pos, start_quat = self.get_free_body_pose(body_name)
        target_quat = target_quat / np.linalg.norm(target_quat)
        for alpha in np.linspace(0.0, 1.0, max(2, steps)):
            quat = start_quat + alpha * (target_quat - start_quat)
            quat = quat / np.linalg.norm(quat)
            self.set_free_body_pose(body_name, (1.0 - alpha) * start_pos + alpha * target_pos, quat)
            self.step(1, viewer=viewer, sleep=sleep)

    def set_sack_visual_emphasis(
        self,
        sack_name: str,
        static_alpha: float = 0.30,
        shell_alpha: float = 0.10,
        dynamic_alpha: float = 0.58,
    ):
        for geom_id in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if not geom_name.startswith(f"{sack_name}_"):
                continue
            if geom_name.endswith("_visual"):
                if "_deform_" in geom_name:
                    self.model.geom_rgba[geom_id, 3] = dynamic_alpha
                elif geom_name == f"{sack_name}_visual":
                    self.model.geom_rgba[geom_id, 3] = static_alpha
                else:
                    self.model.geom_rgba[geom_id, 3] = shell_alpha

    def set_neighbor_visual_emphasis(self, alpha: float = 0.80):
        target_prefix = f"{self.scene.target_name}_"
        for geom_id in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if geom_name.startswith(target_prefix):
                continue
            if geom_name.endswith("_visual"):
                self.model.geom_rgba[geom_id, 3] = alpha

    def peek_metrics(self) -> EpisodeMetrics:
        return self.evaluator.evaluate(
            self.data,
            target_name=self.scene.target_name,
            target_origin_xy=self.target_origin_xy,
            pre_lift_pos=self.pre_lift_target_pos,
            pre_lift_scoop_pos=self.pre_lift_scoop_pos,
        )

    def finalize_metrics(self) -> EpisodeMetrics:
        return self.peek_metrics()

    def mark_pre_lift_state(self):
        self.pre_lift_target_pos = self.data.xpos[self.target_body_id].copy()
        scoop_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "scoop_tool")
        self.pre_lift_scoop_pos = self.data.xpos[scoop_body_id].copy()

    def settle_and_check(self, settle_seconds: float = 5.0, viewer=None, sleep: bool = False) -> SettleReport:
        """5초 동안 장면을 가라앉힌 뒤 안정성을 판정한다."""

        settle_steps = max(1, int(round(settle_seconds / self.model.opt.timestep)))
        window_steps = max(40, int(round(0.5 / self.model.opt.timestep)))
        position_window: list[np.ndarray] = []
        max_linear_speed = 0.0
        max_angular_speed = 0.0
        non_finite_state = False

        for step_idx in range(settle_steps):
            self.step(1, viewer=viewer, sleep=sleep)
            if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
                non_finite_state = True
                break

            if step_idx >= settle_steps - window_steps:
                positions = np.array([self.data.xpos[body_id].copy() for body_id in self.sack_body_ids], dtype=np.float64)
                position_window.append(positions)
                for body_id in self.sack_body_ids:
                    cvel = self.data.cvel[body_id]
                    max_angular_speed = max(max_angular_speed, float(np.linalg.norm(cvel[:3])))
                    max_linear_speed = max(max_linear_speed, float(np.linalg.norm(cvel[3:])))

        if position_window:
            start_positions = position_window[0]
            end_positions = position_window[-1]
            max_position_drift = float(np.max(np.linalg.norm(end_positions - start_positions, axis=1)))
            min_body_height = float(np.min(end_positions[:, 2]))
        else:
            final_positions = np.array([self.data.xpos[body_id].copy() for body_id in self.sack_body_ids], dtype=np.float64)
            max_position_drift = 0.0
            min_body_height = float(np.min(final_positions[:, 2]))

        failure_tags: list[str] = []
        if non_finite_state:
            failure_tags.append("non_finite_state")
        if max_linear_speed > 0.080 or max_angular_speed > 1.60:
            failure_tags.append("residual_motion")
        if max_position_drift > 0.030:
            failure_tags.append("drift_after_settle")
        if min_body_height < 0.045:
            failure_tags.append("body_too_low")

        return SettleReport(
            settle_seconds=settle_seconds,
            settle_steps=settle_steps,
            stable=not failure_tags,
            max_linear_speed=max_linear_speed,
            max_angular_speed=max_angular_speed,
            max_position_drift=max_position_drift,
            min_body_height=min_body_height,
            failure_tags=failure_tags,
        )

    def _sync_mocap_from_ctrl_joints(self):
        gripper_pos, gripper_euler = self._get_ctrl_state(self.gripper_ctrl_qpos_adr)
        scoop_pos, scoop_euler = self._get_ctrl_state(self.scoop_ctrl_qpos_adr)
        self.data.mocap_pos[self.gripper_mocap_id] = gripper_pos
        self.data.mocap_quat[self.gripper_mocap_id] = self.euler_to_quat(gripper_euler)
        self.data.mocap_pos[self.scoop_mocap_id] = scoop_pos
        self.data.mocap_quat[self.scoop_mocap_id] = self.euler_to_quat(scoop_euler)

    def _set_ctrl_state(self, qpos_adrs, act_ids, pos: np.ndarray, euler: np.ndarray):
        values = [pos[0], pos[1], pos[2], euler[0], euler[1], euler[2]]
        for adr, act_id, value in zip(qpos_adrs, act_ids, values):
            self.data.qpos[adr] = value
            self.data.ctrl[act_id] = value

    def _get_ctrl_state(self, qpos_adrs):
        values = np.array([self.data.qpos[adr] for adr in qpos_adrs], dtype=np.float64)
        return values[:3], values[3:]

    def save_episode_log(self, baseline_name: str, metrics: EpisodeMetrics):
        payload = {
            "episode_id": self.scene.episode_id,
            "seed": self.scene.seed,
            "baseline": baseline_name,
            "benchmark": {
                "name": self.scene.benchmark_name,
                "scope": "task-driven benchmark",
                "research_question": self.scene.research_question,
                "target_case": self.scene.target_case.to_dict() if self.scene.target_case is not None else None,
            },
            "settle_report": self.settle_report.to_dict() if self.settle_report is not None else None,
            "target_name": self.scene.target_name,
            "target_variant": self.scene.target_variant,
            "target_pile_difficulty": self.scene.target_pile_difficulty,
            "scene_xml": str(self.scene.xml_path),
            "sacks": [
                {
                    "name": sack.name,
                    "variant": sack.variant.name,
                    "pos": sack.pos,
                    "euler": sack.euler,
                    "exposed_face": sack.exposed_face,
                    "stack_level": sack.stack_level,
                    "is_target": sack.is_target,
                    "mesh_file": sack.mesh_file,
                    "mesh_scale": sack.mesh_scale,
                    "fill_ratio": sack.fill_ratio,
                    "top_collapse": sack.top_collapse,
                    "side_bulge": sack.side_bulge,
                    "flattening": sack.flattening,
                    "pile_difficulty": sack.pile_difficulty,
                    "uncertainty_tags": list(sack.uncertainty_tags),
                    "benchmark_case_id": sack.benchmark_case_id,
                }
                for sack in self.scene.sacks
            ],
            "metrics": metrics.to_dict(),
        }
        out_path = self.log_dir / f"{self.scene.episode_id}_{baseline_name}.json"
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        history_path = self.log_dir / "episode_history.jsonl"
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def euler_to_quat(euler_xyz: np.ndarray) -> np.ndarray:
        roll, pitch, yaw = euler_xyz
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        quat = np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            dtype=np.float64,
        )
        return quat / np.linalg.norm(quat)

    @staticmethod
    def quat_to_euler(quat: np.ndarray) -> np.ndarray:
        w, x, y, z = quat
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            pitch = np.sign(sinp) * (np.pi / 2.0)
        else:
            pitch = np.arcsin(sinp)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return np.array([roll, pitch, yaw], dtype=np.float64)
