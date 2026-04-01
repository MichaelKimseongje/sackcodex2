from __future__ import annotations

import json
import time
from pathlib import Path

import mujoco
import numpy as np

from .evaluation import Evaluator, EpisodeMetrics
from .scene_generator import EpisodeScene
from .visualization import Visualizer


class SackPileEnv:
    """scene 실행, baseline trajectory 적용, 로그 저장을 담당한다."""

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

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.set_gripper_pose(np.array([0.36, -0.33, 0.30]), self.euler_to_quat(np.array([0.0, -np.pi / 2.0, 0.0])))
        self.set_scoop_pose(np.array([0.42, 0.28, 0.20]), self.euler_to_quat(np.array([0.0, 0.0, 0.0])))
        self.set_gripper_width(self.open_width)
        # 초기 hover/tilt 상태에서 pile이 먼저 가라앉도록 충분히 settle시킨다.
        self.step(320)
        self.target_origin_xy = self.data.xpos[self.target_body_id][:2].copy()
        self.pre_lift_target_pos = None
        self.pre_lift_scoop_pos = None

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
                metrics = self.peek_metrics()
                self.visualizer.update(viewer, self.data, metrics, self.scene.target_name)
                if sleep:
                    time.sleep(self.model.opt.timestep)

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
            "target_name": self.scene.target_name,
            "target_variant": self.scene.target_variant,
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
