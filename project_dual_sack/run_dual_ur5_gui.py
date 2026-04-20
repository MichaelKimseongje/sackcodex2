from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk

import mujoco
import mujoco.viewer
import numpy as np

from scenario_builder import (
    CONNECTED_COLUMN_COUNT,
    INNER_BOTTOM_PANEL_COUNT,
    OUTER_BACK_COUNT,
    OUTER_BOTTOM_EDGE_COUNT,
    OUTER_FRONT_COUNT,
    OUTER_LOWER_COUNT,
    OUTER_SHOULDER_COUNT,
    OUTER_SIDE_COUNT,
    OUT_DIR,
    SCENARIO_NAMES,
    TOP_SEAM_COUNT,
    write_scene_xml,
)


POSE_SCHEMA = "project_dual_sack_ur5e_gui_pose_v1"
POSE_DIR = OUT_DIR / "poses"
JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
ARM_PREFIXES = {
    "left": "ur5e_2f140",
    "right": "ur5e_scoop",
}
EE_SITE_NAMES = {
    "left": "robotiq_2f140_center_site",
    "right": "scoop_tip_site",
}
EE_LIMITS = {
    "x": (-0.75, 0.75),
    "y": (-0.85, 0.45),
    "z": (0.02, 0.65),
}
ROBOTIQ_2F140_MAX_GAP_M = 0.140
ROBOTIQ_2F140_MIN_GAP_M = 0.004
TOP_CENTER_SITE = f"site_top_seam_{TOP_SEAM_COUNT // 2:02d}"
TOP_RAIL_SITE = "site_top_grasp_rail_center"
PANEL_CENTER_INDEX = TOP_SEAM_COUNT // 2
LOWER_LEFT_SITE = f"site_lower_left_{PANEL_CENTER_INDEX:02d}"
LOWER_RIGHT_SITE = f"site_lower_right_{PANEL_CENTER_INDEX:02d}"
BOTTOM_CENTER_SITE = "site_bottom_center"
SHOULDER_CENTER_SITE_LEFT = f"site_upper_left_{PANEL_CENTER_INDEX:02d}"
SHOULDER_CENTER_SITE_RIGHT = f"site_upper_right_{PANEL_CENTER_INDEX:02d}"


@dataclass
class JointControl:
    arm: str
    joint_name: str
    actuator_name: str
    target_var: tk.DoubleVar
    current_var: tk.StringVar


@dataclass
class EEControl:
    arm: str
    site_name: str
    target_vars: dict[str, tk.DoubleVar]
    current_var: tk.StringVar


@dataclass
class AdaptiveJointBaseline:
    joint_name: str
    joint_id: int
    dof_id: int
    qpos_id: int
    joint_type: int
    stiffness: float
    damping: float
    joint_range: np.ndarray


class DualSackUR5Env:
    """GUI가 다루는 MuJoCo scene wrapper입니다."""

    def __init__(self, *, scenario: str) -> None:
        self.scenario = scenario
        self.scene_path = write_scene_xml(scenario, include_robots=True)
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        self.show_physics = True
        self.show_inner = False
        self.show_ballast = False
        self.show_visual_skin = False
        self.show_robot_collision_proxy = False
        self.show_contacts = True
        self.show_contact_forces = False
        self.show_site_frames = True
        self.show_body_frames = False
        self.show_site_labels = False
        self.show_transparent = False
        self.contact_patch_monitor_enabled = True
        self.active_compliant_patch_count = 0
        self.active_compliant_joint_count = 0
        self.paused = False
        self.left_joint_names = [f"{ARM_PREFIXES['left']}_{joint}" for joint in JOINT_NAMES]
        self.right_joint_names = [f"{ARM_PREFIXES['right']}_{joint}" for joint in JOINT_NAMES]
        self.left_actuator_names = [f"{name}_act" for name in self.left_joint_names]
        self.right_actuator_names = [f"{name}_act" for name in self.right_joint_names]
        self.gripper_actuator_names = ["finger_left_slide_act", "finger_right_slide_act"]
        self._initial_site_pos: dict[str, np.ndarray] = {}
        self._adaptive_joint_baseline = self._build_adaptive_joint_baseline()
        self._patch_site_joints = self._build_patch_site_joint_map()
        self.reset()

    def _joint_id(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"joint not found: {name}")
        return jid

    def _joint_qpos_address(self, name: str) -> int:
        return int(self.model.jnt_qposadr[self._joint_id(name)])

    def _joint_dof_address(self, name: str) -> int:
        return int(self.model.jnt_dofadr[self._joint_id(name)])

    def _actuator_id(self, name: str) -> int:
        aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise KeyError(f"actuator not found: {name}")
        return aid

    def _site_id(self, name: str) -> int:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            raise KeyError(f"site not found: {name}")
        return sid

    def _body_id(self, name: str) -> int:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(f"body not found: {name}")
        return bid

    def _maybe_joint_id(self, name: str) -> int | None:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return None if jid < 0 else int(jid)

    def _maybe_site_id(self, name: str) -> int | None:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        return None if sid < 0 else int(sid)

    def _build_adaptive_joint_baseline(self) -> dict[str, AdaptiveJointBaseline]:
        prefixes = (
            "top_grasp_rail_",
            "top_seam_band_",
            "upper_",
            "lower_",
            "bottom_",
            "top_edge_occlusion_",
        )
        baseline: dict[str, AdaptiveJointBaseline] = {}
        for jid in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if not name or not name.startswith(prefixes):
                continue
            dof_id = int(self.model.jnt_dofadr[jid])
            qpos_id = int(self.model.jnt_qposadr[jid])
            baseline[name] = AdaptiveJointBaseline(
                joint_name=name,
                joint_id=int(jid),
                dof_id=dof_id,
                qpos_id=qpos_id,
                joint_type=int(self.model.jnt_type[jid]),
                stiffness=float(self.model.jnt_stiffness[jid]),
                damping=float(self.model.dof_damping[dof_id]),
                joint_range=self.model.jnt_range[jid].copy(),
            )
        return baseline

    def _add_patch_site(self, mapping: dict[str, list[str]], site: str, joints: list[str]) -> None:
        if self._maybe_site_id(site) is None:
            return
        valid = [joint for joint in joints if self._maybe_joint_id(joint) is not None]
        if valid:
            mapping[site] = valid

    def _build_patch_site_joint_map(self) -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = {}
        for i in range(TOP_SEAM_COUNT):
            self._add_patch_site(mapping, f"site_top_seam_{i:02d}", [f"top_seam_band_{i:02d}_hinge"])
        self._add_patch_site(mapping, TOP_RAIL_SITE, ["top_grasp_rail_pitch"])
        for side in ("left", "right"):
            for i in range(TOP_SEAM_COUNT):
                self._add_patch_site(mapping, f"site_upper_{side}_{i:02d}", [f"upper_{side}_{i:02d}_hinge"])
                self._add_patch_site(mapping, f"site_shoulder_{side}_{i:02d}", [f"upper_{side}_{i:02d}_hinge"])
                self._add_patch_site(mapping, f"site_lower_{side}_{i:02d}", [f"lower_{side}_{i:02d}_hinge"])
                self._add_patch_site(mapping, f"site_bottom_edge_{side}_{i:02d}", [f"bottom_{i:02d}_hinge"])
        self._add_patch_site(mapping, BOTTOM_CENTER_SITE, [f"bottom_{PANEL_CENTER_INDEX:02d}_hinge"])
        self._add_patch_site(mapping, "site_top_edge_occlusion_left", ["top_edge_occlusion_left_hinge"])
        self._add_patch_site(mapping, "site_top_edge_occlusion_right", ["top_edge_occlusion_right_hinge"])
        return mapping

    def actuator_ctrl(self, actuator_name: str) -> float:
        return float(self.data.ctrl[self._actuator_id(actuator_name)])

    def set_actuator_ctrl(self, actuator_name: str, value: float) -> float:
        aid = self._actuator_id(actuator_name)
        low, high = self.model.actuator_ctrlrange[aid]
        clipped = float(np.clip(value, low, high))
        self.data.ctrl[aid] = clipped
        return clipped

    def set_joint_target_rad(self, joint_name: str, target_rad: float) -> None:
        self.set_actuator_ctrl(f"{joint_name}_act", target_rad)

    def add_joint_delta_rad(self, actuator_name: str, delta_rad: float) -> float:
        return self.set_actuator_ctrl(actuator_name, self.actuator_ctrl(actuator_name) + delta_rad)

    def arm_joint_names(self, arm: str) -> list[str]:
        return self.left_joint_names if arm == "left" else self.right_joint_names

    def arm_actuator_names(self, arm: str) -> list[str]:
        return self.left_actuator_names if arm == "left" else self.right_actuator_names

    def end_effector_pos(self, arm: str) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        return self.data.site_xpos[self._site_id(EE_SITE_NAMES[arm])].copy()

    def site_pos(self, name: str) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        return self.data.site_xpos[self._site_id(name)].copy()

    def gripper_gap(self) -> float:
        q_left = self.data.qpos[self._joint_qpos_address("finger_left_slide")]
        q_right = self.data.qpos[self._joint_qpos_address("finger_right_slide")]
        return float(np.clip(ROBOTIQ_2F140_MAX_GAP_M + q_left - q_right, ROBOTIQ_2F140_MIN_GAP_M, ROBOTIQ_2F140_MAX_GAP_M))

    def set_gripper_gap(self, gap_m: float, *, immediate: bool = False) -> None:
        gap = float(np.clip(gap_m, ROBOTIQ_2F140_MIN_GAP_M, ROBOTIQ_2F140_MAX_GAP_M))
        close_each = 0.5 * (ROBOTIQ_2F140_MAX_GAP_M - gap)
        left = -close_each
        right = close_each
        self.set_actuator_ctrl("finger_left_slide_act", left)
        self.set_actuator_ctrl("finger_right_slide_act", right)
        if immediate:
            self.data.qpos[self._joint_qpos_address("finger_left_slide")] = left
            self.data.qpos[self._joint_qpos_address("finger_right_slide")] = right
            mujoco.mj_forward(self.model, self.data)

    def _set_home_pose(self) -> None:
        home_deg = {
            "left": [-90.0, -80.0, 120.0, -140.0, -90.0, 0.0],
            "right": [-90.0, -80.0, 140.0, -90.0, 90.0, 0.0],
        }
        for arm, values in home_deg.items():
            for joint_name, deg in zip(self.arm_joint_names(arm), values):
                qaddr = self._joint_qpos_address(joint_name)
                rad = float(np.deg2rad(deg))
                self.data.qpos[qaddr] = rad
                self.set_joint_target_rad(joint_name, rad)
        self.set_gripper_gap(ROBOTIQ_2F140_MAX_GAP_M, immediate=True)

    def reset(self) -> None:
        self.restore_panel_compliance()
        mujoco.mj_resetData(self.model, self.data)
        self._set_home_pose()
        mujoco.mj_forward(self.model, self.data)
        self._initial_site_pos = {
            name: self.site_pos(name)
            for name in (
                TOP_CENTER_SITE,
                TOP_RAIL_SITE,
                SHOULDER_CENTER_SITE_LEFT,
                SHOULDER_CENTER_SITE_RIGHT,
                LOWER_LEFT_SITE,
                LOWER_RIGHT_SITE,
                BOTTOM_CENTER_SITE,
                f"site_bottom_edge_left_{PANEL_CENTER_INDEX:02d}",
                f"site_bottom_edge_right_{PANEL_CENTER_INDEX:02d}",
                *[f"site_inner_bottom_{i:02d}" for i in range(INNER_BOTTOM_PANEL_COUNT)],
            )
        }
        self.update_contact_adaptive_compliance()

    def step(self) -> None:
        if not self.paused:
            self.update_contact_adaptive_compliance()
            mujoco.mj_step(self.model, self.data)

    def restore_panel_compliance(self) -> None:
        for baseline in self._adaptive_joint_baseline.values():
            self.model.jnt_stiffness[baseline.joint_id] = baseline.stiffness
            self.model.dof_damping[baseline.dof_id] = baseline.damping
            self.model.jnt_range[baseline.joint_id] = baseline.joint_range
        self.active_compliant_patch_count = 0
        self.active_compliant_joint_count = 0

    def _finger_pad_positions(self) -> list[np.ndarray]:
        mujoco.mj_forward(self.model, self.data)
        positions: list[np.ndarray] = []
        for name in ("finger_left_pad_site", "finger_right_pad_site"):
            try:
                positions.append(self.data.site_xpos[self._site_id(name)].copy())
            except KeyError:
                continue
        return positions

    def _pad_contact_present(self) -> bool:
        pad_tokens = ("robotiq_2f140_left_pad", "robotiq_2f140_right_pad")
        bag_tokens = (
            "top_grasp_rail",
            "top_seam_band_",
            "upper_",
            "lower_",
            "bottom_",
            "top_edge_occlusion_",
        )
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            names = [
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or "",
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or "",
            ]
            has_pad = any(any(token in name for token in pad_tokens) for name in names)
            has_bag = any(any(name.startswith(token) or token in name for token in bag_tokens) for name in names)
            if has_pad and has_bag:
                return True
        return False

    def _active_patch_joints_from_finger_proximity(self) -> tuple[set[str], int]:
        pads = self._finger_pad_positions()
        if not pads:
            return set(), 0
        pad_contact = self._pad_contact_present()
        gap = self.gripper_gap()
        near_radius = 0.058 if pad_contact else (0.044 if gap < 0.115 else 0.032)
        ranked: list[tuple[float, str]] = []
        for site_name in self._patch_site_joints:
            sid = self._maybe_site_id(site_name)
            if sid is None:
                continue
            spos = self.data.site_xpos[sid]
            dist = min(float(np.linalg.norm(spos - pad)) for pad in pads)
            if dist <= near_radius:
                ranked.append((dist, site_name))
        ranked.sort(key=lambda item: item[0])
        active_sites = [site for _, site in ranked[:8]]
        active_joints: set[str] = set()
        for site in active_sites:
            active_joints.update(self._patch_site_joints[site])
        return active_joints, len(active_sites)

    def update_contact_adaptive_compliance(self) -> None:
        """접촉 근접 patch 수만 모니터링하고 joint range는 변경하지 않는다."""
        if not self.contact_patch_monitor_enabled:
            self.restore_panel_compliance()
            return
        active_joints, active_patch_count = self._active_patch_joints_from_finger_proximity()
        self.restore_panel_compliance()
        self.active_compliant_patch_count = active_patch_count
        self.active_compliant_joint_count = len(active_joints)
        return
        """Legacy contact-triggered joint retuning block is disabled.
            self.restore_panel_compliance()
            return
        active_joints, active_patch_count = self._active_patch_joints_from_finger_proximity()
        active_joint_count = 0
        for name, baseline in self._adaptive_joint_baseline.items():
            jid = baseline.joint_id
            qpos = float(self.data.qpos[baseline.qpos_id])
            if name in active_joints:
                active_joint_count += 1
                if baseline.joint_type == mujoco.mjtJoint.mjJNT_HINGE:
                    legacy_joint_range_change_removed = True
                    self.model.jnt_stiffness[jid] = max(0.006, baseline.stiffness * 0.10)
                    self.model.dof_damping[baseline.dof_id] = max(0.020, baseline.damping * 0.38)
                elif baseline.joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
                    span = max(float(np.max(np.abs(baseline.joint_range))), 0.060)
                    self.model.jnt_range[jid] = np.array([-span, span], dtype=np.float64)
                    self.model.jnt_stiffness[jid] = max(0.006, baseline.stiffness * 0.08)
                    self.model.dof_damping[baseline.dof_id] = max(0.018, baseline.damping * 0.32)
            else:
                low, high = baseline.joint_range
                margin = math.radians(6.0) if baseline.joint_type == mujoco.mjtJoint.mjJNT_HINGE else 0.006
                if low - margin <= qpos <= high + margin:
                    self.model.jnt_range[jid] = baseline.joint_range
                    self.model.jnt_stiffness[jid] = baseline.stiffness
                    self.model.dof_damping[baseline.dof_id] = baseline.damping
                else:
                    # 패널이 넓게 접힌 뒤 바로 hard-limit으로 되돌리면 튈 수 있으므로,
                    # 원래 rest shape 쪽으로 회복될 때까지 soft guard만 먼저 복구합니다.
                    self.model.jnt_stiffness[jid] = max(0.012, baseline.stiffness * 0.65)
                    self.model.dof_damping[baseline.dof_id] = max(0.030, baseline.damping * 0.85)
        self.active_compliant_patch_count = active_patch_count
        self.active_compliant_joint_count = active_joint_count

        """

    def solve_ee_position_ik(self, arm: str, target_xyz: np.ndarray, *, iterations: int = 80) -> dict[str, float | bool | int]:
        site_id = self._site_id(EE_SITE_NAMES[arm])
        joint_names = self.arm_joint_names(arm)
        qaddrs = [self._joint_qpos_address(name) for name in joint_names]
        daddrs = [self._joint_dof_address(name) for name in joint_names]
        joint_ids = [self._joint_id(name) for name in joint_names]
        target = np.asarray(target_xyz, dtype=np.float64)

        success = False
        err_norm = float("inf")
        for it in range(iterations):
            mujoco.mj_forward(self.model, self.data)
            err = target - self.data.site_xpos[site_id]
            err_norm = float(np.linalg.norm(err))
            if err_norm < 0.006:
                success = True
                break
            jacp = np.zeros((3, self.model.nv), dtype=np.float64)
            jacr = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
            j = jacp[:, daddrs]
            damping = 0.015
            dq = j.T @ np.linalg.solve(j @ j.T + damping * np.eye(3), 0.45 * err)
            dq = np.clip(dq, -0.045, 0.045)
            for qaddr, jid, delta in zip(qaddrs, joint_ids, dq):
                low, high = self.model.jnt_range[jid]
                self.data.qpos[qaddr] = float(np.clip(self.data.qpos[qaddr] + delta, low, high))

        for joint_name in joint_names:
            self.set_joint_target_rad(joint_name, self.data.qpos[self._joint_qpos_address(joint_name)])
        mujoco.mj_forward(self.model, self.data)
        return {"success": success, "error_m": err_norm, "iterations": it + 1}

    def nearest_grasp_site(self, reference_xyz: np.ndarray) -> tuple[str, np.ndarray]:
        names = [f"site_top_seam_{i:02d}" for i in range(TOP_SEAM_COUNT)]
        names += ["site_top_edge_occlusion_left", "site_top_edge_occlusion_right"]
        names += [f"site_upper_left_{i:02d}" for i in range(TOP_SEAM_COUNT)]
        names += [f"site_upper_right_{i:02d}" for i in range(TOP_SEAM_COUNT)]
        candidates: list[tuple[float, str, np.ndarray]] = []
        for name in names:
            try:
                pos = self.site_pos(name)
            except KeyError:
                continue
            candidates.append((float(np.linalg.norm(pos - reference_xyz)), name, pos))
        _, name, pos = min(candidates, key=lambda item: item[0])
        return name, pos

    def deformation_metrics(self) -> dict[str, float]:
        def disp(name: str) -> float:
            return 1000.0 * float(np.linalg.norm(self.site_pos(name) - self._initial_site_pos[name]))

        left = self.site_pos(LOWER_LEFT_SITE)
        right = self.site_pos(LOWER_RIGHT_SITE)
        left0 = self._initial_site_pos[LOWER_LEFT_SITE]
        right0 = self._initial_site_pos[LOWER_RIGHT_SITE]
        belly_opening = 1000.0 * abs(float(np.linalg.norm(left - right) - np.linalg.norm(left0 - right0)))
        bottom_drop = 1000.0 * max(0.0, float(self._initial_site_pos[BOTTOM_CENTER_SITE][2] - self.site_pos(BOTTOM_CENTER_SITE)[2]))
        for i in range(INNER_BOTTOM_PANEL_COUNT):
            name = f"site_inner_bottom_{i:02d}"
            if name in self._initial_site_pos:
                bottom_drop = max(bottom_drop, 1000.0 * max(0.0, float(self._initial_site_pos[name][2] - self.site_pos(name)[2])))
        bottom_rollup = 0.0
        for name in (
            f"site_bottom_edge_left_{PANEL_CENTER_INDEX:02d}",
            f"site_bottom_edge_right_{PANEL_CENTER_INDEX:02d}",
        ):
            if name in self._initial_site_pos:
                bottom_rollup = max(bottom_rollup, 1000.0 * max(0.0, float(self.site_pos(name)[2] - self._initial_site_pos[name][2])))
        shoulder = 0.5 * (disp(SHOULDER_CENTER_SITE_LEFT) + disp(SHOULDER_CENTER_SITE_RIGHT))
        return {
            "shoulder_deflection_mm": shoulder,
            "top_patch_change_mm": max(disp(TOP_CENTER_SITE), disp(TOP_RAIL_SITE)),
            "lower_belly_opening_mm": belly_opening,
            "bottom_sag_mm": bottom_drop,
            "bottom_edge_rollup_mm": bottom_rollup,
        }


class DualSackUR5Gui:
    def __init__(self, root: tk.Tk, env: DualSackUR5Env, lock: threading.RLock, stop_event: threading.Event) -> None:
        self.root = root
        self.env = env
        self.lock = lock
        self.stop_event = stop_event
        self.updating = False
        self.joint_controls: list[JointControl] = []
        self.ee_controls: dict[str, EEControl] = {}
        self.ee_after_ids: dict[str, str] = {}
        self.show_physics_var = tk.BooleanVar(value=True)
        self.show_inner_var = tk.BooleanVar(value=False)
        self.show_ballast_var = tk.BooleanVar(value=False)
        self.show_visual_skin_var = tk.BooleanVar(value=False)
        self.show_robot_collision_proxy_var = tk.BooleanVar(value=False)
        self.show_contacts_var = tk.BooleanVar(value=True)
        self.show_contact_forces_var = tk.BooleanVar(value=False)
        self.show_site_frames_var = tk.BooleanVar(value=True)
        self.show_body_frames_var = tk.BooleanVar(value=False)
        self.show_site_labels_var = tk.BooleanVar(value=False)
        self.show_transparent_var = tk.BooleanVar(value=False)
        self.contact_patch_monitor_var = tk.BooleanVar(value=True)
        self.pause_var = tk.BooleanVar(value=False)
        self.gripper_var = tk.DoubleVar(value=ROBOTIQ_2F140_MAX_GAP_M * 1000.0)
        self.status_var = tk.StringVar(value="ready")
        self.deform_var = tk.StringVar(value="deformation: -")
        self.bag_var = tk.StringVar(value="bag_frame: -")
        self.root.title(f"Dual UR5e + Robotiq 2F-140 + Scoop | {env.scenario}")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._build()
        self.sync_all_ee_targets()
        self.refresh()

    def _build(self) -> None:
        header = tk.Frame(self.root, padx=8, pady=6)
        header.pack(fill=tk.X)
        tk.Label(header, text="Dual UR5e Sack Benchmark GUI", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        tk.Label(header, text=f"scenario={self.env.scenario} | 각도는 degree, xyz는 meter 단위입니다.").pack(anchor="w")

        options = tk.Frame(header)
        options.pack(fill=tk.X, pady=(4, 0))
        tk.Checkbutton(options, text="physics patches", variable=self.show_physics_var, command=self.on_view_options).pack(side=tk.LEFT)
        tk.Checkbutton(options, text="inner load shell", variable=self.show_inner_var, command=self.on_view_options).pack(side=tk.LEFT, padx=(8, 0))
        tk.Checkbutton(options, text="ballast masses", variable=self.show_ballast_var, command=self.on_view_options).pack(side=tk.LEFT, padx=(8, 0))
        tk.Checkbutton(options, text="visual skin overlay", variable=self.show_visual_skin_var, command=self.on_view_options).pack(side=tk.LEFT, padx=(8, 0))
        tk.Checkbutton(options, text="robot collision proxy", variable=self.show_robot_collision_proxy_var, command=self.on_view_options).pack(side=tk.LEFT, padx=(8, 0))
        tk.Checkbutton(options, text="contacts", variable=self.show_contacts_var, command=self.on_view_options).pack(side=tk.LEFT, padx=(8, 0))
        tk.Checkbutton(options, text="contact force", variable=self.show_contact_forces_var, command=self.on_view_options).pack(side=tk.LEFT, padx=(8, 0))
        tk.Checkbutton(options, text="site frames", variable=self.show_site_frames_var, command=self.on_view_options).pack(side=tk.LEFT, padx=(8, 0))
        tk.Checkbutton(options, text="body frames", variable=self.show_body_frames_var, command=self.on_view_options).pack(side=tk.LEFT, padx=(8, 0))
        tk.Checkbutton(options, text="site labels", variable=self.show_site_labels_var, command=self.on_view_options).pack(side=tk.LEFT, padx=(8, 0))
        tk.Checkbutton(options, text="transparent", variable=self.show_transparent_var, command=self.on_view_options).pack(side=tk.LEFT, padx=(8, 0))
        tk.Checkbutton(options, text="contact patch monitor", variable=self.contact_patch_monitor_var, command=self.on_view_options).pack(side=tk.LEFT, padx=(8, 0))
        tk.Checkbutton(options, text="pause sim", variable=self.pause_var, command=self.on_pause).pack(side=tk.LEFT, padx=(12, 0))
        tk.Button(options, text="Reset", command=self.reset).pack(side=tk.LEFT, padx=(12, 0))
        tk.Button(options, text="Save pose", command=self.save_pose).pack(side=tk.LEFT, padx=(6, 0))
        tk.Button(options, text="Load pose", command=self.load_pose).pack(side=tk.LEFT, padx=(6, 0))

        main = tk.Frame(self.root, padx=8, pady=4)
        main.pack(fill=tk.BOTH, expand=True)
        self._build_arm(main, "left", "UR5e A + Robotiq 2F-140", 0)
        self._build_arm(main, "right", "UR5e B + Scoop", 1)

        bottom = tk.LabelFrame(self.root, text="Sack deformation monitor", padx=8, pady=6)
        bottom.pack(fill=tk.X, padx=8, pady=(0, 8))
        tk.Label(bottom, textvariable=self.bag_var, anchor="w").pack(fill=tk.X)
        tk.Label(bottom, textvariable=self.deform_var, anchor="w").pack(fill=tk.X)
        tk.Label(bottom, textvariable=self.status_var, fg="#245a9a", anchor="w").pack(fill=tk.X)

    def _build_arm(self, parent: tk.Frame, arm: str, title: str, column: int) -> None:
        frame = tk.LabelFrame(parent, text=title, padx=6, pady=6)
        frame.grid(row=0, column=column, sticky="nsew", padx=5)
        parent.grid_columnconfigure(column, weight=1)
        for index, (joint_name, actuator_name) in enumerate(zip(self.env.arm_joint_names(arm), self.env.arm_actuator_names(arm))):
            row = tk.Frame(frame)
            row.pack(fill=tk.X, pady=1)
            short = joint_name.replace(ARM_PREFIXES[arm] + "_", "")
            tk.Label(row, text=f"J{index + 1} {short}", width=22, anchor="w").pack(side=tk.LEFT)
            initial = np.rad2deg(self.env.actuator_ctrl(actuator_name))
            var = tk.DoubleVar(value=float(initial))
            slider = tk.Scale(
                row,
                from_=-180,
                to=180,
                resolution=0.2,
                orient=tk.HORIZONTAL,
                length=230,
                variable=var,
                command=lambda _v, n=actuator_name, v=var: self.on_joint(n, v),
            )
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
            tk.Button(row, text="-", width=3, command=lambda n=actuator_name: self.nudge_joint(n, -2.0)).pack(side=tk.LEFT)
            tk.Button(row, text="+", width=3, command=lambda n=actuator_name: self.nudge_joint(n, 2.0)).pack(side=tk.LEFT)
            cur = tk.StringVar(value="cur: -")
            tk.Label(row, textvariable=cur, width=12, anchor="e").pack(side=tk.LEFT)
            self.joint_controls.append(JointControl(arm, joint_name, actuator_name, var, cur))

        self._build_ee(frame, arm)
        if arm == "left":
            grip = tk.LabelFrame(frame, text="Robotiq 2F-140 opening", padx=6, pady=4)
            grip.pack(fill=tk.X, pady=(8, 0))
            tk.Scale(
                grip,
                from_=ROBOTIQ_2F140_MIN_GAP_M * 1000.0,
                to=ROBOTIQ_2F140_MAX_GAP_M * 1000.0,
                resolution=0.5,
                orient=tk.HORIZONTAL,
                variable=self.gripper_var,
                label="jaw gap [mm]",
                command=self.on_gripper,
            ).pack(fill=tk.X)
            buttons = tk.Frame(grip)
            buttons.pack(fill=tk.X)
            tk.Button(buttons, text="Open 140mm", command=lambda: self.set_gripper_mm(140.0)).pack(side=tk.LEFT)
            tk.Button(buttons, text="Close", command=lambda: self.set_gripper_mm(4.0)).pack(side=tk.LEFT, padx=(6, 0))
            tk.Button(buttons, text="Move to nearest patch", command=self.move_to_nearest_patch).pack(side=tk.LEFT, padx=(6, 0))

    def _build_ee(self, parent: tk.LabelFrame, arm: str) -> None:
        frame = tk.LabelFrame(parent, text=f"End effector xyz / {EE_SITE_NAMES[arm]}", padx=6, pady=4)
        frame.pack(fill=tk.X, pady=(8, 0))
        current = tk.StringVar(value="current xyz: -")
        tk.Label(frame, textvariable=current, anchor="w").pack(fill=tk.X)
        target_vars = {axis: tk.DoubleVar(value=0.0) for axis in ("x", "y", "z")}
        self.ee_controls[arm] = EEControl(arm, EE_SITE_NAMES[arm], target_vars, current)
        for axis in ("x", "y", "z"):
            row = tk.Frame(frame)
            row.pack(fill=tk.X)
            tk.Label(row, text=axis, width=2).pack(side=tk.LEFT)
            low, high = EE_LIMITS[axis]
            tk.Scale(
                row,
                from_=low,
                to=high,
                resolution=0.005,
                orient=tk.HORIZONTAL,
                variable=target_vars[axis],
                length=190,
                command=lambda _v, a=arm: self.schedule_ik(a),
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)
            tk.Button(row, text="-", width=3, command=lambda a=arm, ax=axis: self.nudge_ee(a, ax, -0.01)).pack(side=tk.LEFT)
            tk.Button(row, text="+", width=3, command=lambda a=arm, ax=axis: self.nudge_ee(a, ax, 0.01)).pack(side=tk.LEFT)
        buttons = tk.Frame(frame)
        buttons.pack(fill=tk.X, pady=(4, 0))
        tk.Button(buttons, text="Use current", command=lambda a=arm: self.sync_ee_target(a)).pack(side=tk.LEFT)
        tk.Button(buttons, text="Apply xyz IK", command=lambda a=arm: self.apply_ik(a)).pack(side=tk.LEFT, padx=(6, 0))

    def on_view_options(self) -> None:
        with self.lock:
            self.env.show_physics = bool(self.show_physics_var.get())
            self.env.show_inner = bool(self.show_inner_var.get())
            self.env.show_ballast = bool(self.show_ballast_var.get())
            self.env.show_visual_skin = bool(self.show_visual_skin_var.get())
            self.env.show_robot_collision_proxy = bool(self.show_robot_collision_proxy_var.get())
            self.env.show_contacts = bool(self.show_contacts_var.get())
            self.env.show_contact_forces = bool(self.show_contact_forces_var.get())
            self.env.show_site_frames = bool(self.show_site_frames_var.get())
            self.env.show_body_frames = bool(self.show_body_frames_var.get())
            self.env.show_site_labels = bool(self.show_site_labels_var.get())
            self.env.show_transparent = bool(self.show_transparent_var.get())
            self.env.contact_patch_monitor_enabled = bool(self.contact_patch_monitor_var.get())
            if not self.env.contact_patch_monitor_enabled:
                self.env.restore_panel_compliance()

    def on_pause(self) -> None:
        with self.lock:
            self.env.paused = bool(self.pause_var.get())

    def on_joint(self, actuator_name: str, var: tk.DoubleVar) -> None:
        if self.updating:
            return
        with self.lock:
            self.env.set_actuator_ctrl(actuator_name, float(np.deg2rad(var.get())))
        self.status_var.set(f"{actuator_name} = {var.get():.1f} deg")

    def nudge_joint(self, actuator_name: str, delta_deg: float) -> None:
        with self.lock:
            new_rad = self.env.add_joint_delta_rad(actuator_name, float(np.deg2rad(delta_deg)))
        self.updating = True
        try:
            for control in self.joint_controls:
                if control.actuator_name == actuator_name:
                    control.target_var.set(float(np.rad2deg(new_rad)))
                    break
        finally:
            self.updating = False

    def on_gripper(self, value: str) -> None:
        if self.updating:
            return
        with self.lock:
            self.env.set_gripper_gap(float(value) / 1000.0)

    def set_gripper_mm(self, gap_mm: float) -> None:
        with self.lock:
            self.env.set_gripper_gap(gap_mm / 1000.0, immediate=True)
            gap = self.env.gripper_gap() * 1000.0
        self.updating = True
        try:
            self.gripper_var.set(gap)
        finally:
            self.updating = False

    def schedule_ik(self, arm: str) -> None:
        if self.updating:
            return
        old = self.ee_after_ids.get(arm)
        if old is not None:
            self.root.after_cancel(old)
        self.ee_after_ids[arm] = self.root.after(140, lambda a=arm: self.apply_ik(a))

    def apply_ik(self, arm: str) -> None:
        old = self.ee_after_ids.pop(arm, None)
        if old is not None:
            try:
                self.root.after_cancel(old)
            except tk.TclError:
                pass
        target = np.array([self.ee_controls[arm].target_vars[a].get() for a in ("x", "y", "z")], dtype=np.float64)
        with self.lock:
            result = self.env.solve_ee_position_ik(arm, target)
            targets = {c.actuator_name: self.env.actuator_ctrl(c.actuator_name) for c in self.joint_controls if c.arm == arm}
        self.updating = True
        try:
            for control in self.joint_controls:
                if control.actuator_name in targets:
                    control.target_var.set(float(np.rad2deg(targets[control.actuator_name])))
        finally:
            self.updating = False
        self.status_var.set(f"{arm} IK success={result['success']} error={float(result['error_m']):.4f} m")

    def nudge_ee(self, arm: str, axis: str, delta: float) -> None:
        var = self.ee_controls[arm].target_vars[axis]
        var.set(round(float(var.get() + delta), 4))
        self.apply_ik(arm)

    def sync_all_ee_targets(self) -> None:
        for arm in self.ee_controls:
            self.sync_ee_target(arm)

    def sync_ee_target(self, arm: str) -> None:
        with self.lock:
            xyz = self.env.end_effector_pos(arm)
        self.updating = True
        try:
            for axis, value in zip(("x", "y", "z"), xyz):
                self.ee_controls[arm].target_vars[axis].set(round(float(value), 4))
        finally:
            self.updating = False

    def move_to_nearest_patch(self) -> None:
        with self.lock:
            current = self.env.end_effector_pos("left")
            name, target = self.env.nearest_grasp_site(current)
        target = target + np.array([0.0, 0.0, 0.035])
        self.updating = True
        try:
            for axis, value in zip(("x", "y", "z"), target):
                self.ee_controls["left"].target_vars[axis].set(round(float(value), 4))
        finally:
            self.updating = False
        self.apply_ik("left")
        self.status_var.set(f"left UR5e moved near {name}")

    def reset(self) -> None:
        with self.lock:
            self.env.reset()
        self.sync_all_ee_targets()
        self.status_var.set("reset")

    def save_pose(self) -> None:
        POSE_DIR.mkdir(parents=True, exist_ok=True)
        path = filedialog.asksaveasfilename(
            title="Save pose",
            initialdir=str(POSE_DIR),
            initialfile=f"dual_sack_pose_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            defaultextension=".json",
            filetypes=[("JSON pose", "*.json")],
        )
        if not path:
            return
        with self.lock:
            payload = {
                "schema": POSE_SCHEMA,
                "scenario": self.env.scenario,
                "joint_targets_deg": {c.actuator_name: float(np.rad2deg(self.env.actuator_ctrl(c.actuator_name))) for c in self.joint_controls},
                "gripper_gap_mm": float(self.env.gripper_gap() * 1000.0),
            }
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def load_pose(self) -> None:
        path = filedialog.askopenfilename(title="Load pose", initialdir=str(POSE_DIR), filetypes=[("JSON pose", "*.json")])
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            messagebox.showerror("Load pose failed", str(exc))
            return
        if payload.get("schema") != POSE_SCHEMA:
            messagebox.showerror("Load pose failed", f"unknown schema: {payload.get('schema')}")
            return
        with self.lock:
            for name, deg in payload.get("joint_targets_deg", {}).items():
                try:
                    self.env.set_actuator_ctrl(name, float(np.deg2rad(deg)))
                except KeyError:
                    continue
            if "gripper_gap_mm" in payload:
                self.env.set_gripper_gap(float(payload["gripper_gap_mm"]) / 1000.0, immediate=True)
        self.sync_all_ee_targets()

    def refresh(self) -> None:
        with self.lock:
            self.updating = True
            try:
                for control in self.joint_controls:
                    target_deg = float(np.rad2deg(self.env.actuator_ctrl(control.actuator_name)))
                    control.target_var.set(target_deg)
                    qaddr = self.env._joint_qpos_address(control.joint_name)
                    control.current_var.set(f"cur {np.rad2deg(self.env.data.qpos[qaddr]):6.1f}")
                self.gripper_var.set(self.env.gripper_gap() * 1000.0)
                for arm, control in self.ee_controls.items():
                    xyz = self.env.end_effector_pos(arm)
                    control.current_var.set(f"current xyz: x={xyz[0]: .3f}, y={xyz[1]: .3f}, z={xyz[2]: .3f}")
                bag_id = self.env._body_id("bag_frame")
                bag = self.env.data.xpos[bag_id]
                self.bag_var.set(f"bag_frame xyz [m]: x={bag[0]: .3f}, y={bag[1]: .3f}, z={bag[2]: .3f}")
                metrics = self.env.deformation_metrics()
                self.deform_var.set(
                    "deformation [mm]: "
                    f"shoulder={metrics['shoulder_deflection_mm']:.2f}, "
                    f"top_patch={metrics['top_patch_change_mm']:.2f}, "
                    f"lower_belly_open={metrics['lower_belly_opening_mm']:.2f}, "
                    f"bottom_sag={metrics['bottom_sag_mm']:.2f}, "
                    f"bottom_rollup={metrics['bottom_edge_rollup_mm']:.2f} | "
                    f"near_contact_patches={self.env.active_compliant_patch_count}, "
                    f"near_contact_joints={self.env.active_compliant_joint_count}"
                )
            finally:
                self.updating = False
        if not self.stop_event.is_set():
            self.root.after(120, self.refresh)

    def on_close(self) -> None:
        self.stop_event.set()
        self.root.after(50, self.root.destroy)


def viewer_loop(env: DualSackUR5Env, lock: threading.RLock, stop_event: threading.Event, *, speed: float) -> None:
    sleep_dt = env.model.opt.timestep / max(speed, 1e-6)
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.38])
        viewer.cam.distance = 2.65
        viewer.cam.azimuth = 132.0
        viewer.cam.elevation = -24.0
        viewer.opt.geomgroup[:] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
        while viewer.is_running() and not stop_event.is_set():
            start = time.perf_counter()
            with lock:
                viewer.opt.geomgroup[1] = bool(env.show_physics)
                viewer.opt.geomgroup[2] = bool(env.show_inner)
                viewer.opt.geomgroup[3] = bool(env.show_visual_skin)
                viewer.opt.geomgroup[4] = bool(env.show_ballast)
                viewer.opt.geomgroup[5] = bool(env.show_robot_collision_proxy)
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = bool(env.show_contacts)
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = bool(env.show_contact_forces)
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = bool(env.show_transparent)
                if env.show_site_frames:
                    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
                elif env.show_body_frames:
                    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
                else:
                    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
                viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE if env.show_site_labels else mujoco.mjtLabel.mjLABEL_NONE
                env.step()
                viewer.sync()
            remain = sleep_dt - (time.perf_counter() - start)
            if remain > 0:
                time.sleep(remain)
    stop_event.set()


def smoke_test(scenario: str) -> int:
    env = DualSackUR5Env(scenario=scenario)
    left_before = env.end_effector_pos("left")
    _, target = env.nearest_grasp_site(left_before)
    result = env.solve_ee_position_ik("left", target + np.array([0.0, 0.0, 0.04]))
    env.set_gripper_gap(0.020, immediate=True)
    for _ in range(200):
        env.step()
    metrics = env.deformation_metrics()
    print(f"scene_xml={env.scene_path}")
    print(f"scenario={scenario}")
    print(f"left_ee_before={left_before.tolist()}")
    print(f"left_ik={result}")
    print(f"gripper_gap_mm={env.gripper_gap() * 1000.0:.2f}")
    print(f"deformation_metrics={metrics}")
    print("gui_smoke_pass=True")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="GUI control for dual UR5e + Robotiq 2F-140 + Scoop sack benchmark")
    parser.add_argument("--scenario", choices=SCENARIO_NAMES, default="underfilled")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        return smoke_test(args.scenario)

    env = DualSackUR5Env(scenario=args.scenario)
    lock = threading.RLock()
    stop_event = threading.Event()
    thread = threading.Thread(target=viewer_loop, args=(env, lock, stop_event), kwargs={"speed": args.speed}, daemon=True)
    thread.start()

    root = tk.Tk()
    DualSackUR5Gui(root, env, lock, stop_event)
    try:
        root.mainloop()
    finally:
        stop_event.set()
        thread.join(timeout=2.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
