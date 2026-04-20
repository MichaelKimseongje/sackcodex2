from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Optional
import tkinter as tk

import mujoco
import mujoco.viewer
import numpy as np

from dual_ur5_low_fill_env import (
    DualUR5LowFillEnv,
    KeyboardJointStepper,
    JOINT_STEP_DEG_DEFAULT,
    collect_shell_body_ids,
)


POSE_SCHEMA = "dual_ur5_low_fill_pose_v1"
POSE_DIR = Path(__file__).resolve().parent / "poses"
EE_STEP_M_DEFAULT = 0.01
EE_SLIDER_LIMITS = {
    "x": (-0.60, 1.20),
    "y": (-1.20, 1.40),
    "z": (0.00, 1.20),
}


@dataclass
class JointControl:
    arm: str
    index: int
    joint_name: str
    actuator_name: str
    target_var: tk.DoubleVar
    current_var: tk.StringVar
    slider: tk.Scale


@dataclass
class EndEffectorControl:
    arm: str
    site_name: str
    target_vars: dict[str, tk.DoubleVar]
    sliders: dict[str, tk.Scale]
    current_var: tk.StringVar


class DualUR5Gui:
    """MuJoCo viewer와 분리된 로봇 목표각 조작 GUI."""

    def __init__(
        self,
        root: tk.Tk,
        env: DualUR5LowFillEnv,
        lock: threading.RLock,
        stop_event: threading.Event,
        *,
        step_deg: float,
    ) -> None:
        self.root = root
        self.env = env
        self.lock = lock
        self.stop_event = stop_event
        self.step_deg = float(step_deg)
        self.updating = False
        self.joint_controls: list[JointControl] = []
        self.ee_controls: dict[str, EndEffectorControl] = {}
        self.ee_after_ids: dict[str, str] = {}
        self.ee_auto_sync_until: dict[str, float] = {}
        self.shell_body_ids = collect_shell_body_ids(self.env.model)
        self.bag_frame_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "bag_frame")

        self.bag_frame_var = tk.StringVar(value="bag_frame: -")
        self.shell_center_var = tk.StringVar(value="shell_center: -")
        self.gripper_var = tk.DoubleVar(value=self.env.left_gripper_gap() * 1000.0)
        self.ee_step_var = tk.DoubleVar(value=EE_STEP_M_DEFAULT)
        self.status_var = tk.StringVar(value="ready")

        self.root.title(getattr(self.env, "gui_title", "Dual UR5 Low-Fill Sack Control"))
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._build()
        self.sync_all_ee_targets_from_current()
        self.refresh_from_sim()

    def _build(self) -> None:
        header = tk.Frame(self.root, padx=10, pady=8)
        header.pack(fill=tk.X)
        tk.Label(
            header,
            text=getattr(self.env, "gui_header", "Dual UR5 + low-fill sack controller"),
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w")
        tk.Label(
            header,
            text="관절 목표각은 degree 단위입니다. MuJoCo viewer 창은 카메라 확인용, 이 창은 조작용입니다.",
        ).pack(anchor="w")

        main = tk.Frame(self.root, padx=10, pady=4)
        main.pack(fill=tk.BOTH, expand=True)

        self._build_arm_frame(main, "left", "Left UR5 / 2F gripper", self.env.left_joint_names, self.env.left_actuator_names, 0)
        self._build_arm_frame(main, "right", "Right UR5 / scoop", self.env.right_joint_names, self.env.right_actuator_names, 1)
        self._build_bottom(main)

    def _build_arm_frame(
        self,
        parent: tk.Frame,
        arm: str,
        title: str,
        joint_names: list[str],
        actuator_names: list[str],
        column: int,
    ) -> None:
        frame = tk.LabelFrame(parent, text=title, padx=8, pady=8)
        frame.grid(row=0, column=column, sticky="nsew", padx=5, pady=4)
        parent.grid_columnconfigure(column, weight=1)

        for index, (joint_name, actuator_name) in enumerate(zip(joint_names, actuator_names)):
            actuator_id = self.env._actuator_id(actuator_name)
            low_rad, high_rad = self.env.model.actuator_ctrlrange[actuator_id]
            low_deg, high_deg = np.rad2deg([low_rad, high_rad])
            initial_deg = np.rad2deg(self.env.actuator_ctrl(actuator_name))

            row = tk.Frame(frame)
            row.pack(fill=tk.X, pady=2)

            label = joint_name.replace(f"{arm}_", "").replace("_joint", "")
            tk.Label(row, text=f"J{index + 1} {label}", width=15, anchor="w").pack(side=tk.LEFT)

            target_var = tk.DoubleVar(value=float(initial_deg))
            slider = tk.Scale(
                row,
                from_=float(low_deg),
                to=float(high_deg),
                resolution=0.1,
                orient=tk.HORIZONTAL,
                variable=target_var,
                length=230,
                command=lambda _value, name=actuator_name, var=target_var: self.on_joint_slider(name, var),
            )
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

            tk.Button(row, text="-", width=3, command=lambda name=actuator_name: self.nudge_joint(name, -self.step_deg)).pack(side=tk.LEFT, padx=(4, 1))
            tk.Button(row, text="+", width=3, command=lambda name=actuator_name: self.nudge_joint(name, +self.step_deg)).pack(side=tk.LEFT, padx=(1, 4))

            current_var = tk.StringVar(value="cur: -")
            tk.Label(row, textvariable=current_var, width=12, anchor="e").pack(side=tk.LEFT)

            self.joint_controls.append(
                JointControl(
                    arm=arm,
                    index=index,
                    joint_name=joint_name,
                    actuator_name=actuator_name,
                    target_var=target_var,
                    current_var=current_var,
                    slider=slider,
                )
            )

        self._build_ee_frame(frame, arm)

        if arm == "left":
            grip = tk.LabelFrame(frame, text="2F gripper", padx=6, pady=5)
            grip.pack(fill=tk.X, pady=(8, 0))
            self.gripper_slider = tk.Scale(
                grip,
                from_=0.0,
                to=self.env.left_gripper_gap() * 1000.0,
                resolution=0.2,
                orient=tk.HORIZONTAL,
                label="pad gap [mm]",
                variable=self.gripper_var,
                command=self.on_gripper_slider,
            )
            self.gripper_slider.pack(fill=tk.X)

    def _build_ee_frame(self, parent: tk.LabelFrame, arm: str) -> None:
        site_name = self.env.end_effector_site_name(arm)
        frame = tk.LabelFrame(parent, text=f"End effector xyz / {site_name}", padx=6, pady=5)
        frame.pack(fill=tk.X, pady=(8, 0))

        current_var = tk.StringVar(value="current xyz: -")
        tk.Label(frame, textvariable=current_var, anchor="w").pack(fill=tk.X)

        target_vars = {axis: tk.DoubleVar(value=0.0) for axis in ("x", "y", "z")}
        sliders: dict[str, tk.Scale] = {}
        self.ee_controls[arm] = EndEffectorControl(
            arm=arm,
            site_name=site_name,
            target_vars=target_vars,
            sliders=sliders,
            current_var=current_var,
        )

        for axis in ("x", "y", "z"):
            row = tk.Frame(frame)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=axis, width=2, anchor="w").pack(side=tk.LEFT)
            low, high = EE_SLIDER_LIMITS[axis]
            slider = tk.Scale(
                row,
                from_=low,
                to=high,
                resolution=0.005,
                orient=tk.HORIZONTAL,
                variable=target_vars[axis],
                length=170,
                command=lambda _value, a=arm: self.schedule_ee_ik(a),
            )
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
            sliders[axis] = slider
            tk.Entry(row, textvariable=target_vars[axis], width=9).pack(side=tk.LEFT)
            tk.Button(row, text="-", width=3, command=lambda a=arm, ax=axis: self.nudge_ee(a, ax, -self.ee_step_var.get())).pack(side=tk.LEFT, padx=(4, 1))
            tk.Button(row, text="+", width=3, command=lambda a=arm, ax=axis: self.nudge_ee(a, ax, +self.ee_step_var.get())).pack(side=tk.LEFT, padx=(1, 4))

        buttons = tk.Frame(frame)
        buttons.pack(fill=tk.X, pady=(4, 0))
        tk.Button(buttons, text="Use current", command=lambda a=arm: self.sync_ee_target_from_current(a)).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(buttons, text="Apply xyz IK", command=lambda a=arm: self.apply_ee_ik(a)).pack(side=tk.LEFT)

        step_row = tk.Frame(frame)
        step_row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(step_row, text="xyz step [m]", width=10, anchor="w").pack(side=tk.LEFT)
        tk.Entry(step_row, textvariable=self.ee_step_var, width=8).pack(side=tk.LEFT)

    def _build_bottom(self, parent: tk.Frame) -> None:
        bottom = tk.Frame(parent)
        bottom.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=8)

        pose = tk.LabelFrame(bottom, text="Pose", padx=8, pady=6)
        pose.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        tk.Button(pose, text="Save pose JSON", command=self.save_pose).pack(fill=tk.X, pady=2)
        tk.Button(pose, text="Load pose JSON", command=self.load_pose).pack(fill=tk.X, pady=2)
        tk.Button(pose, text="Home reset", command=self.home_reset).pack(fill=tk.X, pady=2)
        tk.Button(pose, text="Move 2F to nearest grasp", command=self.move_left_to_nearest_grasp).pack(fill=tk.X, pady=(8, 2))
        tk.Button(pose, text="Close gripper", command=self.close_left_gripper).pack(fill=tk.X, pady=2)

        bag = tk.LabelFrame(bottom, text="Sack position", padx=8, pady=6)
        bag.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(bag, textvariable=self.bag_frame_var, anchor="w").pack(fill=tk.X)
        tk.Label(bag, textvariable=self.shell_center_var, anchor="w").pack(fill=tk.X)
        tk.Label(bag, textvariable=self.status_var, anchor="w", fg="#245a9a").pack(fill=tk.X, pady=(6, 0))

    def on_joint_slider(self, actuator_name: str, var: tk.DoubleVar) -> None:
        if self.updating:
            return
        arm = self.arm_for_actuator(actuator_name)
        with self.lock:
            actuator_id = self.env._actuator_id(actuator_name)
            low, high = self.env.model.actuator_ctrlrange[actuator_id]
            target_rad = float(np.clip(np.deg2rad(var.get()), low, high))
            self.env.data.ctrl[actuator_id] = target_rad
        if arm is not None:
            self.enable_ee_auto_sync(arm)
        self.status_var.set(f"{actuator_name} target = {var.get():.1f} deg")

    def on_gripper_slider(self, value: str) -> None:
        if self.updating:
            return
        with self.lock:
            self.env.set_left_gripper_gap(float(value) / 1000.0, immediate=True)
        self.status_var.set(f"left gripper pad gap = {float(value):.1f} mm")

    def nudge_joint(self, actuator_name: str, delta_deg: float) -> None:
        arm = self.arm_for_actuator(actuator_name)
        with self.lock:
            value_rad = self.env.add_actuator_delta(actuator_name, np.deg2rad(delta_deg))
            value_deg = float(np.rad2deg(value_rad))
        self._set_joint_var(actuator_name, value_deg)
        if arm is not None:
            self.enable_ee_auto_sync(arm)
        self.status_var.set(f"{actuator_name} target = {value_deg:.1f} deg")

    def arm_for_actuator(self, actuator_name: str) -> Optional[str]:
        if actuator_name in self.env.left_actuator_names:
            return "left"
        if actuator_name in self.env.right_actuator_names:
            return "right"
        return None

    def enable_ee_auto_sync(self, arm: str, duration_s: float = 2.0) -> None:
        # joint 조작 후에는 EE target slider가 실제 EE 위치를 잠깐 따라오게 해서 x/y/z 조작 시 튐을 막는다.
        self.ee_auto_sync_until[arm] = time.perf_counter() + duration_s

    def disable_ee_auto_sync(self, arm: str) -> None:
        self.ee_auto_sync_until[arm] = 0.0

    def sync_all_ee_targets_from_current(self) -> None:
        for arm in self.ee_controls:
            self.sync_ee_target_from_current(arm)

    def sync_ee_target_from_current(self, arm: str) -> None:
        control = self.ee_controls[arm]
        with self.lock:
            xyz = self.env.end_effector_pos(arm)
        self.updating = True
        try:
            for axis, value in zip(("x", "y", "z"), xyz):
                control.target_vars[axis].set(round(float(value), 4))
        finally:
            self.updating = False
        self.status_var.set(f"{arm} EE target synced from current")

    def schedule_ee_ik(self, arm: str) -> None:
        if self.updating:
            return
        self.disable_ee_auto_sync(arm)
        previous_id = self.ee_after_ids.get(arm)
        if previous_id is not None:
            self.root.after_cancel(previous_id)
        self.ee_after_ids[arm] = self.root.after(120, lambda a=arm: self.apply_ee_ik(a))

    def apply_ee_ik(self, arm: str) -> None:
        self.disable_ee_auto_sync(arm)
        previous_id = self.ee_after_ids.pop(arm, None)
        if previous_id is not None:
            try:
                self.root.after_cancel(previous_id)
            except tk.TclError:
                pass

        control = self.ee_controls[arm]
        try:
            target_xyz = np.array([control.target_vars[axis].get() for axis in ("x", "y", "z")], dtype=np.float64)
        except tk.TclError as exc:
            messagebox.showerror("IK target error", f"Invalid xyz target: {exc}")
            return

        with self.lock:
            result = self.env.solve_ee_position_ik(arm, target_xyz)
            target_degs = {
                actuator_name: float(np.rad2deg(self.env.actuator_ctrl(actuator_name)))
                for actuator_name in self.env.arm_actuator_names(arm)
            }

        for actuator_name, value_deg in target_degs.items():
            self._set_joint_var(actuator_name, value_deg)

        status = "OK" if result["success"] else "approx"
        self.status_var.set(f"{arm} EE IK {status}: error={result['error_m']:.4f} m, iter={result['iterations']}")

    def nudge_ee(self, arm: str, axis: str, delta_m: float) -> None:
        self.disable_ee_auto_sync(arm)
        control = self.ee_controls[arm]
        self.updating = True
        try:
            control.target_vars[axis].set(round(float(control.target_vars[axis].get() + delta_m), 4))
        finally:
            self.updating = False
        self.apply_ee_ik(arm)

    def move_left_to_nearest_grasp(self) -> None:
        with self.lock:
            reference_xyz = self.env.end_effector_pos("left")
            if hasattr(self.env, "nearest_grasp_target"):
                site_name, target_xyz = self.env.nearest_grasp_target(reference_xyz)
            else:
                site_name = self.env.nearest_grasp_site_name(reference_xyz)
                target_xyz = self.env.site_pos(site_name)
            target_xyz = target_xyz + np.array([0.0, 0.0, 0.010], dtype=np.float64)
        control = self.ee_controls["left"]
        self.updating = True
        try:
            for axis, value in zip(("x", "y", "z"), target_xyz):
                control.target_vars[axis].set(round(float(value), 4))
        finally:
            self.updating = False
        self.apply_ee_ik("left")
        self.status_var.set(f"left 2F moved toward {site_name}")

    def close_left_gripper(self) -> None:
        with self.lock:
            self.env.set_left_gripper_gap(self.env.left_gripper_grasp_gap, immediate=True)
            gap_mm = self.env.left_gripper_gap() * 1000.0
        self.updating = True
        try:
            self.gripper_var.set(gap_mm)
        finally:
            self.updating = False
        self.status_var.set(f"left 2F closed: gap={gap_mm:.1f} mm")

    def home_reset(self) -> None:
        with self.lock:
            self.env.reset()
        self.status_var.set("home pose reset")
        for arm in self.ee_controls:
            self.enable_ee_auto_sync(arm, duration_s=0.5)
        self.sync_all_ee_targets_from_current()
        self.refresh_from_sim()

    def save_pose(self) -> None:
        POSE_DIR.mkdir(parents=True, exist_ok=True)
        default_name = f"dual_ur5_pose_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = filedialog.asksaveasfilename(
            title="Save pose",
            initialdir=str(POSE_DIR),
            initialfile=default_name,
            defaultextension=".json",
            filetypes=[("JSON pose", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return

        with self.lock:
            payload = {
                "schema": POSE_SCHEMA,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "with_ballast": self.env.with_ballast,
                "joint_targets_deg": {
                    control.actuator_name: float(np.rad2deg(self.env.actuator_ctrl(control.actuator_name)))
                    for control in self.joint_controls
                },
                "ee_targets_xyz_m": {
                    arm: {axis: float(control.target_vars[axis].get()) for axis in ("x", "y", "z")}
                    for arm, control in self.ee_controls.items()
                },
                "left_gripper_gap_mm": float(self.env.left_gripper_gap() * 1000.0),
            }

        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self.status_var.set(f"saved pose: {Path(path).name}")

    def load_pose(self) -> None:
        path = filedialog.askopenfilename(
            title="Load pose",
            initialdir=str(POSE_DIR),
            filetypes=[("JSON pose", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            messagebox.showerror("Load pose failed", str(exc))
            return

        if payload.get("schema") != POSE_SCHEMA:
            messagebox.showerror("Load pose failed", f"unknown pose schema: {payload.get('schema')}")
            return

        targets = payload.get("joint_targets_deg", {})
        with self.lock:
            for control in self.joint_controls:
                if control.actuator_name not in targets:
                    continue
                actuator_id = self.env._actuator_id(control.actuator_name)
                low, high = self.env.model.actuator_ctrlrange[actuator_id]
                self.env.data.ctrl[actuator_id] = float(np.clip(np.deg2rad(targets[control.actuator_name]), low, high))
                arm = self.arm_for_actuator(control.actuator_name)
                if arm is not None:
                    self.enable_ee_auto_sync(arm)
            if "left_gripper_gap_mm" in payload:
                self.env.set_left_gripper_gap(float(payload["left_gripper_gap_mm"]) / 1000.0, immediate=True)
            elif "left_gripper_opening_mm" in payload:
                self.env.set_left_gripper(float(payload["left_gripper_opening_mm"]) / 1000.0, immediate=True)

            self.updating = True
            try:
                for arm, target in payload.get("ee_targets_xyz_m", {}).items():
                    if arm not in self.ee_controls:
                        continue
                    self.disable_ee_auto_sync(arm)
                    for axis in ("x", "y", "z"):
                        if axis in target:
                            self.ee_controls[arm].target_vars[axis].set(float(target[axis]))
            finally:
                self.updating = False

        self.status_var.set(f"loaded pose: {Path(path).name}")
        self.refresh_from_sim()

    def _set_joint_var(self, actuator_name: str, value_deg: float) -> None:
        self.updating = True
        try:
            for control in self.joint_controls:
                if control.actuator_name == actuator_name:
                    control.target_var.set(float(value_deg))
                    break
        finally:
            self.updating = False

    def refresh_from_sim(self) -> None:
        with self.lock:
            self.updating = True
            try:
                for control in self.joint_controls:
                    target_deg = float(np.rad2deg(self.env.actuator_ctrl(control.actuator_name)))
                    control.target_var.set(target_deg)
                    qpos_addr = self.env._joint_qpos_address(control.joint_name)
                    current_deg = float(np.rad2deg(self.env.data.qpos[qpos_addr]))
                    control.current_var.set(f"cur: {current_deg:7.2f}°")

                gripper_mm = float(self.env.left_gripper_gap() * 1000.0)
                self.gripper_var.set(gripper_mm)

                for arm, control in self.ee_controls.items():
                    ee_xyz = self.env.end_effector_pos(arm)
                    control.current_var.set(
                        f"current xyz [m]: x={ee_xyz[0]: .3f}, y={ee_xyz[1]: .3f}, z={ee_xyz[2]: .3f}"
                    )
                    if self.ee_auto_sync_until.get(arm, 0.0) > time.perf_counter():
                        for axis, value in zip(("x", "y", "z"), ee_xyz):
                            control.target_vars[axis].set(round(float(value), 4))

                if self.bag_frame_id >= 0:
                    frame_xyz = self.env.data.xpos[self.bag_frame_id].copy()
                    self.bag_frame_var.set(
                        f"bag_frame xyz [m]: x={frame_xyz[0]: .3f}, y={frame_xyz[1]: .3f}, z={frame_xyz[2]: .3f}"
                    )

                if self.shell_body_ids:
                    shell_xyz = np.asarray([self.env.data.xpos[body_id] for body_id in self.shell_body_ids], dtype=np.float64)
                    center = np.mean(shell_xyz, axis=0)
                    self.shell_center_var.set(
                        f"shell center xyz [m]: x={center[0]: .3f}, y={center[1]: .3f}, z={center[2]: .3f}"
                    )
            finally:
                self.updating = False

        if not self.stop_event.is_set():
            self.root.after(120, self.refresh_from_sim)

    def on_close(self) -> None:
        self.stop_event.set()
        self.root.after(50, self.root.destroy)


def viewer_loop(
    env: DualUR5LowFillEnv,
    lock: threading.RLock,
    stop_event: threading.Event,
    *,
    speed: float,
    keyboard_control: bool,
    joint_step_deg: float,
) -> None:
    key_controller = KeyboardJointStepper(env, joint_step_deg=joint_step_deg) if keyboard_control else None
    sleep_dt = env.model.opt.timestep / max(speed, 1e-6)

    with mujoco.viewer.launch_passive(
        env.model,
        env.data,
        key_callback=key_controller.handle_key if key_controller is not None else None,
    ) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = np.array([0.58, 0.0, 0.35])
        viewer.cam.distance = 1.55
        viewer.cam.azimuth = 130.0
        viewer.cam.elevation = -22.0
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXFACE] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXEDGE] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXVERT] = True
        viewer.opt.geomgroup[:] = True

        while viewer.is_running() and not stop_event.is_set():
            start = time.perf_counter()
            with lock:
                env.step()
                viewer.sync()
            remaining = sleep_dt - (time.perf_counter() - start)
            if remaining > 0:
                time.sleep(remaining)

    stop_event.set()


def smoke_test(with_ballast: bool) -> int:
    env = DualUR5LowFillEnv(with_ballast=with_ballast)
    for actuator_name in env.left_actuator_names + env.right_actuator_names:
        value_deg = np.rad2deg(env.actuator_ctrl(actuator_name))
        if not np.isfinite(value_deg):
            print(f"nonfinite target: {actuator_name}")
            return 1
    left_target = env.end_effector_pos("left") + np.array([0.01, 0.0, 0.0], dtype=np.float64)
    right_target = env.end_effector_pos("right") + np.array([0.0, 0.0, 0.01], dtype=np.float64)
    left_ik = env.solve_ee_position_ik("left", left_target)
    right_ik = env.solve_ee_position_ik("right", right_target)
    grasp_site_name = env.nearest_grasp_site_name(env.end_effector_pos("left"))
    grasp_target = env.site_pos(grasp_site_name)
    grasp_ik = env.solve_ee_position_ik("left", grasp_target)
    env.set_left_gripper_gap(0.0, immediate=True)
    bag_frame_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "bag_frame")
    shell_count = len(collect_shell_body_ids(env.model))
    print(f"scene_xml={env.scene_path}")
    print(f"with_ballast={with_ballast}")
    print(f"bag_frame_id={bag_frame_id}")
    print(f"shell_body_count={shell_count}")
    print(f"left_ee_ik={left_ik}")
    print(f"right_ee_ik={right_ik}")
    print(f"nearest_grasp_site={grasp_site_name}")
    print(f"nearest_grasp_ik={grasp_ik}")
    print(f"nearest_grasp_site_pos={grasp_target.tolist()}")
    print(f"left_gripper_gap_m={env.left_gripper_gap()}")
    print("gui_smoke_pass=True")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual UR5 low-fill sack GUI controller")
    parser.add_argument("--no-ballast", action="store_true", help="use shell-only low-fill sack")
    parser.add_argument("--speed", type=float, default=1.0, help="viewer simulation speed multiplier")
    parser.add_argument("--joint-step-deg", type=float, default=JOINT_STEP_DEG_DEFAULT, help="joint +/- button and arrow-key step in degrees")
    parser.add_argument("--keyboard-control", action="store_true", help="enable keyboard control inside the MuJoCo viewer")
    parser.add_argument("--no-keyboard-control", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--smoke-test", action="store_true", help="load the scene without opening GUI windows")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with_ballast = not args.no_ballast
    if args.smoke_test:
        return smoke_test(with_ballast)

    env = DualUR5LowFillEnv(with_ballast=with_ballast)
    env.print_summary()

    lock = threading.RLock()
    stop_event = threading.Event()
    viewer_thread = threading.Thread(
        target=viewer_loop,
        kwargs={
            "env": env,
            "lock": lock,
            "stop_event": stop_event,
            "speed": args.speed,
            "keyboard_control": args.keyboard_control and not args.no_keyboard_control,
            "joint_step_deg": args.joint_step_deg,
        },
        daemon=True,
    )
    viewer_thread.start()

    root = tk.Tk()
    DualUR5Gui(root, env, lock, stop_event, step_deg=args.joint_step_deg)
    try:
        root.mainloop()
    finally:
        stop_event.set()
        viewer_thread.join(timeout=2.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
