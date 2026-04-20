from __future__ import annotations

import argparse
import json
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from generate_sack_mesh import JAW_CLOSED_GAP, JAW_OPEN_GAP, OUTPUT_DIR, available_scenarios
from top_grasp_sim import (
    GraspPatch,
    QUALITY_THRESHOLD,
    _apply_hold_surrogate,
    _capture_point_indices,
    _make_hold_patch,
    _refine_target_z_from_settled_shell,
    _target_for_trial,
    bilateral_contact_balance,
    bundle_thickness_proxy,
    compute_load_margin,
    load_runtime,
    overlap_ratios,
    set_gripper,
)


POSE_DIR = OUTPUT_DIR / "manual_poses"


@dataclass
class ManualState:
    center: np.ndarray
    gap: float
    hold_active: bool = False
    hold_patch: GraspPatch | None = None


def instant_grasp_metrics(runtime, center: np.ndarray, gap: float) -> dict[str, float | int | str | bool]:
    # 수동 모드에서는 pull-test를 자동 실행하지 않고, 현재 jaw 안의 local patch만 즉시 평가한다.
    captured = _capture_point_indices(runtime, center, gap)
    ratios = overlap_ratios(runtime.labels, captured)
    thickness = bundle_thickness_proxy(runtime, captured)
    balance = bilateral_contact_balance(runtime, captured, center, gap)
    instant_quality = compute_load_margin(
        captured_shell_points=len(captured),
        bundle_thickness=thickness,
        balance=balance,
        contact_persistence_ms=180.0 if len(captured) >= 2 else 0.0,
        pull_test_slip_mm=0.0 if len(captured) >= 2 else 999.0,
    )
    return {
        "region_label_at_close": str(ratios["region_label_at_close"]),
        "seam_overlap_ratio": float(ratios["seam_overlap_ratio"]),
        "fold_overlap_ratio": float(ratios["fold_overlap_ratio"]),
        "plain_top_overlap_ratio": float(ratios["plain_top_overlap_ratio"]),
        "captured_shell_points": int(len(captured)),
        "bundle_thickness_proxy": float(thickness),
        "bilateral_contact_balance": float(balance),
        "instant_quality_proxy": float(instant_quality),
        "quality_gate_pass": bool(instant_quality >= QUALITY_THRESHOLD and len(captured) >= 2),
    }


class ManualTopGraspGui:
    def __init__(
        self,
        root: tk.Tk,
        runtime,
        state: ManualState,
        lock: threading.RLock,
        stop_event: threading.Event,
        *,
        scenario: str,
        trial: int,
    ) -> None:
        self.root = root
        self.runtime = runtime
        self.state = state
        self.lock = lock
        self.stop_event = stop_event
        self.scenario = scenario
        self.trial = trial
        self.updating = False

        self.x_var = tk.DoubleVar(value=float(state.center[0]))
        self.y_var = tk.DoubleVar(value=float(state.center[1]))
        self.z_var = tk.DoubleVar(value=float(state.center[2]))
        self.gap_var = tk.DoubleVar(value=float(state.gap * 1000.0))
        self.status_var = tk.StringVar(value="수동 조작 준비됨")
        self.metric_var = tk.StringVar(value="")
        self.watch_var = tk.StringVar(
            value=(
                "볼 것: jaw 사이에 shell point가 들어오는지, captured_shell_points가 2 이상인지, "
                "quality가 threshold를 넘는지, z를 올릴 때 slip/drop이 생기는지"
            )
        )

        self.root.title(f"Manual top grasp GUI - {scenario} trial {trial}")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._build()
        self.refresh_metrics()

    def _build(self) -> None:
        main = tk.Frame(self.root, padx=10, pady=10)
        main.pack(fill=tk.BOTH, expand=True)

        info = tk.LabelFrame(main, text="What to watch", padx=8, pady=6)
        info.pack(fill=tk.X)
        tk.Label(info, textvariable=self.watch_var, wraplength=720, justify=tk.LEFT, anchor="w").pack(fill=tk.X)

        controls = tk.LabelFrame(main, text="Manual 2F end-effector control", padx=8, pady=6)
        controls.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        self._add_slider(controls, "x [m]", self.x_var, -0.16, 0.16, 0.001)
        self._add_slider(controls, "y [m]", self.y_var, -0.12, 0.12, 0.001)
        self._add_slider(controls, "z [m]", self.z_var, 0.08, 0.34, 0.001)
        self._add_slider(controls, "gripper gap [mm]", self.gap_var, 0.0, 160.0, 1.0)

        buttons = tk.LabelFrame(main, text="Actions", padx=8, pady=6)
        buttons.pack(fill=tk.X, pady=(8, 0))
        tk.Button(buttons, text="Move above target ROI", command=self.move_above_target).pack(side=tk.LEFT, padx=3)
        tk.Button(buttons, text="Move to target ROI", command=self.move_to_target).pack(side=tk.LEFT, padx=3)
        tk.Button(buttons, text="Open", command=self.open_gripper).pack(side=tk.LEFT, padx=3)
        tk.Button(buttons, text="Close / pinch", command=self.close_gripper).pack(side=tk.LEFT, padx=3)
        tk.Button(buttons, text="Lift +20 mm", command=lambda: self.nudge_z(0.020)).pack(side=tk.LEFT, padx=3)
        tk.Button(buttons, text="Lower -20 mm", command=lambda: self.nudge_z(-0.020)).pack(side=tk.LEFT, padx=3)

        buttons2 = tk.Frame(main)
        buttons2.pack(fill=tk.X, pady=(6, 0))
        tk.Button(buttons2, text="Enable hold if quality OK", command=self.enable_hold_if_quality_ok).pack(side=tk.LEFT, padx=3)
        tk.Button(buttons2, text="Disable hold", command=self.disable_hold).pack(side=tk.LEFT, padx=3)
        tk.Button(buttons2, text="Reset scene", command=self.reset_scene).pack(side=tk.LEFT, padx=3)
        tk.Button(buttons2, text="Save manual pose", command=self.save_pose).pack(side=tk.LEFT, padx=3)

        metrics = tk.LabelFrame(main, text="Live grasp metrics", padx=8, pady=6)
        metrics.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        tk.Label(metrics, textvariable=self.metric_var, justify=tk.LEFT, anchor="w", font=("Consolas", 10)).pack(fill=tk.BOTH)
        tk.Label(main, textvariable=self.status_var, anchor="w", fg="#245a9a").pack(fill=tk.X, pady=(6, 0))

    def _add_slider(self, parent: tk.Widget, label: str, var: tk.DoubleVar, low: float, high: float, resolution: float) -> None:
        frame = tk.Frame(parent)
        frame.pack(fill=tk.X, pady=3)
        tk.Label(frame, text=label, width=18, anchor="w").pack(side=tk.LEFT)
        slider = tk.Scale(
            frame,
            from_=low,
            to=high,
            resolution=resolution,
            orient=tk.HORIZONTAL,
            variable=var,
            command=lambda _value: self.on_slider_changed(),
            length=520,
        )
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def current_center_gap(self) -> tuple[np.ndarray, float]:
        return (
            np.array([self.x_var.get(), self.y_var.get(), self.z_var.get()], dtype=np.float64),
            max(0.0, self.gap_var.get() / 1000.0),
        )

    def on_slider_changed(self) -> None:
        if self.updating:
            return
        center, gap = self.current_center_gap()
        with self.lock:
            self.state.center = center
            self.state.gap = gap
            self.state.hold_active = False
            self.state.hold_patch = None
        self.status_var.set(f"manual EE xyz=({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}), gap={gap * 1000.0:.1f} mm")

    def set_controls(self, center: np.ndarray | None = None, gap: float | None = None) -> None:
        self.updating = True
        try:
            if center is not None:
                self.x_var.set(float(center[0]))
                self.y_var.set(float(center[1]))
                self.z_var.set(float(center[2]))
            if gap is not None:
                self.gap_var.set(float(gap * 1000.0))
        finally:
            self.updating = False
        center_now, gap_now = self.current_center_gap()
        with self.lock:
            self.state.center = center_now
            self.state.gap = gap_now
            self.state.hold_active = False
            self.state.hold_patch = None

    def target_roi(self) -> np.ndarray:
        planned = _target_for_trial(self.scenario, self.trial)
        with self.lock:
            return _refine_target_z_from_settled_shell(self.runtime, planned)

    def move_above_target(self) -> None:
        target = self.target_roi() + np.array([0.0, 0.0, 0.090], dtype=np.float64)
        self.set_controls(target, JAW_OPEN_GAP)
        self.status_var.set("target ROI 위로 이동: 먼저 여기서 카메라로 위치를 확인하세요")

    def move_to_target(self) -> None:
        self.set_controls(self.target_roi(), JAW_OPEN_GAP)
        self.status_var.set("target ROI로 이동: gap을 줄여 실제로 잡히는지 확인하세요")

    def open_gripper(self) -> None:
        center, _gap = self.current_center_gap()
        self.set_controls(center, JAW_OPEN_GAP)
        self.status_var.set("gripper opened")

    def close_gripper(self) -> None:
        center, _gap = self.current_center_gap()
        self.set_controls(center, JAW_CLOSED_GAP)
        self.status_var.set("gripper closed: live metric에서 captured_shell_points를 확인하세요")

    def nudge_z(self, delta: float) -> None:
        center, gap = self.current_center_gap()
        center[2] += delta
        self.set_controls(center, gap)
        self.status_var.set(f"z moved by {delta * 1000.0:+.0f} mm")

    def enable_hold_if_quality_ok(self) -> None:
        center, gap = self.current_center_gap()
        with self.lock:
            captured = _capture_point_indices(self.runtime, center, gap)
            metrics = instant_grasp_metrics(self.runtime, center, gap)
            if metrics["quality_gate_pass"]:
                self.state.hold_patch = _make_hold_patch(self.runtime, captured, center)
                self.state.hold_active = True
                self.status_var.set("hold surrogate enabled on the currently trapped local patch")
            else:
                self.state.hold_active = False
                self.state.hold_patch = None
                self.status_var.set("quality가 낮아 hold surrogate를 켜지 않았습니다")

    def disable_hold(self) -> None:
        with self.lock:
            self.state.hold_active = False
            self.state.hold_patch = None
        self.status_var.set("hold surrogate disabled")

    def reset_scene(self) -> None:
        with self.lock:
            mujoco.mj_resetData(self.runtime.model, self.runtime.data)
            self.state.hold_active = False
            self.state.hold_patch = None
        self.move_above_target()
        self.status_var.set("scene reset")

    def save_pose(self) -> None:
        POSE_DIR.mkdir(parents=True, exist_ok=True)
        center, gap = self.current_center_gap()
        payload = {
            "schema": "manual_top_grasp_pose.v1",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "scenario": self.scenario,
            "trial": self.trial,
            "center_xyz_m": center.tolist(),
            "gripper_gap_mm": gap * 1000.0,
        }
        path = POSE_DIR / f"manual_top_grasp_{self.scenario}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self.status_var.set(f"saved pose: {path}")

    def refresh_metrics(self) -> None:
        try:
            center, gap = self.current_center_gap()
            with self.lock:
                metrics = instant_grasp_metrics(self.runtime, center, gap)
                hold_active = self.state.hold_active
                bag_nan = not (
                    np.all(np.isfinite(self.runtime.data.qpos))
                    and np.all(np.isfinite(self.runtime.data.qvel))
                )

            text = "\n".join(
                [
                    f"scenario              : {self.scenario} / trial {self.trial}",
                    f"region_label_at_close : {metrics['region_label_at_close']}  (analysis only)",
                    f"seam_overlap_ratio    : {metrics['seam_overlap_ratio']:.2f}",
                    f"fold_overlap_ratio    : {metrics['fold_overlap_ratio']:.2f}",
                    f"plain_top_overlap     : {metrics['plain_top_overlap_ratio']:.2f}",
                    f"captured_shell_points : {metrics['captured_shell_points']}",
                    f"bundle_thickness_proxy: {metrics['bundle_thickness_proxy']:.4f} m",
                    f"bilateral_balance     : {metrics['bilateral_contact_balance']:.3f}",
                    f"instant_quality_proxy : {metrics['instant_quality_proxy']:.3f} / threshold {QUALITY_THRESHOLD:.2f}",
                    f"quality_gate_pass     : {metrics['quality_gate_pass']}",
                    f"hold_surrogate_active : {hold_active}",
                    f"nonfinite             : {bag_nan}",
                ]
            )
            self.metric_var.set(text)
        finally:
            if not self.stop_event.is_set():
                self.root.after(150, self.refresh_metrics)

    def on_close(self) -> None:
        self.stop_event.set()
        self.root.after(50, self.root.destroy)


def viewer_loop(runtime, state: ManualState, lock: threading.RLock, stop_event: threading.Event, *, speed: float) -> None:
    sleep_dt = runtime.model.opt.timestep / max(speed, 1e-6)
    with mujoco.viewer.launch_passive(runtime.model, runtime.data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.17])
        viewer.cam.distance = 0.75
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -18.0
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXFACE] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXEDGE] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXVERT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

        while viewer.is_running() and not stop_event.is_set():
            step_start = time.perf_counter()
            with lock:
                center = state.center.copy()
                gap = float(state.gap)
                hold_patch = state.hold_patch if state.hold_active else None
                set_gripper(runtime, center, gap)
                _apply_hold_surrogate(runtime, hold_patch, center)
                mujoco.mj_step(runtime.model, runtime.data)
            viewer.sync()
            remaining = sleep_dt - (time.perf_counter() - step_start)
            if remaining > 0:
                time.sleep(remaining)
    stop_event.set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="manual top grasp GUI")
    parser.add_argument("--scenario", choices=available_scenarios(), default="simple_fold")
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--speed", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    xml_path, runtime = load_runtime(args.scenario)
    planned = _target_for_trial(args.scenario, args.trial)
    start_center = planned + np.array([0.0, 0.0, 0.090], dtype=np.float64)
    state = ManualState(center=start_center, gap=JAW_OPEN_GAP)
    set_gripper(runtime, state.center, state.gap)
    mujoco.mj_forward(runtime.model, runtime.data)

    print(f"xml={xml_path}")
    print("manual_gui=true")
    print("Use GUI sliders for x/y/z and gripper gap. MuJoCo viewer camera remains mouse-controlled.")

    lock = threading.RLock()
    stop_event = threading.Event()
    thread = threading.Thread(target=viewer_loop, args=(runtime, state, lock, stop_event), kwargs={"speed": args.speed}, daemon=True)
    thread.start()

    root = tk.Tk()
    ManualTopGraspGui(root, runtime, state, lock, stop_event, scenario=args.scenario, trial=args.trial)
    root.mainloop()
    stop_event.set()
    thread.join(timeout=2.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
