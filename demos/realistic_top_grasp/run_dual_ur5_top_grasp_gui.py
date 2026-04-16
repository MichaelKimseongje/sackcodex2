from __future__ import annotations

import argparse
import sys
import threading
import time
import tkinter as tk
from pathlib import Path

import mujoco
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
LOW_FILL_DIR = ROOT_DIR.parent / "01_low_fill"
if str(LOW_FILL_DIR) not in sys.path:
    sys.path.insert(0, str(LOW_FILL_DIR))

from dual_ur5_top_grasp_env import DualUR5TopGraspEnv  # noqa: E402
from generate_sack_mesh import available_content_cases, available_scenarios  # noqa: E402
from run_dual_ur5_gui import DualUR5Gui, JOINT_STEP_DEG_DEFAULT, collect_shell_body_ids, viewer_loop  # noqa: E402
from run_dual_ur5_top_grasp_autograsp import (  # noqa: E402
    APPROACH_OFFSET_Z,
    FOLLOW_MOVE_DELTA,
    PREGRASP_OFFSET_Z,
    PRECAPTURE_GRIPPER_GAP_M,
    STAGE_CLOSE_SECONDS,
    STAGE_HOLD_SECONDS,
    STAGE_INITIAL_HOLD_SECONDS,
    STAGE_MOVE_SECONDS,
    apply_hold_patch,
    choose_grasp_target,
    compute_hold_metrics,
    current_grasp_center,
    label_for_shell_body_id,
    make_hold_patch,
    OPEN_GRIPPER_GAP_M,
)


AUTO_STEP_SECONDS = 0.01
AUTO_VISUAL_PINCH_FINAL_GAP_M = 0.012
AUTO_STABLE_LATCH_GAP_M = 0.026
AUTO_MIN_SAFE_GAP_M = 0.014
AUTO_MAX_PATCH_SPEED_MPS = 0.90
AUTO_MAX_QVEL_NORM = 35.0


class TopGraspGui(DualUR5Gui):
    """상단 파지 데모용 GUI: 수동 조작 위에 자동 파지 버튼을 추가한다."""

    def __init__(
        self,
        root: tk.Tk,
        env: DualUR5TopGraspEnv,
        lock: threading.RLock,
        stop_event: threading.Event,
        *,
        step_deg: float,
    ) -> None:
        self.auto_thread: threading.Thread | None = None
        self.auto_stop_event = threading.Event()
        self.auto_assist_var = tk.BooleanVar(master=root, value=True)
        self.auto_eccentric_var = tk.BooleanVar(master=root, value=True)
        self.auto_label_var = tk.StringVar(master=root, value="auto")
        self.content_bias_var = tk.StringVar(master=root, value="eccentric fill: -")
        super().__init__(root, env, lock, stop_event, step_deg=step_deg)
        self._compact_inherited_controls()
        self._build_auto_panel()

    def _build_auto_panel(self) -> None:
        panel = tk.LabelFrame(self.root, text="visual_demo_assisted", padx=6, pady=4)
        children = self.root.winfo_children()
        pack_options = {"fill": tk.X, "padx": 8, "pady": (0, 4)}
        if len(children) >= 2:
            panel.pack(before=children[1], **pack_options)
        else:
            panel.pack(**pack_options)

        tk.Label(
            panel,
            text="시연용 assist surrogate 모드입니다. 순수 접촉 평가는 run_contact_only_eval.py를 사용합니다.",
            anchor="w",
        ).pack(fill=tk.X)

        controls = tk.Frame(panel)
        controls.pack(fill=tk.X, pady=(2, 0))
        tk.Button(controls, text="Auto hold + 5s check", command=self.start_auto_grasp, padx=4, pady=1).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(controls, text="Stop auto", command=self.stop_auto_grasp, padx=4, pady=1).pack(side=tk.LEFT, padx=(0, 8))
        tk.Checkbutton(controls, text="assist surrogate", variable=self.auto_assist_var).pack(side=tk.LEFT, padx=(0, 12))
        tk.Checkbutton(controls, text="eccentric fill", variable=self.auto_eccentric_var).pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(controls, text="target label").pack(side=tk.LEFT)
        tk.OptionMenu(controls, self.auto_label_var, "auto", "seam", "fold", "plain_top").pack(side=tk.LEFT, padx=(4, 0))

        patch_buttons = tk.Frame(panel)
        patch_buttons.pack(fill=tk.X, pady=(3, 0))
        tk.Label(patch_buttons, text="try local patch:").pack(side=tk.LEFT, padx=(0, 4))
        for label in ("seam", "fold", "plain_top"):
            tk.Button(
                patch_buttons,
                text=f"Try {label}",
                command=lambda selected=label: self.start_auto_grasp_for_label(selected),
                padx=4,
                pady=1,
            ).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(
            patch_buttons,
            text="Move selected only",
            command=self.move_left_to_selected_label,
            padx=4,
            pady=1,
        ).pack(side=tk.LEFT, padx=(8, 4))
        tk.Label(panel, textvariable=self.content_bias_var, anchor="w", fg="#7a3a00").pack(fill=tk.X, pady=(2, 0))

    def _compact_inherited_controls(self) -> None:
        self._compact_widget_tree(self.root)

    def _compact_widget_tree(self, widget: tk.Widget) -> None:
        if isinstance(widget, tk.Button):
            try:
                widget.configure(padx=2, pady=1)
            except tk.TclError:
                pass
        elif isinstance(widget, tk.Scale):
            try:
                length = int(float(widget.cget("length")))
                widget.configure(length=min(length, 180), width=10, sliderlength=18)
            except (tk.TclError, ValueError):
                pass
        elif isinstance(widget, tk.LabelFrame):
            try:
                widget.configure(padx=5, pady=5)
            except tk.TclError:
                pass

        for child in widget.winfo_children():
            self._compact_widget_tree(child)

    def start_auto_grasp(self) -> None:
        if self.auto_thread is not None and self.auto_thread.is_alive():
            self.status_var.set("auto grasp is already running")
            return
        self.auto_stop_event.clear()
        self.auto_thread = threading.Thread(target=self._auto_grasp_worker, daemon=True)
        self.auto_thread.start()
        self.status_var.set("auto grasp started")

    def start_auto_grasp_for_label(self, label: str) -> None:
        self.auto_label_var.set(label)
        self.start_auto_grasp()

    def stop_auto_grasp(self) -> None:
        self.auto_stop_event.set()
        self.status_var.set("auto grasp stop requested")

    def _set_content_bias_var_threadsafe(self, text: str) -> None:
        def _set() -> None:
            if not self.stop_event.is_set():
                self.content_bias_var.set(text)

        try:
            self.root.after(0, _set)
        except tk.TclError:
            pass

    def move_left_to_selected_label(self) -> None:
        requested = self.auto_label_var.get()
        with self.lock:
            if hasattr(self.env, "target_for_label"):
                site_name, _body_id, target_xyz = self.env.target_for_label(requested)
            else:
                reference_xyz = self.env.end_effector_pos("left")
                site_name, target_xyz = self.env.nearest_grasp_target(reference_xyz)
            if self.auto_eccentric_var.get() and hasattr(self.env, "set_content_bias_from_grasp"):
                bias_info = self.env.set_content_bias_from_grasp(target_xyz)
            else:
                bias_info = None
            target_xyz = target_xyz + np.array([0.0, 0.0, APPROACH_OFFSET_Z], dtype=np.float64)

        if bias_info is not None:
            self.content_bias_var.set(
                "eccentric fill: bias=({bias_x:.3f}, {bias_y:.3f}) grasp_to_content=({grasp_to_content_x:.3f}, {grasp_to_content_y:.3f}, {grasp_to_content_z:.3f}) m".format(
                    **bias_info
                )
            )

        control = self.ee_controls["left"]
        self.updating = True
        try:
            for axis, value in zip(("x", "y", "z"), target_xyz):
                control.target_vars[axis].set(round(float(value), 4))
        finally:
            self.updating = False
        self.apply_ee_ik("left")
        self.status_var.set(f"left 2F moved toward {site_name}")

    def _set_status_threadsafe(self, text: str) -> None:
        def _set() -> None:
            if not self.stop_event.is_set():
                self.status_var.set(text)

        try:
            self.root.after(0, _set)
        except tk.TclError:
            pass

    def _wait_or_stop(self, seconds: float, hold_patch=None) -> bool:
        end_time = time.perf_counter() + seconds
        while time.perf_counter() < end_time:
            if self.stop_event.is_set() or self.auto_stop_event.is_set():
                return False
            if hold_patch is not None:
                with self.lock:
                    active = apply_hold_patch(self.env, hold_patch, self.env.end_effector_pos("left"))
                if not active:
                    reason = "unknown"
                    if hold_patch is not None:
                        reason = str(hold_patch.capture_quality.get("release_reason", "unknown"))
                    self._set_status_threadsafe(f"assist released: {reason}")
                    return False
            time.sleep(min(AUTO_STEP_SECONDS, max(0.0, end_time - time.perf_counter())))
        return True

    def _solve_left_to(self, target_xyz: np.ndarray) -> dict[str, float | bool | int]:
        with self.lock:
            return self.env.solve_ee_position_ik(
                "left",
                target_xyz,
                iterations=160,
                tolerance=0.004,
                damping=0.08,
                max_step_deg=5.0,
            )

    def _set_gripper_gap_gradual(self, start_gap: float, end_gap: float, seconds: float, hold_patch=None) -> bool:
        step_count = max(1, int(seconds / AUTO_STEP_SECONDS))
        for step_index in range(step_count):
            if self.stop_event.is_set() or self.auto_stop_event.is_set():
                return False
            alpha = (step_index + 1) / step_count
            gap = (1.0 - alpha) * start_gap + alpha * end_gap
            with self.lock:
                self.env.set_left_gripper_gap(gap, immediate=False)
                if hold_patch is not None:
                    apply_hold_patch(self.env, hold_patch, self.env.end_effector_pos("left"))
            time.sleep(AUTO_STEP_SECONDS)
        return True

    def _patch_mean_speed(self, hold_patch) -> float:
        if hold_patch is None or not hold_patch.body_ids:
            return 0.0
        with self.lock:
            speeds = [
                float(np.linalg.norm(self.env.data.cvel[body_id, 3:6]))
                for body_id in hold_patch.body_ids
            ]
        return float(np.mean(speeds)) if speeds else 0.0

    def _auto_snapshot(self) -> dict[str, np.ndarray]:
        # GUI 자동 시연 중 발산 조짐이 보이면 직전 안정 상태로 되돌리기 위한 최소 상태입니다.
        return {
            "qpos": self.env.data.qpos.copy(),
            "qvel": self.env.data.qvel.copy(),
            "mocap_pos": self.env.data.mocap_pos.copy(),
            "mocap_quat": self.env.data.mocap_quat.copy(),
        }

    def _restore_auto_snapshot(self, snapshot: dict[str, np.ndarray], gap: float) -> None:
        self.env.data.qpos[:] = snapshot["qpos"]
        self.env.data.qvel[:] = snapshot["qvel"]
        self.env.data.mocap_pos[:] = snapshot["mocap_pos"]
        self.env.data.mocap_quat[:] = snapshot["mocap_quat"]
        self.env.data.xfrc_applied[:, :] = 0.0
        self.env.set_left_gripper_gap(gap, immediate=True)
        mujoco.mj_forward(self.env.model, self.env.data)

    def _close_until_stable_latch(
        self,
        start_gap: float,
        target_body_id: int,
        seconds: float,
        assist: bool,
        initial_hold_patch=None,
    ):
        """시연용: 발산 직전까지 무리하게 닫지 않고, 안정 latch가 잡힌 gap에서 멈춘다."""
        step_count = max(1, int(seconds / AUTO_STEP_SECONDS))
        hold_patch = initial_hold_patch
        stable_gap = start_gap
        previous_gap = start_gap
        with self.lock:
            previous_snapshot = self._auto_snapshot()
        for step_index in range(step_count):
            if self.stop_event.is_set() or self.auto_stop_event.is_set():
                return None, previous_gap, False
            alpha = (step_index + 1) / step_count
            commanded_gap = (1.0 - alpha) * start_gap + alpha * AUTO_MIN_SAFE_GAP_M
            with self.lock:
                current_snapshot = self._auto_snapshot()
                self.env.set_left_gripper_gap(commanded_gap, immediate=False)
                if assist and hold_patch is None and commanded_gap <= PRECAPTURE_GRIPPER_GAP_M:
                    hold_patch = make_hold_patch(
                        self.env,
                        self.env.end_effector_pos("left"),
                        required_body_id=target_body_id,
                        allow_visual_pinch_latch=True,
                    )
                    if hold_patch is not None and not bool(hold_patch.capture_quality.get("guarded_latch_approved", False)):
                        hold_patch = None
                if hold_patch is not None:
                    apply_hold_patch(self.env, hold_patch, self.env.end_effector_pos("left"))
                nonfinite = not (np.all(np.isfinite(self.env.data.qpos)) and np.all(np.isfinite(self.env.data.qvel)))
                qvel_norm = float(np.linalg.norm(self.env.data.qvel))
            patch_speed = self._patch_mean_speed(hold_patch)
            if nonfinite or patch_speed > AUTO_MAX_PATCH_SPEED_MPS or qvel_norm > AUTO_MAX_QVEL_NORM:
                with self.lock:
                    self._restore_auto_snapshot(previous_snapshot, previous_gap)
                    if assist and hold_patch is None:
                        hold_patch = make_hold_patch(
                            self.env,
                            self.env.end_effector_pos("left"),
                            required_body_id=target_body_id,
                            allow_visual_pinch_latch=True,
                        )
                        if hold_patch is not None and not bool(hold_patch.capture_quality.get("guarded_latch_approved", False)):
                            hold_patch = None
                    if hold_patch is not None:
                        apply_hold_patch(self.env, hold_patch, self.env.end_effector_pos("left"))
                print(
                    "gui_auto_close_stopped_before_instability=True "
                    f"gap_m={previous_gap:.4f} patch_speed_mps={patch_speed:.3f} qvel_norm={qvel_norm:.3f}"
                )
                return hold_patch, previous_gap, hold_patch is not None
            if hold_patch is not None and bool(hold_patch.capture_quality.get("area_capture_approved", False)):
                stable_gap = commanded_gap
                print(
                    "gui_auto_close_stopped_on_contact_area=True "
                    f"gap_m={stable_gap:.4f} "
                    f"area_proxy_m2={float(hold_patch.capture_quality.get('pad_contact_area_proxy_m2', 0.0)):.5f} "
                    f"captured_points={int(hold_patch.capture_quality.get('captured_shell_points', 0))}"
                )
                return hold_patch, stable_gap, True
            if hold_patch is not None and commanded_gap <= AUTO_STABLE_LATCH_GAP_M:
                stable_gap = commanded_gap
                print(f"gui_auto_close_stopped_at_stable_latch=True gap_m={stable_gap:.4f}")
                return hold_patch, stable_gap, True
            previous_snapshot = current_snapshot
            previous_gap = commanded_gap
            time.sleep(AUTO_STEP_SECONDS)
        return hold_patch, previous_gap, hold_patch is not None

    def _auto_grasp_worker(self) -> None:
        preferred = self.auto_label_var.get()
        preferred_label = None if preferred == "auto" else preferred
        assist = bool(self.auto_assist_var.get())
        hold_patch = None

        try:
            self._set_status_threadsafe("auto: choose target")
            print("mode=visual_demo_assisted")
            print(f"requested_target_label={preferred}")
            with self.lock:
                self.env.set_left_gripper_gap(OPEN_GRIPPER_GAP_M, immediate=True)
                target_label, target_body_id, target_pos = choose_grasp_target(self.env, preferred_label=preferred_label)
                if self.auto_eccentric_var.get() and hasattr(self.env, "set_content_bias_from_grasp"):
                    bias_info = self.env.set_content_bias_from_grasp(target_pos)
                    target_pos = self.env.data.xpos[target_body_id].copy()
                else:
                    bias_info = None
                grasp_center = current_grasp_center(self.env, target_body_id)
                pregrasp = grasp_center + np.array([0.0, 0.0, PREGRASP_OFFSET_Z], dtype=np.float64)

            print(f"gui_auto_chosen_target_label={target_label}")
            print(f"gui_auto_chosen_target_body={target_body_id}")
            print(f"gui_auto_chosen_target_xyz={target_pos.tolist()}")
            print(f"gui_eccentric_fill_bias={bias_info}")
            if bias_info is not None:
                self._set_content_bias_var_threadsafe(
                    "eccentric fill: bias=({bias_x:.3f}, {bias_y:.3f}) grasp_to_content=({grasp_to_content_x:.3f}, {grasp_to_content_y:.3f}, {grasp_to_content_z:.3f}) m".format(
                        **bias_info
                    )
                )

            self._set_status_threadsafe(f"auto: move above {target_label}")
            ik_pre = self._solve_left_to(pregrasp)
            print(f"gui_auto_ik_pregrasp={ik_pre}")
            if not self._wait_or_stop(STAGE_MOVE_SECONDS):
                return

            with self.lock:
                target_label, target_body_id, target_pos = choose_grasp_target(self.env, preferred_label=preferred_label)
            print(f"gui_auto_reacquired_target_label={target_label}")
            print(f"gui_auto_reacquired_target_body={target_body_id}")
            print(f"gui_auto_reacquired_target_xyz={target_pos.tolist()}")

            self._set_status_threadsafe("auto: approach current shell point")
            with self.lock:
                approach = current_grasp_center(self.env, target_body_id) + np.array([0.0, 0.0, APPROACH_OFFSET_Z], dtype=np.float64)
            ik_approach = self._solve_left_to(approach)
            print(f"gui_auto_ik_approach={ik_approach}")
            if not self._wait_or_stop(STAGE_MOVE_SECONDS):
                return

            self._set_status_threadsafe("auto: pre-capture close")
            with self.lock:
                start_gap = self.env.left_gripper_gap()
            if not self._set_gripper_gap_gradual(start_gap, PRECAPTURE_GRIPPER_GAP_M, 0.45):
                return

            self._set_status_threadsafe("auto: move to grasp point")
            with self.lock:
                grasp_center = current_grasp_center(self.env, target_body_id)
            ik_grasp = self._solve_left_to(grasp_center)
            print(f"gui_auto_ik_grasp={ik_grasp}")
            if not self._wait_or_stop(STAGE_MOVE_SECONDS):
                return

            self._set_status_threadsafe("auto: refine jaw contact")
            with self.lock:
                grasp_center = current_grasp_center(self.env, target_body_id)
            ik_final_grasp = self._solve_left_to(grasp_center)
            print(f"gui_auto_ik_final_grasp={ik_final_grasp}")
            if not self._wait_or_stop(0.5 * STAGE_MOVE_SECONDS):
                return

            if assist:
                with self.lock:
                    hold_patch = make_hold_patch(
                        self.env,
                        self.env.end_effector_pos("left"),
                        required_body_id=target_body_id,
                        allow_visual_pinch_latch=True,
                    )
                print(f"gui_auto_preclose_pinch_latch_body_count={0 if hold_patch is None else len(hold_patch.body_ids)}")
                print(f"gui_auto_preclose_pinch_latch_quality={None if hold_patch is None else hold_patch.capture_quality}")
                if hold_patch is not None and not bool(hold_patch.capture_quality.get("guarded_latch_approved", False)):
                    hold_patch = None

            self._set_status_threadsafe("auto: close gripper")
            with self.lock:
                start_gap = self.env.left_gripper_gap()
            if assist:
                close_hold_patch, stable_gap, close_ok = self._close_until_stable_latch(
                    start_gap,
                    target_body_id,
                    STAGE_CLOSE_SECONDS,
                    assist,
                    initial_hold_patch=hold_patch,
                )
                if not close_ok:
                    self._set_status_threadsafe("auto: no stable pinch latch found")
                    print(f"gui_auto_no_stable_pinch_latch=True gap_m={stable_gap:.4f}")
                    return
                hold_patch = close_hold_patch or hold_patch
                print(f"gui_auto_stable_close_gap_m={stable_gap:.4f}")
            else:
                if not self._set_gripper_gap_gradual(start_gap, 0.0, STAGE_CLOSE_SECONDS):
                    return
            with self.lock:
                closed_target_xyz = self.env.data.xpos[target_body_id].copy()

            if assist and hold_patch is None:
                with self.lock:
                    hold_patch = make_hold_patch(
                        self.env,
                        self.env.end_effector_pos("left"),
                        required_body_id=target_body_id,
                        allow_visual_pinch_latch=True,
                    )
            print(f"gui_auto_assist_hold={assist}")
            print(f"gui_auto_hold_patch_body_count={0 if hold_patch is None else len(hold_patch.body_ids)}")
            print(f"gui_auto_hold_patch_capture_quality={None if hold_patch is None else hold_patch.capture_quality}")
            if hold_patch is not None and not bool(hold_patch.capture_quality.get("guarded_latch_approved", False)):
                print("gui_auto_guarded_latch_activated=False")
                hold_patch = None
            elif hold_patch is not None:
                print("gui_auto_guarded_latch_activated=True")
            if hold_patch is not None and hold_patch.body_ids and target_body_id not in hold_patch.body_ids:
                target_body_id = hold_patch.body_ids[0]
                target_label = label_for_shell_body_id(self.env, target_body_id, target_label)
                with self.lock:
                    closed_target_xyz = self.env.data.xpos[target_body_id].copy()
                print(f"gui_auto_actual_captured_body={target_body_id}")
                print(f"gui_auto_actual_captured_label={target_label}")

            self._set_status_threadsafe("auto: hold only, no transport yet")
            if not self._wait_or_stop(STAGE_INITIAL_HOLD_SECONDS, hold_patch=hold_patch):
                return
            with self.lock:
                hold_start_target_xyz = self.env.data.xpos[target_body_id].copy()
                hold_start_ee_xyz = self.env.end_effector_pos("left")
                follow_target = hold_start_ee_xyz + FOLLOW_MOVE_DELTA

            self._set_status_threadsafe(f"auto: 5s follow validation")
            ik_follow = self._solve_left_to(follow_target)
            print(f"gui_auto_ik_follow_validation={ik_follow}")
            if not self._wait_or_stop(STAGE_HOLD_SECONDS, hold_patch=hold_patch):
                return

            with self.lock:
                metrics = compute_hold_metrics(
                    self.env,
                    target_body_id,
                    closed_target_xyz,
                    hold_start_target_xyz,
                    hold_start_ee_xyz,
                    follow_command_xyz=follow_target,
                )
            print(f"gui_auto_hold_validation_seconds={STAGE_HOLD_SECONDS:.1f}")
            print(f"gui_auto_hold_validation_metrics={metrics}")
            verdict = "PASS" if metrics["pass_fail"] else "CHECK"
            self._set_status_threadsafe(
                f"auto {verdict}: follow={metrics['follow_ratio']:.2f}, "
                f"target_move={metrics['target_follow_along_m']:.3f}m, slip={metrics['hold_slip_m']:.3f}m"
            )

            if assist and hold_patch is not None and hold_patch.body_ids:
                self._set_status_threadsafe(
                    f"auto {verdict}: holding after 5s check; press Stop auto to release"
                )
                while not self.stop_event.is_set() and not self.auto_stop_event.is_set():
                    with self.lock:
                        active = apply_hold_patch(self.env, hold_patch, self.env.end_effector_pos("left"))
                    if not active:
                        reason = str(hold_patch.capture_quality.get("release_reason", "unknown"))
                        self._set_status_threadsafe(f"assist released: {reason}")
                        break
                    time.sleep(AUTO_STEP_SECONDS)
            else:
                self._set_status_threadsafe(f"auto {verdict}: done without persistent assist")
        except Exception as exc:
            self._set_status_threadsafe(f"auto grasp failed: {exc}")
            print(f"gui_auto_grasp_failed={exc!r}")
        finally:
            with self.lock:
                self.env.data.xfrc_applied[:, :] = 0.0

    def on_close(self) -> None:
        self.auto_stop_event.set()
        super().on_close()


def smoke_test(scenario_name: str, *, content_case: str, with_content_support: bool) -> int:
    env = DualUR5TopGraspEnv(scenario_name=scenario_name, content_case=content_case, with_content_support=with_content_support)
    shell_count = len(collect_shell_body_ids(env.model))
    nearest_site, nearest_target = env.nearest_grasp_target(env.end_effector_pos("left"))
    left_ik = env.solve_ee_position_ik("left", nearest_target + np.array([0.0, 0.0, 0.010], dtype=np.float64))
    bag_frame_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "bag_frame")
    print(f"scene_xml={env.scene_path}")
    print(f"top_grasp_scenario={scenario_name}")
    print(f"content_case={content_case}")
    print(f"with_content_support={with_content_support}")
    print(f"bag_frame_id={bag_frame_id}")
    print(f"shell_body_count={shell_count}")
    print(f"nearest_grasp_site={nearest_site}")
    print(f"nearest_grasp_site_pos={nearest_target.tolist()}")
    print(f"left_nearest_site_ik={left_ik}")
    print("dual_ur5_top_grasp_gui_smoke_pass=True")
    return 0 if shell_count > 0 and bool(left_ik["success"]) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual UR5 manual GUI for top-grasp sack scenarios")
    parser.add_argument("--scenario", choices=available_scenarios(), default="exposed_seam")
    parser.add_argument("--content-case", choices=available_content_cases(), default="underfilled")
    parser.add_argument("--no-content-support", action="store_true", help="disable the internal 3-clump content surrogate")
    parser.add_argument("--speed", type=float, default=1.0, help="viewer simulation speed multiplier")
    parser.add_argument("--joint-step-deg", type=float, default=JOINT_STEP_DEG_DEFAULT)
    parser.add_argument("--keyboard-control", action="store_true", help="enable keyboard control inside the MuJoCo viewer")
    parser.add_argument("--smoke-test", action="store_true", help="load the scene without opening GUI windows")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with_content_support = not args.no_content_support
    if args.smoke_test:
        return smoke_test(args.scenario, content_case=args.content_case, with_content_support=with_content_support)

    env = DualUR5TopGraspEnv(
        scenario_name=args.scenario,
        content_case=args.content_case,
        with_content_support=with_content_support,
    )
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
            "keyboard_control": args.keyboard_control,
            "joint_step_deg": args.joint_step_deg,
        },
        daemon=True,
    )
    viewer_thread.start()

    root = tk.Tk()
    TopGraspGui(root, env, lock, stop_event, step_deg=args.joint_step_deg)
    try:
        root.mainloop()
    finally:
        stop_event.set()
        viewer_thread.join(timeout=2.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
