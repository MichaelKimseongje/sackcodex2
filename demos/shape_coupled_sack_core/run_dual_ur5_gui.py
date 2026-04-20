from __future__ import annotations

import argparse
import importlib.util
import sys
import threading
from pathlib import Path

import tkinter as tk
import numpy as np

from dual_ur5_shape_coupled_env import JOINT_STEP_DEG_DEFAULT, DualUR5ShapeCoupledEnv, smoke_test
from scenario_builder import available_scenarios


ROOT_DIR = Path(__file__).resolve().parent
LOW_FILL_DIR = ROOT_DIR.parent / "01_low_fill"
if str(LOW_FILL_DIR) not in sys.path:
    sys.path.insert(0, str(LOW_FILL_DIR))

_LOW_GUI_SPEC = importlib.util.spec_from_file_location("shape_core_low_fill_gui", LOW_FILL_DIR / "run_dual_ur5_gui.py")
if _LOW_GUI_SPEC is None or _LOW_GUI_SPEC.loader is None:
    raise RuntimeError("failed to load dual UR5 GUI module")
_LOW_GUI_MODULE = importlib.util.module_from_spec(_LOW_GUI_SPEC)
sys.modules[_LOW_GUI_SPEC.name] = _LOW_GUI_MODULE
_LOW_GUI_SPEC.loader.exec_module(_LOW_GUI_MODULE)
DualUR5Gui = _LOW_GUI_MODULE.DualUR5Gui
viewer_loop = _LOW_GUI_MODULE.viewer_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual UR5 + 2F/scoop GUI for shape-coupled sack core")
    parser.add_argument("--scenario", choices=available_scenarios(), default="underfilled")
    parser.add_argument("--post-release", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--joint-step-deg", type=float, default=JOINT_STEP_DEG_DEFAULT)
    parser.add_argument("--keyboard-control", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke_test:
        return smoke_test(args.scenario, post_release=args.post_release)

    env = DualUR5ShapeCoupledEnv(scenario=args.scenario, post_release=args.post_release)
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
    DualUR5Gui(root, env, lock, stop_event, step_deg=args.joint_step_deg)

    shape_frame = tk.LabelFrame(root, text="Shape-coupled sack response", padx=10, pady=6)
    shape_frame.pack(fill=tk.X, padx=10, pady=(0, 8))
    shape_var = tk.StringVar(value="shape response: -")
    enabled_var = tk.BooleanVar(value=env.shape_response_enabled)
    tk.Label(
        shape_frame,
        textvariable=shape_var,
        justify=tk.LEFT,
        anchor="w",
        font=("Consolas", 9),
    ).pack(fill=tk.X)

    def on_toggle_shape_response() -> None:
        with lock:
            env.shape_response_enabled = bool(enabled_var.get())
        state = "on" if env.shape_response_enabled else "off"
        shape_var.set(f"shape response {state}")

    tk.Checkbutton(
        shape_frame,
        text="Enable contact-driven shape response",
        variable=enabled_var,
        command=on_toggle_shape_response,
    ).pack(anchor="w")
    bag_move = tk.Frame(shape_frame)
    bag_move.pack(fill=tk.X, pady=(4, 0))
    tk.Label(bag_move, text="Sack nudge 2cm:", width=16, anchor="w").pack(side=tk.LEFT)

    def nudge_bag(dx: float, dy: float, dz: float) -> None:
        with lock:
            env.nudge_bag_world(np.array([dx, dy, dz], dtype=np.float64))

    for label, delta in (
        ("X-", (-0.02, 0.0, 0.0)),
        ("X+", (0.02, 0.0, 0.0)),
        ("Y-", (0.0, -0.02, 0.0)),
        ("Y+", (0.0, 0.02, 0.0)),
        ("Z-", (0.0, 0.0, -0.02)),
        ("Z+", (0.0, 0.0, 0.02)),
    ):
        tk.Button(bag_move, text=label, width=4, command=lambda d=delta: nudge_bag(*d)).pack(side=tk.LEFT, padx=1)

    def reset_sack_pose() -> None:
        with lock:
            env.reset_bag_pose()

    tk.Button(
        bag_move,
        text="Reset sack pose",
        command=reset_sack_pose,
    ).pack(side=tk.LEFT, padx=(8, 1))

    def refresh_shape_panel() -> None:
        if stop_event.is_set():
            return
        with lock:
            lines = env.shape_status_lines()
            enabled_text = "on" if env.shape_response_enabled else "off"
        shape_var.set(f"response={enabled_text}\n" + "\n".join(lines))
        root.after(160, refresh_shape_panel)

    refresh_shape_panel()
    try:
        root.mainloop()
    finally:
        stop_event.set()
        viewer_thread.join(timeout=2.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
