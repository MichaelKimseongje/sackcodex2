from __future__ import annotations

import argparse
import importlib.util
import sys
import threading
from pathlib import Path

import tkinter as tk

from scenario_builder import available_scenarios
from dual_ur5_rigid_patch_env import DualUR5RigidPatchEnv, JOINT_STEP_DEG_DEFAULT, smoke_test


ROOT_DIR = Path(__file__).resolve().parent
LOW_FILL_DIR = ROOT_DIR.parent / "01_low_fill"
if str(LOW_FILL_DIR) not in sys.path:
    sys.path.insert(0, str(LOW_FILL_DIR))

_LOW_GUI_SPEC = importlib.util.spec_from_file_location("low_fill_dual_ur5_gui", LOW_FILL_DIR / "run_dual_ur5_gui.py")
if _LOW_GUI_SPEC is None or _LOW_GUI_SPEC.loader is None:
    raise RuntimeError("failed to load low-fill dual UR5 GUI module")
_LOW_GUI_MODULE = importlib.util.module_from_spec(_LOW_GUI_SPEC)
sys.modules[_LOW_GUI_SPEC.name] = _LOW_GUI_MODULE
_LOW_GUI_SPEC.loader.exec_module(_LOW_GUI_MODULE)
DualUR5Gui = _LOW_GUI_MODULE.DualUR5Gui
viewer_loop = _LOW_GUI_MODULE.viewer_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual UR5 + 2F/scoop controller for sealed articulated sack v2")
    parser.add_argument("--scenario", choices=available_scenarios(), default="underfilled")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--joint-step-deg", type=float, default=JOINT_STEP_DEG_DEFAULT)
    parser.add_argument("--keyboard-control", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke_test:
        return smoke_test(args.scenario)

    env = DualUR5RigidPatchEnv(scenario=args.scenario)
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
    try:
        root.mainloop()
    finally:
        stop_event.set()
        viewer_thread.join(timeout=2.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
