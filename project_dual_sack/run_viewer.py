from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from scenario_builder import SCENARIO_NAMES, write_scene_xml


def run_viewer(
    *,
    scenario: str,
    include_robots: bool = True,
    show_physics: bool = True,
    show_inner: bool = False,
    show_ballast: bool = False,
    show_skin: bool = False,
    speed: float = 1.0,
) -> None:
    xml_path = write_scene_xml(scenario, include_robots=include_robots)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sleep_dt = model.opt.timestep / max(speed, 1e-6)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.38], dtype=np.float64)
        viewer.cam.distance = 2.65
        viewer.cam.azimuth = 130.0
        viewer.cam.elevation = -24.0
        viewer.opt.geomgroup[:] = True
        # group 1: visible articulated outer shell
        # group 2: hidden inner load shell
        # group 3: physics-free visual skin overlay
        # group 4: distributed ballast masses
        # group 5: legacy hidden compatibility patches / hinge cues
        viewer.opt.geomgroup[1] = bool(show_physics)
        viewer.opt.geomgroup[2] = bool(show_inner)
        viewer.opt.geomgroup[3] = bool(show_skin)
        viewer.opt.geomgroup[4] = bool(show_ballast)
        viewer.opt.geomgroup[5] = False
        print(f"scene_xml={xml_path}")
        print(
            f"scenario={scenario}, include_robots={include_robots}, "
            f"show_physics={show_physics}, show_inner={show_inner}, "
            f"show_ballast={show_ballast}, show_skin={show_skin}"
        )
        print("마우스 우클릭 드래그로 카메라 이동, Ctrl+마우스로 body perturb가 가능합니다.")
        print("기본 화면은 outer articulated shell만 보여줍니다. 내부 ballast는 --show-inner를 켜야 보입니다.")
        while viewer.is_running():
            start = time.perf_counter()
            mujoco.mj_step(model, data)
            viewer.sync()
            remain = sleep_dt - (time.perf_counter() - start)
            if remain > 0:
                time.sleep(remain)


def main() -> int:
    parser = argparse.ArgumentParser(description="Open the dual-robot twin-shell sack benchmark scene")
    parser.add_argument("--scenario", choices=SCENARIO_NAMES, default="baseline_filled")
    parser.add_argument("--no-robots", action="store_true")
    parser.add_argument("--hide-physics", action="store_true")
    parser.add_argument("--show-inner", action="store_true")
    parser.add_argument("--show-ballast", action="store_true")
    parser.add_argument("--show-skin", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    run_viewer(
        scenario=args.scenario,
        include_robots=not args.no_robots,
        show_physics=not args.hide_physics,
        show_inner=args.show_inner,
        show_ballast=args.show_ballast,
        show_skin=args.show_skin,
        speed=args.speed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
