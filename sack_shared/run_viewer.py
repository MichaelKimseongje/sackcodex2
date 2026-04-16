from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from build_shared_sack import write_scene_xml


def run_viewer(*, speed: float = 1.0, show_physics: bool = False) -> None:
    xml_path = write_scene_xml()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sleep_dt = model.opt.timestep / max(speed, 1e-6)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.12], dtype=np.float64)
        viewer.cam.distance = 0.72
        viewer.cam.azimuth = 130.0
        viewer.cam.elevation = -20.0
        viewer.opt.geomgroup[:] = True
        # group 1은 물리용 articulated patch입니다. 기본은 숨겨서 포대 외형만 봅니다.
        viewer.opt.geomgroup[1] = bool(show_physics)

        print(f"scene_xml={xml_path}")
        print("viewer controls: 마우스 우클릭/휠로 카메라 이동, Ctrl+마우스로 body perturb")
        print(f"show_physics={show_physics}  # True이면 내부 articulated patch 골격을 함께 표시합니다.")
        while viewer.is_running():
            start = time.perf_counter()
            mujoco.mj_step(model, data)
            viewer.sync()
            remain = sleep_dt - (time.perf_counter() - start)
            if remain > 0:
                time.sleep(remain)


def main() -> int:
    parser = argparse.ArgumentParser(description="Open MuJoCo viewer for the shared articulated sack skeleton")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--show-physics", action="store_true")
    args = parser.parse_args()
    run_viewer(speed=args.speed, show_physics=args.show_physics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
