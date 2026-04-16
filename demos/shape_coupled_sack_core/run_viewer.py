from __future__ import annotations

import argparse
import time

import mujoco

try:
    import mujoco.viewer
except Exception:  # pragma: no cover
    mujoco_viewer = None
else:
    mujoco_viewer = mujoco.viewer

from build_shape_coupled_sack import write_scene_xml
from scenario_builder import available_scenarios


def main() -> int:
    parser = argparse.ArgumentParser(description="View shape-coupled sack core scene")
    parser.add_argument("--scenario", choices=available_scenarios(), default="underfilled")
    parser.add_argument("--post-release", action="store_true")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    xml_path = write_scene_xml(args.scenario, post_release=args.post_release)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(f"scene_xml={xml_path}")
    print(f"scenario={args.scenario}")
    print(f"post_release={args.post_release}")
    print(f"nbody={model.nbody}")
    print(f"ngeom={model.ngeom}")
    print(f"nflex={int(getattr(model, 'nflex', 0))}")

    if args.no_viewer:
        for _ in range(int(1.0 / model.opt.timestep)):
            mujoco.mj_step(model, data)
        print("loaded_and_stepped=true")
        return 0

    if mujoco_viewer is None:
        raise RuntimeError("mujoco.viewer를 사용할 수 없습니다.")
    with mujoco_viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [0.0, 0.0, 0.22]
        viewer.cam.distance = 1.5
        viewer.cam.azimuth = 130
        viewer.cam.elevation = -22
        sleep_dt = model.opt.timestep / max(args.speed, 1e-6)
        while viewer.is_running():
            start = time.perf_counter()
            mujoco.mj_step(model, data)
            viewer.sync()
            rest = sleep_dt - (time.perf_counter() - start)
            if rest > 0:
                time.sleep(rest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
