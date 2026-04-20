from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MuJoCo viewer로 bag_base.xml을 로드하고 실시간 시뮬레이션합니다."
    )
    parser.add_argument(
        "--xml",
        type=Path,
        default=Path(__file__).with_name("bag_base.xml"),
        help="로드할 MJCF 경로",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="실시간 배속. 1.0이면 기본 속도입니다.",
    )
    parser.add_argument(
        "--scenario",
        choices=("base", "low_fill"),
        default="base",
        help="base 또는 low_fill scenario를 선택합니다.",
    )
    parser.add_argument(
        "--fill-mode",
        choices=("ballast1", "clump3"),
        default="ballast1",
        help="low_fill scenario에서 사용할 내부 surrogate 모드입니다.",
    )
    parser.add_argument(
        "--fixed-camera",
        action="store_true",
        help="기본 자유 카메라 대신 overview 고정 카메라를 사용합니다.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scenario == "low_fill":
        from scenario_low_fill import make_low_fill, stage_low_fill_demo

        xml_path = make_low_fill(fill_mode=args.fill_mode).resolve()
    else:
        xml_path = args.xml.resolve()

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    print(f"xml={xml_path}")
    print(f"scenario={args.scenario}")
    if args.scenario == "low_fill":
        print(f"fill_mode={args.fill_mode}")
    print(f"body: bag_frame -> id={mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'bag_frame')}")
    print(f"geom: floor -> id={mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')}")
    print(f"flex: bag_shell -> id={mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, 'bag_shell')}")
    print("마우스로 카메라 이동이 가능합니다. low_fill은 시작 시 내부 clump를 올려두고 settle되는 데모로 실행됩니다.")

    if args.scenario == "low_fill":
        stage_low_fill_demo(model, data, fill_mode=args.fill_mode)

    timestep = model.opt.timestep
    sleep_dt = timestep / max(args.speed, 1e-6)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        if args.fixed_camera:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overview")
        else:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.14])
            viewer.cam.distance = 1.35
            viewer.cam.azimuth = 132.0
            viewer.cam.elevation = -18.0

        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXFACE] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXEDGE] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXVERT] = True

        while viewer.is_running():
            step_start = time.perf_counter()
            mujoco.mj_step(model, data)
            viewer.sync()

            elapsed = time.perf_counter() - step_start
            remaining = sleep_dt - elapsed
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
