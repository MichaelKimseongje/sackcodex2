"""반강체 sack surrogate 장면을 MuJoCo viewer로 실행한다."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco

try:
    import mujoco.viewer
except Exception:  # pragma: no cover - headless 환경 대비
    mujoco_viewer = None
else:
    mujoco_viewer = mujoco.viewer

from build_sack_surrogate import write_scene_xml
from scenario_builder import available_scenarios


def load_scene(scenario: str, post_release: bool = False) -> tuple[mujoco.MjModel, mujoco.MjData, Path]:
    """선택한 scenario XML을 생성하고 MuJoCo model/data를 반환한다."""
    xml_path = write_scene_xml(scenario, post_release=post_release)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, xml_path


def run_passive_viewer(model: mujoco.MjModel, data: mujoco.MjData, speed: float) -> None:
    """마우스 카메라 조작이 가능한 passive viewer loop."""
    if mujoco_viewer is None:
        raise RuntimeError("mujoco.viewer를 사용할 수 없습니다. GUI 가능한 Python 환경에서 실행해 주세요.")

    dt = float(model.opt.timestep)
    with mujoco_viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.distance = 2.2
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        viewer.cam.lookat[:] = [0.0, 0.0, 0.25]

        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            elapsed = time.time() - step_start
            sleep_time = max(0.0, dt / max(speed, 1e-6) - elapsed)
            if sleep_time:
                time.sleep(sleep_time)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=available_scenarios(), default="underfilled")
    parser.add_argument(
        "--post-release",
        action="store_true",
        help="post_separation_sag에서 hidden support 제거 후 장면을 연다.",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="viewer 재생 속도 배율")
    parser.add_argument("--no-viewer", action="store_true", help="로드와 짧은 stepping만 확인")
    args = parser.parse_args()

    model, data, xml_path = load_scene(args.scenario, post_release=args.post_release)
    nflex = int(getattr(model, "nflex", 0))
    print(f"scene_xml={xml_path}")
    print(f"scenario={args.scenario}")
    print(f"post_release={args.post_release}")
    print(f"nbody={model.nbody}")
    print(f"ngeom={model.ngeom}")
    print(f"nflex={nflex}  # 0이어야 함: flex/soft shell 미사용")

    if args.no_viewer:
        for _ in range(int(1.0 / model.opt.timestep)):
            mujoco.mj_step(model, data)
        print("loaded_and_stepped=true")
        return 0

    run_passive_viewer(model, data, args.speed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
