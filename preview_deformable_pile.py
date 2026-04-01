from __future__ import annotations

import argparse
import time
from pathlib import Path


def main():
    try:
        import mujoco
        import mujoco.viewer
    except ModuleNotFoundError as exc:
        raise SystemExit("MuJoCo Python 패키지가 필요합니다.") from exc

    from mujoco_sack_pile.deformable_generator import DeformablePileGenerator

    parser = argparse.ArgumentParser(description="MuJoCo deformable sack pile preview")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-id", type=str, default="deformable_preview")
    parser.add_argument("--sack-count", type=int, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--fixed-camera", action="store_true")
    parser.add_argument("--settle-steps", type=int, default=300)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    xml_path = DeformablePileGenerator(base_dir).generate(seed=args.seed, episode_id=args.episode_id, sack_count=args.sack_count)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print(f"scene_xml={xml_path}")
    print("mode=deformable_preview")
    print("설명: baseline 없이 deformable sack pile이 가라앉고 얽히는 장면을 확인하는 모드입니다.")

    if args.headless:
        for _ in range(args.settle_steps):
            mujoco.mj_step(model, data)
        print("headless_settle_done=true")
        return

    with mujoco.viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True) as viewer:
        if args.fixed_camera:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overview")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
