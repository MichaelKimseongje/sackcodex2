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

    from mujoco_sack_pile.qualitative_reference import QualitativeReferenceGenerator

    parser = argparse.ArgumentParser(description="MuJoCo qualitative shell reference preview")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-id", type=str, default="reference_preview")
    parser.add_argument("--sack-count", type=int, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--fixed-camera", action="store_true")
    parser.add_argument("--settle-steps", type=int, default=300)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    xml_path = QualitativeReferenceGenerator(base_dir).generate(seed=args.seed, episode_id=args.episode_id, sack_count=args.sack_count)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print(f"scene_xml={xml_path}")
    print("mode=qualitative_reference")
    print("scope=auxiliary_reference_only")
    print("설명: 이 모드는 benchmark 정량 평가가 아니라 shell proxy의 정성적 settling/contact 장면을 보는 용도입니다.")

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
