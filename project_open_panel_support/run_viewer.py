"""MuJoCo viewer for the open-panel support-state prototype."""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer

from builder import write_scene_xml
from open_panel_env import OpenPanelSupportEnv


def scripted_controls(env: OpenPanelSupportEnv) -> None:
    """viewer에서 자동 close/insertion을 천천히 보여주기 위한 간단한 sequence."""

    t = env.data.time
    if t < 0.8:
        close = 0.0
        scoop = 0.0
        lift = 0.0
    elif t < 2.0:
        close = 0.052 * ((t - 0.8) / 1.2)
        scoop = 0.0
        lift = 0.0
    elif t < 3.0:
        close = 0.052
        scoop = 0.0
        lift = 0.035 * ((t - 2.0) / 1.0)
    elif t < 5.0:
        close = 0.052
        scoop = 0.205 * ((t - 3.0) / 2.0)
        lift = 0.035
    else:
        close = 0.052
        scoop = 0.205
        lift = 0.035
    env.set_controls(left_close=close, right_close=close, scoop_depth=scoop, gripper_lift=lift)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-script", action="store_true", help="자동 gripper/scoop sequence 없이 viewer control bar로 조작")
    parser.add_argument("--speed", type=float, default=1.0, help="실시간 대비 simulation speed")
    args = parser.parse_args()

    xml_path = write_scene_xml()
    env = OpenPanelSupportEnv(xml_path)
    env.settle(0.25)

    print(f"scene_xml={xml_path}")
    print("viewer: control bar에서 left/right_finger_close_act, scoop_insert_act를 직접 조작할 수 있습니다.")
    print("guarded grasp는 조건 통과 시에만 guarded_grasp_connect가 활성화됩니다.")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        last = time.time()
        while viewer.is_running():
            now = time.time()
            elapsed = now - last
            last = now
            sim_steps = max(1, int((elapsed * args.speed) / env.model.opt.timestep))
            for _ in range(sim_steps):
                if not args.no_script:
                    scripted_controls(env)
                env.step(1, guarded_update=True)
            viewer.sync()
            time.sleep(0.002)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
