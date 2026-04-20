"""Gripper 없이 open-panel sack hinge motion을 보여주는 viewer."""

from __future__ import annotations

import argparse
import math
import time

import mujoco
import mujoco.viewer
import numpy as np

from joint_only_builder import write_joint_only_scene_xml


HINGE_JOINTS = (
    "hinge_top_panel",
    "hinge_left_side_panel",
    "hinge_right_side_panel",
    "hinge_left_bottom_panel",
    "hinge_right_bottom_panel",
)

ACTUATORS = (
    "top_angle_act",
    "left_side_angle_act",
    "right_side_angle_act",
    "left_bottom_angle_act",
    "right_bottom_angle_act",
)


def name_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise KeyError(name)
    return obj_id


def apply_joint_motion(model: mujoco.MjModel, data: mujoco.MjData, actuator_ids: list[int]) -> None:
    """자루 단면이 접혔다 펴지는 것을 눈으로 보기 위한 자동 joint target."""

    t = data.time
    wave = math.sin(2.0 * math.pi * 0.22 * t)
    wave2 = math.sin(2.0 * math.pi * 0.22 * t + 0.55)

    # 좌우 side/bottom은 반대 방향으로 움직여 단면이 오므라들고 펴지는 모습을 만든다.
    targets = np.array(
        [
            0.12 * wave,
            -0.42 * wave2,
            0.42 * wave2,
            0.35 * wave,
            -0.35 * wave,
        ],
        dtype=float,
    )
    for aid, target in zip(actuator_ids, targets):
        data.ctrl[aid] = target


def joint_angles_deg(model: mujoco.MjModel, data: mujoco.MjData, joint_ids: list[int]) -> dict[str, float]:
    angles = {}
    for jid in joint_ids:
        qadr = model.jnt_qposadr[jid]
        angles[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)] = math.degrees(float(data.qpos[qadr]))
    return angles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-script", action="store_true", help="자동 흔들림 없이 control bar로 hinge actuator를 직접 조작")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--print-every", type=float, default=1.0, help="joint angle 출력 주기 초")
    args = parser.parse_args()

    xml_path = write_joint_only_scene_xml()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    actuator_ids = [name_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ACTUATORS]
    joint_ids = [name_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in HINGE_JOINTS]

    print(f"scene_xml={xml_path}")
    print("이 scene에는 gripper/scoop이 없습니다. control bar의 *_angle_act를 움직이면 panel hinge 각도가 변합니다.")

    last_wall = time.time()
    last_print = 0.0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY
        while viewer.is_running():
            now = time.time()
            elapsed = now - last_wall
            last_wall = now
            sim_steps = max(1, int((elapsed * args.speed) / model.opt.timestep))

            for _ in range(sim_steps):
                if not args.no_script:
                    apply_joint_motion(model, data, actuator_ids)
                mujoco.mj_step(model, data)

            if data.time - last_print >= args.print_every:
                last_print = data.time
                angles = joint_angles_deg(model, data, joint_ids)
                compact = ", ".join(f"{k}={v:+.1f}deg" for k, v in angles.items())
                print(f"t={data.time:.2f}s | {compact}")

            viewer.sync()
            time.sleep(0.002)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
