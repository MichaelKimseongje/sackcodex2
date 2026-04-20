"""Joint-only scene에서 panel hinge 각도가 실제로 변하는지 검증한다."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import mujoco
import numpy as np

from joint_only_builder import PROJECT_DIR, write_joint_only_scene_xml
from run_joint_demo import ACTUATORS, HINGE_JOINTS, apply_joint_motion, name_id


OUT_DIR = PROJECT_DIR / "out"


def main() -> int:
    xml_path = write_joint_only_scene_xml()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    actuator_ids = [name_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in ACTUATORS]
    joint_ids = [name_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in HINGE_JOINTS]

    rows = []
    samples = {name: [] for name in HINGE_JOINTS}
    steps = int(4.0 / model.opt.timestep)
    for step in range(steps):
        apply_joint_motion(model, data, actuator_ids)
        mujoco.mj_step(model, data)
        if step % 20 == 0:
            row = {"time": float(data.time)}
            for name, jid in zip(HINGE_JOINTS, joint_ids):
                q = math.degrees(float(data.qpos[model.jnt_qposadr[jid]]))
                row[name + "_deg"] = q
                samples[name].append(q)
            rows.append(row)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "joint_motion_timeseries.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    ranges = {name: float(max(vals) - min(vals)) for name, vals in samples.items()}
    finite = bool(np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel)))
    moving = all(value > 2.0 for value in ranges.values())
    pass_fail = finite and moving

    print(f"scene_xml={xml_path}")
    print(f"joint_motion_csv={csv_path}")
    for name, value in ranges.items():
        print(f"{name}_range_deg={value:.2f}")
    print(f"finite={finite}")
    print(f"joint_angles_change_pass={moving}")
    print(f"pass_fail={pass_fail}")
    return 0 if pass_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
