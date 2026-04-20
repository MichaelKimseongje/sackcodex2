from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import mujoco
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="bag_base.xml이 정상 로드되고 폭주 없이 settle되는지 검사합니다."
    )
    parser.add_argument(
        "--xml",
        type=Path,
        default=Path(__file__).with_name("bag_base.xml"),
        help="검사할 MJCF 경로",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=3.0,
        help="검사할 시뮬레이션 시간(초)",
    )
    parser.add_argument(
        "--tail-seconds",
        type=float,
        default=0.2,
        help="settle 판정을 위한 마지막 구간 길이(초)",
    )
    parser.add_argument(
        "--peak-qvel-limit",
        type=float,
        default=25.0,
        help="전체 구간 최대 qvel 한계",
    )
    parser.add_argument(
        "--tail-qvel-limit",
        type=float,
        default=0.05,
        help="마지막 구간 평균 qvel 한계",
    )
    parser.add_argument(
        "--min-shell-height",
        type=float,
        default=-0.02,
        help="shell body 중심의 최소 허용 높이",
    )
    return parser.parse_args()


def shell_body_ids(model: mujoco.MjModel) -> list[int]:
    ids: list[int] = []
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name and name.startswith("bag_shell_"):
            ids.append(body_id)
    return ids


def main() -> int:
    args = parse_args()
    xml_path = args.xml.resolve()

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    shell_ids = shell_body_ids(model)
    if not shell_ids:
        print("validation_passed=false")
        print("reason=no_shell_bodies")
        return 1

    total_steps = max(1, math.ceil(args.seconds / model.opt.timestep))
    tail_steps = max(1, math.ceil(args.tail_seconds / model.opt.timestep))

    peak_qvel = 0.0
    min_shell_z = float("inf")
    max_contacts = 0
    tail_samples: list[float] = []
    nonfinite = False

    for _ in range(total_steps):
        mujoco.mj_step(model, data)

        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            nonfinite = True
            break

        current_peak = float(np.max(np.abs(data.qvel))) if data.qvel.size else 0.0
        peak_qvel = max(peak_qvel, current_peak)
        max_contacts = max(max_contacts, int(data.ncon))

        shell_positions = data.xpos[shell_ids, 2]
        min_shell_z = min(min_shell_z, float(np.min(shell_positions)))

        tail_samples.append(current_peak)
        if len(tail_samples) > tail_steps:
            tail_samples.pop(0)

    tail_mean_qvel = float(np.mean(tail_samples)) if tail_samples else float("inf")
    tail_max_qvel = float(np.max(tail_samples)) if tail_samples else float("inf")

    passed = (
        not nonfinite
        and peak_qvel <= args.peak_qvel_limit
        and tail_mean_qvel <= args.tail_qvel_limit
        and min_shell_z >= args.min_shell_height
    )

    print(f"xml={xml_path}")
    print(f"bag_frame_id={mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'bag_frame')}")
    print(f"floor_id={mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')}")
    print(f"bag_shell_id={mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, 'bag_shell')}")
    print(f"shell_body_count={len(shell_ids)}")
    print(f"simulated_seconds={data.time:.3f}")
    print(f"nonfinite={nonfinite}")
    print(f"peak_qvel={peak_qvel:.6f}")
    print(f"tail_mean_qvel={tail_mean_qvel:.6f}")
    print(f"tail_max_qvel={tail_max_qvel:.6f}")
    print(f"min_shell_height={min_shell_z:.6f}")
    print(f"max_contacts={max_contacts}")
    print(f"validation_passed={passed}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
