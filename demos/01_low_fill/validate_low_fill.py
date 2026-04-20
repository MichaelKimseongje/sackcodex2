from __future__ import annotations

import argparse
import math

import mujoco
import numpy as np

from low_fill_builder import (
    BAG_SHELL_BODY_PREFIX,
    DEMO_SECONDS,
    apply_ballast_impulse,
    collect_body_ids_by_prefix,
    collect_internal_body_ids,
    compute_shell_spans,
    count_escaped_internal_bodies,
    load_scene,
    shell_positions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="standalone underfilled sack validation")
    parser.add_argument(
        "--with-ballast",
        action="store_true",
        help="Version B 단일 ballast까지 함께 검증합니다.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=DEMO_SECONDS,
        help="settle 검증 시간",
    )
    parser.add_argument(
        "--impulse-seconds",
        type=float,
        default=0.8,
        help="ballast lateral impulse 이후 추가 검증 시간",
    )
    return parser.parse_args()


def validate_low_fill(with_ballast: bool, seconds: float, impulse_seconds: float) -> dict[str, float | int | bool | str]:
    xml_path, model, data = load_scene(with_ballast=with_ballast)

    shell_body_ids = collect_body_ids_by_prefix(model, BAG_SHELL_BODY_PREFIX)
    if not shell_body_ids:
        raise RuntimeError("bag shell bodies not found by prefix")

    total_steps = max(1, math.ceil(seconds / model.opt.timestep))
    nonfinite = False
    peak_qvel = 0.0
    escaped_internal_bodies = 0

    for _ in range(total_steps):
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            nonfinite = True
            break

        peak_qvel = max(peak_qvel, float(np.max(np.abs(data.qvel))) if data.qvel.size else 0.0)
        escaped_internal_bodies = max(
            escaped_internal_bodies,
            count_escaped_internal_bodies(model, data, shell_body_ids),
        )

    shell_xyz = shell_positions(data, shell_body_ids)
    upper_span_x, lower_span_x, bag_height = compute_shell_spans(shell_xyz)

    if with_ballast and not nonfinite:
        apply_ballast_impulse(model, data, magnitude=0.35)
        impulse_steps = max(1, math.ceil(impulse_seconds / model.opt.timestep))
        for _ in range(impulse_steps):
            mujoco.mj_step(model, data)
            if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
                nonfinite = True
                break

            peak_qvel = max(peak_qvel, float(np.max(np.abs(data.qvel))) if data.qvel.size else 0.0)
            escaped_internal_bodies = max(
                escaped_internal_bodies,
                count_escaped_internal_bodies(model, data, shell_body_ids),
            )

    upper_ratio = upper_span_x / max(lower_span_x, 1e-9)
    internal_body_count = len(collect_internal_body_ids(model))
    pass_fail = bool(
        not nonfinite
        and upper_ratio <= 0.85
        and bag_height >= 0.12
        and escaped_internal_bodies == 0
    )

    return {
        "xml": str(xml_path),
        "with_ballast": with_ballast,
        "shell_body_prefix": BAG_SHELL_BODY_PREFIX,
        "shell_body_count": len(shell_body_ids),
        "internal_body_count": internal_body_count,
        "upper_span_x": float(upper_span_x),
        "lower_span_x": float(lower_span_x),
        "bag_height": float(bag_height),
        "escaped_internal_bodies": int(escaped_internal_bodies),
        "upper_to_lower_ratio": float(upper_ratio),
        "peak_qvel": float(peak_qvel),
        "nonfinite": bool(nonfinite),
        "pass_fail": pass_fail,
    }


def main() -> int:
    args = parse_args()
    result = validate_low_fill(
        with_ballast=args.with_ballast,
        seconds=args.seconds,
        impulse_seconds=args.impulse_seconds,
    )

    for key in (
        "xml",
        "with_ballast",
        "shell_body_prefix",
        "shell_body_count",
        "internal_body_count",
        "upper_span_x",
        "lower_span_x",
        "bag_height",
        "escaped_internal_bodies",
        "upper_to_lower_ratio",
        "peak_qvel",
        "nonfinite",
        "pass_fail",
    ):
        print(f"{key}={result[key]}")

    return 0 if result["pass_fail"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

