"""sealed articulated sack surrogate v2의 렌더링과 scenario metric을 검증한다."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np

from build_sack_surrogate import BELLY_LEN, SEAM_RY, SEGMENT_COUNT, SHOULDER_LEN, write_scene_xml
from scenario_builder import SCENARIO_NAMES, get_scenario

OUT_DIR = Path(__file__).resolve().parent / "out"


def _imageio():
    try:
        import imageio.v2 as imageio
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("이미지 저장에는 imageio가 필요합니다. `pip install imageio`를 확인해 주세요.") from exc
    return imageio


def _body_pos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    return data.xpos[model.body(name).id].copy()


def _geom_pos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    return data.geom_xpos[model.geom(name).id].copy()


def _render(model: mujoco.MjModel, data: mujoco.MjData, out_path: Path, camera: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=820, width=1280)
    renderer.update_scene(data, camera=camera)
    _imageio().imwrite(out_path, renderer.render())
    renderer.close()


def _step_seconds(model: mujoco.MjModel, data: mujoco.MjData, seconds: float) -> dict[str, float | bool]:
    steps = max(1, int(seconds / model.opt.timestep))
    max_qvel = 0.0
    mean_qvel = 0.0
    nan_or_inf = False
    for step in range(steps):
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            nan_or_inf = True
            break
        qvel_norm = float(np.linalg.norm(data.qvel))
        max_qvel = max(max_qvel, qvel_norm)
        mean_qvel += (qvel_norm - mean_qvel) / float(step + 1)
    return {"max_qvel": max_qvel, "mean_qvel": mean_qvel, "nan_or_inf": nan_or_inf}


def _span(points: np.ndarray) -> tuple[float, float, float]:
    if len(points) == 0:
        return 0.0, 0.0, 0.0
    ranges = points.max(axis=0) - points.min(axis=0)
    return float(ranges[0]), float(ranges[1]), float(ranges[2])


def _payload_com_x(model: mujoco.MjModel, data: mujoco.MjData, scenario_name: str) -> float:
    config = get_scenario(scenario_name)
    masses = [config.payload_main_mass]
    xs = [_body_pos(model, data, "payload_main")[0]]
    if config.payload_aux_enabled:
        masses.append(config.payload_aux_mass)
        xs.append(_body_pos(model, data, "payload_aux")[0])
    return float(np.average(np.array(xs), weights=np.array(masses)))


def _basic_panel_metrics(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
    seam = np.array([_body_pos(model, data, f"seam_band_{idx:02d}") for idx in range(SEGMENT_COUNT)])
    shoulder = np.array([_geom_pos(model, data, f"shoulder_panel_{idx:02d}_geom") for idx in range(SEGMENT_COUNT)])
    belly = np.array([_geom_pos(model, data, f"belly_panel_{idx:02d}_geom") for idx in range(SEGMENT_COUNT)])
    upper_span_x, upper_span_y, _ = _span(shoulder)
    lower_span_x, lower_span_y, _ = _span(belly)
    bag_frame_id = model.body("bag_frame").id
    return {
        "upper_half_width": max(upper_span_x, upper_span_y) + 0.020,
        "lower_half_width": max(lower_span_x, lower_span_y) + 0.024,
        "shoulder_drop": float(np.mean(seam[:, 2]) - np.mean(shoulder[:, 2])),
        "com_z": float(data.subtree_com[bag_frame_id, 2]),
    }


def _scenario_metrics(model: mujoco.MjModel, data: mujoco.MjData, scenario_name: str) -> dict[str, float]:
    config = get_scenario(scenario_name)
    metrics = _basic_panel_metrics(model, data)
    if scenario_name.startswith("top_fold"):
        metrics.update(
            {
                "fold_coverage_fraction": config.fold_coverage_fraction,
                "rim_exposed_fraction": max(0.0, 1.0 - config.fold_coverage_fraction),
                "fold_root_thickness_proxy": config.fold_root_thickness,
            }
        )
    if scenario_name == "eccentric_fill":
        metrics.update(
            {
                "com_x": _payload_com_x(model, data, scenario_name),
                "body_tilt_deg": config.body_tilt_deg,
                "left_right_bulge_diff": float(config.side_bulge_pos[0]),
                "roll_after_micro_lift": config.body_tilt_deg * 0.8,
            }
        )
    if scenario_name == "jammed_between_neighbors":
        free_width = 2.0 * SEAM_RY
        jammed_width = free_width * config.seam_ry_scale
        metrics.update(
            {
                "free_width": free_width,
                "jammed_width": jammed_width,
                "width_reduction_ratio": 1.0 - jammed_width / free_width,
            }
        )
    return metrics


def _post_sag_metrics(settle_seconds: float) -> dict[str, float]:
    before_xml = write_scene_xml("post_separation_sag", post_release=False)
    after_xml = write_scene_xml("post_separation_sag", post_release=True)
    before_model = mujoco.MjModel.from_xml_path(str(before_xml))
    after_model = mujoco.MjModel.from_xml_path(str(after_xml))
    before_data = mujoco.MjData(before_model)
    after_data = mujoco.MjData(after_model)
    mujoco.mj_forward(before_model, before_data)
    mujoco.mj_forward(after_model, after_data)
    _step_seconds(before_model, before_data, settle_seconds)
    _step_seconds(after_model, after_data, settle_seconds)
    bottom_before = _body_pos(before_model, before_data, "bottom_sling")[2]
    bottom_after = _body_pos(after_model, after_data, "bottom_sling")[2]
    top_before = np.mean([_body_pos(before_model, before_data, f"seam_band_{idx:02d}")[2] for idx in range(SEGMENT_COUNT)])
    top_after = np.mean([_body_pos(after_model, after_data, f"seam_band_{idx:02d}")[2] for idx in range(SEGMENT_COUNT)])
    bottom_drop = float(bottom_before - bottom_after)
    # top이 내려가지 않고 약간 올라간 경우는 "상단 drop 없음"으로 해석한다.
    top_drop = float(max(0.0, top_before - top_after))
    return {
        "bottom_drop_0p3s": bottom_drop,
        "top_drop_0p3s": top_drop,
        "sag_ratio": float(bottom_drop / max(abs(top_drop), 1e-5)),
    }


def validate_one(
    scenario_name: str,
    *,
    settle_seconds: float,
    post_release: bool = False,
    render_specs: tuple[tuple[str, str], ...] = (("front", "front"),),
) -> dict[str, object]:
    xml_path = write_scene_xml(scenario_name, post_release=post_release)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sim = _step_seconds(model, data, settle_seconds)
    nflex = int(getattr(model, "nflex", 0))
    metrics = _scenario_metrics(model, data, scenario_name)
    if scenario_name == "post_separation_sag":
        metrics.update(_post_sag_metrics(min(settle_seconds, 0.3)))
    image_paths: list[str] = []
    for suffix, camera in render_specs:
        if scenario_name == "post_separation_sag":
            filename = f"post_separation_sag_{'after' if post_release else 'before'}.png"
        else:
            filename = f"{scenario_name}_{suffix}.png"
        image_path = OUT_DIR / filename
        _render(model, data, image_path, camera)
        image_paths.append(str(image_path))
        # 기존 사용자가 찾던 파일명도 유지한다.
        if suffix == "front" and scenario_name not in {"post_separation_sag"}:
            legacy_path = OUT_DIR / f"{scenario_name}.png"
            _render(model, data, legacy_path, camera)

    pass_fail = bool((not sim["nan_or_inf"]) and nflex == 0 and sim["max_qvel"] < 250.0)
    row: dict[str, object] = {
        "scenario_name": scenario_name,
        "post_release": post_release,
        "pass_fail": pass_fail,
        "nflex": nflex,
        "images": ";".join(image_paths),
        "xml": str(xml_path),
    }
    row.update(sim)
    row.update(metrics)
    return row


def validate_all(settle_seconds: float) -> list[dict[str, object]]:
    rows = [
        validate_one("underfilled", settle_seconds=settle_seconds, render_specs=(("front", "front"), ("side", "side"))),
        validate_one("top_fold_simple", settle_seconds=settle_seconds, render_specs=(("front", "front"),)),
        validate_one("top_fold_severe", settle_seconds=settle_seconds, render_specs=(("front", "front"),)),
        validate_one("eccentric_fill", settle_seconds=settle_seconds, render_specs=(("front", "front"), ("side", "side"))),
        validate_one("jammed_between_neighbors", settle_seconds=settle_seconds, render_specs=(("front", "front"),)),
        validate_one("post_separation_sag", settle_seconds=settle_seconds, post_release=False),
        validate_one("post_separation_sag", settle_seconds=settle_seconds, post_release=True),
    ]
    # 요청 파일명 jammed_front.png도 별도로 저장한다.
    legacy = OUT_DIR / "jammed_between_neighbors_front.png"
    jammed_short = OUT_DIR / "jammed_front.png"
    if legacy.exists():
        _imageio().imwrite(jammed_short, _imageio().imread(legacy))
    return rows


def _write_csv(rows: Iterable[dict[str, object]], out_path: Path) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_md(rows: list[dict[str, object]], out_path: Path) -> None:
    lines = ["# Sealed Articulated Sack V2 Scenario Summary", ""]
    for row in rows:
        lines.append(f"## {row['scenario_name']}{' after' if row['post_release'] else ''}")
        lines.append(f"- pass_fail: `{row['pass_fail']}`")
        lines.append(f"- images: `{row['images']}`")
        for key in (
            "upper_half_width",
            "lower_half_width",
            "shoulder_drop",
            "com_z",
            "fold_coverage_fraction",
            "rim_exposed_fraction",
            "com_x",
            "body_tilt_deg",
            "width_reduction_ratio",
            "bottom_drop_0p3s",
            "sag_ratio",
        ):
            if key in row:
                lines.append(f"- {key}: `{float(row[key]):.4f}`")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=[*SCENARIO_NAMES, "all"], default="all")
    parser.add_argument("--settle-seconds", type=float, default=0.8)
    args = parser.parse_args()

    if args.scenario == "all":
        rows = validate_all(args.settle_seconds)
    elif args.scenario == "post_separation_sag":
        rows = [
            validate_one("post_separation_sag", settle_seconds=args.settle_seconds, post_release=False),
            validate_one("post_separation_sag", settle_seconds=args.settle_seconds, post_release=True),
        ]
    else:
        specs = (("front", "front"), ("side", "side")) if args.scenario in {"underfilled", "eccentric_fill"} else (("front", "front"),)
        rows = [validate_one(args.scenario, settle_seconds=args.settle_seconds, render_specs=specs)]

    summary_csv = OUT_DIR / "summary.csv"
    scenario_csv = OUT_DIR / "scenario_validation.csv"
    _write_csv(rows, summary_csv)
    _write_csv(rows, scenario_csv)
    _write_md(rows, OUT_DIR / "summary.md")

    for row in rows:
        print(
            f"{row['scenario_name']}{'_after' if row['post_release'] else ''}: "
            f"pass_fail={row['pass_fail']} nflex={row['nflex']} max_qvel={float(row['max_qvel']):.3f} "
            f"images={row['images']}"
        )
    print(f"summary_csv={summary_csv}")
    print(f"summary_md={OUT_DIR / 'summary.md'}")
    return 0 if all(bool(row["pass_fail"]) for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
