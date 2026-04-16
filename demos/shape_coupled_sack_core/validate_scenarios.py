from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np

from build_shape_coupled_sack import OUT_DIR, SEGMENT_COUNT, write_scene_xml
from scenario_builder import SCENARIO_NAMES, get_scenario


def _imageio():
    try:
        import imageio.v2 as imageio
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("imageio가 필요합니다. `pip install imageio`를 확인해 주세요.") from exc
    return imageio


def _body_pos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    return data.xpos[model.body(name).id].copy()


def _geom_pos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    return data.geom_xpos[model.geom(name).id].copy()


def _render(model: mujoco.MjModel, data: mujoco.MjData, path: Path, camera: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, width=1280, height=820)
    renderer.update_scene(data, camera=camera)
    _imageio().imwrite(path, renderer.render())
    renderer.close()


def _step(model: mujoco.MjModel, data: mujoco.MjData, seconds: float) -> dict[str, float | bool]:
    steps = max(1, int(seconds / model.opt.timestep))
    mean_qvel = 0.0
    max_qvel = 0.0
    nan_or_inf = False
    for i in range(steps):
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            nan_or_inf = True
            break
        qvel_norm = float(np.linalg.norm(data.qvel))
        max_qvel = max(max_qvel, qvel_norm)
        mean_qvel += (qvel_norm - mean_qvel) / float(i + 1)
    return {"mean_qvel": mean_qvel, "max_qvel": max_qvel, "nan_or_inf": nan_or_inf}


def _basic_metrics(model: mujoco.MjModel, data: mujoco.MjData, scenario: str) -> dict[str, float | str]:
    shoulder = np.array([_geom_pos(model, data, f"shoulder_panel_{i:02d}_geom") for i in range(SEGMENT_COUNT)])
    belly = np.array([_geom_pos(model, data, f"belly_panel_{i:02d}_geom") for i in range(SEGMENT_COUNT)])
    seam = np.array([_body_pos(model, data, f"seam_band_{i:02d}") for i in range(SEGMENT_COUNT)])
    upper_width = float(max(shoulder[:, 0].ptp(), shoulder[:, 1].ptp()) + 0.022)
    lower_width = float(max(belly[:, 0].ptp(), belly[:, 1].ptp()) + 0.030)
    bag_id = model.body("bag_frame").id
    preferred_label = "seam"
    preferred_rank = 1
    if scenario == "underfilled":
        preferred_label = "shoulder"
    if scenario.startswith("top_fold"):
        preferred_label = "fold_root"
    return {
        "upper_half_width": upper_width,
        "lower_half_width": lower_width,
        "shoulder_drop": float(np.mean(seam[:, 2]) - np.mean(shoulder[:, 2])),
        "com_z": float(data.subtree_com[bag_id, 2]),
        "preferred_patch_rank": preferred_rank,
        "preferred_patch_label": preferred_label,
    }


def _scenario_metrics(model: mujoco.MjModel, data: mujoco.MjData, scenario: str) -> dict[str, float | str]:
    config = get_scenario(scenario)
    metrics = _basic_metrics(model, data, scenario)
    if scenario.startswith("top_fold"):
        metrics.update(
            {
                "fold_coverage_fraction": config.fold_coverage_fraction,
                "rim_exposed_fraction": 1.0 - config.fold_coverage_fraction,
                "fold_root_thickness_proxy": config.fold_root_thickness,
            }
        )
    return metrics


def _sag_metrics(settle_seconds: float) -> dict[str, float]:
    before_xml = write_scene_xml("post_separation_sag", post_release=False)
    after_xml = write_scene_xml("post_separation_sag", post_release=True)
    before_model = mujoco.MjModel.from_xml_path(str(before_xml))
    after_model = mujoco.MjModel.from_xml_path(str(after_xml))
    before_data = mujoco.MjData(before_model)
    after_data = mujoco.MjData(after_model)
    mujoco.mj_forward(before_model, before_data)
    mujoco.mj_forward(after_model, after_data)
    _step(before_model, before_data, min(settle_seconds, 0.3))
    _step(after_model, after_data, min(settle_seconds, 0.3))
    bottom_before = _body_pos(before_model, before_data, "bottom_sling")[2]
    bottom_after = _body_pos(after_model, after_data, "bottom_sling")[2]
    top_before = np.mean([_body_pos(before_model, before_data, f"seam_band_{i:02d}")[2] for i in range(SEGMENT_COUNT)])
    top_after = np.mean([_body_pos(after_model, after_data, f"seam_band_{i:02d}")[2] for i in range(SEGMENT_COUNT)])
    bottom_drop = float(max(0.0, bottom_before - bottom_after))
    top_drop = float(max(0.0, top_before - top_after))
    return {
        "bottom_drop_0p3s": bottom_drop,
        "top_drop_0p3s": top_drop,
        "sag_ratio": float(bottom_drop / max(top_drop, 1e-5)),
    }


def validate_one(
    scenario: str,
    *,
    settle_seconds: float,
    post_release: bool = False,
    cameras: tuple[str, ...] = ("front",),
) -> dict[str, object]:
    xml = write_scene_xml(scenario, post_release=post_release)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sim = _step(model, data, settle_seconds)
    nflex = int(getattr(model, "nflex", 0))
    metrics = _scenario_metrics(model, data, scenario)
    if scenario == "post_separation_sag":
        metrics.update(_sag_metrics(settle_seconds))

    image_paths: list[str] = []
    for camera in cameras:
        if scenario == "post_separation_sag":
            filename = f"post_separation_sag_{'after' if post_release else 'before'}.png"
        elif scenario == "baseline_filled":
            filename = f"baseline_{camera}.png"
        else:
            filename = f"{scenario}_{camera}.png"
        path = OUT_DIR / filename
        _render(model, data, path, camera)
        image_paths.append(str(path))

    row: dict[str, object] = {
        "scenario_name": scenario,
        "post_release": post_release,
        "nflex": nflex,
        "pass_fail": bool(nflex == 0 and not sim["nan_or_inf"] and sim["max_qvel"] < 250.0),
        "images": ";".join(image_paths),
        "xml": str(xml),
    }
    row.update(sim)
    row.update(metrics)
    return row


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_md(rows: list[dict[str, object]], path: Path) -> None:
    lines = ["# Shape-Coupled Sack Core Summary", ""]
    for row in rows:
        suffix = " after" if row["post_release"] else ""
        lines.append(f"## {row['scenario_name']}{suffix}")
        lines.append(f"- pass_fail: `{row['pass_fail']}`")
        lines.append(f"- images: `{row['images']}`")
        for key in (
            "upper_half_width",
            "lower_half_width",
            "shoulder_drop",
            "com_z",
            "preferred_patch_label",
            "fold_coverage_fraction",
            "rim_exposed_fraction",
            "bottom_drop_0p3s",
            "top_drop_0p3s",
            "sag_ratio",
        ):
            if key in row and row[key] != "":
                lines.append(f"- {key}: `{row[key]}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _make_montage() -> None:
    imageio = _imageio()
    specs = [
        ("baseline_front.png", "underfilled_front.png", "compare_baseline_underfilled.png"),
        ("baseline_front.png", "top_fold_simple_front.png", "compare_baseline_top_fold_simple.png"),
        ("baseline_front.png", "top_fold_severe_front.png", "compare_baseline_top_fold_severe.png"),
        ("post_separation_sag_before.png", "post_separation_sag_after.png", "compare_post_sag_before_after.png"),
    ]
    for left_name, right_name, out_name in specs:
        left = OUT_DIR / left_name
        right = OUT_DIR / right_name
        if not left.exists() or not right.exists():
            continue
        left_img = imageio.imread(left)
        right_img = imageio.imread(right)
        h = min(left_img.shape[0], right_img.shape[0])
        montage = np.concatenate([left_img[:h], right_img[:h]], axis=1)
        imageio.imwrite(OUT_DIR / out_name, montage)


def validate_all(settle_seconds: float) -> list[dict[str, object]]:
    rows = [
        validate_one("baseline_filled", settle_seconds=settle_seconds, cameras=("front",)),
        validate_one("underfilled", settle_seconds=settle_seconds, cameras=("front", "side")),
        validate_one("top_fold_simple", settle_seconds=settle_seconds, cameras=("front",)),
        validate_one("top_fold_severe", settle_seconds=settle_seconds, cameras=("front",)),
        validate_one("post_separation_sag", settle_seconds=settle_seconds, post_release=False, cameras=("front",)),
        validate_one("post_separation_sag", settle_seconds=settle_seconds, post_release=True, cameras=("front",)),
    ]
    _make_montage()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate/render shape-coupled sack core")
    parser.add_argument("--scenario", choices=[*SCENARIO_NAMES, "all"], default="all")
    parser.add_argument("--settle-seconds", type=float, default=0.8)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.scenario == "all":
        rows = validate_all(args.settle_seconds)
    elif args.scenario == "post_separation_sag":
        rows = [
            validate_one("post_separation_sag", settle_seconds=args.settle_seconds, post_release=False),
            validate_one("post_separation_sag", settle_seconds=args.settle_seconds, post_release=True),
        ]
    else:
        cams = ("front", "side") if args.scenario == "underfilled" else ("front",)
        rows = [validate_one(args.scenario, settle_seconds=args.settle_seconds, cameras=cams)]

    _write_csv(rows, OUT_DIR / "summary.csv")
    _write_md(rows, OUT_DIR / "summary.md")
    for row in rows:
        print(
            f"{row['scenario_name']}{'_after' if row['post_release'] else ''}: "
            f"pass_fail={row['pass_fail']} nflex={row['nflex']} max_qvel={float(row['max_qvel']):.3f} "
            f"images={row['images']}"
        )
    print(f"summary_csv={OUT_DIR / 'summary.csv'}")
    print(f"summary_md={OUT_DIR / 'summary.md'}")
    return 0 if all(bool(row["pass_fail"]) for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
