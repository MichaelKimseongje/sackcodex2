from __future__ import annotations

import argparse
import csv
from pathlib import Path

import mujoco
import numpy as np

from build_shared_sack import OUT_DIR, PANEL_GRID_X, PANEL_GRID_Y, write_scene_xml
from render_shared_sack import render_shared_sack


VISUAL_ONLY_PREFIXES = (
    "visual_skin",
    "sealed_top_cap_visual",
    "sealed_top_stitch_visual",
    "sealed_bottom_stitch_visual",
    "side_bulge_cue",
)

PHYSICS_BODY_PREFIXES = (
    "bag_frame",
    "seam_band_",
    "top_panel_",
    "bottom_panel_",
    "corner_fold_patch_",
    "bottom_sling",
    "fold_flap_",
    "payload_main",
    "payload_aux",
)


def _names(model: mujoco.MjModel, obj_type: mujoco.mjtObj, count: int) -> list[str]:
    names: list[str] = []
    for obj_id in range(count):
        name = mujoco.mj_id2name(model, obj_type, obj_id)
        if name:
            names.append(name)
    return names


def _settle(model: mujoco.MjModel, data: mujoco.MjData, seconds: float) -> dict[str, float | bool]:
    steps = max(1, int(seconds / model.opt.timestep))
    max_qvel = 0.0
    mean_qvel = 0.0
    nonfinite = False
    for step in range(steps):
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            nonfinite = True
            break
        qvel_norm = float(np.linalg.norm(data.qvel))
        max_qvel = max(max_qvel, qvel_norm)
        mean_qvel += (qvel_norm - mean_qvel) / float(step + 1)
    return {"max_qvel": max_qvel, "mean_qvel": mean_qvel, "nonfinite": nonfinite}


def _classify_names(names: list[str], prefixes: tuple[str, ...]) -> list[str]:
    return [name for name in names if any(name == prefix or name.startswith(prefix) for prefix in prefixes)]


def _geom_names_by_contact(model: mujoco.MjModel, *, visual_only: bool) -> list[str]:
    names: list[str] = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if not name:
            continue
        is_visual = int(model.geom_contype[geom_id]) == 0 and int(model.geom_conaffinity[geom_id]) == 0
        if is_visual == visual_only:
            names.append(name)
    return names


def validate_shared_sack(*, settle_seconds: float = 1.0, render: bool = True) -> dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    xml_path = write_scene_xml()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    settle = _settle(model, data, settle_seconds)

    body_names = _names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    joint_names = _names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    geom_names = _names(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom)
    visual_only_bodies = _classify_names(body_names, VISUAL_ONLY_PREFIXES)
    visual_only_geoms = _geom_names_by_contact(model, visual_only=True)
    visual_only_parts = visual_only_bodies + visual_only_geoms
    physics_bodies = _classify_names(body_names, PHYSICS_BODY_PREFIXES)
    physics_geoms = _geom_names_by_contact(model, visual_only=False)
    physics_parts = physics_bodies + physics_geoms
    movable_patch_joints = [
        name
        for name in joint_names
        if name.startswith(("top_panel_", "bottom_panel_", "seam_band_", "corner_fold_patch_", "fold_flap_"))
        and name.endswith("_hinge")
    ]
    expected_patch_count = 2 * PANEL_GRID_X * PANEL_GRID_Y + 8 + 4 + 2

    outputs = render_shared_sack(settle_seconds=settle_seconds) if render else {}
    row: dict[str, object] = {
        "xml": str(xml_path),
        "nbody": int(model.nbody),
        "njnt": int(model.njnt),
        "ngeom": int(model.ngeom),
        "nflex": int(getattr(model, "nflex", 0)),
        "movable_patch_count": len(movable_patch_joints),
        "expected_movable_patch_count": expected_patch_count,
        "has_single_bag_frame": body_names.count("bag_frame") == 1,
        "has_freejoint": "bag_frame_freejoint" in joint_names,
        "has_payload_main": "payload_main" in body_names,
        "has_payload_aux": "payload_aux" in body_names,
        "has_neighbors": "neighbor_left" in body_names and "neighbor_right" in body_names,
        "has_hidden_support": "hidden_support" in body_names,
        "has_visual_skin": "visual_skin" in body_names,
        "has_flat_pillow_surface": "top_surface_panels" in body_names and "bottom_surface_panels" in body_names,
        "has_fold_flaps": "fold_flap_1" in body_names and "fold_flap_2" in body_names,
        "has_sealed_top_cap_visual": "sealed_top_cap_visual" in body_names,
        "pass_fail": bool(
            body_names.count("bag_frame") == 1
            and "bag_frame_freejoint" in joint_names
            and len(movable_patch_joints) == expected_patch_count
            and int(getattr(model, "nflex", 0)) == 0
            and not settle["nonfinite"]
            and float(settle["max_qvel"]) < 80.0
            and "top_surface_panels" in body_names
            and "bottom_surface_panels" in body_names
            and "fold_flap_1" in body_names
            and "fold_flap_2" in body_names
        ),
        "body_names": ";".join(body_names),
        "joint_names": ";".join(joint_names),
        "visual_only_parts": ";".join(visual_only_parts),
        "physics_parts": ";".join(physics_parts),
        "front_image": outputs.get("front", ""),
        "side_image": outputs.get("side", ""),
        "top_angled_image": outputs.get("top_angled", ""),
    }
    row.update(settle)

    csv_path = OUT_DIR / "shared_sack_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    md_path = OUT_DIR / "shared_sack_summary.md"
    lines = [
        "# Shared Articulated Sack Skeleton Summary",
        "",
        f"- pass_fail: `{row['pass_fail']}`",
        f"- xml: `{row['xml']}`",
        f"- nflex: `{row['nflex']}`",
        f"- movable_patch_count: `{row['movable_patch_count']}`",
        f"- body_count: `{row['nbody']}`",
        f"- joint_count: `{row['njnt']}`",
        f"- front_image: `{row['front_image']}`",
        f"- side_image: `{row['side_image']}`",
        f"- top_angled_image: `{row['top_angled_image']}`",
        "",
        "## Body Names",
        "",
        row["body_names"].replace(";", "\n"),
        "",
        "## Joint Names",
        "",
        row["joint_names"].replace(";", "\n"),
        "",
        "## Visual-Only Parts",
        "",
        row["visual_only_parts"].replace(";", "\n"),
        "",
        "## Physics Parts",
        "",
        row["physics_parts"].replace(";", "\n"),
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    row["summary_csv"] = str(csv_path)
    row["summary_md"] = str(md_path)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the shared articulated sack skeleton")
    parser.add_argument("--settle-seconds", type=float, default=1.0)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    row = validate_shared_sack(settle_seconds=args.settle_seconds, render=not args.no_render)
    for key in (
        "pass_fail",
        "xml",
        "nflex",
        "movable_patch_count",
        "body_names",
        "joint_names",
        "visual_only_parts",
        "physics_parts",
        "front_image",
        "side_image",
        "top_angled_image",
        "summary_csv",
        "summary_md",
    ):
        print(f"{key}={row[key]}")
    return 0 if bool(row["pass_fail"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
