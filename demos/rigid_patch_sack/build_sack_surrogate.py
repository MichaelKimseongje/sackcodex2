from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from scenario_builder import ScenarioConfig, get_scenario


ROOT_DIR = Path(__file__).resolve().parent
GENERATED_DIR = ROOT_DIR / "generated"
OUT_DIR = ROOT_DIR / "out"

SEGMENT_COUNT = 8
SEAM_RX = 0.105
SEAM_RY = 0.056
SEAM_Z = 0.118
SHOULDER_LEN = 0.082
BELLY_LEN = 0.094
BOTTOM_Z = -0.122

TIMESTEP = 0.001
GRAVITY = "0 0 -9.81"

SCENE_XML_PATH = GENERATED_DIR / "scene_rigid_patch_sack.xml"


def _fmt(values: tuple[float, ...] | list[float] | np.ndarray) -> str:
    return " ".join(f"{float(value):.6f}" for value in values)


def _add_geom(body: ET.Element, **attrib: str) -> ET.Element:
    defaults = {
        "friction": "1.25 0.08 0.006",
        "condim": "4",
        "solref": "0.026 1",
        "solimp": "0.82 0.96 0.001",
    }
    defaults.update(attrib)
    return ET.SubElement(body, "geom", defaults)


def _segment_angle(index: int) -> float:
    return 2.0 * math.pi * index / SEGMENT_COUNT


def _seam_point(index: int, config: ScenarioConfig) -> np.ndarray:
    angle = _segment_angle(index)
    return np.array(
        [
            SEAM_RX * config.seam_rx_scale * math.cos(angle),
            SEAM_RY * config.seam_ry_scale * math.sin(angle),
            SEAM_Z,
        ],
        dtype=np.float64,
    )


def _segment_pose(index: int, config: ScenarioConfig) -> tuple[np.ndarray, float, float]:
    start = _seam_point(index, config)
    end = _seam_point((index + 1) % SEGMENT_COUNT, config)
    mid = 0.5 * (start + end)
    angle = math.atan2(mid[1], mid[0])
    tangent_yaw_deg = math.degrees(angle + math.pi / 2.0)
    segment_width = float(np.linalg.norm(end - start))
    return mid, tangent_yaw_deg, segment_width


def _add_floor(worldbody: ET.Element) -> None:
    _add_geom(
        worldbody,
        name="floor",
        type="plane",
        pos="0 0 0",
        size="2.0 2.0 0.05",
        rgba="0.90 0.90 0.88 1",
        friction="1.5 0.05 0.005",
    )


def _add_camera_light(worldbody: ET.Element) -> None:
    ET.SubElement(worldbody, "light", {"name": "key_light", "pos": "0.4 -0.7 1.5", "dir": "-0.3 0.4 -1"})
    ET.SubElement(worldbody, "camera", {"name": "overview", "pos": "0.86 -0.86 0.72", "xyaxes": "0.72 0.69 0 -0.34 0.35 0.87"})
    ET.SubElement(worldbody, "camera", {"name": "front", "pos": "0.78 0 0.36", "xyaxes": "0 1 0 -0.25 0 0.97"})
    ET.SubElement(worldbody, "camera", {"name": "side", "pos": "0 -0.78 0.36", "xyaxes": "1 0 0 0 0.25 0.97"})


def _add_eval_gripper(worldbody: ET.Element) -> None:
    """run_eval.py에서 쓰는 간단한 mocap 2F gripper다."""
    for name, sign, rgba in (
        ("gripper_left_mocap", -1.0, "0.15 0.35 0.95 0.85"),
        ("gripper_right_mocap", 1.0, "0.95 0.25 0.12 0.85"),
    ):
        body = ET.SubElement(
            worldbody,
            "body",
            {"name": name, "mocap": "true", "pos": f"0 {sign * 0.080:.6f} 0.230"},
        )
        _add_geom(
            body,
            name=f"{name}_pad",
            type="box",
            size="0.020 0.008 0.055",
            rgba=rgba,
            friction="2.2 0.10 0.008",
            solref="0.030 1",
            solimp="0.78 0.94 0.001",
        )
        ET.SubElement(body, "site", {"name": f"{name}_site", "pos": "0 0 0", "size": "0.006", "rgba": rgba})
    center = ET.SubElement(worldbody, "body", {"name": "gripper_center_mocap", "mocap": "true", "pos": "0 0 0.230"})
    ET.SubElement(center, "site", {"name": "gripper_center_site", "pos": "0 0 0", "size": "0.006", "rgba": "1 1 0 0.8"})


def _add_seam_band(bag_frame: ET.Element, index: int, config: ScenarioConfig) -> None:
    """상단 봉합선처럼 보이는 실제 grasp candidate patch."""
    mid, yaw_deg, width = _segment_pose(index, config)
    body = ET.SubElement(
        bag_frame,
        "body",
        {"name": f"seam_band_{index:02d}", "pos": _fmt(mid), "euler": f"0 0 {yaw_deg:.6f}"},
    )
    _add_geom(
        body,
        name=f"seam_band_{index:02d}_geom",
        type="box",
        size=f"{0.45 * width:.6f} 0.006000 0.005000",
        mass="0.016",
        rgba="0.36 0.20 0.08 0.88",
    )
    ET.SubElement(body, "site", {"name": f"grasp_seam_{index:02d}", "pos": "0 0 0", "size": "0.005", "rgba": "1 0.55 0.12 0.75"})
    # 기존 평가/GUI와의 호환용 alias site다.
    ET.SubElement(body, "site", {"name": f"grasp_rim_{index:02d}", "pos": "0 0 0", "size": "0.003", "rgba": "1 0.55 0.12 0.18"})


def _add_shoulder_panel(bag_frame: ET.Element, index: int, config: ScenarioConfig) -> None:
    """upper_skirt 대신 쓰는 넓은 rigid cloth panel."""
    mid, yaw_deg, width = _segment_pose(index, config)
    body = ET.SubElement(
        bag_frame,
        "body",
        {"name": f"shoulder_panel_{index:02d}", "pos": _fmt(mid + np.array([0.0, 0.0, -0.012])), "euler": f"0 0 {yaw_deg:.6f}"},
    )
    ET.SubElement(
        body,
        "joint",
        {
            "name": f"shoulder_panel_{index:02d}_hinge",
            "type": "hinge",
            "axis": "1 0 0",
            "limited": "true",
            "range": "-38 48",
            "damping": "9.0",
            "stiffness": "2.6",
        },
    )
    _add_geom(
        body,
        name=f"shoulder_panel_{index:02d}_geom",
        type="box",
        pos=f"0 {0.5 * config.shoulder_inward:.6f} {-0.5 * SHOULDER_LEN:.6f}",
        size=f"{0.50 * width:.6f} 0.010000 {0.5 * SHOULDER_LEN:.6f}",
        mass="0.028",
        rgba="0.72 0.52 0.27 0.86",
    )
    ET.SubElement(
        body,
        "site",
        {
            "name": f"grasp_shoulder_{index:02d}",
            "pos": f"0 {0.35 * config.shoulder_inward:.6f} {-0.35 * SHOULDER_LEN:.6f}",
            "size": "0.006",
            "rgba": "0.10 0.75 0.25 0.80",
        },
    )


def _add_belly_panel(bag_frame: ET.Element, index: int, config: ScenarioConfig) -> None:
    """lower_skirt 대신 쓰는 하부 bulge/sag panel."""
    mid, yaw_deg, width = _segment_pose(index, config)
    lower_anchor = mid + np.array([0.0, 0.0, -0.083 - 0.25 * config.lower_extra_drop])
    body = ET.SubElement(
        bag_frame,
        "body",
        {"name": f"belly_panel_{index:02d}", "pos": _fmt(lower_anchor), "euler": f"0 0 {yaw_deg:.6f}"},
    )
    ET.SubElement(
        body,
        "joint",
        {
            "name": f"belly_panel_{index:02d}_hinge",
            "type": "hinge",
            "axis": "1 0 0",
            "limited": "true",
            "range": "-46 58",
            "damping": "10.5",
            "stiffness": "1.8",
        },
    )
    _add_geom(
        body,
        name=f"belly_panel_{index:02d}_geom",
        type="box",
        pos=f"0 {-0.5 * config.belly_outward:.6f} {-0.5 * (BELLY_LEN + config.lower_extra_drop):.6f}",
        size=f"{0.53 * width:.6f} 0.012000 {0.5 * (BELLY_LEN + config.lower_extra_drop):.6f}",
        mass="0.034",
        rgba="0.64 0.43 0.21 0.88",
    )


def _add_bottom_sling(bag_frame: ET.Element, config: ScenarioConfig) -> None:
    """골조처럼 보이지 않도록 낮고 넓은 하부 sling으로 내부 하중을 받친다."""
    sling_z = BOTTOM_Z - 0.45 * config.lower_extra_drop
    body = ET.SubElement(bag_frame, "body", {"name": "bottom_sling", "pos": f"0 0 {sling_z:.6f}"})
    _add_geom(
        body,
        name="bottom_sling_pad",
        type="box",
        size="0.082 0.048 0.008",
        mass="0.065",
        rgba="0.58 0.35 0.15 0.78",
    )
    _add_geom(
        body,
        name="bottom_sling_cross",
        type="box",
        euler="0 0 34",
        size="0.072 0.010 0.010",
        mass="0.022",
        rgba="0.46 0.26 0.10 0.74",
    )
    # 기존 코드와의 호환용 이름이다.
    ET.SubElement(body, "site", {"name": "bottom_cradle_site", "pos": "0 0 0", "size": "0.004", "rgba": "0.8 0.4 0.1 0.7"})


def _fold_half_length(config: ScenarioConfig, flap_index: int) -> float:
    circumference = 2.0 * math.pi * math.sqrt((SEAM_RX * SEAM_RX + SEAM_RY * SEAM_RY) / 2.0)
    if config.fold2_enabled:
        fraction = 0.36 if flap_index == 1 else 0.30
    else:
        fraction = max(config.fold_coverage_fraction, 0.30)
    return min(0.108, 0.5 * circumference * fraction)


def _add_fold_flaps(bag_frame: ET.Element, config: ScenarioConfig) -> None:
    for flap_index, enabled, angle_deg, x_shift, y_shift in (
        (1, config.fold1_enabled, config.fold1_angle_deg, config.fold1_x_shift, 0.030),
        (2, config.fold2_enabled, config.fold2_angle_deg, config.fold2_x_shift, 0.018),
    ):
        body = ET.SubElement(
            bag_frame,
            "body",
            {
                "name": f"fold_root_flap_{flap_index}",
                "pos": f"{x_shift:.6f} {y_shift:.6f} {SEAM_Z + 0.012 - 0.010 * (flap_index - 1):.6f}",
                "euler": f"{angle_deg:.6f} 0 0",
            },
        )
        ET.SubElement(
            body,
            "joint",
            {
                "name": f"fold_root_flap_{flap_index}_hinge",
                "type": "hinge",
                "axis": "1 0 0",
                "limited": "true",
                "range": "-125 22",
                "damping": "7.5",
                "stiffness": "2.8",
            },
        )
        half_len = _fold_half_length(config, flap_index)
        _add_geom(
            body,
            name=f"fold_root_flap_{flap_index}_geom",
            type="box",
            pos="0 0.026 -0.012",
            size=f"{half_len:.6f} 0.012000 0.026000",
            mass="0.024" if enabled else "0.001",
            rgba="0.70 0.50 0.27 0.88" if enabled else "0.70 0.50 0.27 0.05",
            contype="1" if enabled else "0",
            conaffinity="1" if enabled else "0",
        )
        if enabled:
            _add_geom(
                body,
                name=f"fold_root_flap_{flap_index}_roll",
                type="capsule",
                fromto=f"{-0.85 * half_len:.6f} 0.004 0.017 {0.85 * half_len:.6f} 0.004 0.017",
                size=f"{config.fold_root_thickness:.6f}",
                mass="0.006",
                rgba="0.42 0.24 0.09 0.92",
                contype="0",
                conaffinity="0",
            )
            for wrinkle_idx, x0 in enumerate((-0.055, 0.0, 0.055)):
                _add_geom(
                    body,
                    name=f"fold_root_flap_{flap_index}_wrinkle_{wrinkle_idx}",
                    type="capsule",
                    fromto=f"{x0 - 0.020:.6f} 0.020 0.012 {x0 + 0.016:.6f} 0.050 -0.004",
                    size="0.0024",
                    mass="0.001",
                    rgba="0.26 0.14 0.05 0.62",
                    contype="0",
                    conaffinity="0",
                )
        ET.SubElement(body, "site", {"name": f"grasp_fold_{flap_index}", "pos": "0 0.026 -0.012", "size": "0.005", "rgba": "0.1 0.2 1 0.35"})
        # 기존 코드 호환용 alias site다.
        ET.SubElement(body, "site", {"name": f"grasp_fold_root_{flap_index}", "pos": "0 0.026 -0.012", "size": "0.004", "rgba": "0.1 0.2 1 0.15"})


def _add_payloads(bag_frame: ET.Element, config: ScenarioConfig) -> None:
    for body_name, pos, size, mass, enabled, rgba in (
        ("payload_main", config.payload_main_pos, config.payload_main_size, config.payload_main_mass, True, "0.42 0.22 0.10 0.72"),
        ("payload_aux", config.payload_aux_pos, config.payload_aux_size, config.payload_aux_mass, config.payload_aux_enabled, "0.60 0.28 0.12 0.68"),
    ):
        body = ET.SubElement(bag_frame, "body", {"name": body_name, "pos": _fmt(pos)})
        for axis_name, axis, limit in (("x", "1 0 0", "-0.030 0.030"), ("y", "0 1 0", "-0.026 0.026"), ("z", "0 0 1", "-0.020 0.020")):
            ET.SubElement(
                body,
                "joint",
                {
                    "name": f"{body_name}_{axis_name}",
                    "type": "slide",
                    "axis": axis,
                    "limited": "true",
                    "range": limit,
                    "damping": "20",
                    "stiffness": "3.0",
                },
            )
        _add_geom(
            body,
            name=f"{body_name}_geom",
            type="ellipsoid",
            size=_fmt(size),
            mass=f"{mass:.4f}",
            rgba=rgba if enabled else "0.60 0.28 0.12 0.05",
            contype="1" if enabled else "0",
            conaffinity="1" if enabled else "0",
        )
    if config.side_bulge_enabled:
        bulge = ET.SubElement(bag_frame, "body", {"name": "side_bulge", "pos": _fmt(config.side_bulge_pos)})
        _add_geom(
            bulge,
            name="side_bulge_geom",
            type="ellipsoid",
            size=_fmt(config.side_bulge_size),
            mass="0.045",
            rgba="0.76 0.46 0.20 0.62",
            contype="1",
            conaffinity="1",
        )


def _add_visual_skin(bag_frame: ET.Element, config: ScenarioConfig) -> None:
    """visual only sealed hull. 물리는 broad panels/payload가 담당한다."""
    ux, uy, uz = config.visual_upper_scale
    lx, ly, lz = config.visual_lower_scale
    sx, sy, sz = config.visual_center_shift
    upper = ET.SubElement(bag_frame, "body", {"name": "visual_upper_hull", "pos": f"{sx:.6f} {sy:.6f} {0.042 + sz:.6f}"})
    _add_geom(
        upper,
        name="visual_upper_hull_geom",
        type="ellipsoid",
        size=f"{SEAM_RX * ux:.6f} {SEAM_RY * uy:.6f} {0.080 * uz:.6f}",
        rgba="0.78 0.62 0.38 0.30",
        contype="0",
        conaffinity="0",
        mass="0.001",
    )
    lower = ET.SubElement(bag_frame, "body", {"name": "visual_lower_hull", "pos": f"{sx:.6f} {sy:.6f} {-0.072 + sz - 0.30 * config.lower_extra_drop:.6f}"})
    _add_geom(
        lower,
        name="visual_lower_hull_geom",
        type="ellipsoid",
        size=f"{SEAM_RX * lx:.6f} {SEAM_RY * ly:.6f} {0.092 * lz + 0.35 * config.lower_extra_drop:.6f}",
        rgba="0.72 0.50 0.25 0.40",
        contype="0",
        conaffinity="0",
        mass="0.001",
    )
    lower_fill = ET.SubElement(bag_frame, "body", {"name": "visual_lower_fill", "pos": _fmt(config.payload_main_pos)})
    _add_geom(
        lower_fill,
        name="visual_lower_fill_geom",
        type="ellipsoid",
        size=f"{config.payload_main_size[0] * 1.15:.6f} {config.payload_main_size[1] * 1.12:.6f} {config.payload_main_size[2] * 1.05:.6f}",
        rgba="0.42 0.22 0.10 0.34",
        contype="0",
        conaffinity="0",
        mass="0.001",
    )


def _add_top_cap_visual(bag_frame: ET.Element, config: ScenarioConfig) -> None:
    """큰 tabletop plate가 아니라 작은 sealed cap과 물리 seam band만 남긴다."""
    cap = ET.SubElement(bag_frame, "body", {"name": "sealed_top_cap_visual", "pos": f"0 0 {SEAM_Z + 0.009:.6f}"})
    _add_geom(
        cap,
        name="sealed_top_cap_visual_geom",
        type="ellipsoid",
        size=f"{SEAM_RX * 0.58:.6f} {SEAM_RY * config.seam_ry_scale * 0.50:.6f} 0.006000",
        rgba="0.80 0.62 0.36 0.58",
        contype="0",
        conaffinity="0",
        mass="0.001",
    )
    seam_line = ET.SubElement(bag_frame, "body", {"name": "sealed_top_stitch_visual", "pos": f"0 0 {SEAM_Z + 0.017:.6f}"})
    _add_geom(
        seam_line,
        name="sealed_top_stitch_visual_geom",
        type="capsule",
        fromto="-0.064 0 0 0.064 0 0",
        size="0.0035",
        rgba="0.24 0.13 0.05 0.76",
        contype="0",
        conaffinity="0",
        mass="0.001",
    )


def _add_neighbors(worldbody: ET.Element, config: ScenarioConfig) -> None:
    for side, y_sign in (("left", 1.0), ("right", -1.0)):
        body = ET.SubElement(
            worldbody,
            "body",
            {
                "name": f"neighbor_{side}",
                "pos": f"0 {y_sign * config.neighbor_gap_y:.6f} 0.145",
                "euler": f"0 0 {y_sign * 8.0:.3f}",
            },
        )
        _add_geom(
            body,
            name=f"neighbor_{side}_geom",
            type="ellipsoid",
            size="0.128 0.050 0.060",
            mass="0.65",
            rgba="0.50 0.42 0.30 0.72" if config.neighbor_enabled else "0.50 0.42 0.30 0.05",
            contype="1" if config.neighbor_enabled else "0",
            conaffinity="1" if config.neighbor_enabled else "0",
            friction="1.8 0.12 0.01",
        )


def _add_hidden_support(worldbody: ET.Element, config: ScenarioConfig) -> None:
    if not config.support_enabled:
        return
    support = ET.SubElement(worldbody, "body", {"name": "temporary_bottom_support", "pos": "0 0 0.055"})
    _add_geom(
        support,
        name="temporary_bottom_support_geom",
        type="box",
        size="0.100 0.060 0.012",
        rgba="0.10 0.65 0.95 0.24",
        friction="0.9 0.03 0.003",
    )


def _add_bag(worldbody: ET.Element, config: ScenarioConfig) -> None:
    bag_frame = ET.SubElement(
        worldbody,
        "body",
        {"name": "bag_frame", "pos": "0 0 0.188", "euler": f"0 {config.body_tilt_deg:.6f} 0"},
    )
    ET.SubElement(bag_frame, "freejoint", {"name": "bag_frame_freejoint"})
    ET.SubElement(bag_frame, "site", {"name": "bag_frame_origin", "pos": "0 0 0", "size": "0.007", "rgba": "1 0 0 0.7"})
    for index in range(SEGMENT_COUNT):
        _add_seam_band(bag_frame, index, config)
    for index in range(SEGMENT_COUNT):
        _add_shoulder_panel(bag_frame, index, config)
    for index in range(SEGMENT_COUNT):
        _add_belly_panel(bag_frame, index, config)
    _add_bottom_sling(bag_frame, config)
    _add_fold_flaps(bag_frame, config)
    _add_payloads(bag_frame, config)
    _add_visual_skin(bag_frame, config)
    _add_top_cap_visual(bag_frame, config)


def build_scene_tree(config: ScenarioConfig, *, include_eval_gripper: bool = True) -> ET.Element:
    root = ET.Element("mujoco", {"model": f"sealed_articulated_sack_v2_{config.name}"})
    ET.SubElement(root, "compiler", {"angle": "degree", "autolimits": "true"})
    ET.SubElement(
        root,
        "option",
        {
            "timestep": f"{TIMESTEP:.6f}",
            "gravity": GRAVITY,
            "integrator": "implicitfast",
            "solver": "Newton",
            "iterations": "100",
            "ls_iterations": "24",
            "cone": "elliptic",
            "jacobian": "sparse",
            "impratio": "4",
        },
    )
    ET.SubElement(root, "size", {"nconmax": "900", "njmax": "1800"})
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"offwidth": "1280", "offheight": "820"})
    worldbody = ET.SubElement(root, "worldbody")
    _add_floor(worldbody)
    _add_camera_light(worldbody)
    if include_eval_gripper:
        _add_eval_gripper(worldbody)
    _add_neighbors(worldbody, config)
    _add_hidden_support(worldbody, config)
    _add_bag(worldbody, config)
    ET.indent(root, space="  ")
    return root


def write_scene_xml(
    scenario_name: str = "underfilled",
    output_path: Path | None = None,
    *,
    post_release: bool = False,
    include_eval_gripper: bool = True,
) -> Path:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    config = get_scenario(scenario_name, post_release=post_release)
    if output_path is None:
        suffix = "_after" if post_release else ""
        output_path = GENERATED_DIR / f"scene_rigid_patch_sack_{scenario_name}{suffix}.xml"
    root = build_scene_tree(config, include_eval_gripper=include_eval_gripper)
    output_path.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    return output_path


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build sealed articulated sack surrogate v2 MJCF scene")
    parser.add_argument("--scenario", default="underfilled")
    parser.add_argument("--post-release", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-eval-gripper", action="store_true")
    args = parser.parse_args()
    path = write_scene_xml(args.scenario, args.output, post_release=args.post_release, include_eval_gripper=not args.no_eval_gripper)
    print(f"scene_xml={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
