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
BASE_RX = 0.108
BASE_RY = 0.058
SEAM_Z = 0.126
SHOULDER_LEN = 0.090
BELLY_LEN = 0.104
BOTTOM_Z = -0.132
TIMESTEP = 0.001


def _fmt(values: tuple[float, ...] | list[float] | np.ndarray) -> str:
    return " ".join(f"{float(v):.6f}" for v in values)


def _geom(body: ET.Element, **attrib: str) -> ET.Element:
    defaults = {
        "friction": "1.35 0.08 0.006",
        "condim": "4",
        "solref": "0.026 1",
        "solimp": "0.82 0.96 0.001",
    }
    defaults.update(attrib)
    return ET.SubElement(body, "geom", defaults)


def _angle(index: int) -> float:
    return 2.0 * math.pi * index / SEGMENT_COUNT


def _ellipse_point(index: int, rx: float, ry: float, z: float) -> np.ndarray:
    theta = _angle(index)
    return np.array([rx * math.cos(theta), ry * math.sin(theta), z], dtype=np.float64)


def _segment_pose(index: int, rx: float, ry: float, z: float) -> tuple[np.ndarray, float, float]:
    start = _ellipse_point(index, rx, ry, z)
    end = _ellipse_point((index + 1) % SEGMENT_COUNT, rx, ry, z)
    mid = 0.5 * (start + end)
    radial_angle = math.atan2(mid[1], mid[0])
    yaw_deg = math.degrees(radial_angle + math.pi / 2.0)
    width = float(np.linalg.norm(end - start))
    return mid, yaw_deg, width


def _add_floor(worldbody: ET.Element) -> None:
    # floor는 낙하/마찰/전도 확인에 필요한 충돌면입니다. 자루 관찰을 방해하지 않도록 기본 시각화는 약하게 둡니다.
    _geom(
        worldbody,
        name="floor",
        type="plane",
        pos="0 0 0",
        size="2.0 2.0 0.05",
        rgba="0.91 0.90 0.86 0.18",
        friction="1.5 0.05 0.005",
    )


def _add_cameras(worldbody: ET.Element) -> None:
    ET.SubElement(worldbody, "light", {"name": "key_light", "pos": "0.4 -0.7 1.5", "dir": "-0.3 0.4 -1"})
    ET.SubElement(worldbody, "camera", {"name": "front", "pos": "0.80 0 0.38", "xyaxes": "0 1 0 -0.24 0 0.97"})
    ET.SubElement(worldbody, "camera", {"name": "side", "pos": "0 -0.78 0.38", "xyaxes": "1 0 0 0 0.24 0.97"})
    ET.SubElement(worldbody, "camera", {"name": "overview", "pos": "0.85 -0.85 0.72", "xyaxes": "0.72 0.69 0 -0.34 0.35 0.87"})


def _add_eval_gripper(worldbody: ET.Element) -> None:
    for name, sign, rgba in (
        ("gripper_left_mocap", -1.0, "0.15 0.35 0.95 0.85"),
        ("gripper_right_mocap", 1.0, "0.95 0.25 0.12 0.85"),
    ):
        body = ET.SubElement(worldbody, "body", {"name": name, "mocap": "true", "pos": f"0 {sign * 0.080:.6f} 0.230"})
        _geom(
            body,
            name=f"{name}_pad",
            type="box",
            size="0.020 0.008 0.055",
            rgba=rgba,
            friction="2.3 0.12 0.010",
            solref="0.030 1",
            solimp="0.78 0.94 0.001",
        )
        ET.SubElement(body, "site", {"name": f"{name}_site", "pos": "0 0 0", "size": "0.006", "rgba": rgba})
    center = ET.SubElement(worldbody, "body", {"name": "gripper_center_mocap", "mocap": "true", "pos": "0 0 0.230"})
    ET.SubElement(center, "site", {"name": "gripper_center_site", "pos": "0 0 0", "size": "0.006", "rgba": "1 1 0 0.8"})


def _add_eval_scoop(worldbody: ET.Element) -> None:
    # 형상-지지 반응 데모용 단순 스쿠프입니다. 물리용 skin이 아니라 rigid support plate입니다.
    body = ET.SubElement(worldbody, "body", {"name": "scoop_mocap", "mocap": "true", "pos": "-0.20 0 0.080"})
    _geom(
        body,
        name="scoop_plate",
        type="box",
        size="0.105 0.052 0.006",
        rgba="0.12 0.42 0.86 0.82",
        friction="1.7 0.08 0.006",
        solref="0.035 1",
        solimp="0.76 0.94 0.001",
    )
    _geom(
        body,
        name="scoop_back_lip",
        type="box",
        pos="-0.095 0 0.019",
        size="0.010 0.052 0.020",
        rgba="0.08 0.25 0.58 0.82",
        friction="1.7 0.08 0.006",
        solref="0.035 1",
        solimp="0.76 0.94 0.001",
    )
    ET.SubElement(body, "site", {"name": "scoop_support_site", "pos": "0.035 0 0.010", "size": "0.006", "rgba": "0 1 1 0.85"})


def _add_seam_band(bag: ET.Element, index: int, config: ScenarioConfig) -> None:
    rx = BASE_RX * config.top_radius_scale
    ry = BASE_RY * config.top_radius_scale
    mid, yaw, width = _segment_pose(index, rx, ry, SEAM_Z)
    body = ET.SubElement(bag, "body", {"name": f"seam_band_{index:02d}", "pos": _fmt(mid), "euler": f"0 0 {yaw:.6f}"})
    _geom(
        body,
        name=f"seam_band_{index:02d}_geom",
        type="box",
        size=f"{0.47 * width:.6f} 0.006000 0.005000",
        mass="0.016",
        rgba="0.35 0.20 0.08 0.90",
    )
    ET.SubElement(body, "site", {"name": f"grasp_seam_{index:02d}", "pos": "0 0 0", "size": "0.005", "rgba": "1 0.55 0.12 0.90"})


def _add_shoulder_panel(bag: ET.Element, index: int, config: ScenarioConfig) -> None:
    rx = BASE_RX * config.top_radius_scale
    ry = BASE_RY * config.top_radius_scale
    mid, yaw, width = _segment_pose(index, rx, ry, SEAM_Z - 0.010)
    body = ET.SubElement(bag, "body", {"name": f"shoulder_panel_{index:02d}", "pos": _fmt(mid), "euler": f"0 0 {yaw:.6f}"})
    ET.SubElement(
        body,
        "joint",
        {
            "name": f"shoulder_panel_{index:02d}_hinge",
            "type": "hinge",
            "axis": "1 0 0",
            "limited": "true",
            "range": "-44 52",
            "damping": f"{config.shoulder_damping:.3f}",
            "stiffness": f"{config.shoulder_stiffness:.3f}",
        },
    )
    _geom(
        body,
        name=f"shoulder_panel_{index:02d}_geom",
        type="box",
        pos=f"0 {config.shoulder_inward:.6f} {-0.5 * SHOULDER_LEN:.6f}",
        size=f"{0.56 * width:.6f} 0.010000 {0.5 * SHOULDER_LEN:.6f}",
        mass="0.030",
        rgba="0.84 0.58 0.27 0.10",
    )
    cue_y = max(config.shoulder_inward + 0.018, 0.030)
    _geom(
        body,
        name=f"shoulder_panel_{index:02d}_shape_cue",
        type="capsule",
        fromto=f"{-0.45 * width:.6f} {cue_y:.6f} {-0.020:.6f} {0.45 * width:.6f} {cue_y:.6f} {-0.070:.6f}",
        size="0.0035",
        mass="0.001",
        rgba="0.18 0.08 0.02 0.95",
        contype="0",
        conaffinity="0",
    )
    for wrinkle_idx, x_offset in enumerate((-0.28 * width, 0.0, 0.28 * width)):
        _geom(
            body,
            name=f"shoulder_panel_{index:02d}_wrinkle_{wrinkle_idx}",
            type="capsule",
            fromto=f"{x_offset - 0.006:.6f} {cue_y + 0.002:.6f} -0.018 {x_offset + 0.006:.6f} {cue_y - 0.002:.6f} -0.080",
            size="0.0022",
            mass="0.0002",
            rgba="0.24 0.11 0.03 0.82",
            contype="0",
            conaffinity="0",
        )
    ET.SubElement(
        body,
        "site",
        {
            "name": f"grasp_shoulder_{index:02d}",
            "pos": f"0 {0.7 * config.shoulder_inward:.6f} {-0.35 * SHOULDER_LEN:.6f}",
            "size": "0.006",
            "rgba": "0.10 0.75 0.25 0.90",
        },
    )


def _add_belly_panel(bag: ET.Element, index: int, config: ScenarioConfig) -> None:
    rx = BASE_RX * config.lower_radius_scale
    ry = BASE_RY * config.lower_radius_scale
    anchor_z = SEAM_Z - SHOULDER_LEN * 0.88
    mid, yaw, width = _segment_pose(index, rx, ry, anchor_z)
    body = ET.SubElement(bag, "body", {"name": f"belly_panel_{index:02d}", "pos": _fmt(mid), "euler": f"0 0 {yaw:.6f}"})
    ET.SubElement(
        body,
        "joint",
        {
            "name": f"belly_panel_{index:02d}_hinge",
            "type": "hinge",
            "axis": "1 0 0",
            "limited": "true",
            "range": "-54 64",
            "damping": f"{config.belly_damping:.3f}",
            "stiffness": f"{config.belly_stiffness:.3f}",
        },
    )
    height = BELLY_LEN + config.lower_extra_drop
    _geom(
        body,
        name=f"belly_panel_{index:02d}_geom",
        type="box",
        pos=f"0 {-config.belly_outward:.6f} {-0.5 * height:.6f}",
        size=f"{0.58 * width:.6f} 0.012000 {0.5 * height:.6f}",
        mass="0.038",
        rgba="0.70 0.42 0.18 0.10",
    )
    cue_y = -max(config.belly_outward + 0.018, 0.028)
    _geom(
        body,
        name=f"belly_panel_{index:02d}_shape_cue",
        type="capsule",
        fromto=f"{-0.46 * width:.6f} {cue_y:.6f} {-0.030:.6f} {0.46 * width:.6f} {cue_y:.6f} {-0.085:.6f}",
        size="0.0038",
        mass="0.001",
        rgba="0.16 0.07 0.02 0.96",
        contype="0",
        conaffinity="0",
    )
    for wrinkle_idx, x_offset in enumerate((-0.30 * width, 0.0, 0.30 * width)):
        _geom(
            body,
            name=f"belly_panel_{index:02d}_wrinkle_{wrinkle_idx}",
            type="capsule",
            fromto=f"{x_offset - 0.008:.6f} {cue_y - 0.002:.6f} -0.024 {x_offset + 0.008:.6f} {cue_y + 0.002:.6f} {-0.78 * height:.6f}",
            size="0.0024",
            mass="0.0002",
            rgba="0.22 0.09 0.02 0.84",
            contype="0",
            conaffinity="0",
        )


def _add_bottom_sling(bag: ET.Element, config: ScenarioConfig) -> None:
    body = ET.SubElement(bag, "body", {"name": "bottom_sling", "pos": f"0 0 {BOTTOM_Z - 0.45 * config.lower_extra_drop:.6f}"})
    ET.SubElement(
        body,
        "joint",
        {
            "name": "bottom_sling_sag",
            "type": "slide",
            "axis": "0 0 1",
            "limited": "true",
            "range": f"{config.bottom_slide_range[0]:.6f} {config.bottom_slide_range[1]:.6f}",
            "damping": f"{config.bottom_slide_damping:.3f}",
            "stiffness": f"{config.bottom_slide_stiffness:.3f}",
        },
    )
    _geom(
        body,
        name="bottom_sling_pad",
        type="box",
        size="0.086 0.050 0.008",
        mass="0.070",
        rgba="0.55 0.33 0.14 0.10",
    )
    ET.SubElement(body, "site", {"name": "bottom_sling_site", "pos": "0 0 0", "size": "0.005", "rgba": "0.8 0.35 0.08 0.8"})


def _fold_half_length(config: ScenarioConfig, flap_index: int) -> float:
    circumference = 2.0 * math.pi * math.sqrt((BASE_RX * BASE_RX + BASE_RY * BASE_RY) / 2.0)
    if config.fold2_enabled:
        fraction = 0.36 if flap_index == 1 else 0.32
    else:
        fraction = max(config.fold_coverage_fraction, 0.30)
    return min(0.115, 0.5 * circumference * fraction)


def _add_fold_flaps(bag: ET.Element, config: ScenarioConfig) -> None:
    for flap_index, enabled, angle, x_shift, y_shift in (
        (1, config.fold1_enabled, config.fold1_angle_deg, -0.018, 0.030),
        (2, config.fold2_enabled, config.fold2_angle_deg, 0.034, 0.018),
    ):
        body = ET.SubElement(
            bag,
            "body",
            {
                "name": f"fold_root_flap_{flap_index}",
                "pos": f"{x_shift:.6f} {y_shift:.6f} {SEAM_Z + 0.012 - 0.010 * (flap_index - 1):.6f}",
                "euler": f"{angle:.6f} 0 0",
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
                "damping": "8.0",
                "stiffness": "2.8",
            },
        )
        half_len = _fold_half_length(config, flap_index)
        _geom(
            body,
            name=f"fold_root_flap_{flap_index}_geom",
            type="box",
            pos="0 0.028 -0.012",
            size=f"{half_len:.6f} 0.012000 0.026000",
            mass="0.024" if enabled else "0.001",
            rgba="0.70 0.50 0.27 0.90" if enabled else "0.70 0.50 0.27 0.04",
            contype="1" if enabled else "0",
            conaffinity="1" if enabled else "0",
        )
        if enabled:
            _geom(
                body,
                name=f"fold_root_flap_{flap_index}_roll",
                type="capsule",
                fromto=f"{-0.85 * half_len:.6f} 0.004 0.017 {0.85 * half_len:.6f} 0.004 0.017",
                size=f"{config.fold_root_thickness:.6f}",
                mass="0.006",
                rgba="0.40 0.22 0.08 0.95",
                contype="0",
                conaffinity="0",
            )
            for wrinkle_idx, x0 in enumerate((-0.055, 0.0, 0.055)):
                _geom(
                    body,
                    name=f"fold_root_flap_{flap_index}_wrinkle_{wrinkle_idx}",
                    type="capsule",
                    fromto=f"{x0 - 0.020:.6f} 0.022 0.012 {x0 + 0.016:.6f} 0.052 -0.004",
                    size="0.0024",
                    mass="0.001",
                    rgba="0.25 0.13 0.05 0.66",
                    contype="0",
                    conaffinity="0",
                )
        ET.SubElement(body, "site", {"name": f"grasp_fold_{flap_index}", "pos": "0 0.028 -0.012", "size": "0.005", "rgba": "0.1 0.2 1 0.40"})


def _add_payload(bag: ET.Element, config: ScenarioConfig) -> None:
    body = ET.SubElement(bag, "body", {"name": "payload_main", "pos": _fmt(config.payload_pos)})
    for axis_name, axis, limit in (("x", "1 0 0", "-0.028 0.028"), ("y", "0 1 0", "-0.024 0.024"), ("z", "0 0 1", "-0.020 0.020")):
        ET.SubElement(
            body,
            "joint",
            {
                "name": f"payload_main_{axis_name}",
                "type": "slide",
                "axis": axis,
                "limited": "true",
                "range": limit,
                "damping": "20",
                "stiffness": "3.0",
            },
        )
    _geom(
        body,
        name="payload_main_geom",
        type="ellipsoid",
        size=_fmt(config.payload_size),
        mass=f"{config.payload_mass:.4f}",
        rgba="0.42 0.22 0.10 0.72",
    )


def _add_visual_hulls(bag: ET.Element, config: ScenarioConfig) -> None:
    # 기준 외피는 위/아래가 대칭인 하나의 sealed hull로 둡니다.
    # 저충진/처짐 차이는 hull scale이 아니라 내부 payload와 articulated panel 반응으로 표현합니다.
    rx_scale = 0.5 * (config.top_radius_scale + config.lower_radius_scale)
    hull_scale_x = 0.5 * (config.upper_hull_scale[0] + config.lower_hull_scale[0])
    hull_scale_y = 0.5 * (config.upper_hull_scale[1] + config.lower_hull_scale[1])
    hull = ET.SubElement(bag, "body", {"name": "visual_symmetric_sealed_hull", "pos": "0 0 -0.004"})
    _geom(
        hull,
        name="visual_symmetric_sealed_hull_geom",
        type="ellipsoid",
        size=f"{BASE_RX * rx_scale * hull_scale_x:.6f} {BASE_RY * rx_scale * hull_scale_y:.6f} 0.145000",
        rgba="0.76 0.55 0.28 0.26",
        contype="0",
        conaffinity="0",
        mass="0.001",
    )
    upper = ET.SubElement(bag, "body", {"name": "visual_upper_hull", "pos": "0 0 0.052"})
    _geom(
        upper,
        name="visual_upper_hull_geom",
        type="ellipsoid",
        size=f"{BASE_RX * rx_scale * hull_scale_x:.6f} {BASE_RY * rx_scale * hull_scale_y:.6f} 0.078000",
        rgba="0.78 0.62 0.38 0.04",
        contype="0",
        conaffinity="0",
        mass="0.001",
    )
    lower = ET.SubElement(bag, "body", {"name": "visual_lower_hull", "pos": "0 0 -0.060"})
    _geom(
        lower,
        name="visual_lower_hull_geom",
        type="ellipsoid",
        size=f"{BASE_RX * rx_scale * hull_scale_x:.6f} {BASE_RY * rx_scale * hull_scale_y:.6f} 0.078000",
        rgba="0.72 0.50 0.25 0.04",
        contype="0",
        conaffinity="0",
        mass="0.001",
    )
    cap = ET.SubElement(bag, "body", {"name": "sealed_top_cap_visual", "pos": f"0 0 {SEAM_Z + 0.008:.6f}"})
    _geom(
        cap,
        name="sealed_top_cap_visual_geom",
        type="ellipsoid",
        size=f"{BASE_RX * config.top_radius_scale * 0.54:.6f} {BASE_RY * config.top_radius_scale * 0.48:.6f} 0.006000",
        rgba="0.82 0.64 0.36 0.45",
        contype="0",
        conaffinity="0",
        mass="0.001",
    )
    stitch = ET.SubElement(bag, "body", {"name": "sealed_top_stitch_visual", "pos": f"0 0 {SEAM_Z + 0.016:.6f}"})
    _geom(
        stitch,
        name="sealed_top_stitch_visual_geom",
        type="capsule",
        fromto="-0.064 0 0 0.064 0 0",
        size="0.0035",
        rgba="0.22 0.12 0.05 0.75",
        contype="0",
        conaffinity="0",
        mass="0.001",
    )
    bottom_stitch = ET.SubElement(bag, "body", {"name": "sealed_bottom_stitch_visual", "pos": f"0 0 {BOTTOM_Z - 0.010:.6f}"})
    _geom(
        bottom_stitch,
        name="sealed_bottom_stitch_visual_geom",
        type="capsule",
        fromto="-0.060 0 0 0.060 0 0",
        size="0.0030",
        rgba="0.22 0.12 0.05 0.42",
        contype="0",
        conaffinity="0",
        mass="0.001",
    )
    if config.name == "underfilled":
        wrinkle = ET.SubElement(bag, "body", {"name": "upper_empty_wrinkle_visual", "pos": "0 0 0.070"})
        for i, y in enumerate((-0.022, 0.0, 0.022)):
            _geom(
                wrinkle,
                name=f"upper_empty_wrinkle_{i}",
                type="capsule",
                fromto=f"-0.040 {y:.6f} 0 0.040 {0.45 * y:.6f} -0.003",
                size="0.0024",
                rgba="0.25 0.13 0.05 0.55",
                contype="0",
                conaffinity="0",
                mass="0.001",
            )


def _add_hidden_support(worldbody: ET.Element, config: ScenarioConfig) -> None:
    if not config.support_enabled:
        return
    support = ET.SubElement(worldbody, "body", {"name": "hidden_support", "pos": "0 0 0.058"})
    _geom(
        support,
        name="hidden_support_geom",
        type="box",
        size="0.104 0.062 0.012",
        rgba="0.10 0.65 0.95 0.24",
        friction="0.9 0.03 0.003",
    )


def _add_bag(worldbody: ET.Element, config: ScenarioConfig) -> None:
    bag = ET.SubElement(worldbody, "body", {"name": "bag_frame", "pos": "0 0 0.188"})
    ET.SubElement(bag, "freejoint", {"name": "bag_frame_freejoint"})
    ET.SubElement(bag, "site", {"name": "bag_frame_origin", "pos": "0 0 0", "size": "0.006", "rgba": "1 0 0 0.7"})
    _geom(
        bag,
        name="bag_mouse_handle",
        type="sphere",
        pos="0 0 0.020",
        size="0.020",
        mass="0.001",
        rgba="0.10 0.35 1.00 0.24",
        contype="0",
        conaffinity="0",
    )
    for i in range(SEGMENT_COUNT):
        _add_seam_band(bag, i, config)
    for i in range(SEGMENT_COUNT):
        _add_shoulder_panel(bag, i, config)
    for i in range(SEGMENT_COUNT):
        _add_belly_panel(bag, i, config)
    _add_bottom_sling(bag, config)
    _add_fold_flaps(bag, config)
    _add_payload(bag, config)
    _add_visual_hulls(bag, config)


def build_scene_tree(
    config: ScenarioConfig,
    *,
    include_eval_gripper: bool = True,
    include_eval_scoop: bool = False,
) -> ET.Element:
    root = ET.Element("mujoco", {"model": f"shape_coupled_sack_core_{config.name}"})
    ET.SubElement(root, "compiler", {"angle": "degree", "autolimits": "true"})
    ET.SubElement(
        root,
        "option",
        {
            "timestep": f"{TIMESTEP:.6f}",
            "gravity": "0 0 -9.81",
            "integrator": "implicitfast",
            "solver": "Newton",
            "iterations": "100",
            "ls_iterations": "24",
            "jacobian": "sparse",
            "cone": "elliptic",
            "impratio": "4",
        },
    )
    ET.SubElement(root, "size", {"nconmax": "900", "njmax": "1800"})
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"offwidth": "1280", "offheight": "820"})
    worldbody = ET.SubElement(root, "worldbody")
    _add_floor(worldbody)
    _add_cameras(worldbody)
    if include_eval_gripper:
        _add_eval_gripper(worldbody)
    if include_eval_scoop:
        _add_eval_scoop(worldbody)
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
    include_eval_scoop: bool = False,
) -> Path:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    config = get_scenario(scenario_name, post_release=post_release)
    suffix = "_after" if post_release else ""
    if output_path is None:
        output_path = GENERATED_DIR / f"scene_shape_coupled_sack_{scenario_name}{suffix}.xml"
    root = build_scene_tree(config, include_eval_gripper=include_eval_gripper, include_eval_scoop=include_eval_scoop)
    output_path.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    return output_path


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build shape-coupled semi-deformable sack surrogate")
    parser.add_argument("--scenario", default="underfilled")
    parser.add_argument("--post-release", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-eval-gripper", action="store_true")
    parser.add_argument("--eval-scoop", action="store_true")
    args = parser.parse_args()
    path = write_scene_xml(
        args.scenario,
        args.output,
        post_release=args.post_release,
        include_eval_gripper=not args.no_eval_gripper,
        include_eval_scoop=args.eval_scoop,
    )
    print(f"scene_xml={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
