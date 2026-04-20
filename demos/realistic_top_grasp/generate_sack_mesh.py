from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
GENERATED_DIR = ROOT_DIR / "generated"
OUTPUT_DIR = ROOT_DIR / "out"

# MuJoCo flex 안정성 파라미터: DEM/입자 충전 없이 2D shell만 사용한다.
TIMESTEP = 0.001
SOLVER = "Newton"
CONE_TYPE = "elliptic"
IMPRATIO = 10.0
INTEGRATOR = "implicitfast"
ITERATIONS = 120
LS_ITERATIONS = 30
JACOBIAN = "sparse"
SHELL_RADIUS = 0.0045
SHELL_DAMPING = 18.0
SHELL_FRICTION = "0.65 0.03 0.003"
SHELL_SOLREF = "0.028 1"
SHELL_SOLIMP = "0.84 0.96 0.001"
SHELL_CONDIM = "3"
SELF_COLLISION_MODE = "none"
SELF_COLLISION_MODES = ("none", "auto")
NOSLIP_ITERATIONS = 0
CCD_MODES = ("off", "on")
PAD_PROFILES = ("flat", "shallow_concave", "lip")
PAD_CONDIM_OPTIONS = (3, 4)
VERTCOLLIDE_MODES = ("false", "true")
NATIVECCD_MODES = ("off", "on")
# MuJoCo 3.1.6은 XML에서 nativeccd/vertcollide 속성을 받지 않는다.
# 옵션은 로그 비교용으로 유지하되, XML에는 안전한 항목만 기록한다.
NATIVECCD_XML_SUPPORTED = False
VERTCOLLIDE_XML_SUPPORTED = False

# 파지 품질 데모용 마대자루 shell 형상 파라미터.
RING_COUNT = 12
BAG_FRAME_Z = 0.165
BAG_MASS = 0.34
FRAME_MASS = 0.001
TOP_CENTER_Z = 0.068
TOP_RING_RX = 0.086
TOP_RING_RY = 0.040
TOP_RING_Z = 0.088
MID_RING_RX = 0.098
MID_RING_RY = 0.050
MID_RING_Z = -0.020
LOWER_RING_RX = 0.104
LOWER_RING_RY = 0.054
LOWER_RING_Z = -0.122
BOTTOM_CENTER_Z = -0.158

# 내부 충전 surrogate: DEM/입자 대신 낮은 단일 ellipsoid만 써서 하부 부피만 남긴다.
# 너무 크게 잡으면 자루가 꽉 찬 강체처럼 보이고 top grasp가 어려워진다.
CONTENT_SUPPORT_BODY = "bag_content_support"
CONTENT_SUPPORT_POS = "0.020 -0.010 -0.080"
CONTENT_SUPPORT_SIZE = "0.085 0.046 0.070"
CONTENT_SUPPORT_MASS = "0.55"
CONTENT_SUPPORT_RGBA = "0.55 0.30 0.10 0.78"
CONTENT_SUPPORT_FRICTION = "0.35 0.02 0.002"
CONTENT_SUPPORT_JOINT_RANGES = {
    "x": (-0.040, 0.040),
    "y": (-0.032, 0.032),
    "z": (-0.012, 0.018),
}

# 연구용 내부 충진 surrogate: DEM 입자 대신 3개의 제한된 clump로 coarse load redistribution만 표현한다.
CONTENT_CLUMP_PREFIX = "bag_content_clump"
CONTENT_CLUMP_NAMES = (
    f"{CONTENT_CLUMP_PREFIX}_central",
    f"{CONTENT_CLUMP_PREFIX}_left",
    f"{CONTENT_CLUMP_PREFIX}_right",
)
CONTENT_CLUMP_FRICTION = "0.38 0.02 0.002"
CONTENT_CLUMP_JOINT_RANGES = {
    "central": {
        "x": (-0.018, 0.018),
        "y": (-0.016, 0.016),
        "z": (-0.010, 0.008),
    },
    "left": {
        "x": (-0.018, 0.020),
        "y": (-0.016, 0.016),
        "z": (-0.010, 0.010),
    },
    "right": {
        "x": (-0.020, 0.018),
        "y": (-0.016, 0.016),
        "z": (-0.010, 0.010),
    },
}


@dataclass(frozen=True)
class ContentClumpSpec:
    name: str
    role: str
    pos: tuple[float, float, float]
    size: tuple[float, float, float]
    mass: float
    rgba: str


CONTENT_CLUMP_CASES: dict[str, tuple[ContentClumpSpec, ContentClumpSpec, ContentClumpSpec]] = {
    "underfilled": (
        ContentClumpSpec("central", "central_bulk", (0.000, 0.000, -0.112), (0.050, 0.032, 0.038), 0.31, "0.55 0.31 0.12 0.68"),
        ContentClumpSpec("left", "lateral_support", (-0.042, -0.004, -0.126), (0.032, 0.024, 0.028), 0.12, "0.48 0.24 0.10 0.62"),
        ContentClumpSpec("right", "lateral_support", (0.042, 0.004, -0.126), (0.032, 0.024, 0.028), 0.12, "0.48 0.24 0.10 0.62"),
    ),
    "eccentric": (
        ContentClumpSpec("central", "central_bulk", (0.018, -0.006, -0.106), (0.047, 0.031, 0.037), 0.25, "0.55 0.31 0.12 0.68"),
        ContentClumpSpec("left", "lateral_support", (-0.036, -0.010, -0.122), (0.029, 0.023, 0.027), 0.08, "0.42 0.21 0.08 0.58"),
        ContentClumpSpec("right", "lateral_support", (0.052, -0.004, -0.106), (0.039, 0.027, 0.033), 0.22, "0.65 0.36 0.14 0.78"),
    ),
    "support_sag": (
        ContentClumpSpec("central", "central_bulk", (0.000, 0.000, -0.122), (0.045, 0.030, 0.035), 0.23, "0.52 0.29 0.11 0.66"),
        ContentClumpSpec("left", "lateral_support", (-0.048, 0.010, -0.108), (0.036, 0.026, 0.032), 0.17, "0.60 0.33 0.13 0.72"),
        ContentClumpSpec("right", "lateral_support", (0.046, -0.014, -0.130), (0.034, 0.025, 0.030), 0.15, "0.49 0.25 0.10 0.64"),
    ),
}

# 2F gripper는 두 개의 mocap jaw로만 표현해서 IK 없이 local graspability를 검증한다.
JAW_PAD_HALF_X = 0.006
JAW_PAD_HALF_Y = 0.044
JAW_PAD_HALF_Z = 0.046
JAW_LIP_HALF_X = 0.0030
JAW_LIP_HALF_Z = 0.0065
JAW_OPEN_GAP = 0.145
JAW_CLOSED_GAP = 0.034
JAW_FRICTION = "4.0 0.30 0.03"

BAG_SHELL_BODY_PREFIX = "bag_shell_"
LEFT_JAW_BODY = "left_jaw_mocap"
RIGHT_JAW_BODY = "right_jaw_mocap"


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    description: str
    nominal_target_xy: tuple[float, float]
    nominal_target_z_offset: float = -0.006


SCENARIOS: dict[str, ScenarioSpec] = {
    "exposed_seam": ScenarioSpec(
        name="exposed_seam",
        description="상단 림의 일부가 살짝 솟아 seam처럼 노출된 경우",
        nominal_target_xy=(0.0, 0.026),
        nominal_target_z_offset=-0.004,
    ),
    "simple_fold": ScenarioSpec(
        name="simple_fold",
        description="상단 일부가 안쪽으로 접혀 plain top과 fold가 섞인 경우",
        nominal_target_xy=(-0.006, 0.010),
        nominal_target_z_offset=-0.010,
    ),
    "severe_fold": ScenarioSpec(
        name="severe_fold",
        description="상단이 더 깊게 뭉쳐 두꺼운 fold bundle이 생긴 경우",
        nominal_target_xy=(0.014, 0.000),
        nominal_target_z_offset=-0.013,
    ),
}


def available_scenarios() -> tuple[str, ...]:
    return tuple(SCENARIOS.keys())


def available_content_cases() -> tuple[str, ...]:
    return tuple(CONTENT_CLUMP_CASES.keys())


def ring_points(radius_x: float, radius_y: float, z: float) -> list[list[float]]:
    points: list[list[float]] = []
    for index in range(RING_COUNT):
        angle = 2.0 * math.pi * index / RING_COUNT
        points.append([radius_x * math.cos(angle), radius_y * math.sin(angle), z])
    return points


def flex_elements() -> str:
    top_center = 0
    top_start = 1
    mid_start = top_start + RING_COUNT
    lower_start = mid_start + RING_COUNT
    bottom_center = lower_start + RING_COUNT
    elements: list[tuple[int, int, int]] = []

    for index in range(RING_COUNT):
        next_index = (index + 1) % RING_COUNT
        top_i = top_start + index
        top_next = top_start + next_index
        mid_i = mid_start + index
        mid_next = mid_start + next_index
        lower_i = lower_start + index
        lower_next = lower_start + next_index

        elements.append((top_center, top_i, top_next))
        elements.append((top_i, mid_i, mid_next))
        elements.append((top_i, mid_next, top_next))
        elements.append((mid_i, lower_i, lower_next))
        elements.append((mid_i, lower_next, mid_next))
        elements.append((bottom_center, lower_next, lower_i))

    return "\n  ".join(f"{a} {b} {c}" for a, b, c in elements)


def format_points(points: list[list[float]]) -> str:
    return " ".join(f"{coord:.6f}" for point in points for coord in point)


def add_single_content_support(bag_frame: ET.Element) -> None:
    """자루 하부/중부 체적을 유지하는 단일 내부 충전 surrogate를 추가한다."""
    body = ET.SubElement(bag_frame, "body", {"name": CONTENT_SUPPORT_BODY, "pos": CONTENT_SUPPORT_POS})
    joint_specs = (
        ("x", "1 0 0", CONTENT_SUPPORT_JOINT_RANGES["x"], "24"),
        ("y", "0 1 0", CONTENT_SUPPORT_JOINT_RANGES["y"], "24"),
        ("z", "0 0 1", CONTENT_SUPPORT_JOINT_RANGES["z"], "32"),
    )
    for axis_name, axis, (low, high), damping in joint_specs:
        ET.SubElement(
            body,
            "joint",
            {
                "name": f"{CONTENT_SUPPORT_BODY}_{axis_name}",
                "type": "slide",
                "axis": axis,
                "limited": "true",
                "range": f"{low:.3f} {high:.3f}",
                "damping": damping,
            },
        )
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"{CONTENT_SUPPORT_BODY}_geom",
            "type": "ellipsoid",
            "size": CONTENT_SUPPORT_SIZE,
            "mass": CONTENT_SUPPORT_MASS,
            "rgba": CONTENT_SUPPORT_RGBA,
            "condim": "3",
            "friction": CONTENT_SUPPORT_FRICTION,
            "solref": "0.020 1",
            "solimp": "0.86 0.96 0.001",
        },
    )


def add_three_clump_content_support(bag_frame: ET.Element, content_case: str = "underfilled") -> None:
    """DEM 대신 3-clump로 내부 하중 재분배를 표현한다."""
    if content_case not in CONTENT_CLUMP_CASES:
        raise ValueError(f"unknown content case: {content_case}")

    for spec in CONTENT_CLUMP_CASES[content_case]:
        body_name = f"{CONTENT_CLUMP_PREFIX}_{spec.name}"
        body = ET.SubElement(
            bag_frame,
            "body",
            {
                "name": body_name,
                "pos": f"{spec.pos[0]:.6f} {spec.pos[1]:.6f} {spec.pos[2]:.6f}",
            },
        )
        for axis_name, axis, damping in (("x", "1 0 0", "18"), ("y", "0 1 0", "18"), ("z", "0 0 1", "24")):
            low, high = CONTENT_CLUMP_JOINT_RANGES[spec.name][axis_name]
            ET.SubElement(
                body,
                "joint",
                {
                    "name": f"{body_name}_{axis_name}",
                    "type": "slide",
                    "axis": axis,
                    "limited": "true",
                    "range": f"{low:.3f} {high:.3f}",
                    "damping": damping,
                },
            )
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{body_name}_geom",
                "type": "ellipsoid",
                "size": f"{spec.size[0]:.6f} {spec.size[1]:.6f} {spec.size[2]:.6f}",
                "mass": f"{spec.mass:.4f}",
                "rgba": spec.rgba,
                "condim": "3",
                "friction": CONTENT_CLUMP_FRICTION,
                "solref": "0.020 1",
                "solimp": "0.86 0.96 0.001",
                "contype": "2",
                "conaffinity": "1",
            },
        )


def _base_points() -> list[list[float]]:
    return [
        [0.0, 0.0, TOP_CENTER_Z],
        *ring_points(TOP_RING_RX, TOP_RING_RY, TOP_RING_Z),
        *ring_points(MID_RING_RX, MID_RING_RY, MID_RING_Z),
        *ring_points(LOWER_RING_RX, LOWER_RING_RY, LOWER_RING_Z),
        [0.0, 0.0, BOTTOM_CENTER_Z],
    ]


def _default_labels(point_count: int) -> list[str]:
    labels = ["other"] * point_count
    labels[0] = "plain_top"
    for point_index in range(1, 1 + RING_COUNT):
        labels[point_index] = "plain_top"
    return labels


def _angle_for_top_index(point_index: int) -> float:
    ring_index = point_index - 1
    return 2.0 * math.pi * ring_index / RING_COUNT


def make_sack_points(scenario_name: str) -> tuple[list[list[float]], list[str]]:
    if scenario_name not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario_name}")

    points = _base_points()
    labels = _default_labels(len(points))

    if scenario_name == "exposed_seam":
        # seam은 성공 규칙이 아니라 국소 형상 라벨이다. 여기서는 앞쪽 림을 살짝 두껍게 보이도록 만든다.
        for point_index in range(1, 1 + RING_COUNT):
            point = points[point_index]
            if point[1] > 0.017:
                labels[point_index] = "seam"
                point[1] *= 1.10
                point[2] += 0.012
            elif abs(point[1]) <= 0.010 and point[0] > 0.0:
                labels[point_index] = "seam"
                point[2] += 0.006

    elif scenario_name == "simple_fold":
        # 단순 fold는 앞쪽 상단 림 일부가 안쪽/아래쪽으로 들어온 초기 rest mesh로 표현한다.
        points[0][2] -= 0.014
        labels[0] = "fold"
        mid_start = 1 + RING_COUNT
        for point_index in range(1, 1 + RING_COUNT):
            point = points[point_index]
            angle = _angle_for_top_index(point_index)
            if math.sin(angle) > -0.10:
                labels[point_index] = "fold"
                labels[mid_start + (point_index - 1)] = "fold"
                point[0] *= 0.72
                point[1] *= 0.48
                point[2] -= 0.026
            elif point[0] < -0.03:
                point[2] += 0.006

    elif scenario_name == "severe_fold":
        # severe fold는 같은 판정식을 쓰되, 실제로는 더 두껍고 비대칭인 local bundle을 만든다.
        points[0][0] -= 0.020
        points[0][2] -= 0.026
        labels[0] = "fold"
        mid_start = 1 + RING_COUNT
        for point_index in range(1, 1 + RING_COUNT):
            point = points[point_index]
            angle = _angle_for_top_index(point_index)
            if math.sin(angle) > -0.45:
                labels[point_index] = "fold"
                labels[mid_start + (point_index - 1)] = "fold"
                point[0] = 0.45 * point[0] - 0.020 * math.cos(angle)
                point[1] = 0.28 * point[1] + 0.006 * math.sin(2.0 * angle)
                point[2] -= 0.038 + 0.008 * max(math.cos(angle), 0.0)
            else:
                point[2] += 0.004

    return points, labels


def nominal_grasp_center(scenario_name: str) -> np.ndarray:
    points, _labels = make_sack_points(scenario_name)
    spec = SCENARIOS[scenario_name]
    top_z = max(point[2] for point in points[: 1 + RING_COUNT])
    return np.array(
        [
            spec.nominal_target_xy[0],
            spec.nominal_target_xy[1],
            BAG_FRAME_Z + top_z + spec.nominal_target_z_offset,
        ],
        dtype=np.float64,
    )


def build_scene_tree(
    scenario_name: str,
    content_case: str = "underfilled",
    selfcollide_mode: str = SELF_COLLISION_MODE,
    noslip_iterations: int = NOSLIP_ITERATIONS,
    multiccd_mode: str = "off",
    nativeccd_mode: str = "off",
    pad_profile: str = "lip",
    pad_condim: int = 4,
    vertcollide_mode: str = "false",
    shell_thickness_scale: float = 1.0,
) -> ET.Element:
    if content_case not in CONTENT_CLUMP_CASES:
        raise ValueError(f"unknown content case: {content_case}")
    if selfcollide_mode not in SELF_COLLISION_MODES:
        raise ValueError(f"unknown selfcollide_mode: {selfcollide_mode}")
    if multiccd_mode not in CCD_MODES:
        raise ValueError(f"unknown multiccd_mode: {multiccd_mode}")
    if nativeccd_mode not in NATIVECCD_MODES:
        raise ValueError(f"unknown nativeccd_mode: {nativeccd_mode}")
    if pad_profile not in PAD_PROFILES:
        raise ValueError(f"unknown pad_profile: {pad_profile}")
    if int(pad_condim) not in PAD_CONDIM_OPTIONS:
        raise ValueError(f"unknown pad_condim: {pad_condim}")
    if vertcollide_mode not in VERTCOLLIDE_MODES:
        raise ValueError(f"unknown vertcollide_mode: {vertcollide_mode}")
    if shell_thickness_scale <= 0.0:
        raise ValueError("shell_thickness_scale must be positive")

    points, _labels = make_sack_points(scenario_name)
    shell_radius = SHELL_RADIUS * float(shell_thickness_scale)
    root = ET.Element("mujoco", {"model": f"realistic_top_grasp_{scenario_name}"})
    root.append(
        ET.Comment(
            "task-driven top graspability benchmark; labels are analysis-only, "
            f"timestep={TIMESTEP}, solver={SOLVER}, cone={CONE_TYPE}, impratio={IMPRATIO}, "
            f"shell_radius={shell_radius}, shell_damping={SHELL_DAMPING}, selfcollide={selfcollide_mode}, "
            f"multiccd={multiccd_mode}, nativeccd_requested={nativeccd_mode}, "
            f"vertcollide_requested={vertcollide_mode}, pad_profile={pad_profile}, pad_condim={pad_condim}"
        )
    )

    ET.SubElement(root, "compiler", {"angle": "radian", "coordinate": "local"})
    option = ET.SubElement(
        root,
        "option",
        {
            "timestep": f"{TIMESTEP:.6f}",
            "gravity": "0 0 -9.81",
            "integrator": INTEGRATOR,
            "solver": SOLVER,
            "iterations": str(ITERATIONS),
            "ls_iterations": str(LS_ITERATIONS),
            "jacobian": JACOBIAN,
            "cone": CONE_TYPE,
            "impratio": str(IMPRATIO),
            "noslip_iterations": str(int(noslip_iterations)),
        },
    )
    ET.SubElement(option, "flag", {"multiccd": "enable" if multiccd_mode == "on" else "disable"})
    ET.SubElement(root, "size", {"memory": "512M", "nconmax": "6000"})
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"offwidth": "1280", "offheight": "720"})

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(
        worldbody,
        "light",
        {"name": "key_light", "pos": "0.3 -0.5 1.7", "dir": "-0.2 0.3 -1", "diffuse": "0.9 0.9 0.9"},
    )
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": "overview",
            "pos": "0.74 -1.10 0.55",
            "xyaxes": "0.83 0.56 0 -0.20 0.30 0.93",
        },
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "size": "2 2 0.1",
            "rgba": "0.91 0.91 0.88 1",
            "friction": "1.1 0.03 0.003",
            "condim": "3",
        },
    )

    bag_frame = ET.SubElement(worldbody, "body", {"name": "bag_frame", "pos": f"0 0 {BAG_FRAME_Z:.6f}"})
    ET.SubElement(
        bag_frame,
        "inertial",
        {"pos": "0 0 0", "mass": f"{FRAME_MASS:.4f}", "diaginertia": "1e-6 1e-6 1e-6"},
    )
    ET.SubElement(bag_frame, "freejoint", {"name": "bag_frame_freejoint"})
    flexcomp = ET.SubElement(
        bag_frame,
        "flexcomp",
        {
            "name": "bag_shell",
            "type": "direct",
            "dim": "2",
            "mass": f"{BAG_MASS:.3f}",
            "radius": f"{shell_radius:.4f}",
            "rgba": "0.74 0.57 0.34 0.94",
            "point": format_points(points),
            "element": flex_elements(),
        },
    )
    ET.SubElement(
        flexcomp,
        "contact",
        {
            "condim": SHELL_CONDIM,
            "selfcollide": selfcollide_mode,
            "internal": "false",
            "friction": SHELL_FRICTION,
            "solref": SHELL_SOLREF,
            "solimp": SHELL_SOLIMP,
        },
    )
    ET.SubElement(flexcomp, "edge", {"equality": "true", "damping": f"{SHELL_DAMPING:.1f}"})
    add_three_clump_content_support(bag_frame, content_case)

    for body_name, sign, rgba in (
        (LEFT_JAW_BODY, -1.0, "0.15 0.35 0.95 0.88"),
        (RIGHT_JAW_BODY, 1.0, "0.95 0.28 0.15 0.88"),
    ):
        body = ET.SubElement(
            worldbody,
            "body",
            {
                "name": body_name,
                "mocap": "true",
                "pos": f"{sign * (JAW_OPEN_GAP * 0.5 + JAW_PAD_HALF_X):.6f} 0 0.42",
            },
        )
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{body_name}_pad",
                "type": "box",
                "size": f"{JAW_PAD_HALF_X:.6f} {JAW_PAD_HALF_Y:.6f} {JAW_PAD_HALF_Z:.6f}",
                "rgba": rgba,
                "friction": JAW_FRICTION,
                "condim": str(int(pad_condim)),
                "margin": "0.0015",
                "solref": "0.012 1",
                "solimp": "0.90 0.98 0.001",
            },
        )
        if pad_profile in {"lip", "shallow_concave"}:
            inner_sign = -sign
            lip_half_x = JAW_LIP_HALF_X if pad_profile == "lip" else JAW_LIP_HALF_X * 0.55
            lip_half_z = JAW_LIP_HALF_Z if pad_profile == "lip" else JAW_LIP_HALF_Z * 0.55
            lip_offset_x = JAW_PAD_HALF_X + lip_half_x * 0.85
            lip_offset_z = JAW_PAD_HALF_Z - lip_half_z
            for lip_index, z_sign in enumerate((-1.0, 1.0)):
                ET.SubElement(
                    body,
                    "geom",
                    {
                        "name": f"{body_name}_inward_lip_{lip_index}",
                        "type": "box",
                        "pos": f"{inner_sign * lip_offset_x:.6f} 0 {z_sign * lip_offset_z:.6f}",
                        "size": f"{lip_half_x:.6f} {JAW_PAD_HALF_Y:.6f} {lip_half_z:.6f}",
                        "rgba": rgba,
                        "friction": JAW_FRICTION,
                        "condim": str(int(pad_condim)),
                        "margin": "0.0015",
                        "solref": "0.012 1",
                        "solimp": "0.90 0.98 0.001",
                    },
        )
        capture_site = "left_capture_site" if body_name == LEFT_JAW_BODY else "right_capture_site"
        ET.SubElement(body, "site", {"name": f"{body_name}_site", "pos": "0 0 0", "size": "0.004", "rgba": rgba})
        ET.SubElement(body, "site", {"name": capture_site, "pos": "0 0 0", "size": "0.006", "rgba": rgba})

    center_body = ET.SubElement(worldbody, "body", {"name": "center_capture_mocap", "mocap": "true", "pos": "0 0 0.42"})
    ET.SubElement(center_body, "site", {"name": "center_capture_site", "pos": "0 0 0", "size": "0.006", "rgba": "1 1 0 0.75"})

    return root


def write_scene_xml(
    scenario_name: str,
    output_path: Path | None = None,
    content_case: str = "underfilled",
    selfcollide_mode: str = SELF_COLLISION_MODE,
    noslip_iterations: int = NOSLIP_ITERATIONS,
    multiccd_mode: str = "off",
    nativeccd_mode: str = "off",
    pad_profile: str = "lip",
    pad_condim: int = 4,
    vertcollide_mode: str = "false",
    shell_thickness_scale: float = 1.0,
) -> Path:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = GENERATED_DIR / f"scene_realistic_top_grasp_{scenario_name}_{content_case}.xml"
    root = build_scene_tree(
        scenario_name,
        content_case=content_case,
        selfcollide_mode=selfcollide_mode,
        noslip_iterations=noslip_iterations,
        multiccd_mode=multiccd_mode,
        nativeccd_mode=nativeccd_mode,
        pad_profile=pad_profile,
        pad_condim=pad_condim,
        vertcollide_mode=vertcollide_mode,
        shell_thickness_scale=shell_thickness_scale,
    )
    ET.indent(root, space="  ")
    output_path.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    return output_path
