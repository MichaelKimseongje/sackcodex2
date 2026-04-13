from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
GENERATED_DIR = ROOT_DIR / "generated"
OUTPUT_DIR = ROOT_DIR / "out"

# 핵심 물리 파라미터
TIMESTEP = 0.001
SOLVER = "Newton"
CONE_TYPE = "elliptic"
IMPRATIO = 10.0
SHELL_RADIUS = 0.0045
SHELL_THICKNESS = 0.0030
SHELL_DAMPING = 18.0
SELF_COLLISION_MODE = "none"
VERTCOLLIDE = False

# 안정성 관련 기본값
INTEGRATOR = "implicitfast"
ITERATIONS = 120
LS_ITERATIONS = 30
JACOBIAN = "sparse"
FLOOR_FRICTION = "0.9 0.02 0.002"
SHELL_FRICTION = "0.4 0.01 0.001"
SHELL_SOLREF = "0.03 1"
SHELL_SOLIMP = "0.85 0.95 0.001"
SHELL_CONDIM = "3"

# underfilled pouch surrogate 형상
BAG_FRAME_POS_Z = 0.157
BAG_MASS = 0.28
FRAME_MASS = 0.001
FRAME_DIAGINERTIA = "1e-6 1e-6 1e-6"
PIN_TOP_RIM = False
FREE_FRAME = True
TOP_CENTER_Z = 0.065
TOP_RING_RADIUS = 0.032
TOP_RING_Z = 0.100
MID_RING_RADIUS = 0.048
MID_RING_Z = -0.015
LOWER_RING_RADIUS = 0.078
LOWER_RING_Z = -0.120
BOTTOM_CENTER_Z = -0.152

# 선택적 단일 ballast
BALLAST_BODY_NAME = "bag_ballast"
BALLAST_POS = "0 0 -0.108"
BALLAST_SIZE = "0.028 0.022 0.030"
BALLAST_MASS = "0.34"

# 이름 prefix
BAG_SHELL_BODY_PREFIX = "bag_shell_"
BALLAST_BODY_PREFIX = "bag_ballast"

# 렌더/교란 기본값
RENDER_WIDTH = 1280
RENDER_HEIGHT = 720
DEMO_SECONDS = 3.0
DISTURB_AT_SECONDS = 1.8
DISTURB_CAPTURE_DELAY = 0.2
FRAME_COUNT = 60


ELEMENT_TEXT = """
  0 1 2
  0 2 3
  0 3 4
  0 4 5
  0 5 6
  0 6 7
  0 7 8
  0 8 1
  1 2 10
  1 10 9
  2 3 11
  2 11 10
  3 4 12
  3 12 11
  4 5 13
  4 13 12
  5 6 14
  5 14 13
  6 7 15
  6 15 14
  7 8 16
  7 16 15
  8 1 9
  8 9 16
  9 10 18
  9 18 17
  10 11 19
  10 19 18
  11 12 20
  11 20 19
  12 13 21
  12 21 20
  13 14 22
  13 22 21
  14 15 23
  14 23 22
  15 16 24
  15 24 23
  16 9 17
  16 17 24
  25 18 17
  25 19 18
  25 20 19
  25 21 20
  25 22 21
  25 23 22
  25 24 23
  25 17 24
""".strip()


def _ring_points(radius: float, z: float) -> list[list[float]]:
    points: list[list[float]] = []
    for index in range(8):
        angle = 2.0 * math.pi * index / 8.0
        points.append([radius * math.cos(angle), radius * math.sin(angle), z])
    return points


def _underfilled_points() -> list[list[float]]:
    return [
        [0.0, 0.0, TOP_CENTER_Z],
        *_ring_points(TOP_RING_RADIUS, TOP_RING_Z),
        *_ring_points(MID_RING_RADIUS, MID_RING_Z),
        *_ring_points(LOWER_RING_RADIUS, LOWER_RING_Z),
        [0.0, 0.0, BOTTOM_CENTER_Z],
    ]


def _format_points(points: list[list[float]]) -> str:
    return " ".join(f"{coord:.6f}" for point in points for coord in point)


def _parameter_comment() -> str:
    return (
        f"timestep={TIMESTEP}, solver={SOLVER}, cone={CONE_TYPE}, impratio={IMPRATIO}, "
        f"shell_radius={SHELL_RADIUS}, shell_thickness={SHELL_THICKNESS}, "
        f"shell_damping={SHELL_DAMPING}, self_collision_mode={SELF_COLLISION_MODE}, "
        f"vertcollide={VERTCOLLIDE} "
        "[MuJoCo 3.1.6 direct flexcomp에서는 thickness/vertcollide가 XML 속성으로 직접 지원되지 않아 "
        "문서화 파라미터로 유지하고, 실제 안정성은 radius/contact/edge damping으로 제어한다.]"
        f" [free_frame={FREE_FRAME}, pin_top_rim={PIN_TOP_RIM}]"
    )


def _add_ballast(bag_frame: ET.Element) -> None:
    body = ET.SubElement(bag_frame, "body", {"name": BALLAST_BODY_NAME, "pos": BALLAST_POS})
    joint_specs = (
        ("x", "1 0 0", "-0.020 0.020", "10"),
        ("y", "0 1 0", "-0.020 0.020", "10"),
        ("z", "0 0 1", "-0.010 0.016", "14"),
    )
    for axis_name, axis, limits, damping in joint_specs:
        ET.SubElement(
            body,
            "joint",
            {
                "name": f"{BALLAST_BODY_NAME}_{axis_name}",
                "type": "slide",
                "axis": axis,
                "limited": "true",
                "range": limits,
                "damping": damping,
            },
        )

    ET.SubElement(
        body,
        "geom",
        {
            "name": f"{BALLAST_BODY_NAME}_geom",
            "type": "ellipsoid",
            "size": BALLAST_SIZE,
            "mass": BALLAST_MASS,
            "rgba": "0.48 0.22 0.15 1",
            "condim": SHELL_CONDIM,
            "friction": "0.35 0.01 0.001",
        },
    )


def build_scene_tree(with_ballast: bool = False) -> ET.Element:
    root = ET.Element("mujoco", {"model": "demo_low_fill"})
    root.append(ET.Comment(_parameter_comment()))

    ET.SubElement(root, "compiler", {"angle": "radian", "coordinate": "local"})
    ET.SubElement(
        root,
        "option",
        {
            "timestep": str(TIMESTEP),
            "gravity": "0 0 -9.81",
            "integrator": INTEGRATOR,
            "solver": SOLVER,
            "iterations": str(ITERATIONS),
            "ls_iterations": str(LS_ITERATIONS),
            "jacobian": JACOBIAN,
            "cone": CONE_TYPE,
            "impratio": str(IMPRATIO),
        },
    )
    ET.SubElement(root, "size", {"memory": "256M", "nconmax": "4000"})

    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "global",
        {"offwidth": str(RENDER_WIDTH), "offheight": str(RENDER_HEIGHT)},
    )

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(
        worldbody,
        "light",
        {
            "name": "key_light",
            "pos": "0.2 -0.4 1.8",
            "dir": "-0.1 0.2 -1",
            "diffuse": "0.9 0.9 0.9",
        },
    )
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": "overview",
            "pos": "0.78 -1.24 0.63",
            "xyaxes": "0.84 0.54 0 -0.22 0.34 0.91",
        },
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "size": "2 2 0.1",
            "rgba": "0.92 0.92 0.92 1",
            "friction": FLOOR_FRICTION,
            "condim": SHELL_CONDIM,
        },
    )

    bag_frame = ET.SubElement(worldbody, "body", {"name": "bag_frame", "pos": f"0 0 {BAG_FRAME_POS_Z:.3f}"})
    if FREE_FRAME:
        ET.SubElement(
            bag_frame,
            "inertial",
            {
                "pos": "0 0 0",
                "mass": f"{FRAME_MASS:.4f}",
                "diaginertia": FRAME_DIAGINERTIA,
            },
        )
        ET.SubElement(bag_frame, "freejoint", {"name": "bag_frame_freejoint"})
    ET.SubElement(
        bag_frame,
        "site",
        {"name": "bag_frame_origin", "pos": "0 0 0", "size": "0.004", "rgba": "0.8 0.15 0.15 1"},
    )

    flexcomp = ET.SubElement(
        bag_frame,
        "flexcomp",
        {
            "name": "bag_shell",
            "type": "direct",
            "dim": "2",
            "mass": f"{BAG_MASS:.3f}",
            "radius": f"{SHELL_RADIUS:.4f}",
            "rgba": "0.75 0.60 0.36 1",
            "point": _format_points(_underfilled_points()),
            "element": ELEMENT_TEXT,
        },
    )
    ET.SubElement(
        flexcomp,
        "contact",
        {
            "condim": SHELL_CONDIM,
            "selfcollide": SELF_COLLISION_MODE,
            "internal": "false",
            "friction": SHELL_FRICTION,
            "solref": SHELL_SOLREF,
            "solimp": SHELL_SOLIMP,
        },
    )
    ET.SubElement(
        flexcomp,
        "edge",
        {
            "equality": "true",
            "damping": f"{SHELL_DAMPING:.1f}",
        },
    )
    if PIN_TOP_RIM:
        for pin_id in range(1, 9):
            ET.SubElement(flexcomp, "pin", {"id": str(pin_id)})

    if with_ballast:
        _add_ballast(bag_frame)

    return root


def write_scene_xml(with_ballast: bool = False, output_path: Path | None = None) -> Path:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        suffix = "ballast1" if with_ballast else "shell_only"
        output_path = GENERATED_DIR / f"low_fill_{suffix}.xml"

    root = build_scene_tree(with_ballast=with_ballast)
    ET.indent(root, space="  ")
    output_path.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    return output_path


def load_scene(with_ballast: bool = False) -> tuple[Path, mujoco.MjModel, mujoco.MjData]:
    xml_path = write_scene_xml(with_ballast=with_ballast)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    return xml_path, model, data


def collect_body_ids_by_prefix(model: mujoco.MjModel, prefix: str) -> list[int]:
    body_ids: list[int] = []
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name and name.startswith(prefix):
            body_ids.append(body_id)
    return body_ids


def collect_shell_body_ids(model: mujoco.MjModel) -> list[int]:
    return collect_body_ids_by_prefix(model, BAG_SHELL_BODY_PREFIX)


def collect_internal_body_ids(model: mujoco.MjModel) -> list[int]:
    return collect_body_ids_by_prefix(model, BALLAST_BODY_PREFIX)


def shell_positions(data: mujoco.MjData, shell_body_ids: list[int]) -> np.ndarray:
    return np.asarray(data.xpos[shell_body_ids], dtype=np.float64)


def compute_shell_spans(shell_body_positions: np.ndarray) -> tuple[float, float, float]:
    z_values = shell_body_positions[:, 2]
    upper_threshold = float(np.percentile(z_values, 67))
    lower_threshold = float(np.percentile(z_values, 33))

    upper_slice = shell_body_positions[z_values >= upper_threshold]
    lower_slice = shell_body_positions[z_values <= lower_threshold]

    upper_span_x = float(np.max(upper_slice[:, 0]) - np.min(upper_slice[:, 0]))
    lower_span_x = float(np.max(lower_slice[:, 0]) - np.min(lower_slice[:, 0]))
    bag_height = float(np.max(z_values) - np.min(z_values))
    return upper_span_x, lower_span_x, bag_height


def count_escaped_internal_bodies(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    shell_body_ids: list[int],
    *,
    margin_xy: float = 0.03,
    margin_z: float = 0.03,
) -> int:
    internal_body_ids = collect_internal_body_ids(model)
    if not internal_body_ids:
        return 0

    shell_xyz = shell_positions(data, shell_body_ids)
    bounds_min = np.min(shell_xyz, axis=0) - np.array([margin_xy, margin_xy, margin_z], dtype=np.float64)
    bounds_max = np.max(shell_xyz, axis=0) + np.array([margin_xy, margin_xy, margin_z], dtype=np.float64)

    escaped = 0
    for body_id in internal_body_ids:
        position = np.asarray(data.xpos[body_id], dtype=np.float64)
        if np.any(position < bounds_min) or np.any(position > bounds_max):
            escaped += 1
    return escaped


def make_render_option() -> mujoco.MjvOption:
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    option.flags[mujoco.mjtVisFlag.mjVIS_FLEXFACE] = True
    option.flags[mujoco.mjtVisFlag.mjVIS_FLEXEDGE] = True
    option.flags[mujoco.mjtVisFlag.mjVIS_FLEXVERT] = True
    return option


def apply_upper_shell_disturbance(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    shell_body_ids = collect_shell_body_ids(model)
    if not shell_body_ids:
        return

    shell_xyz = shell_positions(data, shell_body_ids)
    z_values = shell_xyz[:, 2]
    threshold = float(np.percentile(z_values, 67))

    for body_id in shell_body_ids:
        if data.xpos[body_id, 2] < threshold:
            continue

        dof_address = model.body_dofadr[body_id]
        if dof_address < 0:
            continue

        data.qvel[dof_address + 0] += 0.18
        data.qvel[dof_address + 2] -= 0.10


def apply_ballast_impulse(model: mujoco.MjModel, data: mujoco.MjData, magnitude: float = 0.35) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{BALLAST_BODY_NAME}_x")
    if joint_id < 0:
        return

    dof_address = model.jnt_dofadr[joint_id]
    data.qvel[dof_address] += magnitude


def apply_disturbance(model: mujoco.MjModel, data: mujoco.MjData, *, with_ballast: bool) -> None:
    apply_upper_shell_disturbance(model, data)
    if with_ballast:
        apply_ballast_impulse(model, data, magnitude=0.35)


def default_output_dir(with_ballast: bool = False) -> Path:
    suffix = "version_b_ballast1" if with_ballast else "version_a_shell_only"
    return OUTPUT_DIR / suffix
