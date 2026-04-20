from __future__ import annotations

import argparse
import math
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
GENERATED_DIR = ROOT_DIR / "generated"
OUT_DIR = ROOT_DIR / "out"

TIMESTEP = 0.001
SACK_LENGTH = 0.420
SACK_WIDTH = 0.240
SACK_THICKNESS = 0.150
SACK_Z = 0.105
TOP_SEAM_COUNT = 5
CONNECTED_COLUMN_COUNT = 7
CONNECTED_LAYER_COUNT = 3
CONNECTED_BOTTOM_SEGMENT_COUNT = 5
OUTER_FRONT_COUNT = 7
OUTER_BACK_COUNT = 7
OUTER_SHOULDER_COUNT = 5
OUTER_SIDE_COUNT = 4
OUTER_LOWER_COUNT = 8
OUTER_BOTTOM_EDGE_COUNT = 7
SHOULDER_PANEL_COUNT = OUTER_SHOULDER_COUNT
LOWER_BELLY_PANEL_COUNT = OUTER_LOWER_COUNT
INNER_LOAD_PANEL_COUNT = 5
INNER_BOTTOM_PANEL_COUNT = 5
UR5E_ASSET_DIR = Path(r"D:\Michael\2025\01.Research\01.Parceldetection\16.Pybullet\mujoco_menagerie\universal_robots_ur5e\assets")
UR5E_MESH_FILES = (
    "base_0.obj",
    "base_1.obj",
    "shoulder_0.obj",
    "shoulder_1.obj",
    "shoulder_2.obj",
    "upperarm_0.obj",
    "upperarm_1.obj",
    "upperarm_2.obj",
    "upperarm_3.obj",
    "forearm_0.obj",
    "forearm_1.obj",
    "forearm_2.obj",
    "forearm_3.obj",
    "wrist1_0.obj",
    "wrist1_1.obj",
    "wrist1_2.obj",
    "wrist2_0.obj",
    "wrist2_1.obj",
    "wrist2_2.obj",
    "wrist3.obj",
)

SCENARIO_NAMES = (
    "baseline_filled",
    "empty_collapsed",
    "underfilled",
    "top_fold_simple",
    "top_fold_severe",
    "eccentric_fill",
    "jammed_between_neighbors",
    "post_separation_sag",
)


@dataclass(frozen=True)
class ScenarioState:
    """같은 articulated skeleton의 state만 바꾸기 위한 시나리오 파라미터입니다."""

    name: str
    top_width_scale: float = 1.0
    lower_width_scale: float = 1.0
    top_crown_scale: float = 1.0
    lower_bulge_scale: float = 1.0
    shoulder_rest_deg: float = 0.0
    shoulder_stiffness: float = 2.0
    belly_rest_deg: float = 0.0
    belly_stiffness: float = 2.5
    fold_left_deg: float = 0.0
    fold_right_deg: float = 0.0
    fold_coverage_fraction: float = 0.0
    fold_root_thickness: float = 0.010
    payload_main_pos: tuple[float, float, float] = (0.0, 0.0, -0.006)
    payload_aux_pos: tuple[float, float, float] = (0.0, 0.0, -0.004)
    payload_aux_mass: float = 0.04
    fill_volume_scale: float = 1.0
    ballast_mass_scale: float = 1.0
    visual_bulge_y: float = 0.0
    body_tilt_deg: float = 0.0
    neighbor_gap: float = 0.35
    neighbors_active: bool = False
    hidden_support_active: bool = False
    bottom_sling_rest: float = 0.0
    description: str = ""


SCENARIOS: dict[str, ScenarioState] = {
    "baseline_filled": ScenarioState(
        name="baseline_filled",
        fill_volume_scale=1.0,
        shoulder_stiffness=4.2,
        description="대칭적인 밀봉 쌀포대/밀가루포대 기준 형상",
    ),
    "empty_collapsed": ScenarioState(
        name="empty_collapsed",
        top_width_scale=0.94,
        lower_width_scale=0.62,
        top_crown_scale=0.34,
        lower_bulge_scale=0.24,
        shoulder_rest_deg=-28.0,
        shoulder_stiffness=0.18,
        belly_rest_deg=-20.0,
        belly_stiffness=0.35,
        payload_main_pos=(0.0, 0.0, -0.052),
        payload_aux_pos=(0.0, 0.0, -0.052),
        payload_aux_mass=0.002,
        fill_volume_scale=0.05,
        ballast_mass_scale=0.04,
        description="internal support removed; collapsed sealed sack reference",
    ),
    "underfilled": ScenarioState(
        name="underfilled",
        top_width_scale=0.78,
        lower_width_scale=1.08,
        top_crown_scale=0.62,
        lower_bulge_scale=1.15,
        fill_volume_scale=0.62,
        shoulder_rest_deg=-18.0,
        shoulder_stiffness=0.35,
        belly_rest_deg=8.0,
        belly_stiffness=1.8,
        payload_main_pos=(0.0, 0.0, -0.032),
        payload_aux_pos=(0.030, -0.015, -0.030),
        payload_aux_mass=0.02,
        description="윗부분은 비고 하부에 내용물이 몰린 저충진 상태",
    ),
    "top_fold_simple": ScenarioState(
        name="top_fold_simple",
        fold_left_deg=-36.0,
        fold_coverage_fraction=0.30,
        fold_root_thickness=0.016,
        description="상단 한쪽이 접혀 seam 일부를 가리는 상태",
    ),
    "top_fold_severe": ScenarioState(
        name="top_fold_severe",
        top_crown_scale=0.82,
        fold_left_deg=-54.0,
        fold_right_deg=48.0,
        fold_coverage_fraction=0.62,
        fold_root_thickness=0.024,
        description="양쪽 또는 큰 bundle이 seam을 많이 가리는 심한 접힘 상태",
    ),
    "eccentric_fill": ScenarioState(
        name="eccentric_fill",
        payload_main_pos=(0.0, 0.045, -0.012),
        payload_aux_pos=(0.040, 0.072, -0.010),
        payload_aux_mass=0.18,
        visual_bulge_y=0.030,
        body_tilt_deg=4.5,
        shoulder_rest_deg=5.0,
        belly_rest_deg=7.0,
        description="동일 자루 내부 질량이 한쪽으로 치우친 편심 충진 상태",
    ),
    "jammed_between_neighbors": ScenarioState(
        name="jammed_between_neighbors",
        top_width_scale=0.72,
        lower_width_scale=0.76,
        lower_bulge_scale=0.88,
        shoulder_rest_deg=-10.0,
        belly_rest_deg=-8.0,
        neighbor_gap=0.130,
        neighbors_active=True,
        description="양쪽 이웃 자루/벽에 의해 폭이 줄고 삽입 여유가 감소한 끼임 상태",
    ),
    "post_separation_sag": ScenarioState(
        name="post_separation_sag",
        top_crown_scale=0.92,
        lower_bulge_scale=1.12,
        belly_rest_deg=12.0,
        belly_stiffness=1.2,
        payload_main_pos=(0.0, 0.0, -0.025),
        hidden_support_active=True,
        bottom_sling_rest=-0.026,
        description="초기 하부 지지 제거 후 lower belly와 bottom sling이 더 처지는 상태",
    ),
}


def get_scenario(name: str) -> ScenarioState:
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}. Available: {', '.join(SCENARIO_NAMES)}")
    return SCENARIOS[name]


def _fmt(values: tuple[float, ...] | list[float]) -> str:
    return " ".join(f"{float(v):.6f}" for v in values)


def _rad(deg: float) -> float:
    return math.radians(float(deg))


def _fill_geometry_scale(state: ScenarioState) -> tuple[float, float]:
    """내부 지지량을 외피 폭/높이에 반영하기 위한 scale입니다."""
    fill = max(0.0, min(1.25, state.fill_volume_scale))
    height_scale = 0.32 + 0.68 * fill
    return fill, height_scale


def _geom(parent: ET.Element, **attrib: str) -> ET.Element:
    defaults = {
        "group": "1",
        "condim": "4",
        "friction": "1.35 0.06 0.006",
        "solref": "0.028 1",
        "solimp": "0.76 0.94 0.001",
    }
    defaults.update(attrib)
    return ET.SubElement(parent, "geom", defaults)


def _joint(parent: ET.Element, **attrib: str) -> ET.Element:
    defaults = {"damping": "4.0", "stiffness": "1.0", "limited": "true"}
    defaults.update(attrib)
    return ET.SubElement(parent, "joint", defaults)


def _hinge_capsule(parent: ET.Element, *, name: str, fromto: str, size: float = 0.0042) -> ET.Element:
    """넓은 외피 판이 원기둥형 seam 축을 따라 접히는 것처럼 보이게 하는 hinge cue입니다."""
    return _geom(
        parent,
        name=name,
        type="capsule",
        fromto=fromto,
        size=f"{size:.6f}",
        material="mat_ur5_joint_marker",
        group="1",
        contype="0",
        conaffinity="0",
        mass="0.001",
    )


def _make_visual_skin_mesh(state: ScenarioState) -> tuple[str, str]:
    """시나리오 cue를 반영하는 visual-only sealed pillow mesh입니다."""
    nx, ny = 19, 13
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for layer_sign in (1.0, -1.0):
        for iy in range(ny):
            v = -1.0 + 2.0 * iy / (ny - 1)
            for ix in range(nx):
                u = -1.0 + 2.0 * ix / (nx - 1)
                edge = max(abs(u), abs(v))
                is_top = layer_sign > 0
                width_scale = state.top_width_scale if is_top else state.lower_width_scale
                x = 0.5 * SACK_LENGTH * u
                y = 0.5 * SACK_WIDTH * width_scale * v
                crown = state.top_crown_scale if is_top else state.lower_bulge_scale
                zmag = SACK_THICKNESS * (0.055 + 0.34 * crown * max(0.0, 1.0 - edge**2.25))
                # 편심 충진은 heavy side 쪽 하부가 살짝 더 튀어나오도록 visual cue를 둡니다.
                if state.visual_bulge_y and v > 0.20:
                    y += state.visual_bulge_y * (1.0 - abs(u)) * (0.5 if is_top else 1.0)
                    zmag *= 1.0 + 0.10 * (1.0 - edge)
                verts.append((x, y, layer_sign * zmag))

    def vid(layer: int, ix: int, iy: int) -> int:
        return layer * nx * ny + iy * nx + ix

    for iy in range(ny - 1):
        for ix in range(nx - 1):
            a, b, c, d = vid(0, ix, iy), vid(0, ix + 1, iy), vid(0, ix, iy + 1), vid(0, ix + 1, iy + 1)
            faces.extend(((a, b, c), (b, d, c)))
            a, b, c, d = vid(1, ix, iy), vid(1, ix + 1, iy), vid(1, ix, iy + 1), vid(1, ix + 1, iy + 1)
            faces.extend(((a, c, b), (b, c, d)))

    for ix in range(nx - 1):
        for yidx in (0, ny - 1):
            t0, t1 = vid(0, ix, yidx), vid(0, ix + 1, yidx)
            b0, b1 = vid(1, ix, yidx), vid(1, ix + 1, yidx)
            faces.extend(((t0, b0, t1), (t1, b0, b1)) if yidx == 0 else ((t0, t1, b0), (t1, b1, b0)))

    for iy in range(ny - 1):
        for xidx in (0, nx - 1):
            t0, t1 = vid(0, xidx, iy), vid(0, xidx, iy + 1)
            b0, b1 = vid(1, xidx, iy), vid(1, xidx, iy + 1)
            faces.extend(((t0, t1, b0), (t1, b1, b0)) if xidx == 0 else ((t0, b0, t1), (t1, b0, b1)))

    vertex = " ".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts)
    face = " ".join(f"{a} {b} {c}" for a, b, c in faces)
    return vertex, face


def _make_longitudinal_end_cap_mesh(state: ScenarioState) -> tuple[str, str]:
    """길이 방향 끝단을 막는 visual-only pillow end-cap mesh입니다."""
    fill, height_scale = _fill_geometry_scale(state)
    top_z = 0.018 + 0.5 * SACK_THICKNESS * state.top_crown_scale * height_scale
    # 앞/뒤 판은 top seam에서 하단 edge로 내려가며 바깥쪽으로 벌어지는 사선 판입니다.
    top_y = (0.18 + 0.06 * fill) * SACK_WIDTH * state.top_width_scale
    lower_y = (0.26 + 0.20 * fill) * SACK_WIDTH * state.lower_width_scale
    shell_h = 0.138 * (0.92 + 0.08 * state.lower_bulge_scale) * height_scale
    bottom_z = top_z - shell_h
    half_x = 0.005
    yz = [
        (-top_y, top_z),
        (top_y, top_z),
        (lower_y, bottom_z),
        (-lower_y, bottom_z),
    ]
    verts: list[tuple[float, float, float]] = []
    for x in (-half_x, half_x):
        for y, z in yz:
            verts.append((x, y, z))

    faces: list[tuple[int, int, int]] = []
    # end-cap 앞/뒤 polygon을 fan triangulation으로 닫습니다.
    for base, reverse in ((0, False), (len(yz), True)):
        for i in range(1, len(yz) - 1):
            tri = (base, base + i, base + i + 1)
            faces.append((tri[0], tri[2], tri[1]) if reverse else tri)
    # 얇은 두께의 side wall을 닫습니다.
    n = len(yz)
    for i in range(n):
        j = (i + 1) % n
        a, b, c, d = i, j, n + i, n + j
        faces.extend(((a, b, c), (b, d, c)))

    vertex = " ".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts)
    face = " ".join(f"{a} {b} {c}" for a, b, c in faces)
    return vertex, face


def _add_assets(root: ET.Element, state: ScenarioState, *, include_robots: bool) -> None:
    asset = ET.SubElement(root, "asset")
    vertex, face = _make_visual_skin_mesh(state)
    ET.SubElement(asset, "mesh", {"name": "sealed_pillow_skin_mesh", "vertex": vertex, "face": face})
    end_vertex, end_face = _make_longitudinal_end_cap_mesh(state)
    ET.SubElement(asset, "mesh", {"name": "longitudinal_end_cap_mesh", "vertex": end_vertex, "face": end_face})
    ET.SubElement(asset, "material", {"name": "mat_jute", "rgba": "0.72 0.58 0.36 0.36"})
    ET.SubElement(asset, "material", {"name": "mat_shell_panel", "rgba": "0.70 0.55 0.31 1.00"})
    ET.SubElement(asset, "material", {"name": "mat_connected_shell", "rgba": "0.74 0.61 0.40 1.00"})
    ET.SubElement(asset, "material", {"name": "mat_connected_edge", "rgba": "0.46 0.32 0.17 1.00"})
    ET.SubElement(asset, "material", {"name": "mat_front_back_panel", "rgba": "0.70 0.82 0.96 0.96"})
    ET.SubElement(asset, "material", {"name": "mat_side_panel", "rgba": "0.72 0.70 0.95 0.96"})
    ET.SubElement(asset, "material", {"name": "mat_panel_hidden", "rgba": "0.70 0.55 0.31 0.94"})
    ET.SubElement(asset, "material", {"name": "mat_hinge_cylinder", "rgba": "0.43 0.28 0.13 1.00"})
    ET.SubElement(asset, "material", {"name": "mat_seam", "rgba": "0.34 0.22 0.10 1.00"})
    ET.SubElement(asset, "material", {"name": "mat_fold", "rgba": "0.66 0.46 0.22 0.82"})
    ET.SubElement(asset, "material", {"name": "mat_inner_shell", "rgba": "0.18 0.42 0.78 0.34"})
    ET.SubElement(asset, "material", {"name": "mat_ballast", "rgba": "0.30 0.13 0.06 0.26"})
    ET.SubElement(asset, "material", {"name": "mat_payload", "rgba": "0.30 0.13 0.06 0.20"})
    ET.SubElement(asset, "material", {"name": "mat_ur5_base", "rgba": "0.26 0.30 0.34 1"})
    ET.SubElement(asset, "material", {"name": "mat_ur5_link", "rgba": "0.74 0.78 0.80 1"})
    ET.SubElement(asset, "material", {"name": "mat_ur5_joint_marker", "rgba": "1.00 0.48 0.08 1"})
    ET.SubElement(asset, "material", {"name": "mat_gripper", "rgba": "0.08 0.09 0.10 1"})
    ET.SubElement(asset, "material", {"name": "mat_pad", "rgba": "0.02 0.02 0.02 1"})
    ET.SubElement(asset, "material", {"name": "mat_scoop", "rgba": "0.15 0.24 0.30 1"})
    ET.SubElement(asset, "material", {"name": "mat_ur5e_black", "rgba": "0.033 0.033 0.033 1", "specular": "0.5", "shininess": "0.25"})
    ET.SubElement(asset, "material", {"name": "mat_ur5e_jointgray", "rgba": "0.278 0.278 0.278 1", "specular": "0.5", "shininess": "0.25"})
    ET.SubElement(asset, "material", {"name": "mat_ur5e_linkgray", "rgba": "0.82 0.82 0.82 1", "specular": "0.5", "shininess": "0.25"})
    ET.SubElement(asset, "material", {"name": "mat_ur5e_urblue", "rgba": "0.49 0.678 0.8 1", "specular": "0.5", "shininess": "0.25"})
    if include_robots:
        if not UR5E_ASSET_DIR.exists():
            raise FileNotFoundError(f"UR5e mesh asset folder not found: {UR5E_ASSET_DIR}")
        for filename in UR5E_MESH_FILES:
            mesh_path = UR5E_ASSET_DIR / filename
            if not mesh_path.exists():
                raise FileNotFoundError(f"UR5e mesh asset not found: {mesh_path}")
            ET.SubElement(asset, "mesh", {"name": f"ur5e_{Path(filename).stem}", "file": str(mesh_path)})


def _add_world(worldbody: ET.Element) -> None:
    _geom(worldbody, name="floor", type="plane", size="2.4 2.4 0.05", group="0", rgba="0.86 0.84 0.77 0.55", friction="1.55 0.05 0.004")
    ET.SubElement(worldbody, "light", {"name": "key_light", "pos": "0.5 -0.7 1.3", "dir": "-0.25 0.3 -1"})
    ET.SubElement(worldbody, "light", {"name": "fill_light", "pos": "-0.6 0.5 0.9", "dir": "0.4 -0.2 -1", "diffuse": "0.35 0.32 0.28"})
    ET.SubElement(worldbody, "camera", {"name": "front", "pos": "0.58 -0.66 0.32", "xyaxes": "0.76 0.65 0 -0.20 0.23 0.95"})
    ET.SubElement(worldbody, "camera", {"name": "side", "pos": "0.78 0.00 0.25", "xyaxes": "0 1 0 -0.18 0 0.98"})
    ET.SubElement(worldbody, "camera", {"name": "longitudinal_end", "pos": "0.70 0.00 0.20", "xyaxes": "0 1 0 -0.16 0 0.99"})
    ET.SubElement(worldbody, "camera", {"name": "top_angle", "pos": "0.38 -0.48 0.58", "xyaxes": "0.78 0.62 0 -0.50 0.63 0.59"})
    ET.SubElement(worldbody, "camera", {"name": "dual", "pos": "1.95 -2.35 1.25", "xyaxes": "0.78 0.63 0 -0.35 0.43 0.83", "fovy": "75"})


def _add_visual_skin(bag: ET.Element, state: ScenarioState) -> None:
    skin = ET.SubElement(bag, "body", {"name": "visual_skin", "pos": "0 0 0"})
    _geom(skin, name="visual_skin_main", type="mesh", mesh="sealed_pillow_skin_mesh", material="mat_jute", group="3", contype="0", conaffinity="0", mass="0.001")
    cap_z = 0.5 * SACK_THICKNESS * state.top_crown_scale + 0.006
    cap = ET.SubElement(skin, "body", {"name": "sealed_top_cap_visual", "pos": f"0 0 {cap_z:.6f}"})
    _geom(cap, name="sealed_top_cap_visual_geom", type="box", size=f"{0.28*SACK_LENGTH:.6f} {0.030*SACK_WIDTH:.6f} 0.0025", group="3", rgba="0.50 0.34 0.16 0.18", contype="0", conaffinity="0", mass="0.001")
    # 길이 방향 양 끝이 열린 단면처럼 보이지 않도록 visual-only end cap을 둡니다.
    # 물리 계산에는 관여하지 않지만 outer_shell_only 렌더에는 포함됩니다.
    # 길이 방향 끝단을 실제 외피 끝까지 닫아야 side/end view에서 빈 공간처럼 보이지 않습니다.
    end_cap_x = 0.50 * SACK_LENGTH
    for cap_name, x_sign in (("left_end_cap_visual", -1.0), ("right_end_cap_visual", 1.0)):
        end_cap = ET.SubElement(skin, "body", {"name": cap_name, "pos": f"{x_sign * end_cap_x:.6f} 0 0"})
        _geom(
            end_cap,
            name=f"{cap_name}_geom",
            type="mesh",
            mesh="longitudinal_end_cap_mesh",
            material="mat_side_panel",
            group="1",
            rgba="0.72 0.70 0.95 0.92",
            contype="0",
            conaffinity="0",
            mass="0.001",
        )
        ET.SubElement(end_cap, "site", {"name": f"site_{cap_name}", "pos": "0 0 0", "size": "0.0008", "rgba": "1 0.9 0.2 0.00"})
    # 프린트/봉합 cue는 물리에 쓰지 않고 top opening처럼 보이는 것을 막는 시각 요소입니다.
    mark = ET.SubElement(skin, "body", {"name": "visual_print_mark", "pos": "-0.04 -0.01 0.044", "euler": "0 0 -4"})
    _geom(mark, name="visual_print_mark_geom", type="box", size="0.055 0.007 0.001", group="3", rgba="0.07 0.06 0.05 0.18", contype="0", conaffinity="0", mass="0.001")


def _legacy_add_hinge_locked_outer_shell_11_column(bag: ET.Element, state: ScenarioState) -> None:
    """힌지-잠금형 visible outer shell입니다.

    각 판 geom은 자기 body에 고정되고, body에는 hinge joint만 둡니다.
    따라서 visible shell에서 허용되는 상대 운동은 hinge 각도뿐입니다.
    hinge 위치는 UR 링크처럼 부모 body가 회전/이동하면서 종속적으로 따라옵니다.
    """

    root = ET.SubElement(bag, "body", {"name": "visible_articulated_outer_shell", "pos": "0 0 0"})
    top_z = 0.5 * SACK_THICKNESS * state.top_crown_scale + 0.016
    # 내부 ballast가 들어갈 공간을 확보하기 위해 포대 폭을 조금 넓게 잡습니다.
    top_y = 0.30 * SACK_WIDTH * state.top_width_scale
    mid_y = 0.46 * SACK_WIDTH * (0.65 * state.top_width_scale + 0.35 * state.lower_width_scale)
    lower_y = 0.61 * SACK_WIDTH * state.lower_width_scale
    upper_h = 0.052 * max(0.72, state.top_crown_scale)
    mid_h = 0.056
    lower_h = 0.052 * (0.90 + 0.10 * state.lower_bulge_scale)
    bottom_h = 0.024
    seg_len = 0.78 * SACK_LENGTH / TOP_SEAM_COUNT
    start_x = -0.39 * SACK_LENGTH + 0.5 * seg_len
    # 외피 판은 작은 정사각 타일이 아니라 길게 겹치는 직사각 strip으로 보이게 합니다.
    # collision bit는 outer-outer self collision을 피하도록 되어 있어 strip overlap이 수치 발산을 만들지 않습니다.
    outer_strip_half_x = 1.38 * seg_len
    bottom_strip_half_x = 1.28 * seg_len
    shoulder_stiff = max(0.025, state.shoulder_stiffness * 0.075)
    belly_stiff = max(0.035, state.belly_stiffness * 0.055)
    bottom_stiff = 0.030 if state.name == "post_separation_sag" else 0.060

    def _panel_geom(
        parent: ET.Element,
        *,
        name: str,
        pos: tuple[float, float, float],
        size: tuple[float, float, float],
        mass: float,
        material: str = "mat_connected_shell",
    ) -> None:
        _geom(
            parent,
            name=name,
            type="box",
            pos=_fmt(pos),
            size=_fmt(size),
            material=material,
            group="1",
            mass=f"{mass:.4f}",
            contype="2",
            conaffinity="5",
            friction="1.45 0.06 0.006",
            condim="4",
        )

    def _hinged_body(
        parent: ET.Element,
        *,
        body_name: str,
        joint_name: str,
        pos: tuple[float, float, float],
        axis: str,
        range_deg: tuple[float, float],
        springref_deg: float,
        stiffness: float,
        damping: float,
    ) -> ET.Element:
        body = ET.SubElement(parent, "body", {"name": body_name, "pos": _fmt(pos)})
        _joint(
            body,
            name=joint_name,
            type="hinge",
            axis=axis,
            range=f"{range_deg[0]:.3f} {range_deg[1]:.3f}",
            springref=f"{_rad(springref_deg):.6f}",
            stiffness=f"{stiffness:.5f}",
            damping=f"{damping:.5f}",
        )
        return body

    rail = _hinged_body(
        root,
        body_name="top_grasp_rail",
        joint_name="top_grasp_rail_pitch",
        pos=(0.0, 0.0, top_z),
        axis="0 1 0",
        range_deg=(-32.0, 32.0),
        springref_deg=0.0,
        stiffness=0.055,
        damping=0.46,
    )
    _geom(
        rail,
        name="top_grasp_rail_geom",
        type="capsule",
        fromto=f"{-0.405*SACK_LENGTH:.6f} 0 0 {0.405*SACK_LENGTH:.6f} 0 0",
        size="0.0072",
        material="mat_seam",
        group="1",
        mass="0.030",
        contype="2",
        conaffinity="5",
        friction="1.55 0.08 0.008",
        condim="4",
    )
    _hinge_capsule(
        rail,
        name="joint_axis_top_grasp_rail_pitch",
        fromto=f"0 {-0.070*SACK_WIDTH:.6f} 0.012 0 {0.070*SACK_WIDTH:.6f} 0.012",
        size=0.0048,
    )
    _panel_geom(
        rail,
        name="sealed_top_cap_hinge_locked_geom",
        pos=(0.0, 0.0, 0.010),
        size=(0.405 * SACK_LENGTH, 1.06 * top_y, 0.0045),
        mass=0.030,
        material="mat_connected_shell",
    )
    ET.SubElement(rail, "site", {"name": "site_top_grasp_rail_center", "pos": "0 0 0.010", "size": "0.002", "rgba": "1 0.2 0.1 0.65"})

    seam_chain = ET.SubElement(rail, "body", {"name": "top_seam_chain", "pos": "0 0 0"})
    for i in range(TOP_SEAM_COUNT):
        x = start_x + i * seg_len
        seam = _hinged_body(
            seam_chain,
            body_name=f"top_seam_{i:02d}",
            joint_name=f"top_seam_{i:02d}_hinge",
            pos=(x, 0.0, 0.002),
            axis="0 1 0",
            range_deg=(-82.0, 82.0),
            springref_deg=0.0,
            stiffness=0.095,
            damping=0.52,
        )
        _hinge_capsule(
            seam,
            name=f"cyl_hinge_top_seam_{i:02d}",
            fromto=f"{-0.46*seg_len:.6f} 0 0.010 {0.46*seg_len:.6f} 0 0.010",
            size=0.0032,
        )
        _panel_geom(
            seam,
            name=f"top_seam_{i:02d}_geom",
            pos=(0.0, 0.0, 0.004),
            size=(0.48 * seg_len, 0.014, 0.004),
            mass=0.007,
            material="mat_seam",
        )
        ET.SubElement(seam, "site", {"name": f"site_top_seam_{i:02d}", "pos": "0 0 0.012", "size": "0.0005", "rgba": "0.1 0.35 1 0.00"})

    # 좌/우 외피는 top rail에 매달린 종속 hinge chain입니다.
    # 각 column: upper -> mid -> lower -> bottom-edge 순서로 부모-자식 body가 이어집니다.
    for side, ysign in (("left", 1.0), ("right", -1.0)):
        upper_group = ET.SubElement(rail, "body", {"name": f"outer_upper_{side}_segments", "pos": "0 0 0"})
        for i in range(TOP_SEAM_COUNT):
            x = start_x + i * seg_len
            upper = _hinged_body(
                upper_group,
                body_name=f"outer_upper_{side}_{i:02d}",
                joint_name=f"outer_upper_{side}_{i:02d}_hinge",
                pos=(x, ysign * top_y, -0.006),
                axis="1 0 0",
                range_deg=(-98.0, 98.0),
                springref_deg=state.shoulder_rest_deg * ysign,
                stiffness=shoulder_stiff,
                damping=0.42 if state.name == "underfilled" else 0.72,
            )
            _hinge_capsule(
                upper,
                name=f"joint_axis_outer_upper_{side}_{i:02d}",
                fromto=f"{-0.38*seg_len:.6f} 0 0 {0.38*seg_len:.6f} 0 0",
                size=0.0038,
            )
            dy_upper = mid_y - top_y
            _panel_geom(
                upper,
                name=f"outer_upper_{side}_{i:02d}_geom",
                pos=(0.0, ysign * 0.50 * dy_upper, -0.50 * upper_h),
                size=(outer_strip_half_x, max(0.014, 0.50 * abs(dy_upper) + 0.006), 0.50 * upper_h),
                mass=0.020,
            )
            ET.SubElement(upper, "site", {"name": f"site_outer_upper_{side}_{i:02d}", "pos": f"0 {ysign * dy_upper:.6f} {-upper_h:.6f}", "size": "0.0005", "rgba": "0.2 0.8 1 0.00"})
            ET.SubElement(upper, "site", {"name": f"site_outer_shoulder_{side}_{i:02d}", "pos": f"0 {ysign * dy_upper:.6f} {-upper_h:.6f}", "size": "0.0005", "rgba": "0.2 0.8 1 0.00"})

            mid_prefix = "front" if side == "left" else "back"
            mid = _hinged_body(
                upper,
                body_name=f"outer_mid_{mid_prefix}_{i:02d}",
                joint_name=f"outer_mid_{mid_prefix}_{i:02d}_hinge",
                pos=(0.0, ysign * dy_upper, -upper_h),
                axis="1 0 0",
                range_deg=(-104.0, 104.0),
                springref_deg=0.45 * state.shoulder_rest_deg * ysign,
                stiffness=max(0.020, shoulder_stiff * 0.82),
                damping=0.58,
            )
            _hinge_capsule(
                mid,
                name=f"joint_axis_outer_mid_{mid_prefix}_{i:02d}",
                fromto=f"{-0.38*seg_len:.6f} 0 0 {0.38*seg_len:.6f} 0 0",
                size=0.0036,
            )
            dy_mid = lower_y - mid_y
            _panel_geom(
                mid,
                name=f"outer_mid_{mid_prefix}_{i:02d}_geom",
                pos=(0.0, ysign * 0.50 * dy_mid, -0.50 * mid_h),
                size=(outer_strip_half_x, max(0.016, 0.50 * abs(dy_mid) + 0.007), 0.50 * mid_h),
                mass=0.026,
            )
            ET.SubElement(mid, "site", {"name": f"site_outer_mid_{mid_prefix}_{i:02d}", "pos": f"0 {ysign * dy_mid:.6f} {-mid_h:.6f}", "size": "0.0005", "rgba": "0.2 0.8 1 0.00"})

            lower = _hinged_body(
                mid,
                body_name=f"outer_lower_{side}_{i:02d}",
                joint_name=f"outer_lower_{side}_{i:02d}_hinge",
                pos=(0.0, ysign * dy_mid, -mid_h),
                axis="1 0 0",
                range_deg=(-112.0, 112.0),
                springref_deg=state.belly_rest_deg * ysign,
                stiffness=belly_stiff,
                damping=0.54,
            )
            _hinge_capsule(
                lower,
                name=f"joint_axis_outer_lower_{side}_{i:02d}",
                fromto=f"{-0.38*seg_len:.6f} 0 0 {0.38*seg_len:.6f} 0 0",
                size=0.0036,
            )
            _panel_geom(
                lower,
                name=f"outer_lower_{side}_{i:02d}_geom",
                pos=(0.0, ysign * 0.010, -0.50 * lower_h),
                size=(outer_strip_half_x, 0.030 * state.lower_bulge_scale, 0.50 * lower_h),
                mass=0.030,
            )
            ET.SubElement(lower, "site", {"name": f"site_outer_lower_{side}_{i:02d}", "pos": f"0 {ysign * 0.016:.6f} {-lower_h:.6f}", "size": "0.0005", "rgba": "0.2 0.8 1 0.00"})

            bottom = _hinged_body(
                lower,
                body_name=f"outer_bottom_edge_{side}_{i:02d}",
                joint_name=f"outer_bottom_edge_{side}_{i:02d}_hinge",
                pos=(0.0, ysign * 0.012, -lower_h),
                axis="1 0 0",
                range_deg=(-130.0, 130.0),
                springref_deg=-10.0 * ysign if state.name == "post_separation_sag" else 0.0,
                stiffness=bottom_stiff,
                damping=0.32,
            )
            _hinge_capsule(
                bottom,
                name=f"joint_axis_outer_bottom_edge_{side}_{i:02d}",
                fromto=f"{-0.36*seg_len:.6f} 0 0 {0.36*seg_len:.6f} 0 0",
                size=0.0034,
            )
            _panel_geom(
                bottom,
                name=f"outer_bottom_edge_{side}_{i:02d}_geom",
                pos=(0.0, -ysign * 0.50 * lower_y, -0.50 * bottom_h),
                size=(bottom_strip_half_x, 0.72 * lower_y, 0.50 * bottom_h),
                mass=0.024,
            )
            if i == TOP_SEAM_COUNT // 2:
                _panel_geom(
                    bottom,
                    name=f"outer_bottom_closure_{side}_geom",
                    pos=(0.0, -ysign * 0.42 * lower_y, -0.72 * bottom_h),
                    size=(0.42 * SACK_LENGTH, 0.42 * lower_y, 0.006),
                    mass=0.030,
                    material="mat_connected_shell",
                )
            ET.SubElement(bottom, "site", {"name": f"site_outer_bottom_edge_{side}_{i:02d}", "pos": f"0 {-ysign * 0.86 * lower_y:.6f} {-bottom_h:.6f}", "size": "0.0005", "rgba": "1 0.5 0.1 0.00"})

            if side == "left" and i == TOP_SEAM_COUNT // 2:
                bottom_center = _hinged_body(
                    bottom,
                    body_name="outer_bottom_edge_center",
                    joint_name="outer_bottom_edge_center_hinge",
                    pos=(0.0, -0.86 * lower_y, -bottom_h),
                    axis="0 1 0",
                    range_deg=(-95.0, 95.0),
                    springref_deg=-18.0 if state.name == "post_separation_sag" else 0.0,
                    stiffness=bottom_stiff,
                    damping=0.34,
                )
                _hinge_capsule(
                    bottom_center,
                    name="joint_axis_outer_bottom_edge_center",
                    fromto=f"0 {-0.16*SACK_WIDTH:.6f} 0 0 {0.16*SACK_WIDTH:.6f} 0",
                    size=0.0042,
                )
                _panel_geom(
                    bottom_center,
                    name="outer_bottom_edge_center_geom",
                    pos=(0.0, 0.0, -0.50 * bottom_h),
                    size=(0.40 * SACK_LENGTH, 0.20 * SACK_WIDTH * state.lower_width_scale, 0.42 * bottom_h),
                    mass=0.020,
                )
                ET.SubElement(bottom_center, "site", {"name": "site_outer_bottom_edge_center", "pos": "0 0 -0.026", "size": "0.0005", "rgba": "1 0.5 0.1 0.00"})

        ET.SubElement(
            upper_group,
            "site",
            {
                "name": f"site_outer_side_{side}_center",
                "pos": f"0 {ysign * lower_y:.6f} {-upper_h - mid_h - 0.45 * lower_h:.6f}",
                "size": "0.0005",
                "rgba": "0 1 1 0.00",
            },
        )
        center_i = TOP_SEAM_COUNT // 2
        ET.SubElement(
            upper_group,
            "site",
            {
                "name": f"site_outer_lower_{side}_center",
                "pos": f"{start_x + center_i * seg_len:.6f} {ysign * (lower_y + 0.012):.6f} {-upper_h - mid_h - lower_h:.6f}",
                "size": "0.0005",
                "rgba": "0 1 1 0.00",
            },
        )


def _scenario_lateral_bias(state: ScenarioState) -> float:
    return 0.034 if state.name == "eccentric_fill" else 0.0


def _scenario_vertical_bias(state: ScenarioState) -> float:
    return -0.018 if state.name in ("underfilled", "post_separation_sag") else 0.0


def _add_connected_outer_shell(bag: ET.Element, state: ScenarioState) -> None:
    """그림처럼 이어진 3층 외피입니다.

    각 column은 lower -> mid -> upper body가 부모-자식 hinge chain으로 연결됩니다.
    그래서 위쪽을 당기면 child/parent 관계와 tendon coupling 때문에 주변 판 각도가
    같이 변하고, 판의 기준 위치만 고정된 독립 patch처럼 보이지 않습니다.
    """

    root = ET.SubElement(bag, "body", {"name": "connected_outer_shell", "pos": "0 0 0"})
    count = CONNECTED_COLUMN_COUNT
    seg_len = 0.82 * SACK_LENGTH / count
    seg_half = 0.47 * seg_len
    layer_h = 0.315 * SACK_THICKNESS
    z0 = -0.50 * SACK_THICKNESS
    x0 = -0.41 * SACK_LENGTH + 0.5 * seg_len
    lower_half_width = 0.50 * SACK_WIDTH * state.lower_width_scale
    top_half_width = 0.35 * SACK_WIDTH * state.top_width_scale
    inward_step = (lower_half_width - top_half_width) / max(1, CONNECTED_LAYER_COUNT)
    shell_mass = 0.026

    def panel_geom(parent: ET.Element, *, name: str, side_sign: float, layer: str, mass: float = shell_mass) -> None:
        _geom(
            parent,
            name=name,
            type="box",
            pos=f"0 {side_sign * 0.006:.6f} {0.50 * layer_h:.6f}",
            size=f"{seg_half:.6f} 0.008 {0.51 * layer_h:.6f}",
            material="mat_connected_shell",
            group="1",
            mass=f"{mass:.4f}",
            friction="1.45 0.06 0.006",
            condim="4",
        )

    for side, side_sign in (("front", -1.0), ("back", 1.0)):
        side_root = ET.SubElement(root, "body", {"name": f"connected_outer_{side}_shell", "pos": "0 0 0"})
        for i in range(count):
            x = x0 + i * seg_len
            y = side_sign * lower_half_width
            lower = ET.SubElement(side_root, "body", {"name": f"connected_{side}_{i:02d}_lower", "pos": f"{x:.6f} {y:.6f} {z0:.6f}"})
            _joint(lower, name=f"connected_{side}_{i:02d}_lower_hinge", type="hinge", axis="1 0 0", range="-72 72", springref="0", stiffness="0.20", damping="0.85")
            panel_geom(lower, name=f"connected_{side}_{i:02d}_lower_geom", side_sign=side_sign, layer="lower", mass=0.030)
            ET.SubElement(lower, "site", {"name": f"site_connected_{side}_{i:02d}_lower", "pos": f"0 {side_sign * 0.010:.6f} {0.50 * layer_h:.6f}", "size": "0.0005", "rgba": "0.2 0.8 1 0.00"})

            mid = ET.SubElement(lower, "body", {"name": f"connected_{side}_{i:02d}_mid", "pos": f"0 {-side_sign * inward_step:.6f} {layer_h:.6f}"})
            _joint(mid, name=f"connected_{side}_{i:02d}_mid_hinge", type="hinge", axis="1 0 0", range="-86 86", springref="0", stiffness="0.16", damping="0.70")
            panel_geom(mid, name=f"connected_{side}_{i:02d}_mid_geom", side_sign=side_sign, layer="mid", mass=0.026)
            ET.SubElement(mid, "site", {"name": f"site_connected_{side}_{i:02d}_mid", "pos": f"0 {side_sign * 0.010:.6f} {0.50 * layer_h:.6f}", "size": "0.0005", "rgba": "0.2 0.8 1 0.00"})

            upper = ET.SubElement(mid, "body", {"name": f"connected_{side}_{i:02d}_upper", "pos": f"0 {-side_sign * inward_step:.6f} {layer_h:.6f}"})
            _joint(upper, name=f"connected_{side}_{i:02d}_upper_hinge", type="hinge", axis="1 0 0", range="-102 102", springref="0", stiffness="0.11", damping="0.56")
            panel_geom(upper, name=f"connected_{side}_{i:02d}_upper_geom", side_sign=side_sign, layer="upper", mass=0.020)
            ET.SubElement(upper, "site", {"name": f"site_connected_{side}_{i:02d}_upper", "pos": f"0 {side_sign * 0.010:.6f} {0.50 * layer_h:.6f}", "size": "0.0005", "rgba": "0.2 0.8 1 0.00"})

    # 좌우 끝면도 3층으로 올려서 첫 번째 그림 같은 봉합된 다면체 실루엣을 만든다.
    end_half_width = 0.44 * SACK_WIDTH * state.lower_width_scale
    for end, xsign in (("left", -1.0), ("right", 1.0)):
        end_root = ET.SubElement(root, "body", {"name": f"connected_outer_end_{end}", "pos": "0 0 0"})
        x = xsign * 0.43 * SACK_LENGTH
        prev: ET.Element | None = None
        for layer_idx, layer in enumerate(("lower", "mid", "upper")):
            y_shrink = inward_step * layer_idx
            pos = f"{x:.6f} 0 {z0 + layer_idx * layer_h:.6f}" if prev is None else f"{-xsign * 0.012:.6f} 0 {layer_h:.6f}"
            body = ET.SubElement(end_root if prev is None else prev, "body", {"name": f"connected_end_{end}_{layer}", "pos": pos})
            _joint(body, name=f"connected_end_{end}_{layer}_hinge", type="hinge", axis="0 1 0", range="-80 80", springref="0", stiffness="0.15", damping="0.65")
            _geom(
                body,
                name=f"connected_end_{end}_{layer}_geom",
                type="box",
                pos=f"{xsign * 0.004:.6f} 0 {0.50 * layer_h:.6f}",
                size=f"0.008 {max(0.030, end_half_width - y_shrink):.6f} {0.50 * layer_h:.6f}",
                material="mat_connected_shell",
                group="1",
                mass="0.026",
                friction="1.45 0.06 0.006",
                condim="4",
            )
            ET.SubElement(body, "site", {"name": f"site_connected_end_{end}_{layer}", "pos": f"{xsign * 0.008:.6f} 0 {0.50 * layer_h:.6f}", "size": "0.0005", "rgba": "0.2 0.8 1 0.00"})
            prev = body

    # 굵은 파이프가 아니라 봉합선처럼 보이는 얇은 edge cue만 사용한다.
    for side, side_sign in (("front", -1.0), ("back", 1.0)):
        for layer_idx in range(CONNECTED_LAYER_COUNT + 1):
            frac = layer_idx / CONNECTED_LAYER_COUNT
            y = side_sign * (lower_half_width - (lower_half_width - top_half_width) * frac)
            z = z0 + layer_idx * layer_h
            _geom(
                root,
                name=f"connected_edge_{side}_{layer_idx:02d}",
                type="capsule",
                fromto=f"{-0.43 * SACK_LENGTH:.6f} {y:.6f} {z:.6f} {0.43 * SACK_LENGTH:.6f} {y:.6f} {z:.6f}",
                size="0.0032",
                material="mat_connected_edge",
                group="1",
                contype="0",
                conaffinity="0",
                mass="0.001",
            )

    # 바닥부는 고정된 평판이 아니라 center sling과 좌/우 bottom edge가 hinge로 이어진 체인이다.
    # lift 시 top_grasp_rail_lift와 연결된 tendon이 좌/우 edge를 위쪽으로 말아 올린다.
    bottom_root = ET.SubElement(root, "body", {"name": "connected_bottom_shell", "pos": f"0 0 {z0 - 0.004:.6f}"})
    center_len = 0.22 * SACK_LENGTH
    side_len = 0.15 * SACK_LENGTH
    outer_len = 0.13 * SACK_LENGTH
    bottom_width = 0.42 * SACK_WIDTH * state.lower_width_scale

    def bottom_panel(
        parent: ET.Element,
        *,
        name: str,
        pos: str,
        geom_pos: str,
        size: str,
        hinge_range: str,
        stiffness: str,
        damping: str,
        mass: str,
    ) -> ET.Element:
        body = ET.SubElement(parent, "body", {"name": name, "pos": pos})
        _joint(
            body,
            name=f"{name}_hinge",
            type="hinge",
            axis="0 1 0",
            range=hinge_range,
            springref="0",
            stiffness=stiffness,
            damping=damping,
        )
        _geom(
            body,
            name=f"{name}_geom",
            type="box",
            pos=geom_pos,
            size=size,
            material="mat_connected_shell",
            group="1",
            mass=mass,
            friction="1.50 0.07 0.006",
            condim="4",
        )
        return body

    bottom_center = bottom_panel(
        bottom_root,
        name="connected_bottom_center_sling",
        pos="0 0 0",
        geom_pos="0 0 0",
        size=f"{0.50 * center_len:.6f} {bottom_width:.6f} 0.006",
        hinge_range="-34 34",
        stiffness="0.030",
        damping="0.36",
        mass="0.070",
    )
    ET.SubElement(
        bottom_center,
        "site",
        {"name": "site_connected_bottom_center", "pos": "0 0 -0.007", "size": "0.0005", "rgba": "1 0.5 0.1 0.00"},
    )

    for side_name, xsign in (("left", -1.0), ("right", 1.0)):
        inner = bottom_panel(
            bottom_center,
            name=f"connected_bottom_{side_name}_inner",
            pos=f"{xsign * 0.50 * center_len:.6f} 0 0",
            geom_pos=f"{xsign * 0.50 * side_len:.6f} 0 0",
            size=f"{0.50 * side_len:.6f} {0.92 * bottom_width:.6f} 0.0055",
            hinge_range="-96 96",
            stiffness="0.020",
            damping="0.30",
            mass="0.045",
        )
        ET.SubElement(
            inner,
            "site",
            {
                "name": f"site_connected_bottom_{side_name}_inner",
                "pos": f"{xsign * side_len:.6f} 0 -0.006",
                "size": "0.0005",
                "rgba": "1 0.5 0.1 0.00",
            },
        )
        outer = bottom_panel(
            inner,
            name=f"connected_bottom_{side_name}_outer",
            pos=f"{xsign * side_len:.6f} 0 0",
            geom_pos=f"{xsign * 0.50 * outer_len:.6f} 0 0",
            size=f"{0.50 * outer_len:.6f} {0.76 * bottom_width:.6f} 0.005",
            hinge_range="-118 118",
            stiffness="0.016",
            damping="0.24",
            mass="0.034",
        )
        ET.SubElement(
            outer,
            "site",
            {
                "name": f"site_connected_bottom_{side_name}_outer",
                "pos": f"{xsign * outer_len:.6f} 0 -0.005",
                "size": "0.0005",
                "rgba": "1 0.5 0.1 0.00",
            },
        )

    # 상단 cap은 물리 접촉을 크게 만들지 않고 닫힌 자루처럼 읽히는 낮은 visual/physics cue다.
    cap_z = z0 + CONNECTED_LAYER_COUNT * layer_h + 0.004
    _geom(
        root,
        name="connected_top_cap_panel",
        type="box",
        pos=f"0 0 {cap_z:.6f}",
        size=f"{0.39 * SACK_LENGTH:.6f} {0.30 * SACK_WIDTH * state.top_width_scale:.6f} 0.004",
        material="mat_connected_shell",
        group="1",
        contype="0",
        conaffinity="0",
        mass="0.002",
    )


def _add_twin_top_grasp_rail(bag: ET.Element, state: ScenarioState) -> None:
    top_z = 0.5 * SACK_THICKNESS * state.top_crown_scale + 0.012
    rail = ET.SubElement(bag, "body", {"name": "top_grasp_rail", "pos": f"0 0 {top_z:.6f}"})
    # 상단 seam은 bag_frame에 완전히 고정하지 않는다.
    # 2F가 위로 잡아당기면 이 짧은 slide가 먼저 움직이고,
    # tendon coupling을 통해 외피/바닥 panel 각도 변화로 전달된다.
    _joint(
        rail,
        name="top_grasp_rail_lift",
        type="slide",
        axis="0 0 1",
        range="-0.010 0.095",
        springref="0",
        stiffness="0.22",
        damping="1.30",
    )
    _joint(
        rail,
        name="top_grasp_rail_pitch",
        type="hinge",
        axis="0 1 0",
        range="-18 18",
        springref="0",
        stiffness="0.08",
        damping="0.50",
    )
    _geom(
        rail,
        name="top_grasp_rail_geom",
        type="capsule",
        fromto=f"{-0.38*SACK_LENGTH:.6f} 0 0 {0.38*SACK_LENGTH:.6f} 0 0",
        size="0.0065",
        material="mat_seam",
        group="1",
        mass="0.035",
    )
    ET.SubElement(rail, "site", {"name": "site_top_grasp_rail_center", "pos": "0 0 0.010", "size": "0.002", "rgba": "1 0.2 0.1 0.65"})
    seam_chain = ET.SubElement(rail, "body", {"name": "top_seam_chain", "pos": "0 0 0"})
    seg_len = 0.72 * SACK_LENGTH / TOP_SEAM_COUNT
    start_x = -0.36 * SACK_LENGTH + 0.5 * seg_len
    for i in range(TOP_SEAM_COUNT):
        x = start_x + i * seg_len
        body = ET.SubElement(seam_chain, "body", {"name": f"top_seam_{i:02d}", "pos": f"{x:.6f} 0 0.002000"})
        _joint(body, name=f"top_seam_{i:02d}_hinge", type="hinge", axis="0 1 0", range="-70 70", springref="0", stiffness="0.14", damping="0.65")
        _hinge_capsule(
            body,
            name=f"cyl_hinge_top_seam_{i:02d}",
            fromto=f"{-0.46*seg_len:.6f} 0 0.010 {0.46*seg_len:.6f} 0 0.010",
            size=0.0032,
        )
        _geom(body, name=f"top_seam_{i:02d}_geom", type="box", size=f"{0.48*seg_len:.6f} 0.012 0.004", material="mat_seam", group="1", mass="0.008")
        ET.SubElement(body, "site", {"name": f"site_top_seam_{i:02d}", "pos": "0 0 0.010", "size": "0.0005", "rgba": "0.1 0.35 1 0.00"})


def _add_twin_outer_front_back_shell(bag: ET.Element, state: ScenarioState, *, side: str) -> None:
    count = OUTER_FRONT_COUNT if side == "front" else OUTER_BACK_COUNT
    ysign = -1.0 if side == "front" else 1.0
    group = ET.SubElement(bag, "body", {"name": f"outer_{side}_segments", "pos": "0 0 0"})
    y = ysign * 0.48 * SACK_WIDTH * state.lower_width_scale
    z = -0.004 + 0.006 * (state.top_crown_scale - 1.0)
    for i in range(count):
        frac = (i + 0.5) / count - 0.5
        x = frac * 0.82 * SACK_LENGTH
        body = ET.SubElement(group, "body", {"name": f"outer_{side}_shell_{i:02d}", "pos": f"{x:.6f} {y:.6f} {z:.6f}", "euler": f"{-6.0 * ysign:.6f} 0 0"})
        _joint(body, name=f"outer_{side}_shell_{i:02d}_hinge", type="hinge", axis="1 0 0", range="-74 74", springref=f"{_rad(-3.0 * ysign):.6f}", stiffness="0.38", damping="1.45")
        _hinge_capsule(
            body,
            name=f"cyl_hinge_outer_{side}_shell_{i:02d}_top",
            fromto=f"-0.040 {ysign * 0.014:.6f} 0.030 0.040 {ysign * 0.014:.6f} 0.030",
            size=0.0036,
        )
        _hinge_capsule(
            body,
            name=f"cyl_hinge_outer_{side}_shell_{i:02d}_bottom",
            fromto=f"-0.040 {ysign * 0.022:.6f} -0.050 0.040 {ysign * 0.022:.6f} -0.050",
            size=0.0034,
        )
        _geom(body, name=f"outer_{side}_shell_{i:02d}_geom", type="box", pos=f"0 {ysign * 0.020:.6f} -0.016", size="0.039 0.008 0.044", material="mat_panel_hidden", group="5", contype="0", conaffinity="0", mass="0.030")
        ET.SubElement(body, "site", {"name": f"site_outer_{side}_shell_{i:02d}", "pos": f"0 {ysign * 0.024:.6f} -0.048", "size": "0.0005", "rgba": "0 0.7 0.9 0.00"})


def _add_twin_outer_shoulder_shell(bag: ET.Element, state: ScenarioState) -> None:
    group = ET.SubElement(bag, "body", {"name": "outer_shoulder_segments", "pos": "0 0 0"})
    shoulder_stiff = max(0.030, state.shoulder_stiffness * 0.095)
    shoulder_damping = 0.38 if state.name == "underfilled" else (2.10 if state.name == "baseline_filled" else 0.90)
    shoulder_range = 122 if state.name == "underfilled" else 94
    for side, ysign in (("left", 1.0), ("right", -1.0)):
        side_group = ET.SubElement(group, "body", {"name": f"outer_shoulder_shell_{side}", "pos": "0 0 0"})
        for i in range(OUTER_SHOULDER_COUNT):
            frac = (i + 0.5) / OUTER_SHOULDER_COUNT - 0.5
            x = frac * 0.76 * SACK_LENGTH
            y = ysign * 0.32 * SACK_WIDTH * state.top_width_scale
            z = 0.026 * state.top_crown_scale
            body = ET.SubElement(side_group, "body", {"name": f"outer_shoulder_{side}_{i:02d}", "pos": f"{x:.6f} {y:.6f} {z:.6f}", "euler": f"{state.shoulder_rest_deg * ysign:.6f} 0 0"})
            _joint(body, name=f"outer_shoulder_{side}_{i:02d}_hinge", type="hinge", axis="1 0 0", range=f"{-shoulder_range:.1f} {shoulder_range:.1f}", springref=f"{_rad(state.shoulder_rest_deg):.6f}", stiffness=f"{shoulder_stiff:.3f}", damping=f"{shoulder_damping:.3f}")
            _hinge_capsule(
                body,
                name=f"cyl_hinge_outer_shoulder_{side}_{i:02d}",
                fromto="-0.042 0 0.000 0.042 0 0.000",
                size=0.0042,
            )
            _geom(body, name=f"outer_shoulder_{side}_{i:02d}_geom", type="box", pos=f"0 {ysign * 0.023:.6f} -0.008", size="0.040 0.033 0.008", material="mat_panel_hidden", group="5", contype="0", conaffinity="0", mass="0.025")
            ET.SubElement(body, "site", {"name": f"site_outer_shoulder_{side}_{i:02d}", "pos": f"0 {ysign * 0.031:.6f} -0.012", "size": "0.0005", "rgba": "0.1 0.5 1 0.00"})


def _add_twin_outer_side_shell(bag: ET.Element, state: ScenarioState) -> None:
    for side, ysign in (("left", 1.0), ("right", -1.0)):
        group = ET.SubElement(bag, "body", {"name": f"outer_side_shell_segments_{side}", "pos": "0 0 0"})
        y = ysign * 0.52 * SACK_WIDTH * state.lower_width_scale
        for i in range(OUTER_SIDE_COUNT):
            frac = (i + 0.5) / OUTER_SIDE_COUNT - 0.5
            x = frac * 0.68 * SACK_LENGTH
            body = ET.SubElement(group, "body", {"name": f"outer_side_{side}_{i:02d}", "pos": f"{x:.6f} {y:.6f} {-0.008:.6f}", "euler": f"{ysign * -8.0:.6f} 0 0"})
            _joint(body, name=f"outer_side_{side}_{i:02d}_hinge", type="hinge", axis="1 0 0", range="-78 78", springref=f"{_rad(ysign * -5.0):.6f}", stiffness="0.28", damping="1.25")
            _hinge_capsule(
                body,
                name=f"cyl_hinge_outer_side_{side}_{i:02d}",
                fromto="-0.050 0 0.028 0.050 0 0.028",
                size=0.0038,
            )
            _geom(body, name=f"outer_side_{side}_{i:02d}_geom", type="box", pos=f"0 {ysign * 0.012:.6f} -0.010", size="0.048 0.022 0.040", material="mat_panel_hidden", group="5", contype="0", conaffinity="0", mass="0.026")
            ET.SubElement(body, "site", {"name": f"site_outer_side_{side}_{i:02d}", "pos": f"0 {ysign * 0.026:.6f} -0.010", "size": "0.0005", "rgba": "0.1 0.5 1 0.00"})
        ET.SubElement(group, "site", {"name": f"site_outer_side_{side}_center", "pos": f"0 {y:.6f} -0.008", "size": "0.0005", "rgba": "0 1 1 0.00"})


def _add_twin_outer_lower_shell(bag: ET.Element, state: ScenarioState) -> None:
    lower_root = ET.SubElement(bag, "body", {"name": "outer_lower_belly_segments", "pos": "0 0 0"})
    belly_stiff = max(0.08, state.belly_stiffness * 0.09)
    for i in range(OUTER_LOWER_COUNT):
        frac = (i + 0.5) / OUTER_LOWER_COUNT - 0.5
        body = ET.SubElement(lower_root, "body", {"name": f"outer_lower_shell_{i:02d}", "pos": f"{frac * 0.58 * SACK_LENGTH:.6f} 0 {-0.038:.6f}", "euler": f"{state.belly_rest_deg:.6f} 0 0"})
        _joint(body, name=f"outer_lower_shell_{i:02d}_hinge", type="hinge", axis="0 1 0", range="-104 104", springref=f"{_rad(state.belly_rest_deg):.6f}", stiffness=f"{belly_stiff:.3f}", damping="0.72")
        _hinge_capsule(
            body,
            name=f"cyl_hinge_outer_lower_shell_{i:02d}",
            fromto="-0.035 -0.060 0.000 0.035 -0.060 0.000",
            size=0.0038,
        )
        _geom(body, name=f"outer_lower_shell_{i:02d}_geom", type="box", pos="0 0 -0.018", size="0.033 0.058 0.008", material="mat_panel_hidden", group="5", contype="0", conaffinity="0", mass="0.030")
        ET.SubElement(body, "site", {"name": f"site_outer_lower_shell_{i:02d}", "pos": "0 0 -0.034", "size": "0.0005", "rgba": "0 0.7 0.9 0.00"})


def _add_twin_outer_bottom_edge(bag: ET.Element, state: ScenarioState) -> None:
    group = ET.SubElement(bag, "body", {"name": "outer_bottom_edge_segments", "pos": "0 0 0"})
    bottom_rest_deg = -18.0 if state.name == "post_separation_sag" else 0.0
    for i in range(OUTER_BOTTOM_EDGE_COUNT):
        frac = (i + 0.5) / OUTER_BOTTOM_EDGE_COUNT - 0.5
        body = ET.SubElement(group, "body", {"name": f"outer_bottom_edge_{i:02d}", "pos": f"{frac * 0.62 * SACK_LENGTH:.6f} 0 {-0.056:.6f}"})
        _joint(body, name=f"outer_bottom_edge_{i:02d}_hinge", type="hinge", axis="0 1 0", range="-92 92", springref=f"{_rad(bottom_rest_deg):.6f}", stiffness="0.12", damping="0.95")
        _geom(body, name=f"outer_bottom_edge_{i:02d}_geom", type="capsule", fromto="-0.028 0 0 0.028 0 0", size="0.007", material="mat_panel_hidden", group="5", contype="0", conaffinity="0", mass="0.024")
        ET.SubElement(body, "site", {"name": f"site_outer_bottom_edge_{i:02d}", "pos": "0 0 -0.008", "size": "0.0005", "rgba": "1 0.3 0.1 0.00"})
    ET.SubElement(group, "site", {"name": "site_outer_bottom_edge_center", "pos": "0 0 -0.064", "size": "0.0005", "rgba": "1 0.3 0.1 0.00"})


def _legacy_add_twin_inner_load_shell_strip(bag: ET.Element, state: ScenarioState) -> None:
    root = ET.SubElement(bag, "body", {"name": "hidden_inner_load_shell", "pos": f"0 {_scenario_lateral_bias(state):.6f} {_scenario_vertical_bias(state):.6f}"})
    for side_name, ysign in (("front", -1.0), ("back", 1.0)):
        group = ET.SubElement(root, "body", {"name": f"inner_{side_name}_load_shell", "pos": "0 0 0"})
        for i in range(INNER_LOAD_PANEL_COUNT):
            frac = (i + 0.5) / INNER_LOAD_PANEL_COUNT - 0.5
            body = ET.SubElement(group, "body", {"name": f"inner_{side_name}_load_{i:02d}", "pos": f"{frac * 0.56 * SACK_LENGTH:.6f} {ysign * 0.060:.6f} {-0.012:.6f}", "euler": f"{ysign * -4.0:.6f} 0 0"})
            _joint(body, name=f"inner_{side_name}_load_{i:02d}_hinge", type="hinge", axis="1 0 0", range="-70 70", springref=f"{_rad(ysign * -3.0):.6f}", stiffness="0.20", damping="0.95")
            _geom(body, name=f"inner_{side_name}_load_{i:02d}_geom", type="box", pos=f"0 {ysign * 0.008:.6f} -0.008", size="0.038 0.010 0.032", material="mat_inner_shell", group="2", contype="0", conaffinity="0", mass="0.040")
            ET.SubElement(body, "site", {"name": f"site_inner_{side_name}_load_{i:02d}", "pos": f"0 {ysign * 0.010:.6f} -0.030", "size": "0.0005", "rgba": "0 0.3 1 0.00"})
    bottom = ET.SubElement(root, "body", {"name": "inner_bottom_load_shell", "pos": "0 0 0"})
    inner_bottom_rest_deg = -12.0 if state.name == "post_separation_sag" else 0.0
    for i in range(INNER_BOTTOM_PANEL_COUNT):
        frac = (i + 0.5) / INNER_BOTTOM_PANEL_COUNT - 0.5
        body = ET.SubElement(bottom, "body", {"name": f"inner_bottom_load_{i:02d}", "pos": f"{frac * 0.44 * SACK_LENGTH:.6f} 0 {-0.060:.6f}"})
        _joint(body, name=f"inner_bottom_load_{i:02d}_hinge", type="hinge", axis="0 1 0", range="-96 96", springref=f"{_rad(inner_bottom_rest_deg):.6f}", stiffness="0.065", damping="0.62")
        _geom(body, name=f"inner_bottom_load_{i:02d}_geom", type="box", pos="0 0 -0.006", size="0.060 0.050 0.009", material="mat_inner_shell", group="2", contype="0", conaffinity="0", mass="0.070")
        ET.SubElement(body, "site", {"name": f"site_inner_bottom_load_{i:02d}", "pos": "0 0 -0.014", "size": "0.0005", "rgba": "0 0.3 1 0.00"})


def _add_twin_ballast_masses(bag: ET.Element, state: ScenarioState) -> None:
    fill, _height_scale = _fill_geometry_scale(state)
    size_scale = max(0.12, fill ** (1.0 / 3.0))
    mass_scale = max(0.025, state.ballast_mass_scale)
    ballast_alpha = 0.34 if fill > 0.15 else 0.07
    y_bias = 0.050 if state.name == "eccentric_fill" else 0.0
    z_bias = -0.038 if state.name == "empty_collapsed" else (-0.026 if state.name in ("underfilled", "post_separation_sag") else -0.006)
    main_x, main_y, main_z = state.payload_main_pos
    aux_x, aux_y, aux_z = state.payload_aux_pos
    specs = (
        ("ballast_main", (main_x - 0.070, main_y + 0.010, main_z + z_bias), (0.060, 0.034, 0.017), 0.16),
        ("ballast_aux_1", (aux_x + 0.040, aux_y + y_bias, aux_z + z_bias), (0.056, 0.033, 0.016), max(state.payload_aux_mass * 0.55, 0.07)),
        ("ballast_aux_2", (-0.004, -0.034 + 0.4 * y_bias, -0.021 + z_bias), (0.052, 0.032, 0.015), 0.11),
        ("ballast_aux_3", (0.092, 0.024 + 0.7 * y_bias, -0.024 + z_bias), (0.050, 0.030, 0.014), 0.10),
    )
    for name, pos, size, mass in specs:
        body = ET.SubElement(bag, "body", {"name": name, "pos": _fmt(pos)})
        for axis_name, axis, limit in (("x", "1 0 0", "-0.110 0.110"), ("y", "0 1 0", "-0.135 0.135"), ("z", "0 0 1", "-0.105 0.045")):
            _joint(body, name=f"{name}_{axis_name}", type="slide", axis=axis, range=limit, springref="0", stiffness="0.26", damping="2.4")
        scaled_size = tuple(max(0.004, value * size_scale) for value in size)
        _geom(
            body,
            name=f"{name}_geom",
            type="ellipsoid",
            size=_fmt(scaled_size),
            material="mat_ballast",
            rgba=f"0.82 0.13 0.10 {ballast_alpha:.3f}",
            group="4",
            mass=f"{max(0.002, mass * mass_scale):.4f}",
        )
        ET.SubElement(body, "site", {"name": f"site_{name}", "pos": "0 0 0", "size": "0.001", "rgba": "0.6 0 0.6 0.00"})


def _add_twin_optional_top_edge_occlusion(bag: ET.Element, state: ScenarioState) -> None:
    root = ET.SubElement(bag, "body", {"name": "optional_top_edge_occlusion_patch", "pos": "0 0 0"})
    specs = (
        ("top_edge_occlusion_left", 0.18 * SACK_LENGTH, 0.014, state.fold_left_deg),
        ("top_edge_occlusion_right", -0.18 * SACK_LENGTH, -0.014, state.fold_right_deg),
    )
    for name, x, y, deg in specs:
        z = 0.5 * SACK_THICKNESS * state.top_crown_scale + 0.020
        body = ET.SubElement(root, "body", {"name": name, "pos": f"{x:.6f} {y:.6f} {z:.6f}", "euler": f"0 0 {deg:.6f}"})
        _joint(body, name=f"{name}_hinge", type="hinge", axis="0 1 0", range="-105 75", springref=f"{_rad(deg):.6f}", stiffness="0.18", damping="1.20")
        alpha = 0.78 if abs(deg) > 1e-6 else 0.0
        _geom(body, name=f"{name}_geom", type="box", pos="0 0 -0.002", size="0.075 0.032 0.004", material="mat_fold", rgba=f"0.61 0.42 0.20 {alpha:.3f}", group="1", mass="0.010")
        ET.SubElement(body, "site", {"name": f"site_{name}", "pos": "0 0 0.004", "size": "0.0005", "rgba": "0.2 0.2 1 0.00"})


def _add_neighbors_and_support(worldbody: ET.Element, state: ScenarioState) -> None:
    gap = state.neighbor_gap
    for name, y in (("neighbor_left", gap), ("neighbor_right", -gap)):
        body = ET.SubElement(worldbody, "body", {"name": name, "pos": f"0 {y:.6f} {SACK_Z:.6f}"})
        _geom(
            body,
            name=f"{name}_geom",
            type="box",
            size=f"{0.45*SACK_LENGTH:.6f} 0.030 {0.44*SACK_THICKNESS:.6f}",
            group="0" if state.neighbors_active else "1",
            rgba="0.60 0.48 0.30 0.42" if state.neighbors_active else "0.60 0.48 0.30 0.00",
            contype="1" if state.neighbors_active else "0",
            conaffinity="1" if state.neighbors_active else "0",
            friction="1.45 0.05 0.005",
        )
    support = ET.SubElement(worldbody, "body", {"name": "hidden_support", "pos": f"0 0 {SACK_Z - 0.052:.6f}"})
    _geom(
        support,
        name="hidden_support_geom",
        type="box",
        size=f"{0.33*SACK_LENGTH:.6f} {0.30*SACK_WIDTH:.6f} 0.010",
        rgba="0.1 0.6 1 0.20" if state.hidden_support_active else "0.1 0.6 1 0.00",
        contype="1" if state.hidden_support_active else "0",
        conaffinity="1" if state.hidden_support_active else "0",
        friction="0.8 0.03 0.003",
    )


def _add_hinge_locked_outer_shell(bag: ET.Element, state: ScenarioState) -> None:
    """5-slice cross-section 기반 quasi-rigid visible outer shell입니다.

    기존 11-column strip shell을 쓰지 않고, attached diagram처럼 길이 방향 5개 단면을
    top / upper-left / upper-right / lower-left / lower-right / bottom panel로 구성합니다.
    """

    root = ET.SubElement(bag, "body", {"name": "visible_articulated_outer_shell", "pos": "0 0 0"})
    slice_labels = ("left_end", "left_mid", "center", "right_mid", "right_end")
    candidate_labels = ("left", "left_center", "center", "right_center", "right")
    slice_count = TOP_SEAM_COUNT
    slice_span = 0.80 * SACK_LENGTH
    slice_dx = slice_span / max(1, slice_count - 1)
    slice_half_x = 0.60 * slice_dx
    fill, height_scale = _fill_geometry_scale(state)
    top_z = 0.018 + 0.5 * SACK_THICKNESS * state.top_crown_scale * height_scale
    # 앞/뒤 파란 판은 top seam에서 하단 edge로 사선으로 내려가야 상자처럼 보이지 않습니다.
    top_y = (0.18 + 0.06 * fill) * SACK_WIDTH * state.top_width_scale
    lower_y = (0.26 + 0.20 * fill) * SACK_WIDTH * state.lower_width_scale
    upper_h = 0.138 * (0.92 + 0.08 * state.lower_bulge_scale) * height_scale
    lower_h = 0.004
    shoulder_stiff = max(0.006, state.shoulder_stiffness * 0.070)
    lower_stiff = max(0.020, state.belly_stiffness * 0.032)
    bottom_stiff = 0.026 if state.name == "post_separation_sag" else 0.052

    def panel(
        parent: ET.Element,
        *,
        name: str,
        pos: tuple[float, float, float],
        size: tuple[float, float, float],
        mass: float,
        material: str = "mat_connected_shell",
        euler: tuple[float, float, float] | None = None,
    ) -> None:
        attrib = {
            "name": name,
            "type": "box",
            "pos": _fmt(pos),
            "size": _fmt(size),
            "material": material,
            "group": "1",
            "mass": f"{mass:.4f}",
            "contype": "2",
            "conaffinity": "5",
            "friction": "1.45 0.06 0.006",
            "condim": "4",
        }
        if euler is not None:
            attrib["euler"] = _fmt(euler)
        _geom(parent, **attrib)

    def sloped_panel(
        parent: ET.Element,
        *,
        name: str,
        signed_dy: float,
        dz_down: float,
        half_x: float,
        thickness: float,
        mass: float,
        material: str = "mat_connected_shell",
    ) -> None:
        length = math.sqrt(signed_dy * signed_dy + dz_down * dz_down)
        angle_deg = math.degrees(math.atan2(signed_dy, dz_down))
        panel(
            parent,
            name=name,
            pos=(0.0, 0.5 * signed_dy, -0.5 * dz_down),
            size=(half_x, thickness, 0.5 * length),
            mass=mass,
            material=material,
            euler=(angle_deg, 0.0, 0.0),
        )

    def hinge_body(
        parent: ET.Element,
        *,
        body_name: str,
        joint_name: str,
        pos: tuple[float, float, float],
        axis: str,
        range_deg: tuple[float, float],
        springref_deg: float,
        stiffness: float,
        damping: float,
    ) -> ET.Element:
        body = ET.SubElement(parent, "body", {"name": body_name, "pos": _fmt(pos)})
        _joint(
            body,
            name=joint_name,
            type="hinge",
            axis=axis,
            range=f"{range_deg[0]:.3f} {range_deg[1]:.3f}",
            springref=f"{_rad(springref_deg):.6f}",
            stiffness=f"{stiffness:.5f}",
            damping=f"{damping:.5f}",
        )
        return body

    rail = hinge_body(
        root,
        body_name="top_grasp_rail",
        joint_name="top_grasp_rail_pitch",
        pos=(0.0, 0.0, top_z),
        axis="0 1 0",
        range_deg=(-28.0, 28.0),
        springref_deg=0.0,
        stiffness=0.070,
        damping=0.50,
    )
    _geom(
        rail,
        name="top_grasp_rail_geom",
        type="capsule",
        fromto=f"{-0.43*SACK_LENGTH:.6f} 0 0 {0.43*SACK_LENGTH:.6f} 0 0",
        size="0.0080",
        material="mat_seam",
        group="1",
        mass="0.035",
        contype="2",
        conaffinity="5",
        friction="1.55 0.08 0.008",
        condim="4",
    )
    _hinge_capsule(
        rail,
        name="joint_axis_top_grasp_rail_pitch",
        fromto=f"0 {-0.070*SACK_WIDTH:.6f} 0.013 0 {0.070*SACK_WIDTH:.6f} 0.013",
        size=0.0048,
    )
    panel(
        rail,
        name="sealed_top_cap_cross_section_geom",
        pos=(0.0, 0.0, 0.010),
        size=(0.28 * SACK_LENGTH, 0.026, 0.0035),
        mass=0.012,
        material="mat_front_back_panel",
    )
    ET.SubElement(rail, "site", {"name": "site_top_grasp_rail_center", "pos": "0 0 0.012", "size": "0.002", "rgba": "1 0.2 0.1 0.65"})

    for i in range(slice_count):
        x = -0.5 * slice_span + i * slice_dx
        slice_root = ET.SubElement(rail, "body", {"name": f"slice_{i:02d}_{slice_labels[i]}", "pos": f"{x:.6f} 0 0"})
        seam = hinge_body(
            slice_root,
            body_name=f"top_seam_band_{i:02d}",
            joint_name=f"top_seam_band_{i:02d}_hinge",
            pos=(0.0, 0.0, 0.002),
            axis="0 1 0",
            range_deg=(-36.0, 36.0),
            springref_deg=0.0,
            stiffness=0.085,
            damping=0.46,
        )
        _hinge_capsule(
            seam,
            name=f"joint_axis_top_seam_band_{i:02d}",
            fromto=f"{-0.42*slice_dx:.6f} 0 0.010 {0.42*slice_dx:.6f} 0 0.010",
            size=0.0034,
        )
        panel(
            seam,
            name=f"top_seam_band_{i:02d}_geom",
            pos=(0.0, 0.0, 0.004),
            size=(0.52 * slice_dx, top_y, 0.004),
            mass=0.010,
            material="mat_front_back_panel",
        )
        ET.SubElement(seam, "site", {"name": f"site_top_seam_{i:02d}", "pos": "0 0 0.014", "size": "0.0008", "rgba": "0.1 0.35 1 0.00"})
        ET.SubElement(seam, "site", {"name": f"site_top_seam_{candidate_labels[i]}", "pos": "0 0 0.017", "size": "0.0008", "rgba": "0.1 0.35 1 0.00"})

        for side, ysign in (("left", 1.0), ("right", -1.0)):
            upper = hinge_body(
                seam,
                body_name=f"upper_{side}_{i:02d}",
                joint_name=f"upper_{side}_{i:02d}_hinge",
                pos=(0.0, ysign * top_y, -0.004),
                axis="1 0 0",
                range_deg=(-48.0, 48.0),
                springref_deg=state.shoulder_rest_deg * ysign,
                stiffness=shoulder_stiff,
                damping=0.18 if state.name == "underfilled" else 0.68,
            )
            _hinge_capsule(
                upper,
                name=f"joint_axis_upper_{side}_{i:02d}",
                fromto=f"{-0.36*slice_dx:.6f} 0 0 {0.36*slice_dx:.6f} 0 0",
                size=0.0040,
            )
            dy_upper = lower_y - top_y
            sloped_panel(
                upper,
                name=f"upper_{side}_{i:02d}_geom",
                signed_dy=ysign * dy_upper,
                dz_down=upper_h,
                half_x=slice_half_x,
                thickness=0.0085,
                mass=0.052,
                material="mat_front_back_panel",
            )
            ET.SubElement(upper, "site", {"name": f"site_upper_{side}_{i:02d}", "pos": f"0 {ysign * dy_upper:.6f} {-upper_h:.6f}", "size": "0.0007", "rgba": "0.2 0.8 1 0.00"})
            ET.SubElement(upper, "site", {"name": f"site_shoulder_{side}_{i:02d}", "pos": f"0 {ysign * 0.5 * dy_upper:.6f} {-0.5 * upper_h:.6f}", "size": "0.0007", "rgba": "0.2 0.8 1 0.00"})

            lower = hinge_body(
                upper,
                body_name=f"lower_{side}_{i:02d}",
                joint_name=f"lower_{side}_{i:02d}_hinge",
                pos=(0.0, ysign * dy_upper, -upper_h),
                axis="1 0 0",
                range_deg=(-62.0, 62.0),
                springref_deg=state.belly_rest_deg * ysign,
                stiffness=lower_stiff,
                damping=0.44,
            )
            _hinge_capsule(
                lower,
                name=f"joint_axis_lower_{side}_{i:02d}",
                fromto=f"{-0.36*slice_dx:.6f} 0 0 {0.36*slice_dx:.6f} 0 0",
                size=0.0038,
            )
            # lower body는 넓은 옆판이 아니라 하단 조인트/edge connector입니다.
            panel(
                lower,
                name=f"lower_{side}_{i:02d}_geom",
                pos=(0.0, 0.0, 0.0),
                size=(slice_half_x, 0.0060, 0.0045),
                mass=0.018,
                material="mat_connected_edge",
            )
            ET.SubElement(lower, "site", {"name": f"site_lower_{side}_{i:02d}", "pos": "0 0 0", "size": "0.0007", "rgba": "0.2 0.8 1 0.00"})
            ET.SubElement(lower, "site", {"name": f"site_bottom_edge_{side}_{i:02d}", "pos": "0 0 0", "size": "0.0007", "rgba": "1 0.5 0.1 0.00"})

            if side == "left":
                bottom = hinge_body(
                    lower,
                    body_name=f"bottom_{i:02d}",
                    joint_name=f"bottom_{i:02d}_hinge",
                    pos=(0.0, 0.0, 0.0),
                    axis="1 0 0",
                    range_deg=(-70.0, 70.0),
                    springref_deg=-12.0 if state.name == "post_separation_sag" else 0.0,
                    stiffness=bottom_stiff,
                    damping=0.34,
                )
                _hinge_capsule(
                    bottom,
                    name=f"joint_axis_bottom_{i:02d}",
                    fromto=f"{-0.36*slice_dx:.6f} 0 0 {0.36*slice_dx:.6f} 0 0",
                    size=0.0038,
                )
                sloped_panel(
                    bottom,
                name=f"bottom_{i:02d}_geom",
                signed_dy=-2.0 * lower_y,
                dz_down=0.001,
                half_x=slice_half_x,
                thickness=0.0070,
                mass=0.034,
                material="mat_front_back_panel",
            )
                ET.SubElement(bottom, "site", {"name": f"site_bottom_{i:02d}", "pos": f"0 {-lower_y:.6f} -0.001", "size": "0.0007", "rgba": "1 0.5 0.1 0.00"})
                if i == slice_count // 2:
                    ET.SubElement(bottom, "site", {"name": "site_bottom_center", "pos": f"0 {-lower_y:.6f} -0.001", "size": "0.0008", "rgba": "1 0.5 0.1 0.00"})

    # 보라색 옆판은 길이 방향 양 끝에 하나씩 둡니다. 각 판은 Y축 힌지로만 회전합니다.
    for end_name, xsign in (("left", -1.0), ("right", 1.0)):
        side_panel = hinge_body(
            rail,
            body_name=f"side_panel_{end_name}",
            joint_name=f"side_panel_{end_name}_hinge",
            pos=(xsign * 0.5 * slice_span, 0.0, -0.002),
            axis="0 1 0",
            range_deg=(-42.0, 42.0),
            springref_deg=0.0,
            stiffness=0.052,
            damping=0.38,
        )
        _hinge_capsule(
            side_panel,
            name=f"joint_axis_side_panel_{end_name}",
            fromto=f"0 {-lower_y:.6f} 0.008 0 {lower_y:.6f} 0.008",
            size=0.0050,
        )
        panel(
            side_panel,
            name=f"side_panel_{end_name}_geom",
            pos=(0.0, 0.0, -0.5 * upper_h),
            size=(0.010, lower_y, 0.5 * upper_h),
            mass=0.060,
            material="mat_side_panel",
        )
        ET.SubElement(side_panel, "site", {"name": f"site_side_panel_{end_name}_center", "pos": f"0 0 {-0.5 * upper_h:.6f}", "size": "0.0007", "rgba": "0 1 1 0.00"})

    ET.SubElement(root, "site", {"name": "site_side_left_center", "pos": f"{-0.5 * slice_span:.6f} 0 {-0.5 * upper_h:.6f}", "size": "0.0007", "rgba": "0 1 1 0.00"})
    ET.SubElement(root, "site", {"name": "site_side_right_center", "pos": f"{0.5 * slice_span:.6f} 0 {-0.5 * upper_h:.6f}", "size": "0.0007", "rgba": "0 1 1 0.00"})
    ET.SubElement(root, "site", {"name": "site_lower_left_center", "pos": f"0 {lower_y:.6f} {-upper_h:.6f}", "size": "0.0007", "rgba": "0.2 0.8 1 0.00"})
    ET.SubElement(root, "site", {"name": "site_lower_right_center", "pos": f"0 {-lower_y:.6f} {-upper_h:.6f}", "size": "0.0007", "rgba": "0.2 0.8 1 0.00"})


def _add_twin_inner_load_shell(bag: ET.Element, state: ScenarioState) -> None:
    """5-slice hidden inner load shell입니다. 외형이 아니라 하중 경로와 sag를 담당합니다."""
    root = ET.SubElement(bag, "body", {"name": "hidden_inner_load_shell", "pos": f"0 {_scenario_lateral_bias(state):.6f} {_scenario_vertical_bias(state):.6f}"})
    slice_count = TOP_SEAM_COUNT
    slice_span = 0.78 * SACK_LENGTH
    slice_dx = slice_span / max(1, slice_count - 1)
    x0 = -0.5 * slice_span
    inner_upper_rest = -5.0 if state.name == "underfilled" else 0.0
    inner_bottom_rest = -16.0 if state.name == "post_separation_sag" else -4.0
    for i in range(slice_count):
        x = x0 + i * slice_dx
        upper = ET.SubElement(root, "body", {"name": f"inner_upper_{i:02d}", "pos": f"{x:.6f} 0 0.020"})
        _joint(upper, name=f"inner_upper_{i:02d}_hinge", type="hinge", axis="0 1 0", range="-42 42", springref=f"{_rad(inner_upper_rest):.6f}", stiffness="0.095", damping="0.58")
        _geom(upper, name=f"inner_upper_{i:02d}_geom", type="box", pos="0 0 -0.006", size=f"{0.48*slice_dx:.6f} 0.058 0.014", material="mat_inner_shell", group="2", contype="0", conaffinity="0", mass="0.045")
        ET.SubElement(upper, "site", {"name": f"site_inner_upper_{i:02d}", "pos": "0 0 -0.018", "size": "0.0005", "rgba": "0 0.3 1 0.00"})

        lower = ET.SubElement(root, "body", {"name": f"inner_lower_{i:02d}", "pos": f"{x:.6f} 0 -0.026"})
        _joint(lower, name=f"inner_lower_{i:02d}_hinge", type="hinge", axis="0 1 0", range="-58 58", springref="0", stiffness="0.080", damping="0.54")
        _geom(lower, name=f"inner_lower_{i:02d}_geom", type="box", pos="0 0 -0.006", size=f"{0.50*slice_dx:.6f} 0.072 0.018", material="mat_inner_shell", group="2", contype="0", conaffinity="0", mass="0.060")
        ET.SubElement(lower, "site", {"name": f"site_inner_lower_{i:02d}", "pos": "0 0 -0.020", "size": "0.0005", "rgba": "0 0.3 1 0.00"})

        bottom = ET.SubElement(root, "body", {"name": f"inner_bottom_{i:02d}", "pos": f"{x:.6f} 0 -0.072"})
        _joint(bottom, name=f"inner_bottom_{i:02d}_hinge", type="hinge", axis="0 1 0", range="-75 75", springref=f"{_rad(inner_bottom_rest):.6f}", stiffness="0.060", damping="0.46")
        _geom(bottom, name=f"inner_bottom_{i:02d}_geom", type="box", pos="0 0 -0.006", size=f"{0.50*slice_dx:.6f} 0.088 0.012", material="mat_inner_shell", group="2", contype="0", conaffinity="0", mass="0.075")
        ET.SubElement(bottom, "site", {"name": f"site_inner_bottom_{i:02d}", "pos": "0 0 -0.018", "size": "0.0005", "rgba": "0 0.3 1 0.00"})


def _add_bag(worldbody: ET.Element, state: ScenarioState) -> None:
    bag = ET.SubElement(worldbody, "body", {"name": "bag_frame", "pos": f"0 0 {SACK_Z:.6f}", "euler": f"{state.body_tilt_deg:.6f} 0 0"})
    ET.SubElement(bag, "freejoint", {"name": "bag_frame_freejoint"})
    ET.SubElement(bag, "site", {"name": "site_bag_frame", "pos": "0 0 0", "size": "0.0005", "rgba": "1 0 0 0.00"})
    _add_visual_skin(bag, state)
    _add_hinge_locked_outer_shell(bag, state)
    _add_twin_inner_load_shell(bag, state)
    _add_twin_ballast_masses(bag, state)
    _add_twin_optional_top_edge_occlusion(bag, state)
    _add_neighbors_and_support(worldbody, state)


def _legacy_add_strap_tendons_11_column(root: ET.Element) -> None:
    """저차 coordinated shape change를 만드는 약한 strap/tendon surrogate입니다."""

    tendon = ET.SubElement(root, "tendon")

    def fixed(name: str, joints: list[tuple[str, float]], stiffness: float, damping: float) -> None:
        item = ET.SubElement(tendon, "fixed", {"name": name, "springlength": "0", "stiffness": f"{stiffness:.4f}", "damping": f"{damping:.4f}"})
        for joint_name, coef in joints:
            ET.SubElement(item, "joint", {"joint": joint_name, "coef": f"{coef:.4f}"})

    def angle_chain(name: str, joint_names: list[str], stiffness: float = 0.038, damping: float = 0.014) -> None:
        for idx in range(len(joint_names) - 1):
            fixed(
                f"chain_{name}_{idx:02d}",
                [(joint_names[idx], 1.0), (joint_names[idx + 1], -1.0)],
                stiffness,
                damping,
            )

    # 인접 판의 각도를 elastic하게 묶어, 각 판이 따로 놀지 않고 하나의 외피처럼 같이 말리게 한다.
    # Hinge-locked topology 전용 coupling입니다.
    # 여기서 return하기 때문에 아래 legacy connected/rim coupling은 더 이상 XML에 들어가지 않습니다.
    angle_chain("top_seam_angle", [f"top_seam_{i:02d}_hinge" for i in range(TOP_SEAM_COUNT)], 0.050, 0.018)
    angle_chain("inner_front_angle", [f"inner_front_load_{i:02d}_hinge" for i in range(INNER_LOAD_PANEL_COUNT)], 0.030, 0.010)
    angle_chain("inner_back_angle", [f"inner_back_load_{i:02d}_hinge" for i in range(INNER_LOAD_PANEL_COUNT)], 0.030, 0.010)
    angle_chain("inner_bottom_angle", [f"inner_bottom_load_{i:02d}_hinge" for i in range(INNER_BOTTOM_PANEL_COUNT)], 0.032, 0.010)

    for side in ("left", "right"):
        mid_prefix = "front" if side == "left" else "back"
        angle_chain(f"outer_upper_{side}_angle", [f"outer_upper_{side}_{i:02d}_hinge" for i in range(TOP_SEAM_COUNT)], 0.060, 0.018)
        angle_chain(f"outer_mid_{mid_prefix}_angle", [f"outer_mid_{mid_prefix}_{i:02d}_hinge" for i in range(TOP_SEAM_COUNT)], 0.054, 0.016)
        angle_chain(f"outer_lower_{side}_angle", [f"outer_lower_{side}_{i:02d}_hinge" for i in range(TOP_SEAM_COUNT)], 0.050, 0.015)
        angle_chain(f"outer_bottom_edge_{side}_angle", [f"outer_bottom_edge_{side}_{i:02d}_hinge" for i in range(TOP_SEAM_COUNT)], 0.046, 0.014)
        for i in range(TOP_SEAM_COUNT):
            fixed(
                f"couple_top_to_outer_{side}_{i:02d}",
                [
                    (f"top_seam_{i:02d}_hinge", 0.34),
                    (f"outer_upper_{side}_{i:02d}_hinge", -0.28 if side == "left" else 0.28),
                ],
                0.030,
                0.014,
            )
            fixed(
                f"chain_outer_vertical_{side}_{i:02d}_upper_mid",
                [(f"outer_upper_{side}_{i:02d}_hinge", 0.55), (f"outer_mid_{mid_prefix}_{i:02d}_hinge", -0.44)],
                0.072,
                0.020,
            )
            fixed(
                f"chain_outer_vertical_{side}_{i:02d}_mid_lower",
                [(f"outer_mid_{mid_prefix}_{i:02d}_hinge", 0.52), (f"outer_lower_{side}_{i:02d}_hinge", -0.42)],
                0.066,
                0.018,
            )
            fixed(
                f"chain_outer_vertical_{side}_{i:02d}_lower_bottom",
                [(f"outer_lower_{side}_{i:02d}_hinge", 0.46), (f"outer_bottom_edge_{side}_{i:02d}_hinge", -0.38)],
                0.058,
                0.016,
            )

    for i in range(TOP_SEAM_COUNT):
        fixed(f"couple_left_right_upper_{i:02d}", [(f"outer_upper_left_{i:02d}_hinge", 0.34), (f"outer_upper_right_{i:02d}_hinge", 0.34)], 0.020, 0.010)
        fixed(f"couple_left_right_lower_{i:02d}", [(f"outer_lower_left_{i:02d}_hinge", 0.30), (f"outer_lower_right_{i:02d}_hinge", 0.30)], 0.018, 0.010)
        fixed(f"couple_left_right_bottom_edge_{i:02d}", [(f"outer_bottom_edge_left_{i:02d}_hinge", 0.34), (f"outer_bottom_edge_right_{i:02d}_hinge", 0.34)], 0.030, 0.014)
    fixed(
        "couple_bottom_center_to_side_edges",
        [("outer_bottom_edge_center_hinge", 0.62), ("outer_bottom_edge_left_05_hinge", -0.28), ("outer_bottom_edge_right_05_hinge", -0.28)],
        0.040,
        0.016,
    )

    for i in range(INNER_LOAD_PANEL_COUNT):
        outer_i = min(TOP_SEAM_COUNT - 1, round(i * (TOP_SEAM_COUNT - 1) / max(1, INNER_LOAD_PANEL_COUNT - 1)))
        fixed(f"couple_outer_left_to_inner_front_{i:02d}", [(f"outer_mid_front_{outer_i:02d}_hinge", 0.24), (f"inner_front_load_{i:02d}_hinge", -0.34)], 0.013, 0.010)
        fixed(f"couple_outer_right_to_inner_back_{i:02d}", [(f"outer_mid_back_{outer_i:02d}_hinge", 0.24), (f"inner_back_load_{i:02d}_hinge", -0.34)], 0.013, 0.010)

    for i in range(INNER_BOTTOM_PANEL_COUNT):
        outer_i = min(TOP_SEAM_COUNT - 1, round((i + 0.5) * TOP_SEAM_COUNT / INNER_BOTTOM_PANEL_COUNT - 0.5))
        fixed(
            f"couple_outer_bottom_to_inner_bottom_{i:02d}",
            [
                (f"inner_bottom_load_{i:02d}_hinge", 0.40),
                (f"outer_bottom_edge_left_{outer_i:02d}_hinge", -0.26),
                (f"outer_bottom_edge_right_{outer_i:02d}_hinge", 0.26),
            ],
            0.014,
            0.010,
        )

    fixed("couple_bottom_center_to_inner_load", [("outer_bottom_edge_center_hinge", 0.42), ("inner_bottom_load_01_hinge", -0.25), ("ballast_main_z", 0.50)], 0.022, 0.012)
    fixed("couple_ballast_aux1_to_side_bias", [("ballast_aux_1_y", 0.50), ("outer_lower_left_07_hinge", -0.10), ("outer_lower_right_07_hinge", 0.10), ("inner_back_load_03_hinge", -0.13)], 0.012, 0.010)
    fixed("couple_ballast_aux2_to_bottom_left", [("ballast_aux_2_z", 0.62), ("inner_bottom_load_00_hinge", -0.16), ("outer_bottom_edge_left_02_hinge", -0.14)], 0.014, 0.010)
    fixed("couple_ballast_aux3_to_bottom_right", [("ballast_aux_3_z", 0.62), ("inner_bottom_load_02_hinge", -0.16), ("outer_bottom_edge_right_08_hinge", 0.14)], 0.014, 0.010)
    fixed("couple_top_to_bottom_droplet_mode", [("top_seam_05_hinge", 0.18), ("inner_bottom_load_01_hinge", -0.24), ("outer_bottom_edge_center_hinge", -0.30)], 0.018, 0.012)
    fixed("couple_occlusion_left_to_rail", [("top_edge_occlusion_left_hinge", 0.55), ("top_seam_08_hinge", 0.22)], 0.014, 0.012)
    fixed("couple_occlusion_right_to_rail", [("top_edge_occlusion_right_hinge", 0.55), ("top_seam_02_hinge", 0.22)], 0.014, 0.012)
    return

    angle_chain("top_seam_angle", [f"top_seam_{i:02d}_hinge" for i in range(TOP_SEAM_COUNT)], 0.050, 0.018)
    angle_chain("outer_front_angle", [f"outer_front_shell_{i:02d}_hinge" for i in range(OUTER_FRONT_COUNT)], 0.040, 0.014)
    angle_chain("outer_back_angle", [f"outer_back_shell_{i:02d}_hinge" for i in range(OUTER_BACK_COUNT)], 0.040, 0.014)
    angle_chain("outer_shoulder_left_angle", [f"outer_shoulder_left_{i:02d}_hinge" for i in range(OUTER_SHOULDER_COUNT)], 0.045, 0.016)
    angle_chain("outer_shoulder_right_angle", [f"outer_shoulder_right_{i:02d}_hinge" for i in range(OUTER_SHOULDER_COUNT)], 0.045, 0.016)
    angle_chain("outer_side_left_angle", [f"outer_side_left_{i:02d}_hinge" for i in range(OUTER_SIDE_COUNT)], 0.036, 0.012)
    angle_chain("outer_side_right_angle", [f"outer_side_right_{i:02d}_hinge" for i in range(OUTER_SIDE_COUNT)], 0.036, 0.012)
    angle_chain("outer_lower_angle", [f"outer_lower_shell_{i:02d}_hinge" for i in range(OUTER_LOWER_COUNT)], 0.042, 0.014)
    angle_chain("outer_bottom_angle", [f"outer_bottom_edge_{i:02d}_hinge" for i in range(OUTER_BOTTOM_EDGE_COUNT)], 0.040, 0.014)
    angle_chain("inner_front_angle", [f"inner_front_load_{i:02d}_hinge" for i in range(INNER_LOAD_PANEL_COUNT)], 0.030, 0.010)
    angle_chain("inner_back_angle", [f"inner_back_load_{i:02d}_hinge" for i in range(INNER_LOAD_PANEL_COUNT)], 0.030, 0.010)
    angle_chain("inner_bottom_angle", [f"inner_bottom_load_{i:02d}_hinge" for i in range(INNER_BOTTOM_PANEL_COUNT)], 0.032, 0.010)

    # 새 visible connected shell: 세로 3층과 가로 column을 모두 각도 체인으로 묶는다.
    for side in ("front", "back"):
        for layer in ("lower", "mid", "upper"):
            angle_chain(
                f"connected_{side}_{layer}_row",
                [f"connected_{side}_{i:02d}_{layer}_hinge" for i in range(CONNECTED_COLUMN_COUNT)],
                0.070 if layer == "upper" else 0.056,
                0.020,
            )
        for i in range(CONNECTED_COLUMN_COUNT):
            angle_chain(
                f"connected_{side}_{i:02d}_vertical",
                [f"connected_{side}_{i:02d}_lower_hinge", f"connected_{side}_{i:02d}_mid_hinge", f"connected_{side}_{i:02d}_upper_hinge"],
                0.135,
                0.032,
            )
            fixed(
                f"couple_connected_{side}_{i:02d}_top_to_mid",
                [(f"connected_{side}_{i:02d}_upper_hinge", 0.62), (f"connected_{side}_{i:02d}_mid_hinge", -0.38), (f"connected_{side}_{i:02d}_lower_hinge", -0.32)],
                0.105,
                0.028,
            )

    for layer in ("lower", "mid", "upper"):
        angle_chain(
            f"connected_end_left_vertical_{layer}",
            [f"connected_end_left_{layer}_hinge", f"connected_end_right_{layer}_hinge"],
            0.040,
            0.012,
        )
    for end in ("left", "right"):
        angle_chain(
            f"connected_end_{end}_height",
            [f"connected_end_{end}_lower_hinge", f"connected_end_{end}_mid_hinge", f"connected_end_{end}_upper_hinge"],
            0.070,
            0.020,
        )

    # 상단 grasp rail을 들거나 비틀 때 connected shell upper layer가 같이 말리도록 연결한다.
    for i in range(CONNECTED_COLUMN_COUNT):
        seam = min(TOP_SEAM_COUNT - 1, round(i * (TOP_SEAM_COUNT - 1) / max(1, CONNECTED_COLUMN_COUNT - 1)))
        fixed(
            f"couple_top_seam_to_connected_front_{i:02d}",
            [(f"top_seam_{seam:02d}_hinge", 0.52), (f"connected_front_{i:02d}_upper_hinge", -0.30), (f"connected_front_{i:02d}_mid_hinge", -0.20), (f"connected_front_{i:02d}_lower_hinge", -0.12)],
            0.092,
            0.026,
        )
        fixed(
            f"couple_top_seam_to_connected_back_{i:02d}",
            [(f"top_seam_{seam:02d}_hinge", 0.52), (f"connected_back_{i:02d}_upper_hinge", -0.30), (f"connected_back_{i:02d}_mid_hinge", -0.20), (f"connected_back_{i:02d}_lower_hinge", -0.12)],
            0.092,
            0.026,
        )

    # 하부/내용물 쪽 하중이 connected lower shell에도 전달되도록 약하게 묶는다.
    for i in range(CONNECTED_COLUMN_COUNT):
        lower = min(OUTER_LOWER_COUNT - 1, round(i * (OUTER_LOWER_COUNT - 1) / max(1, CONNECTED_COLUMN_COUNT - 1)))
        fixed(
            f"couple_connected_lower_to_old_lower_{i:02d}",
            [(f"connected_front_{i:02d}_lower_hinge", 0.20), (f"connected_back_{i:02d}_lower_hinge", -0.20), (f"outer_lower_shell_{lower:02d}_hinge", -0.18)],
            0.026,
            0.012,
        )

    # top rail을 잡아 올리면 바닥 edge가 직각으로 남지 않고 안쪽/위쪽으로 말리도록 만든다.
    # slide(m)와 hinge(rad)를 섞기 때문에 계수는 "30~50 mm lift -> 약 25~45 deg roll-up" 기준으로 잡았다.
    fixed(
        "couple_top_lift_to_bottom_center_sling",
        [("top_grasp_rail_lift", 1.0), ("connected_bottom_center_sling_hinge", 0.040)],
        80.00,
        4.00,
    )
    fixed(
        "couple_top_lift_to_bottom_left_inner_rollup",
        [("top_grasp_rail_lift", 1.0), ("connected_bottom_left_inner_hinge", -0.060)],
        280.00,
        10.00,
    )
    fixed(
        "couple_top_lift_to_bottom_left_outer_rollup",
        [("top_grasp_rail_lift", 1.0), ("connected_bottom_left_outer_hinge", -0.050)],
        220.00,
        8.00,
    )
    fixed(
        "couple_top_lift_to_bottom_right_inner_rollup",
        [("top_grasp_rail_lift", 1.0), ("connected_bottom_right_inner_hinge", 0.060)],
        280.00,
        10.00,
    )
    fixed(
        "couple_top_lift_to_bottom_right_outer_rollup",
        [("top_grasp_rail_lift", 1.0), ("connected_bottom_right_outer_hinge", 0.050)],
        220.00,
        8.00,
    )
    fixed(
        "couple_bottom_left_inner_outer_follow",
        [("connected_bottom_left_inner_hinge", 0.72), ("connected_bottom_left_outer_hinge", -0.52)],
        12.00,
        0.70,
    )
    fixed(
        "couple_bottom_right_inner_outer_follow",
        [("connected_bottom_right_inner_hinge", 0.72), ("connected_bottom_right_outer_hinge", -0.52)],
        12.00,
        0.70,
    )
    fixed(
        "couple_bottom_center_to_inner_load",
        [("connected_bottom_center_sling_hinge", 0.34), ("inner_bottom_load_01_hinge", -0.22), ("ballast_main_z", 0.55)],
        0.040,
        0.018,
    )
    fixed(
        "couple_bottom_edges_to_inner_load_left",
        [("connected_bottom_left_inner_hinge", 0.22), ("inner_bottom_load_00_hinge", -0.18), ("ballast_aux_2_z", 0.36)],
        0.034,
        0.016,
    )
    fixed(
        "couple_bottom_edges_to_inner_load_right",
        [("connected_bottom_right_inner_hinge", -0.22), ("inner_bottom_load_02_hinge", -0.18), ("ballast_aux_3_z", 0.36)],
        0.034,
        0.016,
    )

    for i in range(OUTER_SHOULDER_COUNT):
        seam = min(TOP_SEAM_COUNT - 1, round(i * (TOP_SEAM_COUNT - 1) / max(1, OUTER_SHOULDER_COUNT - 1)))
        fixed(f"couple_top_rail_to_outer_shoulder_left_{i:02d}", [(f"top_seam_{seam:02d}_hinge", 0.46), (f"outer_shoulder_left_{i:02d}_hinge", -0.36)], 0.018, 0.014)
        fixed(f"couple_top_rail_to_outer_shoulder_right_{i:02d}", [(f"top_seam_{seam:02d}_hinge", 0.46), (f"outer_shoulder_right_{i:02d}_hinge", -0.36)], 0.018, 0.014)

    for i in range(INNER_LOAD_PANEL_COUNT):
        outer_i = min(OUTER_FRONT_COUNT - 1, round(i * (OUTER_FRONT_COUNT - 1) / max(1, INNER_LOAD_PANEL_COUNT - 1)))
        fixed(f"couple_outer_front_to_inner_front_{i:02d}", [(f"outer_front_shell_{outer_i:02d}_hinge", 0.28), (f"inner_front_load_{i:02d}_hinge", -0.42)], 0.012, 0.010)
        fixed(f"couple_outer_back_to_inner_back_{i:02d}", [(f"outer_back_shell_{outer_i:02d}_hinge", 0.28), (f"inner_back_load_{i:02d}_hinge", -0.42)], 0.012, 0.010)

    for i in range(OUTER_LOWER_COUNT):
        shoulder = min(OUTER_SHOULDER_COUNT - 1, round(i * (OUTER_SHOULDER_COUNT - 1) / max(1, OUTER_LOWER_COUNT - 1)))
        fixed(f"couple_shoulder_to_outer_lower_left_{i:02d}", [(f"outer_shoulder_left_{shoulder:02d}_hinge", 0.22), (f"outer_lower_shell_{i:02d}_hinge", -0.36)], 0.012, 0.010)
        fixed(f"couple_shoulder_to_outer_lower_right_{i:02d}", [(f"outer_shoulder_right_{shoulder:02d}_hinge", 0.22), (f"outer_lower_shell_{i:02d}_hinge", 0.36)], 0.012, 0.010)

    for i in range(INNER_BOTTOM_PANEL_COUNT):
        lower = min(OUTER_LOWER_COUNT - 1, round((i + 0.5) * OUTER_LOWER_COUNT / INNER_BOTTOM_PANEL_COUNT - 0.5))
        bottom = min(OUTER_BOTTOM_EDGE_COUNT - 1, round((i + 0.5) * OUTER_BOTTOM_EDGE_COUNT / INNER_BOTTOM_PANEL_COUNT - 0.5))
        fixed(
            f"couple_outer_lower_to_inner_bottom_{i:02d}",
            [(f"inner_bottom_load_{i:02d}_hinge", 0.42), (f"outer_lower_shell_{lower:02d}_hinge", -0.30), (f"outer_bottom_edge_{bottom:02d}_hinge", -0.32)],
            0.010,
            0.008,
        )

    fixed("couple_inner_bottom_to_ballast_main", [("ballast_main_z", 0.80), ("inner_bottom_load_01_hinge", -0.20), ("outer_bottom_edge_03_hinge", -0.16)], 0.020, 0.016)
    fixed("couple_ballast_aux1_to_side_shell", [("ballast_aux_1_y", 0.52), ("outer_side_left_02_hinge", -0.14), ("outer_side_right_02_hinge", 0.14), ("inner_back_load_02_hinge", -0.15)], 0.012, 0.010)
    fixed("couple_ballast_aux2_to_inner_bottom", [("ballast_aux_2_z", 0.66), ("inner_bottom_load_00_hinge", -0.18), ("outer_lower_shell_01_hinge", -0.16)], 0.014, 0.010)
    fixed("couple_ballast_aux3_to_inner_bottom", [("ballast_aux_3_z", 0.64), ("inner_bottom_load_02_hinge", -0.18), ("outer_lower_shell_06_hinge", -0.16)], 0.014, 0.010)
    fixed("couple_top_to_bottom_sag_mode", [("top_seam_05_hinge", 0.18), ("inner_bottom_load_01_hinge", -0.26), ("outer_bottom_edge_03_hinge", -0.34)], 0.016, 0.012)
    fixed("couple_occlusion_left_to_rail", [("top_edge_occlusion_left_hinge", 0.55), ("top_seam_08_hinge", 0.22)], 0.014, 0.012)
    fixed("couple_occlusion_right_to_rail", [("top_edge_occlusion_right_hinge", 0.55), ("top_seam_02_hinge", 0.22)], 0.014, 0.012)


def _add_strap_tendons(root: ET.Element) -> None:
    """5-slice cross-section topology 전용 tendon/coupling입니다."""

    tendon = ET.SubElement(root, "tendon")

    def fixed(name: str, joints: list[tuple[str, float]], stiffness: float, damping: float) -> None:
        item = ET.SubElement(tendon, "fixed", {"name": name, "springlength": "0", "stiffness": f"{stiffness:.4f}", "damping": f"{damping:.4f}"})
        for joint_name, coef in joints:
            ET.SubElement(item, "joint", {"joint": joint_name, "coef": f"{coef:.4f}"})

    def angle_chain(name: str, joint_names: list[str], stiffness: float, damping: float) -> None:
        for idx in range(len(joint_names) - 1):
            fixed(f"chain_{name}_{idx:02d}", [(joint_names[idx], 1.0), (joint_names[idx + 1], -1.0)], stiffness, damping)

    slices = range(TOP_SEAM_COUNT)
    angle_chain("top_seam_band", [f"top_seam_band_{i:02d}_hinge" for i in slices], 0.040, 0.014)
    angle_chain("bottom", [f"bottom_{i:02d}_hinge" for i in slices], 0.036, 0.012)
    angle_chain("inner_upper", [f"inner_upper_{i:02d}_hinge" for i in slices], 0.026, 0.010)
    angle_chain("inner_lower", [f"inner_lower_{i:02d}_hinge" for i in slices], 0.026, 0.010)
    angle_chain("inner_bottom", [f"inner_bottom_{i:02d}_hinge" for i in slices], 0.030, 0.010)

    for side in ("left", "right"):
        angle_chain(f"upper_{side}", [f"upper_{side}_{i:02d}_hinge" for i in slices], 0.040, 0.014)
        angle_chain(f"lower_{side}", [f"lower_{side}_{i:02d}_hinge" for i in slices], 0.038, 0.013)

    for i in slices:
        fixed(
            f"mirror_upper_lr_s{i:02d}",
            [(f"upper_left_{i:02d}_hinge", 0.34), (f"upper_right_{i:02d}_hinge", 0.34)],
            0.018,
            0.010,
        )
        fixed(
            f"mirror_lower_lr_s{i:02d}",
            [(f"lower_left_{i:02d}_hinge", 0.30), (f"lower_right_{i:02d}_hinge", 0.30)],
            0.016,
            0.010,
        )
        fixed(
            f"couple_top_to_upper_left_s{i:02d}",
            [(f"top_seam_band_{i:02d}_hinge", 0.28), (f"upper_left_{i:02d}_hinge", -0.24)],
            0.024,
            0.012,
        )
        fixed(
            f"couple_top_to_upper_right_s{i:02d}",
            [(f"top_seam_band_{i:02d}_hinge", 0.28), (f"upper_right_{i:02d}_hinge", 0.24)],
            0.024,
            0.012,
        )
        fixed(
            f"couple_upper_to_lower_left_s{i:02d}",
            [(f"upper_left_{i:02d}_hinge", 0.46), (f"lower_left_{i:02d}_hinge", -0.34)],
            0.042,
            0.014,
        )
        fixed(
            f"couple_upper_to_lower_right_s{i:02d}",
            [(f"upper_right_{i:02d}_hinge", 0.46), (f"lower_right_{i:02d}_hinge", -0.34)],
            0.042,
            0.014,
        )
        fixed(
            f"bottom_to_lower_left_s{i:02d}",
            [(f"bottom_{i:02d}_hinge", 0.50), (f"lower_left_{i:02d}_hinge", -0.25)],
            0.028,
            0.012,
        )
        fixed(
            f"bottom_to_lower_right_s{i:02d}",
            [(f"bottom_{i:02d}_hinge", 0.50), (f"lower_right_{i:02d}_hinge", 0.25)],
            0.028,
            0.012,
        )
        fixed(
            f"outer_to_inner_upper_s{i:02d}",
            [(f"top_seam_band_{i:02d}_hinge", 0.20), (f"inner_upper_{i:02d}_hinge", -0.30)],
            0.012,
            0.008,
        )
        fixed(
            f"outer_to_inner_lower_s{i:02d}",
            [(f"lower_left_{i:02d}_hinge", 0.16), (f"lower_right_{i:02d}_hinge", -0.16), (f"inner_lower_{i:02d}_hinge", -0.26)],
            0.012,
            0.008,
        )
        fixed(
            f"inner_bottom_to_bottom_panel_s{i:02d}",
            [(f"inner_bottom_{i:02d}_hinge", 0.42), (f"bottom_{i:02d}_hinge", -0.30)],
            0.018,
            0.010,
        )

    center = TOP_SEAM_COUNT // 2
    fixed("couple_ballast_main_to_center_bottom", [("ballast_main_z", 0.56), (f"inner_bottom_{center:02d}_hinge", -0.28), (f"bottom_{center:02d}_hinge", -0.24)], 0.020, 0.012)
    fixed("couple_ballast_aux1_to_side_bias", [("ballast_aux_1_y", 0.48), ("lower_left_04_hinge", -0.12), ("lower_right_04_hinge", 0.12), ("inner_lower_03_hinge", -0.12)], 0.012, 0.010)
    fixed("couple_ballast_aux2_to_left_bottom", [("ballast_aux_2_z", 0.58), ("inner_bottom_01_hinge", -0.18), ("bottom_01_hinge", -0.16)], 0.014, 0.010)
    fixed("couple_ballast_aux3_to_right_bottom", [("ballast_aux_3_z", 0.58), ("inner_bottom_03_hinge", -0.18), ("bottom_03_hinge", -0.16)], 0.014, 0.010)
    fixed("couple_top_to_bottom_droplet_mode", [(f"top_seam_band_{center:02d}_hinge", 0.20), (f"inner_bottom_{center:02d}_hinge", -0.28), (f"bottom_{center:02d}_hinge", -0.30)], 0.018, 0.012)
    fixed("couple_occlusion_left_to_rail", [("top_edge_occlusion_left_hinge", 0.55), ("top_seam_band_03_hinge", 0.22)], 0.014, 0.012)
    fixed("couple_occlusion_right_to_rail", [("top_edge_occlusion_right_hinge", 0.55), ("top_seam_band_01_hinge", 0.22)], 0.014, 0.012)


def _add_shell_loop_closure_equalities(root: ET.Element, state: ScenarioState) -> None:
    """각 slice 단면의 panel 끝점이 벌어지지 않도록 부드러운 폐곡선 제약을 둡니다.

    MuJoCo body hierarchy는 트리 구조라 한 panel을 좌/우 두 부모에 동시에 붙일 수 없습니다.
    그래서 hinge tree로 1차 종속 관계를 만들고, 모든 맞닿는 edge에는 soft connect를 추가합니다.
    """

    equality = ET.SubElement(root, "equality")
    slice_count = TOP_SEAM_COUNT
    slice_span = 0.80 * SACK_LENGTH
    slice_dx = slice_span / max(1, slice_count - 1)
    fill, height_scale = _fill_geometry_scale(state)
    top_z = 0.018 + 0.5 * SACK_THICKNESS * state.top_crown_scale * height_scale
    top_y = (0.18 + 0.06 * fill) * SACK_WIDTH * state.top_width_scale
    lower_y = (0.26 + 0.20 * fill) * SACK_WIDTH * state.lower_width_scale
    upper_h = 0.138 * (0.92 + 0.08 * state.lower_bulge_scale) * height_scale
    seam_joint_z = top_z - 0.002
    lower_joint_z = seam_joint_z - upper_h
    bottom_right_z = lower_joint_z - 0.001
    tilt = math.radians(state.body_tilt_deg)
    c, s = math.cos(tilt), math.sin(tilt)

    def world_anchor(x: float, y: float, z: float) -> str:
        wy = y * c - z * s
        wz = y * s + z * c + SACK_Z
        return f"{x:.6f} {wy:.6f} {wz:.6f}"

    def connect(name: str, body1: str, body2: str, x: float, y: float, z: float, *, solref: str = "0.045 1.0") -> None:
        ET.SubElement(
            equality,
            "connect",
            {
                "name": name,
                "body1": body1,
                "body2": body2,
                "anchor": world_anchor(x, y, z),
                "solref": solref,
                "solimp": "0.86 0.96 0.0015",
            },
        )

    for i in range(slice_count):
        x = -0.5 * slice_span + i * slice_dx
        # top seam과 좌/우 upper panel이 같은 hinge line을 공유하도록 닫습니다.
        connect(f"loop_top_upper_left_{i:02d}", f"top_seam_band_{i:02d}", f"upper_left_{i:02d}", x, top_y, seam_joint_z)
        connect(f"loop_top_upper_right_{i:02d}", f"top_seam_band_{i:02d}", f"upper_right_{i:02d}", x, -top_y, seam_joint_z)

        # 긴 side panel과 하단 edge connector의 접합선입니다.
        connect(f"loop_upper_lower_left_{i:02d}", f"upper_left_{i:02d}", f"lower_left_{i:02d}", x, lower_y, lower_joint_z)
        connect(f"loop_upper_lower_right_{i:02d}", f"upper_right_{i:02d}", f"lower_right_{i:02d}", x, -lower_y, lower_joint_z)

        # bottom panel은 lower_left tree에 매달고, 좌/우 edge 모두 soft closure로 폐곡선화합니다.
        connect(f"loop_lower_bottom_left_{i:02d}", f"lower_left_{i:02d}", f"bottom_{i:02d}", x, lower_y, lower_joint_z, solref="0.035 1.0")
        connect(f"loop_lower_bottom_right_{i:02d}", f"lower_right_{i:02d}", f"bottom_{i:02d}", x, -lower_y, bottom_right_z, solref="0.035 1.0")

    # 길이 방향 양끝의 보라색 side panel도 앞/뒤/상/하 edge에 묶어 열린 끝단을 줄입니다.
    for end_name, i in (("left", 0), ("right", slice_count - 1)):
        x = -0.5 * slice_span + i * slice_dx
        connect(f"loop_side_{end_name}_top_front", f"side_panel_{end_name}", f"upper_left_{i:02d}", x, lower_y, seam_joint_z, solref="0.040 1.0")
        connect(f"loop_side_{end_name}_top_back", f"side_panel_{end_name}", f"upper_right_{i:02d}", x, -lower_y, seam_joint_z, solref="0.040 1.0")
        connect(f"loop_side_{end_name}_bottom_front", f"side_panel_{end_name}", f"lower_left_{i:02d}", x, lower_y, lower_joint_z, solref="0.040 1.0")
        connect(f"loop_side_{end_name}_bottom_back", f"side_panel_{end_name}", f"lower_right_{i:02d}", x, -lower_y, lower_joint_z, solref="0.040 1.0")


def _add_ur5e_joint_marker(parent: ET.Element, *, name: str, radius: float = 0.026) -> None:
    _geom(
        parent,
        name=f"{name}_joint_marker",
        type="sphere",
        size=f"{radius:.6f}",
        material="mat_ur5_joint_marker",
        group="0",
        contype="0",
        conaffinity="0",
        mass="0.001",
    )


def _add_ur5_arm(parent: ET.Element, *, name: str, base_pos: tuple[float, float, float], yaw_deg: float, tool: str) -> None:
    """UR5e에 가까운 6DoF kinematic/visual surrogate입니다.

    arm link는 자루와 직접 충돌하지 않게 하고, 실제 접촉은 tool pad/scoop만 담당합니다.
    이렇게 해야 팔 링크가 자루/다른 로봇과 얽혀 보이거나 수치적으로 끼는 일을 줄일 수 있습니다.
    """
    base = ET.SubElement(parent, "body", {"name": f"{name}_base", "pos": _fmt(base_pos), "euler": f"0 0 {yaw_deg:.6f}"})
    _geom(base, name=f"{name}_base_geom", type="cylinder", size="0.070 0.045", material="mat_ur5_base", group="0", contype="0", conaffinity="0", mass="1.4")

    shoulder = ET.SubElement(base, "body", {"name": f"{name}_shoulder_link", "pos": "0 0 0.145"})
    _joint(shoulder, name=f"{name}_shoulder_pan_joint", type="hinge", axis="0 0 1", range="-360 360", stiffness="0", damping="5.0")
    _add_ur5e_joint_marker(shoulder, name=f"{name}_shoulder_pan", radius=0.034)
    _geom(shoulder, name=f"{name}_shoulder_cylinder", type="cylinder", size="0.045 0.045", material="mat_ur5_link", group="0", contype="0", conaffinity="0", mass="0.35")

    upper = ET.SubElement(shoulder, "body", {"name": f"{name}_upper_arm_link", "pos": "0 0 0"})
    _joint(upper, name=f"{name}_shoulder_lift_joint", type="hinge", axis="0 1 0", range="-180 180", stiffness="0", damping="5.0")
    _add_ur5e_joint_marker(upper, name=f"{name}_shoulder_lift", radius=0.030)
    _geom(upper, name=f"{name}_upper_arm_geom", type="capsule", fromto="0 0 0 0.335 0 0", size="0.024", material="mat_ur5_link", group="0", contype="0", conaffinity="0", mass="0.70")

    forearm = ET.SubElement(upper, "body", {"name": f"{name}_forearm_link", "pos": "0.335 0 0"})
    _joint(forearm, name=f"{name}_elbow_joint", type="hinge", axis="0 1 0", range="-180 180", stiffness="0", damping="5.0")
    _add_ur5e_joint_marker(forearm, name=f"{name}_elbow", radius=0.027)
    _geom(forearm, name=f"{name}_forearm_geom", type="capsule", fromto="0 0 0 0.295 0 0", size="0.021", material="mat_ur5_link", group="0", contype="0", conaffinity="0", mass="0.55")

    wrist1 = ET.SubElement(forearm, "body", {"name": f"{name}_wrist_1_link", "pos": "0.295 0 0"})
    _joint(wrist1, name=f"{name}_wrist_1_joint", type="hinge", axis="0 1 0", range="-360 360", stiffness="0", damping="4.0")
    _add_ur5e_joint_marker(wrist1, name=f"{name}_wrist_1", radius=0.023)
    _geom(wrist1, name=f"{name}_wrist_1_geom", type="capsule", fromto="0 0 0 0.070 0 0", size="0.017", material="mat_ur5_link", group="0", contype="0", conaffinity="0", mass="0.18")

    wrist2 = ET.SubElement(wrist1, "body", {"name": f"{name}_wrist_2_link", "pos": "0.070 0 0"})
    _joint(wrist2, name=f"{name}_wrist_2_joint", type="hinge", axis="1 0 0", range="-360 360", stiffness="0", damping="4.0")
    _add_ur5e_joint_marker(wrist2, name=f"{name}_wrist_2", radius=0.021)
    _geom(wrist2, name=f"{name}_wrist_2_geom", type="capsule", fromto="0 0 0 0.062 0 0", size="0.016", material="mat_ur5_link", group="0", contype="0", conaffinity="0", mass="0.16")

    wrist3 = ET.SubElement(wrist2, "body", {"name": f"{name}_wrist_3_link", "pos": "0.062 0 0"})
    _joint(wrist3, name=f"{name}_wrist_3_joint", type="hinge", axis="0 1 0", range="-360 360", stiffness="0", damping="4.0")
    _add_ur5e_joint_marker(wrist3, name=f"{name}_wrist_3", radius=0.020)
    _geom(wrist3, name=f"{name}_wrist_3_geom", type="capsule", fromto="0 0 0 0.050 0 0", size="0.015", material="mat_ur5_link", group="0", contype="0", conaffinity="0", mass="0.14")

    ee = ET.SubElement(wrist3, "body", {"name": f"{name}_ee", "pos": "0.060 0 0"})
    ET.SubElement(ee, "site", {"name": f"{name}_ee_site", "pos": "0 0 0", "size": "0.006", "rgba": "1 0.2 0.2 0.7"})
    if tool == "robotiq_2f140":
        palm = ET.SubElement(ee, "body", {"name": "robotiq_2f140", "pos": "0 0 0", "euler": "0 0 0"})
        _geom(palm, name="robotiq_2f140_palm", type="box", size="0.030 0.028 0.018", material="mat_gripper", group="0", mass="0.14")
        for side, y, axis, slide_range in (
            ("left", 0.070, "0 1 0", "-0.068 0.000"),
            ("right", -0.070, "0 1 0", "0.000 0.068"),
        ):
            finger = ET.SubElement(palm, "body", {"name": f"robotiq_2f140_finger_{side}", "pos": f"0 {y:.6f} -0.012"})
            _joint(finger, name=f"finger_{side}_slide", type="slide", axis=axis, range=slide_range, stiffness="0", damping="12.0")
            _geom(finger, name=f"robotiq_2f140_{side}_knuckle", type="box", size="0.018 0.010 0.022", material="mat_gripper", group="0", contype="0", conaffinity="0", mass="0.030")
            _geom(finger, name=f"robotiq_2f140_{side}_pad", type="box", pos="0 0 -0.045", size="0.014 0.008 0.042", material="mat_pad", group="0", mass="0.035", friction="2.4 0.10 0.010", condim="4")
            ET.SubElement(finger, "site", {"name": f"finger_{side}_pad_site", "pos": "0 0 -0.052", "size": "0.004", "rgba": "1 0.7 0.1 0.75"})
        ET.SubElement(palm, "site", {"name": "robotiq_2f140_center_site", "pos": "0 0 -0.052", "size": "0.005", "rgba": "0.1 0.9 1 0.75"})
    else:
        scoop = ET.SubElement(ee, "body", {"name": "scoop_tool", "pos": "0 0 -0.012", "euler": "0 0 0"})
        _geom(scoop, name="scoop_plate", type="box", size="0.085 0.060 0.004", material="mat_scoop", group="0", mass="0.10", friction="1.25 0.05 0.004")
        _geom(scoop, name="scoop_support_hull", type="capsule", fromto="-0.074 0 0.010 0.088 0 0.010", size="0.018", material="mat_scoop", group="0", mass="0.08", friction="1.35 0.05 0.004")
        _geom(scoop, name="scoop_back_lip", type="box", pos="-0.083 0 0.020", size="0.004 0.060 0.020", material="mat_scoop", group="0", mass="0.025")
        _geom(scoop, name="scoop_side_lip_left", type="box", pos="0 0.058 0.015", size="0.080 0.004 0.014", material="mat_scoop", group="0", mass="0.018")
        _geom(scoop, name="scoop_side_lip_right", type="box", pos="0 -0.058 0.015", size="0.080 0.004 0.014", material="mat_scoop", group="0", mass="0.018")
        ET.SubElement(scoop, "site", {"name": "scoop_tip_site", "pos": "0.085 0 0.006", "size": "0.005", "rgba": "0.1 1 0.6 0.8"})


def _official_inertial(parent: ET.Element, *, mass: str, pos: str, diaginertia: str, quat: str | None = None) -> None:
    attrib = {"mass": mass, "pos": pos, "diaginertia": diaginertia}
    if quat:
        attrib["quat"] = quat
    ET.SubElement(parent, "inertial", attrib)


def _official_ur5e_mesh(parent: ET.Element, *, mesh: str, material: str) -> None:
    # UR5e 링크 mesh는 시각화용입니다. 자루와의 작업 접촉은 gripper pad와 scoop만 사용합니다.
    _geom(parent, type="mesh", mesh=f"ur5e_{mesh}", material=f"mat_ur5e_{material}", group="0", contype="0", conaffinity="0", mass="0.001")


def _robot_bag_collision_proxy(
    parent: ET.Element,
    *,
    name: str,
    geom_type: str,
    size: str,
    pos: str | None = None,
    fromto: str | None = None,
) -> None:
    """UR5e visual mesh와 별도로, 자루와만 충돌하는 보이지 않는 링크 proxy입니다."""
    attrib = {
        "name": name,
        "type": geom_type,
        "size": size,
        "group": "5",
        "rgba": "1 0.45 0.05 0.03",
        "contype": "4",
        "conaffinity": "0",
        "mass": "0.001",
        "friction": "1.10 0.04 0.004",
        "condim": "4",
    }
    if pos is not None:
        attrib["pos"] = pos
    if fromto is not None:
        attrib["fromto"] = fromto
    _geom(parent, **attrib)


def _official_robotiq_2f140(parent: ET.Element) -> None:
    palm = ET.SubElement(parent, "body", {"name": "robotiq_2f140", "pos": "0 0 0", "euler": "0 0 0"})
    _geom(palm, name="robotiq_2f140_palm", type="box", size="0.035 0.032 0.026", material="mat_gripper", group="0", mass="0.16", contype="4", conaffinity="0")
    _geom(palm, name="robotiq_2f140_coupler", type="cylinder", pos="0 0 -0.020", size="0.026 0.012", material="mat_gripper", group="0", mass="0.06", contype="4", conaffinity="0")
    for side, y, axis, slide_range in (
        ("left", 0.070, "0 1 0", "-0.068 0.000"),
        ("right", -0.070, "0 1 0", "0.000 0.068"),
    ):
        finger = ET.SubElement(palm, "body", {"name": f"robotiq_2f140_finger_{side}", "pos": f"0 {y:.6f} 0.025"})
        _joint(finger, name=f"finger_{side}_slide", type="slide", axis=axis, range=slide_range, stiffness="0", damping="12.0")
        _geom(finger, name=f"robotiq_2f140_{side}_outer_link", type="box", size="0.014 0.010 0.048", material="mat_gripper", group="0", contype="4", conaffinity="0", mass="0.030")
        _geom(finger, name=f"robotiq_2f140_{side}_pad", type="box", pos="0 0 0.050", size="0.012 0.008 0.046", material="mat_pad", group="0", mass="0.035", friction="2.4 0.10 0.010", condim="4", contype="4", conaffinity="0")
        ET.SubElement(finger, "site", {"name": f"finger_{side}_pad_site", "pos": "0 0 0.056", "size": "0.004", "rgba": "1 0.7 0.1 0.75"})
    ET.SubElement(palm, "site", {"name": "robotiq_2f140_center_site", "pos": "0 0 0.082", "size": "0.005", "rgba": "0.1 0.9 1 0.75"})


def _official_scoop_tool(parent: ET.Element) -> None:
    scoop = ET.SubElement(parent, "body", {"name": "scoop_tool", "pos": "0 0 0.045", "euler": "-90 90 0"})
    _geom(scoop, name="scoop_plate", type="box", size="0.085 0.060 0.004", material="mat_scoop", group="0", mass="0.10", friction="1.25 0.05 0.004", contype="4", conaffinity="0")
    _geom(scoop, name="scoop_support_hull", type="capsule", fromto="-0.074 0 0.010 0.088 0 0.010", size="0.018", material="mat_scoop", group="0", mass="0.08", friction="1.35 0.05 0.004", contype="4", conaffinity="0")
    _geom(scoop, name="scoop_back_lip", type="box", pos="-0.083 0 0.020", size="0.004 0.060 0.020", material="mat_scoop", group="0", mass="0.025", contype="4", conaffinity="0")
    _geom(scoop, name="scoop_side_lip_left", type="box", pos="0 0.058 0.015", size="0.080 0.004 0.014", material="mat_scoop", group="0", mass="0.018", contype="4", conaffinity="0")
    _geom(scoop, name="scoop_side_lip_right", type="box", pos="0 -0.058 0.015", size="0.080 0.004 0.014", material="mat_scoop", group="0", mass="0.018", contype="4", conaffinity="0")
    ET.SubElement(scoop, "site", {"name": "scoop_tip_site", "pos": "0.085 0 0.006", "size": "0.005", "rgba": "0.1 1 0.6 0.8"})


def _add_official_ur5e_arm(parent: ET.Element, *, name: str, base_pos: tuple[float, float, float], yaw_deg: float, tool: str) -> None:
    """공식 MuJoCo Menagerie UR5e mesh hierarchy를 사용하는 arm입니다."""
    mount = ET.SubElement(parent, "body", {"name": f"{name}_mount", "pos": _fmt(base_pos), "euler": f"0 0 {yaw_deg:.6f}"})
    base = ET.SubElement(mount, "body", {"name": f"{name}_base", "quat": "0 0 0 -1"})
    _official_inertial(base, mass="4.0", pos="0 0 0", diaginertia="0.00443333156 0.00443333156 0.0072")
    _official_ur5e_mesh(base, mesh="base_0", material="black")
    _official_ur5e_mesh(base, mesh="base_1", material="jointgray")
    _robot_bag_collision_proxy(base, name=f"{name}_base_collision", geom_type="cylinder", pos="0 0 0.045", size="0.074 0.055")

    shoulder = ET.SubElement(base, "body", {"name": f"{name}_shoulder_link", "pos": "0 0 0.163"})
    _official_inertial(shoulder, mass="3.7", pos="0 0 0", diaginertia="0.0102675 0.0102675 0.00666")
    _joint(shoulder, name=f"{name}_shoulder_pan_joint", type="hinge", axis="0 0 1", range="-360 360", stiffness="0", damping="5.0", armature="0.1")
    _official_ur5e_mesh(shoulder, mesh="shoulder_0", material="urblue")
    _official_ur5e_mesh(shoulder, mesh="shoulder_1", material="black")
    _official_ur5e_mesh(shoulder, mesh="shoulder_2", material="jointgray")
    _robot_bag_collision_proxy(shoulder, name=f"{name}_shoulder_collision", geom_type="sphere", pos="0 0 0", size="0.060")

    upper = ET.SubElement(shoulder, "body", {"name": f"{name}_upper_arm_link", "pos": "0 0.138 0", "quat": "1 0 1 0"})
    _official_inertial(upper, mass="8.393", pos="0 0 0.2125", diaginertia="0.133886 0.133886 0.0151074")
    _joint(upper, name=f"{name}_shoulder_lift_joint", type="hinge", axis="0 1 0", range="-360 360", stiffness="0", damping="5.0", armature="0.1")
    _official_ur5e_mesh(upper, mesh="upperarm_0", material="linkgray")
    _official_ur5e_mesh(upper, mesh="upperarm_1", material="black")
    _official_ur5e_mesh(upper, mesh="upperarm_2", material="jointgray")
    _official_ur5e_mesh(upper, mesh="upperarm_3", material="urblue")
    _robot_bag_collision_proxy(upper, name=f"{name}_upper_arm_collision", geom_type="capsule", fromto="0 0 0.025 0 0 0.410", size="0.045")

    forearm = ET.SubElement(upper, "body", {"name": f"{name}_forearm_link", "pos": "0 -0.131 0.425"})
    _official_inertial(forearm, mass="2.275", pos="0 0 0.196", diaginertia="0.0311796 0.0311796 0.004095")
    _joint(forearm, name=f"{name}_elbow_joint", type="hinge", axis="0 1 0", range="-180 180", stiffness="0", damping="5.0", armature="0.1")
    _official_ur5e_mesh(forearm, mesh="forearm_0", material="urblue")
    _official_ur5e_mesh(forearm, mesh="forearm_1", material="linkgray")
    _official_ur5e_mesh(forearm, mesh="forearm_2", material="black")
    _official_ur5e_mesh(forearm, mesh="forearm_3", material="jointgray")
    _robot_bag_collision_proxy(forearm, name=f"{name}_forearm_collision", geom_type="capsule", fromto="0 0 0.025 0 0 0.370", size="0.038")

    wrist1 = ET.SubElement(forearm, "body", {"name": f"{name}_wrist_1_link", "pos": "0 0 0.392", "quat": "1 0 1 0"})
    _official_inertial(wrist1, mass="1.219", pos="0 0.127 0", diaginertia="0.0025599 0.0025599 0.0021942")
    _joint(wrist1, name=f"{name}_wrist_1_joint", type="hinge", axis="0 1 0", range="-360 360", stiffness="0", damping="4.0", armature="0.1")
    _official_ur5e_mesh(wrist1, mesh="wrist1_0", material="black")
    _official_ur5e_mesh(wrist1, mesh="wrist1_1", material="urblue")
    _official_ur5e_mesh(wrist1, mesh="wrist1_2", material="jointgray")
    _robot_bag_collision_proxy(wrist1, name=f"{name}_wrist1_collision", geom_type="sphere", pos="0 0.060 0", size="0.044")

    wrist2 = ET.SubElement(wrist1, "body", {"name": f"{name}_wrist_2_link", "pos": "0 0.127 0"})
    _official_inertial(wrist2, mass="1.219", pos="0 0 0.1", diaginertia="0.0025599 0.0025599 0.0021942")
    _joint(wrist2, name=f"{name}_wrist_2_joint", type="hinge", axis="0 0 1", range="-360 360", stiffness="0", damping="4.0", armature="0.1")
    _official_ur5e_mesh(wrist2, mesh="wrist2_0", material="black")
    _official_ur5e_mesh(wrist2, mesh="wrist2_1", material="urblue")
    _official_ur5e_mesh(wrist2, mesh="wrist2_2", material="jointgray")
    _robot_bag_collision_proxy(wrist2, name=f"{name}_wrist2_collision", geom_type="sphere", pos="0 0 0.050", size="0.038")

    wrist3 = ET.SubElement(wrist2, "body", {"name": f"{name}_wrist_3_link", "pos": "0 0 0.1"})
    _official_inertial(wrist3, mass="0.1889", pos="0 0.0771683 0", quat="1 0 0 1", diaginertia="0.000132134 9.90863e-05 9.90863e-05")
    _joint(wrist3, name=f"{name}_wrist_3_joint", type="hinge", axis="0 1 0", range="-360 360", stiffness="0", damping="4.0", armature="0.1")
    _official_ur5e_mesh(wrist3, mesh="wrist3", material="linkgray")
    _robot_bag_collision_proxy(wrist3, name=f"{name}_wrist3_collision", geom_type="sphere", pos="0 0.050 0", size="0.034")
    ET.SubElement(wrist3, "site", {"name": f"{name}_ee_site", "pos": "0 0.1 0", "quat": "-1 1 0 0", "size": "0.006", "rgba": "1 0.2 0.2 0.7"})

    tool_frame = ET.SubElement(wrist3, "body", {"name": f"{name}_tool_frame", "pos": "0 0.1 0", "quat": "-1 1 0 0"})
    if tool == "robotiq_2f140":
        _official_robotiq_2f140(tool_frame)
    else:
        _official_scoop_tool(tool_frame)


def _add_dual_robots(worldbody: ET.Element) -> None:
    robots = ET.SubElement(worldbody, "body", {"name": "dual_robot_frame", "pos": "0 0 0"})
    _add_official_ur5e_arm(robots, name="ur5e_2f140", base_pos=(0.0, -0.55, 0.035), yaw_deg=0.0, tool="robotiq_2f140")
    _add_official_ur5e_arm(robots, name="ur5e_scoop", base_pos=(0.0, 0.55, 0.035), yaw_deg=180.0, tool="scoop")


def _add_actuators(root: ET.Element, include_robots: bool) -> None:
    actuator = ET.SubElement(root, "actuator")
    if include_robots:
        ur5_joints = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
        for prefix in ("ur5e_2f140", "ur5e_scoop"):
            for joint in ur5_joints:
                ET.SubElement(
                    actuator,
                    "position",
                    {"name": f"{prefix}_{joint}_act", "joint": f"{prefix}_{joint}", "kp": "70", "kv": "10", "ctrlrange": "-6.283185 6.283185"},
                )
        for joint, ctrlrange in (("finger_left_slide", "-0.068 0.000"), ("finger_right_slide", "0.000 0.068")):
            ET.SubElement(actuator, "position", {"name": f"{joint}_act", "joint": joint, "kp": "120", "kv": "14", "ctrlrange": ctrlrange})


def build_scene_tree(scenario: str = "baseline_filled", *, include_robots: bool = True) -> ET.Element:
    state = get_scenario(scenario)
    root = ET.Element("mujoco", {"model": f"dual_sack_{scenario}"})
    ET.SubElement(root, "compiler", {"angle": "degree", "autolimits": "true"})
    ET.SubElement(
        root,
        "option",
        {
            "timestep": f"{TIMESTEP:.6f}",
            "gravity": "0 0 -9.81",
            "integrator": "implicitfast",
            "solver": "Newton",
            "iterations": "120",
            "ls_iterations": "30",
            "jacobian": "sparse",
            "cone": "elliptic",
            "impratio": "4",
        },
    )
    ET.SubElement(root, "size", {"nconmax": "3200", "njmax": "4200"})
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"offwidth": "1280", "offheight": "820"})
    _add_assets(root, state, include_robots=include_robots)
    worldbody = ET.SubElement(root, "worldbody")
    _add_world(worldbody)
    _add_bag(worldbody, state)
    if include_robots:
        _add_dual_robots(worldbody)
    _add_strap_tendons(root)
    _add_shell_loop_closure_equalities(root, state)
    _add_actuators(root, include_robots)
    ET.indent(root, space="  ")
    return root


def write_scene_xml(scenario: str = "baseline_filled", *, include_robots: bool = True, path: Path | None = None) -> Path:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "dual" if include_robots else "sack"
    path = path or GENERATED_DIR / f"{scenario}_{suffix}.xml"
    root = build_scene_tree(scenario, include_robots=include_robots)
    # GUI와 평가 스크립트가 같은 scene XML을 동시에 만들 때 빈 파일을 읽지 않도록 atomic replace를 사용합니다.
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    for _ in range(12):
        try:
            tmp_path.replace(path)
            return path
        except PermissionError:
            time.sleep(0.05)
    fallback_path = path.with_name(f"{path.stem}.{os.getpid()}.xml")
    tmp_path.replace(fallback_path)
    return fallback_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a shape-coupled semi-deformable sealed pillow-sack scene")
    parser.add_argument("--scenario", choices=SCENARIO_NAMES, default="baseline_filled")
    parser.add_argument("--no-robots", action="store_true")
    args = parser.parse_args()
    path = write_scene_xml(args.scenario, include_robots=not args.no_robots)
    print(f"scene_xml={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
