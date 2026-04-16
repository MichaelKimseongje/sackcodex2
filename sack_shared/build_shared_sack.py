from __future__ import annotations

import itertools
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
GENERATED_DIR = ROOT_DIR / "generated"
OUT_DIR = ROOT_DIR / "out"

TIMESTEP = 0.001
SACK_WORLD_Z = 0.090

SACK_LENGTH = 0.360
SACK_WIDTH = 0.210
SACK_THICKNESS = 0.078
PANEL_GRID_X = 4
PANEL_GRID_Y = 3


@dataclass(frozen=True)
class SharedSackParams:
    """사진 속 마대자루에 가까운 flat pillow sack 공통 skeleton 파라미터입니다."""

    length: float = SACK_LENGTH
    width: float = SACK_WIDTH
    thickness: float = SACK_THICKNESS
    top_panel_stiffness: float = 1.6
    top_panel_damping: float = 7.5
    bottom_panel_stiffness: float = 2.0
    bottom_panel_damping: float = 8.5
    edge_stiffness: float = 2.4
    edge_damping: float = 8.0
    payload_main_pos: tuple[float, float, float] = (0.015, -0.006, -0.004)
    payload_aux_pos: tuple[float, float, float] = (-0.060, 0.034, -0.006)
    show_neighbors: bool = False
    show_hidden_support: bool = False


def _fmt(values: tuple[float, ...] | list[float]) -> str:
    return " ".join(f"{float(v):.6f}" for v in values)


def _geom(parent: ET.Element, **attrib: str) -> ET.Element:
    defaults = {
        "group": "1",
        "friction": "1.45 0.06 0.006",
        "condim": "4",
        "solref": "0.032 1",
        "solimp": "0.78 0.94 0.001",
    }
    defaults.update(attrib)
    return ET.SubElement(parent, "geom", defaults)


def _make_pillow_mesh(params: SharedSackParams) -> tuple[str, str]:
    """visual-only 밀봉 포대 외피용 closed pillow mesh를 만듭니다."""
    nx, ny = 17, 11
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    for layer_sign in (1.0, -1.0):
        for iy in range(ny):
            v = -1.0 + 2.0 * iy / (ny - 1)
            for ix in range(nx):
                u = -1.0 + 2.0 * ix / (nx - 1)
                edge = max(abs(u), abs(v))
                # 중앙은 살짝 부풀고 edge는 눌린 납작한 베개형 포대 단면입니다.
                zmag = params.thickness * (0.05 + 0.36 * max(0.0, 1.0 - edge**2.2))
                vertices.append((0.5 * params.length * u, 0.5 * params.width * v, layer_sign * zmag))

    def vid(layer: int, ix: int, iy: int) -> int:
        return layer * nx * ny + iy * nx + ix

    for iy in range(ny - 1):
        for ix in range(nx - 1):
            a, b, c, d = vid(0, ix, iy), vid(0, ix + 1, iy), vid(0, ix, iy + 1), vid(0, ix + 1, iy + 1)
            faces.extend(((a, b, c), (b, d, c)))
            a, b, c, d = vid(1, ix, iy), vid(1, ix + 1, iy), vid(1, ix, iy + 1), vid(1, ix + 1, iy + 1)
            faces.extend(((a, c, b), (b, c, d)))

    for ix in range(nx - 1):
        top_a, top_b = vid(0, ix, 0), vid(0, ix + 1, 0)
        bot_a, bot_b = vid(1, ix, 0), vid(1, ix + 1, 0)
        faces.extend(((top_a, bot_a, top_b), (top_b, bot_a, bot_b)))
        top_a, top_b = vid(0, ix, ny - 1), vid(0, ix + 1, ny - 1)
        bot_a, bot_b = vid(1, ix, ny - 1), vid(1, ix + 1, ny - 1)
        faces.extend(((top_a, top_b, bot_a), (top_b, bot_b, bot_a)))

    for iy in range(ny - 1):
        top_a, top_b = vid(0, 0, iy), vid(0, 0, iy + 1)
        bot_a, bot_b = vid(1, 0, iy), vid(1, 0, iy + 1)
        faces.extend(((top_a, top_b, bot_a), (top_b, bot_b, bot_a)))
        top_a, top_b = vid(0, nx - 1, iy), vid(0, nx - 1, iy + 1)
        bot_a, bot_b = vid(1, nx - 1, iy), vid(1, nx - 1, iy + 1)
        faces.extend(((top_a, bot_a, top_b), (top_b, bot_a, bot_b)))

    vertex = " ".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices)
    face = " ".join(f"{a} {b} {c}" for a, b, c in faces)
    return vertex, face


def _add_assets(root: ET.Element, params: SharedSackParams) -> None:
    asset = ET.SubElement(root, "asset")
    vertex, face = _make_pillow_mesh(params)
    ET.SubElement(asset, "mesh", {"name": "sealed_pillow_mesh", "vertex": vertex, "face": face})
    ET.SubElement(asset, "material", {"name": "mat_jute_skin", "rgba": "0.72 0.59 0.39 0.96"})
    ET.SubElement(asset, "material", {"name": "mat_hidden_panel", "rgba": "0.64 0.46 0.26 0.015"})
    ET.SubElement(asset, "material", {"name": "mat_surface_patch", "rgba": "0.78 0.62 0.39 0.12"})
    ET.SubElement(asset, "material", {"name": "mat_edge_seam", "rgba": "0.34 0.21 0.10 0.14"})
    ET.SubElement(asset, "material", {"name": "mat_wrinkle", "rgba": "0.22 0.12 0.04 0.00"})


def _add_world(worldbody: ET.Element) -> None:
    _geom(
        worldbody,
        name="floor",
        type="plane",
        size="2.2 2.2 0.05",
        rgba="0.90 0.88 0.80 0.25",
        group="0",
        friction="1.55 0.05 0.005",
    )
    ET.SubElement(worldbody, "light", {"name": "key_light", "pos": "0.4 -0.7 1.3", "dir": "-0.2 0.35 -1"})
    ET.SubElement(worldbody, "camera", {"name": "front", "pos": "0.56 -0.62 0.28", "xyaxes": "0.74 0.67 0 -0.18 0.20 0.96"})
    ET.SubElement(worldbody, "camera", {"name": "side", "pos": "0 -0.62 0.20", "xyaxes": "1 0 0 0 0.16 0.99"})
    ET.SubElement(worldbody, "camera", {"name": "top_angled", "pos": "0.34 -0.46 0.48", "xyaxes": "0.80 0.60 0 -0.45 0.60 0.66"})


def _panel_positions(params: SharedSackParams) -> list[tuple[int, int, float, float, float, float]]:
    cell_x = params.length / PANEL_GRID_X
    cell_y = params.width / PANEL_GRID_Y
    positions = []
    for ix, iy in itertools.product(range(PANEL_GRID_X), range(PANEL_GRID_Y)):
        x = -0.5 * params.length + (ix + 0.5) * cell_x
        y = -0.5 * params.width + (iy + 0.5) * cell_y
        positions.append((ix, iy, x, y, cell_x, cell_y))
    return positions


def _add_visual_skin(bag: ET.Element, params: SharedSackParams) -> None:
    # 물리에는 참여하지 않는 pillow-like sealed sack silhouette입니다.
    skin = ET.SubElement(bag, "body", {"name": "visual_skin", "pos": "0 0 0"})
    _geom(
        skin,
        name="visual_skin_main_pillow",
        type="mesh",
        mesh="sealed_pillow_mesh",
        material="mat_jute_skin",
        group="0",
        contype="0",
        conaffinity="0",
        mass="0.001",
    )
    # 사진 속 마대자루 모서리는 파이프가 아니라 눌린 봉합선에 가까워서 낮은 box strip으로 표현합니다.
    for name, x, y, euler, length in (
        ("visual_front_edge_seam", 0.0, -0.5 * params.width, "0 0 0", params.length),
        ("visual_back_edge_seam", 0.0, 0.5 * params.width, "0 0 0", params.length),
        ("visual_left_edge_seam", -0.5 * params.length, 0.0, "0 0 90", params.width),
        ("visual_right_edge_seam", 0.5 * params.length, 0.0, "0 0 90", params.width),
    ):
        body = ET.SubElement(skin, "body", {"name": name, "pos": f"{x:.6f} {y:.6f} 0.000000", "euler": euler})
        _geom(
            body,
            name=f"{name}_geom",
            type="box",
            size=f"{0.49 * length:.6f} 0.007500 0.004000",
            rgba="0.55 0.39 0.20 0.16",
            group="0",
            contype="0",
            conaffinity="0",
            mass="0.001",
        )
    for name, x, y in (
        ("visual_corner_front_left", -0.5 * params.length, -0.5 * params.width),
        ("visual_corner_front_right", 0.5 * params.length, -0.5 * params.width),
        ("visual_corner_back_left", -0.5 * params.length, 0.5 * params.width),
        ("visual_corner_back_right", 0.5 * params.length, 0.5 * params.width),
    ):
        body = ET.SubElement(skin, "body", {"name": name, "pos": f"{x:.6f} {y:.6f} 0.000000"})
        _geom(
            body,
            name=f"{name}_geom",
            type="ellipsoid",
            size="0.019 0.012 0.007",
            rgba="0.56 0.40 0.22 0.025",
            group="0",
            contype="0",
            conaffinity="0",
            mass="0.001",
        )
    top_print = ET.SubElement(skin, "body", {"name": "visual_print_mark", "pos": "-0.030 -0.005 0.048", "euler": "0 0 -8"})
    _geom(
        top_print,
        name="visual_print_mark_geom",
        type="box",
        size="0.060 0.010 0.001",
        rgba="0.08 0.08 0.07 0.22",
        group="0",
        contype="0",
        conaffinity="0",
        mass="0.001",
    )
    cap = ET.SubElement(skin, "body", {"name": "sealed_top_cap_visual", "pos": "0 0 0.052"})
    _geom(
        cap,
        name="sealed_top_cap_visual_geom",
        type="ellipsoid",
        size=f"{0.34 * params.length:.6f} {0.30 * params.width:.6f} 0.004000",
        rgba="0.74 0.57 0.34 0.00",
        group="0",
        contype="0",
        conaffinity="0",
        mass="0.001",
    )


def _add_top_surface_panels(bag: ET.Element, params: SharedSackParams) -> None:
    group = ET.SubElement(bag, "body", {"name": "top_surface_panels", "pos": "0 0 0"})
    for ix, iy, x, y, cell_x, cell_y in _panel_positions(params):
        index = ix * PANEL_GRID_Y + iy
        crown = 0.5 * params.thickness + 0.010 * (1.0 - abs(y) / (0.5 * params.width + 1e-6))
        body = ET.SubElement(group, "body", {"name": f"top_panel_{index:02d}", "pos": f"{x:.6f} {y:.6f} {crown:.6f}"})
        ET.SubElement(
            body,
            "joint",
            {
                "name": f"top_panel_{index:02d}_hinge",
                "type": "hinge",
                "axis": "1 0 0",
                "limited": "true",
                "range": "-28 28",
                "damping": f"{params.top_panel_damping:.3f}",
                "stiffness": f"{params.top_panel_stiffness:.3f}",
            },
        )
        _geom(
            body,
            name=f"top_panel_{index:02d}_geom",
            type="box",
            pos=f"0 {0.010 * (1.0 if y >= 0 else -1.0):.6f} 0.000000",
            size=f"{0.46 * cell_x:.6f} {0.44 * cell_y:.6f} 0.004500",
            mass="0.026",
            material="mat_surface_patch",
        )
        _geom(
            body,
            name=f"top_panel_{index:02d}_cloth_visual",
            type="ellipsoid",
            pos="0 0 0.006500",
            size=f"{0.44 * cell_x:.6f} {0.42 * cell_y:.6f} 0.002500",
            rgba="0.74 0.58 0.36 0.00",
            group="1",
            contype="0",
            conaffinity="0",
            mass="0.0002",
        )
        for wrinkle_idx, wy in enumerate((-0.24 * cell_y, 0.0, 0.24 * cell_y)):
            _geom(
                body,
                name=f"top_panel_{index:02d}_wrinkle_{wrinkle_idx}",
                type="capsule",
                fromto=f"{-0.38 * cell_x:.6f} {wy:.6f} 0.006 {0.38 * cell_x:.6f} {0.65 * wy:.6f} 0.006",
                size="0.0008",
                material="mat_wrinkle",
                group="1",
                contype="0",
                conaffinity="0",
                mass="0.0002",
            )
        ET.SubElement(body, "site", {"name": f"grasp_top_panel_{index:02d}", "pos": "0 0 0.010", "size": "0.001", "rgba": "0.1 0.6 1 0.05"})


def _add_bottom_surface_panels(bag: ET.Element, params: SharedSackParams) -> None:
    group = ET.SubElement(bag, "body", {"name": "bottom_surface_panels", "pos": "0 0 0"})
    for ix, iy, x, y, cell_x, cell_y in _panel_positions(params):
        index = ix * PANEL_GRID_Y + iy
        body = ET.SubElement(group, "body", {"name": f"bottom_panel_{index:02d}", "pos": f"{x:.6f} {y:.6f} {-0.5 * params.thickness:.6f}"})
        ET.SubElement(
            body,
            "joint",
            {
                "name": f"bottom_panel_{index:02d}_hinge",
                "type": "hinge",
                "axis": "1 0 0",
                "limited": "true",
                "range": "-22 22",
                "damping": f"{params.bottom_panel_damping:.3f}",
                "stiffness": f"{params.bottom_panel_stiffness:.3f}",
            },
        )
        _geom(
            body,
            name=f"bottom_panel_{index:02d}_geom",
            type="box",
            size=f"{0.46 * cell_x:.6f} {0.44 * cell_y:.6f} 0.004500",
            mass="0.028",
            material="mat_hidden_panel",
        )


def _add_edge_bands(bag: ET.Element, params: SharedSackParams) -> None:
    group = ET.SubElement(bag, "body", {"name": "seam_band", "pos": "0 0 0"})
    specs = [
        ("front", 0.0, -0.5 * params.width, 0.0, params.length, "0 0 0"),
        ("back", 0.0, 0.5 * params.width, 0.0, params.length, "0 0 0"),
        ("left", -0.5 * params.length, 0.0, 0.0, params.width, "0 0 90"),
        ("right", 0.5 * params.length, 0.0, 0.0, params.width, "0 0 90"),
        ("top_front", 0.0, -0.5 * params.width, 0.5 * params.thickness, params.length, "0 0 0"),
        ("top_back", 0.0, 0.5 * params.width, 0.5 * params.thickness, params.length, "0 0 0"),
        ("top_left", -0.5 * params.length, 0.0, 0.5 * params.thickness, params.width, "0 0 90"),
        ("top_right", 0.5 * params.length, 0.0, 0.5 * params.thickness, params.width, "0 0 90"),
    ]
    for index, (label, x, y, z, length, euler) in enumerate(specs):
        body = ET.SubElement(group, "body", {"name": f"seam_band_{index:02d}_{label}", "pos": f"{x:.6f} {y:.6f} {z:.6f}", "euler": euler})
        ET.SubElement(
            body,
            "joint",
            {
                "name": f"seam_band_{index:02d}_{label}_hinge",
                "type": "hinge",
                "axis": "0 1 0",
                "limited": "true",
                "range": "-18 18",
                "damping": f"{params.edge_damping:.3f}",
                "stiffness": f"{params.edge_stiffness:.3f}",
            },
        )
        _geom(
            body,
            name=f"seam_band_{index:02d}_{label}_geom",
            type="box",
            size=f"{0.48 * length:.6f} 0.003800 0.002500",
            mass="0.012",
            material="mat_edge_seam",
        )
        ET.SubElement(body, "site", {"name": f"grasp_seam_{index:02d}", "pos": "0 0 0", "size": "0.001", "rgba": "1 0.55 0.1 0.05"})


def _add_corner_folds(bag: ET.Element, params: SharedSackParams) -> None:
    corners = [
        ("front_left", -0.5 * params.length, -0.5 * params.width, 42),
        ("front_right", 0.5 * params.length, -0.5 * params.width, -42),
        ("back_left", -0.5 * params.length, 0.5 * params.width, -42),
        ("back_right", 0.5 * params.length, 0.5 * params.width, 42),
    ]
    for idx, (label, x, y, yaw) in enumerate(corners, start=1):
        body = ET.SubElement(bag, "body", {"name": f"corner_fold_patch_{idx}_{label}", "pos": f"{x:.6f} {y:.6f} {0.5 * params.thickness:.6f}", "euler": f"0 0 {yaw}"})
        ET.SubElement(
            body,
            "joint",
            {
                "name": f"corner_fold_patch_{idx}_{label}_hinge",
                "type": "hinge",
                "axis": "1 0 0",
                "limited": "true",
                "range": "-75 32",
                "damping": "7.0",
                "stiffness": "2.0",
            },
        )
        _geom(
            body,
            name=f"corner_fold_patch_{idx}_{label}_geom",
            type="box",
            pos="0 0 -0.006",
            size="0.052 0.018 0.010",
            mass="0.012",
            rgba="0.68 0.50 0.29 0.18",
        )
        ET.SubElement(body, "site", {"name": f"grasp_corner_fold_{idx}", "pos": "0 0 0", "size": "0.001", "rgba": "0.2 0.2 1 0.05"})


def _add_fold_flaps(bag: ET.Element, params: SharedSackParams) -> None:
    # 사진 속 마대자루처럼 긴 edge가 접히거나 말린 상황을 공통 skeleton에 남겨둡니다.
    specs = (
        (1, "front_folded_lip", 0.0, -0.5 * params.width - 0.006, 0.5 * params.thickness + 0.004, -18.0),
        (2, "side_folded_lip", -0.5 * params.length - 0.006, 0.0, 0.5 * params.thickness + 0.002, 72.0),
    )
    for flap_index, label, x, y, z, yaw in specs:
        body = ET.SubElement(
            bag,
            "body",
            {
                "name": f"fold_flap_{flap_index}",
                "pos": f"{x:.6f} {y:.6f} {z:.6f}",
                "euler": f"0 0 {yaw:.6f}",
            },
        )
        ET.SubElement(
            body,
            "joint",
            {
                "name": f"fold_flap_{flap_index}_hinge",
                "type": "hinge",
                "axis": "1 0 0",
                "limited": "true",
                "range": "-90 28",
                "damping": "7.5",
                "stiffness": "2.2",
            },
        )
        length = 0.62 * params.length if flap_index == 1 else 0.58 * params.width
        _geom(
            body,
            name=f"fold_flap_{flap_index}_{label}_geom",
            type="box",
            pos="0 0 -0.004",
            size=f"{0.5 * length:.6f} 0.012000 0.005000",
            mass="0.012",
            rgba="0.68 0.50 0.29 0.18",
        )
        _geom(
            body,
            name=f"fold_flap_{flap_index}_{label}_rolled_root",
            type="box",
            pos="0 0 0.004",
            size=f"{0.5 * length:.6f} 0.005500 0.003000",
            mass="0.003",
            rgba="0.30 0.16 0.06 0.16",
            contype="0",
            conaffinity="0",
        )
        ET.SubElement(body, "site", {"name": f"grasp_fold_{flap_index}", "pos": "0 0 0", "size": "0.001", "rgba": "0.1 0.2 1 0.05"})


def _add_bottom_sling(bag: ET.Element, params: SharedSackParams) -> None:
    body = ET.SubElement(bag, "body", {"name": "bottom_sling", "pos": f"0 0 {-0.5 * params.thickness - 0.006:.6f}"})
    ET.SubElement(
        body,
        "joint",
        {
            "name": "bottom_sling_sag",
            "type": "slide",
            "axis": "0 0 1",
            "limited": "true",
            "range": "-0.055 0.012",
            "damping": "14.0",
            "stiffness": "1.2",
        },
    )
    for index, (name, euler, size) in enumerate(
        (
            ("bottom_sling_long", "0 0 0", f"{0.39 * params.length:.6f} 0.012 0.005"),
            ("bottom_sling_cross_a", "0 0 90", f"{0.30 * params.width:.6f} 0.010 0.005"),
            ("bottom_sling_cross_b", "0 0 90", f"{0.30 * params.width:.6f} 0.010 0.005"),
        )
    ):
        pos_x = 0.0 if index == 0 else (-0.18 * params.length if index == 1 else 0.18 * params.length)
        _geom(
            body,
            name=name,
            type="box",
            pos=f"{pos_x:.6f} 0 0",
            euler=euler,
            size=size,
            mass="0.020",
            rgba="0.45 0.26 0.12 0.02",
        )
    ET.SubElement(body, "site", {"name": "bottom_sling_site", "pos": "0 0 0", "size": "0.001", "rgba": "0.8 0.35 0.08 0.05"})


def _add_payloads(bag: ET.Element, params: SharedSackParams) -> None:
    for name, pos, size, mass, rgba in (
        ("payload_main", params.payload_main_pos, (0.138, 0.076, 0.027), 0.55, "0.42 0.22 0.10 0.02"),
        ("payload_aux", params.payload_aux_pos, (0.060, 0.040, 0.018), 0.12, "0.36 0.18 0.08 0.01"),
    ):
        body = ET.SubElement(bag, "body", {"name": name, "pos": _fmt(pos)})
        for axis_name, axis, limit in (
            ("x", "1 0 0", "-0.045 0.045"),
            ("y", "0 1 0", "-0.035 0.035"),
            ("z", "0 0 1", "-0.018 0.018"),
        ):
            ET.SubElement(
                body,
                "joint",
                {
                    "name": f"{name}_{axis_name}",
                    "type": "slide",
                    "axis": axis,
                    "limited": "true",
                    "range": limit,
                    "damping": "18",
                    "stiffness": "2.0",
                },
            )
        _geom(body, name=f"{name}_geom", type="ellipsoid", size=_fmt(size), mass=f"{mass:.4f}", rgba=rgba)


def _add_helpers(worldbody: ET.Element, bag: ET.Element, params: SharedSackParams) -> None:
    cue = ET.SubElement(bag, "body", {"name": "side_bulge_cue", "pos": f"{0.22 * params.length:.6f} {0.22 * params.width:.6f} 0.000000"})
    _geom(
        cue,
        name="side_bulge_cue_geom",
        type="ellipsoid",
        size="0.050 0.025 0.020",
        rgba="0.80 0.50 0.24 0.15",
        contype="0",
        conaffinity="0",
        mass="0.001",
    )
    for name, y, rgba in (("neighbor_left", -0.235, "0.46 0.34 0.20 0.18"), ("neighbor_right", 0.235, "0.46 0.34 0.20 0.18")):
        body = ET.SubElement(worldbody, "body", {"name": name, "pos": f"0 {y:.6f} {SACK_WORLD_Z:.6f}"})
        _geom(
            body,
            name=f"{name}_geom",
            type="ellipsoid",
            size=f"{0.5 * params.length:.6f} {0.5 * params.width:.6f} {0.50 * params.thickness:.6f}",
            rgba=rgba if params.show_neighbors else "0.46 0.34 0.20 0.00",
            contype="1" if params.show_neighbors else "0",
            conaffinity="1" if params.show_neighbors else "0",
            friction="1.4 0.05 0.005",
        )
    support = ET.SubElement(worldbody, "body", {"name": "hidden_support", "pos": f"0 0 {0.5 * SACK_WORLD_Z:.6f}"})
    _geom(
        support,
        name="hidden_support_geom",
        type="box",
        size=f"{0.44 * params.length:.6f} {0.42 * params.width:.6f} 0.010",
        rgba="0.10 0.65 0.95 0.20" if params.show_hidden_support else "0.10 0.65 0.95 0.00",
        contype="1" if params.show_hidden_support else "0",
        conaffinity="1" if params.show_hidden_support else "0",
        friction="0.9 0.03 0.003",
    )


def _add_bag(worldbody: ET.Element, params: SharedSackParams) -> None:
    bag = ET.SubElement(worldbody, "body", {"name": "bag_frame", "pos": f"0 0 {SACK_WORLD_Z:.6f}"})
    ET.SubElement(bag, "freejoint", {"name": "bag_frame_freejoint"})
    ET.SubElement(bag, "site", {"name": "bag_frame_origin", "pos": "0 0 0", "size": "0.001", "rgba": "1 0 0 0.05"})
    _geom(
        bag,
        name="bag_mouse_handle",
        type="sphere",
        pos="0 0 0",
        size="0.025",
        mass="0.001",
        rgba="0.10 0.35 1.00 0.02",
        contype="0",
        conaffinity="0",
    )
    _add_visual_skin(bag, params)
    _add_top_surface_panels(bag, params)
    _add_bottom_surface_panels(bag, params)
    _add_edge_bands(bag, params)
    _add_corner_folds(bag, params)
    _add_fold_flaps(bag, params)
    _add_bottom_sling(bag, params)
    _add_payloads(bag, params)
    _add_helpers(worldbody, bag, params)


def build_scene_tree(params: SharedSackParams | None = None) -> ET.Element:
    params = params or SharedSackParams()
    root = ET.Element("mujoco", {"model": "shared_flat_pillow_sack_skeleton"})
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
    ET.SubElement(root, "size", {"nconmax": "2600", "njmax": "3600"})
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"offwidth": "1280", "offheight": "820"})
    _add_assets(root, params)
    worldbody = ET.SubElement(root, "worldbody")
    _add_world(worldbody)
    _add_bag(worldbody, params)
    ET.indent(root, space="  ")
    return root


def write_scene_xml(path: Path | None = None, params: SharedSackParams | None = None) -> Path:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    path = path or GENERATED_DIR / "scene_shared_sack.xml"
    root = build_scene_tree(params)
    path.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    return path


def main() -> int:
    path = write_scene_xml()
    print(f"scene_xml={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
