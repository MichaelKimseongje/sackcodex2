from __future__ import annotations

import argparse
import math
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
BASE_XML_PATH = ROOT_DIR / "bag_base.xml"
GENERATED_DIR = ROOT_DIR / "out"
SUPPORTED_FILL_MODES = ("ballast1", "clump3")


def _parse_points(point_text: str) -> list[list[float]]:
    values = [float(value) for value in point_text.split()]
    return [values[index:index + 3] for index in range(0, len(values), 3)]


def _format_points(points: list[list[float]]) -> str:
    return " ".join(f"{coord:.6f}" for point in points for coord in point)


def _base_points() -> list[list[float]]:
    root = ET.parse(BASE_XML_PATH).getroot()
    bag_frame = next(body for body in root.find("worldbody").findall("body") if body.get("name") == "bag_frame")
    bag_shell = bag_frame.find("flexcomp")
    return _parse_points(bag_shell.get("point"))


def _make_low_fill_points() -> list[list[float]]:
    points = _base_points()

    # 저충진 surrogate:
    # - top center를 눈에 띄게 더 낮추고
    # - top rim을 안쪽/아래로 당겨
    # - 중간 둘레도 조금 좁혀 상부가 실제로 꺼져 보이게 만든다.
    points[0][2] = 0.090
    for point_id in range(1, 9):
        points[point_id][0] *= 0.84
        points[point_id][1] *= 0.84
        points[point_id][2] -= 0.020
    for point_id in range(9, 17):
        points[point_id][0] *= 0.92
        points[point_id][1] *= 0.92

    return points


def _bag_frame_and_shell(root: ET.Element) -> tuple[ET.Element, ET.Element]:
    worldbody = root.find("worldbody")
    bag_frame = next(body for body in worldbody.findall("body") if body.get("name") == "bag_frame")
    bag_shell = bag_frame.find("flexcomp")
    return bag_frame, bag_shell


def _ensure_contact_block(root: ET.Element) -> ET.Element:
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    return contact


def _add_internal_body(
    bag_frame: ET.Element,
    name: str,
    pos: str,
    size: str,
    mass: str,
) -> None:
    body = ET.SubElement(bag_frame, "body", {"name": name, "pos": pos})
    ET.SubElement(
        body,
        "joint",
        {
            "name": f"{name}_x",
            "type": "slide",
            "axis": "1 0 0",
            "limited": "true",
            "range": "-0.022 0.022",
            "damping": "12",
        },
    )
    ET.SubElement(
        body,
        "joint",
        {
            "name": f"{name}_y",
            "type": "slide",
            "axis": "0 1 0",
            "limited": "true",
            "range": "-0.024 0.024",
            "damping": "12",
        },
    )
    ET.SubElement(
        body,
        "joint",
        {
            "name": f"{name}_z",
            "type": "slide",
            "axis": "0 0 1",
            "limited": "true",
            "range": "-0.015 0.020",
            "damping": "16",
        },
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": f"{name}_geom",
            "type": "ellipsoid",
            "size": size,
            "mass": mass,
            "rgba": "0.45 0.20 0.16 1",
            "condim": "3",
            "friction": "0.3 0.01 0.001",
        },
    )


def make_low_fill(fill_mode: str = "ballast1", output_path: Path | None = None) -> Path:
    if fill_mode not in SUPPORTED_FILL_MODES:
        raise ValueError(f"fill_mode must be one of {SUPPORTED_FILL_MODES}, got {fill_mode!r}")

    root = ET.parse(BASE_XML_PATH).getroot()
    bag_frame, bag_shell = _bag_frame_and_shell(root)

    bag_shell.set("point", _format_points(_make_low_fill_points()))

    # shell-fill contact은 단순하게 유지한다.
    shell_contact = bag_shell.find("contact")
    if shell_contact is not None:
        shell_contact.set("selfcollide", "none")
        shell_contact.set("internal", "false")
        shell_contact.set("condim", "3")
        shell_contact.set("friction", "0.4 0.01 0.001")

    moving_bodies: list[str] = []

    if fill_mode == "ballast1":
        _add_internal_body(
            bag_frame=bag_frame,
            name="low_fill_ballast_0",
            pos="0 0 -0.090",
            size="0.032 0.026 0.043",
            mass="0.65",
        )
        moving_bodies.append("low_fill_ballast_0")
    else:
        clumps = (
            ("low_fill_clump_0", "0.000 0.000 -0.092", "0.024 0.020 0.036", "0.28"),
            ("low_fill_clump_1", "-0.018 0.016 -0.078", "0.018 0.014 0.026", "0.16"),
            ("low_fill_clump_2", "0.020 -0.014 -0.076", "0.018 0.014 0.024", "0.14"),
        )
        for name, pos, size, mass in clumps:
            _add_internal_body(
                bag_frame=bag_frame,
                name=name,
                pos=pos,
                size=size,
                mass=mass,
            )
            moving_bodies.append(name)

        # clump-clump collision은 명시적으로 끈다.
        contact = _ensure_contact_block(root)
        for index, body1 in enumerate(moving_bodies):
            for body2 in moving_bodies[index + 1:]:
                ET.SubElement(contact, "exclude", {"body1": body1, "body2": body2})

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = GENERATED_DIR / f"low_fill_{fill_mode}.xml"

    ET.indent(root, space="  ")
    xml_text = ET.tostring(root, encoding="unicode")
    output_path.write_text(xml_text, encoding="utf-8")
    return output_path


def _shell_mass_from_xml(xml_path: Path) -> float:
    root = ET.parse(xml_path).getroot()
    _, bag_shell = _bag_frame_and_shell(root)
    return float(bag_shell.get("mass", "0.0"))


def _internal_body_ids(model: mujoco.MjModel) -> list[int]:
    body_ids: list[int] = []
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name and name.startswith("low_fill_"):
            body_ids.append(body_id)
    return body_ids


def _set_joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str, value: float) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        return
    data.qpos[model.jnt_qposadr[joint_id]] = value


def stage_low_fill_demo(model: mujoco.MjModel, data: mujoco.MjData, fill_mode: str = "ballast1") -> None:
    # viewer에서 settle 변화가 보이도록 내부 mass를 일시적으로 위쪽/옆쪽으로 올려서 시작한다.
    if fill_mode == "ballast1":
        staged = {
            "low_fill_ballast_0_x": 0.016,
            "low_fill_ballast_0_y": -0.012,
            "low_fill_ballast_0_z": 0.018,
        }
    else:
        staged = {
            "low_fill_clump_0_x": 0.020,
            "low_fill_clump_0_y": 0.000,
            "low_fill_clump_0_z": 0.020,
            "low_fill_clump_1_x": -0.020,
            "low_fill_clump_1_y": 0.020,
            "low_fill_clump_1_z": 0.018,
            "low_fill_clump_2_x": 0.020,
            "low_fill_clump_2_y": -0.020,
            "low_fill_clump_2_z": 0.018,
        }

    for joint_name, value in staged.items():
        _set_joint_qpos(model, data, joint_name, value)
    mujoco.mj_forward(model, data)


def validate_low_fill(
    fill_mode: str = "ballast1",
    *,
    seconds: float = 3.0,
    tail_seconds: float = 0.2,
    output_path: Path | None = None,
) -> dict[str, float | bool | str | int]:
    xml_path = make_low_fill(fill_mode=fill_mode, output_path=output_path)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    total_steps = max(1, math.ceil(seconds / model.opt.timestep))
    tail_steps = max(1, math.ceil(tail_seconds / model.opt.timestep))
    tail_qvel: list[float] = []
    peak_qvel = 0.0
    nonfinite = False

    for _ in range(total_steps):
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            nonfinite = True
            break

        qvel_peak = float(np.max(np.abs(data.qvel))) if data.qvel.size else 0.0
        peak_qvel = max(peak_qvel, qvel_peak)
        tail_qvel.append(qvel_peak)
        if len(tail_qvel) > tail_steps:
            tail_qvel.pop(0)

    flex_verts = np.array(data.flexvert_xpos)
    top_rim_average_height = float(np.mean(flex_verts[1:9, 2]))
    top_center_height = float(flex_verts[0, 2])
    upper_half_mean_height = float(np.mean(flex_verts[:17, 2]))

    shell_mass = _shell_mass_from_xml(xml_path)
    shell_com_height = float(np.mean(flex_verts[:, 2]))

    internal_ids = _internal_body_ids(model)
    internal_masses = np.array([model.body_mass[body_id] for body_id in internal_ids], dtype=float)
    internal_heights = np.array([data.xpos[body_id, 2] for body_id in internal_ids], dtype=float)
    internal_mass_total = float(np.sum(internal_masses))
    internal_com_height = (
        float(np.average(internal_heights, weights=internal_masses))
        if internal_masses.size and internal_mass_total > 0
        else shell_com_height
    )
    center_of_mass_height = (
        (shell_mass * shell_com_height + internal_mass_total * internal_com_height)
        / max(shell_mass + internal_mass_total, 1e-9)
    )

    base_points = _base_points()
    base_top_gap = float(base_points[0][2] - np.mean([point[2] for point in base_points[1:9]]))
    current_top_gap = top_center_height - top_rim_average_height
    upper_half_visibly_collapsed = bool(current_top_gap < 0.5 * base_top_gap)

    result: dict[str, float | bool | str | int] = {
        "xml": str(xml_path),
        "fill_mode": fill_mode,
        "internal_body_count": len(internal_ids),
        "top_rim_average_height": top_rim_average_height,
        "center_of_mass_height": float(center_of_mass_height),
        "upper_half_visibly_collapsed": upper_half_visibly_collapsed,
        "top_center_height": top_center_height,
        "upper_half_mean_height": upper_half_mean_height,
        "base_top_gap": base_top_gap,
        "current_top_gap": float(current_top_gap),
        "peak_qvel": peak_qvel,
        "tail_mean_qvel": float(np.mean(tail_qvel)) if tail_qvel else float("inf"),
        "nonfinite": nonfinite,
    }

    return result


def _print_validation(result: dict[str, float | bool | str | int]) -> None:
    ordered_keys = (
        "xml",
        "fill_mode",
        "internal_body_count",
        "top_rim_average_height",
        "center_of_mass_height",
        "upper_half_visibly_collapsed",
        "top_center_height",
        "upper_half_mean_height",
        "base_top_gap",
        "current_top_gap",
        "peak_qvel",
        "tail_mean_qvel",
        "nonfinite",
    )
    for key in ordered_keys:
        print(f"{key}={result[key]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="low_fill scenario 생성 및 검증 도구")
    parser.add_argument(
        "--fill-mode",
        choices=SUPPORTED_FILL_MODES,
        default="ballast1",
        help="ballast1은 단일 바닥 ballast, clump3은 최대 3개 rigid clump 버전입니다.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="scenario를 생성한 뒤 settle 검증과 low_fill 지표를 출력합니다.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="생성할 low_fill XML 경로",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.validate:
        result = validate_low_fill(fill_mode=args.fill_mode, output_path=args.output)
        _print_validation(result)
        return 0 if not result["nonfinite"] else 1

    xml_path = make_low_fill(fill_mode=args.fill_mode, output_path=args.output)
    print(f"xml={xml_path}")
    print(f"fill_mode={args.fill_mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
