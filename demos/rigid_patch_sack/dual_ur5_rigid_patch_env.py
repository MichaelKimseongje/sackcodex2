from __future__ import annotations

import copy
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from build_sack_surrogate import TIMESTEP, build_scene_tree
from scenario_builder import available_scenarios, get_scenario


ROOT_DIR = Path(__file__).resolve().parent
LOW_FILL_DIR = ROOT_DIR.parent / "01_low_fill"
if str(LOW_FILL_DIR) not in sys.path:
    sys.path.insert(0, str(LOW_FILL_DIR))

from dual_ur5_low_fill_env import (  # noqa: E402
    DUAL_SCENE_PATH as _LOW_FILL_DUAL_SCENE_PATH,
    FLOOR_FRICTION,
    IMPRATIO,
    ITERATIONS,
    JACOBIAN,
    JOINT_STEP_DEG_DEFAULT,
    LS_ITERATIONS,
    SOLVER,
    UR5E_DIR,
    DualUR5LowFillEnv,
    KeyboardJointStepper,
)


DUAL_SCENE_PATH = ROOT_DIR / "generated" / "dual_ur5_rigid_patch_sack.xml"
SACK_WORLD_X = 0.40


def _shift_body_x(body: ET.Element, dx: float) -> None:
    parts = [float(value) for value in body.attrib.get("pos", "0 0 0").split()]
    while len(parts) < 3:
        parts.append(0.0)
    parts[0] += dx
    body.attrib["pos"] = f"{parts[0]:.6f} {parts[1]:.6f} {parts[2]:.6f}"


class DualUR5RigidPatchEnv(DualUR5LowFillEnv):
    """sealed articulated sack v2와 dual UR5 + 2F/scoop을 한 장면에 배치한다."""

    def __init__(self, *, scenario: str = "underfilled", scene_path: Path = DUAL_SCENE_PATH):
        if scenario not in available_scenarios():
            raise ValueError(f"unknown scenario: {scenario}")
        self.scenario = scenario
        self.gui_title = "Dual UR5 Sealed Articulated Sack V2 Control"
        self.gui_header = f"Dual UR5 + 2F/scoop + sealed articulated sack v2 [{scenario}]"
        super().__init__(with_ballast=True, scene_path=scene_path)

    def _sack_world_bodies(self) -> list[ET.Element]:
        sack_root = build_scene_tree(get_scenario(self.scenario), include_eval_gripper=False)
        worldbody = sack_root.find("worldbody")
        if worldbody is None:
            raise RuntimeError("sack worldbody not found")

        keep_names = {"bag_frame", "neighbor_left", "neighbor_right", "temporary_bottom_support"}
        bodies: list[ET.Element] = []
        for child in list(worldbody):
            if child.tag != "body":
                continue
            if child.attrib.get("name") not in keep_names:
                continue
            copied = copy.deepcopy(child)
            _shift_body_x(copied, SACK_WORLD_X)
            bodies.append(copied)
        return bodies

    def scene_xml(self) -> str:
        ur_asset, ur_default, left_body, right_body, left_actuator, right_actuator = self._load_ur5e_dual_parts()
        self._attach_2f_gripper(left_body)
        self._attach_scoop(right_body)

        root = ET.Element("mujoco", {"model": f"dual_ur5_sealed_articulated_sack_v2_{self.scenario}"})
        ET.SubElement(
            root,
            "compiler",
            {
                "angle": "degree",
                "meshdir": str(UR5E_DIR / "assets"),
                "autolimits": "true",
                "inertiafromgeom": "true",
            },
        )
        ET.SubElement(
            root,
            "option",
            {
                "integrator": "implicitfast",
                "timestep": f"{TIMESTEP:.6f}",
                "gravity": "0 0 -9.81",
                "solver": SOLVER,
                "iterations": str(max(ITERATIONS, 100)),
                "ls_iterations": str(max(LS_ITERATIONS, 24)),
                "jacobian": JACOBIAN,
                "cone": "elliptic",
                "impratio": str(IMPRATIO),
            },
        )
        ET.SubElement(root, "size", {"memory": "512M", "nconmax": "8000"})
        ET.SubElement(root, "statistic", {"center": "0.55 0 0.45", "extent": "1.6"})

        visual = ET.SubElement(root, "visual")
        ET.SubElement(visual, "headlight", {"diffuse": "0.6 0.6 0.6", "ambient": "0.1 0.1 0.1", "specular": "0 0 0"})
        ET.SubElement(visual, "rgba", {"haze": "0.15 0.25 0.35 1"})
        ET.SubElement(visual, "global", {"azimuth": "120", "elevation": "-20", "offwidth": "1280", "offheight": "820"})

        asset = ET.SubElement(root, "asset")
        for child in list(ur_asset):
            asset.append(copy.deepcopy(child))
        root.append(copy.deepcopy(ur_default))

        worldbody = ET.SubElement(root, "worldbody")
        ET.SubElement(worldbody, "light", {"name": "main_light", "pos": "0 0 1.8", "dir": "0 0 -1", "directional": "true"})
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": "floor",
                "type": "plane",
                "size": "2 2 0.05",
                "rgba": "0.92 0.92 0.92 1",
                "friction": FLOOR_FRICTION,
                "condim": "4",
            },
        )
        ET.SubElement(worldbody, "camera", {"name": "overview", "pos": "1.65 0 1.0", "xyaxes": "0 1 0 -0.42 0 0.91"})
        ET.SubElement(worldbody, "camera", {"name": "front", "pos": "1.02 0 0.42", "xyaxes": "0 1 0 -0.22 0 0.98"})
        self._add_world_origin_axes(worldbody)
        worldbody.append(left_body)
        worldbody.append(right_body)
        for sack_body in self._sack_world_bodies():
            worldbody.append(sack_body)

        actuator = ET.SubElement(root, "actuator")
        for child in list(left_actuator):
            actuator.append(copy.deepcopy(child))
        for child in list(right_actuator):
            actuator.append(copy.deepcopy(child))
        ET.SubElement(
            actuator,
            "position",
            {"name": "left_finger_l_act", "joint": "left_finger_l", "ctrlrange": "0.010 0.080", "kp": "450", "forcerange": "-60 60"},
        )
        ET.SubElement(
            actuator,
            "position",
            {"name": "left_finger_r_act", "joint": "left_finger_r", "ctrlrange": "0.010 0.080", "kp": "450", "forcerange": "-60 60"},
        )

        ET.indent(root, space="  ")
        return ET.tostring(root, encoding="unicode")

    def grasp_site_names(self) -> list[str]:
        names = [f"grasp_seam_{index:02d}" for index in range(8)]
        names += [f"grasp_shoulder_{index:02d}" for index in range(8)]
        names += ["grasp_fold_1", "grasp_fold_2"]
        return [name for name in names if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name) >= 0]

    def nearest_grasp_target(self, reference_xyz: np.ndarray) -> tuple[str, np.ndarray]:
        site_names = self.grasp_site_names()
        if not site_names:
            raise RuntimeError("no v2 grasp sites found")
        distances = [float(np.linalg.norm(self.site_pos(site_name) - reference_xyz)) for site_name in site_names]
        site_name = site_names[int(np.argmin(distances))]
        return site_name, self.site_pos(site_name)

    def print_summary(self) -> None:
        print(f"scene_xml={self.scene_path}")
        print(f"scenario={self.scenario}")
        print(f"mujoco_version={getattr(mujoco, '__version__', 'unknown')}")
        print(f"left_gripper_site={np.round(self.site_pos('left_gripper_pinch'), 4).tolist()}")
        print(f"right_scoop_site={np.round(self.site_pos('right_scoop_site'), 4).tolist()}")
        print(f"grasp_site_count={len(self.grasp_site_names())}")
        print(f"nflex={int(getattr(self.model, 'nflex', 0))}")


def smoke_test(scenario: str) -> int:
    env = DualUR5RigidPatchEnv(scenario=scenario)
    for actuator_name in env.left_actuator_names + env.right_actuator_names:
        value_deg = np.rad2deg(env.actuator_ctrl(actuator_name))
        if not np.isfinite(value_deg):
            print(f"nonfinite target: {actuator_name}")
            return 1
    left_target = env.end_effector_pos("left") + np.array([0.01, 0.0, 0.0], dtype=np.float64)
    right_target = env.end_effector_pos("right") + np.array([0.0, 0.0, 0.01], dtype=np.float64)
    left_ik = env.solve_ee_position_ik("left", left_target)
    right_ik = env.solve_ee_position_ik("right", right_target)
    site_name, target = env.nearest_grasp_target(env.end_effector_pos("left"))
    grasp_ik = env.solve_ee_position_ik("left", target + np.array([0.0, 0.0, 0.010], dtype=np.float64))
    env.set_left_gripper_gap(0.0, immediate=True)
    print(f"scene_xml={env.scene_path}")
    print(f"scenario={scenario}")
    print(f"left_ee_ik={left_ik}")
    print(f"right_ee_ik={right_ik}")
    print(f"nearest_grasp_site={site_name}")
    print(f"nearest_grasp_site_pos={target.tolist()}")
    print(f"nearest_grasp_ik={grasp_ik}")
    print(f"left_gripper_gap_m={env.left_gripper_gap()}")
    print("dual_ur5_rigid_patch_smoke_pass=True")
    return 0
