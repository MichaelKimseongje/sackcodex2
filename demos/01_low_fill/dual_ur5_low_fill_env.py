from __future__ import annotations

import argparse
import copy
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from low_fill_builder import (
    BAG_FRAME_POS_Z,
    BAG_MASS,
    BALLAST_BODY_NAME,
    BALLAST_MASS,
    BALLAST_POS,
    BALLAST_SIZE,
    CONE_TYPE,
    ELEMENT_TEXT,
    FLOOR_FRICTION,
    FRAME_DIAGINERTIA,
    FRAME_MASS,
    GENERATED_DIR,
    IMPRATIO,
    ITERATIONS,
    JACOBIAN,
    LS_ITERATIONS,
    SHELL_CONDIM,
    SHELL_DAMPING,
    SHELL_FRICTION,
    SHELL_RADIUS,
    SHELL_SOLIMP,
    SHELL_SOLREF,
    SOLVER,
    TIMESTEP,
    _format_points,
    _underfilled_points,
    collect_shell_body_ids,
)


ROOT_DIR = Path(__file__).resolve().parent
MENAGERIE_DIR = Path(r"D:\Michael\2025\01.Research\01.Parceldetection\16.Pybullet\mujoco_menagerie")
UR5E_DIR = MENAGERIE_DIR / "universal_robots_ur5e"
UR5E_XML = UR5E_DIR / "ur5e.xml"
DUAL_SCENE_PATH = GENERATED_DIR / "dual_ur5_low_fill.xml"

LEFT_BASE_POS = np.array([0.0, -0.48, 0.0], dtype=np.float64)
RIGHT_BASE_POS = np.array([0.0, 0.48, 0.0], dtype=np.float64)
LEFT_HOME_DEG = np.array([-92.0, -88.0, 96.0, -98.0, -88.0, 0.0], dtype=np.float64)
RIGHT_HOME_DEG = np.array([-88.0, -88.0, 96.0, -98.0, 88.0, 0.0], dtype=np.float64)

LOW_FILL_WORLD_POS = np.array([0.70, 0.0, BAG_FRAME_POS_Z], dtype=np.float64)
JOINT_STEP_DEG_DEFAULT = 2.0
GRIPPER_STEP_DEFAULT = 0.002

# GLFW key code와 동일한 값이다. MuJoCo Python viewer가 별도 상수를 노출하지 않는 환경이 있어 직접 둔다.
KEY_RIGHT = 262
KEY_LEFT = 263
KEY_DOWN = 264
KEY_UP = 265


class DualUR5LowFillEnv:
    """저충진 sack과 dual UR5 + 2F/scoop end-effector를 한 장면에 배치한다."""

    def __init__(self, *, with_ballast: bool = True, scene_path: Path = DUAL_SCENE_PATH):
        if not UR5E_XML.exists():
            raise FileNotFoundError(f"UR5e XML not found: {UR5E_XML}")

        self.with_ballast = with_ballast
        self.scene_path = scene_path
        self.scene_path.parent.mkdir(parents=True, exist_ok=True)

        self.left_joint_names = [
            "left_shoulder_pan_joint",
            "left_shoulder_lift_joint",
            "left_elbow_joint",
            "left_wrist_1_joint",
            "left_wrist_2_joint",
            "left_wrist_3_joint",
        ]
        self.right_joint_names = [
            "right_shoulder_pan_joint",
            "right_shoulder_lift_joint",
            "right_elbow_joint",
            "right_wrist_1_joint",
            "right_wrist_2_joint",
            "right_wrist_3_joint",
        ]
        self.left_actuator_names = [
            "left_shoulder_pan",
            "left_shoulder_lift",
            "left_elbow",
            "left_wrist_1",
            "left_wrist_2",
            "left_wrist_3",
        ]
        self.right_actuator_names = [
            "right_shoulder_pan",
            "right_shoulder_lift",
            "right_elbow",
            "right_wrist_1",
            "right_wrist_2",
            "right_wrist_3",
        ]
        self.left_finger_actuator_names = ["left_finger_l_act", "left_finger_r_act"]
        self.left_gripper_open = 0.040
        self.left_gripper_close = 0.012

        self.write_scene_xml()
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)
        self.reset()

    def _prefix_tree(self, element: ET.Element, prefix: str) -> None:
        rename_keys = {"name", "joint", "body", "site", "target"}
        for node in element.iter():
            for key, value in list(node.attrib.items()):
                if key in rename_keys:
                    node.attrib[key] = f"{prefix}_{value}"

    def _convert_angle_ranges_to_degree(self, element: ET.Element) -> None:
        for node in element.iter():
            if node.tag != "joint" or "range" not in node.attrib:
                continue
            parts = node.attrib["range"].split()
            if len(parts) != 2:
                continue
            try:
                values = np.array([float(parts[0]), float(parts[1])], dtype=np.float64)
            except ValueError:
                continue

            # UR5e 원본은 radian이고 이 scene은 degree compiler를 쓰므로 joint range만 변환한다.
            # general actuator ctrlrange는 MuJoCo에서 각도 단위 자동 변환 대상이 아니어서 radian 그대로 둔다.
            degree_values = np.rad2deg(values)
            node.attrib["range"] = f"{degree_values[0]:.6f} {degree_values[1]:.6f}"

    def _find_body(self, root: ET.Element, body_name: str) -> ET.Element | None:
        for body in root.iter("body"):
            if body.attrib.get("name") == body_name:
                return body
        return None

    def _load_ur5e_dual_parts(self) -> tuple[ET.Element, ET.Element, ET.Element, ET.Element, ET.Element, ET.Element]:
        root = ET.parse(UR5E_XML).getroot()
        asset = copy.deepcopy(root.find("asset"))
        default = copy.deepcopy(root.find("default"))
        actuator = copy.deepcopy(root.find("actuator"))
        self._convert_angle_ranges_to_degree(default)
        self._convert_angle_ranges_to_degree(actuator)

        worldbody = root.find("worldbody")
        robot_body = copy.deepcopy(list(worldbody)[-1])

        left_body = copy.deepcopy(robot_body)
        right_body = copy.deepcopy(robot_body)
        self._prefix_tree(left_body, "left")
        self._prefix_tree(right_body, "right")
        left_body.attrib["name"] = "left_base"
        right_body.attrib["name"] = "right_base"
        left_body.attrib["pos"] = f"{LEFT_BASE_POS[0]:.6f} {LEFT_BASE_POS[1]:.6f} {LEFT_BASE_POS[2]:.6f}"
        right_body.attrib["pos"] = f"{RIGHT_BASE_POS[0]:.6f} {RIGHT_BASE_POS[1]:.6f} {RIGHT_BASE_POS[2]:.6f}"

        left_actuator = copy.deepcopy(actuator)
        right_actuator = copy.deepcopy(actuator)
        self._prefix_tree(left_actuator, "left")
        self._prefix_tree(right_actuator, "right")
        return asset, default, left_body, right_body, left_actuator, right_actuator

    def _attach_2f_gripper(self, left_body: ET.Element) -> None:
        left_wrist = self._find_body(left_body, "left_wrist_3_link")
        if left_wrist is None:
            raise RuntimeError("left_wrist_3_link not found")

        gripper = ET.fromstring(
            """
        <body name="left_gripper_base" pos="0 0.10 0" quat="-1 1 0 0">
          <geom name="left_gripper_palm" type="box" size="0.022 0.022 0.032" rgba="0.12 0.12 0.12 1"/>
          <body name="left_finger_l_body" pos="0 0 0.067">
            <joint name="left_finger_l" type="slide" axis="0 1 0" limited="true" range="0.012 0.040" damping="8"/>
            <geom name="left_finger_l_pad" type="box" pos="0 0.018 0" size="0.011 0.006 0.034" rgba="0.10 0.10 0.10 1" friction="2.0 0.05 0.01"/>
          </body>
          <body name="left_finger_r_body" pos="0 0 0.067">
            <joint name="left_finger_r" type="slide" axis="0 -1 0" limited="true" range="0.012 0.040" damping="8"/>
            <geom name="left_finger_r_pad" type="box" pos="0 -0.018 0" size="0.011 0.006 0.034" rgba="0.10 0.10 0.10 1" friction="2.0 0.05 0.01"/>
          </body>
          <site name="left_gripper_pinch" pos="0 0 0.067" size="0.006" rgba="1 0 0 1"/>
        </body>
        """
        )
        left_wrist.append(gripper)

    def _attach_scoop(self, right_body: ET.Element) -> None:
        right_wrist = self._find_body(right_body, "right_wrist_3_link")
        if right_wrist is None:
            raise RuntimeError("right_wrist_3_link not found")

        scoop = ET.fromstring(
            """
        <body name="right_scoop_tool" pos="0 0.10 0.05" euler="-90 90 0">
          <geom name="right_scoop_plate" type="box" pos="0 -0.020 0" size="0.055 0.078 0.002" mass="0.05" rgba="0.30 0.30 0.30 1" friction="1.6 0.05 0.01"/>
          <geom name="right_scoop_lip" type="box" pos="0.053 -0.020 0.013" size="0.003 0.078 0.012" mass="0.015" rgba="0.25 0.25 0.25 1" friction="1.6 0.05 0.01"/>
          <geom name="right_scoop_left_rail" type="box" pos="0 0.057 0.014" size="0.055 0.003 0.012" mass="0.012" rgba="0.25 0.25 0.25 1"/>
          <geom name="right_scoop_right_rail" type="box" pos="0 -0.097 0.014" size="0.055 0.003 0.012" mass="0.012" rgba="0.25 0.25 0.25 1"/>
          <geom name="right_scoop_ramp" type="box" pos="0 -0.105 -0.001" euler="0.35 0 0" size="0.055 0.012 0.0015" mass="0.008" rgba="0.30 0.30 0.30 1" friction="1.6 0.05 0.01"/>
          <site name="right_scoop_site" pos="0 -0.090 0.006" size="0.006" rgba="0 0 1 1"/>
          <site name="right_scoop_tip_site" pos="0.053 -0.105 0.002" size="0.005" rgba="0 0.7 1 1"/>
        </body>
        """
        )
        right_wrist.append(scoop)

    def _low_fill_body(self) -> ET.Element:
        bag_frame = ET.Element(
            "body",
            {
                "name": "bag_frame",
                "pos": f"{LOW_FILL_WORLD_POS[0]:.6f} {LOW_FILL_WORLD_POS[1]:.6f} {LOW_FILL_WORLD_POS[2]:.6f}",
            },
        )
        ET.SubElement(
            bag_frame,
            "inertial",
            {"pos": "0 0 0", "mass": f"{FRAME_MASS:.4f}", "diaginertia": FRAME_DIAGINERTIA},
        )
        ET.SubElement(bag_frame, "freejoint", {"name": "bag_frame_freejoint"})
        ET.SubElement(
            bag_frame,
            "site",
            {"name": "bag_frame_origin", "pos": "0 0 0", "size": "0.006", "rgba": "0.8 0.15 0.15 1"},
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
                "rgba": "0.75 0.60 0.36 0.92",
                "point": _format_points(_underfilled_points()),
                "element": ELEMENT_TEXT,
            },
        )
        ET.SubElement(
            flexcomp,
            "contact",
            {
                "condim": SHELL_CONDIM,
                "selfcollide": "none",
                "internal": "false",
                "friction": SHELL_FRICTION,
                "solref": SHELL_SOLREF,
                "solimp": SHELL_SOLIMP,
            },
        )
        ET.SubElement(flexcomp, "edge", {"equality": "true", "damping": f"{SHELL_DAMPING:.1f}"})

        if self.with_ballast:
            ballast = ET.SubElement(bag_frame, "body", {"name": BALLAST_BODY_NAME, "pos": BALLAST_POS})
            joint_specs = (
                ("x", "1 0 0", "-0.030 0.030", "10"),
                ("y", "0 1 0", "-0.018 0.018", "10"),
                ("z", "0 0 1", "-0.014 0.018", "16"),
            )
            for axis_name, axis, limits, damping in joint_specs:
                ET.SubElement(
                    ballast,
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
                ballast,
                "geom",
                {
                    "name": f"{BALLAST_BODY_NAME}_geom",
                    "type": "ellipsoid",
                    "size": BALLAST_SIZE,
                    "mass": BALLAST_MASS,
                    "rgba": "0.46 0.21 0.14 1",
                    "condim": SHELL_CONDIM,
                    "friction": "0.35 0.01 0.001",
                },
            )
        return bag_frame

    def scene_xml(self) -> str:
        ur_asset, ur_default, left_body, right_body, left_actuator, right_actuator = self._load_ur5e_dual_parts()
        self._attach_2f_gripper(left_body)
        self._attach_scoop(right_body)

        root = ET.Element("mujoco", {"model": "dual_ur5_low_fill"})
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
                "timestep": str(TIMESTEP),
                "gravity": "0 0 -9.81",
                "solver": SOLVER,
                "iterations": str(ITERATIONS),
                "ls_iterations": str(LS_ITERATIONS),
                "jacobian": JACOBIAN,
                "cone": CONE_TYPE,
                "impratio": str(IMPRATIO),
            },
        )
        ET.SubElement(root, "size", {"memory": "512M", "nconmax": "8000"})
        ET.SubElement(root, "statistic", {"center": "0.55 0 0.45", "extent": "1.6"})

        visual = ET.SubElement(root, "visual")
        ET.SubElement(visual, "headlight", {"diffuse": "0.6 0.6 0.6", "ambient": "0.1 0.1 0.1", "specular": "0 0 0"})
        ET.SubElement(visual, "rgba", {"haze": "0.15 0.25 0.35 1"})
        ET.SubElement(visual, "global", {"azimuth": "120", "elevation": "-20", "offwidth": "1280", "offheight": "720"})

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
                "condim": SHELL_CONDIM,
            },
        )
        ET.SubElement(worldbody, "camera", {"name": "overview", "pos": "1.65 0 1.0", "xyaxes": "0 1 0 -0.42 0 0.91"})
        worldbody.append(left_body)
        worldbody.append(right_body)
        worldbody.append(self._low_fill_body())

        actuator = ET.SubElement(root, "actuator")
        for child in list(left_actuator):
            actuator.append(copy.deepcopy(child))
        for child in list(right_actuator):
            actuator.append(copy.deepcopy(child))
        ET.SubElement(
            actuator,
            "position",
            {"name": "left_finger_l_act", "joint": "left_finger_l", "ctrlrange": "0.012 0.040", "kp": "1800", "forcerange": "-200 200"},
        )
        ET.SubElement(
            actuator,
            "position",
            {"name": "left_finger_r_act", "joint": "left_finger_r", "ctrlrange": "0.012 0.040", "kp": "1800", "forcerange": "-200 200"},
        )

        ET.indent(root, space="  ")
        return ET.tostring(root, encoding="unicode")

    def write_scene_xml(self) -> Path:
        self.scene_path.write_text(self.scene_xml(), encoding="utf-8")
        return self.scene_path

    def _joint_qpos_address(self, joint_name: str) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise KeyError(f"joint not found: {joint_name}")
        return int(self.model.jnt_qposadr[joint_id])

    def _actuator_id(self, actuator_name: str) -> int:
        actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if actuator_id < 0:
            raise KeyError(f"actuator not found: {actuator_name}")
        return int(actuator_id)

    def actuator_ctrl(self, actuator_name: str) -> float:
        return float(self.data.ctrl[self._actuator_id(actuator_name)])

    def add_actuator_delta(self, actuator_name: str, delta: float) -> float:
        actuator_id = self._actuator_id(actuator_name)
        value = float(self.data.ctrl[actuator_id] + delta)
        if self.model.actuator_ctrllimited[actuator_id]:
            low, high = self.model.actuator_ctrlrange[actuator_id]
            value = float(np.clip(value, low, high))
        self.data.ctrl[actuator_id] = value
        return value

    def set_arm_qpos_deg(self, arm: str, q_deg: np.ndarray) -> None:
        names = self.left_joint_names if arm == "left" else self.right_joint_names
        for joint_name, value in zip(names, np.deg2rad(np.asarray(q_deg, dtype=np.float64))):
            self.data.qpos[self._joint_qpos_address(joint_name)] = float(value)

    def set_arm_target_deg(self, arm: str, q_deg: np.ndarray) -> None:
        names = self.left_actuator_names if arm == "left" else self.right_actuator_names
        for actuator_name, value in zip(names, np.deg2rad(np.asarray(q_deg, dtype=np.float64))):
            self.data.ctrl[self._actuator_id(actuator_name)] = float(value)

    def set_left_gripper(self, opening: float) -> None:
        opening = float(np.clip(opening, self.left_gripper_close, self.left_gripper_open))
        for actuator_name in self.left_finger_actuator_names:
            self.data.ctrl[self._actuator_id(actuator_name)] = opening

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.set_arm_qpos_deg("left", LEFT_HOME_DEG)
        self.set_arm_qpos_deg("right", RIGHT_HOME_DEG)
        self.set_arm_target_deg("left", LEFT_HOME_DEG)
        self.set_arm_target_deg("right", RIGHT_HOME_DEG)
        self.set_left_gripper(self.left_gripper_open)
        mujoco.mj_forward(self.model, self.data)

    def step(self, steps: int = 1) -> None:
        for _ in range(int(steps)):
            mujoco.mj_step(self.model, self.data)

    def print_summary(self) -> None:
        print(f"scene_xml={self.scene_path}")
        print(f"with_ballast={self.with_ballast}")
        print(f"mujoco_version={getattr(mujoco, '__version__', 'unknown')}")
        print(f"left_gripper_site={np.round(self.site_pos('left_gripper_pinch'), 4).tolist()}")
        print(f"right_scoop_site={np.round(self.site_pos('right_scoop_site'), 4).tolist()}")
        print(f"bag_shell_body_count={len(collect_shell_body_ids(self.model))}")

    def site_pos(self, site_name: str) -> np.ndarray:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise KeyError(f"site not found: {site_name}")
        return self.data.site_xpos[site_id].copy()


class KeyboardJointStepper:
    """viewer에서 UR5 관절 목표값을 작은 단위로 바꾸는 키보드 조작기."""

    def __init__(
        self,
        env: DualUR5LowFillEnv,
        *,
        joint_step_deg: float = JOINT_STEP_DEG_DEFAULT,
        gripper_step: float = GRIPPER_STEP_DEFAULT,
    ):
        self.env = env
        self.arm = "left"
        self.joint_index = 0
        self.joint_step_rad = float(np.deg2rad(joint_step_deg))
        self.gripper_step = float(gripper_step)
        self.print_help()
        self.print_selected()

    def arm_actuator_names(self) -> list[str]:
        return self.env.left_actuator_names if self.arm == "left" else self.env.right_actuator_names

    def current_actuator_name(self) -> str:
        return self.arm_actuator_names()[self.joint_index]

    def current_label(self) -> str:
        actuator_name = self.current_actuator_name()
        return f"{self.arm} J{self.joint_index + 1} {actuator_name.removeprefix(self.arm + '_')}"

    def print_help(self) -> None:
        print("\nkeyboard_joint_control=True")
        print("  L/R        : 왼쪽 2F UR5 / 오른쪽 scoop UR5 선택")
        print("  1..6       : 현재 arm의 관절 선택")
        print("  ←/→        : 이전/다음 관절 선택")
        print("  ↑/↓        : 선택 관절 목표각을 작은 단위로 증가/감소")
        print("  O/C        : 왼쪽 2F gripper를 조금 열기/닫기")
        print("  H          : 두 UR5와 gripper를 home pose로 리셋")
        print("  ?          : 이 도움말 다시 출력")

    def print_selected(self) -> None:
        value_deg = np.rad2deg(self.env.actuator_ctrl(self.current_actuator_name()))
        print(f"selected={self.current_label()} target={value_deg:.2f} deg")

    def select_arm(self, arm: str) -> None:
        self.arm = arm
        self.joint_index = min(self.joint_index, len(self.arm_actuator_names()) - 1)
        self.print_selected()

    def select_joint(self, joint_index: int) -> None:
        self.joint_index = int(np.clip(joint_index, 0, len(self.arm_actuator_names()) - 1))
        self.print_selected()

    def nudge_joint(self, direction: float) -> None:
        actuator_name = self.current_actuator_name()
        value = self.env.add_actuator_delta(actuator_name, direction * self.joint_step_rad)
        print(f"{self.current_label()} target={np.rad2deg(value):.2f} deg")

    def nudge_gripper(self, direction: float) -> None:
        actuator_name = self.env.left_finger_actuator_names[0]
        opening = self.env.actuator_ctrl(actuator_name) + direction * self.gripper_step
        self.env.set_left_gripper(opening)
        print(f"left 2F opening={self.env.actuator_ctrl(actuator_name):.4f} m")

    def handle_key(self, keycode: int) -> None:
        if keycode in (ord("L"), ord("l")):
            self.select_arm("left")
        elif keycode in (ord("R"), ord("r")):
            self.select_arm("right")
        elif keycode in (ord("1"), ord("2"), ord("3"), ord("4"), ord("5"), ord("6")):
            self.select_joint(keycode - ord("1"))
        elif keycode == KEY_LEFT:
            self.select_joint(self.joint_index - 1)
        elif keycode == KEY_RIGHT:
            self.select_joint(self.joint_index + 1)
        elif keycode == KEY_UP:
            self.nudge_joint(+1.0)
        elif keycode == KEY_DOWN:
            self.nudge_joint(-1.0)
        elif keycode in (ord("O"), ord("o")):
            self.nudge_gripper(+1.0)
        elif keycode in (ord("C"), ord("c")):
            self.nudge_gripper(-1.0)
        elif keycode in (ord("H"), ord("h")):
            self.env.reset()
            print("robot home pose reset")
            self.print_selected()
        elif keycode in (ord("?"), ord("/")):
            self.print_help()
            self.print_selected()


def run_viewer(
    env: DualUR5LowFillEnv,
    *,
    speed: float = 1.0,
    keyboard_control: bool = True,
    joint_step_deg: float = JOINT_STEP_DEG_DEFAULT,
) -> None:
    env.print_summary()
    sleep_dt = env.model.opt.timestep / max(speed, 1e-6)
    key_controller = KeyboardJointStepper(env, joint_step_deg=joint_step_deg) if keyboard_control else None

    with mujoco.viewer.launch_passive(
        env.model,
        env.data,
        key_callback=key_controller.handle_key if key_controller is not None else None,
    ) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = np.array([0.58, 0.0, 0.35])
        viewer.cam.distance = 1.55
        viewer.cam.azimuth = 130.0
        viewer.cam.elevation = -22.0
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXFACE] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXEDGE] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXVERT] = True

        while viewer.is_running():
            start = time.perf_counter()
            env.step()
            viewer.sync()
            remaining = sleep_dt - (time.perf_counter() - start)
            if remaining > 0:
                time.sleep(remaining)


def validate_dual_scene(with_ballast: bool = True, seconds: float = 2.0) -> dict[str, float | int | bool | str]:
    env = DualUR5LowFillEnv(with_ballast=with_ballast)
    total_steps = max(1, int(np.ceil(seconds / env.model.opt.timestep)))
    nonfinite = False
    peak_qvel = 0.0
    for _ in range(total_steps):
        env.step()
        if not np.all(np.isfinite(env.data.qpos)) or not np.all(np.isfinite(env.data.qvel)):
            nonfinite = True
            break
        peak_qvel = max(peak_qvel, float(np.max(np.abs(env.data.qvel))) if env.data.qvel.size else 0.0)

    return {
        "scene_xml": str(env.scene_path),
        "with_ballast": with_ballast,
        "nonfinite": nonfinite,
        "peak_qvel": peak_qvel,
        "bag_shell_body_count": len(collect_shell_body_ids(env.model)),
        "left_gripper_site_z": float(env.site_pos("left_gripper_pinch")[2]),
        "right_scoop_site_z": float(env.site_pos("right_scoop_site")[2]),
        "pass_fail": bool(not nonfinite),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="dual UR5 + 2F/scoop + low-fill flex sack viewer")
    parser.add_argument("--no-ballast", action="store_true", help="단일 ballast 없이 shell-only low-fill sack을 사용합니다.")
    parser.add_argument("--headless", action="store_true", help="viewer 없이 로드/짧은 안정성만 확인합니다.")
    parser.add_argument("--speed", type=float, default=1.0, help="viewer 실행 속도 배수")
    parser.add_argument("--seconds", type=float, default=2.0, help="headless 검증 시간")
    parser.add_argument("--joint-step-deg", type=float, default=JOINT_STEP_DEG_DEFAULT, help="up/down key joint target step in degrees")
    parser.add_argument("--no-keyboard-control", action="store_true", help="disable keyboard step control in viewer")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with_ballast = not args.no_ballast
    if args.headless:
        result = validate_dual_scene(with_ballast=with_ballast, seconds=args.seconds)
        for key, value in result.items():
            print(f"{key}={value}")
        return 0 if result["pass_fail"] else 1

    env = DualUR5LowFillEnv(with_ballast=with_ballast)
    run_viewer(
        env,
        speed=args.speed,
        keyboard_control=not args.no_keyboard_control,
        joint_step_deg=args.joint_step_deg,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
