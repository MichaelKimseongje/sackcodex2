import argparse
import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


class DualUR5eOfficial0330:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.menagerie_dir = Path(r"D:\Michael\2025\01.Research\01.Parceldetection\16.Pybullet\mujoco_menagerie")
        self.ur5e_dir = self.menagerie_dir / "universal_robots_ur5e"
        self.source_xml = self.ur5e_dir / "ur5e.xml"
        self.scene_path = self.base_dir / "mujoco_dual_ur5e_official_0330.xml"

        self.left_base_pos = np.array([0.0, -0.45, 0.0], dtype=np.float64)
        self.right_base_pos = np.array([0.0, 0.45, 0.0], dtype=np.float64)
        self.left_home = np.array([-1.57, -1.57, 1.57, -1.57, -1.57, 0.0], dtype=np.float64)
        self.right_home = np.array([-1.57, -1.57, 1.57, -1.57, 1.57, 0.0], dtype=np.float64)

        self._write_scene_xml()
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.data = mujoco.MjData(self.model)

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

        self.set_arm_qpos("left", self.left_home)
        self.set_arm_qpos("right", self.right_home)
        mujoco.mj_forward(self.model, self.data)
        self.sync_ctrl_to_qpos()

    def _prefix_tree(self, element: ET.Element, prefix: str):
        rename_keys = {"name", "joint", "body", "site", "target"}
        for node in element.iter():
            for key, val in list(node.attrib.items()):
                if key not in rename_keys:
                    continue
                node.attrib[key] = f"{prefix}_{val}"

    def _load_prefixed_robot(self, prefix: str, base_pos: np.ndarray):
        root = ET.parse(self.source_xml).getroot()
        default = root.find("default")
        worldbody = root.find("worldbody")
        actuator = root.find("actuator")
        if default is None or worldbody is None or actuator is None:
            raise RuntimeError("Unexpected UR5e XML structure")

        default_copy = copy.deepcopy(default)
        body_elems = [copy.deepcopy(child) for child in list(worldbody)]
        robot_body = body_elems[-1]
        self._prefix_tree(robot_body, prefix)
        robot_body.attrib["pos"] = f"{base_pos[0]:.6f} {base_pos[1]:.6f} {base_pos[2]:.6f}"

        actuator_copy = copy.deepcopy(actuator)
        self._prefix_tree(actuator_copy, prefix)
        return default_copy, robot_body, actuator_copy

    def _scene_xml(self) -> str:
        src_root = ET.parse(self.source_xml).getroot()
        asset = src_root.find("asset")
        if asset is None:
            raise RuntimeError("No asset block found in UR5e XML")

        default_copy, left_body, left_act = self._load_prefixed_robot("left", self.left_base_pos)
        _unused_default, right_body, right_act = self._load_prefixed_robot("right", self.right_base_pos)

        root = ET.Element("mujoco", {"model": "dual_ur5e_official_0330"})
        ET.SubElement(root, "compiler", {
            "angle": "radian",
            "meshdir": str(self.ur5e_dir / "assets"),
            "autolimits": "true",
        })
        ET.SubElement(root, "option", {
            "integrator": "implicitfast",
            "timestep": "0.002",
            "gravity": "0 0 -9.81",
        })
        ET.SubElement(root, "size", {"memory": "128M", "nconmax": "2000"})
        ET.SubElement(root, "statistic", {"center": "0.45 0 0.45", "extent": "1.2"})

        visual = ET.SubElement(root, "visual")
        ET.SubElement(visual, "headlight", {
            "diffuse": "0.6 0.6 0.6",
            "ambient": "0.1 0.1 0.1",
            "specular": "0 0 0",
        })
        ET.SubElement(visual, "rgba", {"haze": "0.15 0.25 0.35 1"})
        ET.SubElement(visual, "global", {"azimuth": "120", "elevation": "-20"})

        root.append(copy.deepcopy(asset))
        root.append(default_copy)

        worldbody = ET.SubElement(root, "worldbody")
        ET.SubElement(worldbody, "light", {
            "name": "main_light",
            "pos": "0 0 1.8",
            "dir": "0 0 -1",
            "directional": "true",
        })
        ET.SubElement(worldbody, "geom", {
            "name": "floor",
            "type": "plane",
            "size": "0 0 0.05",
            "rgba": "0.92 0.92 0.92 1",
        })
        ET.SubElement(worldbody, "camera", {
            "name": "overview",
            "pos": "1.6 0 1.0",
            "xyaxes": "0 1 0 -0.40 0 0.92",
        })
        worldbody.append(left_body)
        worldbody.append(right_body)

        actuator = ET.SubElement(root, "actuator")
        for child in list(left_act):
            actuator.append(copy.deepcopy(child))
        for child in list(right_act):
            actuator.append(copy.deepcopy(child))

        return ET.tostring(root, encoding="unicode")

    def _write_scene_xml(self):
        self.scene_path.write_text(self._scene_xml(), encoding="utf-8")

    def _joint_qadr(self, joint_name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        return int(self.model.jnt_qposadr[jid])

    def _act_id(self, act_name: str) -> int:
        return int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name))

    def arm_qpos(self, arm: str) -> np.ndarray:
        names = self.left_joint_names if arm == "left" else self.right_joint_names
        return np.array([self.data.qpos[self._joint_qadr(name)] for name in names], dtype=np.float64)

    def set_arm_qpos(self, arm: str, q: np.ndarray):
        names = self.left_joint_names if arm == "left" else self.right_joint_names
        for name, val in zip(names, q):
            self.data.qpos[self._joint_qadr(name)] = float(val)

    def set_arm_target(self, arm: str, q: np.ndarray):
        acts = self.left_actuator_names if arm == "left" else self.right_actuator_names
        for act_name, val in zip(acts, q):
            self.data.ctrl[self._act_id(act_name)] = float(val)

    def sync_ctrl_to_qpos(self):
        self.set_arm_target("left", self.arm_qpos("left"))
        self.set_arm_target("right", self.arm_qpos("right"))

    def step(self, n: int = 1):
        for _ in range(int(n)):
            mujoco.mj_step(self.model, self.data)

    def move_arm_smooth(self, arm: str, q_target: np.ndarray, steps: int = 240):
        q0 = self.arm_qpos(arm)
        for alpha in np.linspace(0.0, 1.0, max(2, steps)):
            q = (1.0 - alpha) * q0 + alpha * q_target
            self.set_arm_target(arm, q)
            other = "right" if arm == "left" else "left"
            self.set_arm_target(other, self.arm_qpos(other))
            self.step(1)

    def ee_pos(self, arm: str) -> np.ndarray:
        site_name = "left_attachment_site" if arm == "left" else "right_attachment_site"
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        return self.data.site_xpos[sid].copy()


def print_summary(sim: DualUR5eOfficial0330):
    print("scene_xml", sim.scene_path)
    print("mujoco_version", getattr(mujoco, "__version__", "unknown"))
    print("mode", "official_ur5e_dual")
    print("left_q", np.round(sim.arm_qpos("left"), 4).tolist())
    print("right_q", np.round(sim.arm_qpos("right"), 4).tolist())
    print("left_ee", np.round(sim.ee_pos("left"), 4).tolist())
    print("right_ee", np.round(sim.ee_pos("right"), 4).tolist())


def run_gui(sim: DualUR5eOfficial0330):
    print_summary(sim)
    mujoco.viewer.launch_from_path(str(sim.scene_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Launch the MuJoCo viewer.")
    parser.add_argument("--headless", action="store_true", help="Run without the MuJoCo viewer.")
    args = parser.parse_args()

    sim = DualUR5eOfficial0330()
    if args.gui or not args.headless:
        run_gui(sim)
        return
    print_summary(sim)


if __name__ == "__main__":
    main()


