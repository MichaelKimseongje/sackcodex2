import argparse
import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


class DualUR5e2FScoopSack0330:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.menagerie_dir = Path(r"D:\Michael\2025\01.Research\01.Parceldetection\16.Pybullet\mujoco_menagerie")
        self.ur5e_dir = self.menagerie_dir / "universal_robots_ur5e"
        self.ur5e_xml = self.ur5e_dir / "ur5e.xml"
        self.scene_path = self.base_dir / "mujoco_dual_ur5e_2f_scoop_sack_0330.xml"

        self.left_base_pos = np.array([0.0, -0.45, 0.0], dtype=np.float64)
        self.right_base_pos = np.array([0.0, 0.45, 0.0], dtype=np.float64)
        self.left_home_deg = np.array([-90.0, -90.0, 90.0, -90.0, -90.0, 0.0], dtype=np.float64)
        self.right_home_deg = np.array([-90.0, -90.0, 90.0, -90.0, 90.0, 0.0], dtype=np.float64)

        self.sack_mesh_scale = 0.07
        self.sack_pos0 = np.array([0.72, 0.0, 0.135], dtype=np.float64)
        self.band_pos0 = np.array([0.72, 0.0, 0.223], dtype=np.float64)
        self.band_half = np.array([0.016, 0.055, 0.008], dtype=np.float64)
        self.connect_offsets = [
            np.array([+0.012, +0.045, 0.0], dtype=np.float64),
            np.array([+0.012, -0.045, 0.0], dtype=np.float64),
            np.array([-0.012, +0.045, 0.0], dtype=np.float64),
            np.array([-0.012, -0.045, 0.0], dtype=np.float64),
        ]

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
        self.left_finger_act_names = ["left_finger_l_act", "left_finger_r_act"]
        self.left_gripper_open = 0.040
        self.left_gripper_close = 0.012

        self.band_free_qadr = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "band_free")]
        self.band_free_dadr = self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "band_free")]
        self.grasp_active = False
        self.grasp_offset = np.zeros(3, dtype=np.float64)
        self.band_quat_hold = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        self.set_arm_qpos_deg("left", self.left_home_deg)
        self.set_arm_qpos_deg("right", self.right_home_deg)
        self.set_left_gripper(self.left_gripper_open)
        mujoco.mj_forward(self.model, self.data)
        self.sync_ctrl_to_qpos()

    def _prefix_tree(self, element: ET.Element, prefix: str):
        rename_keys = {"name", "joint", "body", "site", "target"}
        for node in element.iter():
            for key, val in list(node.attrib.items()):
                if key in rename_keys:
                    node.attrib[key] = f"{prefix}_{val}"

    def _find_body(self, root: ET.Element, body_name: str):
        for body in root.iter("body"):
            if body.attrib.get("name") == body_name:
                return body
        return None

    def _convert_angle_ranges_to_degree(self, element: ET.Element):
        angle_keys = ("range", "ctrlrange")
        for node in element.iter():
            for key in angle_keys:
                if key not in node.attrib:
                    continue
                parts = node.attrib[key].split()
                if len(parts) != 2:
                    continue
                try:
                    vals = [float(parts[0]), float(parts[1])]
                except ValueError:
                    continue
                deg = np.rad2deg(vals)
                node.attrib[key] = f"{deg[0]:.6f} {deg[1]:.6f}"

    def _load_ur5e_dual_parts(self):
        root = ET.parse(self.ur5e_xml).getroot()
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
        left_body.attrib["pos"] = f"{self.left_base_pos[0]:.6f} {self.left_base_pos[1]:.6f} {self.left_base_pos[2]:.6f}"
        right_body.attrib["pos"] = f"{self.right_base_pos[0]:.6f} {self.right_base_pos[1]:.6f} {self.right_base_pos[2]:.6f}"

        left_act = copy.deepcopy(actuator)
        right_act = copy.deepcopy(actuator)
        self._prefix_tree(left_act, "left")
        self._prefix_tree(right_act, "right")
        return asset, default, left_body, right_body, left_act, right_act

    def _payload_xml(self) -> str:
        payload_specs = [
            (-0.010, -0.020, 0.000),
            (0.012, 0.018, 0.010),
            (0.000, 0.000, 0.022),
            (-0.015, 0.020, -0.005),
        ]
        chunks = []
        for i, (x, y, z) in enumerate(payload_specs):
            chunks.append(
                f'''
      <body name="payload{i}" pos="{x:.4f} {y:.4f} {z:.4f}">
        <joint name="payload{i}_x" type="slide" axis="1 0 0" limited="true" range="-0.025 0.025" damping="14"/>
        <joint name="payload{i}_y" type="slide" axis="0 1 0" limited="true" range="-0.040 0.040" damping="14"/>
        <joint name="payload{i}_z" type="slide" axis="0 0 1" limited="true" range="-0.015 0.050" damping="16"/>
        <geom type="sphere" size="0.016" mass="0.90" rgba="0.62 0.22 0.20 1" contype="0" conaffinity="0"/>
      </body>'''
            )
        return "\n".join(chunks)

    def _equality_xml(self) -> str:
        lines = []
        for offset in self.connect_offsets:
            anchor = self.band_pos0 + np.array([offset[0], offset[1], -self.band_half[2]], dtype=np.float64)
            lines.append(
                f'    <connect body1="band" body2="sack" anchor="{anchor[0]:.6f} {anchor[1]:.6f} {anchor[2]:.6f}" solref="0.030 1" solimp="0.88 0.96 0.004"/>'
            )
        return "\n".join(lines)

    def _scene_xml(self) -> str:
        ur_asset, ur_default, left_body, right_body, left_act, right_act = self._load_ur5e_dual_parts()

        left_wrist = self._find_body(left_body, "left_wrist_3_link")
        right_wrist = self._find_body(right_body, "right_wrist_3_link")
        if left_wrist is None or right_wrist is None:
            raise RuntimeError("wrist_3_link not found")

        gripper = ET.fromstring(
            '''
        <body name="left_gripper_base" pos="0 0.10 0" quat="-1 1 0 0">
          <geom type="box" size="0.020 0.020 0.030" rgba="0.12 0.12 0.12 1"/>
          <body name="left_finger_l_body" pos="0 0 0.065">
            <joint name="left_finger_l" type="slide" axis="0 1 0" limited="true" range="0.012 0.040" damping="8"/>
            <geom type="box" pos="0 0.018 0" size="0.010 0.006 0.032" rgba="0.10 0.10 0.10 1" friction="2.0 0.05 0.01"/>
          </body>
          <body name="left_finger_r_body" pos="0 0 0.065">
            <joint name="left_finger_r" type="slide" axis="0 -1 0" limited="true" range="0.012 0.040" damping="8"/>
            <geom type="box" pos="0 -0.018 0" size="0.010 0.006 0.032" rgba="0.10 0.10 0.10 1" friction="2.0 0.05 0.01"/>
          </body>
          <site name="left_gripper_pinch" pos="0 0 0.065" size="0.006" rgba="1 0 0 1"/>
        </body>'''
        )
        left_wrist.append(gripper)

        scoop = ET.fromstring(
            '''
        <body name="right_scoop_tool" pos="0 0.10 0.05" euler="-90 90 0">
          <geom type="box" pos="0 -0.020 0" size="0.050 0.070 0.001" mass="0.04" rgba="0.30 0.30 0.30 1" friction="1.6 0.05 0.01"/>
          <geom type="box" pos="0.048 -0.020 0.011" size="0.002 0.070 0.010" mass="0.015" rgba="0.30 0.30 0.30 1" friction="1.6 0.05 0.01"/>
          <geom type="box" pos="-0.048 -0.020 0.011" size="0.002 0.070 0.010" mass="0.015" rgba="0.30 0.30 0.30 1" friction="1.6 0.05 0.01"/>
          <geom type="box" pos="0 0.048 0.011" size="0.050 0.002 0.010" mass="0.012" rgba="0.30 0.30 0.30 1" friction="1.6 0.05 0.01"/>
          <geom type="box" pos="0 -0.1005 -0.001" euler="0.35 0 0" size="0.050 0.0105 0.001" mass="0.006" rgba="0.30 0.30 0.30 1" friction="1.6 0.05 0.01"/>
          <site name="right_scoop_site" pos="0 -0.090 0.006" size="0.006" rgba="0 0 1 1"/>
        </body>'''
        )
        right_wrist.append(scoop)

        payload_xml = self._payload_xml()
        equality_xml = self._equality_xml()

        root = ET.Element("mujoco", {"model": "dual_ur5e_2f_scoop_sack_0330"})
        ET.SubElement(
            root,
            "compiler",
            {
                "angle": "degree",
                "meshdir": str(self.ur5e_dir / "assets"),
                "autolimits": "true",
                "inertiafromgeom": "true",
            },
        )
        ET.SubElement(root, "option", {"integrator": "implicitfast", "timestep": "0.002", "gravity": "0 0 -9.81"})
        ET.SubElement(root, "size", {"memory": "256M", "nconmax": "4000"})
        ET.SubElement(root, "statistic", {"center": "0.45 0 0.45", "extent": "1.4"})

        visual = ET.SubElement(root, "visual")
        ET.SubElement(visual, "headlight", {"diffuse": "0.6 0.6 0.6", "ambient": "0.1 0.1 0.1", "specular": "0 0 0"})
        ET.SubElement(visual, "rgba", {"haze": "0.15 0.25 0.35 1"})
        ET.SubElement(visual, "global", {"azimuth": "120", "elevation": "-20"})

        asset = ET.SubElement(root, "asset")
        for child in list(ur_asset):
            asset.append(copy.deepcopy(child))
        ET.SubElement(
            asset,
            "mesh",
            {
                "name": "sack9_obj",
                "file": str(self.base_dir / "object" / "sack9.obj"),
                "scale": f"{self.sack_mesh_scale:.3f} {self.sack_mesh_scale:.3f} {self.sack_mesh_scale:.3f}",
            },
        )

        root.append(copy.deepcopy(ur_default))

        worldbody = ET.SubElement(root, "worldbody")
        ET.SubElement(worldbody, "light", {"name": "main_light", "pos": "0 0 1.8", "dir": "0 0 -1", "directional": "true"})
        ET.SubElement(worldbody, "geom", {"name": "floor", "type": "plane", "size": "0 0 0.05", "rgba": "0.92 0.92 0.92 1"})
        ET.SubElement(worldbody, "camera", {"name": "overview", "pos": "1.7 0 1.0", "xyaxes": "0 1 0 -0.40 0 0.92"})
        worldbody.append(left_body)
        worldbody.append(right_body)

        band = ET.fromstring(
            f'''
        <body name="band" pos="{self.band_pos0[0]:.6f} {self.band_pos0[1]:.6f} {self.band_pos0[2]:.6f}">
          <freejoint name="band_free"/>
          <geom name="grasp_band" type="box" size="{self.band_half[0]:.6f} {self.band_half[1]:.6f} {self.band_half[2]:.6f}" mass="0.10" rgba="0.50 0.36 0.20 1" friction="2.0 0.05 0.01"/>
          <site name="handle_grasp_site" pos="0 0 0" size="0.008" rgba="1 0 0 1"/>
        </body>'''
        )
        worldbody.append(band)

        sack = ET.fromstring(
            f'''
        <body name="sack" pos="{self.sack_pos0[0]:.6f} {self.sack_pos0[1]:.6f} {self.sack_pos0[2]:.6f}">
          <freejoint name="sack_free"/>
          <geom name="sack_visual" type="mesh" mesh="sack9_obj" rgba="0.90 0.85 0.70 0.52" contype="0" conaffinity="0"/>
          <geom name="sack_bottom" type="box" pos="0 0 -0.055" size="0.058 0.088 0.006" mass="0.16" rgba="0.75 0.67 0.52 0.35" friction="1.3 0.05 0.01"/>
          <body name="sack_wall_l_body" pos="0 0.084 0.010">
            <joint name="sack_wall_l_slide" type="slide" axis="0 -1 0" limited="true" range="-0.018 0.010" damping="18" stiffness="900" armature="0.003"/>
            <geom name="sack_wall_l" type="box" pos="0 0 0" size="0.058 0.006 0.070" mass="0.05" rgba="0.75 0.67 0.52 0.28" friction="1.3 0.05 0.01"/>
          </body>
          <body name="sack_wall_r_body" pos="0 -0.084 0.010">
            <joint name="sack_wall_r_slide" type="slide" axis="0 1 0" limited="true" range="-0.018 0.010" damping="18" stiffness="900" armature="0.003"/>
            <geom name="sack_wall_r" type="box" pos="0 0 0" size="0.058 0.006 0.070" mass="0.05" rgba="0.75 0.67 0.52 0.28" friction="1.3 0.05 0.01"/>
          </body>
          <body name="sack_wall_f_body" pos="0.054 0 0.010">
            <joint name="sack_wall_f_slide" type="slide" axis="-1 0 0" limited="true" range="-0.015 0.010" damping="18" stiffness="900" armature="0.003"/>
            <geom name="sack_wall_f" type="box" pos="0 0 0" size="0.006 0.088 0.070" mass="0.05" rgba="0.75 0.67 0.52 0.28" friction="1.3 0.05 0.01"/>
          </body>
          <body name="sack_wall_b_body" pos="-0.054 0 0.010">
            <joint name="sack_wall_b_slide" type="slide" axis="1 0 0" limited="true" range="-0.015 0.010" damping="18" stiffness="900" armature="0.003"/>
            <geom name="sack_wall_b" type="box" pos="0 0 0" size="0.006 0.088 0.070" mass="0.05" rgba="0.75 0.67 0.52 0.28" friction="1.3 0.05 0.01"/>
          </body>
          <site name="sack_com_site" pos="0 0 0.01" size="0.006" rgba="0 1 0 1"/>
{payload_xml}
        </body>'''
        )
        worldbody.append(sack)

        equality = ET.SubElement(root, "equality")
        for line in equality_xml.splitlines():
            equality.append(ET.fromstring(line.strip()))

        actuator = ET.SubElement(root, "actuator")
        for child in list(left_act):
            actuator.append(copy.deepcopy(child))
        for child in list(right_act):
            actuator.append(copy.deepcopy(child))
        ET.SubElement(actuator, "position", {"name": "left_finger_l_act", "joint": "left_finger_l", "ctrlrange": "0.012 0.040", "kp": "1800", "forcerange": "-200 200"})
        ET.SubElement(actuator, "position", {"name": "left_finger_r_act", "joint": "left_finger_r", "ctrlrange": "0.012 0.040", "kp": "1800", "forcerange": "-200 200"})

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

    def arm_qpos_deg(self, arm: str) -> np.ndarray:
        return np.rad2deg(self.arm_qpos(arm))

    def set_arm_qpos(self, arm: str, q: np.ndarray):
        names = self.left_joint_names if arm == "left" else self.right_joint_names
        for name, val in zip(names, q):
            self.data.qpos[self._joint_qadr(name)] = float(val)

    def set_arm_qpos_deg(self, arm: str, q_deg: np.ndarray):
        self.set_arm_qpos(arm, np.deg2rad(np.asarray(q_deg, dtype=np.float64)))

    def set_arm_target(self, arm: str, q: np.ndarray):
        acts = self.left_actuator_names if arm == "left" else self.right_actuator_names
        for act_name, val in zip(acts, q):
            self.data.ctrl[self._act_id(act_name)] = float(val)

    def set_arm_target_deg(self, arm: str, q_deg: np.ndarray):
        self.set_arm_target(arm, np.deg2rad(np.asarray(q_deg, dtype=np.float64)))

    def set_left_gripper(self, opening: float):
        opening = float(np.clip(opening, self.left_gripper_close, self.left_gripper_open))
        for act_name in self.left_finger_act_names:
            self.data.ctrl[self._act_id(act_name)] = opening

    def sync_ctrl_to_qpos(self):
        self.set_arm_target_deg("left", self.arm_qpos_deg("left"))
        self.set_arm_target_deg("right", self.arm_qpos_deg("right"))
        self.set_left_gripper(self.left_gripper_open)

    def site_pos(self, site_name: str) -> np.ndarray:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        return self.data.site_xpos[sid].copy()

    def body_pos(self, body_name: str) -> np.ndarray:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.data.xpos[bid].copy()

    def body_quat(self, body_name: str) -> np.ndarray:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.data.xquat[bid].copy()

    def step(self, n: int = 1):
        for _ in range(int(n)):
            if self.grasp_active:
                grip = self.site_pos("left_gripper_pinch")
                self.data.qpos[self.band_free_qadr:self.band_free_qadr + 3] = grip - self.grasp_offset
                self.data.qpos[self.band_free_qadr + 3:self.band_free_qadr + 7] = self.band_quat_hold
                self.data.qvel[self.band_free_dadr:self.band_free_dadr + 6] = 0.0
            mujoco.mj_step(self.model, self.data)

    def payload_center_of_mass(self) -> np.ndarray:
        names = [
            "sack",
            "payload0",
            "payload1",
            "payload2",
            "payload3",
            "sack_wall_l_body",
            "sack_wall_r_body",
            "sack_wall_f_body",
            "sack_wall_b_body",
        ]
        masses = []
        centers = []
        for name in names:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            masses.append(float(self.model.body_mass[bid]))
            centers.append(self.data.xipos[bid].copy())
        masses = np.asarray(masses, dtype=np.float64)
        centers = np.asarray(centers, dtype=np.float64)
        return (centers * masses[:, None]).sum(axis=0) / masses.sum()


def print_summary(sim: DualUR5e2FScoopSack0330):
    print("scene_xml", sim.scene_path)
    print("mujoco_version", getattr(mujoco, "__version__", "unknown"))
    print("mode", "official_ur5e_quasisoft_scoop_sack")
    print("left_q_deg", np.round(sim.arm_qpos_deg("left"), 3).tolist())
    print("right_q_deg", np.round(sim.arm_qpos_deg("right"), 3).tolist())
    print("left_pinch", np.round(sim.site_pos("left_gripper_pinch"), 4).tolist())
    print("right_scoop", np.round(sim.site_pos("right_scoop_site"), 4).tolist())
    print("payload_com", np.round(sim.payload_center_of_mass(), 4).tolist())


def run_gui(sim: DualUR5e2FScoopSack0330):
    print_summary(sim)
    mujoco.viewer.launch_from_path(str(sim.scene_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    sim = DualUR5e2FScoopSack0330()
    if args.gui or not args.headless:
        run_gui(sim)
        return
    print_summary(sim)


if __name__ == "__main__":
    main()


