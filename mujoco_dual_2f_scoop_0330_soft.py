import argparse
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


@dataclass
class ArmSpec:
    name: str
    base_pos: tuple[float, float, float]
    color: str
    has_gripper: bool = False
    has_scoop: bool = False


LEFT_ARM = ArmSpec("left", (0.10, -0.42, 0.0), "0.73 0.73 0.76 1", has_gripper=True)
RIGHT_ARM = ArmSpec("right", (0.10, 0.42, 0.0), "0.76 0.73 0.73 1", has_scoop=True)


class MuJoCoDualSoftSack0330:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.scene_path = self.base_dir / "mujoco_dual_2f_scoop_0330_soft.xml"
        self.sack_mesh_scale = 0.07
        self.sack_pos0 = np.array([0.72, 0.0, 0.135], dtype=np.float64)
        self.band_pos0 = np.array([0.72, 0.0, 0.215], dtype=np.float64)
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

        self.left_joints = [f"left_j{i}" for i in range(1, 7)]
        self.right_joints = [f"right_j{i}" for i in range(1, 7)]
        self.left_actuators = [f"act_left_j{i}" for i in range(1, 7)]
        self.right_actuators = [f"act_right_j{i}" for i in range(1, 7)]
        self.left_finger_acts = ["act_left_finger_l", "act_left_finger_r"]

        self.band_free_qadr = self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "band_free")]
        self.band_free_dadr = self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "band_free")]

        self.left_q_home = np.array([0.35, -1.10, 1.55, -0.95, -1.20, 0.20], dtype=np.float64)
        self.right_q_home = np.array([-0.35, -1.00, 1.45, -0.90, 1.10, 0.10], dtype=np.float64)
        self.right_support = np.array([-0.45, -0.95, 1.25, -1.15, 1.20, 0.20], dtype=np.float64)
        self.gripper_open = 0.040
        self.gripper_close = 0.012
        self.grasp_active = False
        self.grasp_offset = np.zeros(3, dtype=np.float64)
        self.band_quat_hold = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        self._set_arm_qpos("left", self.left_q_home)
        self._set_arm_qpos("right", self.right_q_home)
        self._set_left_gripper(self.gripper_open)
        mujoco.mj_forward(self.model, self.data)
        self._sync_ctrl_to_qpos()

    def _arm_body_xml(self, spec: ArmSpec) -> str:
        name = spec.name
        bx, by, bz = spec.base_pos
        tool = ""
        if spec.has_gripper:
            tool = f'''
                  <body name="{name}_tool" pos="0.14 0 0">
                    <geom type="box" size="0.020 0.020 0.030" rgba="0.15 0.15 0.15 1"/>
                    <body name="{name}_finger_l_body" pos="0 0 0.065">
                      <joint name="{name}_finger_l" type="slide" axis="0 1 0" limited="true" range="0.012 0.040" damping="8"/>
                      <geom type="box" pos="0 0.018 0" size="0.010 0.006 0.032" rgba="0.10 0.10 0.10 1" friction="2.0 0.05 0.01"/>
                    </body>
                    <body name="{name}_finger_r_body" pos="0 0 0.065">
                      <joint name="{name}_finger_r" type="slide" axis="0 -1 0" limited="true" range="0.012 0.040" damping="8"/>
                      <geom type="box" pos="0 -0.018 0" size="0.010 0.006 0.032" rgba="0.10 0.10 0.10 1" friction="2.0 0.05 0.01"/>
                    </body>
                    <site name="{name}_grip_site" pos="0 0 0.065" size="0.006" rgba="1 0 0 1"/>
                  </body>
'''
        if spec.has_scoop:
            tool = f'''
                  <body name="{name}_scoop" pos="0.16 0 0">
                    <geom type="box" pos="0.070 0 0" size="0.070 0.080 0.004" rgba="0.30 0.35 0.42 1" friction="1.2 0.05 0.01"/>
                    <geom type="box" pos="0.130 0 0.025" size="0.004 0.080 0.025" rgba="0.25 0.30 0.36 1" friction="1.2 0.05 0.01"/>
                    <geom type="box" pos="0.070 0.076 0.015" size="0.060 0.004 0.015" rgba="0.25 0.30 0.36 1" friction="1.2 0.05 0.01"/>
                    <geom type="box" pos="0.070 -0.076 0.015" size="0.060 0.004 0.015" rgba="0.25 0.30 0.36 1" friction="1.2 0.05 0.01"/>
                    <site name="{name}_scoop_site" pos="0.050 0 0.028" size="0.006" rgba="0 0 1 1"/>
                  </body>
'''
        return f'''
    <body name="{name}_base" pos="{bx} {by} {bz}">
      <geom type="cylinder" size="0.060 0.050" rgba="0.18 0.18 0.18 1"/>
      <body name="{name}_link1" pos="0 0 0.090">
        <joint name="{name}_j1" type="hinge" axis="0 0 1" range="-3.14 3.14" damping="6"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.120" size="0.050" rgba="{spec.color}"/>
        <body name="{name}_link2" pos="0 0 0.120">
          <joint name="{name}_j2" type="hinge" axis="0 1 0" range="-2.4 2.4" damping="6"/>
          <geom type="capsule" fromto="0 0 0 0.280 0 0" size="0.045" rgba="{spec.color}"/>
          <body name="{name}_link3" pos="0.280 0 0">
            <joint name="{name}_j3" type="hinge" axis="0 1 0" range="-2.8 2.8" damping="5"/>
            <geom type="capsule" fromto="0 0 0 0.260 0 0" size="0.040" rgba="{spec.color}"/>
            <body name="{name}_link4" pos="0.260 0 0">
              <joint name="{name}_j4" type="hinge" axis="0 1 0" range="-3.14 3.14" damping="4"/>
              <geom type="capsule" fromto="0 0 0 0.180 0 0" size="0.032" rgba="{spec.color}"/>
              <body name="{name}_link5" pos="0.180 0 0">
                <joint name="{name}_j5" type="hinge" axis="0 0 1" range="-3.14 3.14" damping="3"/>
                <geom type="capsule" fromto="0 0 0 0.120 0 0" size="0.026" rgba="{spec.color}"/>
                <body name="{name}_link6" pos="0.120 0 0">
                  <joint name="{name}_j6" type="hinge" axis="0 1 0" range="-3.14 3.14" damping="2"/>
                  <geom type="capsule" fromto="0 0 0 0.140 0 0" size="0.022" rgba="{spec.color}"/>
{tool}
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
'''

    def _payload_xml(self) -> str:
        payload_specs = [
            (-0.010, -0.020, 0.000),
            (0.012, 0.018, 0.010),
            (0.000, 0.000, 0.022),
            (-0.015, 0.020, -0.005),
        ]
        chunks = []
        for i, (x, y, z) in enumerate(payload_specs):
            chunks.append(f'''
      <body name="payload{i}" pos="{x:.4f} {y:.4f} {z:.4f}">
        <joint name="payload{i}_x" type="slide" axis="1 0 0" limited="true" range="-0.025 0.025" damping="14"/>
        <joint name="payload{i}_y" type="slide" axis="0 1 0" limited="true" range="-0.040 0.040" damping="14"/>
        <joint name="payload{i}_z" type="slide" axis="0 0 1" limited="true" range="-0.015 0.050" damping="16"/>
        <geom type="sphere" size="0.016" mass="0.90" rgba="0.62 0.22 0.20 1" contype="0" conaffinity="0"/>
      </body>''')
        return "\n".join(chunks)

    def _equality_xml(self) -> str:
        lines = []
        for offset in self.connect_offsets:
            anchor = self.band_pos0 + np.array([offset[0], offset[1], -self.band_half[2]], dtype=np.float64)
            lines.append(
                f'    <connect body1="band" body2="sack" anchor="{anchor[0]:.6f} {anchor[1]:.6f} {anchor[2]:.6f}" solref="0.025 1" solimp="0.90 0.97 0.002"/>'
            )
        return "\n".join(lines)

    def _scene_xml(self) -> str:
        left = self._arm_body_xml(LEFT_ARM)
        right = self._arm_body_xml(RIGHT_ARM)
        payload_xml = self._payload_xml()
        equality_xml = self._equality_xml()
        return f'''<mujoco model="dual_2f_scoop_0330_soft">
  <compiler angle="radian" coordinate="local" meshdir="object" inertiafromgeom="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" iterations="80" tolerance="1e-8"/>
  <size memory="256M" nconmax="4000"/>

  <asset>
    <mesh name="sack_mesh" file="sack9.obj" scale="{self.sack_mesh_scale:.3f} {self.sack_mesh_scale:.3f} {self.sack_mesh_scale:.3f}"/>
  </asset>

  <default>
    <joint armature="0.03" frictionloss="0.02"/>
    <geom condim="4" friction="1.0 0.05 0.01" margin="0.002"/>
    <position kp="900" ctrllimited="true" forcelimited="true" forcerange="-500 500"/>
  </default>

  <worldbody>
    <light pos="0 0 2.2" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="ground" type="plane" size="2 2 0.1" rgba="0.92 0.92 0.92 1"/>
    <camera name="overview" pos="1.35 0 0.95" xyaxes="0 1 0 -0.45 0 0.89"/>

{left}
{right}

    <body name="band" pos="{self.band_pos0[0]:.6f} {self.band_pos0[1]:.6f} {self.band_pos0[2]:.6f}">
      <freejoint name="band_free"/>
      <geom name="grasp_band" type="box" size="{self.band_half[0]:.6f} {self.band_half[1]:.6f} {self.band_half[2]:.6f}" mass="0.10" rgba="0.50 0.36 0.20 1" friction="2.0 0.05 0.01"/>
      <site name="handle_grasp_site" pos="0 0 0" size="0.008" rgba="1 0 0 1"/>
    </body>

    <body name="sack" pos="{self.sack_pos0[0]:.6f} {self.sack_pos0[1]:.6f} {self.sack_pos0[2]:.6f}">
      <freejoint name="sack_free"/>
      <geom name="sack_visual" type="mesh" mesh="sack_mesh" rgba="0.90 0.85 0.70 0.70" contype="0" conaffinity="0"/>
      <geom name="sack_bottom" type="box" pos="0 0 -0.055" size="0.060 0.090 0.005" mass="0.10" rgba="0.75 0.67 0.52 0.20"/>
      <geom name="sack_wall_l" type="box" pos="0 0.085 0.010" size="0.060 0.005 0.070" mass="0.08" rgba="0.75 0.67 0.52 0.18"/>
      <geom name="sack_wall_r" type="box" pos="0 -0.085 0.010" size="0.060 0.005 0.070" mass="0.08" rgba="0.75 0.67 0.52 0.18"/>
      <geom name="sack_wall_f" type="box" pos="0.055 0 0.010" size="0.005 0.090 0.070" mass="0.08" rgba="0.75 0.67 0.52 0.18"/>
      <geom name="sack_wall_b" type="box" pos="-0.055 0 0.010" size="0.005 0.090 0.070" mass="0.08" rgba="0.75 0.67 0.52 0.18"/>
      <site name="sack_com_site" pos="0 0 0.01" size="0.006" rgba="0 1 0 1"/>
{payload_xml}
    </body>
  </worldbody>

  <equality>
{equality_xml}
  </equality>

  <actuator>
    <position name="act_left_j1" joint="left_j1" ctrlrange="-3.14 3.14" forcerange="-220 220"/>
    <position name="act_left_j2" joint="left_j2" ctrlrange="-2.4 2.4" forcerange="-220 220"/>
    <position name="act_left_j3" joint="left_j3" ctrlrange="-2.8 2.8" forcerange="-220 220"/>
    <position name="act_left_j4" joint="left_j4" ctrlrange="-3.14 3.14" forcerange="-320 320"/>
    <position name="act_left_j5" joint="left_j5" ctrlrange="-3.14 3.14" forcerange="-240 240"/>
    <position name="act_left_j6" joint="left_j6" ctrlrange="-3.14 3.14" forcerange="-240 240"/>
    <position name="act_right_j1" joint="right_j1" ctrlrange="-3.14 3.14" forcerange="-220 220"/>
    <position name="act_right_j2" joint="right_j2" ctrlrange="-2.4 2.4" forcerange="-220 220"/>
    <position name="act_right_j3" joint="right_j3" ctrlrange="-2.8 2.8" forcerange="-220 220"/>
    <position name="act_right_j4" joint="right_j4" ctrlrange="-3.14 3.14" forcerange="-320 320"/>
    <position name="act_right_j5" joint="right_j5" ctrlrange="-3.14 3.14" forcerange="-240 240"/>
    <position name="act_right_j6" joint="right_j6" ctrlrange="-3.14 3.14" forcerange="-240 240"/>
    <position name="act_left_finger_l" joint="left_finger_l" ctrlrange="0.012 0.040" kp="1400" forcerange="-240 240"/>
    <position name="act_left_finger_r" joint="left_finger_r" ctrlrange="0.012 0.040" kp="1400" forcerange="-240 240"/>
  </actuator>
</mujoco>
'''

    def _write_scene_xml(self):
        self.scene_path.write_text(self._scene_xml(), encoding="utf-8")

    def _joint_qadr(self, joint_name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        return int(self.model.jnt_qposadr[jid])

    def _joint_dadr(self, joint_name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        return int(self.model.jnt_dofadr[jid])

    def _act_id(self, act_name: str) -> int:
        return int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name))

    def _set_arm_qpos(self, arm: str, q: np.ndarray):
        names = self.left_joints if arm == "left" else self.right_joints
        for name, val in zip(names, q):
            self.data.qpos[self._joint_qadr(name)] = float(val)

    def _arm_qpos(self, arm: str) -> np.ndarray:
        names = self.left_joints if arm == "left" else self.right_joints
        return np.array([self.data.qpos[self._joint_qadr(name)] for name in names], dtype=np.float64)

    def _set_left_gripper(self, opening: float):
        opening = float(np.clip(opening, self.gripper_close, self.gripper_open))
        for act_name in self.left_finger_acts:
            self.data.ctrl[self._act_id(act_name)] = opening

    def _sync_ctrl_to_qpos(self):
        for name in self.left_joints:
            self.data.ctrl[self._act_id(f"act_{name}")] = self.data.qpos[self._joint_qadr(name)]
        for name in self.right_joints:
            self.data.ctrl[self._act_id(f"act_{name}")] = self.data.qpos[self._joint_qadr(name)]
        self._set_left_gripper(self.gripper_open)

    def set_arm_target(self, arm: str, q: np.ndarray):
        acts = self.left_actuators if arm == "left" else self.right_actuators
        for act_name, val in zip(acts, q):
            self.data.ctrl[self._act_id(act_name)] = float(val)

    def site_pos(self, site_name: str) -> np.ndarray:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        return self.data.site_xpos[sid].copy()

    def body_pos(self, body_name: str) -> np.ndarray:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.data.xpos[bid].copy()

    def body_quat(self, body_name: str) -> np.ndarray:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.data.xquat[bid].copy()

    def solve_ik_position(self, arm: str, site_name: str, target_pos: np.ndarray, iters: int = 80) -> np.ndarray:
        q = self._arm_qpos(arm).copy()
        joint_names = self.left_joints if arm == "left" else self.right_joints
        qadr = [self._joint_qadr(n) for n in joint_names]
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        q_backup = self.data.qpos.copy()
        try:
            for _ in range(iters):
                for adr, val in zip(qadr, q):
                    self.data.qpos[adr] = val
                mujoco.mj_forward(self.model, self.data)
                cur = self.data.site_xpos[sid].copy()
                err = np.asarray(target_pos, dtype=np.float64) - cur
                if np.linalg.norm(err) < 2e-3:
                    break
                mujoco.mj_jacSite(self.model, self.data, jacp, jacr, sid)
                cols = [self._joint_dadr(n) for n in joint_names]
                Jq = np.stack([jacp[:, c] for c in cols], axis=1)
                dq = Jq.T @ np.linalg.solve(Jq @ Jq.T + 1e-3 * np.eye(3), err)
                q += 0.65 * dq
        finally:
            self.data.qpos[:] = q_backup
            mujoco.mj_forward(self.model, self.data)
        return q

    def step(self, n: int = 1):
        for _ in range(int(n)):
            if self.grasp_active:
                grip = self.site_pos("left_grip_site")
                target_pos = grip - self.grasp_offset
                self.data.qpos[self.band_free_qadr:self.band_free_qadr + 3] = target_pos
                self.data.qpos[self.band_free_qadr + 3:self.band_free_qadr + 7] = self.band_quat_hold
                self.data.qvel[self.band_free_dadr:self.band_free_dadr + 6] = 0.0
            mujoco.mj_step(self.model, self.data)

    def move_arm_smooth(self, arm: str, q_target: np.ndarray, steps: int = 240):
        q0 = self._arm_qpos(arm)
        for alpha in np.linspace(0.0, 1.0, max(2, steps)):
            q = (1.0 - alpha) * q0 + alpha * q_target
            self.set_arm_target(arm, q)
            self.step(1)

    def close_left_gripper(self, steps: int = 140):
        for val in np.linspace(self.gripper_open, self.gripper_close, max(2, steps)):
            self._set_left_gripper(float(val))
            self.step(1)

    def try_attach_grasp(self, threshold: float = 0.045) -> bool:
        grip = self.site_pos("left_grip_site")
        band_pos = self.body_pos("band")
        handle = self.site_pos("handle_grasp_site")
        if float(np.linalg.norm(grip - handle)) > threshold:
            return False
        self.grasp_active = True
        self.grasp_offset = grip - band_pos
        self.band_quat_hold = self.body_quat("band")
        return True

    def settle(self, steps: int = 300):
        self.step(steps)

    def scripted_demo(self):
        self.set_arm_target("left", self.left_q_home)
        self.set_arm_target("right", self.right_support)
        self._set_left_gripper(self.gripper_open)
        self.step(260)
        handle = self.site_pos("handle_grasp_site")
        pre = handle + np.array([0.0, 0.0, 0.10], dtype=np.float64)
        grasp = handle + np.array([0.0, 0.0, 0.030], dtype=np.float64)
        q_pre = self.solve_ik_position("left", "left_grip_site", pre)
        q_grasp = self.solve_ik_position("left", "left_grip_site", grasp)
        self.move_arm_smooth("left", q_pre, steps=220)
        self.move_arm_smooth("left", q_grasp, steps=220)
        self.close_left_gripper(steps=120)
        attached = self.try_attach_grasp()
        lift = grasp + np.array([0.0, 0.0, 0.12], dtype=np.float64)
        q_lift = self.solve_ik_position("left", "left_grip_site", lift)
        self.move_arm_smooth("left", q_lift, steps=260)
        self.step(120)
        return attached

    def payload_center_of_mass(self) -> np.ndarray:
        names = ["sack", "payload0", "payload1", "payload2", "payload3"]
        masses = []
        centers = []
        for name in names:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            masses.append(float(self.model.body_mass[bid]))
            centers.append(self.data.xipos[bid].copy())
        masses = np.asarray(masses, dtype=np.float64)
        centers = np.asarray(centers, dtype=np.float64)
        return (centers * masses[:, None]).sum(axis=0) / masses.sum()

    def support_load_index(self) -> dict:
        left = sum(abs(float(self.data.actuator_force[self._act_id(a)])) for a in self.left_actuators)
        right = sum(abs(float(self.data.actuator_force[self._act_id(a)])) for a in self.right_actuators)
        total = max(left + right, 1e-9)
        return {
            "left_abs": left,
            "right_abs": right,
            "left_share": left / total,
            "right_share": right / total,
        }


def print_summary(sim: MuJoCoDualSoftSack0330, attached=None, sack0=None, com0=None):
    print("scene_xml", sim.scene_path)
    print("mujoco_version", getattr(mujoco, "__version__", "unknown"))
    print("mode", "compliant_proxy")
    if attached is not None:
        print("attached", attached)
    if sack0 is not None:
        print("sack_delta", (sim.body_pos("sack") - sack0).tolist())
    if com0 is not None:
        print("com_delta", (sim.payload_center_of_mass() - com0).tolist())
    load = sim.support_load_index()
    print("load_share", {k: (v if "abs" in k else round(100.0 * v, 2)) for k, v in load.items()})


def run_gui(sim: MuJoCoDualSoftSack0330):
    print("scene_xml", sim.scene_path)
    print("mujoco_version", getattr(mujoco, "__version__", "unknown"))
    print("mode", "compliant_proxy")
    mujoco.viewer.launch_from_path(str(sim.scene_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Launch the MuJoCo viewer.")
    parser.add_argument("--headless", action="store_true", help="Run without the MuJoCo viewer.")
    parser.add_argument("--demo", action="store_true", help="Run the scripted grasp/lift demo.")
    args = parser.parse_args()

    sim = MuJoCoDualSoftSack0330()
    use_gui = args.gui or not args.headless
    if use_gui:
        run_gui(sim)
        return

    sim.settle(320)
    sack0 = sim.body_pos("sack").copy()
    com0 = sim.payload_center_of_mass().copy()
    attached = None
    if args.demo:
        attached = sim.scripted_demo()
    print_summary(sim, attached=attached, sack0=sack0, com0=com0)


if __name__ == "__main__":
    main()

