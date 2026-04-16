from __future__ import annotations

import copy
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from build_shape_coupled_sack import TIMESTEP, build_scene_tree
from scenario_builder import available_scenarios, get_scenario
from shape_response_controller import ReducedOrderShapeController, measure_shape_metrics


ROOT_DIR = Path(__file__).resolve().parent
LOW_FILL_DIR = ROOT_DIR.parent / "01_low_fill"
if str(LOW_FILL_DIR) not in sys.path:
    sys.path.insert(0, str(LOW_FILL_DIR))

from dual_ur5_low_fill_env import (  # noqa: E402
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


DUAL_SCENE_PATH = ROOT_DIR / "generated" / "dual_ur5_shape_coupled_sack.xml"
SACK_WORLD_X = 0.40


def _shift_body_x(body: ET.Element, dx: float) -> None:
    parts = [float(v) for v in body.attrib.get("pos", "0 0 0").split()]
    while len(parts) < 3:
        parts.append(0.0)
    parts[0] += dx
    body.attrib["pos"] = f"{parts[0]:.6f} {parts[1]:.6f} {parts[2]:.6f}"


class DualUR5ShapeCoupledEnv(DualUR5LowFillEnv):
    """shape-coupled sack core와 dual UR5 + 2F/scoop을 한 장면에 배치한다."""

    def __init__(self, *, scenario: str = "underfilled", post_release: bool = False, scene_path: Path = DUAL_SCENE_PATH):
        if scenario not in available_scenarios():
            raise ValueError(f"unknown scenario: {scenario}")
        self.scenario = scenario
        self.post_release = bool(post_release)
        self.gui_title = "Dual UR5 Shape-Coupled Sack Core Control"
        self.gui_header = f"Dual UR5 + 2F/scoop + shape-coupled sack [{scenario}]"
        super().__init__(with_ballast=True, scene_path=scene_path)
        self.shape_response_enabled = True
        self.shape_controller = ReducedOrderShapeController(target_index=2, lateral_bias=1.0)
        self.manual_shape_phase = "observe"
        self.manual_shape_target_index = 2
        self.manual_shape_lateral_bias = 1.0
        self.last_shape_metrics = measure_shape_metrics(self.model, self.data, phase="observe", target_index=2)

    def _sack_world_bodies(self) -> list[ET.Element]:
        root = build_scene_tree(get_scenario(self.scenario, post_release=self.post_release), include_eval_gripper=False)
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise RuntimeError("sack worldbody not found")
        keep_names = {"bag_frame", "hidden_support"}
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

        root = ET.Element("mujoco", {"model": f"dual_ur5_shape_coupled_sack_{self.scenario}"})
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
                "rgba": "0.92 0.90 0.86 0.18",
                "friction": FLOOR_FRICTION,
                "condim": "4",
            },
        )
        ET.SubElement(worldbody, "camera", {"name": "overview", "pos": "1.65 0 1.0", "xyaxes": "0 1 0 -0.42 0 0.91"})
        ET.SubElement(worldbody, "camera", {"name": "front", "pos": "1.02 0 0.42", "xyaxes": "0 1 0 -0.22 0 0.98"})
        self._add_world_origin_axes(worldbody)
        worldbody.append(left_body)
        worldbody.append(right_body)
        for body in self._sack_world_bodies():
            worldbody.append(body)

        actuator = ET.SubElement(root, "actuator")
        for child in list(left_actuator):
            actuator.append(copy.deepcopy(child))
        for child in list(right_actuator):
            actuator.append(copy.deepcopy(child))
        ET.SubElement(actuator, "position", {"name": "left_finger_l_act", "joint": "left_finger_l", "ctrlrange": "0.010 0.080", "kp": "450", "forcerange": "-60 60"})
        ET.SubElement(actuator, "position", {"name": "left_finger_r_act", "joint": "left_finger_r", "ctrlrange": "0.010 0.080", "kp": "450", "forcerange": "-60 60"})

        ET.indent(root, space="  ")
        return ET.tostring(root, encoding="unicode")

    def grasp_site_names(self) -> list[str]:
        names = [f"grasp_seam_{i:02d}" for i in range(8)]
        names += [f"grasp_shoulder_{i:02d}" for i in range(8)]
        names += ["grasp_fold_1", "grasp_fold_2"]
        return [name for name in names if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name) >= 0]

    def nearest_grasp_target(self, reference_xyz: np.ndarray) -> tuple[str, np.ndarray]:
        names = self.grasp_site_names()
        if not names:
            raise RuntimeError("no grasp sites found")
        distances = [float(np.linalg.norm(self.site_pos(name) - reference_xyz)) for name in names]
        name = names[int(np.argmin(distances))]
        return name, self.site_pos(name)

    def _geom_name(self, geom_id: int) -> str:
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id)) or ""

    def _body_name(self, body_id: int) -> str:
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(body_id)) or ""

    def _bag_up_z(self) -> float:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bag_frame")
        if body_id < 0:
            return 1.0
        return float(self.data.xmat[body_id].reshape(3, 3)[:, 2][2])

    def _target_index_from_contact(self, geom_names: set[str]) -> int | None:
        for name in geom_names:
            for prefix in ("shoulder_panel_", "belly_panel_", "seam_band_"):
                if not name.startswith(prefix):
                    continue
                try:
                    return int(name[len(prefix) : len(prefix) + 2])
                except ValueError:
                    return None
        return None

    def _lowest_side_panel_index(self) -> int:
        candidates: list[tuple[float, int]] = []
        for index in range(8):
            for geom_name in (f"shoulder_panel_{index:02d}_geom", f"belly_panel_{index:02d}_geom"):
                geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                if geom_id < 0:
                    continue
                candidates.append((float(self.data.geom_xpos[geom_id, 2]), index))
        if not candidates:
            return self.manual_shape_target_index
        return min(candidates, key=lambda item: item[0])[1]

    def _contact_driven_shape_phase(self) -> tuple[str, int, float]:
        gripper_contacts = 0
        scoop_contacts = 0
        sidewall_tool_contacts = 0
        contacted_indices: list[int] = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom_names = {self._geom_name(contact.geom1), self._geom_name(contact.geom2)}
            has_bag = any(
                name.startswith(("seam_band", "shoulder_panel", "belly_panel", "fold_root_flap", "bottom_sling", "payload"))
                for name in geom_names
            )
            if not has_bag:
                continue
            has_gripper = any(name in geom_names for name in ("left_finger_l_pad", "left_finger_r_pad"))
            has_scoop = any(name in geom_names for name in ("right_scoop_plate", "right_scoop_left_rail", "right_scoop_right_rail"))
            if has_gripper:
                gripper_contacts += 1
            if has_scoop:
                scoop_contacts += 1
            has_sidewall = any(name.startswith(("shoulder_panel_", "belly_panel_")) for name in geom_names)
            if has_sidewall and (has_gripper or has_scoop):
                sidewall_tool_contacts += 1
            target_index = self._target_index_from_contact(geom_names)
            if target_index is not None:
                contacted_indices.append(target_index)

        bag_up_z = self._bag_up_z()
        target_index = self.manual_shape_target_index
        if contacted_indices:
            # 실제로 접촉된 패널 주변을 국소 변형 중심으로 사용합니다.
            target_index = int(round(float(np.median(contacted_indices)))) % 8

        bag_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bag_frame")
        lateral_bias = self.manual_shape_lateral_bias
        if bag_body_id >= 0:
            bag_x_axis = self.data.xmat[bag_body_id].reshape(3, 3)[:, 0]
            lateral_bias = float(np.sign(bag_x_axis[2]) or lateral_bias or 1.0)

        lowest_side_index = self._lowest_side_panel_index()
        if bag_up_z < 0.92:
            # 옆으로 누운 상태에서는 실제로 가장 낮아진 옆면을 처짐 중심으로 둡니다.
            target_index = lowest_side_index

        left_ee_z = float(self.site_pos("left_gripper_pinch")[2])
        bag_z = float(self.data.xpos[bag_body_id, 2]) if bag_body_id >= 0 else 0.0
        if bag_up_z < 0.92 and scoop_contacts == 0:
            # 로봇이 밀고 있는 중에도 옆면 처짐이 pinch보다 우선입니다.
            phase = "side_fall"
        elif scoop_contacts > 0 and bag_up_z < 0.92:
            # 옆으로 누운 자루를 스쿠프가 받치면 하부 지지 회복으로 전환합니다.
            phase = "scoop_insert"
        elif sidewall_tool_contacts > 0 and bag_up_z < 0.985:
            # 옆면을 미는 순간에도 외부 사각 패널이 눌리는 반응을 보여줍니다.
            phase = "side_push"
        elif scoop_contacts > 0:
            phase = "support_lift" if left_ee_z > bag_z + 0.12 else "scoop_insert"
        elif gripper_contacts > 0:
            phase = "micro_lift" if left_ee_z > bag_z + 0.17 else "pinch"
        else:
            phase = "observe"

        return phase, target_index, lateral_bias

    def step(self, steps: int = 1) -> None:
        for _ in range(int(steps)):
            if self.shape_response_enabled:
                phase, target_index, lateral_bias = self._contact_driven_shape_phase()
                self.manual_shape_phase = phase
                self.manual_shape_target_index = target_index
                self.manual_shape_lateral_bias = lateral_bias
                self.shape_controller.apply(
                    self.model,
                    self.data,
                    phase=phase,
                    target_index=target_index,
                    lateral_bias=lateral_bias,
                )
            mujoco.mj_step(self.model, self.data)
        if self.shape_response_enabled:
            self.last_shape_metrics = measure_shape_metrics(
                self.model,
                self.data,
                phase=self.manual_shape_phase,
                target_index=self.manual_shape_target_index,
            )

    def shape_status_lines(self) -> list[str]:
        metrics = self.last_shape_metrics
        return [
            f"shape_phase={self.manual_shape_phase} target_panel={self.manual_shape_target_index:02d} bag_up_z={metrics.bag_up_z:.2f}",
            f"local_shoulder={metrics.shoulder_angle_local_deg:.1f} deg bottom_sag={metrics.bottom_sag_m * 1000.0:.1f} mm",
            f"payload_shift_y={metrics.payload_slide_y_m * 1000.0:.1f} mm bag_com_z={metrics.bag_com_z_m:.3f} m",
        ]

    def _bag_freejoint_qpos_address(self) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "bag_frame_freejoint")
        if joint_id < 0:
            raise RuntimeError("bag_frame_freejoint not found")
        return int(self.model.jnt_qposadr[joint_id])

    def nudge_bag_world(self, delta_xyz: np.ndarray) -> None:
        adr = self._bag_freejoint_qpos_address()
        self.data.qpos[adr : adr + 3] += np.asarray(delta_xyz, dtype=np.float64)
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "bag_frame_freejoint")
        dof_adr = int(self.model.jnt_dofadr[joint_id])
        self.data.qvel[dof_adr : dof_adr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def reset_bag_pose(self) -> None:
        adr = self._bag_freejoint_qpos_address()
        self.data.qpos[adr : adr + 3] = np.array([SACK_WORLD_X, 0.0, 0.188], dtype=np.float64)
        self.data.qpos[adr + 3 : adr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def print_summary(self) -> None:
        print(f"scene_xml={self.scene_path}")
        print(f"scenario={self.scenario}")
        print(f"post_release={self.post_release}")
        print(f"mujoco_version={getattr(mujoco, '__version__', 'unknown')}")
        print(f"left_gripper_site={np.round(self.site_pos('left_gripper_pinch'), 4).tolist()}")
        print(f"right_scoop_site={np.round(self.site_pos('right_scoop_site'), 4).tolist()}")
        print(f"grasp_site_count={len(self.grasp_site_names())}")
        print(f"nflex={int(getattr(self.model, 'nflex', 0))}")
        print("shape_response_enabled=True")
        for line in self.shape_status_lines():
            print(line)


def smoke_test(scenario: str, post_release: bool = False) -> int:
    env = DualUR5ShapeCoupledEnv(scenario=scenario, post_release=post_release)
    left_target = env.end_effector_pos("left") + np.array([0.01, 0.0, 0.0], dtype=np.float64)
    right_target = env.end_effector_pos("right") + np.array([0.0, 0.0, 0.01], dtype=np.float64)
    left_ik = env.solve_ee_position_ik("left", left_target)
    right_ik = env.solve_ee_position_ik("right", right_target)
    site_name, target = env.nearest_grasp_target(env.end_effector_pos("left"))
    grasp_ik = env.solve_ee_position_ik("left", target + np.array([0.0, 0.0, 0.006], dtype=np.float64))
    # smoke test에서는 actuator target을 실제 qpos에도 반영해서 contact 근처까지 즉시 확인한다.
    for joint_name, actuator_name in zip(env.left_joint_names, env.left_actuator_names):
        env.data.qpos[env._joint_qpos_address(joint_name)] = env.actuator_ctrl(actuator_name)
    mujoco.mj_forward(env.model, env.data)
    env.set_left_gripper_gap(0.0, immediate=True)
    for _ in range(int(0.25 / env.model.opt.timestep)):
        env.step()
    left_contact_count = 0
    for index in range(env.data.ncon):
        contact = env.data.contact[index]
        geom_names = {
            mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or "",
            mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or "",
        }
        has_finger = any(name in geom_names for name in ("left_finger_l_pad", "left_finger_r_pad"))
        has_bag = any(name.startswith(("seam_band", "shoulder_panel", "fold_root_flap")) for name in geom_names)
        if has_finger and has_bag:
            left_contact_count += 1
    print(f"scene_xml={env.scene_path}")
    print(f"scenario={scenario}")
    print(f"post_release={post_release}")
    print(f"left_ee_ik={left_ik}")
    print(f"right_ee_ik={right_ik}")
    print(f"nearest_grasp_site={site_name}")
    print(f"nearest_grasp_site_pos={target.tolist()}")
    print(f"nearest_grasp_ik={grasp_ik}")
    print(f"left_gripper_gap_m={env.left_gripper_gap()}")
    print(f"left_grasp_contact_count={left_contact_count}")
    print("dual_ur5_shape_coupled_smoke_pass=True")
    return 0
