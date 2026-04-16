from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
LOW_FILL_DIR = ROOT_DIR.parent / "01_low_fill"
if str(LOW_FILL_DIR) not in sys.path:
    sys.path.insert(0, str(LOW_FILL_DIR))

from dual_ur5_low_fill_env import (  # noqa: E402
    DualUR5LowFillEnv,
    FRAME_DIAGINERTIA,
    FRAME_MASS,
    LOW_FILL_WORLD_POS,
    SHELL_CONDIM,
    SHELL_DAMPING,
    SHELL_FRICTION,
    SHELL_RADIUS,
    SHELL_SOLIMP,
    SHELL_SOLREF,
    collect_shell_body_ids,
)
from generate_sack_mesh import (  # noqa: E402
    BAG_MASS,
    CONTENT_CLUMP_JOINT_RANGES,
    CONTENT_CLUMP_NAMES,
    CONTENT_SUPPORT_BODY,
    CONTENT_SUPPORT_JOINT_RANGES,
    GENERATED_DIR,
    RING_COUNT,
    SCENARIOS,
    add_three_clump_content_support,
    available_content_cases,
    flex_elements,
    format_points,
    make_sack_points,
)


DUAL_TOP_GRASP_SCENE_PATH = GENERATED_DIR / "dual_ur5_top_grasp.xml"


class DualUR5TopGraspEnv(DualUR5LowFillEnv):
    """기존 Dual UR5 GUI에서 top grasp scenario shell을 직접 조작하기 위한 환경."""

    gui_title = "Dual UR5 Top-Grasp Sack Control"
    gui_header = "Dual UR5 + top-grasp sack scenario controller"

    def __init__(
        self,
        *,
        scenario_name: str = "simple_fold",
        content_case: str = "underfilled",
        scene_path: Path = DUAL_TOP_GRASP_SCENE_PATH,
        with_content_support: bool = True,
    ):
        if scenario_name not in SCENARIOS:
            raise ValueError(f"unknown scenario: {scenario_name}")
        if content_case not in available_content_cases():
            raise ValueError(f"unknown content case: {content_case}")
        self.scenario_name = scenario_name
        self.content_case = content_case
        self.with_content_support = with_content_support
        self._scenario_labels: list[str] = []
        super().__init__(with_ballast=False, scene_path=scene_path)

    def _load_ur5e_dual_parts(self):
        parts = super()._load_ur5e_dual_parts()
        ur_asset, ur_default, left_body, right_body, left_actuator, right_actuator = parts
        for robot_body in (left_body, right_body):
            for geom in robot_body.iter("geom"):
                # top grasp demo에서는 팔 링크가 자루를 치지 않게 하고, tool contact만 남긴다.
                geom.attrib["contype"] = "0"
                geom.attrib["conaffinity"] = "0"
        return ur_asset, ur_default, left_body, right_body, left_actuator, right_actuator

    def _attach_2f_gripper(self, left_body: ET.Element) -> None:
        super()._attach_2f_gripper(left_body)
        for geom in left_body.iter("geom"):
            if geom.attrib.get("name") not in {"left_finger_l_pad", "left_finger_r_pad"}:
                continue
            # flex shell이 gripper close 순간 튀지 않도록 Robotiq 2F-140 pad 접촉은 부드럽게 둔다.
            geom.attrib["friction"] = "2.2 0.10 0.008"
            geom.attrib["margin"] = "0.0015"
            geom.attrib["solref"] = "0.034 1"
            geom.attrib["solimp"] = "0.76 0.93 0.001"
        for finger_body in left_body.iter("body"):
            body_name = finger_body.attrib.get("name", "")
            if body_name not in {"left_finger_l_body", "left_finger_r_body"}:
                continue
            if any((geom.attrib.get("name") or "").endswith("_inward_lip_0") for geom in finger_body.iter("geom")):
                continue
            inner_y_sign = -1.0 if body_name.endswith("_l_body") else 1.0
            for lip_index, z_sign in enumerate((-1.0, 1.0)):
                # jaw 내부로 아주 얕은 턱을 추가해 flex shell이 닫힘 순간 옆으로 빠지는 현상을 줄인다.
                ET.SubElement(
                    finger_body,
                    "geom",
                    {
                        "name": f"{body_name}_inward_lip_{lip_index}",
                        "type": "box",
                        "pos": f"0 {inner_y_sign * 0.0125:.6f} {0.044 + z_sign * 0.058:.6f}",
                        "size": "0.022 0.0020 0.0070",
                        "rgba": "0.08 0.08 0.08 1",
                        "friction": "2.2 0.10 0.008",
                        "condim": "4",
                        "margin": "0.0015",
                        "solref": "0.034 1",
                        "solimp": "0.76 0.93 0.001",
                    },
                )

    def _low_fill_body(self) -> ET.Element:
        points, labels = make_sack_points(self.scenario_name)
        self._scenario_labels = labels
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

        # 분석 라벨은 site 색으로만 보여준다. 성공/실패나 hold 판정에는 쓰지 않는다.
        for point_index in range(0, 1 + RING_COUNT):
            point = points[point_index]
            label = labels[point_index] if point_index < len(labels) else "other"
            rgba = {
                "seam": "1.0 0.45 0.05 1",
                "fold": "0.15 0.70 1.0 1",
                "plain_top": "0.10 0.90 0.25 1",
            }.get(label, "0.6 0.6 0.6 1")
            ET.SubElement(
                bag_frame,
                "site",
                {
                    "name": f"bag_grasp_site_{point_index:02d}",
                    "pos": f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}",
                    "size": "0.008",
                    "rgba": rgba,
                },
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
                "rgba": "0.74 0.57 0.34 0.62",
                "point": format_points(points),
                "element": flex_elements(),
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
        if self.with_content_support:
            add_three_clump_content_support(bag_frame, self.content_case)
        return bag_frame

    def grasp_site_names(self) -> list[str]:
        return [f"bag_grasp_site_{index:02d}" for index in range(0, 1 + RING_COUNT)]

    def shell_point_body_id(self, point_index: int) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"bag_shell_{point_index}")
        if body_id < 0:
            raise KeyError(f"shell body not found: bag_shell_{point_index}")
        return int(body_id)

    def nearest_grasp_target(self, reference_xyz: np.ndarray) -> tuple[str, np.ndarray]:
        """현재 변형된 shell point 중 접근 가능한 상단 후보 위치를 반환한다."""
        reference_xyz = np.asarray(reference_xyz, dtype=np.float64)
        shell_ids = collect_shell_body_ids(self.model)
        if not shell_ids:
            site_name = self.nearest_grasp_site_name(reference_xyz)
            return site_name, self.site_pos(site_name)

        candidate_items: list[tuple[str, np.ndarray]] = []
        for point_index in range(0, min(1 + RING_COUNT, len(shell_ids))):
            label = self._scenario_labels[point_index] if point_index < len(self._scenario_labels) else "other"
            if label not in {"seam", "fold", "plain_top"}:
                continue
            body_id = self.shell_point_body_id(point_index)
            candidate_items.append((f"bag_shell_{point_index:02d}_{label}", self.data.xpos[body_id].copy()))

        if not candidate_items:
            site_name = self.nearest_grasp_site_name(reference_xyz)
            return site_name, self.site_pos(site_name)

        distances = [float(np.linalg.norm(position - reference_xyz)) for _name, position in candidate_items]
        return candidate_items[int(np.argmin(distances))]

    def _top_label_targets(self, allowed_labels: set[str]) -> list[tuple[str, int, np.ndarray]]:
        candidates: list[tuple[str, int, np.ndarray]] = []
        for point_index in range(0, 1 + RING_COUNT):
            label = self._scenario_labels[point_index] if point_index < len(self._scenario_labels) else "other"
            if label not in allowed_labels:
                continue
            body_id = self.shell_point_body_id(point_index)
            candidates.append((f"bag_shell_{point_index:02d}_{label}", body_id, self.data.xpos[body_id].copy()))
        return candidates

    def candidate_targets_for_label(self, requested_label: str) -> list[tuple[str, int, np.ndarray]]:
        """label은 후보 선택용일 뿐이며 성공/실패 판정에는 쓰지 않는다."""
        grasp_labels = {"seam", "fold", "plain_top"}
        if requested_label == "auto":
            return self._top_label_targets(grasp_labels)
        if requested_label not in grasp_labels:
            raise ValueError(f"unknown requested label: {requested_label}")

        candidates: list[tuple[str, int, np.ndarray]] = []
        for point_index in range(0, 1 + RING_COUNT):
            label = self._scenario_labels[point_index] if point_index < len(self._scenario_labels) else "other"
            if label != requested_label:
                continue
            body_id = self.shell_point_body_id(point_index)
            candidates.append((f"bag_shell_{point_index:02d}_{label}", body_id, self.data.xpos[body_id].copy()))
        return candidates

    def target_for_label(self, requested_label: str) -> tuple[str, int, np.ndarray]:
        candidates = self.candidate_targets_for_label(requested_label)
        if not candidates:
            candidates = self._top_label_targets({"seam", "fold", "plain_top"})
        if not candidates:
            name, pos = self.nearest_grasp_target(self.end_effector_pos("left"))
            point_index = 0
            if name.startswith("bag_shell_"):
                try:
                    point_index = int(name.split("_")[2])
                except (IndexError, ValueError):
                    point_index = 0
            return name, self.shell_point_body_id(point_index), pos

        reference = self.end_effector_pos("left")
        bag_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bag_frame")
        bag_xy = self.data.xpos[bag_body_id, :2].copy() if bag_body_id >= 0 else np.zeros(2, dtype=np.float64)

        def graspability_score(item: tuple[str, int, np.ndarray]) -> float:
            _name, _body_id, pos = item
            distance_to_ee = float(np.linalg.norm(pos - reference))
            floor_clearance = max(float(pos[2]) - 0.035, 0.0)
            near_floor_penalty = max(0.060 - float(pos[2]), 0.0)
            far_from_bag_center = float(np.linalg.norm(pos[:2] - bag_xy))
            # 자루가 쓰러진 경우에도 현재 위치 기준으로 높고, 바닥에서 떨어져 있고,
            # 왼쪽 2F gripper가 무리 없이 접근할 수 있는 patch를 우선한다.
            return (
                3.00 * float(pos[2])
                + 0.80 * floor_clearance
                - 0.35 * distance_to_ee
                - 2.20 * near_floor_penalty
                - 0.08 * far_from_bag_center
            )

        name, body_id, pos = max(candidates, key=graspability_score)
        return name, body_id, pos.copy()

    def content_support_body_id(self) -> int:
        central_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, CONTENT_CLUMP_NAMES[0])
        if central_id >= 0:
            return int(central_id)
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, CONTENT_SUPPORT_BODY)

    def content_support_body_ids(self) -> list[int]:
        body_ids = [
            int(body_id)
            for body_name in CONTENT_CLUMP_NAMES
            if (body_id := mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)) >= 0
        ]
        if body_ids:
            return body_ids
        single_id = self.content_support_body_id()
        return [int(single_id)] if single_id >= 0 else []

    def content_support_pos(self) -> np.ndarray | None:
        body_ids = self.content_support_body_ids()
        if not body_ids:
            return None
        masses = np.asarray([self.model.body_mass[body_id] for body_id in body_ids], dtype=np.float64)
        positions = np.asarray([self.data.xpos[body_id].copy() for body_id in body_ids], dtype=np.float64)
        total_mass = float(np.sum(masses))
        if total_mass <= 1e-9:
            return np.mean(positions, axis=0)
        return (positions * masses[:, None]).sum(axis=0) / total_mass

    def set_content_bias_from_grasp(self, grasp_world_xyz: np.ndarray) -> dict[str, float] | None:
        """잡는 위치 대비 내부 질량이 한쪽으로 치우치는 편심 충진 surrogate를 적용한다."""
        if not self.content_support_body_ids():
            return None
        bag_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "bag_frame")
        if bag_body_id < 0:
            return None

        grasp_world_xyz = np.asarray(grasp_world_xyz, dtype=np.float64)
        bag_xyz = self.data.xpos[bag_body_id].copy()
        local_xy = grasp_world_xyz[:2] - bag_xyz[:2]
        base_shift = {
            "x": float(-0.28 * local_xy[0]),
            "y": float(-0.40 * local_xy[1]),
            "z": 0.0,
        }
        desired: dict[str, float] = {}
        clump_scale = {"central": 0.45, "left": 0.70, "right": 0.85}
        used_clumps = False
        for clump_role in ("central", "left", "right"):
            body_name = f"bag_content_clump_{clump_role}"
            if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name) < 0:
                continue
            used_clumps = True
            for axis_name in ("x", "y", "z"):
                low, high = CONTENT_CLUMP_JOINT_RANGES[clump_role][axis_name]
                value = float(np.clip(base_shift[axis_name] * clump_scale[clump_role], low, high))
                desired[f"{clump_role}_{axis_name}"] = value
                joint_name = f"{body_name}_{axis_name}"
                qpos_address = self._joint_qpos_address(joint_name)
                dof_address = self._joint_dof_address(joint_name)
                self.data.qpos[qpos_address] = value
                self.data.qvel[dof_address] = 0.0
        if not used_clumps:
            desired = {
                "x": float(np.clip(base_shift["x"], *CONTENT_SUPPORT_JOINT_RANGES["x"])),
                "y": float(np.clip(base_shift["y"], *CONTENT_SUPPORT_JOINT_RANGES["y"])),
                "z": 0.0,
            }
            for axis_name, value in desired.items():
                joint_name = f"{CONTENT_SUPPORT_BODY}_{axis_name}"
                try:
                    qpos_address = self._joint_qpos_address(joint_name)
                    dof_address = self._joint_dof_address(joint_name)
                except KeyError:
                    continue
                self.data.qpos[qpos_address] = value
                self.data.qvel[dof_address] = 0.0
        mujoco.mj_forward(self.model, self.data)

        content_pos = self.content_support_pos()
        if content_pos is None:
            return desired
        return {
            "bias_x": base_shift["x"],
            "bias_y": base_shift["y"],
            "bias_z": base_shift["z"],
            "content_case": self.content_case,
            "grasp_to_content_x": float(content_pos[0] - grasp_world_xyz[0]),
            "grasp_to_content_y": float(content_pos[1] - grasp_world_xyz[1]),
            "grasp_to_content_z": float(content_pos[2] - grasp_world_xyz[2]),
        }

    def print_summary(self) -> None:
        super().print_summary()
        print(f"top_grasp_scenario={self.scenario_name}")
        print(f"content_case={self.content_case}")
        print(f"with_content_support={self.with_content_support}")
        print("site_color_legend=seam:orange, fold:blue, plain_top:green")
