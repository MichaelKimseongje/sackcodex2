from __future__ import annotations

from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from build_shape_coupled_sack import SEGMENT_COUNT


@dataclass
class ShapeMetrics:
    """자루 surrogate의 저차원 형상 상태를 기록합니다."""

    phase: str
    time_s: float
    upper_half_width: float
    lower_half_width: float
    shoulder_angle_mean_deg: float
    shoulder_angle_local_deg: float
    belly_angle_mean_deg: float
    bottom_sag_m: float
    payload_slide_x_m: float
    payload_slide_y_m: float
    payload_slide_z_m: float
    payload_world_z_m: float
    bag_com_x_m: float
    bag_com_y_m: float
    bag_com_z_m: float
    gripper_bag_contact_count: int
    scoop_bag_contact_count: int
    shape_change_score: float
    support_margin_proxy: float
    bag_up_z: float
    side_fall_bias: float

    def row(self) -> dict[str, float | int | str]:
        return asdict(self)


def _obj_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    return mujoco.mj_name2id(model, obj_type, name)


def _joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> float:
    joint_id = _obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        return 0.0
    return float(data.qpos[model.jnt_qposadr[joint_id]])


def _joint_qvel(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> float:
    joint_id = _obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        return 0.0
    return float(data.qvel[model.jnt_dofadr[joint_id]])


def _apply_joint_pd(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    name: str,
    target: float,
    *,
    kp: float,
    kd: float,
) -> None:
    joint_id = _obj_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        return
    qpos_addr = int(model.jnt_qposadr[joint_id])
    dof_addr = int(model.jnt_dofadr[joint_id])
    err = target - float(data.qpos[qpos_addr])
    vel = float(data.qvel[dof_addr])
    data.qfrc_applied[dof_addr] += kp * err - kd * vel


def _apply_world_force(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    force_xyz: np.ndarray,
    tracker: np.ndarray | None = None,
) -> None:
    body_id = _obj_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return
    force = np.asarray(force_xyz, dtype=np.float64)
    data.xfrc_applied[body_id, 0:3] += force
    if tracker is not None:
        tracker[body_id, 0:3] += force


def _geom_pos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    geom_id = _obj_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if geom_id < 0:
        return np.zeros(3, dtype=np.float64)
    return data.geom_xpos[geom_id].copy()


def _body_pos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    body_id = _obj_id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        return np.zeros(3, dtype=np.float64)
    return data.xpos[body_id].copy()


def _contact_counts(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[int, int]:
    gripper_count = 0
    scoop_count = 0
    for index in range(data.ncon):
        contact = data.contact[index]
        names = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or "",
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or "",
        }
        has_bag = any(
            name.startswith(("seam_band", "shoulder_panel", "belly_panel", "fold_root_flap", "bottom_sling", "payload"))
            for name in names
        )
        if not has_bag:
            continue
        if any(name in names for name in ("gripper_left_mocap_pad", "gripper_right_mocap_pad")):
            gripper_count += 1
        if any(name in names for name in ("scoop_plate", "scoop_back_lip")):
            scoop_count += 1
    return gripper_count, scoop_count


def measure_shape_metrics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    phase: str,
    target_index: int = 2,
    initial: ShapeMetrics | None = None,
) -> ShapeMetrics:
    shoulder_pos = np.array([_geom_pos(model, data, f"shoulder_panel_{i:02d}_geom") for i in range(SEGMENT_COUNT)])
    belly_pos = np.array([_geom_pos(model, data, f"belly_panel_{i:02d}_geom") for i in range(SEGMENT_COUNT)])
    upper_width = float(max(np.ptp(shoulder_pos[:, 0]), np.ptp(shoulder_pos[:, 1])) + 0.022)
    lower_width = float(max(np.ptp(belly_pos[:, 0]), np.ptp(belly_pos[:, 1])) + 0.030)
    shoulder_angles = np.array([_joint_qpos(model, data, f"shoulder_panel_{i:02d}_hinge") for i in range(SEGMENT_COUNT)])
    belly_angles = np.array([_joint_qpos(model, data, f"belly_panel_{i:02d}_hinge") for i in range(SEGMENT_COUNT)])
    target_index = int(target_index) % SEGMENT_COUNT
    bag_id = _obj_id(model, mujoco.mjtObj.mjOBJ_BODY, "bag_frame")
    payload_pos = _body_pos(model, data, "payload_main")
    gripper_contacts, scoop_contacts = _contact_counts(model, data)
    bottom_sag = _joint_qpos(model, data, "bottom_sling_sag")
    support_margin = float(scoop_contacts * 0.010 - max(0.0, -bottom_sag) * 0.35)
    if bag_id >= 0:
        bag_com = data.subtree_com[bag_id].copy()
        bag_up_z = float(data.xmat[bag_id].reshape(3, 3)[:, 2][2])
        bag_x_axis = data.xmat[bag_id].reshape(3, 3)[:, 0]
        side_fall_bias = float(np.sign(bag_x_axis[2]) or 1.0)
    else:
        bag_com = np.zeros(3, dtype=np.float64)
        bag_up_z = 1.0
        side_fall_bias = 1.0
    local_change = (
        abs(float(np.rad2deg(shoulder_angles[target_index])))
        + abs(float(np.rad2deg(np.mean(belly_angles))))
        + abs(bottom_sag) * 800.0
        + abs(_joint_qpos(model, data, "payload_main_y")) * 450.0
    )
    if initial is not None:
        local_change += abs(upper_width - initial.upper_half_width) * 120.0
        local_change += abs(lower_width - initial.lower_half_width) * 120.0
    return ShapeMetrics(
        phase=phase,
        time_s=float(data.time),
        upper_half_width=upper_width,
        lower_half_width=lower_width,
        shoulder_angle_mean_deg=float(np.rad2deg(np.mean(np.abs(shoulder_angles)))),
        shoulder_angle_local_deg=float(np.rad2deg(shoulder_angles[target_index])),
        belly_angle_mean_deg=float(np.rad2deg(np.mean(np.abs(belly_angles)))),
        bottom_sag_m=bottom_sag,
        payload_slide_x_m=_joint_qpos(model, data, "payload_main_x"),
        payload_slide_y_m=_joint_qpos(model, data, "payload_main_y"),
        payload_slide_z_m=_joint_qpos(model, data, "payload_main_z"),
        payload_world_z_m=float(payload_pos[2]),
        bag_com_x_m=float(bag_com[0]),
        bag_com_y_m=float(bag_com[1]),
        bag_com_z_m=float(bag_com[2]),
        gripper_bag_contact_count=gripper_contacts,
        scoop_bag_contact_count=scoop_contacts,
        shape_change_score=float(local_change),
        support_margin_proxy=support_margin,
        bag_up_z=bag_up_z,
        side_fall_bias=side_fall_bias,
    )


class ReducedOrderShapeController:
    """접촉/지지 상태를 저차원 패널 모드와 payload 이동으로 연결합니다.

    이 컨트롤러는 정확한 천 재료 모델이 아니라, 자루에서 중요한
    국소 눌림, 하부 처짐, 내부 하중 편차를 안정적으로 재현하기 위한
    task-driven shape coupling입니다.
    """

    def __init__(self, *, target_index: int = 2, lateral_bias: float = 1.0):
        self.target_index = int(target_index) % SEGMENT_COUNT
        self.lateral_bias = float(np.sign(lateral_bias) or 1.0)
        self._last_xfrc_applied: np.ndarray | None = None

    def _neighbors(self) -> tuple[int, int, int]:
        return ((self.target_index - 1) % SEGMENT_COUNT, self.target_index, (self.target_index + 1) % SEGMENT_COUNT)

    def apply(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        phase: str,
        target_index: int | None = None,
        lateral_bias: float | None = None,
    ) -> None:
        if target_index is not None:
            self.target_index = int(target_index) % SEGMENT_COUNT
        if lateral_bias is not None and abs(lateral_bias) > 1e-9:
            self.lateral_bias = float(np.sign(lateral_bias))
        data.qfrc_applied[:] = 0.0
        if self._last_xfrc_applied is None or self._last_xfrc_applied.shape != data.xfrc_applied.shape:
            self._last_xfrc_applied = np.zeros_like(data.xfrc_applied)
        else:
            # 이전 step에서 이 컨트롤러가 넣은 외력만 제거합니다.
            # MuJoCo viewer의 Ctrl+마우스 perturb 외력은 지우지 않아야 물체 드래그가 됩니다.
            data.xfrc_applied[:] -= self._last_xfrc_applied
            self._last_xfrc_applied.fill(0.0)
        shoulder_targets = np.zeros(SEGMENT_COUNT, dtype=np.float64)
        belly_targets = np.zeros(SEGMENT_COUNT, dtype=np.float64)
        payload_target = np.zeros(3, dtype=np.float64)
        bottom_target = 0.0
        shoulder_kp = 78.0
        shoulder_kd = 4.4
        belly_kp = 62.0
        belly_kd = 4.0
        bottom_kp = 360.0
        bottom_kd = 18.0
        payload_kp = 120.0
        payload_kd = 8.0

        if phase in {"pinch", "tug_test"}:
            # 그리퍼가 누른 국소 patch만 더 접히고, 주변 patch는 약하게 따라옵니다.
            for rank, idx in enumerate(self._neighbors()):
                shoulder_targets[idx] = np.deg2rad(-26.0 if rank == 1 else -10.0)
                belly_targets[idx] = np.deg2rad(-7.0)
            payload_target[1] = 0.006 * self.lateral_bias
            payload_target[2] = -0.004
        elif phase == "micro_lift":
            # 상단은 따라 올라가지만 하부와 payload는 늦게 따라오므로 sag가 생깁니다.
            for rank, idx in enumerate(self._neighbors()):
                shoulder_targets[idx] = np.deg2rad(-34.0 if rank == 1 else -16.0)
                belly_targets[idx] = np.deg2rad(-13.0)
            payload_target[1] = 0.016 * self.lateral_bias
            payload_target[2] = -0.018
            bottom_target = -0.048
        elif phase == "scoop_insert":
            # 스쿠프가 받치기 시작하면 하부 처짐이 줄고 belly panel이 받침 방향으로 정렬됩니다.
            belly_targets[:] = np.deg2rad(-6.0)
            payload_target[1] = 0.008 * self.lateral_bias
            payload_target[2] = -0.006
            bottom_target = -0.016
        elif phase == "side_push":
            # 옆면을 밀면 완전히 넘어지기 전에도 local sidewall이 먼저 눌립니다.
            for idx in range(SEGMENT_COUNT):
                delta = min((idx - self.target_index) % SEGMENT_COUNT, (self.target_index - idx) % SEGMENT_COUNT)
                weight = max(0.0, 1.0 - 0.42 * float(delta))
                shoulder_targets[idx] = np.deg2rad(-4.0 - 18.0 * weight)
                belly_targets[idx] = np.deg2rad(-5.0 - 22.0 * weight)
                down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
                _apply_world_force(model, data, f"shoulder_panel_{idx:02d}", down * (0.9 + 2.1 * weight), self._last_xfrc_applied)
                _apply_world_force(model, data, f"belly_panel_{idx:02d}", down * (1.2 + 3.6 * weight), self._last_xfrc_applied)
            payload_target[1] = 0.014 * self.lateral_bias
            payload_target[2] = -0.004
            bottom_target = -0.006
            shoulder_kp = 48.0
            shoulder_kd = 8.5
            belly_kp = 50.0
            belly_kd = 8.5
            bottom_kp = 90.0
            payload_kp = 75.0
        elif phase == "support_lift":
            belly_targets[:] = np.deg2rad(-1.5)
            shoulder_targets[:] = np.deg2rad(-2.0)
            payload_target[1] = 0.004 * self.lateral_bias
            payload_target[2] = -0.002
            bottom_target = -0.001
        elif phase == "side_fall":
            # 자루가 옆으로 누우면 처짐은 자루 로컬축이 아니라 월드 중력 방향으로 발생해야 합니다.
            # 따라서 큰 hinge target으로 억지 변형을 만들지 않고, 낮은 쪽 panel들에 아래 방향 하중을 줍니다.
            for idx in range(SEGMENT_COUNT):
                delta = min((idx - self.target_index) % SEGMENT_COUNT, (self.target_index - idx) % SEGMENT_COUNT)
                low_side_weight = max(0.0, 1.0 - 0.33 * float(delta))
                shoulder_targets[idx] = np.deg2rad(-5.0 - 24.0 * low_side_weight)
                belly_targets[idx] = np.deg2rad(-6.0 - 24.0 * low_side_weight)
                down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
                _apply_world_force(model, data, f"shoulder_panel_{idx:02d}", down * (1.4 + 4.8 * low_side_weight), self._last_xfrc_applied)
                _apply_world_force(model, data, f"belly_panel_{idx:02d}", down * (2.0 + 7.0 * low_side_weight), self._last_xfrc_applied)
            _apply_world_force(model, data, "bottom_sling", np.array([0.0, 0.0, -8.0], dtype=np.float64), self._last_xfrc_applied)
            _apply_world_force(model, data, "payload_main", np.array([0.0, 0.0, -6.0], dtype=np.float64), self._last_xfrc_applied)
            payload_target[1] = 0.024 * self.lateral_bias
            payload_target[2] = -0.004
            bottom_target = -0.006
            shoulder_kp = 42.0
            shoulder_kd = 11.0
            belly_kp = 42.0
            belly_kd = 11.0
            bottom_kp = 70.0
            bottom_kd = 24.0
            payload_kp = 65.0
            payload_kd = 12.0

        for i, target in enumerate(shoulder_targets):
            _apply_joint_pd(model, data, f"shoulder_panel_{i:02d}_hinge", float(target), kp=shoulder_kp, kd=shoulder_kd)
        for i, target in enumerate(belly_targets):
            _apply_joint_pd(model, data, f"belly_panel_{i:02d}_hinge", float(target), kp=belly_kp, kd=belly_kd)
        _apply_joint_pd(model, data, "bottom_sling_sag", float(bottom_target), kp=bottom_kp, kd=bottom_kd)
        _apply_joint_pd(model, data, "payload_main_x", float(payload_target[0]), kp=payload_kp, kd=payload_kd)
        _apply_joint_pd(model, data, "payload_main_y", float(payload_target[1]), kp=payload_kp, kd=payload_kd)
        _apply_joint_pd(model, data, "payload_main_z", float(payload_target[2]), kp=payload_kp + 20.0, kd=payload_kd + 1.0)


def joint_snapshot(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, float]:
    """디버깅용으로 주요 shape DOF를 사람이 읽기 쉬운 dict로 반환합니다."""
    values: dict[str, float] = {}
    for i in range(SEGMENT_COUNT):
        values[f"shoulder_{i:02d}_deg"] = float(np.rad2deg(_joint_qpos(model, data, f"shoulder_panel_{i:02d}_hinge")))
        values[f"belly_{i:02d}_deg"] = float(np.rad2deg(_joint_qpos(model, data, f"belly_panel_{i:02d}_hinge")))
    for name in ("bottom_sling_sag", "payload_main_x", "payload_main_y", "payload_main_z"):
        values[name] = _joint_qpos(model, data, name)
    return values
