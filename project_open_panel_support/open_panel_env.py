"""Open-panel support-state prototype runtime utilities.

여기서는 MuJoCo 모델을 로드하고, guarded grasp와 support-state 지표를
Python callback 방식으로 계산한다. soft body나 DEM은 사용하지 않는다.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np

from builder import PROJECT_DIR, SCENE_XML, write_scene_xml


OUT_DIR = PROJECT_DIR / "out"


@dataclass
class SupportMetrics:
    """지원상태 평가를 위한 한 시점의 지표."""

    time: float
    sag_index: float
    effective_com_offset: float
    effective_com_offset_x: float
    effective_com_offset_y: float
    scoop_load_transfer: float
    peel_ratio: float
    support_margin: float
    insertion_depth: float
    scoop_contact_force: float
    guarded_grasp_active: bool
    guarded_grasp_accepted: bool
    left_contact_present: bool
    right_contact_present: bool
    trapped_patch_count: int
    bilateral_contact_balance: float
    tangential_slip_proxy: float
    finger_gap: float


class OpenPanelSupportEnv:
    """Open-panel MuJoCo 환경 wrapper."""

    hidden_mass_names = ("hidden_mass_C", "hidden_mass_L", "hidden_mass_R")
    target_panel_geoms = (
        "geom_top_panel",
        "geom_left_side_panel",
        "geom_right_side_panel",
        "geom_left_bottom_panel",
        "geom_right_bottom_panel",
    )
    capture_sites = ("site_top_center", "site_left_side_center", "site_right_side_center")

    def __init__(self, xml_path: Path | str | None = None):
        if xml_path is None:
            xml_path = write_scene_xml()
        self.xml_path = Path(xml_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        self._ids = self._build_name_index()
        self.guard_eq_id = self._id(mujoco.mjtObj.mjOBJ_EQUALITY, "guarded_grasp_connect")
        self.scoop_act_id = self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, "scoop_insert_act")
        self.gripper_lift_act_id = self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_lift_act")
        self.left_act_id = self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, "left_finger_close_act")
        self.right_act_id = self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, "right_finger_close_act")
        self.scoop_joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, "scoop_insert")

        self.scoop_geom_ids = {
            self._id(mujoco.mjtObj.mjOBJ_GEOM, "geom_scoop_plate"),
            self._id(mujoco.mjtObj.mjOBJ_GEOM, "geom_scoop_lip"),
        }
        self.left_pad_geom_id = self._id(mujoco.mjtObj.mjOBJ_GEOM, "geom_left_finger_pad")
        self.right_pad_geom_id = self._id(mujoco.mjtObj.mjOBJ_GEOM, "geom_right_finger_pad")
        self.target_geom_ids = {self._id(mujoco.mjtObj.mjOBJ_GEOM, name) for name in self.target_panel_geoms}

        self.total_hidden_weight = self._hidden_mass_total() * 9.81
        self.nominal_top_bottom_gap = 0.0
        self.nominal_top_z = 0.0
        self.nominal_bottom_z = 0.0
        self.prev_top_site = None
        self.prev_metric_time = 0.0
        self.last_tangential_slip_mm = 0.0
        self.contact_persistence_s = 0.0
        self.guarded_grasp_accepted = False
        self.guarded_grasp_active = False

        self.reset()

    def _build_name_index(self) -> dict[str, dict[str, int]]:
        """디버깅용 이름 인덱스."""

        return {
            "body": {self.model.body(i).name: i for i in range(self.model.nbody)},
            "joint": {self.model.joint(i).name: i for i in range(self.model.njnt)},
            "geom": {self.model.geom(i).name: i for i in range(self.model.ngeom)},
            "site": {self.model.site(i).name: i for i in range(self.model.nsite)},
        }

    def _id(self, obj_type: mujoco.mjtObj, name: str) -> int:
        obj_id = mujoco.mj_name2id(self.model, obj_type, name)
        if obj_id < 0:
            raise KeyError(f"MuJoCo object not found: {name}")
        return obj_id

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.eq_active[self.guard_eq_id] = 0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.guarded_grasp_active = False
        self.guarded_grasp_accepted = False
        self.contact_persistence_s = 0.0
        self.prev_top_site = self.site_world("site_top_center").copy()
        self.prev_metric_time = float(self.data.time)
        self.set_nominal_shape_reference()

    def step(self, n: int = 1, guarded_update: bool = True) -> None:
        for _ in range(n):
            if guarded_update:
                self.update_guarded_grasp()
            mujoco.mj_step(self.model, self.data)

    def settle(self, seconds: float = 0.4) -> None:
        steps = max(1, int(seconds / self.model.opt.timestep))
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
        self.set_nominal_shape_reference()

    def set_controls(self, left_close: float | None = None, right_close: float | None = None,
                     scoop_depth: float | None = None, gripper_lift: float | None = None) -> None:
        """position actuator target을 설정한다."""

        if left_close is not None:
            self.data.ctrl[self.left_act_id] = float(np.clip(left_close, 0.0, 0.060))
        if right_close is not None:
            self.data.ctrl[self.right_act_id] = float(np.clip(right_close, 0.0, 0.060))
        if scoop_depth is not None:
            self.data.ctrl[self.scoop_act_id] = float(np.clip(scoop_depth, 0.0, 0.265))
        if gripper_lift is not None:
            self.data.ctrl[self.gripper_lift_act_id] = float(np.clip(gripper_lift, -0.035, 0.090))

    def site_world(self, name: str) -> np.ndarray:
        sid = self._id(mujoco.mjtObj.mjOBJ_SITE, name)
        return self.data.site_xpos[sid].copy()

    def body_world(self, name: str) -> np.ndarray:
        bid = self._id(mujoco.mjtObj.mjOBJ_BODY, name)
        return self.data.xpos[bid].copy()

    def bag_local(self, world_pos: np.ndarray) -> np.ndarray:
        """world 좌표를 bag_frame local 좌표로 변환한다."""

        bid = self._id(mujoco.mjtObj.mjOBJ_BODY, "bag_frame")
        origin = self.data.xpos[bid]
        rot = self.data.xmat[bid].reshape(3, 3)
        return rot.T @ (world_pos - origin)

    def site_local(self, name: str) -> np.ndarray:
        return self.bag_local(self.site_world(name))

    def set_nominal_shape_reference(self) -> None:
        top_z = self.site_local("site_top_center")[2]
        bottom_z = self._bottom_local_z()
        self.nominal_top_z = float(top_z)
        self.nominal_bottom_z = float(bottom_z)
        self.nominal_top_bottom_gap = float(top_z - bottom_z)

    def _bottom_local_z(self) -> float:
        left = self.site_local("site_left_bottom_center")[2]
        right = self.site_local("site_right_bottom_center")[2]
        return float(0.5 * (left + right))

    def _hidden_mass_total(self) -> float:
        return sum(self.model.body_mass[self._id(mujoco.mjtObj.mjOBJ_BODY, name)] for name in self.hidden_mass_names)

    def hidden_mass_com_world(self) -> np.ndarray:
        total = 0.0
        weighted = np.zeros(3)
        for name in self.hidden_mass_names:
            bid = self._id(mujoco.mjtObj.mjOBJ_BODY, name)
            mass = self.model.body_mass[bid]
            weighted += mass * self.data.xipos[bid]
            total += mass
        return weighted / max(total, 1e-9)

    def _contact_force_sum(self, geom_ids: Iterable[int]) -> float:
        geom_ids = set(geom_ids)
        total = 0.0
        force = np.zeros(6)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 in geom_ids or contact.geom2 in geom_ids:
                mujoco.mj_contactForce(self.model, self.data, i, force)
                total += max(0.0, float(force[0]))
        return total

    def _contact_force_between(self, geom_a: int, geom_b_set: set[int]) -> float:
        total = 0.0
        force = np.zeros(6)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            pair = {contact.geom1, contact.geom2}
            if geom_a in pair and pair.intersection(geom_b_set):
                mujoco.mj_contactForce(self.model, self.data, i, force)
                total += max(0.0, float(force[0]))
        return total

    def _finger_gap(self) -> float:
        left_y = self.site_world("site_left_pad_inner")[1]
        right_y = self.site_world("site_right_pad_inner")[1]
        return abs(left_y - right_y)

    def _capture_patch_count(self) -> int:
        left = self.site_world("site_left_pad_inner")
        right = self.site_world("site_right_pad_inner")
        y_min, y_max = sorted((left[1], right[1]))
        center = self.site_world("site_gripper_center")
        count = 0
        for name in self.capture_sites:
            p = self.site_world(name)
            inside_y = y_min - 0.010 <= p[1] <= y_max + 0.010
            inside_x = abs(p[0] - center[0]) <= 0.055
            inside_z = abs(p[2] - center[2]) <= 0.080
            if inside_x and inside_y and inside_z:
                count += 1
        return count

    def _update_slip_proxy(self) -> float:
        top = self.site_world("site_top_center")
        now = float(self.data.time)
        if self.prev_top_site is None:
            self.prev_top_site = top.copy()
            self.prev_metric_time = now
            self.last_tangential_slip_mm = 0.0
            return 0.0
        dt = max(now - self.prev_metric_time, self.model.opt.timestep)
        horizontal_delta = top[:2] - self.prev_top_site[:2]
        # 한 스텝 displacement를 mm proxy로 기록한다. 너무 큰 값은 slip/escape 신호로 해석한다.
        self.last_tangential_slip_mm = float(np.linalg.norm(horizontal_delta) * 1000.0)
        self.prev_top_site = top.copy()
        self.prev_metric_time = now
        return self.last_tangential_slip_mm

    def guarded_grasp_conditions(self) -> dict[str, float | bool | int]:
        """Latch 활성화 전후 공통으로 쓰는 guarded grasp 조건."""

        left_force = self._contact_force_between(self.left_pad_geom_id, self.target_geom_ids)
        right_force = self._contact_force_between(self.right_pad_geom_id, self.target_geom_ids)
        left_contact = left_force > 0.03
        right_contact = right_force > 0.03
        trapped = self._capture_patch_count()
        gap = self._finger_gap()
        slip = self.last_tangential_slip_mm
        balance = min(left_force, right_force) / max(max(left_force, right_force), 1e-6)
        peel_ratio = self._peel_ratio()
        return {
            "left_contact_present": left_contact,
            "right_contact_present": right_contact,
            "trapped_patch_count": trapped,
            "finger_gap": gap,
            "tangential_slip_proxy": slip,
            "bilateral_contact_balance": balance,
            "peel_ratio": peel_ratio,
        }

    def update_guarded_grasp(self) -> None:
        """조건을 만족할 때만 selected local patch surrogate connect를 켠다."""

        self._update_slip_proxy()
        cond = self.guarded_grasp_conditions()
        bilateral = bool(cond["left_contact_present"] and cond["right_contact_present"])
        if bilateral and cond["trapped_patch_count"] >= 1:
            self.contact_persistence_s += self.model.opt.timestep
        else:
            self.contact_persistence_s = 0.0

        if not self.guarded_grasp_active:
            accepted = (
                bilateral
                and cond["trapped_patch_count"] >= 1
                and 0.018 <= cond["finger_gap"] <= 0.150
                and cond["bilateral_contact_balance"] >= 0.18
                and cond["tangential_slip_proxy"] <= 3.5
                and cond["peel_ratio"] <= 2.8
                and self.contact_persistence_s >= 0.040
            )
            if accepted:
                self.data.eq_active[self.guard_eq_id] = 1
                self.guarded_grasp_active = True
                self.guarded_grasp_accepted = True

        else:
            # Release 조건: gripper가 열렸거나 양손 접촉이 오래 깨지면 latch를 해제한다.
            if cond["finger_gap"] > 0.165 or (not bilateral and self.contact_persistence_s <= 0.0):
                self.data.eq_active[self.guard_eq_id] = 0
                self.guarded_grasp_active = False

    def _peel_ratio(self) -> float:
        top_z = self.site_local("site_top_center")[2]
        bottom_z = self._bottom_local_z()
        top_delta = top_z - self.nominal_top_z
        bottom_delta = bottom_z - self.nominal_bottom_z
        return float(abs(top_delta - bottom_delta) / (abs(top_delta) + 0.005))

    def compute_metrics(self) -> SupportMetrics:
        """현재 data에서 support-state 출력 지표를 계산한다."""

        top_z = self.site_local("site_top_center")[2]
        bottom_z = self._bottom_local_z()
        # 열린 패널 prototype에서는 gap 증가/감소 모두 형상 변화 신호로 본다.
        sag_index = abs((top_z - bottom_z) - self.nominal_top_bottom_gap) * 1000.0

        com_world = self.hidden_mass_com_world()
        com_local = self.bag_local(com_world)
        com_offset_x = float(com_local[0] * 1000.0)
        com_offset_y = float(com_local[1] * 1000.0)
        effective_com_offset = float(np.linalg.norm(com_local[:2]) * 1000.0)

        scoop_force = self._contact_force_sum(self.scoop_geom_ids)
        scoop_load_transfer = scoop_force / max(self.total_hidden_weight, 1e-9)

        support_margin = self._support_margin_mm(com_world)
        insertion_depth = self._joint_qpos("scoop_insert")
        cond = self.guarded_grasp_conditions()

        return SupportMetrics(
            time=float(self.data.time),
            sag_index=float(sag_index),
            effective_com_offset=effective_com_offset,
            effective_com_offset_x=com_offset_x,
            effective_com_offset_y=com_offset_y,
            scoop_load_transfer=float(scoop_load_transfer),
            peel_ratio=float(self._peel_ratio()),
            support_margin=float(support_margin),
            insertion_depth=float(insertion_depth),
            scoop_contact_force=float(scoop_force),
            guarded_grasp_active=bool(self.guarded_grasp_active),
            guarded_grasp_accepted=bool(self.guarded_grasp_accepted),
            left_contact_present=bool(cond["left_contact_present"]),
            right_contact_present=bool(cond["right_contact_present"]),
            trapped_patch_count=int(cond["trapped_patch_count"]),
            bilateral_contact_balance=float(cond["bilateral_contact_balance"]),
            tangential_slip_proxy=float(cond["tangential_slip_proxy"]),
            finger_gap=float(cond["finger_gap"]),
        )

    def _support_margin_mm(self, com_world: np.ndarray) -> float:
        scoop_center = self.site_world("site_scoop_center")
        half_x = 0.245
        half_y = 0.058
        margin_x = half_x - abs(com_world[0] - scoop_center[0])
        margin_y = half_y - abs(com_world[1] - scoop_center[1])
        return float(min(margin_x, margin_y) * 1000.0)

    def _joint_qpos(self, joint_name: str) -> float:
        jid = self._id(mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        adr = self.model.jnt_qposadr[jid]
        return float(self.data.qpos[adr])

    def runtime_inventory(self) -> dict[str, object]:
        """로드된 model 기준 body/joint/site inventory."""

        body_names = [self.model.body(i).name for i in range(self.model.nbody)]
        joint_names = [self.model.joint(i).name for i in range(self.model.njnt)]
        site_names = [self.model.site(i).name for i in range(self.model.nsite)]
        legacy_names = [name for name in body_names if any(k in name for k in ("rim_ring", "upper_skirt", "lower_skirt", "bottom_cradle"))]
        return {
            "xml_path": str(self.xml_path),
            "body_count": self.model.nbody,
            "joint_count": self.model.njnt,
            "site_count": self.model.nsite,
            "panel_bodies": [name for name in body_names if name.endswith("_panel")],
            "hidden_mass_bodies": [name for name in body_names if name.startswith("hidden_mass_")],
            "scoop_bodies": [name for name in body_names if "scoop" in name],
            "gripper_bodies": [name for name in body_names if "gripper" in name or "finger" in name],
            "legacy_body_names_still_remaining": legacy_names,
            "body_names": body_names,
            "joint_names": joint_names,
            "site_names": site_names,
        }


def run_scripted_trial(env: OpenPanelSupportEnv, seconds: float = 3.8, sample_every: int = 10) -> list[SupportMetrics]:
    """간단한 close -> scoop insert -> hold sequence를 실행한다."""

    rows: list[SupportMetrics] = []
    env.reset()
    env.set_controls(left_close=0.0, right_close=0.0, scoop_depth=0.0, gripper_lift=0.0)
    env.settle(0.25)

    total_steps = int(seconds / env.model.opt.timestep)
    for step in range(total_steps):
        t = step * env.model.opt.timestep
        if t < 0.45:
            close = 0.0
            scoop = 0.0
            lift = 0.0
        elif t < 1.10:
            alpha = (t - 0.45) / 0.65
            close = 0.052 * alpha
            scoop = 0.0
            lift = 0.0
        elif t < 1.55:
            close = 0.052
            scoop = 0.0
            lift = 0.035 * ((t - 1.10) / 0.45)
        elif t < 2.75:
            alpha = (t - 1.55) / 1.20
            close = 0.052
            scoop = 0.205 * alpha
            lift = 0.035
        else:
            close = 0.052
            scoop = 0.205
            lift = 0.035

        env.set_controls(left_close=close, right_close=close, scoop_depth=scoop, gripper_lift=lift)
        env.step(1, guarded_update=True)

        if step % sample_every == 0:
            rows.append(env.compute_metrics())

    return rows


def save_metrics(rows: list[SupportMetrics], out_dir: Path = OUT_DIR) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "eval_metrics.csv"
    json_path = out_dir / "eval_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    final = asdict(rows[-1])
    final["max_sag_index"] = max(row.sag_index for row in rows)
    final["max_scoop_load_transfer"] = max(row.scoop_load_transfer for row in rows)
    final["min_support_margin"] = min(row.support_margin for row in rows)
    final["guarded_grasp_ever_accepted"] = any(row.guarded_grasp_accepted for row in rows)
    final["samples"] = len(rows)
    json_path.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    return csv_path, json_path


def check_finite(env: OpenPanelSupportEnv) -> bool:
    return bool(np.all(np.isfinite(env.data.qpos)) and np.all(np.isfinite(env.data.qvel)))
