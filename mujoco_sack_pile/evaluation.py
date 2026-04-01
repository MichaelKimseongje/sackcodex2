from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import mujoco
import numpy as np


@dataclass
class EpisodeMetrics:
    """에피소드 평가 결과."""

    support_state_score: float
    support_success: bool
    scoop_insertion_depth: float
    micro_lift_stability: float
    slip_distance: float
    tilt_deg: float
    dropped: bool
    contact_count: int
    failure_tags: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


class Evaluator:
    """연구 질문에 맞는 휴리스틱 평가 함수를 제공한다."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model

    def evaluate(
        self,
        data: mujoco.MjData,
        target_name: str,
        target_origin_xy: np.ndarray,
        pre_lift_pos: np.ndarray | None = None,
        pre_lift_scoop_pos: np.ndarray | None = None,
    ) -> EpisodeMetrics:
        target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, target_name)
        scoop_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "scoop_tool")
        support_geom_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{target_name}_core_geom"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{target_name}_bottom_support"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{target_name}_side_left"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{target_name}_side_right"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{target_name}_front_panel"),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{target_name}_back_panel"),
        }
        scoop_geom_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("scoop_plate", "scoop_lip", "scoop_left_rail", "scoop_right_rail", "scoop_backstop")
        }

        target_pos = data.xpos[target_body_id].copy()
        target_xmat = data.xmat[target_body_id].reshape(3, 3)
        scoop_pos = data.xpos[scoop_body_id].copy()
        scoop_xmat = data.xmat[scoop_body_id].reshape(3, 3)

        relative = scoop_xmat.T @ (target_pos - scoop_pos)
        insertion_depth = float(np.clip(relative[0] + 0.020, 0.0, 0.18))
        slip_distance = float(np.linalg.norm(target_pos[:2] - target_origin_xy[:2]))
        if pre_lift_pos is not None and pre_lift_scoop_pos is not None:
            slip_distance = float(
                np.linalg.norm((target_pos[:2] - scoop_pos[:2]) - (pre_lift_pos[:2] - pre_lift_scoop_pos[:2]))
            )
        tilt_deg = self._tilt_deg(target_xmat[:, 2])
        dropped = bool(target_pos[2] < 0.055)

        support_contacts = 0
        all_target_contacts = 0
        for contact in self._iter_contacts(data):
            if bool({contact.geom1, contact.geom2} & support_geom_ids) and (
                contact.geom1 in scoop_geom_ids or contact.geom2 in scoop_geom_ids
            ):
                support_contacts += 1
            target_prefix = f"{target_name}_"
            g1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
            g2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
            if g1.startswith(target_prefix) or g2.startswith(target_prefix):
                all_target_contacts += 1

        lift_gain = 0.0
        if pre_lift_pos is not None:
            lift_gain = max(0.0, float(target_pos[2] - pre_lift_pos[2]))
        micro_lift_stability = float(
            np.clip(
                0.55 * min(lift_gain / 0.030, 1.0)
                + 0.30 * max(0.0, 1.0 - tilt_deg / 30.0)
                + 0.15 * max(0.0, 1.0 - slip_distance / 0.12),
                0.0,
                1.0,
            )
        )

        support_score = float(
            np.clip(
                0.35 * min(support_contacts / 4.0, 1.0)
                + 0.25 * min(insertion_depth / 0.10, 1.0)
                + 0.20 * micro_lift_stability
                + 0.10 * max(0.0, 1.0 - tilt_deg / 35.0)
                + 0.10 * max(0.0, 1.0 - slip_distance / 0.12),
                0.0,
                1.0,
            )
        )
        support_success = bool(
            support_score >= 0.49 and insertion_depth >= 0.050 and micro_lift_stability >= 0.40 and not dropped and tilt_deg < 24.0
        )

        failure_tags: list[str] = []
        if support_contacts == 0 and insertion_depth < 0.08:
            failure_tags.append("no_support_contact")
        if insertion_depth < 0.045:
            failure_tags.append("shallow_insertion")
        if micro_lift_stability < 0.40:
            failure_tags.append("unstable_micro_lift")
        if slip_distance > 0.10:
            failure_tags.append("slip")
        if tilt_deg > 24.0:
            failure_tags.append("tilt")
        if dropped:
            failure_tags.append("drop")
        if not failure_tags and not support_success:
            failure_tags.append("support_not_formed")

        return EpisodeMetrics(
            support_state_score=support_score,
            support_success=support_success,
            scoop_insertion_depth=insertion_depth,
            micro_lift_stability=micro_lift_stability,
            slip_distance=slip_distance,
            tilt_deg=tilt_deg,
            dropped=dropped,
            contact_count=all_target_contacts,
            failure_tags=failure_tags,
        )

    @staticmethod
    def _tilt_deg(up_axis: np.ndarray) -> float:
        cos_angle = float(np.clip(np.dot(up_axis, np.array([0.0, 0.0, 1.0])), -1.0, 1.0))
        return float(np.degrees(np.arccos(cos_angle)))

    @staticmethod
    def _iter_contacts(data: mujoco.MjData) -> Iterable[mujoco.MjContact]:
        for i in range(data.ncon):
            yield data.contact[i]
