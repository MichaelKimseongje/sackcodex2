from __future__ import annotations

import mujoco
import numpy as np


class Visualizer:
    """contact point와 insertion depth, score overlay를 marker로 표현한다."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model

    def update(self, viewer, data: mujoco.MjData, metrics, target_name: str):
        if viewer is None:
            return
        viewer.user_scn.ngeom = 0
        self._add_contact_markers(viewer, data)
        self._add_insertion_marker(viewer, data, metrics)
        self._add_score_overlay(viewer, data, metrics)
        viewer.sync()

    def _add_contact_markers(self, viewer, data: mujoco.MjData):
        for i in range(min(data.ncon, 48)):
            contact = data.contact[i]
            rgba = np.array([1.0, 0.15, 0.15, 0.75], dtype=np.float32)
            self._append_sphere(viewer, contact.pos, 0.005, rgba)

    def _add_insertion_marker(self, viewer, data: mujoco.MjData, metrics):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "scoop_tip_site")
        center_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "scoop_center_site")
        tip = data.site_xpos[site_id].copy()
        center = data.site_xpos[center_id].copy()
        delta = center - tip
        line_end = tip + np.clip(metrics.scoop_insertion_depth / 0.12, 0.0, 1.0) * delta
        self._append_capsule(viewer, tip, line_end, 0.003, np.array([0.15, 0.55, 1.0, 0.85], dtype=np.float32))

    def _add_score_overlay(self, viewer, data: mujoco.MjData, metrics):
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "score_overlay_site")
        origin = data.site_xpos[site_id].copy()
        height = 0.020 + 0.110 * metrics.support_state_score
        top = origin + np.array([0.0, 0.0, height], dtype=np.float64)
        color = np.array(
            [1.0 - metrics.support_state_score, 0.25 + 0.65 * metrics.support_state_score, 0.15, 0.85],
            dtype=np.float32,
        )
        self._append_capsule(viewer, origin, top, 0.010, color)

    @staticmethod
    def _append_sphere(viewer, pos, radius, rgba):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            return
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([radius, radius, radius], dtype=np.float64),
            pos=np.array(pos, dtype=np.float64),
            mat=np.eye(3).ravel(),
            rgba=rgba,
        )
        viewer.user_scn.ngeom += 1

    @staticmethod
    def _append_capsule(viewer, from_pos, to_pos, radius, rgba):
        if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
            return
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=np.array([radius, radius, radius], dtype=np.float64),
            pos=np.zeros(3, dtype=np.float64),
            mat=np.eye(3).ravel(),
            rgba=rgba,
        )
        mujoco.mjv_makeConnector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            radius,
            from_pos[0],
            from_pos[1],
            from_pos[2],
            to_pos[0],
            to_pos[1],
            to_pos[2],
        )
        viewer.user_scn.ngeom += 1
