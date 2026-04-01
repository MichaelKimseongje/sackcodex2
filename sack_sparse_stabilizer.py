import json
import math

import pybullet as p


class SparseSackStabilizer:
    def __init__(
        self,
        obj_path,
        json_path,
        base_position,
        base_orientation,
        scale=0.05,
        mass=0.12,
        neo_mu=80.0,
        neo_lambda=120.0,
        neo_damping=0.03,
        friction_coeff=0.9,
        collision_margin=0.001,
        stabilization_gain=4.0,
        helper_radius=0.006,
        use_gui_helpers=False,
    ):
        self.obj_path = obj_path
        self.json_path = json_path
        self.base_position = [float(x) for x in base_position]
        self.base_orientation = [float(x) for x in base_orientation]
        self.scale = float(scale)
        self.mass = float(mass)
        self.neo_mu = float(neo_mu)
        self.neo_lambda = float(neo_lambda)
        self.neo_damping = float(neo_damping)
        self.friction_coeff = float(friction_coeff)
        self.collision_margin = float(collision_margin)
        self.stabilization_gain = float(stabilization_gain)
        self.helper_radius = float(helper_radius)
        self.use_gui_helpers = bool(use_gui_helpers)

        self.position_blend = min(0.02 * self.stabilization_gain, 0.08)
        self.height_correction_ratio = min(0.015 * self.stabilization_gain, 0.06)
        self.grasp_relaxation = 0.22

        self.rest_shape = self._load_rest_shape(self.json_path)
        self.obj_vertices = self._load_obj_vertices(self.obj_path)

        self.top_idx = self.rest_shape["Top_ring"]["indices"]
        self.mid_idx = self.rest_shape["Mid_ring"]["indices"]
        self.bot_idx = self.rest_shape["Bot_ring"]["indices"]

        self.top_center_local = self.rest_shape["Top_ring"]["center"]
        self.mid_center_local = self.rest_shape["Mid_ring"]["center"]
        self.bot_center_local = self.rest_shape["Bot_ring"]["center"]

        self.rest_top_center = self._world_from_local(self.top_center_local)
        self.rest_mid_center = self._world_from_local(self.mid_center_local)
        self.rest_bot_center = self._world_from_local(self.bot_center_local)
        self.rest_height = self.scale * float(self.rest_shape["rest_height"])
        self.rest_body_center = [
            (self.rest_top_center[0] + self.rest_mid_center[0] + self.rest_bot_center[0]) / 3.0,
            (self.rest_top_center[1] + self.rest_mid_center[1] + self.rest_bot_center[1]) / 3.0,
            (self.rest_top_center[2] + self.rest_mid_center[2] + self.rest_bot_center[2]) / 3.0,
        ]
        self.rest_top_offset = [self.rest_top_center[i] - self.rest_body_center[i] for i in range(3)]
        self.rest_mid_offset = [self.rest_mid_center[i] - self.rest_body_center[i] for i in range(3)]
        self.rest_bot_offset = [self.rest_bot_center[i] - self.rest_body_center[i] for i in range(3)]

        self.sack_id = self._create_soft_sack()
        self.top_ref = self._create_helper_body(self.rest_top_center, [1.0, 0.2, 0.2, 0.65])
        self.mid_ref = self._create_helper_body(self.rest_mid_center, [0.2, 1.0, 0.2, 0.65])
        self.bot_ref = self._create_helper_body(self.rest_bot_center, [0.2, 0.4, 1.0, 0.65])

        self._anchor_ring_subset(self.top_ref, self.top_center_local, self.top_idx, 8)
        self._anchor_ring_subset(self.mid_ref, self.mid_center_local, self.mid_idx, 10)
        self._anchor_ring_subset(self.bot_ref, self.bot_center_local, self.bot_idx, 6)

    def _load_rest_shape(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_obj_vertices(self, obj_path):
        verts = []
        with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    _, xs, ys, zs = line.split()[:4]
                    verts.append([float(xs), float(ys), float(zs)])
        return verts

    def _scaled_local(self, local_point):
        return [self.scale * float(x) for x in local_point]

    def _world_from_local(self, local_point):
        world_pos, _ = p.multiplyTransforms(
            self.base_position,
            self.base_orientation,
            self._scaled_local(local_point),
            [0, 0, 0, 1],
        )
        return list(world_pos)

    def _create_soft_sack(self):
        sack_id = p.loadSoftBody(
            fileName=self.obj_path,
            basePosition=self.base_position,
            baseOrientation=self.base_orientation,
            scale=self.scale,
            mass=self.mass,
            useNeoHookean=0,
            NeoHookeanMu=self.neo_mu,
            NeoHookeanLambda=self.neo_lambda,
            NeoHookeanDamping=self.neo_damping,
            useMassSpring=1,
            useBendingSprings=1,
            springElasticStiffness=36.0,
            springDampingStiffness=0.18,
            springDampingAllDirections=1,
            frictionCoeff=self.friction_coeff,
            useSelfCollision=1,
            useFaceContact=1,
            repulsionStiffness=180.0,
            collisionMargin=self.collision_margin,
        )
        p.changeVisualShape(sack_id, -1, rgbaColor=[0.9, 0.85, 0.7, 0.98])
        p.changeVisualShape(sack_id, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED)
        return sack_id

    def _create_helper_body(self, world_pos, rgba):
        if self.use_gui_helpers:
            visual = p.createVisualShape(p.GEOM_SPHERE, radius=self.helper_radius, rgbaColor=rgba)
        else:
            visual = -1
        return p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual,
            basePosition=world_pos,
        )

    def sample_indices(self, indices, n):
        if not indices:
            return []
        if n >= len(indices):
            return list(indices)
        sampled = []
        last_pos = -1
        max_pos = len(indices) - 1
        for i in range(n):
            pos = int(round(i * max_pos / max(n - 1, 1)))
            if pos <= last_pos:
                pos = min(last_pos + 1, max_pos)
            sampled.append(indices[pos])
            last_pos = pos
        return sampled

    def _anchor_ring_subset(self, helper_id, helper_center_local, ring_indices, anchor_count):
        selected = self.sample_indices(ring_indices, anchor_count)
        for vidx in selected:
            rest_vertex_local = self.obj_vertices[vidx]
            local_offset = [
                self.scale * (rest_vertex_local[0] - helper_center_local[0]),
                self.scale * (rest_vertex_local[1] - helper_center_local[1]),
                self.scale * (rest_vertex_local[2] - helper_center_local[2]),
            ]
            p.createSoftBodyAnchor(self.sack_id, int(vidx), helper_id, -1, local_offset)

    def _get_soft_vertices(self):
        mesh_data = p.getMeshData(self.sack_id, -1, flags=p.MESH_DATA_SIMULATION_MESH)
        return mesh_data[1]

    def _ring_center(self, verts, indices):
        n = float(len(indices))
        if n <= 0:
            return [0.0, 0.0, 0.0]
        acc = [0.0, 0.0, 0.0]
        for idx in indices:
            v = verts[idx]
            acc[0] += float(v[0])
            acc[1] += float(v[1])
            acc[2] += float(v[2])
        return [acc[0] / n, acc[1] / n, acc[2] / n]

    def _normalize(self, vec, fallback=(0.0, 0.0, 1.0)):
        norm = math.sqrt(sum(float(v) * float(v) for v in vec))
        if norm < 1e-8:
            return [float(fallback[0]), float(fallback[1]), float(fallback[2])], 0.0
        return [float(v) / norm for v in vec], norm

    def _blend_pos(self, current, target, alpha):
        return [
            float(current[0]) + alpha * (float(target[0]) - float(current[0])),
            float(current[1]) + alpha * (float(target[1]) - float(current[1])),
            float(current[2]) + alpha * (float(target[2]) - float(current[2])),
        ]

    def _set_helper_pos(self, body_id, target_pos, alpha):
        current_pos, current_orn = p.getBasePositionAndOrientation(body_id)
        new_pos = self._blend_pos(current_pos, target_pos, alpha)
        p.resetBasePositionAndOrientation(body_id, new_pos, current_orn)

    def is_grasping(self, gripper_body_ids=None):
        if gripper_body_ids is None:
            return False
        if not isinstance(gripper_body_ids, (list, tuple, set)):
            gripper_body_ids = [gripper_body_ids]
        for body_id in gripper_body_ids:
            if body_id is None:
                continue
            if p.getContactPoints(bodyA=self.sack_id, bodyB=body_id):
                return True
        return False

    def maintain_shape(self, gain_scale=1.0):
        verts = self._get_soft_vertices()
        top_center = self._ring_center(verts, self.top_idx)
        mid_center = self._ring_center(verts, self.mid_idx)
        bot_center = self._ring_center(verts, self.bot_idx)
        body_center = [
            float(sum(v[0] for v in verts) / len(verts)),
            float(sum(v[1] for v in verts) / len(verts)),
            float(sum(v[2] for v in verts) / len(verts)),
        ]

        cur_height = float(top_center[2] - bot_center[2])
        target_height = cur_height + gain_scale * self.height_correction_ratio * (self.rest_height - cur_height)
        z_scale = target_height / max(self.rest_height, 1e-6)

        top_target = [
            body_center[0] + self.rest_top_offset[0],
            body_center[1] + self.rest_top_offset[1],
            body_center[2] + z_scale * self.rest_top_offset[2],
        ]
        mid_target = [
            body_center[0] + self.rest_mid_offset[0],
            body_center[1] + self.rest_mid_offset[1],
            body_center[2] + z_scale * self.rest_mid_offset[2],
        ]
        bot_target = [
            body_center[0] + self.rest_bot_offset[0],
            body_center[1] + self.rest_bot_offset[1],
            body_center[2] + z_scale * self.rest_bot_offset[2],
        ]

        alpha = max(0.0, min(1.0, gain_scale * self.position_blend))
        self._set_helper_pos(self.top_ref, top_target, alpha)
        self._set_helper_pos(self.mid_ref, mid_target, alpha)
        self._set_helper_pos(self.bot_ref, bot_target, alpha)

    def step(self, gripper_body_ids=None):
        gain_scale = self.grasp_relaxation if self.is_grasping(gripper_body_ids) else 1.0
        self.maintain_shape(gain_scale=gain_scale)
