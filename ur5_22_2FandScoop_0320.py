import time
import numpy as np
import os, json
import math

import pybullet as p
import pybullet_data

"""
ts:로그 확인
t_unix: UNIX epoch time (초)/속도/가속도 추정 시 사용 가능
Left_X, Left_Y, Left_Z, Right_X, Right_Y, Right_Z: Left / Right EE 목표값
Left_Roll, Left_Pitch, Left_Yaw, Right_Roll, Right_Pitch, Right_Yaw: 엔드이펙터의 롤/피치/요 [rad]
qL, qR: 실제 로봇 관절 각도/강화학습에서 가장 좋은 “행동” 형태
eeL, eeR: 실제 도달한 EE 포즈 (결과)

20260109
충돌 제한 기구학 기반 캘리브레이션으로 확인
"""

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def resolve_existing_path(candidates):
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return candidates[-1] if candidates else None

class DualUR5EEGuiIK:
    def __init__(self,
                 gui=True,
                 urdf_path_left="ur5/ur5.urdf",
                 urdf_path_right="ur5/ur5.urdf",
                 logpath='./configs/teleop_log.jsonl',
                 Clothobj="./cloth_z_up.obj",
                 base_pos_left=(0.0, -0.3, 0.0),
                 base_pos_right=(0.0,  0.3, 0.0),
                 base_ori_left=(0, 0, 0),
                 base_ori_right=(0, 0, 0),
                 time_step=1/240.0):

        self.gui = gui
        self.dt = time_step
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
        p.setGravity(0, 0, -9.8)

        p.loadURDF("plane.urdf")

        p.setPhysicsEngineParameter(
            fixedTimeStep=self.dt,
            numSubSteps=2, # 4 -> 1  (체감 제일 큼)#ur5_16
            numSolverIterations=80, # 150 -> 80#ur5_16
            deterministicOverlappingPairs=1
        )

        left_urdf = self._resolve_local_path(urdf_path_left)
        right_urdf = self._resolve_local_path(urdf_path_right)

        self._add_urdf_search_paths(left_urdf)
        self._add_urdf_search_paths(right_urdf)

        self.urL = p.loadURDF(
            left_urdf,
            basePosition=base_pos_left,
            baseOrientation=p.getQuaternionFromEuler(base_ori_left),
            useFixedBase=True,
            flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        )
        self.urR = p.loadURDF(
            right_urdf,
            basePosition=base_pos_right,
            baseOrientation=p.getQuaternionFromEuler(base_ori_right),
            useFixedBase=True,
            flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        )

        self.jL = self._get_arm_revolute_joints(self.urL, 6)
        self.jR = self._get_arm_revolute_joints(self.urR, 6)

        self.eeL = self._find_link(self.urL, ["ee_link", "tool0", "wrist_3_link"])
        self.eeR = self._find_link(self.urR, ["ee_link", "tool0", "wrist_3_link"])

        
        self.plateL = self._find_link_or_none(self.urL, ["plate_link", "robotiq_arg2f_base_link", "tool0", "ee_link"])
        self.plateR = self._find_link_or_none(self.urR, ["plate_link", "tool0", "ee_link"])
        self._set_tool_friction(self.urL, self.plateL)
        self._set_tool_friction(self.urR, self.plateR)

        self.left_gripper_joint = self._find_joint_by_name(self.urL, "finger_joint")
        if self.left_gripper_joint is not None:
            self.left_gripper_range = [0.0, 0.085]
            self.left_gripper_open = 0.70
            self.left_gripper_close = 0.00
            self.left_gripper_target = self.left_gripper_open
            self.left_gripper_mimic = {
                "finger_joint": 1.0,
                "left_inner_finger_joint": 1.0,
                "right_inner_finger_joint": 1.0,
                "left_inner_knuckle_joint": -1.0,
                "right_outer_knuckle_joint": -1.0,
                "right_inner_knuckle_joint": -1.0,
            }
            self.left_gripper_joint_map = {
                name: self._find_joint_by_name(self.urL, name)
                for name in self.left_gripper_mimic
            }
            self._set_left_gripper_contact_friction()
            self.set_left_gripper_opening(self.left_gripper_open)
        else:
            self.left_gripper_range = None
            self.left_gripper_open = None
            self.left_gripper_close = None
            self.left_gripper_target = None
            self.left_gripper_mimic = {}
            self.left_gripper_joint_map = {}

        self.maxF_L = [120, 120, 120, 20, 20, 20]
        self.maxF_R = [120, 120, 120, 20, 20, 20]
        
        
        self.torque_limit_L = [150,150,150,28,28,28]
        self.torque_limit_R = [150,150,150,28,28,28]

        self.kp = 0.35        # positionGain (0.05~0.25 사이 추천)
        self.kd = 0.8         # velocityGain 느낌 (PyBullet setJointMotorControl2에서 velocityGain으로 사용)
        self.maxVel = 2.5     # 관절 최대 속도 제한(너무 크면 튐)

        self.joint_damping = [0.08]*6  # 0.02~0.2 추천

        self.alpha = 0.15     # 0.1~0.3 추천 (작을수록 덜 튐)

        self.homeL = [-1.57, -1.54,  1.34, -1.37, -1.57, 0.001]
        self.homeR = [ 1.57, -1.54,  1.34, -1.37, -1.57, 0.001]
        self._reset_arm(self.urL, self.jL, self.homeL)
        self._reset_arm(self.urR, self.jR, self.homeR)

        self.targetL = self._get_ee_pose(self.urL, self.eeL)  # (pos[3], rpy[3])
        self.targetR = self._get_ee_pose(self.urR, self.eeR)

        self.filtL = np.array(self.targetL[0] + self.targetL[1], dtype=np.float32)
        self.filtR = np.array(self.targetR[0] + self.targetR[1], dtype=np.float32)
        
        self.prev_L_ee = np.array(self.filtL, dtype=np.float32)
        self.prev_R_ee = np.array(self.filtR, dtype=np.float32)
        self.prev_L_q  = np.array([p.getJointState(self.urL, jid)[0] for jid in self.jL], dtype=np.float32)
        self.prev_R_q  = np.array([p.getJointState(self.urR, jid)[0] for jid in self.jR], dtype=np.float32)
        
        self.alpha_q = 0.25
        self.filtqL = self.prev_L_q.copy()
        self.filtqR = self.prev_R_q.copy()
        
        self.ctrl_thresh = 3e-3

        self.scale = 0.15
        self.mass_each = 0.20  # sheet mass (tune)

        sackpath = r"D:/Michael/2025/01.Research/01.Parceldetection/16.Pybullet/bullet3-master (1)/bullet3-master/examples/pybullet/examples/DeformableTest/Sackcodex2/object/sack9.obj"
        SACK_OBJ = resolve_existing_path([
            os.path.join(os.path.dirname(__file__), "Sackcodex2", "object", "sack9.obj"),
            sackpath,
        ])

        self.sack_obj_path = SACK_OBJ
        self.sack_scale = 0.05
        self.sack_mass = 0.40
        self.sack_spawn_pos = [0.50, 0.17, 0.20]
        self.sack_spawn_orn = p.getQuaternionFromEuler([0, 1.57, 0])
        self.sack_rgba = [0.9, 0.85, 0.7, 0.7]
        self.sack_collision_box_scale = [0.95, 0.97, 0.95]
        self.sack_wall_thickness = 0.009
        self.sack_inner_margin = 0.012

        self.sack_local_vertices = self._load_obj_vertices(self.sack_obj_path)
        self.sack_local_bbox_min = self.sack_local_vertices.min(axis=0)
        self.sack_local_bbox_max = self.sack_local_vertices.max(axis=0)
        self.sack_local_bbox_center = 0.5 * (self.sack_local_bbox_min + self.sack_local_bbox_max)
        self.sack_local_mean = self.sack_local_vertices.mean(axis=0)
        self.sack_visual_offset = np.zeros(3, dtype=np.float32)
        self.sack_local_size = (self.sack_local_bbox_max - self.sack_local_bbox_min) * self.sack_scale

        self.sack_id = self._create_rigid_sack(self.sack_spawn_pos, self.sack_spawn_orn)
        self.sack_content_ids, self.sack_content_constraint_ids = self._spawn_sack_internal_spheres()
        self.shape_restore_enabled = False
        self.initial_pos = None
        self.initial_center = None
        self.initial_offsets = None
        self.initial_radius = None

        self.sack_debug_line_ids = {"x": None, "y": None, "z": None}
        self.sack_debug_text_id = None
        self.sack_debug_axis_len = 0.07
        self.sack_debug_update_sec = 0.5
        self._sack_last_debug_t = 0.0

        self.enable_robot_debug = False
        self.enable_sack_debug = False
        self.robot_debug_update_sec = 0.2
        self._robot_last_debug_t = 0.0
        self.robot_joint_marker_ids = {"L": [None]*len(self.jL), "R": [None]*len(self.jR)}
        self.robot_ee_marker_ids = {"L": [None, None, None], "R": [None, None, None]}
        self.robot_joint_marker_half = 0.006
        self.robot_ee_marker_half = 0.008

        self.force_far_from_sack = False
        self.forced_far_point = None
        self.forced_yz_angle_range_deg = (25.0, 30.0)

        self._update_sack_debug(force=True)
        
        

        # s=2
        # mass0=0.003#kg단위
        # bean_ids = self.spawn_clump_grid(ClumpType=3,
        #     r=0.003*s, vis_scale=0.003*s,mass=mass0*(s**3),
            
            
        #     xs=(0.48,0.50,0.52), #7
        #     ys=(-0.02,0,0.02), #11
        #     zs=(0.185,0.2,0.215), #4
        #     )

        

        self.ui = {}
        
        self._prev_btn = {
            "L_COPY_EE2J": 0, "L_COPY_J2EE": 0,
            "R_COPY_EE2J": 0, "R_COPY_J2EE": 0
        }
        
        self._readback_id_L = None
        self._readback_id_R = None

        
        self._prev_save = 0
        self._prev_reset = 0
        
        self.torque_text_ids_L = [None]*len(self.jL)
        self.torque_text_ids_R = [None]*len(self.jR)

        self.torque_limit_L = [150,150,150,28,28,28]
        self.torque_limit_R = [150,150,150,28,28,28]

        self._torque_last_print_t = 0.0
        self._torque_print_interval = 0.25  # 0.25초에 1번만 출력
        self._torque_ratio_th = 1.0         # 1.0이면 리밋 초과만, 0.8이면 80% 경고
        
        self.sat_count_L = 0
        self.sat_count_R = 0
        self.sat_warn_frames = 30   # 240Hz 기준 0.125초 (원하면 60=0.25초)
        
        self.cmd = {
            "L_mode": 0,  # 0=EE,1=J
            "R_mode": 0,
            "L_ee": np.array(self.filtL, dtype=np.float32),  # 6
            "R_ee": np.array(self.filtR, dtype=np.float32),
            "L_q":  np.array(self.homeL, dtype=np.float32),  # 6
            "R_q":  np.array(self.homeR, dtype=np.float32),
            "sleep": self.dt,
        }

    
    def create_peanut_collision(self,r=0.003, d=0.0035):
        col = p.createCollisionShapeArray(
            shapeTypes=[p.GEOM_SPHERE, p.GEOM_SPHERE],
            radii=[r, r],
            collisionFramePositions=[[0, 0, -d/2], [0, 0, +d/2]],
        )
        return col

    def create_bean_visual(self,mesh_path, s=0.003):
        vis = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=mesh_path,
            meshScale=[s, s, s],
        )
        return vis
    
    
    def create_clump_collision(self, r, offsets):
        """
        offsets: [[x,y,z], ...] 각 구의 중심 오프셋 (m)
        r: 각 구 반지름 (m)
        """
        n = len(offsets)
        col = p.createCollisionShapeArray(
            shapeTypes=[p.GEOM_SPHERE]*n,
            radii=[r]*n,
            collisionFramePositions=offsets,
        )
        return col

    def create_clump_visual(self, r, offsets, rgba=(0.4,0.3,0.2,1.0)):
        n = len(offsets)
        vis = p.createVisualShapeArray(
            shapeTypes=[p.GEOM_SPHERE]*n,
            radii=[r]*n,
            visualFramePositions=offsets,
            rgbaColors=[rgba]*n
        )
        return vis

    
    def offsets_tri(self, d):
        return [
            [0.0,     0.0,   0.0],
            [d,       0.0,   0.0],
            [0.5*d, 0.866*d, 0.0],
        ]
    
    def offsets_tetra(self, d):
        return [
            [0.0,     0.0,    0.0],
            [d,       0.0,    0.0],
            [0.5*d, 0.866*d,  0.0],
            [0.5*d, 0.2887*d, 0.816*d],
        ]
    
    def spawn_clump_grid(self, xs, ys, zs,
                     r=0.003, d=None, ClumpType=3,
                     mass=0.003, dyn=None,
                     use_mesh_visual=False, mesh_path=None, vis_scale=0.003):

        if d is None:
            d = 1.5 * r  # 시작값
    
        if ClumpType == 3:
            offsets = self.offsets_tri(d)
        elif ClumpType == 4:
            offsets = self.offsets_tetra(d)
        else:
            raise ValueError("n must be 3 or 4")
    
        col = self.create_clump_collision(r, offsets)
    
        if use_mesh_visual and mesh_path is not None:
            vis = self.create_bean_visual(mesh_path, vis_scale)  # 네 기존 mesh visual
        else:
            vis = self.create_clump_visual(r, offsets)
    
        if dyn is None:
            dyn = dict(
                lateralFriction=3.0,
                spinningFriction=0.05,
                rollingFriction=0.08,     # ✅ 0.02 -> 0.05~0.15 올려봐 (쌓임에 도움)
                restitution=0.0,
                linearDamping=0.9,
                angularDamping=0.9,
            )
    
        ids=[]
        for x in xs:
            for y in ys:
                for z in zs:
                    orn = p.getQuaternionFromEuler(np.random.uniform(-np.pi, np.pi, 3))
                    bid = p.createMultiBody(
                        baseMass=mass,
                        baseCollisionShapeIndex=col,
                        baseVisualShapeIndex=vis,
                        basePosition=[x,y,z],
                        baseOrientation=orn
                    )
                    p.changeDynamics(bid, -1, **dyn)
                    ids.append(bid)
        return ids

    
    def spawn_peanut_grid(self,mesh_path, xs, ys, zs,
                          r=0.003, d=0.0035, vis_scale=0.003,
                          mass=0.003, dyn=None):
        col = self.create_peanut_collision(r, d)
        vis = self.create_bean_visual(mesh_path, vis_scale)
    
        if dyn is None:
            
            dyn = dict(
                lateralFriction=3.0,
                spinningFriction=0.02,
                rollingFriction=0.02,     # ✅ 핵심: 구슬-유체화 방지
                restitution=0.0,
                linearDamping=0.9,
                angularDamping=0.9,
            )
    
        ids=[]
        for x in xs:
            for y in ys:
                for z in zs:
                    orn = p.getQuaternionFromEuler(np.random.uniform(-np.pi, np.pi, 3))
                    bid = p.createMultiBody(
                        baseMass=mass,
                        baseCollisionShapeIndex=col,
                        baseVisualShapeIndex=vis,
                        basePosition=[x,y,z],
                        baseOrientation=orn
                    )
                    p.changeDynamics(bid, -1, **dyn)
                    ids.append(bid)
        return ids

    def _get_arm_revolute_joints(self, rid, target_dofs=6):
        joints = []
        for j in range(p.getNumJoints(rid)):
            if p.getJointInfo(rid, j)[2] == p.JOINT_REVOLUTE:
                joints.append(j)
        return joints[:target_dofs]

    def _resolve_local_path(self, path):
        if os.path.isabs(path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"URDF path does not exist: {path}")
            return path

        candidates = [
            os.path.abspath(path),
            os.path.join(self.base_dir, path),
        ]
        resolved = resolve_existing_path(candidates)
        if resolved is None or not os.path.exists(resolved):
            raise FileNotFoundError(
                "Cannot resolve URDF path. "
                f"input='{path}', tried={candidates}"
            )
        return os.path.abspath(resolved)

    def _add_urdf_search_paths(self, urdf_file):
        urdf_dir = os.path.dirname(os.path.abspath(urdf_file))
        p.setAdditionalSearchPath(urdf_dir)
        parent_dir = os.path.dirname(urdf_dir)
        if parent_dir and parent_dir != urdf_dir:
            p.setAdditionalSearchPath(parent_dir)

    def _find_link(self, rid, candidates):
        name_to_idx = {}
        for j in range(p.getNumJoints(rid)):
            link_name = p.getJointInfo(rid, j)[12].decode("utf-8")
            name_to_idx[link_name] = j
        for c in candidates:
            if c in name_to_idx:
                return name_to_idx[c]
        return p.getNumJoints(rid) - 1

    def _find_link_or_none(self, rid, candidates):
        name_to_idx = {}
        for j in range(p.getNumJoints(rid)):
            link_name = p.getJointInfo(rid, j)[12].decode("utf-8")
            name_to_idx[link_name] = j
        for c in candidates:
            if c in name_to_idx:
                return name_to_idx[c]
        return None

    def _find_joint_by_name(self, rid, joint_name):
        for j in range(p.getNumJoints(rid)):
            if p.getJointInfo(rid, j)[1].decode("utf-8") == joint_name:
                return j
        return None

    def _set_link_friction(self, rid, link_idx, lateral=2.0, spinning=0.02, rolling=0.0):
        try:
            p.changeDynamics(
                rid,
                link_idx,
                lateralFriction=float(lateral),
                spinningFriction=float(spinning),
                rollingFriction=float(rolling),
                restitution=0.0,
                frictionAnchor=1,
            )
        except TypeError:
            p.changeDynamics(
                rid,
                link_idx,
                lateralFriction=float(lateral),
                spinningFriction=float(spinning),
                rollingFriction=float(rolling),
                restitution=0.0,
            )

    def _set_tool_friction(self, rid, link_idx):
        if link_idx is None:
            return
        self._set_link_friction(rid, link_idx, lateral=2.0, spinning=0.02, rolling=0.0)

    def _set_left_gripper_contact_friction(self):
        if self.plateL is None:
            return

        gripper_links = set(self._collect_descendant_links(self.urL, self.plateL))
        gripper_links.add(self.plateL)

        tip_links = [
            "left_inner_finger_pad",
            "right_inner_finger_pad",
            "left_inner_finger",
            "right_inner_finger",
            "left_outer_finger",
            "right_outer_finger",
        ]
        for link_name in tip_links:
            link_idx = self._find_link_or_none(self.urL, [link_name])
            if link_idx is not None:
                gripper_links.add(link_idx)

        for link_idx in sorted(gripper_links):
            self._set_link_friction(self.urL, link_idx, lateral=8.0, spinning=0.2, rolling=0.0)

    def set_left_gripper_opening(self, opening):
        if self.left_gripper_joint is None:
            return
        target = float(clamp(opening, self.left_gripper_close, self.left_gripper_open))
        self.left_gripper_target = target
        for name, mul in self.left_gripper_mimic.items():
            jid = self.left_gripper_joint_map.get(name)
            if jid is None:
                continue
            p.setJointMotorControl2(
                self.urL,
                jid,
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(mul * target),
                force=140.0,
                positionGain=0.3,
                velocityGain=1.0,
                maxVelocity=2.0,
            )

    def _gripper_main_joint_from_opening_length(self, opening_length):
        lo, hi = self.left_gripper_range
        length = float(clamp(opening_length, lo, hi))
        ratio = (length - 0.010) / 0.1143
        ratio = clamp(ratio, -1.0, 1.0)
        angle = 0.715 - math.asin(ratio)
        return float(clamp(angle, self.left_gripper_close, self.left_gripper_open))

    def set_left_gripper_opening_length(self, opening_length):
        if self.left_gripper_joint is None:
            return
        main_joint = self._gripper_main_joint_from_opening_length(opening_length)
        self.set_left_gripper_opening(main_joint)

    def _reset_arm(self, rid, joint_ids, q):
        for jid, qi in zip(joint_ids, q):
            p.resetJointState(rid, jid, qi)
        for jid in joint_ids:
            p.setJointMotorControl2(rid, jid, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

    def _get_ee_pose(self, rid, ee_idx):
        ls = p.getLinkState(rid, ee_idx)
        pos = list(ls[4])  # worldLinkFramePosition
        orn = ls[5]
        rpy = list(p.getEulerFromQuaternion(orn))
        return pos, rpy

    def contact_force_sum(self, bodyA, bodyB=None):
        pts = p.getContactPoints(bodyA=bodyA, bodyB=bodyB) if bodyB is not None else p.getContactPoints(bodyA=bodyA)
        return sum(cp[9] for cp in pts)  # normal force

    def _smooth(self, prev, x, alpha):
        return (1.0 - alpha) * prev + alpha * x

    def _ik_to_joints(self, rid, ee_idx, joint_ids, target_vec6, rest_pose):
        pos = target_vec6[:3].tolist()
        rpy = target_vec6[3:].tolist()
        orn = p.getQuaternionFromEuler(rpy)
    
        if hasattr(p, "calculateInverseKinematics2"):
            try:
                q = p.calculateInverseKinematics2(
                    bodyUniqueId=rid,
                    endEffectorLinkIndices=[ee_idx],
                    targetPositions=[pos],
                    targetOrientations=[orn],
                    jointIndices=joint_ids,
                    jointDamping=getattr(self, "joint_damping", None),
                    restPoses=rest_pose,
                )
                return list(q[:len(joint_ids)])
            except TypeError:
                try:
                    q = p.calculateInverseKinematics2(
                        rid,
                        [ee_idx],
                        [pos],
                        [orn],
                        jointIndices=joint_ids,
                        jointDamping=getattr(self, "joint_damping", None),
                        restPoses=rest_pose,
                    )
                    return list(q[:len(joint_ids)])
                except Exception:
                    pass  # IK1 fallback
    
        q = p.calculateInverseKinematics(
            bodyUniqueId=rid,
            endEffectorLinkIndex=ee_idx,
            targetPosition=pos,
            targetOrientation=orn,
            jointDamping=getattr(self, "joint_damping", None),
            restPoses=rest_pose,
        )
    
        if len(q) >= len(joint_ids):
            return list(q[:len(joint_ids)])
        return list(q)

    

    def _apply_q(self, rid, joint_ids, q, max_forces):
        for jid, qi, mf in zip(joint_ids, q, max_forces):
            p.setJointMotorControl2(
                rid, jid,
                controlMode=p.POSITION_CONTROL,
                targetPosition=qi,
                force=mf,
                positionGain=self.kp,
                velocityGain=self.kd,
                maxVelocity=self.maxVel
            )
            
    
    def is_sustained_sat(self, sat, tag="L"):
        if tag=="L":
            self.sat_count_L = self.sat_count_L + 1 if sat else 0
            return self.sat_count_L >= self.sat_warn_frames
        else:
            self.sat_count_R = self.sat_count_R + 1 if sat else 0
            return self.sat_count_R >= self.sat_warn_frames

    def _draw_torque_texts(self, rid, joint_ids, taus, limits, text_ids, prefix="L"):
        """관절 근처에 토크 표시. limit 초과면 빨간색."""
        for k, (jid, tau, lim) in enumerate(zip(joint_ids, taus, limits)):
            try:
                ls = p.getLinkState(rid, jid, computeForwardKinematics=True)
                pos = ls[4]  # worldLinkFramePosition
            except:
                pos, _ = p.getBasePositionAndOrientation(rid)

            ratio = abs(tau) / max(lim, 1e-6)
            over = ratio >= 1.0

            color = [1, 0, 0] if over else [0, 1, 0]   # red if over else green
            txt = f"{prefix}J{k+1}: {tau:+.2f} / {lim:.1f} Nm ({ratio*100:.0f}%)"

            if text_ids[k] is None:
                text_ids[k] = p.addUserDebugText(
                    txt, pos,
                    textColorRGB=color,
                    textSize=1.2,
                    lifeTime=0,  # 0이면 유지됨(우리가 replace로 업데이트)
                )
            else:
                text_ids[k] = p.addUserDebugText(
                    txt, pos,
                    textColorRGB=color,
                    textSize=1.2,
                    lifeTime=0,
                    replaceItemUniqueId=text_ids[k]
                )
                
    def _check_torque_over_and_print(self, robot_id, joint_ids, limits, tag="L", limit="ON"):
        taus = []
        for jid in joint_ids:
            js = p.getJointState(robot_id, jid)
            taus.append(float(js[3]))  # applied motor torque (approx)
    
        now = time.time()
        if now - self._torque_last_print_t < self._torque_print_interval:
            return taus
    
        ratios = []
        for tau, lim in zip(taus, limits):
            if lim <= 0:
                ratios.append(0.0)
            else:
                ratios.append(abs(tau) / float(lim))
    
        if limit.upper() == "OFF":
            msg = " | ".join([f"J{i+1}:{taus[i]:+.2f}Nm/{limits[i]:.1f}(x{ratios[i]:.2f})"
                              for i in range(len(taus))])
            print(f"[TORQUE {tag}] {msg}")
            self._torque_last_print_t = now
            return taus
    
        over = [(i, taus[i], limits[i], ratios[i]) for i in range(len(taus)) if ratios[i] >= self._torque_ratio_th]
        if over:
            msg = " | ".join([f"J{i+1}:{tau:+.2f}Nm/{lim:.1f}(x{ratio:.2f})" for (i, tau, lim, ratio) in over])
            print(f"[TORQUE OVER {tag}] {msg}")
            self._torque_last_print_t = now
    
        return taus

    def torque_saturation_ratio(self, robot_id, joint_ids, maxF):
        taus = [float(p.getJointState(robot_id, jid)[3]) for jid in joint_ids]
        ratios = [abs(t)/mf if mf>1e-6 else 0.0 for t, mf in zip(taus, maxF)]
        sat = any(r > 0.95 for r in ratios)  # 95% 이상이면 포화로 판단
        return taus, ratios, sat
    
    
    def _load_obj_vertices(self, obj_path):
        verts = []
        with open(obj_path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if line.startswith("v "):
                    parts = line.split()
                    if len(parts) >= 4:
                        verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if not verts:
            raise ValueError(f"No vertices found in OBJ: {obj_path}")
        return np.array(verts, dtype=np.float32)

    def _transform_points(self, base_pos, base_orn, local_points):
        rot = np.array(p.getMatrixFromQuaternion(base_orn), dtype=np.float32).reshape(3, 3)
        pts = np.array(local_points, dtype=np.float32)
        return (pts @ rot.T) + np.array(base_pos, dtype=np.float32)

    def _create_rigid_sack(self, base_pos, base_orn):
        outer = 0.5 * np.array(self.sack_local_size, dtype=np.float32)
        outer = outer * np.array(self.sack_collision_box_scale, dtype=np.float32)
        wall_t = float(self.sack_wall_thickness)
        inner = np.maximum(outer - wall_t, 0.012)
        center = np.array(self.sack_visual_offset, dtype=np.float32)

        half_extents = [
            [outer[0], 0.5 * wall_t, outer[2]],
            [outer[0], 0.5 * wall_t, outer[2]],
            [0.5 * wall_t, inner[1], outer[2]],
            [0.5 * wall_t, inner[1], outer[2]],
            [inner[0], inner[1], 0.5 * wall_t],
            [inner[0], inner[1], 0.5 * wall_t],
        ]
        positions = [
            [center[0], center[1] - inner[1] - 0.5 * wall_t, center[2]],
            [center[0], center[1] + inner[1] + 0.5 * wall_t, center[2]],
            [center[0] - inner[0] - 0.5 * wall_t, center[1], center[2]],
            [center[0] + inner[0] + 0.5 * wall_t, center[1], center[2]],
            [center[0], center[1], center[2] - inner[2] - 0.5 * wall_t],
            [center[0], center[1], center[2] + inner[2] + 0.5 * wall_t],
        ]
        col = p.createCollisionShapeArray(
            shapeTypes=[p.GEOM_BOX] * len(half_extents),
            halfExtents=[list(map(float, he)) for he in half_extents],
            collisionFramePositions=[list(map(float, pp)) for pp in positions],
            collisionFrameOrientations=[[0, 0, 0, 1]] * len(half_extents),
        )
        vis = p.createVisualShape(
            p.GEOM_MESH,
            fileName=self.sack_obj_path,
            meshScale=[self.sack_scale] * 3,
            rgbaColor=self.sack_rgba,
            visualFramePosition=(-self.sack_local_mean * self.sack_scale).tolist(),
        )
        sack_id = p.createMultiBody(
            baseMass=self.sack_mass,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=base_pos,
            baseOrientation=base_orn,
        )
        p.changeDynamics(
            sack_id, -1,
            lateralFriction=1.4,
            spinningFriction=0.06,
            rollingFriction=0.0,
            restitution=0.0,
            linearDamping=0.06,
            angularDamping=0.16,
        )
        return sack_id

    def _spawn_sack_internal_spheres(self):
        base_pos, base_orn = p.getBasePositionAndOrientation(self.sack_id)
        outer = 0.5 * np.array(self.sack_local_size, dtype=np.float32)
        outer = outer * np.array(self.sack_collision_box_scale, dtype=np.float32)
        inner = np.maximum(outer - float(self.sack_wall_thickness) - float(self.sack_inner_margin), 0.012)
        center = np.array(self.sack_visual_offset, dtype=np.float32)

        radius = 0.0043
        xs = center[0] + np.linspace(-0.24 * inner[0], 0.24 * inner[0], 3)
        ys = center[1] + np.linspace(-0.12 * inner[1], 0.12 * inner[1], 1)
        zs = center[2] + np.linspace(-0.24 * inner[2], 0.24 * inner[2], 2)

        col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[0.52, 0.44, 0.30, 1.0])
        ids = []
        cids = []
        sphere_mass = 0.60
        for x in xs:
            for y in ys:
                for z in zs:
                    local_pos = [float(x), float(y), float(z)]
                    wpos = self._transform_points(base_pos, base_orn, [local_pos])[0]
                    bid = p.createMultiBody(
                        baseMass=sphere_mass,
                        baseCollisionShapeIndex=col,
                        baseVisualShapeIndex=vis,
                        basePosition=wpos.tolist(),
                        baseOrientation=[0, 0, 0, 1],
                    )
                    p.changeDynamics(
                        bid, -1,
                        lateralFriction=1.1,
                        spinningFriction=0.02,
                        rollingFriction=0.0005,
                        restitution=0.0,
                        linearDamping=0.04,
                        angularDamping=0.04,
                        contactProcessingThreshold=0.0,
                        ccdSweptSphereRadius=radius * 0.9,
                    )
                    try:
                        p.changeDynamics(bid, -1, ccdMotionThreshold=radius * 0.5)
                    except Exception:
                        pass
                    ids.append(bid)
        return ids, cids

    def border_indices_from_verts(self, verts, edge_band=0):
        """
        verts: p.getMeshData(...)[1]  ([(x,y,z), ...])
        edge_band: 0이면 가장 바깥 1줄, 1이면 2줄 ...
        return: border_indices(list[int]), N_grid(or None)
        """
        numv = len(verts)
        N = int(round(math.sqrt(numv)))

        if N * N == numv:
            idxs = []
            for idx in range(numv):
                i = idx // N
                j = idx % N
                if (i <= edge_band) or (j <= edge_band) or (i >= N-1-edge_band) or (j >= N-1-edge_band):
                    idxs.append(idx)
            return idxs, N

        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        approxN = max(int(round(math.sqrt(numv))), 2)
        dx = (x_max - x_min) / (approxN - 1)
        dy = (y_max - y_min) / (approxN - 1)
        thx = (edge_band + 0.5) * dx + 1e-6
        thy = (edge_band + 0.5) * dy + 1e-6

        idxs = []
        for idx, (x, y, z) in enumerate(verts):
            is_edge = (abs(x - x_min) < thx) or (abs(x - x_max) < thx) or (abs(y - y_min) < thy) or (abs(y - y_max) < thy)
            if is_edge:
                idxs.append(idx)
        return idxs, None
    
    def spawn_mesh_grid(self,
        visual_mesh_path,
        collision_mesh_path,   # ✅ convex hull obj
        mesh_scale=0.003,
        mass=0.001,
        xs=(0.43, 0.50, 0.57),
        ys=(-0.14, -0.07, 0.0, 0.07, 0.14),
        zs=(0.03, 0.1),
        quat=None,
        dyn_kwargs=None,
    ):
        if quat is None:
            quat = p.getQuaternionFromEuler([0, 0, 0])
    
        visual_mesh_path = os.path.abspath(visual_mesh_path)
        collision_mesh_path = os.path.abspath(collision_mesh_path)
    
        p.setAdditionalSearchPath(os.path.dirname(visual_mesh_path))
    
        col = p.createCollisionShape(
            p.GEOM_MESH,
            fileName=collision_mesh_path,
            meshScale=[mesh_scale]*3
        )
        vis = p.createVisualShape(
            p.GEOM_MESH,
            fileName=visual_mesh_path,
            meshScale=[mesh_scale]*3
        )
    
        if dyn_kwargs is None:
            dyn_kwargs = dict(
                lateralFriction=2.0,
                spinningFriction=0.2,
                rollingFriction=0.05,
                restitution=0.0,
                linearDamping=0.6,
                angularDamping=0.6,
            )
    
        ids = []
        for x in xs:
            for y in ys:
                for z in zs:
                    bid = p.createMultiBody(
                        baseMass=mass,
                        baseCollisionShapeIndex=col,
                        baseVisualShapeIndex=vis,
                        basePosition=[x, y, z],
                        baseOrientation=quat
                    )
                    p.changeDynamics(bid, -1, **dyn_kwargs)
                    ids.append(bid)
        return ids

    

    
    def spawn_object_grid(self,
        objecturdf="sphere_small.urdf",
        sphere_scale=1.0,
        zs=(0.03, 0.1),
        xs=(0.43, 0.50, 0.57),
        ys=(-0.14, -0.07, 0.0, 0.07, 0.14),
        quat=None,
        dyn_kwargs=None,
    ):
        """
        동일 URDF 구체를 격자 형태로 생성하고, 동역학 파라미터를 일괄 적용.
        반환: sphere_ids(list)
        """
        if quat is None:
            quat = p.getQuaternionFromEuler([0, 0, 0])
    
        if dyn_kwargs is None:
            dyn_kwargs = dict(
                mass = 0.05,
                lateralFriction=1.5,
                spinningFriction=0.05,
                rollingFriction=0.0,
                restitution=0.0,
                linearDamping=0.8,   
                angularDamping=0.8,
            )
    
        sphere_ids = []
        for x in xs:
            for y in ys:
                for z in zs:
                    sid = p.loadURDF(
                        objecturdf,
                        [x, y, z],
                        quat,
                        globalScaling=sphere_scale
                    )
                    p.changeDynamics(sid, -1, **dyn_kwargs)
                    sphere_ids.append(sid)
    
        return sphere_ids
    
    
    def min_border_distance_xy(self,cloth_bottom, cloth_top, edge_band=1):
        vb, vb_raw = self.get_soft_verts(cloth_bottom)
        vt, vt_raw = self.get_soft_verts(cloth_top)
        bb, _ = self.border_indices_from_verts(vb_raw, edge_band=edge_band)
        bt, _ = self.border_indices_from_verts(vt_raw, edge_band=edge_band)
    
        vb_b = vb[bb][:,:2]
        vt_b = vt[bt][:,:2]
    
        min_d = 1e9
        step = max(len(vt_b)//200, 1)  # 너무 많으면 샘플링 (최대 200개 정도)
        for pt in vt_b[::step]:
            d = vb_b - pt
            dd = np.min(np.sum(d*d, axis=1))
            if dd < min_d:
                min_d = dd
        return float(np.sqrt(min_d))
    
    def get_soft_verts(self, soft_id):
        return np.zeros((0, 3), dtype=np.float32), []

    def _capture_softbody_vertices(self, soft_id):
        return None

    def _best_fit_rotation(self, rest_offsets, current_offsets):
        return np.eye(3, dtype=np.float32)

    def _rotation_matrix_to_quaternion(self, rot):
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    def _iter_robot_contact_pairs(self):
        return []

    def _sack_has_robot_contact(self):
        return False

    def _sack_has_robot_near_contact(self, distance=None):
        return False

    def apply_shape_restoration(self, soft_id, initial_pos, k=30.0, damping=1.0):
        return

    def _get_sack_state(self):
        pos, orn = p.getBasePositionAndOrientation(self.sack_id)
        rpy = p.getEulerFromQuaternion(orn)
        return {
            "center": np.array(pos, dtype=np.float32),
            "rpy": np.array(rpy, dtype=np.float32),
            "quat": np.array(orn, dtype=np.float32),
            "size": np.array(self.sack_local_size, dtype=np.float32),
        }

    def get_sack_approach_target(self, clearance=0.10, side_offset=0.0):
        state = self._get_sack_state()
        if state is None:
            return None

        center = np.array(state["center"], dtype=np.float32)
        target = center.copy()
        target[1] += float(side_offset)
        target[2] += float(clearance)
        return target

    def _update_sack_debug(self, force=False):
        if not getattr(self, "enable_sack_debug", True):
            return
        now = time.time()
        if (not force) and (now - self._sack_last_debug_t < self.sack_debug_update_sec):
            return

        state = self._get_sack_state()
        if state is None:
            return

        c = state["center"]
        r, pch, y = state["rpy"]
        lx, wy, hz = state["size"]

        rot = np.array(p.getMatrixFromQuaternion(state["quat"]), dtype=np.float32).reshape(3, 3)
        colors = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}
        axes = {"x": rot[:, 0], "y": rot[:, 1], "z": rot[:, 2]}

        for key in ("x", "y", "z"):
            end = (c + self.sack_debug_axis_len * axes[key]).tolist()
            self.sack_debug_line_ids[key] = p.addUserDebugLine(
                c.tolist(),
                end,
                lineColorRGB=colors[key],
                lineWidth=3,
                lifeTime=0,
                replaceItemUniqueId=self.sack_debug_line_ids[key] if self.sack_debug_line_ids[key] is not None else -1,
            )

        label = f"Sack C=({c[0]:.3f},{c[1]:.3f},{c[2]:.3f}) RPY=({r:.2f},{pch:.2f},{y:.2f})"
        self.sack_debug_text_id = p.addUserDebugText(
            label,
            c.tolist(),
            textColorRGB=[1, 1, 0],
            textSize=1.2,
            lifeTime=0,
            replaceItemUniqueId=self.sack_debug_text_id if self.sack_debug_text_id is not None else -1,
        )

        self._sack_last_debug_t = now

    def _draw_cross_marker(self, point, color, half_len, marker_id):
        x, y, z = point
        segs = [
            ([x-half_len, y, z], [x+half_len, y, z]),
            ([x, y-half_len, z], [x, y+half_len, z]),
            ([x, y, z-half_len], [x, y, z+half_len]),
        ]
        if marker_id is None:
            marker_id = [None, None, None]
        for i, (a, b) in enumerate(segs):
            marker_id[i] = p.addUserDebugLine(
                a, b,
                lineColorRGB=color,
                lineWidth=4,
                lifeTime=0,
                replaceItemUniqueId=marker_id[i] if marker_id[i] is not None else -1,
            )
        return marker_id

    def _aabb_corners(self, aabb_min, aabb_max):
        mn = np.array(aabb_min, dtype=np.float32)
        mx = np.array(aabb_max, dtype=np.float32)
        return np.array([
            [mn[0], mn[1], mn[2]], [mn[0], mn[1], mx[2]],
            [mn[0], mx[1], mn[2]], [mn[0], mx[1], mx[2]],
            [mx[0], mn[1], mn[2]], [mx[0], mn[1], mx[2]],
            [mx[0], mx[1], mn[2]], [mx[0], mx[1], mx[2]],
        ], dtype=np.float32)

    def _transform_local_points(self, frame_pos, frame_orn, local_points):
        rot = np.array(p.getMatrixFromQuaternion(frame_orn), dtype=np.float32).reshape(3, 3)
        lp = np.array(local_points, dtype=np.float32)
        return (lp @ rot.T) + np.array(frame_pos, dtype=np.float32)

    def _shape_points_local(self, geom_type, dims):
        if geom_type == p.GEOM_BOX:
            hx, hy, hz = 0.5*np.array(dims[:3], dtype=np.float32)
            return np.array([
                [-hx,-hy,-hz], [-hx,-hy,+hz], [-hx,+hy,-hz], [-hx,+hy,+hz],
                [+hx,-hy,-hz], [+hx,-hy,+hz], [+hx,+hy,-hz], [+hx,+hy,+hz],
            ], dtype=np.float32)
        if geom_type == p.GEOM_SPHERE:
            r = float(dims[0]) if len(dims) > 0 else 0.0
            return np.array([[+r,0,0],[-r,0,0],[0,+r,0],[0,-r,0],[0,0,+r],[0,0,-r]], dtype=np.float32)
        if geom_type in (p.GEOM_CYLINDER, p.GEOM_CAPSULE):
            r = float(dims[0]) if len(dims) > 0 else 0.0
            h = float(dims[1]) if len(dims) > 1 else 0.0
            hz = 0.5*h
            return np.array([
                [+r,0,+hz],[-r,0,+hz],[0,+r,+hz],[0,-r,+hz],
                [+r,0,-hz],[-r,0,-hz],[0,+r,-hz],[0,-r,-hz],
            ], dtype=np.float32)
        return None

    def _collect_descendant_links(self, rid, root_link_idx):
        descendants = []
        for link_idx in range(p.getNumJoints(rid)):
            cur = link_idx
            while cur != -1:
                if cur == root_link_idx:
                    descendants.append(link_idx)
                    break
                cur = p.getJointInfo(rid, cur)[16]  # parent link index
        return descendants

    def _collect_link_candidate_points(self, rid, link_idx):
        pts_world = []
        col_data = p.getCollisionShapeData(rid, link_idx)
        if col_data:
            for cd in col_data:
                geom_type = int(cd[2])
                dims = cd[3]
                local_pos = cd[5]
                local_orn = cd[6]
                local_pts = self._shape_points_local(geom_type, dims)
                if local_pts is None:
                    aabb_min, aabb_max = p.getAABB(rid, link_idx)
                    pts_world.append(self._aabb_corners(aabb_min, aabb_max))
                    continue
                frame_pos, frame_orn = p.multiplyTransforms([0,0,0], [0,0,0,1], local_pos, local_orn)
                pts_world.append(self._transform_local_points(frame_pos, frame_orn, local_pts))

        if not pts_world:
            aabb_min, aabb_max = p.getAABB(rid, link_idx)
            pts_world.append(self._aabb_corners(aabb_min, aabb_max))

        link_state = p.getLinkState(rid, link_idx, computeForwardKinematics=True)
        link_pos, link_orn = link_state[4], link_state[5]
        return np.vstack([self._transform_local_points(link_pos, link_orn, pts) for pts in pts_world])

    def _get_gripper_extreme_points(self, rid, joint6_idx, fallback_link_idx):
        """
        joint6 기준으로 그리퍼 후보점에서
        - 가장 먼 2점(far1, far2)
        - 가장 가까운 1점(near)
        을 반환.

        NOTE:
        - 이전 구현은 end-slice/PCA를 쓰면서 축 추정 오차가 누적될 수 있었음.
        - 여기서는 요청대로 "거리 기준"을 직접 적용.
        - 후보 링크는 가능하면 plate_link(실제 그리퍼 본체)를 우선 사용.
        """
        joint6_pos = np.array(
            p.getLinkState(rid, joint6_idx, computeForwardKinematics=True)[4],
            dtype=np.float32,
        )

        preferred = self.plateL if rid == self.urL else self.plateR
        candidate_links = [preferred] if preferred is not None else [fallback_link_idx]

        pts = []
        for link_idx in candidate_links:
            try:
                pts.append(self._collect_link_candidate_points(rid, link_idx))
            except Exception:
                continue

        if not pts:
            pts = [self._collect_link_candidate_points(rid, fallback_link_idx)]

        all_pts = np.vstack(pts)
        dist = np.linalg.norm(all_pts - joint6_pos, axis=1)

        near = all_pts[int(np.argmin(dist))]
        far_idx = np.argsort(dist)[-2:]
        far1, far2 = all_pts[far_idx[0]], all_pts[far_idx[1]]

        if self.force_far_from_sack and self.forced_far_point is not None:
            far1 = self.forced_far_point.copy()
            far2 = self.forced_far_point.copy()

        return far1, far2, near, joint6_pos

    def _update_robot_realtime_debug(self, force=False):
        if not getattr(self, "enable_robot_debug", False):
            return
        now = time.time()
        if (not force) and (now - self._robot_last_debug_t < self.robot_debug_update_sec):
            return

        arms = [("L", self.urL, self.jL, self.eeL), ("R", self.urR, self.jR, self.eeR)]
        for arm, rid, jids, ee_idx in arms:
            joint_msgs = []
            for i, jid in enumerate(jids):
                ls = p.getLinkState(rid, jid, computeForwardKinematics=True)
                pos = np.array(ls[4], dtype=np.float32)
                joint_msgs.append(f"J{i+1}=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f})")
                self.robot_joint_marker_ids[arm][i] = self._draw_cross_marker(
                    pos.tolist(), [1, 0, 0], self.robot_joint_marker_half, self.robot_joint_marker_ids[arm][i]
                )

            joint6_idx = jids[-1]
            far1, far2, near_pt, j6 = self._get_gripper_extreme_points(rid, joint6_idx, ee_idx)
            yz_ang = self._yz_angle_from_xy_plane_deg(far1, near_pt)
            lo, hi = self.forced_yz_angle_range_deg
            tag = "OK" if (lo <= yz_ang <= hi) else "OUT"

            self.robot_ee_marker_ids[arm][0] = self._draw_cross_marker(
                far1.tolist(), [0, 0, 1], self.robot_ee_marker_half, self.robot_ee_marker_ids[arm][0]
            )
            self.robot_ee_marker_ids[arm][1] = self._draw_cross_marker(
                far2.tolist(), [0, 0, 1], self.robot_ee_marker_half, self.robot_ee_marker_ids[arm][1]
            )
            self.robot_ee_marker_ids[arm][2] = self._draw_cross_marker(
                near_pt.tolist(), [0, 1, 1], self.robot_ee_marker_half * 1.3, self.robot_ee_marker_ids[arm][2]
            )

        self._robot_last_debug_t = now

    
    def set_forced_far_from_sack(self, enabled=True):
        state = self._get_sack_state()
        if state is None:
            print("[WARN] sack state unavailable; far override not changed")
            return None

        c = state["center"]
        size = state["size"]
        length_y = float(size[1])
        target = np.array([float(c[0]), float(c[1] - length_y - 0.1), 0.1], dtype=np.float32)

        self.force_far_from_sack = bool(enabled)
        self.forced_far_point = target
        st = "ON" if self.force_far_from_sack else "OFF"
        print(f"[FORCE_FAR_FROM_SACK] {st} target=({target[0]:+.3f},{target[1]:+.3f},{target[2]:+.3f})")
        return target.copy()

    def _yz_angle_from_xy_plane_deg(self, p_far, p_near):
        v = np.array(p_far, dtype=np.float32) - np.array(p_near, dtype=np.float32)
        ang = np.degrees(np.arctan2(abs(float(v[2])), max(abs(float(v[1])), 1e-9)))
        return float(ang)
    

    
    def check_dual_collision(self,robotA, robotB, safety_dist=0.02):
        p.performCollisionDetection()
    
        contacts = p.getContactPoints(bodyA=robotA, bodyB=robotB)
        if len(contacts) > 0:
            return True, 0.0, contacts  # 충돌
    
        near = p.getClosestPoints(bodyA=robotA, bodyB=robotB, distance=safety_dist)
        if len(near) > 0:
            min_d = min([pt[8] for pt in near])  # pt[8] = contact distance
            return False, float(min_d), near     # 충돌은 아니지만 위험

        return False, float("inf"), []