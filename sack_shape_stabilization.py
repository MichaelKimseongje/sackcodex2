import json
import math
import os
import time

import pybullet as p
import pybullet_data


# Tunable parameters.
MASS = 0.18
NEO_HOOKEAN_MU = 140.0
NEO_HOOKEAN_LAMBDA = 220.0
NEO_HOOKEAN_DAMPING = 0.04
FRICTION_COEFF = 0.8
COLLISION_MARGIN = 0.003
STABILIZATION_GAIN = 10.0

SPRING_ELASTIC_STIFFNESS = 24.0
SPRING_DAMPING_STIFFNESS = 0.08
REPULSION_STIFFNESS = 400.0
HELPER_MASS = 0.03
HELPER_RADIUS = 0.018
TIME_STEP = 1.0 / 240.0
BASE_POSITION = [0.0, 0.0, 1.15]
BASE_ORIENTATION_EULER = [0.0, 0.0, 0.0]
MESH_SCALE = 0.08

TOP_ANCHOR_COUNT = 8
MID_ANCHOR_COUNT = 10
BOT_ANCHOR_COUNT = 6

HEIGHT_GAIN = STABILIZATION_GAIN
MID_ALIGN_GAIN = 0.45 * STABILIZATION_GAIN
XY_CENTER_GAIN = 0.35 * STABILIZATION_GAIN
XY_DAMPING = 0.7
VERTICAL_DAMPING = 0.5


OBJ_PATH = r"D:/Michael/2025/01.Research/01.Parceldetection/16.Pybullet/bullet3-master (1)/bullet3-master/examples/pybullet/examples/DeformableTest/Sackcodex2/object/sack8.obj"
JSON_PATH = r"D:/Michael/2025/01.Research/01.Parceldetection/16.Pybullet/bullet3-master (1)/bullet3-master/examples/pybullet/examples/DeformableTest/Sackcodex2/object/sack_rest_shape8.json"


def load_rest_shape(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_obj_vertices(obj_path):
    verts = []
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                _, xs, ys, zs = line.split()[:4]
                verts.append([float(xs), float(ys), float(zs)])
    return verts


def sample_indices(indices, n):
    """Select evenly spaced entries from a sorted index list."""
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


def scaled_point(local_point):
    return [MESH_SCALE * float(x) for x in local_point]


def world_from_local(local_point, base_pos, base_orn):
    world_pos, _ = p.multiplyTransforms(base_pos, base_orn, scaled_point(local_point), [0, 0, 0, 1])
    return list(world_pos)


def create_helper_body(world_pos, rgba):
    visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=HELPER_RADIUS,
        rgbaColor=rgba,
    )
    body_id = p.createMultiBody(
        baseMass=HELPER_MASS,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=visual,
        basePosition=world_pos,
    )
    p.changeDynamics(
        body_id,
        -1,
        linearDamping=0.9,
        angularDamping=0.9,
    )
    return body_id


def anchor_ring_subset(soft_id, helper_id, helper_center_local, ring_indices, obj_vertices, n_anchors):
    selected = sample_indices(ring_indices, n_anchors)
    for vidx in selected:
        rest_vertex_local = obj_vertices[vidx]
        local_offset = [
            MESH_SCALE * (rest_vertex_local[0] - helper_center_local[0]),
            MESH_SCALE * (rest_vertex_local[1] - helper_center_local[1]),
            MESH_SCALE * (rest_vertex_local[2] - helper_center_local[2]),
        ]
        p.createSoftBodyAnchor(
            soft_id,
            int(vidx),
            helper_id,
            -1,
            local_offset,
        )


def apply_force(body_id, force):
    pos, _ = p.getBasePositionAndOrientation(body_id)
    p.applyExternalForce(
        objectUniqueId=body_id,
        linkIndex=-1,
        forceObj=[float(force[0]), float(force[1]), float(force[2])],
        posObj=pos,
        flags=p.WORLD_FRAME,
    )


def maintain_shape(top_ref, mid_ref, bot_ref, rest_height, top_radius, mid_radius):
    top_pos, _ = p.getBasePositionAndOrientation(top_ref)
    mid_pos, _ = p.getBasePositionAndOrientation(mid_ref)
    bot_pos, _ = p.getBasePositionAndOrientation(bot_ref)

    top_vel, _ = p.getBaseVelocity(top_ref)
    mid_vel, _ = p.getBaseVelocity(mid_ref)
    bot_vel, _ = p.getBaseVelocity(bot_ref)

    current_height = float(top_pos[2] - bot_pos[2])
    height_error = float(rest_height - current_height)

    top_force = [0.0, 0.0, 0.5 * HEIGHT_GAIN * height_error]
    bot_force = [0.0, 0.0, -0.5 * HEIGHT_GAIN * height_error]

    target_mid = [
        0.5 * (top_pos[0] + bot_pos[0]),
        0.5 * (top_pos[1] + bot_pos[1]),
        0.5 * (top_pos[2] + bot_pos[2]),
    ]
    mid_force = [
        MID_ALIGN_GAIN * (target_mid[0] - mid_pos[0]),
        MID_ALIGN_GAIN * (target_mid[1] - mid_pos[1]),
        MID_ALIGN_GAIN * (target_mid[2] - mid_pos[2]),
    ]

    for i in (0, 1):
        top_force[i] += -XY_CENTER_GAIN * top_pos[i] - XY_DAMPING * top_vel[i]
        mid_force[i] += -XY_CENTER_GAIN * mid_pos[i] - XY_DAMPING * mid_vel[i]
        bot_force[i] += -XY_CENTER_GAIN * bot_pos[i] - XY_DAMPING * bot_vel[i]

    top_force[2] += -VERTICAL_DAMPING * top_vel[2]
    mid_force[2] += -VERTICAL_DAMPING * mid_vel[2]
    bot_force[2] += -VERTICAL_DAMPING * bot_vel[2]

    # Light radial bias: keep top and mid helper centers from drifting too far in XY.
    top_xy = math.hypot(top_pos[0], top_pos[1])
    mid_xy = math.hypot(mid_pos[0], mid_pos[1])
    if top_xy > max(top_radius * 0.12, 1e-6):
        scale = -0.15 * STABILIZATION_GAIN / top_xy
        top_force[0] += scale * top_pos[0]
        top_force[1] += scale * top_pos[1]
    if mid_xy > max(mid_radius * 0.12, 1e-6):
        scale = -0.15 * STABILIZATION_GAIN / mid_xy
        mid_force[0] += scale * mid_pos[0]
        mid_force[1] += scale * mid_pos[1]

    apply_force(top_ref, top_force)
    apply_force(mid_ref, mid_force)
    apply_force(bot_ref, bot_force)


def is_grasping(gripper_id, sack_id):
    if gripper_id is None:
        return False
    return len(p.getContactPoints(bodyA=gripper_id, bodyB=sack_id)) > 0


def main():
    if not os.path.exists(OBJ_PATH):
        raise FileNotFoundError(f"OBJ not found: {OBJ_PATH}")
    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(f"JSON not found: {JSON_PATH}")

    rest_data = load_rest_shape(JSON_PATH)
    obj_vertices = load_obj_vertices(OBJ_PATH)

    top_idx = rest_data["Top_ring"]["indices"]
    mid_idx = rest_data["Mid_ring"]["indices"]
    bot_idx = rest_data["Bot_ring"]["indices"]
    top_center_local = rest_data["Top_ring"]["center"]
    mid_center_local = rest_data["Mid_ring"]["center"]
    bot_center_local = rest_data["Bot_ring"]["center"]
    rest_height = MESH_SCALE * float(rest_data["rest_height"])
    top_radius = MESH_SCALE * float(rest_data["Top_ring"]["avg_radius"])
    mid_radius = MESH_SCALE * float(rest_data["Mid_ring"]["avg_radius"])

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
    p.setGravity(0, 0, -9.81)
    p.setPhysicsEngineParameter(
        fixedTimeStep=TIME_STEP,
        numSubSteps=2,
        numSolverIterations=100,
        deterministicOverlappingPairs=1,
    )
    p.loadURDF("plane.urdf")

    base_orn = p.getQuaternionFromEuler(BASE_ORIENTATION_EULER)

    # Load the sack as a deformable body using NeoHookean + self-collision.
    sack_id = p.loadSoftBody(
        fileName=OBJ_PATH,
        basePosition=BASE_POSITION,
        baseOrientation=base_orn,
        scale=MESH_SCALE,
        mass=MASS,
        useNeoHookean=1,
        NeoHookeanMu=NEO_HOOKEAN_MU,
        NeoHookeanLambda=NEO_HOOKEAN_LAMBDA,
        NeoHookeanDamping=NEO_HOOKEAN_DAMPING,
        useMassSpring=1,
        useBendingSprings=1,
        springElasticStiffness=SPRING_ELASTIC_STIFFNESS,
        springDampingStiffness=SPRING_DAMPING_STIFFNESS,
        springDampingAllDirections=1,
        frictionCoeff=FRICTION_COEFF,
        useSelfCollision=1,
        useFaceContact=1,
        repulsionStiffness=REPULSION_STIFFNESS,
        collisionMargin=COLLISION_MARGIN,
    )
    p.changeVisualShape(sack_id, -1, rgbaColor=[0.86, 0.78, 0.58, 0.96])
    p.changeVisualShape(sack_id, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED)

    # Create helper references at the rest-shape ring centers.
    top_ref = create_helper_body(world_from_local(top_center_local, BASE_POSITION, base_orn), [1.0, 0.25, 0.25, 0.85])
    mid_ref = create_helper_body(world_from_local(mid_center_local, BASE_POSITION, base_orn), [0.25, 1.0, 0.25, 0.85])
    bot_ref = create_helper_body(world_from_local(bot_center_local, BASE_POSITION, base_orn), [0.25, 0.45, 1.0, 0.85])

    # Sparse anchors: enough to stabilize the overall silhouette without freezing the full mesh.
    anchor_ring_subset(sack_id, top_ref, top_center_local, top_idx, obj_vertices, TOP_ANCHOR_COUNT)
    anchor_ring_subset(sack_id, mid_ref, mid_center_local, mid_idx, obj_vertices, MID_ANCHOR_COUNT)
    anchor_ring_subset(sack_id, bot_ref, bot_center_local, bot_idx, obj_vertices, BOT_ANCHOR_COUNT)

    # Plug a real 2F gripper body id here when integrating with your robot scene.
    gripper_id = None

    while p.isConnected():
        if not is_grasping(gripper_id, sack_id):
            maintain_shape(top_ref, mid_ref, bot_ref, rest_height, top_radius, mid_radius)
        p.stepSimulation()
        time.sleep(TIME_STEP)


if __name__ == "__main__":
    main()


# Integration note:
# - Replace gripper_id = None with your real 2F gripper bodyUniqueId and keep is_grasping(gripper_id, sack_id).
# - Increase HEIGHT_GAIN / MID_ALIGN_GAIN / XY_CENTER_GAIN or helper anchor counts for stronger stabilization.
# - Lower those gains, lower NEO_HOOKEAN_MU/LAMBDA, or reduce anchor counts for softer behavior during grasp.
