# Geometry Specification Table

이 표는 `project_dual_sack`의 현재 구현 기준입니다. 자루는 하나의 shared articulated sealed pillow-sack skeleton이고, scenario는 같은 skeleton의 state/parameter만 바꿉니다.

## Sack Skeleton

| Part | Body name | Joint | Initial pose | Size | Scenario state change | Role |
|---|---|---|---|---|---|---|
| 기준 body | `bag_frame` | `freejoint` | 바닥 위 중심, `z ~= 0.095 m` | 전체 기준 `0.42 x 0.24 x 0.09 m` | 모든 scenario에서 동일 skeleton 유지, pose/tilt만 state로 변화 가능 | physics |
| 외피 | `visual_skin` | 없음 | `bag_frame`에 attached | closed pillow mesh | `top_width_scale`, `lower_width_scale`, `top_crown_scale`, `visual_bulge_y`로 silhouette 변화 | visual-only |
| 밀봉 상단 cap | `sealed_top_cap_visual` | 없음 | top seam 위 낮은 위치 | low-profile box/cap | underfilled에서 crown 낮아짐, fold scenario에서 seam 노출 cue와 함께 보임 | visual-only |
| 상단 seam chain | `top_seam_chain/top_seam_00..06` | passive hinge | 자루 상단 중앙 축 | segment 약 `0.04 x 0.012 x 0.004 m` | fold coverage에 따라 exposed fraction 감소, top grasp candidate 변화 | physics/grasp candidate |
| 왼쪽 shoulder | `shoulder_left_panels/shoulder_left_00..02` | passive hinge + spring/damping/limit | 좌측 상부 경사면 | broad panel | underfilled에서 inward rest angle 증가, poke/contact force로 deflection 발생 | physics |
| 오른쪽 shoulder | `shoulder_right_panels/shoulder_right_00..02` | passive hinge + spring/damping/limit | 우측 상부 경사면 | broad panel | baseline은 좌우 대칭, eccentric/fold/jam에서 비대칭 가능 | physics |
| 정면 face | `front_face_panels/front_face_00..03` | passive hinge | 정면 edge 부근 | broad panel | jammed/underfilled에서 폭과 경사 cue 변화 | physics |
| 후면 face | `back_face_panels/back_face_00..03` | passive hinge | 후면 edge 부근 | broad panel | 정면과 대칭적으로 자루 두께 유지 | physics |
| 좌측 gusset | `side_gusset_left/side_gusset_left_00..01` | passive hinge | 좌측 옆 접힘면 | broad panel | jammed에서 inward squeeze, eccentric에서 side bulge cue | physics |
| 우측 gusset | `side_gusset_right/side_gusset_right_00..01` | passive hinge | 우측 옆 접힘면 | broad panel | jammed에서 inward squeeze | physics |
| 하부 belly | `lower_belly_panels/lower_belly_00..03` | passive hinge + spring/damping/limit | 하부 지지면 | broad panel | underfilled lower bulge, scoop insertion opening, sag response | physics |
| 하부 sling | `bottom_sling` | passive slide | 하부 중심 | support plate | hidden support release 후 `bottom_sag_mm` 증가 | physics |
| 왼쪽 fold | `fold_patch_left` | passive hinge | top seam root 부근 | folded patch | simple/severe에서 coverage 증가, brushing/preload로 약간 변화 | physics |
| 오른쪽 fold | `fold_patch_right` | passive hinge | top seam root 부근 | folded patch | severe fold에서 활성 | physics |
| 주 payload | `payload_main` | slide x/y/z + spring/damping | 내부 중심 또는 하부 | ellipsoid | underfilled는 하부, eccentric은 한쪽으로 이동 | physics |
| 보조 payload | `payload_aux` | slide x/y/z + spring/damping | 내부 | ellipsoid | eccentric에서 보조 질량 및 CoM offset 증가 | physics |
| 숨은 지지 | `hidden_support` | 없음 | 자루 하부 아래 | thin box | post_separation_sag에서 초기 on, release test에서 off | physics switch |
| 왼쪽 이웃 | `neighbor_left` | 없음 | 자루 좌측 | rigid blocker | jammed_between_neighbors에서 bag width reduction | physics |
| 오른쪽 이웃 | `neighbor_right` | 없음 | 자루 우측 | rigid blocker | jammed_between_neighbors에서 bag width reduction | physics |

## Dual Robot

| Part | Body / Joint name | Joint | Initial pose | Size | Scenario state change | Role |
|---|---|---|---|---|---|---|
| UR5e A base | `ur5e_2f140_base` | fixed base | `(0.0, -0.55, 0.035)`, yaw `0 deg` | MuJoCo Menagerie UR5e mesh `base_0/base_1` | scenario와 무관 | Robot A base |
| UR5e A joints | `ur5e_2f140_shoulder_pan_joint`, `shoulder_lift_joint`, `elbow_joint`, `wrist_1_joint`, `wrist_2_joint`, `wrist_3_joint` | 6 hinge joints with UR5e mesh links | home pose from GUI | Menagerie UR5e kinematic tree and visual meshes | GUI slider/IK로 제어 | top patch grasp robot |
| Robotiq 2F-140 | `robotiq_2f140` | attached to UR5e A EE | wrist 끝단 | max gap `140 mm` surrogate | jaw gap GUI로 제어 | 2F grasp |
| 2F fingers | `robotiq_2f140_finger_left/right`, joints `finger_left_slide`, `finger_right_slide` | slide joints | open gap `140 mm` | pad length 약 `84 mm` | close/open GUI로 제어 | local patch capture/contact |
| UR5e B base | `ur5e_scoop_base` | fixed base | `(0.0, 0.55, 0.035)`, yaw `180 deg` | MuJoCo Menagerie UR5e mesh `base_0/base_1` | scenario와 무관 | Robot B base |
| UR5e B joints | `ur5e_scoop_shoulder_pan_joint`, `shoulder_lift_joint`, `elbow_joint`, `wrist_1_joint`, `wrist_2_joint`, `wrist_3_joint` | 6 hinge joints with UR5e mesh links | home pose from GUI | Menagerie UR5e kinematic tree and visual meshes | GUI slider/IK로 제어 | scoop support robot |
| Scoop tool | `scoop_tool` | attached to UR5e B EE | wrist 끝단 | plate `0.17 x 0.12 m` 수준 | GUI/IK로 삽입 위치 제어 | under-support insertion |
| Scoop plate/lips | `scoop_plate`, `scoop_back_lip`, `scoop_side_lip_left/right` | fixed geoms | scoop local frame | front-open shallow scoop | scenario와 무관 | bottom support formation |

참고: 현재 로봇은 `mujoco_menagerie/universal_robots_ur5e`의 UR5e mesh hierarchy를 사용합니다. 자루 벤치마크 안정성을 위해 UR5e arm link mesh는 visual/inertial 용도로 두고, 실제 작업 접촉은 Robotiq pad와 scoop tool만 담당하도록 구성했습니다.

## Shape Change Readout

GUI의 `show physics patches`를 켜면 실제 움직이는 articulated patch가 보입니다. 기본 외피 `visual_skin`은 physics-free이므로, 외피 mesh 자체가 cloth처럼 실시간으로 찌그러지는 것은 아닙니다. 대신 현재 구현에서 실제 형상 변화는 아래 물리 관절에서 발생하고 GUI에 수치로 표시됩니다.

| Metric | Physical source |
|---|---|
| `shoulder_deflection_mm` | `shoulder_left/right_*_hinge`와 tip site displacement |
| `top_patch_change_mm` | `top_seam_03` passive hinge/site displacement |
| `lower_belly_opening_mm` | `lower_belly_01/02` tip distance change |
| `bottom_sag_mm` | `bottom_sling_slide` displacement |
