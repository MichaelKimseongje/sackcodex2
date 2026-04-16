# Sealed Articulated Sack Surrogate V2

이 데모는 full soft sack simulator가 아닙니다. MuJoCo `flex`, `flexcomp`, DEM, 다입자 충진을 쓰지 않고, 안정적인 rigid/articulated body로 `support-state formation under uncertainty`를 평가하기 위한 task-driven sack surrogate입니다.

## 핵심 구조

- `bag_frame`: freejoint를 가진 자루 root body입니다.
- `sealed_top_cap_visual`: visual only의 작은 low-profile 밀봉 cap입니다. 큰 tabletop 판처럼 보이지 않게 축소했습니다.
- `seam_band_00..07`: 실제 grasp candidate인 물리 봉합선 segment입니다. 화면에서는 상단 봉합선처럼 보입니다.
- `shoulder_panel_00..07`: 기존 rod/capsule upper skirt를 대체한 넓은 rigid cloth panel입니다. hinge, damping, limit이 있습니다.
- `belly_panel_00..07`: 하부 bulge와 sag를 표현하는 넓은 rigid cloth panel입니다.
- `bottom_sling`: 골조처럼 보이는 cradle 대신 낮고 넓은 하부 sling으로 내부 하중을 받칩니다.
- `fold_root_flap_1`, `fold_root_flap_2`: seam band root에서 시작하는 folded top patch입니다.
- `payload_main`, `payload_aux`: 내부 충진과 편심 충진을 표현하는 ellipsoid rigid body입니다.
- `neighbor_left`, `neighbor_right`: 이웃 자루와 끼인 상태를 표현하는 blocker입니다.
- `visual_upper_hull`, `visual_lower_hull`: sealed sack appearance를 위한 visual only 외피입니다. contact에는 참여하지 않습니다.

## Scenario

- `underfilled`: 겉은 밀봉된 자루이고, 내부 payload가 아래쪽에만 있습니다. shoulder panel은 inward droop bias를 가지며, upper half width가 lower half width보다 작게 기록됩니다.
- `top_fold_simple`: `fold_root_flap_1`만 사용해 seam circumference의 약 32%를 덮습니다. exposed seam patch가 남아 있습니다.
- `top_fold_severe`: `fold_root_flap_1`, `fold_root_flap_2`가 겹쳐 약 62%의 seam을 덮습니다. simple보다 graspable top patch가 적습니다.
- `eccentric_fill`: `payload_main + payload_aux`와 lower bulge가 한쪽으로 이동해 lateral COM offset과 비대칭 silhouette을 만듭니다.
- `jammed_between_neighbors`: 양쪽 neighbor가 bag 폭을 줄여 끼임 상태를 만듭니다.
- `post_separation_sag`: support 제거 후 belly panels와 bottom sling이 상단보다 더 아래로 처지는 상태를 보여줍니다.

## Scenario 이미지와 metric 생성

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/rigid_patch_sack/validate_scenarios.py --scenario all
```

저장 위치:

```text
D:\Michael\2025\01.Research\01.Parceldetection\16.Pybullet\bullet3-master (1)\bullet3-master\examples\pybullet\examples\DeformableTest\Sackcodex2\demos\rigid_patch_sack\out
```

주요 출력:

- `underfilled_front.png`
- `underfilled_side.png`
- `top_fold_simple_front.png`
- `top_fold_severe_front.png`
- `eccentric_fill_front.png`
- `eccentric_fill_side.png`
- `jammed_front.png`
- `post_separation_sag_before.png`
- `post_separation_sag_after.png`
- `summary.csv`
- `summary.md`

## Viewer

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/rigid_patch_sack/run_viewer.py --scenario underfilled
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/rigid_patch_sack/run_viewer.py --scenario top_fold_severe
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/rigid_patch_sack/run_viewer.py --scenario eccentric_fill
```

## 2F Grasp 평가

평가 후보는 `seam band`, `fold flap`, `shoulder patch`입니다.

`contact_only_eval`은 pure contact lower bound입니다. latch/connect를 켜지 않습니다.

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/rigid_patch_sack/run_eval.py --mode contact_only_eval --scenario underfilled --target-label auto
```

`qualification_gated_connect`는 pure friction proof가 아니라 task-driven local patch stabilization surrogate입니다. 조건을 통과한 selected/captured patch 1개 또는 최대 2개 body에만 connect force를 적용합니다. bag 전체를 rigid weld하거나 parenting하지 않습니다.

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/rigid_patch_sack/run_eval.py --mode qualification_gated_connect --scenario underfilled --target-label auto
```

`visual_demo`는 시연용입니다.

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/rigid_patch_sack/run_eval.py --mode visual_demo --scenario top_fold_simple --target-label fold
```

## Dual UR5 GUI

왼쪽 UR5에는 Robotiq-style 2F gripper, 오른쪽 UR5에는 scoop gripper가 붙어 있습니다. 기존 GUI처럼 joint degree slider와 end-effector x/y/z IK slider를 사용할 수 있습니다.

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/rigid_patch_sack/run_dual_ur5_gui.py --scenario underfilled
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/rigid_patch_sack/run_dual_ur5_gui.py --scenario top_fold_simple
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/rigid_patch_sack/run_dual_ur5_gui.py --scenario eccentric_fill
```

GUI smoke test:

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/rigid_patch_sack/run_dual_ur5_gui.py --scenario underfilled --smoke-test
```

## 해석 주의

- 이 모델은 정확한 sack material simulator가 아닙니다.
- visual hull/cap은 sealed sack appearance를 위한 visual only cue입니다.
- 물리는 `seam_band`, `shoulder_panel`, `belly_panel`, `bottom_sling`, `fold_root_flap`, `payload`가 담당합니다.
- `qualification_gated_connect` 성공은 순수 2F 마찰 파지 증명이 아닙니다. local patch capture가 task qualification을 통과했을 때 support-state formation 실험을 안정화하는 surrogate입니다.
