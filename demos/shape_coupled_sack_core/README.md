# Shape-Coupled Semi-Deformable Sack Core

이 데모는 full soft sack simulator가 아닙니다. MuJoCo `flex`, `flexcomp`, DEM을 사용하지 않고, rigid/articulated panel 기반으로 자루의 핵심 shape mode를 안정적으로 근사합니다.

목표는 `dual robot 2F + scoop`의 support-state formation under uncertainty 평가입니다.

## 핵심 아이디어

기존 rigid proxy처럼 외형만 고정하지 않고, 각 scenario의 상태가 형상과 파지 후보에 연결되도록 만들었습니다.

```text
state -> shape -> grasp/support response
```

예:

```text
underfilled
payload_main below
-> shoulder panels inward
-> upper half narrow, lower half wide
-> preferred patch shifts to shoulder
```

## 구현된 scenario

- `baseline_filled`: 비교 기준이 되는 충진 자루입니다.
- `underfilled`: 내부 payload가 아래쪽에 있고, shoulder panel은 inward droop, belly panel은 lower bulge를 만듭니다.
- `top_fold_simple`: fold root flap 1개가 seam circumference의 약 32%를 덮습니다.
- `top_fold_severe`: fold root flap 2개가 seam circumference의 약 64%를 덮습니다.
- `post_separation_sag`: support 제거 후 bottom_sling과 belly panels가 top보다 더 크게 처집니다.

## 모델 구조

- `bag_frame`: freejoint를 가진 자루 root body입니다.
- `sealed_top_cap_visual`: 작은 visual-only 밀봉 cap입니다.
- `seam_band_00..07`: 상단 봉합선이자 2F grasp candidate입니다.
- `shoulder_panel_00..07`: 넓은 rigid cloth panel입니다. 저충진에서 안쪽으로 좁아집니다.
- `belly_panel_00..07`: 하부 rigid cloth panel입니다. 저충진/처짐에서 하부 bulge와 sag를 만듭니다.
- `bottom_sling`: 하부 support structure입니다. post-separation에서 아래로 움직일 수 있습니다.
- `fold_root_flap_1`, `fold_root_flap_2`: 접힌 상단 천 patch입니다.
- `payload_main`: 내부 충진을 나타내는 ellipsoid rigid body입니다.
- `visual_upper_hull`, `visual_lower_hull`: sealed sack appearance를 위한 visual-only hull입니다.

## 구조 정리 기준

| 구성 요소 | 결정 | 이유 |
|---|---|---|
| `floor` | 유지하되 거의 투명하게 표시 | 바닥 접촉, 마찰, 넘어짐, 스쿠프 삽입 평가에 필요합니다. 큰 사각판처럼 보일 필요는 없어서 시각화를 약하게 낮췄습니다. |
| `bag_frame` freejoint | 유지 | 자루 전체가 이동, 회전, 전도되어야 하므로 필요합니다. |
| `payload_main` | 유지 | 저충진, 내부 하중 이동, 옆으로 쓰러졌을 때 내용물이 한쪽으로 치우치는 현상을 표현합니다. |
| `seam_band` | 유지 | 밀봉된 상단 봉합선과 2F 파지 후보를 표현합니다. |
| `shoulder_panel`, `belly_panel` | 물리용으로 유지, 기본 화면에서는 거의 숨김 | 발산을 줄이고 형상 변화량을 측정하기 위한 내부 물리 구조입니다. 보이면 자루가 사각 강체처럼 보이므로 투명도를 낮췄습니다. |
| 큰 외부 사각 cloth patch | 삭제 | 자루답지 않고 박스처럼 보이게 만들어 제거했습니다. |
| panel wrinkle capsule | 추가 | 사각 패널 대신 움직이는 얇은 주름선으로 접힘/처짐을 보여줍니다. |
| `visual_symmetric_sealed_hull` | 추가 | 위/아래/좌우가 대칭인 기준 밀봉 자루 외피입니다. |
| `visual_upper_hull`, `visual_lower_hull` | 거의 투명하게 유지 | 기존 코드 호환용 이름은 남기되, 기준 외형을 비대칭으로 만들지 않도록 시각 영향은 낮췄습니다. |
| `sealed_top_stitch_visual`, `sealed_bottom_stitch_visual` | 유지/추가 | 상단과 하단 모두 봉합선 cue를 가져 위아래 외형이 한쪽만 특이해 보이지 않도록 했습니다. |

## 1대1 대응

| 실제 현상 | 모델 대응 | 봐야 할 것 | metric |
|---|---|---|---|
| 저충진 | payload가 아래, shoulder inward, belly outward | 위가 좁고 아래가 넓은 밀봉 자루 | `upper_half_width < lower_half_width`, 낮은 `com_z`, `preferred_patch_label=shoulder` |
| 단순 상단 접힘 | fold flap 1개 | 상단 seam 일부만 접힌 천이 덮음 | `fold_coverage_fraction=0.32`, `rim_exposed_fraction=0.68` |
| 심한 상단 접힘 | fold flap 2개 | 상단 seam 대부분이 겹친 접힘으로 가려짐 | `fold_coverage_fraction=0.64`, `rim_exposed_fraction=0.36` |
| 분리 직후 처짐 | support 제거 후 bottom_sling slide/drop | after에서 하부가 top보다 더 내려감 | `bottom_drop_0p3s > top_drop_0p3s`, `sag_ratio > 1` |

## 이미지와 metric 저장

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/shape_coupled_sack_core/validate_scenarios.py --scenario all
```

출력 폴더:

```text
D:\Michael\2025\01.Research\01.Parceldetection\16.Pybullet\bullet3-master (1)\bullet3-master\examples\pybullet\examples\DeformableTest\Sackcodex2\demos\shape_coupled_sack_core\out
```

주요 출력:

- `baseline_front.png`
- `underfilled_front.png`
- `underfilled_side.png`
- `top_fold_simple_front.png`
- `top_fold_severe_front.png`
- `post_separation_sag_before.png`
- `post_separation_sag_after.png`
- `compare_baseline_underfilled.png`
- `compare_baseline_top_fold_simple.png`
- `compare_baseline_top_fold_severe.png`
- `compare_post_sag_before_after.png`
- `summary.csv`
- `summary.md`

## 접촉 기반 형상변화 데모

정지 이미지가 아니라, 그리퍼가 상단 patch를 누르고 닫은 뒤 미세 리프트를 하고, 스쿠프가 하부를 받치면서 자루의 형상 모드가 어떻게 바뀌는지 보여줍니다.

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/shape_coupled_sack_core/run_shape_response_demo.py --scenario underfilled
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/shape_coupled_sack_core/run_shape_response_demo.py --scenario top_fold_simple
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/shape_coupled_sack_core/run_shape_response_demo.py --scenario post_separation_sag --post-release
```

출력 폴더:

```text
D:\Michael\2025\01.Research\01.Parceldetection\16.Pybullet\bullet3-master (1)\bullet3-master\examples\pybullet\examples\DeformableTest\Sackcodex2\demos\shape_coupled_sack_core\out\shape_response\<scenario>
```

주요 파일:

- `observe.png`: 접촉 전 초기 형상입니다.
- `pinch_shape_change.png`: 2F가 닫히며 local patch가 눌리고 shoulder panel이 접힌 상태입니다.
- `micro_lift_sag.png`: 상단을 살짝 들어 올릴 때 하부와 payload가 늦게 따라오며 처짐이 생긴 상태입니다.
- `scoop_support.png`: 스쿠프가 하부 지지 영역에 들어가 support state를 만들기 시작한 상태입니다.
- `support_lift_recovered_shape.png`: 2F와 scoop가 함께 받칠 때 처짐이 줄어드는 상태입니다.
- `shape_response_summary.csv`: `upper_half_width`, `lower_half_width`, `shoulder_angle_local_deg`, `bottom_sag_m`, `payload_slide_y_m`, `bag_com_z_m` 변화가 저장됩니다.
- `shape_response_demo.mp4`: 전체 단계가 짧은 영상으로 저장됩니다. mp4 저장이 안 되는 환경이면 `frames/`에 PNG sequence가 저장됩니다.

이 데모에서 보는 핵심은 외형만 바뀌는 것이 아니라 `패널 각도`, `하부 sling 처짐`, `내부 payload 위치`, `자루 CoM`이 함께 변한다는 점입니다.

## Viewer

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/shape_coupled_sack_core/run_viewer.py --scenario underfilled
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/shape_coupled_sack_core/run_viewer.py --scenario top_fold_severe
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/shape_coupled_sack_core/run_viewer.py --scenario post_separation_sag --post-release
```

## Dual UR5 GUI

왼쪽 UR5에는 2F gripper, 오른쪽 UR5에는 scoop gripper가 붙습니다. 기존 GUI처럼 joint degree slider와 end-effector x/y/z IK slider로 조작할 수 있습니다.

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/shape_coupled_sack_core/run_dual_ur5_gui.py --scenario underfilled
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/shape_coupled_sack_core/run_dual_ur5_gui.py --scenario top_fold_simple
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/shape_coupled_sack_core/run_dual_ur5_gui.py --scenario post_separation_sag --post-release
```

GUI에는 `Move 2F to nearest grasp` 버튼이 있습니다. 이 버튼은 `seam`, `shoulder`, `fold` grasp site 중 가까운 후보로 왼쪽 2F gripper를 이동시킵니다. 이후 `Close gripper`로 닫을 수 있습니다.

Smoke test:

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/shape_coupled_sack_core/run_dual_ur5_gui.py --scenario underfilled --smoke-test
```

## 해석 주의

이 모델은 정확한 cloth/material simulator가 아닙니다. 대신 자루 취급에서 중요한 저차원 형상 변화 모드를 rigid/articulated panel로 근사합니다.

```text
full soft deformation 전체 X
task-relevant shape mode O
```

따라서 “자루 물리엔진”이라기보다 “shape-coupled task-driven sack benchmark”로 설명하는 것이 맞습니다.
