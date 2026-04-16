# Shape-Coupled Hinge-Locked Sack Benchmark

이 프로젝트는 MuJoCo에서 `support-state formation under shape and pile uncertainty`를 보기 위한 task-driven sack surrogate입니다.

정확한 직물, 공기, 입자 흐름, 마대 재질을 복제하는 full soft sack simulator가 아닙니다. 목표는 2F gripper와 scoop가 자루의 상단 patch를 잡고, 하부 지지 상태를 만들고, 안전 이송이 가능한지 평가하는 안정적인 benchmark를 만드는 것입니다.

## 현재 자루 구현

현재 main topology는 `hinge-locked visible articulated outer shell + hidden inner load shell + distributed ballast masses`입니다.

- `bag_frame`: freejoint가 있는 자루 전체 기준 body입니다.
- `top_grasp_rail`: 2F gripper가 접근하는 상단 봉합 edge입니다.
- `top_seam_00..10`: 상단 edge를 여러 segment로 나눈 grasp candidate입니다.
- `outer_upper_left/right_00..10`: 상단 shoulder 쪽 visible outer shell panel입니다.
- `outer_mid_front/back_00..10`: 자루 몸통 중간 visible outer shell panel입니다.
- `outer_lower_left/right_00..10`: 하부 belly visible outer shell panel입니다.
- `outer_bottom_edge_left/right_00..10`: 하단 edge를 따라 움직이는 bottom shell panel입니다.
- `outer_bottom_edge_center`: 하단 중앙 폐쇄감을 주는 hinge-linked center panel입니다.
- `outer_bottom_closure_left/right_geom`: 뒤집었을 때 하부가 크게 열린 것처럼 보이지 않도록 좌우 bottom edge에 붙인 겹침 closure panel입니다.
- `hidden_inner_load_shell`: 외형이 아니라 내부 하중 경로와 bottom sag를 담당하는 숨김 load shell입니다.
- `ballast_main`, `ballast_aux_1..3`: DEM 대신 사용하는 coarse distributed ballast mass입니다.
- `visual_skin`: physics-free 외형 overlay입니다. 평가와 디버그에서는 꺼서 실제 physics patch를 봅니다.

## 중요한 물리 단순화

MuJoCo body hierarchy는 기본적으로 tree 구조라서, 실제 천 포대처럼 모든 가장자리를 완전한 폐루프(closed loop)로 동시에 묶기 어렵습니다. 그래서 현재 하단은 `outer_bottom_edge_left/right`, `outer_bottom_edge_center`, `outer_bottom_closure_left/right_geom`, 그리고 bottom-edge tendon coupling을 조합해서 “닫힌 하부처럼 보이고 움직이는” 구조로 만들었습니다.

이 모델에서 움직이는 것은 panel의 위치를 임의로 순간 이동시키는 방식이 아닙니다.

- 각 visible panel geom은 자기 body에 고정되어 있습니다.
- 각 body는 부모 body에 hinge joint로 연결됩니다.
- 실제 움직임은 hinge angle 변화와 부모-자식 body chain의 종속 운동에서 나옵니다.
- visible shell은 slide joint를 사용하지 않습니다.
- 각 판은 정사각형 tile이 아니라 x방향으로 긴 직사각형 strip입니다.
- 상단은 `sealed_top_cap_hinge_locked_geom`로 덮어서 top opening처럼 보이지 않게 했습니다.
- 하단은 겹침 closure panel과 center hinge panel로 뒤집었을 때 열림을 줄였습니다.

## 로봇 충돌 처리

UR5e mesh는 Menagerie mesh를 사용한 visual geometry입니다. mesh 자체는 복잡하고 불안정할 수 있어서 기본적으로 visual-only입니다.

자루와 로봇이 서로 통과하지 않도록, UR5e 링크에는 보이지 않는 collision proxy를 추가했습니다.

- collision proxy group: `5`
- 기본 GUI에서는 숨김
- GUI에서 `robot collision proxy`를 켜면 확인 가능
- proxy는 자루와만 충돌하도록 `contype=4`, `conaffinity=0`으로 설정했습니다.
- 자루 physics panel은 robot proxy와 충돌하도록 `conaffinity=5`를 사용합니다.
- Robotiq 2F-140 palm/finger/pad와 scoop plate/support hull도 자루와 충돌합니다.

## 실행 방법

Dual UR5e + Robotiq 2F-140 + scoop GUI:

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' project_dual_sack/run_dual_ur5_gui.py --scenario underfilled
```

사용 가능한 scenario:

```text
baseline_filled
underfilled
top_fold_simple
top_fold_severe
eccentric_fill
jammed_between_neighbors
post_separation_sag
```

## GUI에서 볼 항목

- `physics patches`: 실제 물리 visible outer shell panel입니다.
- `inner load shell`: 숨겨진 내부 load shell입니다.
- `ballast masses`: 내부 coarse ballast mass입니다.
- `visual skin overlay`: physics-free 외형 overlay입니다.
- `robot collision proxy`: 자루와 충돌하는 UR5e proxy입니다.
- `contacts`: 로봇과 자루 panel의 contact point입니다.
- `contact force`: 접촉력 표시입니다.
- `site frames`, `body frames`: hinge-linked body/site 기준축입니다.
- `deformation monitor`: `shoulder_deflection_mm`, `top_patch_change_mm`, `lower_belly_opening_mm`, `bottom_sag_mm`, `bottom_edge_rollup_mm`를 표시합니다.

## 진단 실행

빠른 CSV 진단:

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' project_dual_sack/diagnose_rigid_like.py --scenario underfilled --no-render --no-video
```

전체 scenario 진단:

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' project_dual_sack/diagnose_rigid_like.py --scenario all
```

출력 경로:

```text
project_dual_sack/out/rld/
```

주요 출력:

- `summary.csv`
- `summary.md`
- `deformation_timeseries.csv`
- `joint_response.csv`
- `topology_diff.md`
- debug render/video 파일

## 해석 주의

이 모델은 pure friction proof가 아니고, full soft sack simulator도 아닙니다.

연구적으로는 다음을 보기 위한 reduced-order surrogate입니다.

- 상단 edge/shoulder patch를 잡았을 때 local top deformation이 생기는가
- lower belly와 bottom edge가 scoop insertion에 반응하는가
- underfilled에서 shoulder response가 더 커지는가
- eccentric fill에서 COM offset과 roll bias가 생기는가
- post-separation에서 bottom drop이 top drop보다 커지는가

최종 과업은 top patch search 자체가 아니라 dual robot의 support-state formation과 safe transport입니다.
