# Cross-Section Quasi-Rigid Sack Benchmark

이 프로젝트는 MuJoCo에서 `support-state formation under shape and pile uncertainty`를 평가하기 위한 task-driven sack surrogate입니다. 정확한 직물, 입자, 공기, 마찰 재질을 복제하는 full soft simulator가 아니라, 2F gripper와 scoop가 top patch graspability, load transfer, safe transport를 실험할 수 있게 만든 안정적인 benchmark입니다.

## Topology Single Source Of Truth

현재 런타임 모델의 기준은 `topology_unification_report.md`입니다. 실제 MuJoCo model에서 아래 개수가 맞는지 `inspect_topology_runtime.py`가 직접 검증합니다.

```text
visible outer shell = 30 bodies
hidden inner shell  = 15 bodies
ballast             = 3 or 4 bodies
longitudinal slices = 5
seam windows        = 5
```

## Bag Size

```text
SACK_LENGTH    = 0.420 m
SACK_WIDTH     = 0.240 m
SACK_THICKNESS = 0.150 m
SACK_Z         = 0.105 m
```

## Runtime Body Naming

Visible outer shell은 5개 slice마다 정확히 6개 body를 가집니다.

```text
top_seam_band_00..04
upper_left_00..04
upper_right_00..04
lower_left_00..04
lower_right_00..04
bottom_00..04
```

Hidden inner load shell은 5개 slice마다 정확히 3개 body를 가집니다.

```text
inner_upper_00..04
inner_lower_00..04
inner_bottom_00..04
```

Ballast는 DEM이 아니라 coarse load redistribution용 제한 slide body입니다.

```text
ballast_main
ballast_aux_1
ballast_aux_2
ballast_aux_3
```

Legacy topology인 `rim_ring`, `upper_skirt`, `lower_skirt`, `bottom_cradle`, `outer_upper_*`, `outer_lower_*`, `inner_*_load_*`는 런타임 모델에 남아 있으면 실패로 판정합니다.

## Run

Dual UR5e + Robotiq 2F-140 + scoop GUI:

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' project_dual_sack/run_dual_ur5_gui.py --scenario underfilled
```

Runtime topology inventory와 렌더 증거 생성:

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' project_dual_sack/inspect_topology_runtime.py --scenario underfilled
```

출력 경로:

```text
project_dual_sack/out/runtime_topology_inventory/
```

생성되는 주요 파일:

```text
actual_runtime_body_inventory.csv
actual_runtime_joint_inventory.csv
actual_runtime_site_inventory.csv
topology_mismatch_report.md
outer_shell_only.png
inner_shell_only.png
ballast_only.png
overlay.png
top_view.png
side_view.png
perspective_view.png
```

## Scenarios

```text
baseline_filled
underfilled
top_fold_simple
top_fold_severe
eccentric_fill
jammed_between_neighbors
post_separation_sag
```

## Interpretation

이 모델은 pure friction proof가 아니며 full soft sack simulator도 아닙니다. 실제 물리는 articulated outer shell, hidden inner shell, distributed ballast가 담당하고, optional visual skin은 외형 보조용입니다. 최종 과업은 top patch search 자체가 아니라 dual robot의 support-state formation과 safe transport입니다.
