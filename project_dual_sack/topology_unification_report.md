# Revised Topology Unification Report

이 문서가 현재 `project_dual_sack`의 topology single source of truth입니다. 기준은 `right_slice_shell_contract.png`의 오른쪽 단면을 길이 방향으로 5개 복제한 cross-section-inspired quasi-rigid sack surrogate입니다.

## Acceptance Inventory

| Category | Expected runtime count | Count rule |
|---|---:|---|
| visible outer shell bodies | 30 | end-cap visuals 제외 |
| hidden inner shell bodies | 15 | 5 slices x 3 inner bodies |
| ballast bodies | 3 or 4 | DEM 아님, coarse load surrogate |
| longitudinal slices | 5 | `left_end`, `left_mid`, `center`, `right_mid`, `right_end` |
| seam candidate windows | 5 | left / left-center / center / right-center / right |
| end-cap visual bodies | 2 | visible outer shell 30개에는 포함하지 않음 |

## Revised Body Hierarchy

```text
bag_frame
  visual_skin
    sealed_top_cap_visual
    left_end_cap_visual
    right_end_cap_visual
    visual_print_mark

  visible_articulated_outer_shell
    top_grasp_rail
      slice_00_left_end
        top_seam_band_00
          upper_left_00
            lower_left_00
          upper_right_00
            lower_right_00
        bottom_00
      slice_01_left_mid
        ...
      slice_04_right_end
        top_seam_band_04
          upper_left_04
            lower_left_04
          upper_right_04
            lower_right_04
        bottom_04

  hidden_inner_load_shell
    inner_upper_00..04
    inner_lower_00..04
    inner_bottom_00..04

  ballast_main
  ballast_aux_1
  ballast_aux_2
  ballast_aux_3

  optional_top_edge_occlusion_patch
    top_edge_occlusion_left
    top_edge_occlusion_right
```

## End-Cap Design Table

| Name | Type | Physics | Render group | Purpose |
|---|---|---:|---:|---|
| `left_end_cap_visual` | visual-only body with named geom | no | 1 | left longitudinal end가 뚫려 보이지 않게 닫음 |
| `right_end_cap_visual` | visual-only body with named geom | no | 1 | right longitudinal end가 뚫려 보이지 않게 닫음 |
| `left_end_cap_physics` | optional thin physics panel | default off | debug only | 필요 시 end-side contact 보조 |
| `right_end_cap_physics` | optional thin physics panel | default off | debug only | 필요 시 end-side contact 보조 |

기본 구현은 visual-only입니다. end-cap을 physics-bearing body로 강하게 넣으면 자루 끝이 rigid wall처럼 굳을 수 있으므로, 현재 연구 단계에서는 sealed appearance 보장을 우선합니다.

## Runtime Body Names

Visible outer shell, 30 bodies:

```text
top_seam_band_00..04
upper_left_00..04
upper_right_00..04
lower_left_00..04
lower_right_00..04
bottom_00..04
```

Hidden inner shell, 15 bodies:

```text
inner_upper_00..04
inner_lower_00..04
inner_bottom_00..04
```

Ballast:

```text
ballast_main
ballast_aux_1
ballast_aux_2
ballast_aux_3
```

Seam candidate windows:

```text
site_top_seam_left
site_top_seam_left_center
site_top_seam_center
site_top_seam_right_center
site_top_seam_right
```

## Render Acceptance List

```text
outer_shell_only.png
inner_shell_only.png
ballast_only.png
overlay.png
front_view.png
side_view.png
longitudinal_end_view.png
top_view.png
```

`outer_shell_only.png`는 30개 articulated outer shell과 2개 end-cap visual closure가 함께 보여야 합니다. `longitudinal_end_view.png`는 길이 방향 끝단이 열린 단면처럼 보이지 않는지 확인하기 위한 전용 렌더입니다.

## Runtime Failure Conditions

아래 이름이 로드된 MuJoCo model에 남아 있으면 topology mismatch입니다.

```text
rim_ring
upper_skirt
lower_skirt
bottom_cradle
outer_upper_*
outer_lower_*
outer_mid_*
outer_bottom_*
inner_front_load_*
inner_back_load_*
inner_bottom_load_*
payload_main
payload_aux
rigid_core
central_core
```

## Runtime Proof Command

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' project_dual_sack/inspect_topology_runtime.py --scenario underfilled
```
