# Hinge-Locked Twin-Shell Topology Diff

이 파일은 현재 generated XML의 main sack topology를 요약합니다.

## Removed From Main Topology

| old concept | status | reason |
|---|---|---|
| `rim_ring / upper_skirt / lower_skirt / bottom_cradle` | not used as main bag body | central rim/skirt 느낌을 제거하고 hinge-locked panel chain으로 대체 |
| `connected_outer_shell` | demoted to unused legacy builder code | main XML generation path에서 호출하지 않음 |
| `top_grasp_rail_lift` | removed from generated XML | visible shell에 slide를 쓰지 않기 위해 제거 |
| central large rigid core | not used | distributed ballast masses로 대체 |

## New Body Hierarchy

```text
bag_frame
  visual_skin                         # physics-free overlay
  visible_articulated_outer_shell
    top_grasp_rail                    # hinge only
      top_seam_chain
        top_seam_00..10               # hinge + fixed seam patch
      outer_upper_left/right_segments
        outer_upper_left/right_00..10 # hinge body, panel fixed to body
          outer_mid_front/back_00..10
            outer_lower_left/right_00..10
              outer_bottom_edge_left/right_00..10
      outer_bottom_edge_center
  hidden_inner_load_shell
    inner_front_load_00..04
    inner_back_load_00..04
    inner_bottom_load_00..02
  ballast_main
  ballast_aux_1
  ballast_aux_2
  ballast_aux_3
  optional_top_edge_occlusion_patch
```

## Motion Policy

- visible outer shell panel geom은 해당 hinge body에 고정됨
- visible outer shell에는 slide joint 없음
- panel 위치 변화는 부모 hinge chain의 회전 때문에 종속적으로 발생
- ballast slide는 내부 질량 재분포 surrogate에만 사용

## Grasp Candidate Bodies

- `top_grasp_rail`
- `top_seam_00..10`
- `outer_upper_left/right_00..10`
- `top_edge_occlusion_left/right`

## Scoop Support Bodies

- `outer_lower_left/right_00..10`
- `outer_bottom_edge_left/right_00..10`
- `outer_bottom_edge_center`
- `inner_bottom_load_00..02`

## Coupling

- `chain_outer_vertical_*`: upper -> mid -> lower -> bottom 종속 angle chain
- `couple_left_right_*`: 좌우 외피가 따로 놀지 않게 하는 약한 대칭 coupling
- `couple_outer_*_to_inner_*`: visible outer shell과 hidden inner load shell 연결
- `couple_ballast_*`: distributed ballast와 inner/bottom response 연결
