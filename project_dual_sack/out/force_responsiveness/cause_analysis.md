# Force Responsiveness Cause Analysis

이 표는 `project_dual_sack/out/force_responsiveness/summary.csv`와 `summary.md`를 기준으로 작성한 원인 분석입니다.

중요 해석:

- `visual_skin`은 physics-free이므로 변형 판정에 사용하지 않습니다.
- `bag_frame`은 freejoint라서 world-frame motion과 bag-frame local deformation을 분리해서 봐야 합니다.
- 현재 결과는 완전 강체는 아닙니다. 모든 scenario에서 `bag_frame_local_deformation_mm`가 2.221-16.783 mm로 측정되어 `rigid_like_flag=False`입니다.
- 하지만 `top_patch_change_mm`와 `lower_belly_opening_mm`는 거의 0에 가까워, 사용자가 기대하는 "접촉부가 눈에 띄게 찌그러지는 자루"로는 아직 부족합니다.

## 1. 테스트별 원인 분석표

| 테스트 | 관찰 결과 | local deformation vs whole-body motion | 가장 의심되는 원인 | 결론 |
|---|---:|---|---|---|
| shoulder poke test | `shoulder_deflection_mm=0.040-0.298 mm` | per-test local deformation은 대략 `0.736-2.082 mm`지만 shoulder tip 자체 변위는 작음 | `shoulder_*_hinge`의 damping이 크고, 패널 tip/force 작용점의 moment arm이 작음. `visual_skin`도 shoulder panel pose를 따라가지 않음 | shoulder는 실제로 조금 반응하지만, 사용자가 눈으로 보는 외피 변형으로는 약함 |
| top preload deformation test | `top_patch_change_mm=0.000-0.002 mm` | world translation은 `0.533-1.860 mm`, local deformation은 `0.650-1.780 mm`인데 top seam 자체는 거의 안 움직임 | `top_seam_*_hinge` stiffness/damping이 가장 큼. joint range도 `-16 16 deg`로 좁고, downward preload가 hinge 회전으로 잘 변환되지 않음 | top patch는 현재 rigid-like 병목 |
| lateral squeeze test | summary field는 없지만 per-test metric이 `0.067-0.282 mm` 수준 | local deformation은 대략 `0.687-1.904 mm`, whole-body motion은 비교적 작음 | `side_gusset_*_hinge`가 stiff하고 side panel이 visual skin과 분리됨. squeeze가 broad side collapse로 연결되지 않음 | 옆면 압축은 물리 패치 내부에서는 조금 보이나 외형 변화가 약함 |
| scoop insertion deformation test | `lower_belly_opening_mm=0.027-0.036 mm` | world translation은 scenario에 따라 큼. eccentric은 `20.940 mm`, post sag는 `2.991 mm`. local deformation은 있지만 lower belly gap은 거의 안 열림 | `lower_belly_*_hinge` damping이 크고, lower belly pair가 서로 벌어지는 coupling이 없음. payload slide가 하부 개구를 따라오지 않음 | scoop 삽입은 whole-body motion이 섞이고 local opening은 부족 |
| support release sag test | baseline/fold/jammed는 `bottom_sag_mm=0`, post sag는 `12.959 mm` | post sag는 world translation `22.474 mm`, local deformation `16.783 mm`로 둘 다 큼 | `bottom_sling_slide`는 sag scenario에서는 작동하지만 일반 scenario에서는 hidden support/release 조건이 없음. stiffness/damping이 커서 baseline에서는 sag가 억제됨 | sag 현상은 post scenario에만 의도적으로 잘 나타남 |
| fold brushing test | simple `0.700->0.710`, severe `0.380->0.385` | fold site local change는 `0.083-0.161 mm` 수준 | `fold_patch_*_hinge` damping이 높고, fold root가 visual skin/top seam과 약하게만 연결됨 | fold는 label/cue로는 보이지만 brushing response는 약함 |

## 2. Joint별 stiffness 병목표

| 부품 / joint | 현재 설정 위치 | 현재 성향 | rigid-like하게 보이는 이유 | 우선 수정 |
|---|---|---|---|---|
| `top_seam_*_hinge` | `scenario_builder.py`, `_add_top_seam_chain()` | `stiffness=2.8`, `damping=8.0`, `range=-16 16` | top preload를 줘도 `top_patch_change_mm`가 거의 0. visual skin cap도 seam을 따라 움직이지 않음 | stiffness 감소, damping 감소, joint limit 확대, visual skin top coupling |
| `shoulder_left/right_*_hinge` | `_add_panel_group()`, shoulder panels | baseline `stiffness=2.0`, underfilled `1.15`, `damping=7.0`, `range=-42 42` | shoulder poke metric이 매우 작음. underfilled도 `0.040 mm`로 너무 작음 | stiffness 감소, damping 감소, joint limit 확대, shoulder visual coupling |
| `side_gusset_*_hinge` | `_add_side_gussets()` | `stiffness=1.8`, `damping=7.0`, `range=-38 38` | lateral squeeze가 broad side collapse로 보이지 않음 | stiffness 감소, damping 감소, limit 확대, side-belly strap coupling |
| `lower_belly_*_hinge` | `_add_panel_group()`, lower belly | baseline `stiffness=2.5`, underfilled `1.8`, sag `1.2`, `damping=8.5`, `range=-42 42` | scoop insertion에서 bag이 움직이지만 lower belly gap이 거의 안 열림 | stiffness 감소, damping 감소, limit 확대, lower pair tendon/strap coupling |
| `bottom_sling_slide` | `_add_bottom_sling()` | `stiffness=1.15`, `damping=12.0`, `range=-0.070 0.014` | post sag 외에는 bottom sag가 억제됨. sag 이후 회복도 둔함 | damping 감소, slide range 확대, payload-bottom coupling |
| `payload_main/aux_x/y/z` | `_add_payloads()` | `stiffness=2.2`, `damping=18.0`, x/y/z slide range 작음 | 내부 질량 이동이 외피 변형으로 충분히 전달되지 않음 | payload slide range 확대, damping 감소, payload-belly coupling |

## 3. Visual coupling 원인표

| visual 요소 | 현재 상태 | 변형이 안 보이는 이유 | 필요한 수정 |
|---|---|---|---|
| `visual_skin_main` | `bag_frame`에 붙은 physics-free mesh | physics patch가 움직여도 mesh vertex가 패치 pose를 따라가지 않음 | visual skin coupling 수정 |
| `sealed_top_cap_visual_geom` | visual only box | top seam이 움직여도 cap은 별도 변형 없음 | top seam site 기반 cap pose/scale update |
| `visual_print_mark_geom` | visual only marking | 자루 변형과 관계없이 고정 | 유지 가능. 단 변형 설명에는 사용 금지 |
| `mat_panel_hidden` physics patches | alpha가 낮음 | 실제 움직이는 패치가 기본 화면에서 거의 안 보임 | overlay/debug에서는 alpha 증가, 일반 화면에서는 visual skin coupling 필요 |

## 4. 수정 항목 1:1 대응 제안

| 원인 | stiffness 조정 | damping 조정 | joint limit 조정 | payload slide range 조정 | tendon/strap coupling 추가 | visual skin coupling 수정 |
|---|---|---|---|---|---|---|
| top preload가 top seam 변형으로 연결되지 않음 | `top_seam_*_hinge`: `2.8 -> 0.6-1.0` | `8.0 -> 1.5-3.0` | `+-16 deg -> +-35-45 deg` | 불필요 | top seam과 shoulder root를 약하게 묶는 strap | top skin crown vertices가 `top_seam_*` 평균 pose를 따르도록 수정 |
| shoulder poke에서 shoulder tip 변위가 작음 | shoulder hinge: baseline `2.0 -> 0.6-0.9`, underfilled `1.15 -> 0.25-0.45` | `7.0 -> 1.5-2.5` | `+-42 deg -> +-65 deg` | underfilled에서 payload z/y slide 확대 | left/right shoulder와 belly를 묶는 side strap | upper half width/shoulder drop이 shoulder panel pose를 반영하도록 수정 |
| lateral squeeze가 broad side collapse로 보이지 않음 | side gusset: `1.8 -> 0.5-0.8` | `7.0 -> 1.5-2.5` | `+-38 deg -> +-60 deg` | eccentric/jammed에서 y slide 확대 | side gusset과 lower belly 사이 diagonal strap | side silhouette가 side_gusset pose를 반영하도록 수정 |
| scoop insertion 때 lower belly opening이 거의 없음 | lower belly: `2.5/1.8/1.2 -> 0.4-0.8` | `8.5 -> 1.5-3.0` | `+-42 deg -> +-70 deg` | payload z/y range 확대 | lower_belly_01/02 pair opening strap 또는 scoop-contact tendon | lower visual skin underside가 lower belly sites를 따라 열리도록 수정 |
| post sag 외 scenario에서 bottom sag가 없음 | bottom_sling: `1.15 -> 0.4-0.7` | `12.0 -> 2.0-4.0` | slide `[-0.070,0.014] -> [-0.110,0.030]` | payload z range 확대 | payload-bottom_sling vertical strap | bottom skin vertices가 `site_bottom_sling` z를 반영하도록 수정 |
| fold brushing response가 작음 | fold patch: `1.8 -> 0.4-0.8` | `8.0 -> 1.5-3.0` | `[-70,35] -> [-90,60] deg` | 불필요 | fold root와 top seam 사이 weak strap | fold visual patch가 hinge qpos에 따라 더 크게 회전/두께 변화 |
| world-frame motion이 deformation metric에 섞임 | bag_frame 자체 stiffness 없음 | 해당 없음 | 해당 없음 | payload damping 조정으로 전체 흔들림 감소 | payload-shell coupling으로 하중 전달 분산 | metric은 항상 bag-frame local 기준으로 유지 |

## 5. 우선순위

1. `visual_skin_main` coupling을 먼저 수정해야 합니다. 현재 실제 패치가 조금 움직여도 외피가 따라오지 않아 사용자는 "안 변한다"고 봅니다.
2. `top_seam_*_hinge`, `shoulder_*_hinge`, `lower_belly_*_hinge`를 먼저 부드럽게 해야 합니다. 이 세 부위가 로봇 파지/스쿱 지지와 직접 연결됩니다.
3. lower belly pair와 shoulder-belly 사이에는 tendon/strap-like coupling이 필요합니다. 개별 패널만 움직이면 자루처럼 coordinated collapse가 보이지 않습니다.
4. payload slide range와 damping은 eccentric/sag에서만 크게 조정하는 편이 좋습니다. 모든 scenario에서 너무 크게 풀면 젤리처럼 보일 수 있습니다.

## 6. 현재 모델에 대한 결론

현재 모델은 완전한 단일 강체는 아닙니다. `bag_frame_local_deformation_mm`가 모든 scenario에서 0보다 충분히 크기 때문입니다.

하지만 사용자가 기대하는 마대자루형 변형, 즉 로봇이 누른 위치 주변이 눈에 띄게 눌리고, 스쿱이 들어갈 때 하부가 열리고, 접힌 부분이 brushing에 따라 visibly 바뀌는 수준은 아직 부족합니다.

따라서 다음 구현은 full soft로 돌아가는 것이 아니라, 위 표의 순서대로 passive joint를 부드럽게 하고 tendon/strap coupling과 visual skin coupling을 추가하는 것이 가장 타당합니다.
