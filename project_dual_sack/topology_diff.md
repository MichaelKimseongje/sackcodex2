# Topology Diff

이 파일은 이전 mixed topology에서 현재 5-slice cross-section topology로 바뀐 내용을 요약합니다. 실제 판정은 `inspect_topology_runtime.py`가 로드된 MuJoCo model에서 직접 수행합니다.

## Removed From Runtime

| Old runtime concept | Status |
|---|---|
| `rim_ring` | removed |
| `upper_skirt` | removed |
| `lower_skirt` | removed |
| `bottom_cradle` | removed |
| 11-column / 100+ strip panel topology | removed |
| `outer_upper_*`, `outer_lower_*`, `outer_bottom_edge_*` main topology | removed from runtime |
| `inner_front_load_*`, `inner_back_load_*`, `inner_bottom_load_*` | removed from runtime |
| `payload_main`, `payload_aux` central payload naming | replaced by distributed ballast bodies |

## New Runtime Topology

| Category | Body names | Count |
|---|---|---:|
| longitudinal slice roots | `slice_00_left_end` .. `slice_04_right_end` | 5 |
| visible outer shell | `top_seam_band_00..04` | 5 |
| visible outer shell | `upper_left_00..04`, `upper_right_00..04` | 10 |
| visible outer shell | `lower_left_00..04`, `lower_right_00..04` | 10 |
| visible outer shell | `bottom_00..04` | 5 |
| hidden inner shell | `inner_upper_00..04` | 5 |
| hidden inner shell | `inner_lower_00..04` | 5 |
| hidden inner shell | `inner_bottom_00..04` | 5 |
| ballast | `ballast_main`, `ballast_aux_1`, `ballast_aux_2`, `ballast_aux_3` | 4 |
| support / neighbor | `hidden_support`, `neighbor_left`, `neighbor_right` | 3 |

## Candidate Seam Windows

| Window | Site |
|---|---|
| left | `site_top_seam_left` |
| left-center | `site_top_seam_left_center` |
| center | `site_top_seam_center` |
| right-center | `site_top_seam_right_center` |
| right | `site_top_seam_right` |

## Coupling Policy

Adjacent slices are coupled by fixed tendons. Left and right panels are softly mirrored, but asymmetry can still appear from ballast offset, neighbor jamming, or robot contact. The whole bag is never fixed as one rigid body.
