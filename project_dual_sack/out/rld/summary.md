# Force-Responsive Articulated Sack Diagnostic

이 리포트는 full soft/flex/DEM이 아니라 visible articulated outer shell 기반 surrogate가 실제로 local deformation을 만드는지 확인한 결과입니다.

진단 모드:
- `free`: 자루 전체가 자유롭게 움직이는 기본 task 상태입니다.
- `anchored`: 진단 동안만 약한 world anchor를 걸어 whole-body slip과 local deformation을 분리합니다.

판정 규칙:
- world-frame motion만 크고 bag-frame local deformation이 작으면 rigid-like로 판단합니다.
- anchored mode에서도 local deformation이 작으면 topology 자체가 너무 rigid-like하다고 판단합니다.
- anchored mode에서는 변형이 있고 free mode에서만 미끄러지면 topology보다 support/friction 조건 문제가 더 큽니다.

## underfilled
- rigid_like rows: `0/14`
- free: max local `458.866 mm`, world translation `153.689 mm`, world rotation `109.263 deg`
- anchored: max local `455.028 mm`, world translation `68.521 mm`, world rotation `58.130 deg`

| test | mode | rigid_like | local_mm | world_mm | shoulder_mm | top_mm | belly_open_mm | bottom_sag_mm | fold_exposed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shoulder_poke_left | free | False | 43.4838 | 12.1922 | 21.4253 | 0.2118 | 41.4866 | 6.6315 | 1.000->1.000 |
| shoulder_poke_right | free | False | 44.3406 | 12.2392 | 21.5755 | 0.2144 | 42.1456 | 6.7888 | 1.000->1.000 |
| top_preload | free | False | 325.3381 | 153.6890 | 60.8987 | 212.0274 | 213.6116 | 324.4114 | 1.000->1.000 |
| side_push | free | False | 3.1362 | 0.1166 | 2.5157 | 0.2529 | 2.2227 | 1.6301 | 1.000->1.000 |
| scoop_insert | free | False | 458.8660 | 78.0571 | 851.8275 | 5.4289 | 370.2255 | 457.8849 | 1.000->1.000 |
| support_release | free | False | 0.8102 | 0.5366 | 0.2715 | 0.3381 | 0.8025 | 0.5701 | 1.000->1.000 |
| fold_brushing | free | False | 0.6447 | 0.4262 | 0.2177 | 0.2531 | 0.6346 | 0.4594 | 1.000->1.000 |
| shoulder_poke_left | anchored | False | 38.7979 | 10.4092 | 20.4076 | 0.2129 | 37.0530 | 6.5681 | 1.000->1.000 |
| shoulder_poke_right | anchored | False | 38.8650 | 10.4322 | 20.4264 | 0.2139 | 37.1375 | 6.5751 | 1.000->1.000 |
| top_preload | anchored | False | 233.5376 | 68.5207 | 43.3376 | 151.1187 | 152.1765 | 222.8706 | 1.000->1.000 |
| side_push | anchored | False | 3.1322 | 0.1130 | 2.5139 | 0.2527 | 2.2194 | 1.6299 | 1.000->1.000 |
| scoop_insert | anchored | False | 455.0283 | 32.5796 | 837.9266 | 2.4987 | 365.5018 | 455.0214 | 1.000->1.000 |
| support_release | anchored | False | 0.7922 | 0.5192 | 0.2634 | 0.3380 | 0.7822 | 0.5485 | 1.000->1.000 |
| fold_brushing | anchored | False | 0.6325 | 0.4146 | 0.2123 | 0.2529 | 0.6211 | 0.4441 | 1.000->1.000 |

## Explicit Conclusion
- anchored diagnostic에서 local deformation이 측정되어 topology 전체를 단일 강체로 보기는 어렵습니다.
- free diagnostic에서도 bag-frame local deformation이 측정되어 단순 whole-body slip만 발생한 것은 아닙니다.
- top preload와 lower-belly/scoop insertion에서 목표 수준의 local shape response가 측정되었습니다.
- max bag-frame local deformation: `458.866 mm`
- max top_patch_change_mm: `212.027 mm`
- max lower_belly_opening_mm: `370.226 mm`
- simple fold exposed fraction delta: `0.000`
- post_separation_sag max bottom_sag_mm: `0.000 mm`
