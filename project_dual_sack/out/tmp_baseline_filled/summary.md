# Force-Responsive Articulated Sack Diagnostic

이 리포트는 full soft/flex/DEM이 아니라 visible articulated outer shell 기반 surrogate가 실제로 local deformation을 만드는지 확인한 결과입니다.

진단 모드:
- `free`: 자루 전체가 자유롭게 움직이는 기본 task 상태입니다.
- `anchored`: 진단 동안만 약한 world anchor를 걸어 whole-body slip과 local deformation을 분리합니다.

판정 규칙:
- world-frame motion만 크고 bag-frame local deformation이 작으면 rigid-like로 판단합니다.
- anchored mode에서도 local deformation이 작으면 topology 자체가 너무 rigid-like하다고 판단합니다.
- anchored mode에서는 변형이 있고 free mode에서만 미끄러지면 topology보다 support/friction 조건 문제가 더 큽니다.

## baseline_filled
- rigid_like rows: `0/14`
- free: max local `21.703 mm`, world translation `22.872 mm`, world rotation `8.007 deg`
- anchored: max local `22.924 mm`, world translation `21.913 mm`, world rotation `3.236 deg`

| test | mode | rigid_like | local_mm | world_mm | shoulder_mm | top_mm | belly_open_mm | bottom_sag_mm | fold_exposed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shoulder_poke_left | free | False | 2.1481 | 4.8242 | 0.5026 | 0.0105 | 0.0805 | 0.3949 | 1.000->1.000 |
| shoulder_poke_right | free | False | 2.1591 | 4.8497 | 0.5064 | 0.0109 | 0.0860 | 0.3940 | 1.000->1.000 |
| top_preload | free | False | 2.1628 | 1.5619 | 0.4983 | 1.1584 | 0.0026 | 0.5590 | 1.000->1.000 |
| side_push | free | False | 2.7425 | 0.2991 | 0.6117 | 0.0110 | 0.0066 | 0.4509 | 1.000->1.000 |
| scoop_insert | free | False | 21.7027 | 22.8717 | 0.6255 | 0.0126 | 10.4077 | 0.0000 | 1.000->1.000 |
| support_release | free | False | 4.1103 | 0.3014 | 0.8494 | 0.0138 | 0.0038 | 0.4536 | 1.000->1.000 |
| fold_brushing | free | False | 2.7475 | 0.3014 | 0.6123 | 0.0122 | 0.0030 | 0.4536 | 1.000->1.000 |
| shoulder_poke_left | anchored | False | 2.1488 | 0.6333 | 0.5013 | 0.0101 | 0.0543 | 0.4082 | 1.000->1.000 |
| shoulder_poke_right | anchored | False | 2.1577 | 0.6211 | 0.5049 | 0.0108 | 0.0590 | 0.4063 | 1.000->1.000 |
| top_preload | anchored | False | 2.1588 | 0.9298 | 0.4957 | 0.9869 | 0.0021 | 0.5967 | 1.000->1.000 |
| side_push | anchored | False | 2.7474 | 0.2905 | 0.6117 | 0.0125 | 0.0066 | 0.4523 | 1.000->1.000 |
| scoop_insert | anchored | False | 22.9241 | 21.9127 | 0.6292 | 0.0128 | 10.2368 | 0.0000 | 1.000->1.000 |
| support_release | anchored | False | 4.1103 | 0.2901 | 0.8494 | 0.0143 | 0.0038 | 0.4529 | 1.000->1.000 |
| fold_brushing | anchored | False | 2.7476 | 0.2901 | 0.6124 | 0.0123 | 0.0030 | 0.4529 | 1.000->1.000 |

## Explicit Conclusion
- anchored diagnostic에서 local deformation이 측정되어 topology 전체를 단일 강체로 보기는 어렵습니다.
- free diagnostic에서도 bag-frame local deformation이 측정되어 단순 whole-body slip만 발생한 것은 아닙니다.
- top preload와 lower-belly/scoop insertion에서 목표 수준의 local shape response가 측정되었습니다.
- max bag-frame local deformation: `22.924 mm`
- max top_patch_change_mm: `1.158 mm`
- max lower_belly_opening_mm: `10.408 mm`
- simple fold exposed fraction delta: `0.000`
- post_separation_sag max bottom_sag_mm: `0.000 mm`
