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
- free: max local `18.987 mm`, world translation `18.677 mm`, world rotation `7.782 deg`
- anchored: max local `18.916 mm`, world translation `21.381 mm`, world rotation `6.477 deg`

| test | mode | rigid_like | local_mm | world_mm | shoulder_mm | top_mm | belly_open_mm | bottom_sag_mm | fold_exposed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shoulder_poke_left | free | False | 3.4236 | 1.6374 | 1.0119 | 0.0085 | 1.0208 | 1.0450 | 1.000->1.000 |
| shoulder_poke_right | free | False | 3.4298 | 6.3243 | 0.7912 | 0.0078 | 0.7162 | 0.3475 | 1.000->1.000 |
| top_preload | free | False | 3.4518 | 2.5308 | 0.5558 | 1.1142 | 1.1899 | 3.2237 | 1.000->1.000 |
| side_push | free | False | 3.9740 | 1.5938 | 0.6150 | 0.0090 | 0.9154 | 0.7820 | 1.000->1.000 |
| scoop_insert | free | False | 18.9869 | 18.6768 | 0.8765 | 0.0088 | 11.2803 | 0.0192 | 1.000->1.000 |
| support_release | free | False | 4.9938 | 2.2602 | 0.7875 | 0.0108 | 1.2977 | 1.6708 | 1.000->1.000 |
| fold_brushing | free | False | 3.9557 | 1.6104 | 0.6008 | 0.0097 | 1.0186 | 0.9451 | 1.000->1.000 |
| shoulder_poke_left | anchored | False | 3.4234 | 1.4859 | 1.0415 | 0.0083 | 0.8822 | 0.3732 | 1.000->1.000 |
| shoulder_poke_right | anchored | False | 3.4309 | 1.1128 | 0.8060 | 0.0081 | 0.7059 | 0.0704 | 1.000->1.000 |
| top_preload | anchored | False | 3.4519 | 1.4522 | 0.5883 | 0.9994 | 1.0226 | 2.4939 | 1.000->1.000 |
| side_push | anchored | False | 3.9752 | 1.4212 | 0.6359 | 0.0088 | 0.8179 | 0.2386 | 1.000->1.000 |
| scoop_insert | anchored | False | 18.9158 | 21.3813 | 0.8703 | 0.0101 | 10.8854 | 0.0065 | 1.000->1.000 |
| support_release | anchored | False | 4.9925 | 2.0037 | 0.8237 | 0.0090 | 1.0842 | 0.9330 | 1.000->1.000 |
| fold_brushing | anchored | False | 3.9558 | 1.4398 | 0.6240 | 0.0084 | 0.8981 | 0.4015 | 1.000->1.000 |

## Explicit Conclusion
- anchored diagnostic에서 local deformation이 측정되어 topology 전체를 단일 강체로 보기는 어렵습니다.
- free diagnostic에서도 bag-frame local deformation이 측정되어 단순 whole-body slip만 발생한 것은 아닙니다.
- top preload와 lower-belly/scoop insertion에서 목표 수준의 local shape response가 측정되었습니다.
- max bag-frame local deformation: `18.987 mm`
- max top_patch_change_mm: `1.114 mm`
- max lower_belly_opening_mm: `11.280 mm`
- simple fold exposed fraction delta: `0.000`
- post_separation_sag max bottom_sag_mm: `0.000 mm`
