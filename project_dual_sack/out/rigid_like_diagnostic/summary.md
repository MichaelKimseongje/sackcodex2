# Rigid-Like Diagnostic Report

이 리포트는 diagnostic-only pass입니다. topology, geometry spec, joint/tendon parameter를 변경하지 않았고 auto-tuning도 수행하지 않았습니다.

진단 모드:
- `free`: 자루가 task/evaluation처럼 자유롭게 움직입니다.
- `anchored`: 짧은 진단 동안만 `bag_frame`에 약한 world anchor force를 적용하여 whole-body slip과 local deformation을 분리합니다.

판정 규칙:
- world-frame motion만 크고 `bag_frame` local deformation이 거의 0이면 rigid-like입니다.
- joint angle 변화와 shape response가 모두 거의 0이면 rigid-like입니다.
- anchored mode에서도 local deformation이 거의 0이면 topology 자체가 너무 rigid-like입니다.
- anchored mode에서는 local deformation이 있으나 free mode에서 slip만 크면 topology보다 support/friction/anchoring 문제가 큽니다.

## underfilled
- rigid_like rows: `0/14`
- free: max local `4.138 mm`, world translation `10.028 mm`, world rotation `3.505 deg`
- anchored: max local `3.596 mm`, world translation `3.274 mm`, world rotation `1.550 deg`

| test | mode | rigid_like | local_mm | world_mm | rot_deg | shoulder_mm | top_mm | belly_open_mm | width_red_mm | bottom_sag_mm | limit_hits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shoulder_poke_left | free | False | 0.5960 | 1.4626 | 0.2864 | 0.0743 | 0.0016 | 0.0000 | 0.0000 | 0.0000 | 1 |
| shoulder_poke_right | free | False | 0.5851 | 0.7598 | 0.1463 | 0.0423 | 0.0016 | 0.0000 | 0.0000 | 0.0000 | 1 |
| top_preload | free | False | 0.5858 | 0.4802 | 0.0489 | 0.0310 | 0.0016 | 0.0000 | 0.0000 | 0.2849 | 1 |
| side_push | free | False | 0.6893 | 0.2682 | 0.1046 | 0.0127 | 0.0019 | 0.0000 | 0.0000 | 0.0000 | 1 |
| scoop_insert | free | False | 4.1383 | 10.0283 | 3.5049 | 0.0133 | 0.0020 | 0.0716 | 0.0000 | 0.0000 | 1 |
| support_release | free | False | 0.8789 | 0.3647 | 0.0262 | 0.0162 | 0.0024 | 0.0000 | 0.0000 | 0.0000 | 1 |
| fold_brushing | free | False | 0.6834 | 0.2674 | 0.0262 | 0.0127 | 0.0019 | 0.0000 | 0.0000 | 0.0000 | 1 |
| shoulder_poke_left | anchored | False | 0.5625 | 1.2325 | 0.1095 | 0.0744 | 0.0016 | 0.0000 | 0.0000 | 0.0000 | 1 |
| shoulder_poke_right | anchored | False | 0.5847 | 0.7304 | 0.1323 | 0.0423 | 0.0016 | 0.0000 | 0.0000 | 0.0000 | 1 |
| top_preload | anchored | False | 0.5855 | 0.4725 | 0.0421 | 0.0310 | 0.0016 | 0.0000 | 0.0000 | 0.2844 | 1 |
| side_push | anchored | False | 0.6887 | 0.2642 | 0.1004 | 0.0127 | 0.0019 | 0.0000 | 0.0000 | 0.0000 | 1 |
| scoop_insert | anchored | False | 3.5959 | 3.2738 | 1.5500 | 0.0133 | 0.0020 | 0.0713 | 0.0000 | 0.0000 | 1 |
| support_release | anchored | False | 0.8788 | 0.3576 | 0.0236 | 0.0162 | 0.0024 | 0.0000 | 0.0000 | 0.0000 | 1 |
| fold_brushing | anchored | False | 0.6831 | 0.2633 | 0.0195 | 0.0127 | 0.0019 | 0.0000 | 0.0000 | 0.0000 | 1 |

## Explicit Conclusion
- anchored diagnostic에서 local deformation이 측정되므로 topology 전체가 완전 rigid-like라고 보기는 어렵습니다.
- free diagnostic에서도 bag-frame local deformation이 함께 측정되어 단순 whole-body slip만 발생한 것은 아닙니다.
- 하지만 top preload 또는 scoop insertion의 국소 shape response가 작아서 support-state surrogate로는 아직 부족합니다.
- max bag-frame local deformation: `4.138 mm`
- max top_patch_change_mm: `0.002 mm`
- max lower_belly_opening_mm: `0.072 mm`
