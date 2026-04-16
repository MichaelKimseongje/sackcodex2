# Shape Change Diagnostics

이 파일은 `visual_skin`이 아니라 실제 articulated physics patches의 움직임을 기준으로 작성됩니다.
`visual_skin`은 physics-free sealed sack silhouette이며, 실시간 변형 판정에는 사용하지 않습니다.

렌더 결과:
- physics patch debug: `project_dual_sack/out/shape_diagnostics/*_physics_patch_debug.png`
- visual skin only: `project_dual_sack/out/shape_diagnostics/*_visual_skin.png`
- overlay: `project_dual_sack/out/shape_diagnostics/*_overlay.png`

## baseline_filled
- tuning_applied: `True`
- rigid_like_flag: `False`
- before tuning: shoulder_deflection_mm=1.037, top_patch_change_mm=1.179, lower_belly_opening_mm=0.024, bottom_sag_mm=0.000
- after tuning: shoulder_deflection_mm=1.217, top_patch_change_mm=0.902, lower_belly_opening_mm=1.582, bottom_sag_mm=0.000, fold_exposed_fraction_before_after=1.000->1.000
- shoulder recovery after poke: `False`
- top reference drop during support release: `0.000 mm`

## underfilled
- tuning_applied: `True`
- rigid_like_flag: `False`
- before tuning: shoulder_deflection_mm=0.286, top_patch_change_mm=0.392, lower_belly_opening_mm=0.024, bottom_sag_mm=0.200
- after tuning: shoulder_deflection_mm=0.351, top_patch_change_mm=0.290, lower_belly_opening_mm=1.559, bottom_sag_mm=0.270, fold_exposed_fraction_before_after=1.000->1.000
- shoulder recovery after poke: `True`
- top reference drop during support release: `0.000 mm`

## top_fold_simple
- tuning_applied: `True`
- rigid_like_flag: `False`
- before tuning: shoulder_deflection_mm=1.041, top_patch_change_mm=1.181, lower_belly_opening_mm=0.025, bottom_sag_mm=0.000
- after tuning: shoulder_deflection_mm=1.213, top_patch_change_mm=0.901, lower_belly_opening_mm=1.576, bottom_sag_mm=0.000, fold_exposed_fraction_before_after=0.700->0.713
- shoulder recovery after poke: `False`
- top reference drop during support release: `0.000 mm`

## top_fold_severe
- tuning_applied: `True`
- rigid_like_flag: `False`
- before tuning: shoulder_deflection_mm=1.011, top_patch_change_mm=1.168, lower_belly_opening_mm=0.024, bottom_sag_mm=0.006
- after tuning: shoulder_deflection_mm=1.261, top_patch_change_mm=0.925, lower_belly_opening_mm=1.549, bottom_sag_mm=0.000, fold_exposed_fraction_before_after=0.380->0.386
- shoulder recovery after poke: `False`
- top reference drop during support release: `0.000 mm`

## eccentric_fill
- tuning_applied: `True`
- rigid_like_flag: `False`
- before tuning: shoulder_deflection_mm=1.883, top_patch_change_mm=2.365, lower_belly_opening_mm=0.020, bottom_sag_mm=0.837
- after tuning: shoulder_deflection_mm=3.147, top_patch_change_mm=3.564, lower_belly_opening_mm=1.404, bottom_sag_mm=0.579, fold_exposed_fraction_before_after=1.000->1.000
- shoulder recovery after poke: `False`
- top reference drop during support release: `0.132 mm`

## jammed_between_neighbors
- tuning_applied: `True`
- rigid_like_flag: `False`
- before tuning: shoulder_deflection_mm=0.334, top_patch_change_mm=0.363, lower_belly_opening_mm=0.024, bottom_sag_mm=0.000
- after tuning: shoulder_deflection_mm=0.319, top_patch_change_mm=0.401, lower_belly_opening_mm=1.308, bottom_sag_mm=0.000, fold_exposed_fraction_before_after=1.000->1.000
- shoulder recovery after poke: `True`
- top reference drop during support release: `0.000 mm`

## post_separation_sag
- tuning_applied: `True`
- rigid_like_flag: `False`
- before tuning: shoulder_deflection_mm=0.817, top_patch_change_mm=1.252, lower_belly_opening_mm=0.027, bottom_sag_mm=26.838
- after tuning: shoulder_deflection_mm=0.258, top_patch_change_mm=2.705, lower_belly_opening_mm=1.641, bottom_sag_mm=29.168, fold_exposed_fraction_before_after=1.000->1.000
- shoulder recovery after poke: `True`
- top reference drop during support release: `11.458 mm`

## Conclusion
모든 선택 scenario에서 force-driven test에 의해 측정 가능한 patch motion이 발생했습니다.
