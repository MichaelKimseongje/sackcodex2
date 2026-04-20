# Force-Responsive Semi-Deformable Diagnostics

이 진단은 `visual_skin`이 아니라 physics patches 기준으로 local deformation을 계산합니다.
`bag_frame`이 freejoint이므로 world-frame motion과 bag-frame aligned local deformation을 분리했습니다.

판정 기준:
- world-frame motion만 크고 bag-frame local deformation이 거의 0이면 `rigid_like_flag=True`입니다.
- bag-frame local deformation이 유의미하면 force-responsive surrogate로 해석합니다.

렌더:
- `*_physics_patch_debug.png`: visual skin을 거의 숨기고 실제 패치를 표시
- `*_visual_skin.png`: physics-free 외피만 표시
- `*_overlay.png`: 외피와 실제 패치를 함께 표시

## baseline_filled
- rigid_like_flag: `False`
- summary: shoulder_deflection_mm=0.121, top_patch_change_mm=0.001, lower_belly_opening_mm=0.033, bottom_sag_mm=0.000, bag_frame_local_deformation_mm=3.038
- rigid body motion: world_frame_bag_translation_mm=2.740, world_frame_bag_rotation_deg=4.907
- fold_exposed_fraction_before_after: `1.000->1.000`
- per-test separation:
  - shoulder_poke: metric_mm=0.121, world_translation_mm=1.127, world_rotation_deg=0.013, local_deformation_mm=2.072
  - top_preload: metric_mm=0.001, world_translation_mm=1.197, world_rotation_deg=0.011, local_deformation_mm=1.772
  - lateral_squeeze: metric_mm=0.282, world_translation_mm=1.104, world_rotation_deg=0.013, local_deformation_mm=1.895
  - scoop_insertion: metric_mm=0.033, world_translation_mm=2.740, world_rotation_deg=4.907, local_deformation_mm=3.038
  - support_release_sag: metric_mm=0.000, world_translation_mm=1.099, world_rotation_deg=0.009, local_deformation_mm=1.300
  - fold_brushing: metric_mm=0.000, world_translation_mm=0.000, world_rotation_deg=0.000, local_deformation_mm=0.000

## underfilled
- rigid_like_flag: `False`
- summary: shoulder_deflection_mm=0.040, top_patch_change_mm=0.001, lower_belly_opening_mm=0.033, bottom_sag_mm=0.000, bag_frame_local_deformation_mm=3.897
- rigid body motion: world_frame_bag_translation_mm=5.029, world_frame_bag_rotation_deg=1.803
- fold_exposed_fraction_before_after: `1.000->1.000`
- per-test separation:
  - shoulder_poke: metric_mm=0.040, world_translation_mm=0.772, world_rotation_deg=0.067, local_deformation_mm=1.052
  - top_preload: metric_mm=0.001, world_translation_mm=0.533, world_rotation_deg=0.037, local_deformation_mm=0.930
  - lateral_squeeze: metric_mm=0.116, world_translation_mm=0.626, world_rotation_deg=0.089, local_deformation_mm=0.978
  - scoop_insertion: metric_mm=0.033, world_translation_mm=5.029, world_rotation_deg=1.803, local_deformation_mm=3.897
  - support_release_sag: metric_mm=0.000, world_translation_mm=0.312, world_rotation_deg=0.020, local_deformation_mm=0.734
  - fold_brushing: metric_mm=0.000, world_translation_mm=0.000, world_rotation_deg=0.000, local_deformation_mm=0.000

## top_fold_simple
- rigid_like_flag: `False`
- summary: shoulder_deflection_mm=0.119, top_patch_change_mm=0.000, lower_belly_opening_mm=0.033, bottom_sag_mm=0.000, bag_frame_local_deformation_mm=3.170
- rigid body motion: world_frame_bag_translation_mm=2.903, world_frame_bag_rotation_deg=6.357
- fold_exposed_fraction_before_after: `0.700->0.710`
- per-test separation:
  - shoulder_poke: metric_mm=0.119, world_translation_mm=1.129, world_rotation_deg=0.016, local_deformation_mm=2.072
  - top_preload: metric_mm=0.000, world_translation_mm=1.199, world_rotation_deg=0.014, local_deformation_mm=1.773
  - lateral_squeeze: metric_mm=0.282, world_translation_mm=1.106, world_rotation_deg=0.014, local_deformation_mm=1.896
  - scoop_insertion: metric_mm=0.033, world_translation_mm=2.903, world_rotation_deg=6.357, local_deformation_mm=3.170
  - support_release_sag: metric_mm=0.000, world_translation_mm=1.101, world_rotation_deg=0.011, local_deformation_mm=1.302
  - fold_brushing: metric_mm=0.161, world_translation_mm=1.138, world_rotation_deg=0.472, local_deformation_mm=1.052

## top_fold_severe
- rigid_like_flag: `False`
- summary: shoulder_deflection_mm=0.140, top_patch_change_mm=0.000, lower_belly_opening_mm=0.033, bottom_sag_mm=0.000, bag_frame_local_deformation_mm=3.140
- rigid body motion: world_frame_bag_translation_mm=3.629, world_frame_bag_rotation_deg=6.361
- fold_exposed_fraction_before_after: `0.380->0.385`
- per-test separation:
  - shoulder_poke: metric_mm=0.140, world_translation_mm=1.120, world_rotation_deg=0.010, local_deformation_mm=2.082
  - top_preload: metric_mm=0.000, world_translation_mm=1.190, world_rotation_deg=0.009, local_deformation_mm=1.780
  - lateral_squeeze: metric_mm=0.281, world_translation_mm=1.097, world_rotation_deg=0.010, local_deformation_mm=1.904
  - scoop_insertion: metric_mm=0.033, world_translation_mm=3.629, world_rotation_deg=6.361, local_deformation_mm=3.140
  - support_release_sag: metric_mm=0.000, world_translation_mm=1.092, world_rotation_deg=0.010, local_deformation_mm=1.305
  - fold_brushing: metric_mm=0.083, world_translation_mm=1.115, world_rotation_deg=0.430, local_deformation_mm=1.003

## eccentric_fill
- rigid_like_flag: `False`
- summary: shoulder_deflection_mm=0.298, top_patch_change_mm=0.001, lower_belly_opening_mm=0.027, bottom_sag_mm=0.779, bag_frame_local_deformation_mm=4.382
- rigid body motion: world_frame_bag_translation_mm=20.940, world_frame_bag_rotation_deg=2.633
- fold_exposed_fraction_before_after: `1.000->1.000`
- per-test separation:
  - shoulder_poke: metric_mm=0.298, world_translation_mm=1.926, world_rotation_deg=1.266, local_deformation_mm=1.669
  - top_preload: metric_mm=0.001, world_translation_mm=1.860, world_rotation_deg=1.187, local_deformation_mm=1.448
  - lateral_squeeze: metric_mm=0.067, world_translation_mm=2.037, world_rotation_deg=1.226, local_deformation_mm=1.535
  - scoop_insertion: metric_mm=0.027, world_translation_mm=20.940, world_rotation_deg=2.633, local_deformation_mm=4.382
  - support_release_sag: metric_mm=0.779, world_translation_mm=1.463, world_rotation_deg=0.911, local_deformation_mm=1.103
  - fold_brushing: metric_mm=0.000, world_translation_mm=0.000, world_rotation_deg=0.000, local_deformation_mm=0.000

## jammed_between_neighbors
- rigid_like_flag: `False`
- summary: shoulder_deflection_mm=0.108, top_patch_change_mm=0.001, lower_belly_opening_mm=0.033, bottom_sag_mm=0.000, bag_frame_local_deformation_mm=2.221
- rigid body motion: world_frame_bag_translation_mm=0.916, world_frame_bag_rotation_deg=0.342
- fold_exposed_fraction_before_after: `1.000->1.000`
- per-test separation:
  - shoulder_poke: metric_mm=0.108, world_translation_mm=0.733, world_rotation_deg=0.229, local_deformation_mm=0.736
  - top_preload: metric_mm=0.001, world_translation_mm=0.644, world_rotation_deg=0.211, local_deformation_mm=0.650
  - lateral_squeeze: metric_mm=0.088, world_translation_mm=0.715, world_rotation_deg=0.219, local_deformation_mm=0.687
  - scoop_insertion: metric_mm=0.033, world_translation_mm=0.916, world_rotation_deg=0.342, local_deformation_mm=2.221
  - support_release_sag: metric_mm=0.000, world_translation_mm=0.592, world_rotation_deg=0.178, local_deformation_mm=0.518
  - fold_brushing: metric_mm=0.000, world_translation_mm=0.000, world_rotation_deg=0.000, local_deformation_mm=0.000

## post_separation_sag
- rigid_like_flag: `False`
- summary: shoulder_deflection_mm=0.124, top_patch_change_mm=0.002, lower_belly_opening_mm=0.036, bottom_sag_mm=12.959, bag_frame_local_deformation_mm=16.783
- rigid body motion: world_frame_bag_translation_mm=22.474, world_frame_bag_rotation_deg=6.437
- fold_exposed_fraction_before_after: `1.000->1.000`
- per-test separation:
  - shoulder_poke: metric_mm=0.124, world_translation_mm=0.827, world_rotation_deg=0.167, local_deformation_mm=2.075
  - top_preload: metric_mm=0.002, world_translation_mm=1.354, world_rotation_deg=0.261, local_deformation_mm=1.773
  - lateral_squeeze: metric_mm=0.281, world_translation_mm=0.682, world_rotation_deg=0.270, local_deformation_mm=1.898
  - scoop_insertion: metric_mm=0.036, world_translation_mm=2.991, world_rotation_deg=0.833, local_deformation_mm=2.214
  - support_release_sag: metric_mm=12.959, world_translation_mm=22.474, world_rotation_deg=6.437, local_deformation_mm=16.783
  - fold_brushing: metric_mm=0.000, world_translation_mm=0.000, world_rotation_deg=0.000, local_deformation_mm=0.000

## Conclusion
선택한 scenario는 world-frame rigid motion만이 아니라 bag-frame local deformation도 측정되어 force-responsive surrogate로 볼 수 있습니다.