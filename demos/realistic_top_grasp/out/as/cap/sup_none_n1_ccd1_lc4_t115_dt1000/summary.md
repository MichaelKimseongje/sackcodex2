# Auto Search And Probe Graspability Summary

This report is generated from auto search + probing.
`contact_only_eval` keeps latch disabled. `qualification_gated_capture` enables a task-driven soft latch only after capture qualification; `qualification_gated_latch_eval` is kept as a backward-compatible alias.

- total_trials: 4
- pass_count: 0
- fail_count: 4
- pass_rate: 0.000
- drop_count: 4
- no_graspable_patch_found_count: 2

| mode | scenario | content_case | requested_label | actual_label | rank | trapped | L/R | escape | slip_mm | follow | latch | no_patch | pass |
|---|---|---|---|---|---:|---:|---|---|---:|---:|---|---|---|
| qualification_gated_capture | exposed_seam | support_sag | auto | seam | 3 | 7 | True/True | False | 18.6 | -0.03 | False | False | False |
| qualification_gated_capture | exposed_seam | support_sag | seam | seam | 3 | 7 | True/True | False | 18.6 | -0.03 | False | False | False |
| qualification_gated_capture | exposed_seam | support_sag | fold | none | -1 | 0 | False/False | True | 999.0 | 0.00 | False | True | False |
| qualification_gated_capture | exposed_seam | support_sag | plain_top | plain_top | 0 | 7 | False/True | False | 31.1 | 0.04 | False | True | False |
