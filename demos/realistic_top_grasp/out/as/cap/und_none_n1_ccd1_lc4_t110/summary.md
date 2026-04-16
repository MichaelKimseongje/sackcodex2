# Auto Search And Probe Graspability Summary

This report is generated from auto search + probing.
`contact_only_eval` keeps latch disabled. `qualification_gated_capture` enables a task-driven soft latch only after capture qualification; `qualification_gated_latch_eval` is kept as a backward-compatible alias.

- total_trials: 1
- pass_count: 0
- fail_count: 1
- pass_rate: 0.000
- drop_count: 1
- no_graspable_patch_found_count: 0

| mode | scenario | content_case | requested_label | actual_label | rank | trapped | L/R | escape | slip_mm | follow | latch | no_patch | pass |
|---|---|---|---|---|---:|---:|---|---|---:|---:|---|---|---|
| qualification_gated_capture | exposed_seam | underfilled | auto | seam | 1 | 3 | True/True | False | 20.3 | 0.08 | True | False | False |
