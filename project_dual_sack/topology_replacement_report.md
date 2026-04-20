# Topology Replacement Report

## Replacement Goal

현재 모델은 `right_slice_shell_contract.png`를 기준으로 한 5-slice cross-section-inspired quasi-rigid sack surrogate입니다. 이번 보완의 목적은 runtime body count 통과에 더해, 길이 방향 양 끝이 열린 단면처럼 보이지 않는 sealed pillow-sack appearance를 명시적으로 만족시키는 것입니다.

## Current Runtime Topology vs Target

| Item | Current target after replacement |
|---|---|
| 11-column / 100+ strip topology | not used |
| central rigid core | not used |
| visible articulated outer shell | 5 slices x 6 bodies = 30 bodies |
| hidden inner load shell | 5 slices x 3 bodies = 15 bodies |
| ballast | 4 limited-slide ballast bodies |
| seam windows | 5 named top seam candidate sites |
| longitudinal end closure | `left_end_cap_visual`, `right_end_cap_visual` |

## End-Cap Closure

The longitudinal end caps are visual-only by default. They are excluded from the 30-body visible outer shell count, but they are rendered in the outer-shell view so the sack does not look open-ended.

```text
left_end_cap_visual
right_end_cap_visual
```

Optional physics-bearing end caps are intentionally not enabled by default:

```text
left_end_cap_physics
right_end_cap_physics
```

## Runtime Proof

Use:

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' project_dual_sack/inspect_topology_runtime.py --scenario underfilled
```

Expected:

```text
visible outer shell = 30
hidden inner shell = 15
ballast = 4
end-cap visuals = 2
legacy body names = none
central rigid core = none
```
