# Sack Support-State Benchmark

이 프로젝트는 MuJoCo 기반 `task-driven benchmark`입니다.

고정 연구 질문:

`support-state formation under shape and pile uncertainty`

즉, 이 구현은 다음을 보기 위한 환경입니다.

- 형상 불확실성이 있는 sack family 3종
- 적재 불확실성이 있는 pile difficulty 4종
- 2F gripper + scoop baseline 3종
- `transport-ready support state` 형성 가능 여부와 failure mode

## 주장 범위

이 구현이 주장하는 것:

- shape / pile uncertainty가 있는 장면을 taxonomy로 나눠 benchmark case를 만든다
- 동일 family 안에서도 초기 형상과 적재 상태가 달라지는 target case를 만든다
- heuristic baseline의 success rate와 failure tags를 비교한다
- support-state 형성 가능성과 실패 양상을 task 수준에서 비교한다

이 구현이 주장하지 않는 것:

- 정확한 sack material simulator
- cloth / FEM 수준의 재질 복원
- 실제 포장지의 기공, 섬유 마찰, 국소 주름을 정밀하게 재현하는 디지털 트윈
- 자루 얽힘 자체를 high-fidelity physics로 복원했다는 주장

## Main Workflow

본선 benchmark 실행 경로:

- `generate_scene.py`
- `run_baseline.py`
- `evaluate_logs.py`

실행 예시:

```bash
python generate_scene.py --seed 7 --episode-id case_preview
python run_baseline.py --seed 7 --preview-only
python run_baseline.py --baseline fixed_2f_scoop_pose --seed 11
python evaluate_logs.py
```

화면으로 장면을 보려면 [run_baseline.py](/d:/Michael/2025/01.Research/01.Parceldetection/16.Pybullet/bullet3-master%20(1)/bullet3-master/examples/pybullet/examples/DeformableTest/Sackcodex2/run_baseline.py)를 사용합니다.

배치 benchmark를 돌릴 때는 `--headless`를 사용합니다.

## Benchmark Taxonomy

Sack family:

- `regular_well_filled`
- `low_fill_top_collapsed`
- `side_bulged_unstable`

Pile difficulty:

- `top_exposed`
- `side_exposed`
- `partially_buried`
- `leaning_wedged`

## Core Files

- `mujoco_sack_pile/benchmark_definition.py`
  - 연구 질문, benchmark 이름, taxonomy 정의
- `mujoco_sack_pile/scene_generator.py`
  - 3~6개 sack pile benchmark case 생성
- `mujoco_sack_pile/environment.py`
  - settle, viewer, 로그 저장, benchmark episode 실행
- `mujoco_sack_pile/evaluation.py`
  - support-state score와 failure tags 계산
- `mujoco_sack_pile/baselines/heuristics.py`
  - baseline 3종 구현
- `mujoco_sack_pile/visualization.py`
  - contact, insertion depth, score overlay 표시

## Viewer Control

`run_baseline.py --manual-control`에서 아래 키를 사용할 수 있습니다.

- `1` / `2`: gripper / scoop 선택
- `W S`: X축 이동
- `A D`: Y축 이동
- `R F`: Z축 이동
- `I K`: pitch
- `J L`: yaw
- `U O`: roll
- `[` `]`: gripper 열기 / 닫기
- `H`: 도움말 다시 출력

오른쪽 control bar에서도 아래 actuator를 직접 조절할 수 있습니다.

- `gripper_ctrl_*`
- `scoop_ctrl_*`
- `left_finger_act`
- `right_finger_act`

MuJoCo viewer에서는 `double-click` 후 `Ctrl + drag`로 객체 perturb도 가능합니다.

## Logs

- 생성 XML: `mujoco_sack_pile/generated/*.xml`
- episode 로그: `mujoco_sack_pile/logs/*.json`
- 누적 로그: `mujoco_sack_pile/logs/episode_history.jsonl`

로그에는 아래 항목이 포함됩니다.

- benchmark 이름과 연구 질문
- target case taxonomy
- sack family / pile difficulty label
- support-state score
- insertion depth
- micro-lift stability
- failure tags

## Experimental Soft Reference

[compare_soft_sack_prototypes.py](/d:/Michael/2025/01.Research/01.Parceldetection/16.Pybullet/bullet3-master%20(1)/bullet3-master/examples/pybullet/examples/DeformableTest/Sackcodex2/compare_soft_sack_prototypes.py)와 [mujoco_sack_pile/soft_sack_prototypes.py](/d:/Michael/2025/01.Research/01.Parceldetection/16.Pybullet/bullet3-master%20(1)/bullet3-master/examples/pybullet/examples/DeformableTest/Sackcodex2/mujoco_sack_pile/soft_sack_prototypes.py)는 메인 benchmark가 아닙니다.

이 경로는 exploratory reference입니다.

- 목적: `mesh-only`와 `mesh + minimal internal support`가 얼마나 쉽게 발산하거나 처지는지 확인
- 상태: known unstable
- 용도: soft representation의 한계를 확인하는 보조 실험
- 비권장 용도: main benchmark, 성능 비교, sim2real 주장의 근거

예시:

```bash
python compare_soft_sack_prototypes.py
python compare_soft_sack_prototypes.py --mode mesh_only
python compare_soft_sack_prototypes.py --mode mesh_with_payload
```

기본 `compare` 모드는 viewer를 띄우지 않고 headless 비교만 수행합니다.

## Uncertainty Preview

[preview_uncertainty_case.py](/d:/Michael/2025/01.Research/01.Parceldetection/16.Pybullet/bullet3-master%20(1)/bullet3-master/examples/pybullet/examples/DeformableTest/Sackcodex2/preview_uncertainty_case.py)는 개별 uncertainty 현상을 하나씩 구분해서 보여주는 preview입니다.

현재 제공하는 현상:

- `underfilled_slack`
- `top_fold_occluded`
- `eccentric_fill`
- `neighbor_contact_wedge`
- `partial_support_sag`

예시:

```bash
python preview_uncertainty_case.py --phenomenon underfilled_slack
python preview_uncertainty_case.py --phenomenon top_fold_occluded
python preview_uncertainty_case.py --phenomenon eccentric_fill
python preview_uncertainty_case.py --phenomenon neighbor_contact_wedge
python preview_uncertainty_case.py --phenomenon partial_support_sag
```

이 preview는 현재 MuJoCo 3.1.6 환경에서 안정적으로 돌리는 것을 우선한 경로입니다.
즉 3D flex trilinear 주 시뮬레이터가 아니라, 현행 proxy-based benchmark scene 중에서 조건을 만족하는 장면을 골라 보여줍니다.

## Phenomenon Demo

[demo_phenomenon.py](/d:/Michael/2025/01.Research/01.Parceldetection/16.Pybullet/bullet3-master%20(1)/bullet3-master/examples/pybullet/examples/DeformableTest/Sackcodex2/demo_phenomenon.py)는 현상별 자동 자극과 반응을 6~10초 내외의 mp4로 저장하는 스크립트입니다.

구현된 현상:

- `underfilled_slack`
- `top_fold_occluded`
- `eccentric_fill`
- `neighbor_contact_wedge`
- `partial_support_sag`

예시:

```bash
python demo_phenomenon.py --phenomenon underfilled_slack --save out/underfilled.mp4
python demo_phenomenon.py --phenomenon top_fold_occluded --save out/top_fold.mp4
python demo_phenomenon.py --phenomenon eccentric_fill --save out/eccentric.mp4
python demo_phenomenon.py --phenomenon neighbor_contact_wedge --save out/wedge.mp4
python demo_phenomenon.py --phenomenon partial_support_sag --save out/partial_support.mp4
```

각 demo는 아래 4단계를 자동으로 보여줍니다.

- `before`
- `stimulus`
- `response`
- `after`

현재 visible response는 static mesh 위에 더해진 `dynamic deform visual layer`와 proxy joint 변화로 표현됩니다.
즉 mesh가 완전한 soft body처럼 변형되는 것은 아니지만, 자극에 따라 top band, side bulge, bottom support, local outline이 실제로 움직이도록 만들었습니다.

## Integrated Demo

[demo_integrated_case.py](/d:/Michael/2025/01.Research/01.Parceldetection/16.Pybullet/bullet3-master%20(1)/bullet3-master/examples/pybullet/examples/DeformableTest/Sackcodex2/demo_integrated_case.py)는 복수의 uncertainty가 한 장면에 동시에 들어간 통합 데모입니다.

예시:

```bash
python demo_integrated_case.py --save out/integrated_case.mp4
```

통합 demo는 아래 순서의 짧은 시퀀스를 보여줍니다.

- `before`
- `shape_tidy`
- `gap_creation`
- `support_formation`
- `stabilize`

저장 경로 예시:

- `out/underfilled.mp4`
- `out/eccentric.mp4`
- `out/integrated_case.mp4`

## Notes

- 현재 benchmark 본체는 shape / pile uncertainty를 가진 `proxy-based task benchmark`입니다.
- 따라서 설득 포인트는 material fidelity가 아니라 `case taxonomy`, `support-state score`, `failure mode` 비교에 있습니다.
- sim2real은 exact deformation matching보다 `failure mode alignment`와 `task-level robustness` 관점에서 접근하는 것이 적절합니다.

## Related Docs

- [BENCHMARK_AUDIT_KO.md](/d:/Michael/2025/01.Research/01.Parceldetection/16.Pybullet/bullet3-master%20(1)/bullet3-master/examples/pybullet/examples/DeformableTest/Sackcodex2/BENCHMARK_AUDIT_KO.md)
- [RESEARCH_POSITIONING_KO.md](/d:/Michael/2025/01.Research/01.Parceldetection/16.Pybullet/bullet3-master%20(1)/bullet3-master/examples/pybullet/examples/DeformableTest/Sackcodex2/RESEARCH_POSITIONING_KO.md)
- [SIM2REAL_POSITIONING_KO.md](/d:/Michael/2025/01.Research/01.Parceldetection/16.Pybullet/bullet3-master%20(1)/bullet3-master/examples/pybullet/examples/DeformableTest/Sackcodex2/SIM2REAL_POSITIONING_KO.md)

## Bag Base

Shared deformable sack surrogate base files:

- `bag_base.xml`
- `run_viewer.py`
- `validate_base.py`

Commands:

```bash
python run_viewer.py
python validate_base.py
```

Stable names reserved for future scenario scripts:

- body: `bag_frame`
- flex: `bag_shell`
- geom: `floor`

## Low Fill Scenario

Low fill only:

```bash
python scenario_low_fill.py --fill-mode ballast1 --validate
python scenario_low_fill.py --fill-mode clump3 --validate
python run_viewer.py --scenario low_fill --fill-mode ballast1
python run_viewer.py --scenario low_fill --fill-mode clump3
```
