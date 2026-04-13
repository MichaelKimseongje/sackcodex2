# Benchmark Audit

## 고정 연구 질문

`support-state formation under shape and pile uncertainty`

## 현재 구현 적합도 점검

### 1. scene generator

판정: `대체로 적합`

- sack family 3종이 존재함
- 3~6개 pile을 무작위 생성함
- 동일 family 내에서도 fill ratio, collapse, bulge, 초기 자세가 달라짐
- target case에 대해 exposure class와 pile class를 라벨링함

보완점:

- pile class는 아직 규칙 기반 분류이며, 실제 얽힘의 고정밀 판정은 아님

### 2. robot setup

판정: `적합`

- 2F gripper와 scoop가 존재함
- baseline 실행과 수동 조작 모두 가능함
- viewer 오른쪽 control bar에서 actuator 조작 가능함

### 3. baseline

판정: `적합`

- heuristic baseline 3종이 존재함
- 연구 질문상 “feasibility check” 용도로는 충분함

보완점:

- 성능이 아직 약해 taxonomy별 경향 비교를 더 쌓을 필요가 있음

### 4. evaluation

판정: `적합`

- support-state success
- scoop insertion depth
- micro-lift stability
- slip / tilt / drop
- failure tags

를 모두 기록함.

### 5. material simulator처럼 보이는 위험 요소

이전 위험 요소:

- `deformable`
- `skin`
- `compliant`
- `material simulator`

정리 결과:

- 핵심 benchmark 경로에서는 `proxy`, `benchmark case`, `uncertainty label` 중심 용어로 수정함
- qualitative preview는 `preview_reference_case.py`로 분리하고 auxiliary reference로 격하함
- 기존 `preview_deformable_pile.py`는 deprecated wrapper로만 남김

### 6. 남아 있는 해석상 한계

- 내부 proxy와 local shape proxy는 형상 다양성을 위한 근사 표현이다
- 실제 포장지 재질, 섬유 마찰, 기공, 국소 주름을 정확히 재현하지 않는다
- 따라서 본 구현은 `material fidelity benchmark`가 아니라 `task-driven benchmark`로 읽혀야 한다

## 최종 판단

현재 구현은 아래 질문에는 맞습니다.

- `support-state formation under shape and pile uncertainty`

현재 구현은 아래 질문에는 맞지 않습니다.

- `정확한 sack material deformation을 시뮬레이션했는가`
- `실제 포장지의 얽힘과 기공을 고정밀 복원했는가`

## 권장 사용 방식

주요 결과:

- `generate_scene.py`
- `run_baseline.py`
- `evaluate_logs.py`

보조 시각화:

- `preview_reference_case.py`

권장 서술:

- `task-driven benchmark`
- `support-state benchmark`
- `shape/pile uncertainty taxonomy`
- `failure mode comparison`
