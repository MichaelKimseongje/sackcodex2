# MuJoCo Sack Pile Handling Prototype

MuJoCo에서 비정형 sack pile handling 연구를 보기 위한 프로토타입 환경입니다. 강화학습은 제외했고, 먼저 `비정형 pile 생성`, `2F gripper + scoop baseline`, `평가/시각화/로그`를 돌리는 데 초점을 맞췄습니다.

## 이번 버전의 핵심 변화

- 자루를 단순 box proxy처럼 보이게 두지 않고, `기존 sack OBJ mesh`를 시각 외형으로 직접 사용합니다.
- 같은 family 안에서도 episode마다 아래 값이 달라집니다.
  - mesh 선택
  - fill ratio
  - top collapse
  - side bulge
  - flattening
  - 초기 roll / pitch / yaw
- scene 생성 시 일부 자루는 약간 공중에서 시작하고 기울어진 채 배치되어, reset 시 MuJoCo 안에서 먼저 `settle`되며 pile을 이룹니다.

## 포함 산출물

- `mujoco_sack_pile/scene_generator.py`
- `mujoco_sack_pile/environment.py`
- `mujoco_sack_pile/baselines/heuristics.py`
- `mujoco_sack_pile/evaluation.py`
- `mujoco_sack_pile/visualization.py`
- `generate_scene.py`
- `run_baseline.py`
- `evaluate_logs.py`

## Sack family

- `regular_well_filled`
- `low_fill_top_collapsed`
- `side_bulged_unstable`

## 자루 모델 구조

- 시각 외형:
  - `object/sack*.obj` mesh 사용
- 내부 동역학:
  - 중심 `ellipsoid/capsule core`
  - 하단 support skin
  - 상단 grip skin
  - 좌/우/전/후 compliant skin
  - local compliant patch
- 완전한 cloth는 아니지만, `같은 자루가 매번 다른 초깃값과 적재 상태를 가지는 비정형 sack pile`을 보는 목적에는 이전 버전보다 훨씬 가깝게 구성했습니다.

## 실행 방법

MuJoCo Python 패키지가 설치되어 있어야 합니다.

```bash
python generate_scene.py --seed 7 --episode-id preview_scene
python run_baseline.py --seed 7 --preview-only
python run_baseline.py --seed 7 --preview-only --manual-control
python run_baseline.py --baseline fixed_2f_scoop_pose --seed 11
python run_baseline.py --baseline fixed_2f_scoop_pose --seed 11 --fixed-camera
python run_baseline.py --baseline scoop_first_gap_creation_regrasp --seed 21 --headless
python evaluate_logs.py
```

## 수동 조작 키

`--manual-control`로 실행하면 viewer 안에서 다음 키를 쓸 수 있습니다.

- `1` / `2`: gripper / scoop 선택
- `W S`: X축 이동
- `A D`: Y축 이동
- `R F`: Z축 이동
- `I K`: pitch 회전
- `J L`: yaw 회전
- `U O`: roll 회전
- `[` `]`: gripper 열기 / 닫기
- `H`: 도움말 다시 출력

## 로그와 생성 파일

- 생성 XML: `mujoco_sack_pile/generated/*.xml`
- episode 로그: `mujoco_sack_pile/logs/*.json`
- 누적 로그: `mujoco_sack_pile/logs/episode_history.jsonl`

로그에는 다음이 포함됩니다.

- 성공/실패 여부
- support-state score 및 세부 metric
- 실패 원인 태그
- target sack variant
- mesh 파일명
- fill ratio / collapse / bulge / flattening
- 각 sack의 초기 위치와 Euler 각도

## 현재 한계

- 완전한 cloth / FEM sack은 아닙니다.
- baseline은 아직 강하지 않고, scene realism을 먼저 올린 상태입니다.
- RL을 붙일 때는 `SackPileEnv`에 observation/action 래퍼를 추가하면 됩니다.
