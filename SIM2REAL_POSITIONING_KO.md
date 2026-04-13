# Sim2Real Positioning

## 결론

현재 benchmark만으로는 `정확한 자루 재질 시뮬레이터`라고 설득하기 어렵습니다.

하지만 연구 질문을 아래처럼 고정하면 충분히 설득 가능한 방향이 있습니다.

`support-state formation under shape and pile uncertainty`

즉, 이 환경은 `재질 정밀 복원`이 아니라 `불확실한 형상과 적재 상태에서 support state를 만들 수 있는가`를 묻는 task-driven benchmark로 사용하는 것이 맞습니다.

## 지금 모델로 주장할 수 있는 것

- 서로 다른 sack family와 pile difficulty에서 baseline이 얼마나 자주 성공하는가
- 어떤 초기 형상과 적재 상태에서 insertion, slip, tilt, drop이 자주 발생하는가
- exact grasp point가 불명확한 상황에서 support-state formation이 가능한가
- heuristic 또는 residual correction이 failure mode를 얼마나 줄이는가

## 지금 모델로 주장하면 안 되는 것

- 실제 포장지 재질과 동일한 deformation이 재현되었다
- 자루 간 얽힘이 실제와 같은 수준으로 물리적으로 복원되었다
- sim contact가 real fabric interlocking을 직접 대체한다
- sim trajectory를 거의 그대로 실기로 옮길 수 있다

## Sim2Real을 설득하려면

핵심은 `exact physics matching`이 아니라 `failure mode alignment`입니다.

즉 아래가 맞아야 합니다.

- sim에서 어려운 case가 real에서도 어렵다
- sim에서 잘 되는 primitive 조합이 real에서도 상대적으로 낫다
- sim failure tag와 real failure tag가 대체로 대응된다

## 권장 주장 구조

1. benchmark 정의

- 3개 family
- 4개 pile difficulty
- support-state score
- baseline / recovery primitive 비교

2. sim 결과

- family별 success rate
- difficulty별 failure tag 분포
- support-state score와 실제 성공 여부의 상관

3. real 소규모 검증

- 각 family와 difficulty에서 소수의 대표 case를 수동으로 구성
- 동일한 primitive sequence를 적용
- sim과 real의 성공 / 실패 경향 비교

4. 해석

- exact material matching은 아니지만
- shape / pile uncertainty 하의 task difficulty ordering과 failure mode는 일정 부분 보존된다고 주장

## 가장 안전한 연구 메시지

추천 메시지는 아래와 같습니다.

`We do not model sack material faithfully. Instead, we use a task-driven benchmark that factorizes shape and pile uncertainty, and we evaluate whether support-state formation policies remain effective under those uncertainties.`

한국어로는 아래가 자연스럽습니다.

`본 환경은 자루 재질을 정밀 복원하는 시뮬레이터가 아니라, 형상 및 적재 불확실성을 분해해 표현한 task-driven benchmark이며, 이러한 불확실성 하에서 support-state formation 전략이 얼마나 유지되는지를 평가하기 위한 환경이다.`

## 실험 설계 권장안

가장 설득력 있는 조합은 아래입니다.

- main benchmark: 현재 proxy-based MuJoCo 환경
- auxiliary qualitative reference: experimental soft reference
- small real study: family × difficulty별 대표 case

이때 논문에서 중심은 항상 main benchmark와 real 비교에 둡니다.

experimental soft reference는 아래 용도로만 씁니다.

- direct mesh soft body가 쉽게 불안정해진다는 점을 보여주는 보조 근거
- 왜 task-driven proxy benchmark를 채택했는지 설명하는 부록 자료

## 다음 단계

- baseline 3종을 family × difficulty별로 충분히 반복 실행
- recovery primitive를 추가해 failure mode별 개선 효과 비교
- real 데이터에서 동일 taxonomy로 장면을 라벨링
- success / failure tag 대응표를 만들기

이 네 가지가 갖춰지면, 지금 모델도 `task-level sim2real study`로는 충분히 설득력을 가질 수 있습니다.
