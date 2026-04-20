# 01_low_fill

## Dual UR5 GUI xyz control

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/01_low_fill/run_dual_ur5_gui.py
```

- Joint target: 양쪽 UR5 관절 목표각을 degree 단위로 조절합니다.
- End effector xyz: 왼쪽은 `left_gripper_pinch`, 오른쪽은 `right_scoop_tip_site`의 world `x, y, z` 목표를 m 단위로 입력하고 `Apply xyz IK`로 이동합니다.
- EE +/- button: `xyz step [m]`만큼 손끝 위치 목표를 조금씩 이동합니다.
- 2F gripper: `pad gap [mm]` 기준으로 조작하며, `0 mm`이면 두 finger pad가 닫히는 상태입니다.
- Pose JSON: joint target degree, EE target xyz, gripper gap을 저장/불러옵니다.

## 2F Grasp Helper

얇은 flex shell 자체는 접힌 부분을 바로 집기 어렵기 때문에, 자루 상단 둘레에 `bag_graspable_band`라는 물리적 seam/grip band를 추가했습니다. 이는 특정 한 점만 잡으라는 뜻이 아니라, 실제 마대자루의 접힌 seam, 천 두께, 잡히는 주름을 둘레 방향으로 단순화한 task-driven grasp layer입니다.

GUI 사용 순서:

- `Move 2F to nearest grasp`: 현재 2F 위치에서 가장 가까운 `bag_grasp_site_XX` 근처로 이동합니다.
- `Close gripper`: gripper를 닫습니다.
- 이후 왼쪽 end-effector의 `z` 값을 올리면 micro-lift처럼 자루가 같이 따라오는지 확인할 수 있습니다.

## Dual UR5 keyboard joint control

Dual UR5 viewer에서는 MuJoCo 오른쪽 control bar 슬라이더와 함께 키보드 미세 조정을 사용할 수 있습니다.

별도 GUI 창으로 degree 단위 관절 조작, 자루 위치 표시, pose JSON 저장/불러오기를 하려면 아래를 실행합니다.

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/01_low_fill/run_dual_ur5_gui.py
```

GUI에서는 각 UR5 관절 목표각을 degree로 조절하고, `bag_frame` 및 shell center 위치를 m 단위로 확인할 수 있습니다. 저장한 pose는 기본적으로 `demos/01_low_fill/poses/*.json`에 둘 수 있습니다.

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/01_low_fill/run_dual_ur5_gui.py --joint-step-deg 0.5
```

기존 MuJoCo viewer만 사용할 수도 있습니다.

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/01_low_fill/run_dual_ur5_low_fill.py
```

- `L` / `R`: 왼쪽 2F UR5 / 오른쪽 scoop UR5 선택
- `1` ... `6`: 현재 arm의 관절 선택
- `←` / `→`: 이전/다음 관절 선택
- `↑` / `↓`: 선택 관절 목표각을 작은 단위로 증가/감소
- `O` / `C`: 왼쪽 2F gripper를 조금 열기/닫기
- `H`: home pose reset

한 번 누를 때 이동량은 기본 `2 deg`이고, 필요하면 아래처럼 바꿀 수 있습니다.

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/01_low_fill/run_dual_ur5_low_fill.py --joint-step-deg 0.5
```

이 데모는 `underfilled sack surrogate` 하나만 보여주는 독립 MuJoCo 예제입니다.  
목표는 `lower third는 비교적 차 있고`, `upper half는 비어 있으며`, `upper shell이 slack/collapsed`로 보이는 현상을 빠르고 안정적으로 확인하는 것입니다.

현재 기본 설정은 `상단 pin 없음`, `bag_frame freejoint 있음`입니다.  
즉 자루 윗부분을 고정한 데모가 아니라, 바닥 위에 놓인 자유 자루에 더 가깝게 구성했습니다.

## 구성

- `low_fill_builder.py`
  - standalone MJCF를 생성합니다.
  - `direct 2D flexcomp shell` 하나를 사용합니다.
  - Version A는 내부 자유 body 없이 shell 형상만으로 underfilled phenotype을 만듭니다.
  - Version B는 바닥 중심 근처에 `단일 ellipsoid ballast` 하나를 추가합니다.
- `run_demo.py`
  - offscreen 렌더를 실행하고 `initial.png`, `settled.png`, `disturbed.png`, `frames/*.png`, `sequence.mp4`를 저장합니다.
  - `--viewer`로 자유 카메라 viewer를 열 수 있습니다.
- `validate_low_fill.py`
  - shell body를 `bag_shell_` prefix로 동적으로 수집합니다.
  - `upper_span_x`, `lower_span_x`, `bag_height`, `escaped_internal_bodies`, `pass_fail`를 계산합니다.

## 노출된 파라미터

`low_fill_builder.py` 상단에 아래 값을 모아 두었습니다.

- `TIMESTEP`
- `SOLVER`
- `CONE_TYPE`
- `IMPRATIO`
- `SHELL_RADIUS`
- `SHELL_THICKNESS`
- `SHELL_DAMPING`
- `SELF_COLLISION_MODE`
- `VERTCOLLIDE`

참고:
- 현재 환경의 MuJoCo `3.1.6` direct `flexcomp` 경로는 `thickness`, `vertcollide`를 XML 속성으로 직접 받지 않습니다.
- 그래서 두 값은 문서화된 조정 파라미터로 유지하고, 실제 안정성은 `radius`, `contact`, `edge damping`으로 맞췄습니다.

## 실행 명령

Version A, offscreen 렌더:

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/01_low_fill/run_demo.py
```

Version B, offscreen 렌더:

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/01_low_fill/run_demo.py --with-ballast
```

Version A, 자유 카메라 viewer:

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/01_low_fill/run_demo.py --viewer
```

Version B, 자유 카메라 viewer:

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/01_low_fill/run_demo.py --viewer --with-ballast
```

Version A 검증:

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/01_low_fill/validate_low_fill.py
```

Version B 검증:

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/01_low_fill/validate_low_fill.py --with-ballast
```

## 저장 경로

- Version A: `demos/01_low_fill/out/version_a_shell_only/`
- Version B: `demos/01_low_fill/out/version_b_ballast1/`

각 폴더에는 아래 파일이 저장됩니다.

- `initial.png`
- `settled.png`
- `disturbed.png`
- `frames/frame_000.png` ... `frames/frame_059.png`
- `sequence.mp4`

## 검증 기준

- 3초 동안 NaN / explosion 없이 실행
- settle 후 `upper_span_x <= 0.85 * lower_span_x`
- `bag_height`가 충분히 유지됨
- Version B에서는 settle 및 lateral impulse 동안 `escaped_internal_bodies == 0`

## Version A / Version B 차이

- `run_demo.py --viewer`
  - Version A입니다.
  - 내부 자유 물체 없이 2D flex shell 형상만으로 underfilled pouch를 보여줍니다.
  - 하단 충전감보다는 "빈 shell이 접히는 저충진 외피"를 확인하는 용도입니다.
- `run_demo.py --viewer --with-ballast`
  - Version B입니다.
  - 하단 중심 근처에 단일 ellipsoid ballast 1개를 넣습니다.
  - 쌀포대처럼 lower third가 더 차 있고 upper half가 비어 보이는 장면은 이 모드가 더 적합합니다.

현재 기본 형상은 원형 cone ring이 아니라, x 방향으로 긴 타원형 rounded pouch 단면을 사용합니다. 상단은 고정하지 않고 `bag_frame`도 freejoint로 둡니다.

## Dual UR5 + Low Fill Sack

기존 환경처럼 왼쪽 UR5에는 2F gripper, 오른쪽 UR5에는 scoop를 붙인 장면도 제공합니다.  
이 장면은 아직 baseline이나 자동 파지 정책이 아니라, `저충진 flex sack + dual UR5 end-effector 배치 확인용`입니다.

저충진 ballast 포함 기본 viewer:

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/01_low_fill/run_dual_ur5_low_fill.py
```

ballast 없이 shell-only viewer:

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/01_low_fill/run_dual_ur5_low_fill.py --no-ballast
```

headless 로드 검증:

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/01_low_fill/run_dual_ur5_low_fill.py --headless --seconds 2.0
```

생성된 XML은 `demos/01_low_fill/generated/dual_ur5_low_fill.xml`에 저장됩니다.
