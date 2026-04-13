# 01_low_fill

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
