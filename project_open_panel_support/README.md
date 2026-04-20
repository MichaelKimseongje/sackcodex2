# Open Panel Support-State Prototype

이 폴더는 최종 sealed sack model이 아니라, **support-state output 계산 가능성**을 확인하기 위한 MuJoCo prototype입니다.

목표는 자루 외형을 사실적으로 재현하는 것이 아니라, 열린 패널 구조에서 다음 출력이 안정적으로 계산되는지 보는 것입니다.

- `sag_index`
- `effective_com_offset`
- `scoop_load_transfer`
- `peel_ratio`
- `support_margin`
- `insertion_depth`
- `scoop_contact_force`

## 모델 구성

- `bag_frame`: freejoint를 가진 기준 body입니다.
- `top_panel`, `left_side_panel`, `right_side_panel`, `left_bottom_panel`, `right_bottom_panel`: 5개 rigid panel입니다.
- 각 panel은 hinge joint, damping, stiffness, range limit을 가집니다.
- `panel_0`은 hidden mass가 아니라 첫 번째 rigid panel입니다.
- 마지막 panel tip과 첫 root는 `close_loop_panel_*_to_root` spatial tendon으로 5-1 soft close-loop 연결을 만듭니다.
- 현재 Yolov9의 MuJoCo 3.1.6에서는 site-to-site equality 문법을 쓸 수 없어, 닫힌 단면 확인용으로 tendon 기반 soft 연결을 사용합니다.
- `hidden_mass_00`, `hidden_mass_01`, ...: DEM 입자가 아니라 coarse clump입니다.
- `hidden_mass_count`로 1~12개까지 생성할 수 있고, 기본값은 6개입니다.
- `hidden_mass_size_scale`로 clump 크기를 줄이거나 키울 수 있습니다. 기본값은 작은 clump용 `0.58`입니다.
- 각 hidden mass는 `bag_frame` 기준 `x/y/z` slide joint 3개를 가지며, range limit으로 chamber 내부에 머무릅니다.
- `hidden_mass free rotation(ball joint)`를 켜면 각 clump가 ball joint로 회전도 할 수 있습니다.
- `hidden_mass_range_x/y/z`는 clump가 움직일 수 있는 chamber 범위입니다. 값이 커질수록 더 자유롭게 움직이지만 panel 밖으로 삐져나갈 가능성도 커집니다.
- `hidden_mass_slide_damping`과 `hidden_mass_slide_armature`를 낮추면 내부 질량이 더 빠르고 자유롭게 움직입니다.
- `scoop_body`: slide joint로 삽입되며 insertion depth와 contact force를 기록합니다.
- `gripper_palm`, `left_finger`, `right_finger`: 간단한 2F gripper입니다.
- `gripper_lift`: close 후 micro-lift/tug-test를 만들기 위한 z축 slide joint입니다.
- `guarded_grasp_connect`: 기본 비활성입니다. 조건을 만족할 때만 Python에서 켭니다.

## Guarded Grasp 의미

이 prototype은 “jaw 안에 들어오면 무조건 attach”를 하지 않습니다.

아래 조건이 맞을 때만 `guarded_grasp_connect`를 활성화합니다.

- 좌/우 finger가 target panel과 모두 접촉
- finger gap 안에 panel patch가 존재
- bilateral contact balance가 충분함
- tangential slip proxy가 작음
- peel ratio가 과도하지 않음
- 짧은 contact persistence가 있음

따라서 이것은 pure friction proof가 아니라, task-output feasibility 확인용 guarded grasp abstraction입니다.

## 실행 방법

Yolov9 conda 환경의 Python을 쓰는 경우:

패널/힌지 형상을 직접 만들고 MuJoCo viewer를 재시작하는 GUI:

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' project_open_panel_support\panel_designer_gui.py
```

GUI에서 가능한 일:

- 설정 탭은 `Geometry`, `Hinge`, `Mass / Loop`, `Run`으로 나뉩니다.
- Log 영역은 아래쪽에 작게 고정되어 있고, 패널 초기각도 표는 오른쪽에 표시됩니다.
- 패널 개수 4~12개 선택. 기본값은 닫힌 단면 확인을 위해 6개입니다.
- 패널 길이, 폭, 두께, 질량 수정
- GUI 시작 기본값은 `panel_designer_builder.py`의 `PanelDesignerConfig`에서 직접 읽습니다.
- hinge axis, range, damping, stiffness, actuator gain 수정
- `Passive free hinge mode` 체크 시 panel actuator를 생성하지 않고 hinge stiffness를 0으로 낮춥니다. 이때 패널은 hinge min/max 범위 안에서 gravity, hidden mass, close-loop, contact에 따라 자유롭게 움직입니다.
- `Passive free hinge mode` 체크 해제 시 기존처럼 `panel_*_angle_deg_act` actuator가 생성되어 MuJoCo control bar에서 degree 단위로 수동 조작합니다.
- Passive mode에서는 `actuator_kp`가 적용되지 않습니다. 변형 속도를 키우려면 `damping`과 `armature`를 낮추거나 `Fast passive response preset`을 사용하세요.
- 내부 질량 움직임은 `hidden_mass_count`, `hidden_mass_size_scale`, `hidden_mass_slide_damping`, `hidden_mass_slide_armature`, `hidden_mass_range_x/y/z`로 조정합니다.
- 첫 번째 panel은 root hinge에서 바로 시작하고, 마지막 panel은 생성 시점부터 첫 root 근처로 돌아오는 6-panel closed template을 기본으로 씁니다.
- 마지막 panel과 첫 root의 close-loop 연결 on/off. 이 연결은 이미 닫힌 단면을 안정화하는 용도이며, 멀리 떨어진 panel을 나중에 끌어당기는 용도가 아닙니다.
- `close_loop_time` 수정. 값이 작을수록 마지막 panel과 첫 root 사이의 안정화가 강합니다. 기본값은 `0.005`입니다.
- 기본 hinge range는 `-90~90 deg`, 기본 actuator gain은 `1.0`입니다. panel을 강제로 고정하기보다 조인트가 실제로 움직이는지 확인하기 위한 설정입니다.
- 각 panel의 초기 각도 수정
- XML 생성
- MuJoCo viewer 실행
- `Open Viewer`를 눌러도 현재 GUI 값으로 XML을 먼저 재생성한 뒤 viewer를 엽니다.
- 수정 후 XML 재생성 + viewer 재시작
- viewer control bar에서 `panel_0_angle_deg_act`, `panel_1_angle_deg_act` 같은 hinge actuator를 degree 단위로 직접 조작
- MuJoCo 내부 hinge 좌표는 radian이지만, 이 actuator는 `general` actuator로 만들어 viewer 입력값 `degree`를 내부 목표각 `radian`으로 변환합니다.

### MjSpec backend

MuJoCo 3.2+에서는 `MjSpec` API로 body, geom, joint, actuator를 프로그래밍 방식으로 추가한 뒤 `compile()` / `to_xml()` 할 수 있습니다.

현재 Yolov9 환경은 확인 결과 `mujoco==3.1.6`이라 `mujoco.MjSpec`가 없습니다. 그래서 GUI는 자동으로 다음 방식으로 동작합니다.

- `MjSpec` 사용 가능: `panel_designer_mjspec_backend.py`가 `MjSpec`으로 body/geom/joint/actuator를 생성
- `MjSpec` 없음: 기존 XML builder fallback으로 동일한 XML 생성

backend 상태만 확인하려면:

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' project_open_panel_support\panel_designer_mjspec_backend.py
```

MuJoCo를 3.2 이상으로 업그레이드하면 같은 GUI가 MjSpec backend를 우선 사용합니다.

자루 panel joint 움직임만 확인하려면 아래를 먼저 실행하세요. 이 장면에는 gripper와 scoop이 없습니다.

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' project_open_panel_support\run_joint_demo.py
```

자동 흔들림 없이 MuJoCo control bar에서 hinge actuator를 직접 조작하려면:

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' project_open_panel_support\run_joint_demo.py --no-script
```

joint 각도가 실제로 변하는지 CSV로 검증하려면:

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' project_open_panel_support\validate_joint_motion.py
```

2F/scoop까지 포함한 support-state prototype을 보려면:

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' project_open_panel_support\run_viewer.py
```

자동 동작 없이 MuJoCo control bar로 직접 조작하려면:

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' project_open_panel_support\run_viewer.py --no-script
```

이 모드에서는 control bar에서 `left_finger_close_act`, `right_finger_close_act`, `gripper_lift_act`, `scoop_insert_act`를 직접 조작할 수 있습니다.

평가 로그를 저장하려면:

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' project_open_panel_support\run_eval.py
```

검증을 실행하려면:

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' project_open_panel_support\validate_open_panel.py
```

## 출력 파일

생성 XML:

```text
project_open_panel_support/generated/open_panel_support_scene.xml
```

평가 결과:

```text
project_open_panel_support/out/eval_metrics.csv
project_open_panel_support/out/eval_summary.json
project_open_panel_support/out/runtime_inventory.json
project_open_panel_support/out/validation_summary.md
```

## 입력 CAD/STL 파일

사용자가 제공한 입력 파일은 아래 경로에 있습니다.

```text
sack/1/plate.stl
sack/1/topplate.stl
```

이번 prototype에서는 수치 안정성을 위해 physics panel을 box geom으로 만들었습니다. STL은 이후 visual-only panel shape 또는 convex mesh panel 비교 실험에 붙일 수 있습니다.

## 이번 prototype에서 하지 않는 것

- full soft body
- membrane shell
- waterbomb closed shell
- grain DEM
- sealed sack appearance
- 실제 직물/마찰/입자 재현

이 prototype은 “자루처럼 보이는지”가 아니라, 2F와 scoop 상호작용에서 support-state 관련 지표를 계산하고 로그로 남길 수 있는지 확인하는 단계입니다.
