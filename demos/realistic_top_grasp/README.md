# Realistic Top Graspability Demo

이 데모는 정확한 자루 재료 시뮬레이터가 아니라, `support-state formation under shape and pile uncertainty` 연구를 위한 task-driven benchmark입니다. 내부 충전물은 DEM/입자 대신 `central bulk mass 1개 + lateral support mass 2개`의 3-clump surrogate로 표현합니다.

## 모드 구분

- `visual_demo_assisted`: GUI/시연용입니다. MuJoCo flex shell이 jaw가 닫히는 순간 빠져나가거나 발산하는 현상을 줄이기 위해 조건부 assist를 사용합니다. 순수 2F 마찰 파지 성능 증거로 해석하면 안 됩니다.
- `contact_only_eval`: 연구용 pure-contact lower bound입니다. auto search와 probing은 수행하지만 latch/assist를 켜지 않습니다.
- `qualification_gated_capture`: task-driven grasp surrogate 평가 모드입니다. jaw 내부 patch가 충분히 포획되고 qualification을 통과한 경우에만 soft latch force를 켭니다.
- `qualification_gated_latch_eval`: 기존 이름과 호환하기 위한 alias입니다. 새 실험명으로는 `qualification_gated_capture`를 권장합니다.

`seam`, `fold`, `plain_top`은 성공/실패 규칙이 아니라 candidate selector와 분석용 label입니다. 실제 close 시점에 잡힌 patch는 `actual_region_label_at_close`로 따로 기록됩니다.

## Capture Rule

`qualification_gated_capture`는 “jaw 안에 들어오면 무조건 고정”이 아닙니다. 아래 조건이 통과되어야만 latch가 켜집니다.

- `left_contact_present == true`
- `right_contact_present == true`
- `trapped_shell_points >= threshold`
- `bundle_thickness_proxy`가 설정 범위 안에 있음
- `bilateral_contact_balance`가 충분함
- `tangential_slip_proxy`가 threshold 이하
- `load_following_ratio`가 threshold 이상
- `jaw_escape == false`

현재 latch 구현은 MuJoCo equality `weld/connect`가 아니라 `adhesion_force_guarded` 계열의 soft force surrogate입니다. 따라서 latch 성공을 pure 2F friction proof로 해석하지 않습니다.

## Close Stabilization

gripper close 단계는 flex shell이 순간적으로 튀는 구간이라 별도로 안정화합니다. `contact_only_eval`과 `qualification_gated_capture`는 close 중 최근 안정 snapshot을 저장하고, shell 속도 spike, jaw escape, 과도한 penetration, NaN 위험이 감지되면 `last-stable rollback`을 수행합니다. rollback 여부는 `rollback_used`로 기록되고, 해당 상태는 `rollback.png`로 저장됩니다.

GUI의 `Auto hold + 5s check`도 같은 의도로 발산 직전 gap에서 멈춥니다. 다만 GUI는 시연용 `visual_demo_assisted`이므로, 여기서 유지되는 latch는 연구용 성공률에 섞지 않습니다.

GUI 시연에서는 Robotiq 2F-140 proxy gripper를 사용합니다. 실제 CAD/동역학 전체 모델은 아니고, 약 140 mm stroke, 긴 fingertip pad, palm/knuckle 외형을 가진 MuJoCo 안정화용 proxy입니다.

gripper가 닫히는 동안 jaw 내부의 local sack patch 면적 proxy가 기준 이상이면 더 닫지 않고 `surface_capture_latch`를 켭니다. 이 latch는 전체 자루가 아니라 접촉된 shell point 2~3개만 Robotiq-style 2F jaw 좌표계에 고정해서 close 순간 발산을 막는 시연용 장치입니다. latch 중에는 x/y 방향은 제한적으로 따라가게 하지만, z축 위쪽 frame follow는 거의 끊어서 하부 shell과 내부 3-clump가 중력 방향으로 처지는 모습을 남깁니다. clump joint range와 shell local offset은 clamp해서 내부물이 밖으로 튀어나와 보이는 현상을 줄입니다.

잡은 상태에서 로봇을 수동으로 크게 움직이면 flex shell에 순간 에너지가 들어가 발산할 수 있습니다. 이를 줄이기 위해 latch target은 gripper 위치로 순간이동하지 않고 low-pass/step-limit로 따라가며, 수동 이동이 너무 크면 `manual_motion_too_fast`로 latch를 release합니다. 따라서 GUI에서 잡은 상태로 이동할 때는 joint slider를 크게 드래그하기보다 xyz nudge나 작은 slider 변화로 천천히 움직이는 것이 안전합니다.

## 물리 디버깅 옵션

평가 스크립트에서 아래 옵션을 비교할 수 있습니다.

- `--multiccd off|on`: MuJoCo 3.1.6에서 지원되는 multiccd flag입니다.
- `--nativeccd off|on`: 현재 설치된 MuJoCo 3.1.6 XML에서는 `nativeccd`가 지원되지 않으므로 요청값만 로그에 남깁니다.
- `--noslip-iterations 0|1|2|3`: noslip solver 반복 수입니다.
- `--pad-condim 3|4`: finger pad contact 차원입니다.
- `--pad-profile flat|shallow_concave|lip`: 평면 pad, 얕은 concave/lip pad, 더 뚜렷한 inward lip pad입니다.
- `--selfcollide-mode none|auto`: flex shell self-collision 설정입니다.
- `--vertcollide false|true`: 현재 MuJoCo 3.1.6 XML에서는 `vertcollide`가 지원되지 않으므로 요청값만 로그에 남깁니다.
- `--shell-thickness-scale`: flex shell radius scale입니다.
- `--close-seconds`: gripper close 시간을 바꿔 close velocity를 조절합니다.
- `--close-timestep 0.001|0.0005|0.0002`: close phase에만 적용되는 작은 timestep입니다.
- `--gripper-kv`: latch/assist force의 stiffness proxy입니다.
- `--gripper-dampratio`: latch/assist force의 damping ratio입니다.
- `--precompression-dwell-seconds`: close 후 tug-test 전 dwell 시간입니다.

## 실행

GUI에서 시연용 assisted hold를 보려면:

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/realistic_top_grasp/run_dual_ur5_top_grasp_gui.py --scenario simple_fold --content-case underfilled
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/realistic_top_grasp/run_dual_ur5_top_grasp_gui.py --scenario simple_fold --content-case eccentric
```

순수 접촉 평가:

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/realistic_top_grasp/run_contact_only_eval.py --scenario simple_fold --content-case underfilled --target-label auto --trials 1
```

조건부 capture surrogate 평가:

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/realistic_top_grasp/run_qualification_gated_capture.py --scenario simple_fold --content-case underfilled --target-label auto --trials 1
```

물리 옵션 비교 예:

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' demos/realistic_top_grasp/run_qualification_gated_capture.py --scenario severe_fold --content-case underfilled --target-label auto --trials 1 --selfcollide-mode auto --noslip-iterations 1 --multiccd on --pad-profile shallow_concave --pad-condim 4 --shell-thickness-scale 1.15 --close-timestep 0.0005
```

## 저장 결과

- CSV: `demos/realistic_top_grasp/out/as/<mode>/<case_and_options>/summary.csv`
- Markdown: `demos/realistic_top_grasp/out/as/<mode>/<case_and_options>/summary.md`
- 이미지: `initial.png`, `candidate_overlay.png`, `close.png`, `rollback.png`, `tug_test.png`, `micro_lift.png`, `latch_on.png`, `final_lift.png`, `lift.png`
- MP4/frame sequence: 각 trial 폴더의 `sequence.mp4`와 `frames/`

`support_sag`와 severe fold에서는 top-only grasp가 실패하거나 latch가 켜지지 않는 결과가 정상적으로 나올 수 있습니다. 그 실패가 이후 scoop/support-state primitive가 필요한 근거입니다.
