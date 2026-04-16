# Shared Flat Pillow Sack Skeleton

이 프로젝트는 사진 속 산업용 마대자루처럼 **납작한 베개형 포대**에 가까운 공통 articulated skeleton을 만듭니다.

정확한 soft sack material simulator가 아니라, 이후 여러 scenario가 같은 골격의 parameter/state만 바꿔서 생성될 수 있는 `shape-coupled semi-deformable surrogate` 기반입니다.

## 핵심 원칙

- `flex`, `flexcomp`, DEM, many-particle fill은 사용하지 않습니다.
- 자루를 단일 rigid body로 만들지 않습니다.
- scenario마다 별도 rigid object를 새로 만들지 않습니다.
- 공통 skeleton은 유지하고, payload 위치, panel joint state, fold state, neighbor 위치, support state만 바꿉니다.
- main physics는 rigid/articulated patch가 담당합니다.
- `visual_skin`은 마대자루처럼 보이기 위한 visual-only layer이고 물리 충돌에는 쓰지 않습니다.

## 사진형 마대자루 대응

사진 속 자루는 세워진 타원형 포대가 아니라, 넓고 납작한 베개형 자루입니다. 그래서 현재 skeleton은 다음 구조를 사용합니다.

- `bag_frame`: 자루 root body입니다. `bag_frame_freejoint`를 가져 이동/회전/전도가 가능합니다.
- `visual_skin`: 물리에 참여하지 않는 납작한 pillow-like sealed sack silhouette입니다.
- `visual_skin_main_pillow`: visual-only closed mesh입니다. 긴 파이프나 외부 골격처럼 보이지 않도록 기본 렌더에서는 물리 patch를 숨기고 이 외피를 보여줍니다.
- `top_surface_panels`: 4 x 3 = 12개 top patch입니다. 윗면의 넓은 천면을 표현합니다.
- `bottom_surface_panels`: 4 x 3 = 12개 bottom patch입니다. 바닥에 눌리는 하부면을 표현합니다.
- `seam_band`: 앞/뒤/좌/우 edge와 top edge를 포함한 8개 rounded seam segment입니다. 주요 파지 후보입니다.
- `corner_fold_patch_1..4`: 네 모서리 접힘/주름 patch입니다.
- `fold_flap_1`, `fold_flap_2`: 긴 edge가 접히거나 말린 부분을 표현하는 공통 flap입니다.
- `bottom_sling`: 하부 지지와 post-separation sag를 위한 3개 support piece와 slide joint입니다.
- `payload_main`, `payload_aux`: 내부 하중과 CoM offset을 표현하는 rigid ellipsoid입니다.
- `side_bulge_cue`: 시각화용 side bulge cue입니다.
- `neighbor_left`, `neighbor_right`: pile/jammed scenario용 blocker입니다.
- `hidden_support`: post-separation sag scenario용 지지체입니다. 기본은 보이지 않고 충돌도 꺼져 있습니다.

## 실행

XML 생성:

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' sack_shared/build_shared_sack.py
```

MuJoCo viewer로 직접 보기:

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' sack_shared/run_viewer.py
```

기본 viewer는 사람이 보는 포대 외형만 보여줍니다. 내부 articulated patch 골격을 같이 보려면:

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' sack_shared/run_viewer.py --show-physics
```

이미지 렌더:

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' sack_shared/render_shared_sack.py
```

검증:

```powershell
& 'C:\Users\Michael3080\anaconda3\envs\Yolov9\python.exe' sack_shared/validate_shared_sack.py
```

## 출력

생성 XML:

```text
sack_shared/generated/scene_shared_sack.xml
```

출력 폴더:

```text
sack_shared/out
```

주요 파일:

- `shared_sack_front.png`
- `shared_sack_side.png`
- `shared_sack_top_angled.png`
- `shared_sack_summary.csv`
- `shared_sack_summary.md`

## 검증 기준

`validate_shared_sack.py`는 다음을 확인합니다.

- `bag_frame`이 하나만 존재하는지
- `bag_frame_freejoint`가 있는지
- `flex`가 없는지
- movable articulated patch가 38개인지
- top/bottom surface panel이 존재하는지
- `fold_flap_1`, `fold_flap_2`가 존재하는지
- `payload_main`, `payload_aux`가 존재하는지
- `visual_skin`이 physics-free인지
- settle 중 NaN/발산이 없는지

## 해석

이 모델은 실제 천/분말 재질을 정확히 해석하지 않습니다. 대신 사진 같은 마대자루의 작업 관련 구조를 안정적인 articulated skeleton으로 표현합니다.

연구에서는 다음처럼 설명하는 것이 안전합니다.

```text
The sack is represented by a shared flat pillow-like articulated skeleton.
It is not a high-fidelity deformable material simulator.
Scenario diversity is introduced by changing state and parameters of the same skeleton, not by replacing the sack with unrelated rigid objects.
```
