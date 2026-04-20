from __future__ import annotations

import argparse
from pathlib import Path

import mujoco

from build_shared_sack import OUT_DIR, write_scene_xml


def _imageio():
    try:
        import imageio.v2 as imageio
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("imageio가 필요합니다. `pip install imageio` 후 다시 실행해 주세요.") from exc
    return imageio


def render_shared_sack(*, settle_seconds: float = 0.8, out_dir: Path = OUT_DIR) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    xml_path = write_scene_xml()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for _ in range(max(1, int(settle_seconds / model.opt.timestep))):
        mujoco.mj_step(model, data)

    renderer = mujoco.Renderer(model, width=1280, height=820)
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[:] = True
    # 기본 결과 이미지는 사람이 보는 마대자루 외형만 보여줍니다.
    # 물리용 articulated patch는 group 1에 두고 숨겨 골격 전시처럼 보이지 않게 합니다.
    scene_option.geomgroup[1] = False
    imageio = _imageio()
    outputs: dict[str, str] = {}
    for camera in ("front", "side", "top_angled"):
        path = out_dir / f"shared_sack_{camera}.png"
        renderer.update_scene(data, camera=camera, scene_option=scene_option)
        imageio.imwrite(path, renderer.render())
        outputs[camera] = str(path)
    renderer.close()
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the shared articulated sack skeleton")
    parser.add_argument("--settle-seconds", type=float, default=0.8)
    args = parser.parse_args()
    outputs = render_shared_sack(settle_seconds=args.settle_seconds)
    for camera, path in outputs.items():
        print(f"{camera}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
