from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import mujoco.viewer
import numpy as np

from low_fill_builder import (
    DEMO_SECONDS,
    DISTURB_AT_SECONDS,
    DISTURB_CAPTURE_DELAY,
    FRAME_COUNT,
    RENDER_HEIGHT,
    RENDER_WIDTH,
    apply_disturbance,
    default_output_dir,
    load_scene,
    make_render_option,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="standalone underfilled sack demo")
    parser.add_argument(
        "--with-ballast",
        action="store_true",
        help="Version B: 하단 중심에 단일 ballast를 추가합니다.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="offscreen 저장 대신 interactive viewer를 엽니다.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="이미지/시퀀스 저장 경로",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=DEMO_SECONDS,
        help="총 시뮬레이션 시간",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=RENDER_WIDTH,
        help="offscreen 렌더 폭",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=RENDER_HEIGHT,
        help="offscreen 렌더 높이",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="viewer 실행 속도 배수",
    )
    return parser.parse_args()


def _capture_rgb(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    scene_option: mujoco.MjvOption,
) -> np.ndarray:
    renderer.update_scene(data, camera="overview", scene_option=scene_option)
    return np.asarray(renderer.render(), dtype=np.uint8)


def _save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, image)


def render_demo(
    *,
    with_ballast: bool,
    output_dir: Path,
    seconds: float,
    width: int,
    height: int,
) -> None:
    xml_path, model, data = load_scene(with_ballast=with_ballast)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    renderer = mujoco.Renderer(model, height=height, width=width)
    scene_option = make_render_option()

    total_steps = max(1, math.ceil(seconds / model.opt.timestep))
    disturb_step = min(total_steps - 1, max(1, math.ceil(DISTURB_AT_SECONDS / model.opt.timestep)))
    settled_step = min(total_steps - 1, max(1, math.ceil(1.2 / model.opt.timestep)))
    disturbed_capture_step = min(
        total_steps - 1,
        disturb_step + max(1, math.ceil(DISTURB_CAPTURE_DELAY / model.opt.timestep)),
    )

    sample_steps = np.linspace(0, total_steps, FRAME_COUNT, dtype=int)
    capture_steps: dict[int, list[tuple[str, int | None]]] = {
        0: [("initial", None)],
        settled_step: [("settled", None)],
        disturbed_capture_step: [("disturbed", None)],
    }
    for frame_index, step in enumerate(sample_steps):
        capture_steps.setdefault(int(step), []).append(("frame", int(frame_index)))

    frame_images: dict[int, np.ndarray] = {}

    def _store_capture_items(step: int) -> None:
        if step not in capture_steps:
            return

        image = _capture_rgb(renderer, data, scene_option)
        for capture_kind, capture_index in capture_steps[step]:
            if capture_kind == "frame" and capture_index is not None:
                frame_images[capture_index] = image.copy()
            else:
                _save_png(output_dir / f"{capture_kind}.png", image)

    _store_capture_items(0)

    for step in range(1, total_steps + 1):
        if step == disturb_step:
            apply_disturbance(model, data, with_ballast=with_ballast)

        mujoco.mj_step(model, data)
        _store_capture_items(step)

    mp4_path = output_dir / "sequence.mp4"
    writer = None
    try:
        writer = imageio.get_writer(mp4_path, fps=20)
    except Exception:
        writer = None

    for frame_index in range(FRAME_COUNT):
        image = frame_images[frame_index]
        _save_png(frames_dir / f"frame_{frame_index:03d}.png", image)
        if writer is not None:
            writer.append_data(image)

    if writer is not None:
        writer.close()

    renderer.close()

    print(f"xml={xml_path}")
    print(f"output_dir={output_dir}")
    print(f"initial={output_dir / 'initial.png'}")
    print(f"settled={output_dir / 'settled.png'}")
    print(f"disturbed={output_dir / 'disturbed.png'}")
    print(f"frames_dir={frames_dir}")
    print(f"mp4={mp4_path}")


def launch_viewer(*, with_ballast: bool, speed: float) -> None:
    xml_path, model, data = load_scene(with_ballast=with_ballast)
    print(f"xml={xml_path}")
    print("free_camera=true")
    print("left-drag=rotate  right-drag=pan  wheel=zoom")

    sleep_dt = model.opt.timestep / max(speed, 1e-6)
    disturb_step = max(1, math.ceil(DISTURB_AT_SECONDS / model.opt.timestep))
    scene_option = make_render_option()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.10])
        viewer.cam.distance = 1.10
        viewer.cam.azimuth = 136.0
        viewer.cam.elevation = -14.0
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXFACE] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXEDGE] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXVERT] = True

        step = 0
        while viewer.is_running():
            step_start = time.perf_counter()
            if step == disturb_step:
                apply_disturbance(model, data, with_ballast=with_ballast)
            mujoco.mj_step(model, data)
            viewer.sync()
            step += 1

            elapsed = time.perf_counter() - step_start
            remaining = sleep_dt - elapsed
            if remaining > 0:
                time.sleep(remaining)


def main() -> int:
    args = parse_args()
    output_dir = (args.output_dir or default_output_dir(with_ballast=args.with_ballast)).resolve()

    if args.viewer:
        launch_viewer(with_ballast=args.with_ballast, speed=args.speed)
        return 0

    render_demo(
        with_ballast=args.with_ballast,
        output_dir=output_dir,
        seconds=args.seconds,
        width=args.width,
        height=args.height,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
