from __future__ import annotations

import argparse
import time
from pathlib import Path


class PreviewControls:
    """현상 preview용 최소 viewer 제어."""

    def __init__(self):
        self.paused = False

    def handle_key(self, keycode: int):
        if keycode == 32:
            self.paused = not self.paused
            print(f"viewer_paused={self.paused}")
        elif keycode in (ord("H"), ord("h")):
            self.print_help()

    @staticmethod
    def print_help():
        print("viewer_help:")
        print("  left drag / right drag / wheel : 카메라 이동")
        print("  double-click body : 객체 선택")
        print("  Ctrl + drag : 선택한 객체 perturb")
        print("  Space : 일시정지 / 재개")
        print("  H : 도움말 다시 출력")


def main():
    try:
        import mujoco
        import mujoco.viewer
    except ModuleNotFoundError as exc:
        raise SystemExit("MuJoCo Python 패키지가 필요합니다. `pip install mujoco` 후 다시 실행해 주세요.") from exc

    from mujoco_sack_pile.environment import SackPileEnv
    from mujoco_sack_pile.phenomenon_presets import (
        format_summary_lines,
        list_phenomena,
        run_partial_support_demo,
        select_scene_for_phenomenon,
    )

    parser = argparse.ArgumentParser(description="개별 uncertainty 현상을 하나씩 구분해서 보여주는 preview runner")
    parser.add_argument("--phenomenon", choices=list_phenomena(), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attempt-limit", type=int, default=48)
    parser.add_argument("--settle-seconds", type=float, default=5.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--fixed-camera", action="store_true")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    scene, preset, summary = select_scene_for_phenomenon(
        base_dir=base_dir,
        phenomenon=args.phenomenon,
        base_seed=args.seed,
        attempt_limit=args.attempt_limit,
    )
    env = SackPileEnv(scene=scene, log_dir=base_dir / "mujoco_sack_pile" / "logs")
    settle_report = env.reset(settle_seconds=args.settle_seconds, verify_stability=True)

    print("note=current_main_path=proxy_based_task_benchmark")
    print("note=current_mujoco=3.1.6")
    print("note=trilinear_3d_flex_is_not_available_in_this_environment")
    for line in format_summary_lines(preset, summary):
        print(line)
    print(f"scene_xml={scene.xml_path}")
    print(f"settle_stable={settle_report.stable}")
    print(f"settle_failure_tags={','.join(settle_report.failure_tags) if settle_report.failure_tags else 'none'}")

    if args.headless:
        if preset.run_support_demo:
            support_report = run_partial_support_demo(env, viewer=None)
            print("support_demo=partial_support_sag")
            print(f"support_success={support_report['support_success']}")
            print(f"support_state_score={support_report['support_state_score']:.3f}")
            print(f"micro_lift_stability={support_report['micro_lift_stability']:.3f}")
            print(f"slip_distance={support_report['slip_distance']:.3f}")
            print(f"tilt_deg={support_report['tilt_deg']:.3f}")
            print(f"dropped={support_report['dropped']}")
            print(
                f"failure_tags={','.join(support_report['failure_tags']) if support_report['failure_tags'] else 'none'}"
            )
        return

    controls = PreviewControls()
    controls.print_help()
    with mujoco.viewer.launch_passive(
        env.model,
        env.data,
        key_callback=controls.handle_key,
        show_left_ui=True,
        show_right_ui=True,
    ) as viewer:
        if args.fixed_camera:
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "overview")

        if preset.run_support_demo:
            support_report = run_partial_support_demo(env, viewer=viewer)
            print("support_demo=partial_support_sag")
            print(f"support_success={support_report['support_success']}")
            print(f"support_state_score={support_report['support_state_score']:.3f}")
            print(f"micro_lift_stability={support_report['micro_lift_stability']:.3f}")
            print(f"slip_distance={support_report['slip_distance']:.3f}")
            print(f"tilt_deg={support_report['tilt_deg']:.3f}")
            print(f"dropped={support_report['dropped']}")
            print(
                f"failure_tags={','.join(support_report['failure_tags']) if support_report['failure_tags'] else 'none'}"
            )

        while viewer.is_running():
            if controls.paused:
                env.render_viewer(viewer)
                time.sleep(env.model.opt.timestep)
                continue
            env.step(1, viewer=viewer, sleep=True)


if __name__ == "__main__":
    main()
