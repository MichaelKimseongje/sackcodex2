from __future__ import annotations

import argparse
import time
from pathlib import Path

EXPERIMENTAL_WARNING = (
    "warning=experimental_soft_reference "
    "(not part of the main support-state benchmark; known to diverge or sag)"
)


def main():
    try:
        import mujoco
        import mujoco.viewer
    except ModuleNotFoundError as exc:
        raise SystemExit("MuJoCo Python 패키지가 필요합니다.") from exc

    from mujoco_sack_pile.soft_sack_prototypes import SoftSackPrototypeGenerator

    parser = argparse.ArgumentParser(
        description="Experimental mesh soft-sack reference: mesh-only vs mesh+internal-support comparison"
    )
    parser.add_argument(
        "--variant",
        choices=("regular_well_filled", "low_fill_top_collapsed", "side_bulged_unstable"),
        default="regular_well_filled",
    )
    parser.add_argument("--mesh-file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-id", type=str, default="soft_sack_compare")
    parser.add_argument("--settle-seconds", type=float, default=0.2)
    parser.add_argument("--mode", choices=("compare", "mesh_only", "mesh_with_payload"), default="compare")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--fixed-camera", action="store_true")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    generator = SoftSackPrototypeGenerator(base_dir)
    spec = generator.sample_spec(variant=args.variant, seed=args.seed, mesh_file=args.mesh_file)

    if args.mode == "compare":
        print(EXPERIMENTAL_WARNING)
        print("main_benchmark_runner=run_baseline.py")
        print("compare_mode=headless_only")
        print("note=use --mode mesh_only or --mode mesh_with_payload to open a viewer")
        reports = generator.compare_headless(spec=spec, episode_id=args.episode_id, settle_seconds=args.settle_seconds)
        print(f"variant={spec.variant}")
        print(f"mesh_file={spec.mesh_file}")
        for mode, report in reports.items():
            print(f"[{mode}]")
            print(f"  xml={report.xml_path}")
            print(f"  stable={report.stable}")
            print(f"  non_finite={report.non_finite}")
            print(f"  tail_mean_qvel={report.tail_mean_qvel:.4f}")
            print(f"  tail_max_qvel={report.tail_max_qvel:.4f}")
            print(f"  peak_contact_count={report.peak_contact_count}")
            print(f"  final_min_geom_height={report.final_min_geom_height:.4f}")
            print(f"  final_z_span={report.final_z_span:.4f}")
            print(f"  failure_tags={','.join(report.failure_tags) if report.failure_tags else 'none'}")
        return

    xml_path = generator.generate_xml(spec=spec, mode=args.mode, episode_id=args.episode_id)
    print(EXPERIMENTAL_WARNING)
    print("main_benchmark_runner=run_baseline.py")
    print(f"variant={spec.variant}")
    print(f"mesh_file={spec.mesh_file}")
    print(f"mode={args.mode}")
    print(f"scene_xml={xml_path}")
    print("viewer_help:")
    print("  left drag / right drag / wheel : 카메라 이동")
    print("  double-click body : 자루 선택")
    print("  Ctrl + drag : 선택한 자루 perturb")
    print("  Space : 일시정지 / 재개")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    paused = False

    def on_key(keycode: int):
        nonlocal paused
        if keycode == 32:
            paused = not paused
            print(f"viewer_paused={paused}")

    if args.headless:
        report = generator.simulate_settle(xml_path=xml_path, settle_seconds=args.settle_seconds, mode=args.mode)
        print(f"stable={report.stable}")
        print(f"tail_mean_qvel={report.tail_mean_qvel:.4f}")
        print(f"tail_max_qvel={report.tail_max_qvel:.4f}")
        print(f"peak_contact_count={report.peak_contact_count}")
        print(f"final_min_geom_height={report.final_min_geom_height:.4f}")
        print(f"final_z_span={report.final_z_span:.4f}")
        print(f"failure_tags={','.join(report.failure_tags) if report.failure_tags else 'none'}")
        return

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=on_key,
        show_left_ui=True,
        show_right_ui=True,
    ) as viewer:
        if args.fixed_camera:
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overview")
        while viewer.is_running():
            if not paused:
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
