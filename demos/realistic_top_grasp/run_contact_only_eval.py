from __future__ import annotations

import argparse
from pathlib import Path

from generate_sack_mesh import (
    CCD_MODES,
    NATIVECCD_MODES,
    PAD_CONDIM_OPTIONS,
    PAD_PROFILES,
    SELF_COLLISION_MODES,
    VERTCOLLIDE_MODES,
    available_content_cases,
    available_scenarios,
)
from top_grasp_sim import CLOSE_TIMESTEP_OPTIONS, run_contact_only_suite


TARGET_LABELS = ("auto", "seam", "fold", "plain_top")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="auto search + probing top graspability evaluation")
    parser.add_argument(
        "--mode",
        choices=("contact_only_eval", "qualification_gated_latch_eval", "qualification_gated_capture"),
        default="contact_only_eval",
    )
    parser.add_argument("--scenario", choices=("all", *available_scenarios()), default="all")
    parser.add_argument("--content-case", choices=available_content_cases(), default="underfilled")
    parser.add_argument("--selfcollide-mode", choices=SELF_COLLISION_MODES, default="none")
    parser.add_argument("--noslip-iterations", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--multiccd", choices=CCD_MODES, default="off")
    parser.add_argument("--nativeccd", choices=NATIVECCD_MODES, default="off")
    parser.add_argument("--pad-profile", choices=PAD_PROFILES, default="lip")
    parser.add_argument("--pad-condim", type=int, choices=PAD_CONDIM_OPTIONS, default=4)
    parser.add_argument("--vertcollide", choices=VERTCOLLIDE_MODES, default="false")
    parser.add_argument("--shell-thickness-scale", type=float, default=1.0)
    parser.add_argument("--close-seconds", type=float, default=0.24)
    parser.add_argument("--precompression-dwell-seconds", type=float, default=0.06)
    parser.add_argument("--close-timestep", type=float, choices=CLOSE_TIMESTEP_OPTIONS, default=0.001)
    parser.add_argument("--gripper-kv", type=float, default=320.0)
    parser.add_argument("--gripper-dampratio", type=float, default=0.20)
    parser.add_argument(
        "--target-label",
        choices=("all", *TARGET_LABELS),
        default="auto",
        help="candidate selector only; never used as a success rule",
    )
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-render", action="store_true", help="skip PNG/frame rendering")
    parser.add_argument("--no-mp4", action="store_true", help="save PNG/frame sequence only")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenarios = available_scenarios() if args.scenario == "all" else (args.scenario,)
    requested_target_labels = TARGET_LABELS if args.target_label == "all" else (args.target_label,)
    results, summary_csv, summary_md = run_contact_only_suite(
        scenarios=scenarios,
        mode=args.mode,
        content_case=args.content_case,
        requested_target_labels=requested_target_labels,
        trials=args.trials,
        output_dir=args.output_dir,
        render=not args.no_render,
        save_mp4=not args.no_mp4,
        width=args.width,
        height=args.height,
        selfcollide_mode=args.selfcollide_mode,
        noslip_iterations=args.noslip_iterations,
        multiccd_mode=args.multiccd,
        nativeccd_mode=args.nativeccd,
        pad_profile=args.pad_profile,
        pad_condim=args.pad_condim,
        vertcollide_mode=args.vertcollide,
        shell_thickness_scale=args.shell_thickness_scale,
        close_seconds=args.close_seconds,
        precompression_dwell_seconds=args.precompression_dwell_seconds,
        close_timestep=args.close_timestep,
        gripper_kv=args.gripper_kv,
        gripper_dampratio=args.gripper_dampratio,
    )
    pass_count = sum(1 for result in results if bool(result["pass_fail"]))
    print(f"mode={args.mode}")
    print(f"content_case={args.content_case}")
    print(f"selfcollide_mode={args.selfcollide_mode}")
    print(f"noslip_iterations={args.noslip_iterations}")
    print(f"multiccd_mode={args.multiccd}")
    print(f"nativeccd_mode={args.nativeccd}")
    print(f"pad_profile={args.pad_profile}")
    print(f"pad_condim={args.pad_condim}")
    print(f"vertcollide_mode={args.vertcollide}")
    print(f"shell_thickness_scale={args.shell_thickness_scale}")
    print(f"close_timestep={args.close_timestep}")
    print(f"gripper_kv={args.gripper_kv}")
    print(f"gripper_dampratio={args.gripper_dampratio}")
    print(f"summary_csv={summary_csv}")
    print(f"summary_md={summary_md}")
    print(f"total_trials={len(results)}")
    print(f"pass_count={pass_count}")
    print(f"fail_count={len(results) - pass_count}")
    print(f"pass_rate={pass_count / max(len(results), 1):.3f}")
    return 0 if results and not any(bool(result["nonfinite"]) for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
