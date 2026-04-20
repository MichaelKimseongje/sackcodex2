from __future__ import annotations

import argparse
import sys
from pathlib import Path

from generate_sack_mesh import OUTPUT_DIR, available_content_cases, available_scenarios
from top_grasp_sim import launch_viewer, run_trial


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="realistic top graspability demo")
    parser.add_argument("--scenario", choices=available_scenarios(), default="simple_fold")
    parser.add_argument("--content-case", choices=available_content_cases(), default="underfilled")
    parser.add_argument("--trial", type=int, default=0, help="trial perturbation index")
    parser.add_argument("--viewer", action="store_true", help="play the scripted benchmark in the MuJoCo viewer")
    parser.add_argument("--manual", action="store_true", help="open the manual top-grasp GUI")
    parser.add_argument("--no-render", action="store_true", help="skip image/mp4 rendering")
    parser.add_argument("--no-mp4", action="store_true", help="save PNG/frame sequence only")
    parser.add_argument("--output-dir", type=Path, default=None, help="result directory")
    parser.add_argument("--speed", type=float, default=1.0, help="viewer playback speed multiplier")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.manual:
        # 수동 GUI는 기존 argparse 진입점을 재사용하므로 필요한 인자만 다시 넘긴다.
        from run_manual_gui import main as manual_main

        sys.argv = [
            sys.argv[0],
            "--scenario",
            args.scenario,
            "--trial",
            str(args.trial),
            "--speed",
            str(args.speed),
        ]
        return manual_main()

    if args.viewer:
        launch_viewer(args.scenario, trial_index=args.trial, speed=args.speed, content_case=args.content_case)
        return 0

    output_dir = args.output_dir or (OUTPUT_DIR / "single_demo" / args.content_case / args.scenario / f"trial_{args.trial:02d}")
    result = run_trial(
        args.scenario,
        content_case=args.content_case,
        trial_index=args.trial,
        output_dir=output_dir,
        render=not args.no_render,
        save_mp4=not args.no_mp4,
    )
    for key, value in result.items():
        print(f"{key}={value}")
    return 0 if not result["nonfinite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
