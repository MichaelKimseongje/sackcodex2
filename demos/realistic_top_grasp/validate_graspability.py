from __future__ import annotations

import argparse
from pathlib import Path

from generate_sack_mesh import OUTPUT_DIR, available_content_cases, available_scenarios
from top_grasp_sim import run_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="validate local top graspability without label-based success rules")
    parser.add_argument("--scenario", choices=("all", *available_scenarios()), default="all")
    parser.add_argument("--content-case", choices=available_content_cases(), default="underfilled")
    parser.add_argument("--trials", type=int, default=4, help="number of approach perturbations per scenario")
    parser.add_argument("--output-dir", type=Path, default=None, help="result directory")
    parser.add_argument("--no-render", action="store_true", help="skip image rendering")
    parser.add_argument("--no-mp4", action="store_true", help="save PNG/frame sequence only")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenarios = available_scenarios() if args.scenario == "all" else (args.scenario,)
    output_dir = args.output_dir or (OUTPUT_DIR / "validation" / args.content_case)
    results, summary_path = run_suite(
        scenarios=scenarios,
        content_case=args.content_case,
        trials=args.trials,
        output_dir=output_dir,
        render=not args.no_render,
        save_mp4=not args.no_mp4,
        width=args.width,
        height=args.height,
    )
    pass_count = sum(1 for result in results if result["pass_fail"])
    print(f"content_case={args.content_case}")
    print(f"summary_csv={summary_path}")
    print(f"total_trials={len(results)}")
    print(f"pass_count={pass_count}")
    print(f"fail_count={len(results) - pass_count}")
    print(f"pass_rate={pass_count / max(len(results), 1):.3f}")
    return 0 if results and not any(result["nonfinite"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
