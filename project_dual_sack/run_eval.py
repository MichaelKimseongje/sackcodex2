from __future__ import annotations

import argparse

from scenario_builder import SCENARIO_NAMES
from validate_dual_support_state import EVAL_MODES, evaluate_all
from validate_scenarios import validate_all


def main() -> int:
    parser = argparse.ArgumentParser(description="Run scenario validation and dual robot support-state evaluation")
    parser.add_argument("--scenario", choices=SCENARIO_NAMES + ("all",), default="all")
    parser.add_argument("--mode", choices=EVAL_MODES + ("all",), default="all")
    parser.add_argument("--skip-scenario-validation", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    if not args.skip_scenario_validation:
        print("[1/2] scenario shape validation")
        validate_all(scenario=args.scenario, render=not args.no_render)
    print("[2/2] dual support-state evaluation")
    rows = evaluate_all(scenario=args.scenario, mode=args.mode, render=not args.no_render)
    success = sum(1 for row in rows if row["pass_fail"])
    print(f"summary: {success}/{len(rows)} pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
