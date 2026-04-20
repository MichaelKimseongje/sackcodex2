from __future__ import annotations

import argparse
from pathlib import Path


def main():
    try:
        import mujoco  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit("MuJoCo Python 패키지가 필요합니다. `pip install mujoco` 후 다시 실행해 주세요.") from exc

    from mujoco_sack_pile.phenomenon_demos import run_integrated_demo

    parser = argparse.ArgumentParser(description="복수 uncertainty가 섞인 통합 support-state demo를 실행하고 mp4로 저장한다.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    save_path = Path(args.save) if args.save is not None else None
    result = run_integrated_demo(
        base_dir=base_dir,
        seed=args.seed,
        save_path=save_path,
    )

    print("demo=integrated_case")
    print(f"scene_xml={result['scene_xml']}")
    print(f"seed={result['seed']}")
    print(f"settle_stable={result['settle_stable']}")
    print(f"settle_failure_tags={','.join(result['settle_failure_tags']) if result['settle_failure_tags'] else 'none'}")
    print(f"final_support_score={result['final_metrics']['support_state_score']:.3f}")
    print(f"final_tilt_deg={result['final_metrics']['tilt_deg']:.3f}")
    print(f"final_failure_tags={','.join(result['final_metrics']['failure_tags']) if result['final_metrics']['failure_tags'] else 'none'}")
    if result["save_path"] is not None:
        print(f"saved_mp4={result['save_path']}")


if __name__ == "__main__":
    main()
