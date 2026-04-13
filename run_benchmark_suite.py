from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from mujoco_sack_pile.baselines.heuristics import BASELINES
from mujoco_sack_pile.benchmark_definition import PILE_DIFFICULTIES
from mujoco_sack_pile.environment import SackPileEnv
from mujoco_sack_pile.scene_generator import SACK_VARIANTS, SceneGenerator


def _seed_for_case(seed_start: int, family_idx: int, difficulty_idx: int, episode_idx: int, attempt_idx: int) -> int:
    """case/episode/attempt 조합마다 고유한 seed를 만든다."""

    return seed_start + family_idx * 1000 + difficulty_idx * 100 + episode_idx * 10 + attempt_idx


def main():
    parser = argparse.ArgumentParser(description="MuJoCo sack pile benchmark suite runner")
    parser.add_argument("--episodes-per-case", type=int, default=1)
    parser.add_argument("--max-scene-attempts", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--sack-count", type=int, default=None)
    parser.add_argument("--settle-seconds", type=float, default=5.0)
    parser.add_argument("--target-families", nargs="*", choices=sorted(SACK_VARIANTS.keys()), default=None)
    parser.add_argument("--pile-difficulties", nargs="*", choices=PILE_DIFFICULTIES, default=None)
    parser.add_argument("--baselines", nargs="*", choices=sorted(BASELINES.keys()), default=None)
    args = parser.parse_args()

    family_names = args.target_families or sorted(SACK_VARIANTS.keys())
    difficulty_names = args.pile_difficulties or list(PILE_DIFFICULTIES)
    baseline_names = args.baselines or sorted(BASELINES.keys())

    base_dir = Path(__file__).resolve().parent
    log_dir = base_dir / "mujoco_sack_pile" / "logs"
    generator = SceneGenerator(base_dir)

    stable_cases: list[tuple[str, str, int, object]] = []
    skipped_cases: list[tuple[str, str, int, list[str]]] = []
    settle_failure_counter = Counter()

    # 먼저 case별로 5초 settle 안정성이 확보되는 scene만 골라낸다.
    for family_idx, family_name in enumerate(family_names):
        for difficulty_idx, difficulty_name in enumerate(difficulty_names):
            for episode_idx in range(args.episodes_per_case):
                stable_scene = None
                final_failure_tags: list[str] = ["scene_generation_failed"]
                for attempt_idx in range(args.max_scene_attempts):
                    seed = _seed_for_case(args.seed_start, family_idx, difficulty_idx, episode_idx, attempt_idx)
                    episode_id = f"{family_name}_{difficulty_name}_ep{episode_idx}_try{attempt_idx}"
                    scene = generator.generate_episode(
                        seed=seed,
                        episode_id=episode_id,
                        sack_count=args.sack_count,
                        target_variant=family_name,
                        pile_difficulty=difficulty_name,
                    )
                    env = SackPileEnv(scene=scene, log_dir=log_dir)
                    settle_report = env.reset(settle_seconds=args.settle_seconds, verify_stability=True)
                    if settle_report.stable:
                        stable_scene = scene
                        print(
                            f"stable_scene family={family_name} difficulty={difficulty_name} "
                            f"episode={episode_idx} seed={seed} case={scene.target_case.case_id if scene.target_case else 'none'}"
                        )
                        break
                    final_failure_tags = settle_report.failure_tags or ["scene_unstable"]
                    settle_failure_counter.update(final_failure_tags)

                if stable_scene is None:
                    skipped_cases.append((family_name, difficulty_name, episode_idx, final_failure_tags))
                    print(
                        f"skipped_unstable_scene family={family_name} difficulty={difficulty_name} "
                        f"episode={episode_idx} tags={','.join(final_failure_tags)}"
                    )
                    continue
                stable_cases.append((family_name, difficulty_name, episode_idx, stable_scene))

    result_rows: list[dict] = []
    failure_counter = Counter()

    # 안정한 scene만 대상으로 baseline 3개를 실행한다.
    for family_name, difficulty_name, episode_idx, scene in stable_cases:
        for baseline_name in baseline_names:
            env = SackPileEnv(scene=scene, log_dir=log_dir)
            settle_report = env.reset(settle_seconds=args.settle_seconds, verify_stability=True)
            if not settle_report.stable:
                settle_failure_counter.update(settle_report.failure_tags or ["scene_unstable"])
                print(
                    f"rerun_unstable_scene family={family_name} difficulty={difficulty_name} "
                    f"episode={episode_idx} baseline={baseline_name} tags={','.join(settle_report.failure_tags)}"
                )
                continue

            BASELINES[baseline_name](env, viewer=None)
            metrics = env.finalize_metrics()
            env.save_episode_log(baseline_name, metrics)

            result_rows.append(
                {
                    "family": family_name,
                    "difficulty": difficulty_name,
                    "episode_idx": episode_idx,
                    "baseline": baseline_name,
                    "success": metrics.support_success,
                    "failure_tags": metrics.failure_tags,
                    "support_state_score": metrics.support_state_score,
                }
            )
            failure_counter.update(metrics.failure_tags)
            print(
                f"baseline_result family={family_name} difficulty={difficulty_name} "
                f"episode={episode_idx} baseline={baseline_name} success={metrics.support_success} "
                f"failure_tags={','.join(metrics.failure_tags) if metrics.failure_tags else 'none'}"
            )

    print(f"stable_scene_count={len(stable_cases)}")
    print(f"skipped_unstable_scene_count={len(skipped_cases)}")
    if settle_failure_counter:
        print("settle_failure_tags:")
        for tag, count in settle_failure_counter.most_common():
            print(f"  {tag}: {count}")

    if not result_rows:
        print("baseline_results=none")
        return

    baseline_rows = defaultdict(list)
    case_rows = defaultdict(list)
    for row in result_rows:
        baseline_rows[row["baseline"]].append(row)
        case_rows[(row["family"], row["difficulty"])].append(row)

    print("baseline_summary:")
    for baseline_name, rows in sorted(baseline_rows.items()):
        success_count = sum(1 for row in rows if row["success"])
        avg_score = sum(row["support_state_score"] for row in rows) / len(rows)
        print(
            f"  {baseline_name}: success_rate={success_count / len(rows):.3f}, "
            f"avg_score={avg_score:.3f}, n={len(rows)}"
        )

    print("case_summary:")
    for (family_name, difficulty_name), rows in sorted(case_rows.items()):
        success_count = sum(1 for row in rows if row["success"])
        print(
            f"  {family_name} / {difficulty_name}: success_rate={success_count / len(rows):.3f}, "
            f"n={len(rows)}"
        )

    print("failure_tags:")
    for tag, count in failure_counter.most_common():
        print(f"  {tag}: {count}")


if __name__ == "__main__":
    main()
