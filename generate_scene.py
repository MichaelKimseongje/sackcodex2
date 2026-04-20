from __future__ import annotations

import argparse
from pathlib import Path

from mujoco_sack_pile.benchmark_definition import PILE_DIFFICULTIES
from mujoco_sack_pile.scene_generator import SceneGenerator
from mujoco_sack_pile.scene_generator import SACK_VARIANTS


def main():
    parser = argparse.ArgumentParser(description="MuJoCo task-driven benchmark scene generator")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-id", type=str, default="scene_preview")
    parser.add_argument("--sack-count", type=int, default=None)
    parser.add_argument("--target-family", choices=sorted(SACK_VARIANTS.keys()), default=None)
    parser.add_argument("--pile-difficulty", choices=PILE_DIFFICULTIES, default=None)
    parser.add_argument("--check-settle", action="store_true")
    parser.add_argument("--settle-seconds", type=float, default=5.0)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    generator = SceneGenerator(base_dir)
    scene = generator.generate_episode(
        seed=args.seed,
        episode_id=args.episode_id,
        sack_count=args.sack_count,
        target_variant=args.target_family,
        pile_difficulty=args.pile_difficulty,
    )
    print(f"scene_xml={scene.xml_path}")
    print(f"benchmark={scene.benchmark_name}")
    print(f"research_question={scene.research_question}")
    print(f"target={scene.target_name}")
    print(f"target_variant={scene.target_variant}")
    print(f"target_pile_difficulty={scene.target_pile_difficulty}")
    if scene.target_case is not None:
        print(f"target_case={scene.target_case.case_id}")
        print(f"target_tags={','.join(scene.target_case.tags)}")
    for sack in scene.sacks:
        print(
            f"{sack.name}: variant={sack.variant.name}, mesh={sack.mesh_file}, "
            f"exposed={sack.exposed_face}, pos={sack.pos}, euler={tuple(round(v, 3) for v in sack.euler)}, "
            f"fill={sack.fill_ratio:.2f}, collapse={sack.top_collapse:.3f}, bulge={sack.side_bulge:.3f}, "
            f"pile_difficulty={sack.pile_difficulty}"
        )

    if args.check_settle:
        from mujoco_sack_pile.environment import SackPileEnv

        env = SackPileEnv(scene=scene, log_dir=base_dir / "mujoco_sack_pile" / "logs")
        settle_report = env.reset(settle_seconds=args.settle_seconds, verify_stability=True)
        print(f"settle_stable={settle_report.stable}")
        print(f"settle_failure_tags={','.join(settle_report.failure_tags) if settle_report.failure_tags else 'none'}")
        print(f"settle_max_linear_speed={settle_report.max_linear_speed:.4f}")
        print(f"settle_max_angular_speed={settle_report.max_angular_speed:.4f}")
        print(f"settle_max_position_drift={settle_report.max_position_drift:.4f}")


if __name__ == "__main__":
    main()
