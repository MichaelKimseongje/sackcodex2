from __future__ import annotations

import argparse
from pathlib import Path

from mujoco_sack_pile.scene_generator import SceneGenerator


def main():
    parser = argparse.ArgumentParser(description="MuJoCo sack pile scene generator")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-id", type=str, default="scene_preview")
    parser.add_argument("--sack-count", type=int, default=None)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    generator = SceneGenerator(base_dir)
    scene = generator.generate_episode(seed=args.seed, episode_id=args.episode_id, sack_count=args.sack_count)
    print(f"scene_xml={scene.xml_path}")
    print(f"target={scene.target_name}")
    print(f"target_variant={scene.target_variant}")
    for sack in scene.sacks:
        print(
            f"{sack.name}: variant={sack.variant.name}, mesh={sack.mesh_file}, "
            f"exposed={sack.exposed_face}, pos={sack.pos}, euler={tuple(round(v, 3) for v in sack.euler)}, "
            f"fill={sack.fill_ratio:.2f}, collapse={sack.top_collapse:.3f}, bulge={sack.side_bulge:.3f}"
        )


if __name__ == "__main__":
    main()
