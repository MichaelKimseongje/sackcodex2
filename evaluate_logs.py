from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="episode 로그 요약 스크립트")
    parser.add_argument(
        "--logfile",
        type=Path,
        default=Path("mujoco_sack_pile/logs/episode_history.jsonl"),
    )
    args = parser.parse_args()

    if not args.logfile.exists():
        raise SystemExit(f"로그 파일이 없습니다: {args.logfile}")

    rows = [json.loads(line) for line in args.logfile.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit("로그가 비어 있습니다.")

    baseline_rows = defaultdict(list)
    failure_counter = Counter()
    for row in rows:
        baseline_rows[row["baseline"]].append(row)
        failure_counter.update(row["metrics"]["failure_tags"])

    print(f"episodes={len(rows)}")
    for baseline, items in sorted(baseline_rows.items()):
        success = sum(1 for item in items if item["metrics"]["support_success"])
        avg_score = sum(item["metrics"]["support_state_score"] for item in items) / len(items)
        avg_depth = sum(item["metrics"]["scoop_insertion_depth"] for item in items) / len(items)
        print(
            f"{baseline}: success_rate={success / len(items):.3f}, "
            f"avg_score={avg_score:.3f}, avg_insertion_depth={avg_depth:.3f}, n={len(items)}"
        )

    print("failure_tags:")
    for tag, count in failure_counter.most_common():
        print(f"  {tag}: {count}")


if __name__ == "__main__":
    main()
