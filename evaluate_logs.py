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
    target_case_rows = defaultdict(list)
    target_shape_rows = defaultdict(list)
    target_difficulty_rows = defaultdict(list)
    settle_failure_counter = Counter()
    failure_counter = Counter()
    for row in rows:
        baseline_rows[row["baseline"]].append(row)
        failure_counter.update(row["metrics"]["failure_tags"])
        settle_report = row.get("settle_report") or {}
        settle_failure_counter.update(settle_report.get("failure_tags") or [])
        target_case = ((row.get("benchmark") or {}).get("target_case") or {})
        target_sack = next(
            (sack for sack in row.get("sacks", []) if sack.get("is_target") or sack.get("name") == row.get("target_name")),
            None,
        )
        shape_family = target_case.get("shape_family") or row.get("target_variant")
        pile_difficulty = target_case.get("pile_difficulty") or row.get("target_pile_difficulty") or (target_sack or {}).get("pile_difficulty")
        case_id = target_case.get("case_id") or (target_sack or {}).get("benchmark_case_id")

        if shape_family:
            target_shape_rows[shape_family].append(row)
        if pile_difficulty:
            target_difficulty_rows[pile_difficulty].append(row)
        if case_id:
            target_case_rows[case_id].append(row)

    print(f"episodes={len(rows)}")
    for baseline, items in sorted(baseline_rows.items()):
        success = sum(1 for item in items if item["metrics"]["support_success"])
        avg_score = sum(item["metrics"]["support_state_score"] for item in items) / len(items)
        avg_depth = sum(item["metrics"]["scoop_insertion_depth"] for item in items) / len(items)
        print(
            f"{baseline}: success_rate={success / len(items):.3f}, "
            f"avg_score={avg_score:.3f}, avg_insertion_depth={avg_depth:.3f}, n={len(items)}"
        )

    if target_shape_rows:
        print("target_shape_family:")
        for shape_family, items in sorted(target_shape_rows.items()):
            success = sum(1 for item in items if item["metrics"]["support_success"])
            print(f"  {shape_family}: success_rate={success / len(items):.3f}, n={len(items)}")

    if target_difficulty_rows:
        print("target_pile_difficulty:")
        for pile_difficulty, items in sorted(target_difficulty_rows.items()):
            success = sum(1 for item in items if item["metrics"]["support_success"])
            print(f"  {pile_difficulty}: success_rate={success / len(items):.3f}, n={len(items)}")

    if target_case_rows:
        print("target_case:")
        for case_id, items in sorted(target_case_rows.items()):
            success = sum(1 for item in items if item["metrics"]["support_success"])
            avg_score = sum(item["metrics"]["support_state_score"] for item in items) / len(items)
            print(f"  {case_id}: success_rate={success / len(items):.3f}, avg_score={avg_score:.3f}, n={len(items)}")

    print("failure_tags:")
    for tag, count in failure_counter.most_common():
        print(f"  {tag}: {count}")

    if settle_failure_counter:
        print("settle_failure_tags:")
        for tag, count in settle_failure_counter.most_common():
            print(f"  {tag}: {count}")


if __name__ == "__main__":
    main()
