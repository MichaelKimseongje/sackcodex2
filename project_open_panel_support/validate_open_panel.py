"""Validation for the open-panel support-state prototype."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np

from builder import write_scene_xml
from open_panel_env import OpenPanelSupportEnv, OUT_DIR, check_finite, run_scripted_trial, save_metrics


REQUIRED_METRICS = (
    "sag_index",
    "effective_com_offset",
    "scoop_load_transfer",
    "peel_ratio",
    "support_margin",
    "insertion_depth",
    "scoop_contact_force",
)


def main() -> int:
    xml_path = write_scene_xml()
    env = OpenPanelSupportEnv(xml_path)
    inventory = env.runtime_inventory()

    stable = True
    env.settle(1.0)
    if not check_finite(env):
        stable = False
    max_abs_qvel_after_settle = float(np.max(np.abs(env.data.qvel))) if env.data.qvel.size else 0.0
    if max_abs_qvel_after_settle > 80.0:
        stable = False

    rows = run_scripted_trial(env, seconds=3.8, sample_every=20)
    csv_path, json_path = save_metrics(rows)
    final = asdict(rows[-1])
    metric_available = all(name in final and np.isfinite(float(final[name])) for name in REQUIRED_METRICS)
    no_legacy = len(inventory["legacy_body_names_still_remaining"]) == 0
    pass_fail = bool(stable and metric_available and no_legacy)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inventory_path = OUT_DIR / "runtime_inventory.json"
    inventory_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_md = OUT_DIR / "validation_summary.md"
    summary_md.write_text(
        "\n".join(
            [
                "# Open Panel Prototype Validation",
                "",
                f"- scene_xml: `{xml_path}`",
                f"- stable_without_nan: `{stable}`",
                f"- max_abs_qvel_after_settle: `{max_abs_qvel_after_settle:.6f}`",
                f"- metric_available: `{metric_available}`",
                f"- no_legacy_topology_names: `{no_legacy}`",
                f"- guarded_grasp_ever_accepted: `{any(row.guarded_grasp_accepted for row in rows)}`",
                f"- metrics_csv: `{csv_path}`",
                f"- summary_json: `{json_path}`",
                f"- runtime_inventory_json: `{inventory_path}`",
                f"- pass_fail: `{pass_fail}`",
            ]
        ),
        encoding="utf-8",
    )

    print(f"scene_xml={xml_path}")
    print(f"runtime_inventory={inventory_path}")
    print(f"metrics_csv={csv_path}")
    print(f"validation_summary={summary_md}")
    print(f"stable_without_nan={stable}")
    print(f"max_abs_qvel_after_settle={max_abs_qvel_after_settle:.6f}")
    print(f"metric_available={metric_available}")
    print(f"guarded_grasp_ever_accepted={any(row.guarded_grasp_accepted for row in rows)}")
    print(f"pass_fail={pass_fail}")
    return 0 if pass_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
