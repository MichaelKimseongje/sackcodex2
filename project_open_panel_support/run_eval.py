"""Open-panel support-state scripted evaluation."""

from __future__ import annotations

from pathlib import Path

from builder import write_scene_xml
from open_panel_env import OpenPanelSupportEnv, run_scripted_trial, save_metrics


def main() -> int:
    xml_path = write_scene_xml()
    env = OpenPanelSupportEnv(xml_path)
    rows = run_scripted_trial(env)
    csv_path, json_path = save_metrics(rows)
    final = rows[-1]

    print(f"scene_xml={xml_path}")
    print(f"metrics_csv={csv_path}")
    print(f"summary_json={json_path}")
    print(f"guarded_grasp_accepted={any(row.guarded_grasp_accepted for row in rows)}")
    print(f"final_insertion_depth_m={final.insertion_depth:.4f}")
    print(f"final_scoop_contact_force_N={final.scoop_contact_force:.4f}")
    print(f"final_scoop_load_transfer={final.scoop_load_transfer:.4f}")
    print(f"final_sag_index_mm={final.sag_index:.3f}")
    print(f"final_effective_com_offset_mm={final.effective_com_offset:.3f}")
    print(f"final_peel_ratio={final.peel_ratio:.3f}")
    print(f"final_support_margin_mm={final.support_margin:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
