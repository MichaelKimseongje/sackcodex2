from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

try:
    import imageio.v2 as imageio
except Exception as exc:  # pragma: no cover - runtime dependency check
    raise RuntimeError("imageio가 필요합니다. `pip install imageio` 후 다시 실행해 주세요.") from exc

from scenario_builder import SCENARIO_NAMES, write_scene_xml


ROOT_DIR = Path(__file__).resolve().parent
OUT_DIR = ROOT_DIR / "out" / "runtime_topology_inventory"

VISIBLE_OUTER_RE = re.compile(r"^(top_seam_band|upper_left|upper_right|lower_left|lower_right|bottom)_\d{2}$")
HIDDEN_INNER_RE = re.compile(r"^inner_(upper|lower|bottom)_\d{2}$")
SLICE_RE = re.compile(r"^slice_\d{2}_(left_end|left_mid|center|right_mid|right_end)$")
SEAM_WINDOW_SITES = (
    "site_top_seam_left",
    "site_top_seam_left_center",
    "site_top_seam_center",
    "site_top_seam_right_center",
    "site_top_seam_right",
)
BALLAST_BODIES = ("ballast_main", "ballast_aux_1", "ballast_aux_2", "ballast_aux_3")
SUPPORT_NEIGHBOR_BODIES = ("hidden_support", "neighbor_left", "neighbor_right")
END_CAP_VISUAL_BODIES = ("left_end_cap_visual", "right_end_cap_visual")

LEGACY_PATTERNS = (
    re.compile(r"^rim_ring"),
    re.compile(r"^upper_skirt"),
    re.compile(r"^lower_skirt"),
    re.compile(r"^bottom_cradle"),
    re.compile(r"^outer_upper_"),
    re.compile(r"^outer_lower_"),
    re.compile(r"^outer_mid_"),
    re.compile(r"^outer_bottom_"),
    re.compile(r"^outer_front_shell_"),
    re.compile(r"^outer_back_shell_"),
    re.compile(r"^outer_shoulder_"),
    re.compile(r"^outer_side_"),
    re.compile(r"^inner_front_load_"),
    re.compile(r"^inner_back_load_"),
    re.compile(r"^inner_bottom_load_"),
    re.compile(r"^top_seam_\d{2}$"),
    re.compile(r"^payload_"),
    re.compile(r".*rigid_core.*"),
    re.compile(r".*central_core.*"),
)
CORE_PATTERNS = (
    re.compile(r".*rigid_core.*"),
    re.compile(r".*central_core.*"),
    re.compile(r".*single_core.*"),
    re.compile(r"^payload_main$"),
    re.compile(r"^payload_aux$"),
)


@dataclass
class RuntimeInventory:
    scenario: str
    xml_path: Path
    body_names: list[str]
    joint_names: list[str]
    site_names: list[str]
    geom_names: list[str]
    visible_outer: list[str]
    hidden_inner: list[str]
    ballast: list[str]
    support_neighbor: list[str]
    end_cap_visuals: list[str]
    robots: list[str]
    slices: list[str]
    seam_windows: list[str]
    indexed_seam_sites: list[str]
    legacy_names: list[str]
    central_core_names: list[str]
    mismatch: list[str]

    @property
    def matches(self) -> bool:
        return not self.mismatch


def _names(model: mujoco.MjModel, obj_type: mujoco.mjtObj, count: int) -> list[str]:
    return [mujoco.mj_id2name(model, obj_type, i) or "" for i in range(count)]


def _is_robot_body(name: str) -> bool:
    return (
        name == "dual_robot_frame"
        or name.startswith("ur5e_2f140")
        or name.startswith("ur5e_scoop")
        or name.startswith("robotiq_2f140")
        or name == "scoop_tool"
    )


def collect_inventory(model: mujoco.MjModel, scenario: str, xml_path: Path) -> RuntimeInventory:
    body_names = _names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)
    joint_names = _names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    site_names = _names(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite)
    geom_names = _names(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom)

    visible_outer = sorted(name for name in body_names if VISIBLE_OUTER_RE.match(name))
    hidden_inner = sorted(name for name in body_names if HIDDEN_INNER_RE.match(name))
    ballast = [name for name in BALLAST_BODIES if name in body_names]
    support_neighbor = [name for name in SUPPORT_NEIGHBOR_BODIES if name in body_names]
    end_cap_visuals = [name for name in END_CAP_VISUAL_BODIES if name in body_names]
    robots = sorted(name for name in body_names if _is_robot_body(name))
    slices = sorted(name for name in body_names if SLICE_RE.match(name))
    seam_windows = [name for name in SEAM_WINDOW_SITES if name in site_names]
    indexed_seam_sites = sorted(name for name in site_names if re.match(r"^site_top_seam_0[0-4]$", name))
    legacy_names = sorted(
        name for name in body_names + joint_names + site_names + geom_names
        if any(pattern.match(name) for pattern in LEGACY_PATTERNS)
    )
    central_core_names = sorted(
        name for name in body_names + geom_names
        if any(pattern.match(name) for pattern in CORE_PATTERNS)
    )

    mismatch: list[str] = []
    if len(visible_outer) != 30:
        mismatch.append(f"visible outer shell count is {len(visible_outer)}, expected 30")
    if len(hidden_inner) != 15:
        mismatch.append(f"hidden inner shell count is {len(hidden_inner)}, expected 15")
    if len(ballast) not in (3, 4):
        mismatch.append(f"ballast count is {len(ballast)}, expected 3 or 4")
    if len(slices) != 5:
        mismatch.append(f"longitudinal slice count is {len(slices)}, expected 5")
    if len(seam_windows) != 5:
        mismatch.append(f"seam candidate window count is {len(seam_windows)}, expected 5")
    if len(end_cap_visuals) != 2:
        mismatch.append(f"end-cap visual count is {len(end_cap_visuals)}, expected 2")
    if legacy_names:
        mismatch.append(f"legacy topology names remain: {', '.join(legacy_names[:24])}")
    if central_core_names:
        mismatch.append(f"single/central rigid core-like names remain: {', '.join(central_core_names)}")

    return RuntimeInventory(
        scenario=scenario,
        xml_path=xml_path,
        body_names=body_names,
        joint_names=joint_names,
        site_names=site_names,
        geom_names=geom_names,
        visible_outer=visible_outer,
        hidden_inner=hidden_inner,
        ballast=ballast,
        support_neighbor=support_neighbor,
        end_cap_visuals=end_cap_visuals,
        robots=robots,
        slices=slices,
        seam_windows=seam_windows,
        indexed_seam_sites=indexed_seam_sites,
        legacy_names=legacy_names,
        central_core_names=central_core_names,
        mismatch=mismatch,
    )


def _write_inventory_csv(path: Path, names: list[str], category_lookup: dict[str, str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("index", "name", "category"))
        writer.writeheader()
        for idx, name in enumerate(names):
            writer.writerow({"index": idx, "name": name, "category": category_lookup.get(name, "other")})


def write_inventory_files(inv: RuntimeInventory, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    body_categories: dict[str, str] = {}
    for name in inv.visible_outer:
        body_categories[name] = "visible_outer_shell"
    for name in inv.hidden_inner:
        body_categories[name] = "hidden_inner_shell"
    for name in inv.ballast:
        body_categories[name] = "ballast"
    for name in inv.support_neighbor:
        body_categories[name] = "support_neighbor"
    for name in inv.end_cap_visuals:
        body_categories[name] = "end_cap_visual"
    for name in inv.robots:
        body_categories[name] = "robots"

    site_categories = {name: "seam_candidate_window" for name in inv.seam_windows}
    site_categories.update({name: "indexed_top_seam_site" for name in inv.indexed_seam_sites})

    _write_inventory_csv(out_dir / "actual_runtime_body_inventory.csv", inv.body_names, body_categories)
    _write_inventory_csv(out_dir / "actual_runtime_joint_inventory.csv", inv.joint_names, {})
    _write_inventory_csv(out_dir / "actual_runtime_site_inventory.csv", inv.site_names, site_categories)

    verdict = "runtime model matches the unification report" if inv.matches else "runtime model does NOT match the unification report"
    md = [
        "# Runtime Topology Inventory",
        "",
        f"- scenario: `{inv.scenario}`",
        f"- xml: `{inv.xml_path}`",
        f"- total bodies: `{len(inv.body_names)}`",
        f"- total joints: `{len(inv.joint_names)}`",
        f"- total sites: `{len(inv.site_names)}`",
        f"- total geoms: `{len(inv.geom_names)}`",
        "",
        "## Body Counts By Category",
        "",
        f"- visible outer shell: `{len(inv.visible_outer)}`",
        f"- hidden inner shell: `{len(inv.hidden_inner)}`",
        f"- ballast: `{len(inv.ballast)}`",
        f"- support / neighbor: `{len(inv.support_neighbor)}`",
        f"- end-cap visuals: `{len(inv.end_cap_visuals)}`",
        f"- robots: `{len(inv.robots)}`",
        f"- longitudinal slices: `{len(inv.slices)}`",
        f"- seam candidate windows: `{len(inv.seam_windows)}`",
        "",
        "## Runtime Body Inventory",
        "",
        ", ".join(inv.body_names),
        "",
        "## Runtime Joint Inventory",
        "",
        ", ".join(inv.joint_names),
        "",
        "## Runtime Site Inventory",
        "",
        ", ".join(inv.site_names),
        "",
        "## Legacy Body Names Still Remaining",
        "",
        ", ".join(inv.legacy_names) if inv.legacy_names else "none",
        "",
        "## Topology Mismatch Report",
        "",
        "\n".join(f"- {item}" for item in inv.mismatch) if inv.mismatch else "- none",
        "",
        f"**{verdict}**",
        "",
    ]
    (out_dir / "topology_mismatch_report.md").write_text("\n".join(md), encoding="utf-8")


def _scene_option(groups: list[int]) -> mujoco.MjvOption:
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = False
    for group in groups:
        opt.geomgroup[group] = True
    return opt


def render_runtime_views(model: mujoco.MjModel, data: mujoco.MjData, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, width=1280, height=820)
    outputs: dict[str, Path] = {}
    render_specs = {
        "outer_shell_only.png": ("front", [1]),
        "inner_shell_only.png": ("front", [2]),
        "ballast_only.png": ("front", [4]),
        "overlay.png": ("front", [0, 1, 2, 3, 4]),
        "front_view.png": ("front", [0, 1, 2, 3, 4]),
        "longitudinal_end_view.png": ("longitudinal_end", [0, 1, 2, 3, 4]),
        "top_view.png": ("top_angle", [0, 1, 2, 3, 4]),
        "side_view.png": ("side", [0, 1, 2, 3, 4]),
        "perspective_view.png": ("dual", [0, 1, 2, 3, 4]),
    }
    for filename, (camera, groups) in render_specs.items():
        renderer.update_scene(data, camera=camera, scene_option=_scene_option(groups))
        image = renderer.render()
        path = out_dir / filename
        imageio.imwrite(path, np.asarray(image))
        outputs[filename] = path
    renderer.close()
    return outputs


def print_inventory(inv: RuntimeInventory, render_outputs: dict[str, Path]) -> None:
    verdict = "runtime model matches the unification report" if inv.matches else "runtime model does NOT match the unification report"
    print(f"scenario={inv.scenario}")
    print(f"scene_xml={inv.xml_path}")
    print("")
    print("actual runtime body inventory:")
    print(", ".join(inv.body_names))
    print("")
    print("actual runtime joint inventory:")
    print(", ".join(inv.joint_names))
    print("")
    print("actual runtime site inventory:")
    print(", ".join(inv.site_names))
    print("")
    print("body counts by category:")
    print(f"  visible outer shell={len(inv.visible_outer)}")
    print(f"  hidden inner shell={len(inv.hidden_inner)}")
    print(f"  ballast={len(inv.ballast)}")
    print(f"  support / neighbor={len(inv.support_neighbor)}")
    print(f"  end-cap visuals={len(inv.end_cap_visuals)}")
    print(f"  robots={len(inv.robots)}")
    print(f"  longitudinal slices={len(inv.slices)}")
    print(f"  seam candidate windows={len(inv.seam_windows)}")
    print("")
    print(f"legacy body names still remaining: {', '.join(inv.legacy_names) if inv.legacy_names else 'none'}")
    print("")
    print("topology mismatch report:")
    if inv.mismatch:
        for item in inv.mismatch:
            print(f"  - {item}")
    else:
        print("  - none")
    print("")
    print("render outputs:")
    for name, path in render_outputs.items():
        print(f"  {name}={path}")
    print("")
    print(verdict)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect the loaded MuJoCo model and prove the runtime sack topology.")
    parser.add_argument("--scenario", choices=SCENARIO_NAMES, default="underfilled")
    parser.add_argument("--no-robots", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    xml_path = write_scene_xml(args.scenario, include_robots=not args.no_robots)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    inv = collect_inventory(model, args.scenario, xml_path)
    write_inventory_files(inv, args.out_dir)
    render_outputs = render_runtime_views(model, data, args.out_dir)
    print_inventory(inv, render_outputs)
    return 0 if inv.matches else 2


if __name__ == "__main__":
    raise SystemExit(main())
