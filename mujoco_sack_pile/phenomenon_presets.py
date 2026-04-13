from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .environment import SackPileEnv
from .scene_generator import EpisodeScene, SceneGenerator


@dataclass(frozen=True)
class PhenomenonPreset:
    """개별 현상을 구분해서 보여주기 위한 scene preset."""

    key: str
    label_ko: str
    explanation_ko: str
    target_variant: str
    pile_difficulty: str
    min_top_collapse: float = 0.0
    min_side_bulge: float = 0.0
    max_fill_ratio: float | None = None
    min_abs_roll: float = 0.0
    require_stack_level: int | None = None
    require_exposed_face: str | None = None
    required_tag: str | None = None
    run_support_demo: bool = False


PHENOMENON_PRESETS: dict[str, PhenomenonPreset] = {
    "underfilled_slack": PhenomenonPreset(
        key="underfilled_slack",
        label_ko="저충진 슬랙형",
        explanation_ko="충진량이 적어 윗부분이 꺼지고 옆면이 상대적으로 느슨해 보이는 경우",
        target_variant="low_fill_top_collapsed",
        pile_difficulty="top_exposed",
        min_top_collapse=0.034,
        max_fill_ratio=0.64,
        require_exposed_face="top",
    ),
    "top_fold_occluded": PhenomenonPreset(
        key="top_fold_occluded",
        label_ko="상단 접힘/가림형",
        explanation_ko="상단 파지면이 꺼지거나 일부 가려져 정면 top grasp가 어려운 경우",
        target_variant="low_fill_top_collapsed",
        pile_difficulty="partially_buried",
        min_top_collapse=0.030,
        max_fill_ratio=0.68,
        require_exposed_face="partial",
        required_tag="cover_contact",
    ),
    "eccentric_fill": PhenomenonPreset(
        key="eccentric_fill",
        label_ko="편심 충진형",
        explanation_ko="한쪽이 더 불룩하고 비대칭으로 눕는 경향이 큰 경우",
        target_variant="side_bulged_unstable",
        pile_difficulty="side_exposed",
        min_side_bulge=0.030,
        min_abs_roll=0.18,
        require_exposed_face="side",
    ),
    "neighbor_contact_wedge": PhenomenonPreset(
        key="neighbor_contact_wedge",
        label_ko="이웃 자루 접촉/끼임형",
        explanation_ko="주변 자루와 기대거나 끼어 삽입 경로가 제한되는 경우",
        target_variant="side_bulged_unstable",
        pile_difficulty="leaning_wedged",
        min_side_bulge=0.028,
        min_abs_roll=0.42,
        require_stack_level=1,
        required_tag="stacked_contact",
    ),
    "partial_support_sag": PhenomenonPreset(
        key="partial_support_sag",
        label_ko="부분 지지/분리 직후 처짐형",
        explanation_ko="지지 도구가 일부만 받칠 때 들어 올리며 기울거나 처지는 반응을 보는 경우",
        target_variant="low_fill_top_collapsed",
        pile_difficulty="side_exposed",
        min_top_collapse=0.030,
        max_fill_ratio=0.64,
        require_exposed_face="side",
        run_support_demo=True,
    ),
}


def list_phenomena() -> tuple[str, ...]:
    return tuple(PHENOMENON_PRESETS.keys())


def select_scene_for_phenomenon(
    *,
    base_dir: Path,
    phenomenon: str,
    base_seed: int,
    attempt_limit: int = 48,
) -> tuple[EpisodeScene, PhenomenonPreset, dict]:
    """조건에 맞는 현상 장면이 나올 때까지 seed를 바꿔가며 찾는다."""

    preset = PHENOMENON_PRESETS[phenomenon]
    generator = SceneGenerator(base_dir)

    for attempt_idx in range(attempt_limit):
        seed = base_seed + attempt_idx
        episode_id = f"{phenomenon}_seed{seed}"
        scene = generator.generate_episode(
            seed=seed,
            episode_id=episode_id,
            target_variant=preset.target_variant,
            pile_difficulty=preset.pile_difficulty,
        )
        target = next(sack for sack in scene.sacks if sack.is_target)
        if not _matches_preset(target, preset):
            continue

        summary = {
            "seed": seed,
            "target_name": target.name,
            "variant": target.variant.name,
            "pile_difficulty": target.pile_difficulty,
            "exposed_face": target.exposed_face,
            "stack_level": target.stack_level,
            "fill_ratio": target.fill_ratio,
            "top_collapse": target.top_collapse,
            "side_bulge": target.side_bulge,
            "flattening": target.flattening,
            "euler": target.euler,
            "uncertainty_tags": list(target.uncertainty_tags),
        }
        return scene, preset, summary

    raise RuntimeError(
        f"현상 preset을 만족하는 장면을 찾지 못했습니다: {phenomenon}, "
        f"base_seed={base_seed}, attempt_limit={attempt_limit}"
    )


def _matches_preset(target, preset: PhenomenonPreset) -> bool:
    """target sack이 preset 조건을 만족하는지 검사한다."""

    if target.top_collapse < preset.min_top_collapse:
        return False
    if target.side_bulge < preset.min_side_bulge:
        return False
    if preset.max_fill_ratio is not None and target.fill_ratio > preset.max_fill_ratio:
        return False
    if abs(target.euler[0]) < preset.min_abs_roll:
        return False
    if preset.require_stack_level is not None and target.stack_level != preset.require_stack_level:
        return False
    if preset.require_exposed_face is not None and target.exposed_face != preset.require_exposed_face:
        return False
    if preset.required_tag is not None and preset.required_tag not in target.uncertainty_tags:
        return False
    return True


def format_summary_lines(preset: PhenomenonPreset, summary: dict) -> list[str]:
    """콘솔 출력용 요약 문자열을 만든다."""

    euler = tuple(round(float(v), 3) for v in summary["euler"])
    return [
        f"phenomenon={preset.key}",
        f"label_ko={preset.label_ko}",
        f"description={preset.explanation_ko}",
        f"seed={summary['seed']}",
        f"target_variant={summary['variant']}",
        f"pile_difficulty={summary['pile_difficulty']}",
        f"exposed_face={summary['exposed_face']}",
        f"stack_level={summary['stack_level']}",
        f"fill_ratio={summary['fill_ratio']:.3f}",
        f"top_collapse={summary['top_collapse']:.3f}",
        f"side_bulge={summary['side_bulge']:.3f}",
        f"flattening={summary['flattening']:.3f}",
        f"target_euler={euler}",
        f"uncertainty_tags={','.join(summary['uncertainty_tags']) if summary['uncertainty_tags'] else 'none'}",
    ]


def run_partial_support_demo(env: SackPileEnv, viewer=None) -> dict:
    """부분 지지 시나리오를 짧게 실행해 처짐/기울기 proxy를 본다."""

    target_center, _ = env.target_state()
    target_side = env.target_site("side_site")
    grip_quat = SackPileEnv.euler_to_quat(np.array([0.0, -np.pi / 2.0, 0.0], dtype=np.float64))
    scoop_quat = SackPileEnv.euler_to_quat(np.array([0.0, 0.0, 0.0], dtype=np.float64))

    # 먼저 scoop를 얕게 넣고, 그 다음 side grip으로 얇은 지지 상태를 만든다.
    env.set_gripper_width(env.open_width)
    env.move_mocap_linear("scoop", target_center + np.array([-0.165, 0.0, -0.010]), scoop_quat, 140, viewer=viewer)
    env.move_mocap_linear("scoop", target_center + np.array([-0.025, 0.0, -0.004]), scoop_quat, 120, viewer=viewer)
    env.move_mocap_linear("gripper", target_side + np.array([0.005, -0.030, 0.070]), grip_quat, 150, viewer=viewer)
    env.move_mocap_linear("gripper", target_side + np.array([0.005, -0.010, 0.022]), grip_quat, 100, viewer=viewer)
    env.set_gripper_width(env.closed_width)
    env.step(120, viewer=viewer)

    env.mark_pre_lift_state()
    env.move_mocap_linear("scoop", target_center + np.array([0.020, 0.0, 0.025]), scoop_quat, 110, viewer=viewer)
    env.move_mocap_linear("gripper", target_side + np.array([0.025, 0.0, 0.090]), grip_quat, 110, viewer=viewer)
    env.step(150, viewer=viewer)

    metrics = env.finalize_metrics()
    return {
        "support_state_score": metrics.support_state_score,
        "support_success": metrics.support_success,
        "micro_lift_stability": metrics.micro_lift_stability,
        "slip_distance": metrics.slip_distance,
        "tilt_deg": metrics.tilt_deg,
        "dropped": metrics.dropped,
        "failure_tags": list(metrics.failure_tags),
    }
