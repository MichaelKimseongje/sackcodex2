from __future__ import annotations

from dataclasses import dataclass, replace


SCENARIO_NAMES = (
    "baseline_filled",
    "underfilled",
    "top_fold_simple",
    "top_fold_severe",
    "post_separation_sag",
)


@dataclass(frozen=True)
class ScenarioConfig:
    """shape-coupled semi-deformable sack surrogate의 scenario 파라미터."""

    name: str
    description: str
    payload_pos: tuple[float, float, float] = (0.0, 0.0, -0.080)
    payload_size: tuple[float, float, float] = (0.072, 0.046, 0.050)
    payload_mass: float = 0.46
    top_radius_scale: float = 1.0
    lower_radius_scale: float = 1.0
    upper_hull_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    lower_hull_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    shoulder_inward: float = 0.010
    belly_outward: float = 0.010
    shoulder_stiffness: float = 2.8
    shoulder_damping: float = 9.0
    belly_stiffness: float = 2.0
    belly_damping: float = 10.0
    lower_extra_drop: float = 0.0
    fold1_enabled: bool = False
    fold2_enabled: bool = False
    fold_coverage_fraction: float = 0.0
    fold_root_thickness: float = 0.008
    fold1_angle_deg: float = -58.0
    fold2_angle_deg: float = -95.0
    support_enabled: bool = False
    post_release: bool = False
    bottom_slide_range: tuple[float, float] = (-0.020, 0.015)
    bottom_slide_stiffness: float = 2.0
    bottom_slide_damping: float = 12.0


BASE_SCENARIOS: dict[str, ScenarioConfig] = {
    "baseline_filled": ScenarioConfig(
        name="baseline_filled",
        description="filled sealed sack baseline with similar upper/lower width",
        payload_pos=(0.0, 0.0, -0.075),
        payload_size=(0.076, 0.048, 0.052),
        payload_mass=0.48,
        top_radius_scale=1.0,
        lower_radius_scale=1.0,
        upper_hull_scale=(1.00, 1.00, 1.00),
        lower_hull_scale=(1.00, 1.00, 1.00),
        shoulder_inward=0.008,
        belly_outward=0.008,
    ),
    "underfilled": ScenarioConfig(
        name="underfilled",
        description="same symmetric sealed sack exterior, low payload and compliant panels create low-fill sag response",
        payload_pos=(0.0, 0.0, -0.135),
        payload_size=(0.070, 0.044, 0.035),
        payload_mass=0.34,
        top_radius_scale=1.0,
        lower_radius_scale=1.0,
        upper_hull_scale=(1.00, 1.00, 1.00),
        lower_hull_scale=(1.00, 1.00, 1.00),
        shoulder_inward=0.030,
        belly_outward=0.014,
        shoulder_stiffness=1.1,
        shoulder_damping=10.5,
        belly_stiffness=1.7,
        belly_damping=11.0,
        lower_extra_drop=0.020,
    ),
    "top_fold_simple": ScenarioConfig(
        name="top_fold_simple",
        description="one fold-root flap covers about one third of the seam while leaving exposed seam",
        fold1_enabled=True,
        fold_coverage_fraction=0.32,
        fold_root_thickness=0.011,
        fold1_angle_deg=-64.0,
    ),
    "top_fold_severe": ScenarioConfig(
        name="top_fold_severe",
        description="two bunched fold-root flaps cover over half of the seam",
        fold1_enabled=True,
        fold2_enabled=True,
        fold_coverage_fraction=0.64,
        fold_root_thickness=0.017,
        fold1_angle_deg=-76.0,
        fold2_angle_deg=-108.0,
        top_radius_scale=0.94,
        shoulder_inward=0.018,
        belly_outward=0.014,
    ),
    "post_separation_sag": ScenarioConfig(
        name="post_separation_sag",
        description="hidden support holds the bottom before release; after release bottom sling and belly panels sag",
        payload_pos=(0.0, 0.0, -0.118),
        payload_size=(0.066, 0.042, 0.040),
        payload_mass=0.42,
        top_radius_scale=0.98,
        lower_radius_scale=1.04,
        shoulder_inward=0.018,
        belly_outward=0.020,
        lower_extra_drop=0.025,
        lower_hull_scale=(1.04, 1.04, 1.15),
        support_enabled=True,
        bottom_slide_range=(-0.018, 0.010),
        bottom_slide_stiffness=3.0,
        bottom_slide_damping=16.0,
    ),
}


def available_scenarios() -> tuple[str, ...]:
    return SCENARIO_NAMES


def get_scenario(name: str, *, post_release: bool = False) -> ScenarioConfig:
    if name not in BASE_SCENARIOS:
        raise ValueError(f"unknown scenario: {name}")
    config = BASE_SCENARIOS[name]
    if name == "post_separation_sag" and post_release:
        return replace(
            config,
            support_enabled=False,
            post_release=True,
            lower_extra_drop=0.070,
            lower_hull_scale=(1.02, 1.02, 1.45),
            bottom_slide_range=(-0.080, 0.010),
            bottom_slide_stiffness=0.4,
            bottom_slide_damping=8.0,
        )
    return config
