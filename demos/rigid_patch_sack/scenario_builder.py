from __future__ import annotations

from dataclasses import dataclass, replace


SCENARIO_NAMES = (
    "underfilled",
    "top_fold_simple",
    "top_fold_severe",
    "eccentric_fill",
    "jammed_between_neighbors",
    "post_separation_sag",
)


@dataclass(frozen=True)
class ScenarioConfig:
    """sealed articulated sack surrogate v2의 scenario별 파라미터."""

    name: str
    description: str
    payload_main_pos: tuple[float, float, float] = (0.0, 0.0, -0.090)
    payload_main_size: tuple[float, float, float] = (0.064, 0.040, 0.040)
    payload_main_mass: float = 0.42
    payload_aux_enabled: bool = False
    payload_aux_pos: tuple[float, float, float] = (0.0, 0.0, -0.095)
    payload_aux_size: tuple[float, float, float] = (0.038, 0.028, 0.028)
    payload_aux_mass: float = 0.14
    shoulder_inward: float = 0.018
    belly_outward: float = 0.014
    lower_extra_drop: float = 0.000
    seam_rx_scale: float = 1.0
    seam_ry_scale: float = 1.0
    visual_upper_scale: tuple[float, float, float] = (0.92, 0.92, 0.92)
    visual_lower_scale: tuple[float, float, float] = (1.00, 1.00, 1.00)
    visual_center_shift: tuple[float, float, float] = (0.0, 0.0, 0.0)
    side_bulge_enabled: bool = False
    side_bulge_pos: tuple[float, float, float] = (0.060, 0.030, -0.055)
    side_bulge_size: tuple[float, float, float] = (0.042, 0.028, 0.040)
    body_tilt_deg: float = 0.0
    fold1_enabled: bool = False
    fold2_enabled: bool = False
    fold1_angle_deg: float = -56.0
    fold2_angle_deg: float = -88.0
    fold1_x_shift: float = -0.020
    fold2_x_shift: float = 0.030
    fold_coverage_fraction: float = 0.0
    fold_root_thickness: float = 0.009
    neighbor_enabled: bool = False
    neighbor_gap_y: float = 0.125
    support_enabled: bool = False
    post_release: bool = False


BASE_SCENARIOS: dict[str, ScenarioConfig] = {
    "underfilled": ScenarioConfig(
        name="underfilled",
        description="sealed sack silhouette with low payload, inward shoulder droop, and low CoM",
        payload_main_pos=(0.0, 0.0, -0.132),
        payload_main_size=(0.070, 0.044, 0.034),
        payload_main_mass=0.34,
        shoulder_inward=0.040,
        belly_outward=0.026,
        lower_extra_drop=0.018,
        visual_upper_scale=(0.72, 0.70, 0.86),
        visual_lower_scale=(1.10, 1.08, 1.04),
        visual_center_shift=(0.0, 0.0, -0.012),
    ),
    "top_fold_simple": ScenarioConfig(
        name="top_fold_simple",
        description="one rolled fold patch covers about one third of the seam band",
        payload_main_pos=(0.0, 0.0, -0.104),
        shoulder_inward=0.018,
        belly_outward=0.012,
        fold1_enabled=True,
        fold1_angle_deg=-64.0,
        fold_coverage_fraction=0.32,
        fold_root_thickness=0.010,
    ),
    "top_fold_severe": ScenarioConfig(
        name="top_fold_severe",
        description="two bunched fold patches cover more than half of the seam band",
        payload_main_pos=(0.0, 0.0, -0.108),
        shoulder_inward=0.024,
        belly_outward=0.016,
        lower_extra_drop=0.006,
        fold1_enabled=True,
        fold2_enabled=True,
        fold1_angle_deg=-74.0,
        fold2_angle_deg=-103.0,
        fold_coverage_fraction=0.62,
        fold_root_thickness=0.015,
    ),
    "eccentric_fill": ScenarioConfig(
        name="eccentric_fill",
        description="payloads and lower bulge are shifted to one side to make lateral CoM offset",
        payload_main_pos=(0.040, 0.018, -0.105),
        payload_main_mass=0.42,
        payload_aux_enabled=True,
        payload_aux_pos=(0.078, 0.030, -0.092),
        payload_aux_size=(0.042, 0.030, 0.030),
        payload_aux_mass=0.18,
        shoulder_inward=0.016,
        belly_outward=0.018,
        visual_center_shift=(0.020, 0.010, -0.004),
        visual_lower_scale=(1.06, 1.03, 1.02),
        side_bulge_enabled=True,
        side_bulge_pos=(0.072, 0.038, -0.056),
        body_tilt_deg=6.0,
    ),
    "jammed_between_neighbors": ScenarioConfig(
        name="jammed_between_neighbors",
        description="neighbor blockers squeeze the sealed panel sack into a narrower jammed state",
        payload_main_pos=(0.0, 0.0, -0.105),
        seam_ry_scale=0.82,
        shoulder_inward=0.025,
        belly_outward=0.006,
        visual_upper_scale=(0.94, 0.76, 0.94),
        visual_lower_scale=(1.00, 0.72, 1.00),
        neighbor_enabled=True,
        neighbor_gap_y=0.103,
    ),
    "post_separation_sag": ScenarioConfig(
        name="post_separation_sag",
        description="bottom support release makes belly panels and bottom sling drop more than the top",
        payload_main_pos=(0.0, 0.0, -0.120),
        payload_main_size=(0.062, 0.040, 0.036),
        shoulder_inward=0.026,
        belly_outward=0.018,
        lower_extra_drop=0.030,
        visual_center_shift=(0.0, 0.0, -0.018),
        visual_lower_scale=(1.02, 1.00, 1.18),
        support_enabled=True,
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
            lower_extra_drop=0.095,
            visual_center_shift=(0.0, 0.0, -0.055),
            visual_lower_scale=(1.00, 0.98, 1.48),
        )
    return config
