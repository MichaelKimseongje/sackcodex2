from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .benchmark_definition import (
    BENCHMARK_NAME,
    PILE_DIFFICULTIES,
    RESEARCH_QUESTION,
    BenchmarkCase,
    build_benchmark_case,
    infer_pile_difficulty,
    validate_pile_difficulty,
)


@dataclass
class SackVariant:
    """자루 family별 형상 범위와 proxy 파라미터를 정의한다."""

    name: str
    rgba: tuple[float, float, float, float]
    mesh_files: tuple[str, ...]
    mesh_scale_range: tuple[float, float]
    core_size: tuple[float, float, float]
    outer_proxy_size: tuple[float, float, float]
    mass_range: tuple[float, float]
    top_offset_range: tuple[float, float]
    top_collapse_range: tuple[float, float]
    side_bulge_range: tuple[float, float]
    pitch_roll_range: tuple[float, float]
    roundedness: float
    top_indentation_gain: float
    side_bulge_gain: float
    lean_pose_gain: float
    exposure_bias: float


@dataclass
class SackPlacement:
    """개별 자루의 초기 배치와 uncertainty 라벨을 담는다."""

    name: str
    variant: SackVariant
    pos: tuple[float, float, float]
    euler: tuple[float, float, float]
    exposed_face: str
    mesh_file: str
    mesh_scale: float
    total_mass: float
    top_offset: float
    top_collapse: float
    side_bulge: float
    fill_ratio: float
    flattening: float
    stack_level: int
    support_bias: float
    pile_difficulty: str
    uncertainty_tags: tuple[str, ...]
    benchmark_case_id: str
    is_target: bool = False


@dataclass
class EpisodeScene:
    """한 episode에 대응하는 benchmark case 메타데이터."""

    episode_id: str
    xml_path: Path
    seed: int
    sacks: list[SackPlacement] = field(default_factory=list)
    target_name: str = ""
    target_variant: str = ""
    target_pile_difficulty: str = ""
    benchmark_name: str = BENCHMARK_NAME
    research_question: str = RESEARCH_QUESTION
    target_case: BenchmarkCase | None = None


SACK_VARIANTS: dict[str, SackVariant] = {
    "regular_well_filled": SackVariant(
        name="regular_well_filled",
        rgba=(0.84, 0.74, 0.48, 1.0),
        mesh_files=("sack7.obj", "sack8.obj", "sack9.obj"),
        mesh_scale_range=(0.066, 0.073),
        core_size=(0.051, 0.041, 0.094),
        outer_proxy_size=(0.074, 0.058, 0.118),
        mass_range=(2.3, 2.9),
        top_offset_range=(0.040, 0.060),
        top_collapse_range=(0.002, 0.010),
        side_bulge_range=(0.004, 0.014),
        pitch_roll_range=(0.10, 0.24),
        roundedness=0.94,
        top_indentation_gain=0.30,
        side_bulge_gain=0.30,
        lean_pose_gain=0.28,
        exposure_bias=0.52,
    ),
    "low_fill_top_collapsed": SackVariant(
        name="low_fill_top_collapsed",
        rgba=(0.82, 0.67, 0.44, 1.0),
        mesh_files=("sack.obj", "sack2.obj", "sack10.obj"),
        mesh_scale_range=(0.066, 0.073),
        core_size=(0.046, 0.036, 0.070),
        outer_proxy_size=(0.075, 0.060, 0.116),
        mass_range=(1.4, 2.0),
        top_offset_range=(0.014, 0.030),
        top_collapse_range=(0.028, 0.052),
        side_bulge_range=(0.004, 0.016),
        pitch_roll_range=(0.16, 0.34),
        roundedness=1.06,
        top_indentation_gain=1.20,
        side_bulge_gain=0.48,
        lean_pose_gain=0.36,
        exposure_bias=0.73,
    ),
    "side_bulged_unstable": SackVariant(
        name="side_bulged_unstable",
        rgba=(0.76, 0.60, 0.40, 1.0),
        mesh_files=("sack3.obj", "sack6Apply.obj", "sack9.obj"),
        mesh_scale_range=(0.067, 0.075),
        core_size=(0.044, 0.034, 0.082),
        outer_proxy_size=(0.078, 0.066, 0.114),
        mass_range=(1.8, 2.5),
        top_offset_range=(0.028, 0.048),
        top_collapse_range=(0.010, 0.022),
        side_bulge_range=(0.026, 0.048),
        pitch_roll_range=(0.32, 0.60),
        roundedness=1.12,
        top_indentation_gain=0.44,
        side_bulge_gain=1.24,
        lean_pose_gain=1.06,
        exposure_bias=0.88,
    ),
}


class SceneGenerator:
    """형상/적재 불확실성 benchmark case와 대응 MJCF를 생성한다."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.generated_dir = self.base_dir / "mujoco_sack_pile" / "generated"
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        self.object_dir = self.base_dir / "object"

    def generate_episode(
        self,
        seed: int,
        episode_id: str,
        sack_count: int | None = None,
        target_variant: str | None = None,
        pile_difficulty: str | None = None,
    ) -> EpisodeScene:
        rng = random.Random(seed)
        count = sack_count if sack_count is not None else rng.randint(3, 6)
        requested_variant = target_variant or rng.choice(tuple(SACK_VARIANTS.keys()))
        requested_difficulty = validate_pile_difficulty(pile_difficulty or rng.choice(PILE_DIFFICULTIES))
        sacks = self._sample_placements(
            rng,
            count,
            target_variant_name=requested_variant,
            target_pile_difficulty=requested_difficulty,
        )
        target = next(sack for sack in sacks if sack.is_target)
        target_case = build_benchmark_case(
            shape_family=target.variant.name,
            pile_difficulty=target.pile_difficulty,
            top_collapse=target.top_collapse,
            side_bulge=target.side_bulge,
            tilt_mag=max(abs(target.euler[0]), abs(target.euler[1])),
        )

        xml_path = self.generated_dir / f"{episode_id}.xml"
        xml_text = self._build_xml(episode_id=episode_id, sacks=sacks)
        xml_path.write_text(xml_text, encoding="utf-8")
        return EpisodeScene(
            episode_id=episode_id,
            xml_path=xml_path,
            seed=seed,
            sacks=sacks,
            target_name=target.name,
            target_variant=target.variant.name,
            target_pile_difficulty=target.pile_difficulty,
            target_case=target_case,
        )

    def _sample_placements(
        self,
        rng: random.Random,
        count: int,
        target_variant_name: str,
        target_pile_difficulty: str,
    ) -> list[SackPlacement]:
        """요청한 family와 difficulty를 만족하는 target을 먼저 배치한다."""

        placements: list[SackPlacement] = []
        target_variant = SACK_VARIANTS[target_variant_name]
        target = self._sample_target_placement(rng, target_variant, target_pile_difficulty)
        target.is_target = True
        placements.append(target)

        family_cycle = list(SACK_VARIANTS.values())
        rng.shuffle(family_cycle)
        for idx in range(1, count):
            if target_pile_difficulty in {"partially_buried", "leaning_wedged"} and idx <= 2:
                placement = self._sample_context_near_target(
                    rng,
                    idx=idx,
                    placements=placements,
                    target=target,
                    target_pile_difficulty=target_pile_difficulty,
                )
            else:
                variant = family_cycle[(idx - 1) % len(family_cycle)]
                if idx > len(family_cycle):
                    variant = rng.choice(tuple(SACK_VARIANTS.values()))
                placement = self._sample_context_placement(rng, idx=idx, placements=placements, variant=variant)
            placements.append(placement)
        return placements

    def _sample_target_placement(
        self,
        rng: random.Random,
        variant: SackVariant,
        pile_difficulty: str,
    ) -> SackPlacement:
        """요청한 pile difficulty에 맞는 target sack을 만든다."""

        pile_difficulty = validate_pile_difficulty(pile_difficulty)
        x = 0.62 + rng.uniform(-0.035, 0.035)
        y = rng.uniform(-0.050, 0.050)
        stack_level = 0
        exposed_face = "top"
        yaw = rng.uniform(-math.pi, math.pi)
        roll = rng.uniform(-0.12, 0.12)
        pitch = rng.uniform(-0.12, 0.12)
        z = 0.118 + rng.uniform(-0.004, 0.008)

        if pile_difficulty == "top_exposed":
            exposed_face = "top"
            y = rng.uniform(-0.070, 0.070)
            z = 0.118 + rng.uniform(-0.004, 0.010)
            roll = rng.uniform(-0.16, 0.16)
            pitch = rng.uniform(-0.16, 0.16)
        elif pile_difficulty == "side_exposed":
            exposed_face = "side"
            y = rng.choice((-1.0, 1.0)) * rng.uniform(0.080, 0.170)
            z = 0.114 + rng.uniform(-0.004, 0.008)
            roll = rng.uniform(-0.22, 0.22)
            pitch = rng.uniform(-0.18, 0.18)
        elif pile_difficulty == "partially_buried":
            exposed_face = "partial"
            y = rng.uniform(-0.040, 0.040)
            z = 0.108 + rng.uniform(-0.004, 0.004)
            roll = rng.uniform(-0.20, 0.20)
            pitch = rng.uniform(-0.20, 0.20)
        elif pile_difficulty == "leaning_wedged":
            stack_level = 1
            exposed_face = "top" if rng.random() < 0.60 else "side"
            z = 0.152 + rng.uniform(-0.006, 0.012)
            lean_sign = rng.choice((-1.0, 1.0))
            roll = lean_sign * rng.uniform(0.36, 0.55)
            pitch = rng.uniform(-0.22, 0.22)
            y = lean_sign * rng.uniform(0.020, 0.085)

        top_collapse = self._sample_top_collapse(rng, variant)
        side_bulge = self._sample_side_bulge(rng, variant)
        fill_ratio = self._sample_fill_ratio(rng, variant)
        flattening = self._sample_flattening(rng, variant, stack_level)
        mesh_file = rng.choice(variant.mesh_files)
        mesh_scale = rng.uniform(*variant.mesh_scale_range) * (0.96 + 0.10 * fill_ratio)
        total_mass = rng.uniform(*variant.mass_range)
        top_offset = rng.uniform(*variant.top_offset_range)
        roll, pitch = self._apply_family_pose_bias(
            rng,
            variant,
            roll=roll,
            pitch=pitch,
            pile_difficulty=pile_difficulty,
        )
        support_bias = rng.uniform(-0.016, 0.016)
        benchmark_case = build_benchmark_case(
            shape_family=variant.name,
            pile_difficulty=pile_difficulty,
            top_collapse=top_collapse,
            side_bulge=side_bulge,
            tilt_mag=max(abs(roll), abs(pitch)),
        )

        return SackPlacement(
            name="sack_0",
            variant=variant,
            pos=(x, y, z),
            euler=(roll, pitch, yaw),
            exposed_face=exposed_face,
            mesh_file=mesh_file,
            mesh_scale=mesh_scale,
            total_mass=total_mass,
            top_offset=top_offset,
            top_collapse=top_collapse,
            side_bulge=side_bulge,
            fill_ratio=fill_ratio,
            flattening=flattening,
            stack_level=stack_level,
            support_bias=support_bias,
            pile_difficulty=benchmark_case.pile_difficulty,
            uncertainty_tags=benchmark_case.tags,
            benchmark_case_id=benchmark_case.case_id,
        )

    def _sample_context_near_target(
        self,
        rng: random.Random,
        idx: int,
        placements: list[SackPlacement],
        target: SackPlacement,
        target_pile_difficulty: str,
    ) -> SackPlacement:
        """부분 매몰과 leaning/wedged 장면은 주변 자루를 의도적으로 가깝게 둔다."""

        variant = rng.choice(tuple(SACK_VARIANTS.values()))
        support_bias = rng.uniform(-0.016, 0.016)
        fill_ratio = self._sample_fill_ratio(rng, variant)
        top_collapse = self._sample_top_collapse(rng, variant)
        side_bulge = self._sample_side_bulge(rng, variant)
        flattening = self._sample_flattening(rng, variant, stack_level=0)
        total_mass = rng.uniform(*variant.mass_range)
        top_offset = rng.uniform(*variant.top_offset_range)
        mesh_file = rng.choice(variant.mesh_files)
        mesh_scale = rng.uniform(*variant.mesh_scale_range) * (0.96 + 0.10 * fill_ratio)

        if target_pile_difficulty == "partially_buried":
            side_sign = rng.choice((-1.0, 1.0))
            offsets = (
                (0.082, 0.050 * side_sign, 0.008),
                (-0.070, -0.055 * side_sign, 0.010),
            )
            exposed_face = "side" if idx == 1 else "top"
            stack_level = 0
            roll = rng.uniform(-0.18, 0.18)
            pitch = rng.uniform(-0.18, 0.18)
        else:
            lean_sign = 1.0 if target.pos[1] >= 0.0 else -1.0
            offsets = (
                (-0.065, 0.050 * lean_sign, -0.032),
                (0.020, -0.082 * lean_sign, -0.024),
            )
            exposed_face = "side"
            stack_level = 0
            roll = rng.uniform(-0.22, 0.22)
            pitch = rng.uniform(-0.20, 0.20)
        roll, pitch = self._apply_family_pose_bias(
            rng,
            variant,
            roll=roll,
            pitch=pitch,
            pile_difficulty=target_pile_difficulty,
        )

        offset = offsets[min(idx - 1, len(offsets) - 1)]
        x = target.pos[0] + offset[0] + rng.uniform(-0.012, 0.012)
        y = target.pos[1] + offset[1] + rng.uniform(-0.012, 0.012)
        z = max(0.106, target.pos[2] + offset[2] + rng.uniform(-0.004, 0.004))
        yaw = rng.uniform(-math.pi, math.pi)
        pile_difficulty = infer_pile_difficulty(exposed_face, stack_level, max(abs(roll), abs(pitch)))
        benchmark_case = build_benchmark_case(
            shape_family=variant.name,
            pile_difficulty=pile_difficulty,
            top_collapse=top_collapse,
            side_bulge=side_bulge,
            tilt_mag=max(abs(roll), abs(pitch)),
        )

        return SackPlacement(
            name=f"sack_{idx}",
            variant=variant,
            pos=(x, y, z),
            euler=(roll, pitch, yaw),
            exposed_face=exposed_face,
            mesh_file=mesh_file,
            mesh_scale=mesh_scale,
            total_mass=total_mass,
            top_offset=top_offset,
            top_collapse=top_collapse,
            side_bulge=side_bulge,
            fill_ratio=fill_ratio,
            flattening=flattening,
            stack_level=stack_level,
            support_bias=support_bias,
            pile_difficulty=benchmark_case.pile_difficulty,
            uncertainty_tags=benchmark_case.tags,
            benchmark_case_id=benchmark_case.case_id,
        )

    def _sample_context_placement(
        self,
        rng: random.Random,
        idx: int,
        placements: list[SackPlacement],
        variant: SackVariant,
    ) -> SackPlacement:
        """나머지 자루는 랜덤 pile context로 배치한다."""

        for _ in range(120):
            radial = rng.uniform(0.04, 0.22)
            angle = rng.uniform(-math.pi, math.pi)
            x = 0.62 + radial * math.cos(angle)
            y = 0.00 + radial * math.sin(angle)
            stack_level = 1 if rng.random() < min(0.16 + 0.03 * idx, 0.32) else 0
            z = 0.107 + rng.uniform(0.000, 0.018) + 0.032 * stack_level
            min_dist = 0.082 if stack_level == 0 else 0.070
            if any(self._xy_distance((x, y), (p.pos[0], p.pos[1])) < min_dist for p in placements):
                continue

            yaw = rng.uniform(-math.pi, math.pi)
            tilt_mag = rng.uniform(*variant.pitch_roll_range)
            roll = rng.uniform(-tilt_mag, tilt_mag)
            pitch = rng.uniform(-tilt_mag, tilt_mag)
            if stack_level > 0:
                roll += rng.uniform(-0.12, 0.12)
                pitch += rng.uniform(-0.12, 0.12)
            roll, pitch = self._apply_family_pose_bias(
                rng,
                variant,
                roll=roll,
                pitch=pitch,
                pile_difficulty="leaning_wedged" if stack_level > 0 else "side_exposed",
            )

            top_collapse = self._sample_top_collapse(rng, variant)
            side_bulge = self._sample_side_bulge(rng, variant)
            fill_ratio = self._sample_fill_ratio(rng, variant)
            flattening = self._sample_flattening(rng, variant, stack_level)
            exposed_face = self._pick_context_face(rng, stack_level, abs(y))
            pile_difficulty = infer_pile_difficulty(exposed_face, stack_level, max(abs(roll), abs(pitch)))
            benchmark_case = build_benchmark_case(
                shape_family=variant.name,
                pile_difficulty=pile_difficulty,
                top_collapse=top_collapse,
                side_bulge=side_bulge,
                tilt_mag=max(abs(roll), abs(pitch)),
            )

            return SackPlacement(
                name=f"sack_{idx}",
                variant=variant,
                pos=(x, y, z),
                euler=(roll, pitch, yaw),
                exposed_face=exposed_face,
                mesh_file=rng.choice(variant.mesh_files),
                mesh_scale=rng.uniform(*variant.mesh_scale_range) * (0.96 + 0.10 * fill_ratio),
                total_mass=rng.uniform(*variant.mass_range),
                top_offset=rng.uniform(*variant.top_offset_range),
                top_collapse=top_collapse,
                side_bulge=side_bulge,
                fill_ratio=fill_ratio,
                flattening=flattening,
                stack_level=stack_level,
                support_bias=rng.uniform(-0.016, 0.016),
                pile_difficulty=benchmark_case.pile_difficulty,
                uncertainty_tags=benchmark_case.tags,
                benchmark_case_id=benchmark_case.case_id,
            )

        # 랜덤 배치가 계속 막히면 안전한 fallback으로 넣는다.
        pile_difficulty = "side_exposed"
        fallback_case = build_benchmark_case(
            shape_family=variant.name,
            pile_difficulty=pile_difficulty,
            top_collapse=sum(variant.top_collapse_range) * 0.5,
            side_bulge=sum(variant.side_bulge_range) * 0.5,
            tilt_mag=0.15,
        )
        return SackPlacement(
            name=f"sack_{idx}",
            variant=variant,
            pos=(0.48 + 0.06 * idx, -0.12 + 0.08 * (idx % 3), 0.122 + 0.018 * (idx // 3)),
            euler=(0.10 * (-1) ** idx, -0.06 * idx, 0.35 * idx),
            exposed_face="side",
            mesh_file=rng.choice(variant.mesh_files),
            mesh_scale=sum(variant.mesh_scale_range) * 0.5,
            total_mass=sum(variant.mass_range) * 0.5,
            top_offset=sum(variant.top_offset_range) * 0.5,
            top_collapse=sum(variant.top_collapse_range) * 0.5,
            side_bulge=sum(variant.side_bulge_range) * 0.5,
            fill_ratio=self._sample_fill_ratio(rng, variant),
            flattening=0.92,
            stack_level=0,
            support_bias=0.0,
            pile_difficulty=fallback_case.pile_difficulty,
            uncertainty_tags=fallback_case.tags + ("fallback_case",),
            benchmark_case_id=fallback_case.case_id,
        )

    @staticmethod
    def _sample_fill_ratio(rng: random.Random, variant: SackVariant) -> float:
        """family별로 fill ratio를 다르게 샘플링한다."""

        if variant.name == "low_fill_top_collapsed":
            return rng.uniform(0.42, 0.68)
        if variant.name == "side_bulged_unstable":
            return rng.uniform(0.56, 0.84)
        return rng.uniform(0.82, 1.05)

    @staticmethod
    def _sample_top_collapse(rng: random.Random, variant: SackVariant) -> float:
        """family별로 상단 함몰 정도를 더 눈에 띄게 샘플링한다."""

        low, high = variant.top_collapse_range
        if variant.name == "low_fill_top_collapsed":
            mid = low + 0.42 * (high - low)
            return rng.uniform(mid, high)
        if variant.name == "side_bulged_unstable":
            return rng.uniform(low, low + 0.72 * (high - low))
        return rng.uniform(low, high)

    @staticmethod
    def _sample_side_bulge(rng: random.Random, variant: SackVariant) -> float:
        """family별로 측면 bulge를 다르게 강조한다."""

        low, high = variant.side_bulge_range
        if variant.name == "side_bulged_unstable":
            mid = low + 0.35 * (high - low)
            return rng.uniform(mid, high)
        if variant.name == "regular_well_filled":
            return rng.uniform(low, low + 0.75 * (high - low))
        return rng.uniform(low, high)

    @staticmethod
    def _sample_flattening(rng: random.Random, variant: SackVariant, stack_level: int) -> float:
        """family별로 바닥에 눌린 정도를 다르게 준다."""

        if variant.name == "low_fill_top_collapsed":
            base = rng.uniform(0.72, 0.90)
        elif variant.name == "side_bulged_unstable":
            base = rng.uniform(0.82, 0.97)
        else:
            base = rng.uniform(0.90, 1.04)
        return base - 0.07 * stack_level

    @staticmethod
    def _apply_family_pose_bias(
        rng: random.Random,
        variant: SackVariant,
        roll: float,
        pitch: float,
        pile_difficulty: str,
    ) -> tuple[float, float]:
        """family 특성에 맞는 기울기 bias를 더해 형상 차이를 키운다."""

        if variant.name == "low_fill_top_collapsed":
            pitch += rng.choice((-1.0, 1.0)) * rng.uniform(0.03, 0.09) * variant.lean_pose_gain
        elif variant.name == "side_bulged_unstable":
            if pile_difficulty == "leaning_wedged" and abs(roll) > 1e-4:
                lean_sign = 1.0 if roll >= 0.0 else -1.0
            else:
                lean_sign = rng.choice((-1.0, 1.0))
            extra_roll = rng.uniform(0.12, 0.22) * variant.lean_pose_gain
            if pile_difficulty == "leaning_wedged":
                extra_roll += rng.uniform(0.10, 0.18)
            roll += lean_sign * extra_roll
            pitch += rng.uniform(-0.10, 0.10) * max(0.6, variant.lean_pose_gain)
            if pile_difficulty == "leaning_wedged" and abs(roll) < 0.42:
                roll = lean_sign * rng.uniform(0.42, 0.62)
        return roll, pitch

    @staticmethod
    def _pick_context_face(rng: random.Random, stack_level: int, lateral_bias: float) -> str:
        """비타깃 자루의 노출 방향을 대략적으로 정한다."""

        if stack_level > 0 and rng.random() < 0.50:
            return "top"
        if lateral_bias > 0.15:
            return "side"
        sample = rng.random()
        if sample < 0.32:
            return "side"
        if sample < 0.66:
            return "top"
        return "partial"

    @staticmethod
    def _xy_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return math.sqrt(dx * dx + dy * dy)

    def _build_xml(self, episode_id: str, sacks: Iterable[SackPlacement]) -> str:
        sacks = list(sacks)
        mesh_assets = "\n".join(self._mesh_asset_xml(sack) for sack in sacks)
        sack_bodies = "\n".join(self._sack_body_xml(sack) for sack in sacks)
        excludes = "\n".join(self._sack_contact_excludes(sacks))
        meshdir = self.object_dir.as_posix()
        return f"""<mujoco model="{episode_id}">
  <compiler angle="radian" coordinate="local" inertiafromgeom="true" meshdir="{meshdir}"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" iterations="100" tolerance="1e-8"/>
  <size memory="768M" nconmax="12000"/>

  <visual>
    <global azimuth="128" elevation="-20" offwidth="1280" offheight="720"/>
    <headlight ambient="0.34 0.34 0.34" diffuse="0.78 0.78 0.78" specular="0.15 0.15 0.15"/>
    <rgba haze="0.11 0.13 0.18 1"/>
  </visual>

  <default>
    <geom condim="4" friction="1.15 0.03 0.01" margin="0.003" solimp="0.93 0.98 0.003" solref="0.012 1"/>
    <joint damping="1.2" armature="0.002"/>
    <position kp="2500" forcelimited="true" forcerange="-420 420"/>
    <motor ctrllimited="true"/>
  </default>

  <asset>
{mesh_assets}
  </asset>

  <worldbody>
    <light name="key" pos="0.85 0.15 1.8" dir="-0.2 -0.1 -1" directional="true"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.93 0.93 0.93 1"/>
    <geom name="work_pad" type="box" pos="0.62 0 0.012" size="0.30 0.30 0.012" rgba="0.60 0.61 0.62 1" friction="1.4 0.05 0.01"/>
    <geom name="corral_front" type="box" pos="0.93 0.00 0.08" size="0.010 0.30 0.08" rgba="0.55 0.55 0.58 0.22" friction="1.0 0.03 0.01"/>
    <geom name="corral_back" type="box" pos="0.31 0.00 0.08" size="0.010 0.30 0.08" rgba="0.55 0.55 0.58 0.18" friction="1.0 0.03 0.01"/>
    <geom name="corral_left" type="box" pos="0.62 0.30 0.08" size="0.30 0.010 0.08" rgba="0.55 0.55 0.58 0.14" friction="1.0 0.03 0.01"/>
    <geom name="corral_right" type="box" pos="0.62 -0.30 0.08" size="0.30 0.010 0.08" rgba="0.55 0.55 0.58 0.14" friction="1.0 0.03 0.01"/>
    <camera name="overview" pos="1.22 0.00 0.92" xyaxes="0 1 0 -0.46 0 0.88"/>

    <body name="gripper_mocap" mocap="true" pos="0.36 -0.33 0.30" quat="0.707107 0 -0.707107 0"/>
    <body name="scoop_mocap" mocap="true" pos="0.42 0.28 0.20" quat="1 0 0 0"/>

    <body name="gripper_ctrl_base" pos="0.36 -0.33 0.30">
      <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>
      <joint name="gripper_ctrl_x" type="slide" axis="1 0 0" limited="true" range="0.20 1.00"/>
      <body name="gripper_ctrl_y_body">
        <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>
        <joint name="gripper_ctrl_y" type="slide" axis="0 1 0" limited="true" range="-0.60 0.60"/>
        <body name="gripper_ctrl_z_body">
          <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>
          <joint name="gripper_ctrl_z" type="slide" axis="0 0 1" limited="true" range="0.04 0.70"/>
          <body name="gripper_ctrl_yaw_body">
            <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>
            <joint name="gripper_ctrl_yaw" type="hinge" axis="0 0 1" limited="true" range="-3.1416 3.1416"/>
            <body name="gripper_ctrl_pitch_body">
              <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>
              <joint name="gripper_ctrl_pitch" type="hinge" axis="0 1 0" limited="true" range="-3.1416 3.1416"/>
              <body name="gripper_ctrl_roll_body">
                <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>
                <joint name="gripper_ctrl_roll" type="hinge" axis="1 0 0" limited="true" range="-3.1416 3.1416"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <body name="scoop_ctrl_base" pos="0.42 0.28 0.20">
      <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>
      <joint name="scoop_ctrl_x" type="slide" axis="1 0 0" limited="true" range="0.20 1.00"/>
      <body name="scoop_ctrl_y_body">
        <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>
        <joint name="scoop_ctrl_y" type="slide" axis="0 1 0" limited="true" range="-0.60 0.60"/>
        <body name="scoop_ctrl_z_body">
          <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>
          <joint name="scoop_ctrl_z" type="slide" axis="0 0 1" limited="true" range="0.04 0.70"/>
          <body name="scoop_ctrl_yaw_body">
            <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>
            <joint name="scoop_ctrl_yaw" type="hinge" axis="0 0 1" limited="true" range="-3.1416 3.1416"/>
            <body name="scoop_ctrl_pitch_body">
              <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>
              <joint name="scoop_ctrl_pitch" type="hinge" axis="0 1 0" limited="true" range="-3.1416 3.1416"/>
              <body name="scoop_ctrl_roll_body">
                <inertial pos="0 0 0" mass="0.001" diaginertia="1e-6 1e-6 1e-6"/>
                <joint name="scoop_ctrl_roll" type="hinge" axis="1 0 0" limited="true" range="-3.1416 3.1416"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <body name="gripper_tool" pos="0.36 -0.33 0.30" quat="0.707107 0 -0.707107 0">
      <freejoint name="gripper_tool_free"/>
      <inertial pos="0 0 0.04" mass="0.08" diaginertia="0.0002 0.0002 0.0002"/>
      <geom name="gripper_palm" type="box" pos="0 0 0.030" size="0.020 0.026 0.030" rgba="0.14 0.14 0.14 1"/>
      <body name="left_finger_body" pos="0 0 0.072">
        <joint name="left_finger_slide" type="slide" axis="0 1 0" limited="true" range="0.010 0.040" damping="12"/>
        <geom name="left_finger_geom" type="box" pos="0 0.020 0" size="0.012 0.006 0.040" rgba="0.10 0.10 0.10 1" friction="2.2 0.05 0.01"/>
      </body>
      <body name="right_finger_body" pos="0 0 0.072">
        <joint name="right_finger_slide" type="slide" axis="0 -1 0" limited="true" range="0.010 0.040" damping="12"/>
        <geom name="right_finger_geom" type="box" pos="0 -0.020 0" size="0.012 0.006 0.040" rgba="0.10 0.10 0.10 1" friction="2.2 0.05 0.01"/>
      </body>
      <site name="gripper_pinch_site" pos="0 0 0.074" size="0.006" rgba="1 0 0 1"/>
    </body>

    <body name="scoop_tool" pos="0.42 0.28 0.20" quat="1 0 0 0">
      <freejoint name="scoop_tool_free"/>
      <inertial pos="0 0 0.02" mass="0.10" diaginertia="0.0003 0.0003 0.0003"/>
      <geom name="scoop_plate" type="box" pos="0.062 0 0" size="0.062 0.085 0.004" rgba="0.28 0.31 0.35 1" friction="1.7 0.05 0.01"/>
      <geom name="scoop_lip" type="box" pos="0.126 0 0.014" size="0.004 0.085 0.014" rgba="0.24 0.27 0.30 1" friction="1.7 0.05 0.01"/>
      <geom name="scoop_left_rail" type="box" pos="0.062 0.082 0.018" size="0.056 0.004 0.018" rgba="0.24 0.27 0.30 1"/>
      <geom name="scoop_right_rail" type="box" pos="0.062 -0.082 0.018" size="0.056 0.004 0.018" rgba="0.24 0.27 0.30 1"/>
      <geom name="scoop_backstop" type="box" pos="0.000 0 0.022" size="0.006 0.082 0.022" rgba="0.24 0.27 0.30 1"/>
      <site name="scoop_tip_site" pos="0.126 0 0.005" size="0.005" rgba="0 0 1 1"/>
      <site name="scoop_center_site" pos="0.060 0 0.020" size="0.005" rgba="0 0.7 1 1"/>
      <site name="score_overlay_site" pos="0.030 0 0.080" size="0.008" rgba="0 1 0 0.8"/>
    </body>

{sack_bodies}
  </worldbody>

  <equality>
    <weld name="gripper_follow" body1="gripper_mocap" body2="gripper_tool" torquescale="400" solref="0.002 1" solimp="0.99 0.999 0.0001"/>
    <weld name="scoop_follow" body1="scoop_mocap" body2="scoop_tool" torquescale="500" solref="0.002 1" solimp="0.99 0.999 0.0001"/>
  </equality>

  <actuator>
    <position name="gripper_ctrl_x_act" joint="gripper_ctrl_x" ctrlrange="0.20 1.00" kp="3500" forcerange="-300 300"/>
    <position name="gripper_ctrl_y_act" joint="gripper_ctrl_y" ctrlrange="-0.60 0.60" kp="3500" forcerange="-300 300"/>
    <position name="gripper_ctrl_z_act" joint="gripper_ctrl_z" ctrlrange="0.04 0.70" kp="3500" forcerange="-300 300"/>
    <position name="gripper_ctrl_yaw_act" joint="gripper_ctrl_yaw" ctrlrange="-3.1416 3.1416" kp="1800" forcerange="-250 250"/>
    <position name="gripper_ctrl_pitch_act" joint="gripper_ctrl_pitch" ctrlrange="-3.1416 3.1416" kp="1800" forcerange="-250 250"/>
    <position name="gripper_ctrl_roll_act" joint="gripper_ctrl_roll" ctrlrange="-3.1416 3.1416" kp="1800" forcerange="-250 250"/>
    <position name="scoop_ctrl_x_act" joint="scoop_ctrl_x" ctrlrange="0.20 1.00" kp="3500" forcerange="-300 300"/>
    <position name="scoop_ctrl_y_act" joint="scoop_ctrl_y" ctrlrange="-0.60 0.60" kp="3500" forcerange="-300 300"/>
    <position name="scoop_ctrl_z_act" joint="scoop_ctrl_z" ctrlrange="0.04 0.70" kp="3500" forcerange="-300 300"/>
    <position name="scoop_ctrl_yaw_act" joint="scoop_ctrl_yaw" ctrlrange="-3.1416 3.1416" kp="1800" forcerange="-250 250"/>
    <position name="scoop_ctrl_pitch_act" joint="scoop_ctrl_pitch" ctrlrange="-3.1416 3.1416" kp="1800" forcerange="-250 250"/>
    <position name="scoop_ctrl_roll_act" joint="scoop_ctrl_roll" ctrlrange="-3.1416 3.1416" kp="1800" forcerange="-250 250"/>
    <position name="left_finger_act" joint="left_finger_slide" ctrlrange="0.010 0.040" kp="2800" forcerange="-220 220"/>
    <position name="right_finger_act" joint="right_finger_slide" ctrlrange="0.010 0.040" kp="2800" forcerange="-220 220"/>
  </actuator>

  <contact>
{excludes}
  </contact>
</mujoco>
"""

    def _mesh_asset_xml(self, sack: SackPlacement) -> str:
        return f'    <mesh name="{sack.name}_visual_mesh" file="{sack.mesh_file}" scale="{sack.mesh_scale:.4f} {sack.mesh_scale:.4f} {sack.mesh_scale * (0.92 + 0.18 * sack.fill_ratio):.4f}"/>'

    def _sack_contact_excludes(self, sacks: Iterable[SackPlacement]) -> Iterable[str]:
        for sack in sacks:
            yield f'    <exclude body1="{sack.name}_top_grasp_proxy_body" body2="{sack.name}_left_side_proxy_body"/>'
            yield f'    <exclude body1="{sack.name}_top_grasp_proxy_body" body2="{sack.name}_right_side_proxy_body"/>'

    def _sack_body_xml(self, sack: SackPlacement) -> str:
        v = sack.variant
        roll, pitch, yaw = sack.euler
        r, g, b, _ = v.rgba

        core_x = v.core_size[0] * (0.92 + 0.10 * sack.fill_ratio)
        core_y = v.core_size[1] * (0.88 + 0.20 * sack.fill_ratio)
        core_z = v.core_size[2] * (0.82 + 0.14 * sack.fill_ratio)
        outer_x = v.outer_proxy_size[0] * (0.96 + 0.08 * sack.fill_ratio)
        outer_y = v.outer_proxy_size[1] * (0.92 + 0.30 * sack.fill_ratio + 0.80 * sack.side_bulge)
        outer_z = v.outer_proxy_size[2] * sack.flattening
        top_indent = sack.top_collapse * (1.15 + v.top_indentation_gain)
        top_z = sack.top_offset + outer_z * 0.64 - top_indent

        side_sign = -1.0 if sack.exposed_face == "side" else 1.0
        front_lean = 0.014 if sack.exposed_face == "partial" else 0.0
        lean_shift_x = front_lean + max(-0.020, min(0.020, pitch * 0.028 * max(0.8, v.lean_pose_gain)))
        lean_shift_y = max(-0.026, min(0.026, roll * 0.024 * max(0.7, v.lean_pose_gain)))
        dominant_bulge = sack.side_bulge * (1.10 + v.side_bulge_gain)
        mass_core = 0.42 * sack.total_mass
        mass_lobes = 0.18 * sack.total_mass
        mass_proxy_panel = 0.08 * sack.total_mass
        rgba_text = f"{r:.3f} {g:.3f} {b:.3f}"
        shell_x = outer_x * (0.84 + 0.06 * v.roundedness)
        shell_y = outer_y * (0.76 + 0.08 * v.roundedness)
        shell_z = outer_z * (0.80 + 0.05 * v.roundedness)
        top_band_x = max(outer_x - 0.022, 0.028)
        top_band_y = max(outer_y - 0.030, 0.020)
        side_half_x = max(outer_x - 0.016, 0.028)
        side_half_y = max(0.012 + 0.45 * dominant_bulge, 0.012)
        side_half_z = max(outer_z - 0.024, 0.030)
        face_half_x = max(0.014 + 0.30 * dominant_bulge, 0.014)
        face_half_y = max(outer_y - 0.012, 0.022)
        face_half_z = max(outer_z - 0.028, 0.028)
        bottom_half_z = max(0.012 + 0.006 * (1.0 - sack.fill_ratio), 0.010)
        bulge_major_offset = 0.020 + 0.80 * dominant_bulge
        bulge_minor_offset = 0.014 + 0.35 * dominant_bulge
        top_crown_z = top_z + 0.012 - 0.22 * top_indent
        top_crown_radius = max(0.014 + 0.006 * v.roundedness, 0.014)

        return f"""
    <body name="{sack.name}" pos="{sack.pos[0]:.4f} {sack.pos[1]:.4f} {sack.pos[2]:.4f}" euler="{roll:.4f} {pitch:.4f} {yaw:.4f}">
      <freejoint name="{sack.name}_free"/>

      <geom name="{sack.name}_visual" type="mesh" mesh="{sack.name}_visual_mesh" rgba="{rgba_text} 0.92" contype="0" conaffinity="0"/>
      <site name="{sack.name}_center_site" pos="0 0 0.000" size="0.006" rgba="0 1 0 1"/>
      <site name="{sack.name}_top_site" pos="{lean_shift_x * 0.4:.4f} {0.006 * side_sign + lean_shift_y * 0.3:.4f} {top_z + 0.028:.4f}" size="0.007" rgba="1 0.5 0 1"/>
      <site name="{sack.name}_side_site" pos="{lean_shift_x:.4f} {side_sign * (outer_y - 0.010 + 0.30 * dominant_bulge):.4f} 0.018" size="0.007" rgba="1 0.8 0 1"/>

      <geom name="{sack.name}_core_geom" type="ellipsoid" pos="0 0 {0.006 - 0.012 * sack.top_collapse:.4f}" size="{core_x:.4f} {core_y:.4f} {core_z:.4f}" mass="{mass_core:.4f}" rgba="{rgba_text} 0.12" friction="1.0 0.03 0.01"/>
      <geom name="{sack.name}_core_front" type="capsule" fromto="{-0.024 + lean_shift_x:.4f} 0 {-0.018:.4f} {0.050 + lean_shift_x:.4f} 0 {0.058:.4f}" size="{0.025 + 0.006 * sack.fill_ratio:.4f}" mass="{0.10 * sack.total_mass:.4f}" rgba="{rgba_text} 0.10"/>
      <geom name="{sack.name}_core_left" type="capsule" fromto="-0.016 {0.018 + bulge_minor_offset:.4f} -0.012 0.020 {0.030 + bulge_major_offset:.4f} 0.054" size="{0.020 + 0.010 * dominant_bulge:.4f}" mass="{mass_lobes:.4f}" rgba="{rgba_text} 0.10"/>
      <geom name="{sack.name}_core_right" type="capsule" fromto="-0.016 {-0.018 - bulge_minor_offset:.4f} -0.012 0.020 {-0.030 - bulge_major_offset:.4f} 0.054" size="{0.018 + 0.008 * dominant_bulge:.4f}" mass="{mass_lobes:.4f}" rgba="{rgba_text} 0.10"/>

      <!-- 자루의 외형은 둥근 shell lobe로 보이게 하고, contact proxy는 반투명하게 숨긴다. -->
      <geom name="{sack.name}_shell_main_visual" type="ellipsoid" pos="{lean_shift_x * 0.5:.4f} {lean_shift_y * 0.5:.4f} {0.002 - 0.28 * top_indent:.4f}" size="{shell_x:.4f} {shell_y:.4f} {shell_z:.4f}" rgba="{rgba_text} 0.16" contype="0" conaffinity="0"/>
      <geom name="{sack.name}_shell_left_visual" type="ellipsoid" pos="0.0040 {bulge_major_offset + lean_shift_y:.4f} 0.0020" size="{shell_x * 0.68:.4f} {0.016 + 0.55 * dominant_bulge:.4f} {shell_z * 0.82:.4f}" rgba="{rgba_text} 0.14" contype="0" conaffinity="0"/>
      <geom name="{sack.name}_shell_right_visual" type="ellipsoid" pos="0.0040 {-bulge_minor_offset + lean_shift_y:.4f} 0.0000" size="{shell_x * 0.64:.4f} {0.014 + 0.25 * dominant_bulge:.4f} {shell_z * 0.80:.4f}" rgba="{rgba_text} 0.12" contype="0" conaffinity="0"/>
      <geom name="{sack.name}_top_left_crown_visual" type="capsule" fromto="{-top_band_x * 0.40:.4f} {top_band_y * 0.36:.4f} {top_crown_z:.4f} {top_band_x * 0.40:.4f} {top_band_y * 0.18:.4f} {top_crown_z + 0.004:.4f}" size="{top_crown_radius:.4f}" rgba="{rgba_text} 0.14" contype="0" conaffinity="0"/>
      <geom name="{sack.name}_top_right_crown_visual" type="capsule" fromto="{-top_band_x * 0.40:.4f} {-top_band_y * 0.18:.4f} {top_crown_z + 0.004:.4f} {top_band_x * 0.40:.4f} {-top_band_y * 0.36:.4f} {top_crown_z:.4f}" size="{top_crown_radius:.4f}" rgba="{rgba_text} 0.14" contype="0" conaffinity="0"/>

      <body name="{sack.name}_bottom_support_proxy_body" pos="{lean_shift_x * 0.18:.4f} {sack.support_bias + lean_shift_y * 0.22:.4f} {-outer_z + 0.016:.4f}">
        <joint name="{sack.name}_bottom_x" type="slide" axis="1 0 0" limited="true" range="-0.012 0.012" damping="18" stiffness="{340 + 180 * sack.fill_ratio:.1f}"/>
        <joint name="{sack.name}_bottom_y" type="slide" axis="0 1 0" limited="true" range="-0.012 0.012" damping="18" stiffness="{340 + 180 * sack.fill_ratio:.1f}"/>
        <joint name="{sack.name}_bottom_z" type="slide" axis="0 0 1" limited="true" range="-0.008 0.020" damping="24" stiffness="{420 + 220 * sack.fill_ratio:.1f}"/>
        <geom name="{sack.name}_bottom_support_proxy" type="ellipsoid" size="{outer_x * 0.92:.4f} {max(outer_y - 0.010, 0.020):.4f} {bottom_half_z:.4f}" mass="{mass_proxy_panel:.4f}" rgba="{rgba_text} 0.10" friction="1.55 0.05 0.01"/>
        <geom name="{sack.name}_bottom_deform_visual" type="ellipsoid" size="{outer_x * 0.98:.4f} {max(outer_y - 0.004, 0.022):.4f} {bottom_half_z * 1.30:.4f}" rgba="{rgba_text} 0.22" contype="0" conaffinity="0"/>
      </body>

      <body name="{sack.name}_top_grasp_proxy_body" pos="{lean_shift_x:.4f} {0.006 * side_sign + lean_shift_y * 0.35:.4f} {top_z:.4f}">
        <joint name="{sack.name}_top_x" type="slide" axis="1 0 0" limited="true" range="-0.018 0.018" damping="12" stiffness="{130 + 70 * sack.fill_ratio:.1f}"/>
        <joint name="{sack.name}_top_y" type="slide" axis="0 1 0" limited="true" range="-0.018 0.018" damping="12" stiffness="{130 + 70 * sack.fill_ratio:.1f}"/>
        <joint name="{sack.name}_top_z" type="slide" axis="0 0 1" limited="true" range="-0.040 0.010" damping="14" stiffness="{90 + 60 * sack.fill_ratio:.1f}"/>
        <geom name="{sack.name}_top_grasp_proxy" type="ellipsoid" size="{top_band_x:.4f} {top_band_y:.4f} {0.012 + 0.18 * top_indent:.4f}" mass="{0.06 * sack.total_mass:.4f}" rgba="{rgba_text} 0.10" friction="2.2 0.05 0.01"/>
        <geom name="{sack.name}_top_deform_visual" type="ellipsoid" size="{top_band_x * 1.04:.4f} {top_band_y * 1.08:.4f} {0.016 + 0.22 * top_indent:.4f}" rgba="{rgba_text} 0.36" contype="0" conaffinity="0"/>
        <geom name="{sack.name}_top_fold_left_visual" type="capsule" fromto="{-top_band_x * 0.48:.4f} {top_band_y * 0.28:.4f} 0.000 {top_band_x * 0.10:.4f} {top_band_y * 0.08:.4f} {0.010 + 0.12 * top_indent:.4f}" size="{top_crown_radius * 0.72:.4f}" rgba="{rgba_text} 0.26" contype="0" conaffinity="0"/>
        <geom name="{sack.name}_top_fold_right_visual" type="capsule" fromto="{-top_band_x * 0.12:.4f} {-top_band_y * 0.08:.4f} {0.010 + 0.12 * top_indent:.4f} {top_band_x * 0.48:.4f} {-top_band_y * 0.28:.4f} 0.000" size="{top_crown_radius * 0.72:.4f}" rgba="{rgba_text} 0.26" contype="0" conaffinity="0"/>
      </body>

      <body name="{sack.name}_left_side_proxy_body" pos="{lean_shift_x * 0.16:.4f} {outer_y + bulge_major_offset + lean_shift_y:.4f} 0.010">
        <joint name="{sack.name}_left_side_slide" type="slide" axis="0 -1 0" limited="true" range="-0.028 0.016" damping="16" stiffness="{150 - 40 * sack.top_collapse + 120 * sack.fill_ratio:.1f}"/>
        <geom name="{sack.name}_left_side_proxy" type="ellipsoid" size="{side_half_x:.4f} {side_half_y:.4f} {side_half_z:.4f}" mass="{0.06 * sack.total_mass:.4f}" rgba="{rgba_text} 0.08" friction="1.25 0.04 0.01"/>
        <geom name="{sack.name}_left_side_deform_visual" type="ellipsoid" size="{side_half_x * 1.03:.4f} {side_half_y * 1.08:.4f} {side_half_z * 1.02:.4f}" rgba="{rgba_text} 0.24" contype="0" conaffinity="0"/>
      </body>

      <body name="{sack.name}_right_side_proxy_body" pos="{lean_shift_x * 0.16:.4f} {-outer_y - bulge_minor_offset + lean_shift_y:.4f} 0.010">
        <joint name="{sack.name}_right_side_slide" type="slide" axis="0 1 0" limited="true" range="-0.028 0.016" damping="16" stiffness="{140 - 40 * sack.top_collapse + 110 * sack.fill_ratio:.1f}"/>
        <geom name="{sack.name}_right_side_proxy" type="ellipsoid" size="{side_half_x * 0.96:.4f} {max(side_half_y * 0.78, 0.010):.4f} {side_half_z * 0.98:.4f}" mass="{0.06 * sack.total_mass:.4f}" rgba="{rgba_text} 0.08" friction="1.25 0.04 0.01"/>
        <geom name="{sack.name}_right_side_deform_visual" type="ellipsoid" size="{side_half_x * 0.99:.4f} {max(side_half_y * 0.86, 0.012):.4f} {side_half_z * 1.00:.4f}" rgba="{rgba_text} 0.24" contype="0" conaffinity="0"/>
      </body>

      <body name="{sack.name}_front_proxy_body" pos="{outer_x - 0.012 + lean_shift_x:.4f} {lean_shift_y * 0.16:.4f} 0.000">
        <joint name="{sack.name}_front_slide" type="slide" axis="-1 0 0" limited="true" range="-0.022 0.012" damping="15" stiffness="{160 + 90 * sack.fill_ratio:.1f}"/>
        <geom name="{sack.name}_front_support_proxy" type="ellipsoid" size="{face_half_x:.4f} {face_half_y:.4f} {face_half_z:.4f}" mass="{0.05 * sack.total_mass:.4f}" rgba="{rgba_text} 0.08" friction="1.20 0.04 0.01"/>
        <geom name="{sack.name}_front_deform_visual" type="ellipsoid" size="{face_half_x * 1.02:.4f} {face_half_y * 1.02:.4f} {face_half_z * 1.02:.4f}" rgba="{rgba_text} 0.20" contype="0" conaffinity="0"/>
      </body>

      <body name="{sack.name}_back_proxy_body" pos="{-outer_x + 0.012 + lean_shift_x * 0.4:.4f} {lean_shift_y * 0.16:.4f} 0.000">
        <joint name="{sack.name}_back_slide" type="slide" axis="1 0 0" limited="true" range="-0.022 0.012" damping="15" stiffness="{150 + 80 * sack.fill_ratio:.1f}"/>
        <geom name="{sack.name}_back_support_proxy" type="ellipsoid" size="{max(face_half_x * 0.92, 0.012):.4f} {face_half_y:.4f} {face_half_z:.4f}" mass="{0.05 * sack.total_mass:.4f}" rgba="{rgba_text} 0.08" friction="1.20 0.04 0.01"/>
        <geom name="{sack.name}_back_deform_visual" type="ellipsoid" size="{max(face_half_x * 0.96, 0.014):.4f} {face_half_y * 1.02:.4f} {face_half_z * 1.02:.4f}" rgba="{rgba_text} 0.20" contype="0" conaffinity="0"/>
      </body>

      <body name="{sack.name}_shape_proxy_body" pos="{lean_shift_x * 0.6:.4f} {side_sign * (0.014 + dominant_bulge) + lean_shift_y * 0.4:.4f} {0.018 + 0.052 * (1.0 - sack.fill_ratio):.4f}">
        <joint name="{sack.name}_bulge_y" type="slide" axis="0 1 0" limited="true" range="-0.020 0.020" damping="8" stiffness="{60 + 120 * max(0.0, 1.0 - sack.fill_ratio):.1f}"/>
        <joint name="{sack.name}_bulge_z" type="slide" axis="0 0 1" limited="true" range="-0.014 0.028" damping="8" stiffness="{70 + 90 * max(0.0, 1.0 - sack.fill_ratio):.1f}"/>
        <geom name="{sack.name}_shape_proxy" type="ellipsoid" size="{0.024 + 0.012 * dominant_bulge:.4f} {0.020 + 0.024 * dominant_bulge:.4f} {0.028 + 0.18 * top_indent:.4f}" mass="{0.04 * sack.total_mass:.4f}" rgba="{rgba_text} 0.10"/>
        <geom name="{sack.name}_shape_deform_visual" type="ellipsoid" size="{0.028 + 0.015 * dominant_bulge:.4f} {0.024 + 0.028 * dominant_bulge:.4f} {0.032 + 0.22 * top_indent:.4f}" rgba="{rgba_text} 0.32" contype="0" conaffinity="0"/>
      </body>
    </body>"""
