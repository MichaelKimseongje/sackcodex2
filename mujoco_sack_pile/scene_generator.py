from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class SackVariant:
    """자루 family별 형상/물성 범위를 정의한다."""

    name: str
    rgba: tuple[float, float, float, float]
    mesh_files: tuple[str, ...]
    mesh_scale_range: tuple[float, float]
    core_size: tuple[float, float, float]
    shell_size: tuple[float, float, float]
    mass_range: tuple[float, float]
    top_offset_range: tuple[float, float]
    top_collapse_range: tuple[float, float]
    side_bulge_range: tuple[float, float]
    pitch_roll_range: tuple[float, float]
    exposure_bias: float


@dataclass
class SackPlacement:
    """개별 자루의 초기 배치와 shape variation 정보를 담는다."""

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
    is_target: bool = False


@dataclass
class EpisodeScene:
    """한 episode에 대응하는 생성 scene 메타데이터."""

    episode_id: str
    xml_path: Path
    seed: int
    sacks: list[SackPlacement] = field(default_factory=list)
    target_name: str = ""
    target_variant: str = ""


SACK_VARIANTS: dict[str, SackVariant] = {
    "regular_well_filled": SackVariant(
        name="regular_well_filled",
        rgba=(0.84, 0.74, 0.48, 1.0),
        mesh_files=("sack7.obj", "sack8.obj", "sack9.obj"),
        mesh_scale_range=(0.064, 0.071),
        core_size=(0.050, 0.040, 0.095),
        shell_size=(0.070, 0.052, 0.118),
        mass_range=(2.3, 2.9),
        top_offset_range=(0.040, 0.060),
        top_collapse_range=(0.000, 0.008),
        side_bulge_range=(0.000, 0.010),
        pitch_roll_range=(0.12, 0.28),
        exposure_bias=0.52,
    ),
    "low_fill_top_collapsed": SackVariant(
        name="low_fill_top_collapsed",
        rgba=(0.82, 0.67, 0.44, 1.0),
        mesh_files=("sack.obj", "sack2.obj", "sack10.obj"),
        mesh_scale_range=(0.064, 0.072),
        core_size=(0.046, 0.035, 0.074),
        shell_size=(0.072, 0.054, 0.118),
        mass_range=(1.4, 2.0),
        top_offset_range=(0.020, 0.038),
        top_collapse_range=(0.018, 0.040),
        side_bulge_range=(0.002, 0.012),
        pitch_roll_range=(0.18, 0.38),
        exposure_bias=0.73,
    ),
    "side_bulged_unstable": SackVariant(
        name="side_bulged_unstable",
        rgba=(0.76, 0.60, 0.40, 1.0),
        mesh_files=("sack3.obj", "sack6Apply.obj", "sack9.obj"),
        mesh_scale_range=(0.066, 0.074),
        core_size=(0.045, 0.034, 0.085),
        shell_size=(0.074, 0.060, 0.114),
        mass_range=(1.8, 2.5),
        top_offset_range=(0.028, 0.048),
        top_collapse_range=(0.008, 0.018),
        side_bulge_range=(0.014, 0.028),
        pitch_roll_range=(0.24, 0.48),
        exposure_bias=0.88,
    ),
}


class SceneGenerator:
    """비정형 sack pile scene과 대응 MJCF를 생성한다."""

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
    ) -> EpisodeScene:
        rng = random.Random(seed)
        count = sack_count if sack_count is not None else rng.randint(3, 6)
        sacks = self._sample_placements(rng, count)
        target = max(
            sacks,
            key=lambda s: s.variant.exposure_bias
            + 0.20 * (1 if s.exposed_face in {"top", "side"} else 0)
            + 0.06 * s.stack_level
            + 0.05 * abs(s.pos[1]),
        )
        target.is_target = True

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
        )

    def _sample_placements(self, rng: random.Random, count: int) -> list[SackPlacement]:
        placements: list[SackPlacement] = []
        exposed_assigned = False
        family_cycle = list(SACK_VARIANTS.values())
        rng.shuffle(family_cycle)

        for idx in range(count):
            variant = family_cycle[idx % len(family_cycle)]
            if idx >= len(family_cycle):
                variant = rng.choice(list(SACK_VARIANTS.values()))

            placed = False
            for attempt in range(120):
                radial = rng.uniform(0.03, 0.21)
                angle = rng.uniform(-math.pi, math.pi)
                x = 0.62 + radial * math.cos(angle)
                y = 0.00 + radial * math.sin(angle)

                if idx == 0:
                    stack_level = 0
                else:
                    stack_level = 1 if rng.random() < min(0.18 + 0.04 * idx, 0.34) else 0

                hover = rng.uniform(0.000, 0.030) + 0.030 * stack_level
                z = 0.105 + hover

                min_dist = 0.105 if stack_level == 0 else 0.085
                if any(self._xy_distance((x, y), (p.pos[0], p.pos[1])) < min_dist for p in placements):
                    continue

                yaw = rng.uniform(-math.pi, math.pi)
                tilt_mag = rng.uniform(*variant.pitch_roll_range)
                pitch = rng.uniform(-tilt_mag, tilt_mag)
                roll = rng.uniform(-tilt_mag, tilt_mag)
                if stack_level > 0:
                    pitch += rng.uniform(-0.16, 0.16)
                    roll += rng.uniform(-0.16, 0.16)

                top_collapse = rng.uniform(*variant.top_collapse_range)
                side_bulge = rng.uniform(*variant.side_bulge_range)
                fill_ratio = rng.uniform(0.72, 1.06)
                if variant.name == "low_fill_top_collapsed":
                    fill_ratio = rng.uniform(0.48, 0.76)
                elif variant.name == "side_bulged_unstable":
                    fill_ratio = rng.uniform(0.60, 0.88)

                flattening = rng.uniform(0.80, 1.08) - 0.10 * stack_level
                exposed_face = self._pick_exposure(rng, idx, count, exposed_assigned, stack_level, abs(y))
                mesh_file = rng.choice(variant.mesh_files)
                mesh_scale = rng.uniform(*variant.mesh_scale_range) * (0.96 + 0.10 * fill_ratio)
                total_mass = rng.uniform(*variant.mass_range)
                top_offset = rng.uniform(*variant.top_offset_range)
                support_bias = rng.uniform(-0.016, 0.016)

                placements.append(
                    SackPlacement(
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
                    )
                )
                exposed_assigned = exposed_assigned or exposed_face in {"top", "side"}
                placed = True
                break

            if not placed:
                variant = rng.choice(list(SACK_VARIANTS.values()))
                placements.append(
                    SackPlacement(
                        name=f"sack_{idx}",
                        variant=variant,
                        pos=(0.50 + 0.06 * idx, -0.12 + 0.08 * (idx % 3), 0.130 + 0.025 * (idx // 3)),
                        euler=(0.15 * (-1) ** idx, -0.08 * idx, 0.45 * idx),
                        exposed_face="side" if not exposed_assigned else "top",
                        mesh_file=rng.choice(variant.mesh_files),
                        mesh_scale=sum(variant.mesh_scale_range) * 0.5,
                        total_mass=sum(variant.mass_range) * 0.5,
                        top_offset=sum(variant.top_offset_range) * 0.5,
                        top_collapse=sum(variant.top_collapse_range) * 0.5,
                        side_bulge=sum(variant.side_bulge_range) * 0.5,
                        fill_ratio=0.8,
                        flattening=0.9,
                        stack_level=0,
                        support_bias=0.0,
                    )
                )
                exposed_assigned = True

        if not exposed_assigned and placements:
            placements[0].exposed_face = "top"
        return placements

    @staticmethod
    def _xy_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return math.sqrt(dx * dx + dy * dy)

    @staticmethod
    def _pick_exposure(
        rng: random.Random,
        idx: int,
        count: int,
        exposed_assigned: bool,
        stack_level: int,
        lateral_bias: float,
    ) -> str:
        if not exposed_assigned and idx == count - 1:
            return "top"
        if stack_level > 0:
            return "top" if rng.random() < 0.62 else "side"
        if lateral_bias > 0.14:
            return "side"
        sample = rng.random()
        if sample < 0.34:
            return "side"
        if sample < 0.70:
            return "top"
        return "partial"

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
    <global azimuth="128" elevation="-20"/>
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
            yield f'    <exclude body1="{sack.name}_top_skin" body2="{sack.name}_left_skin"/>'
            yield f'    <exclude body1="{sack.name}_top_skin" body2="{sack.name}_right_skin"/>'

    def _sack_body_xml(self, sack: SackPlacement) -> str:
        v = sack.variant
        roll, pitch, yaw = sack.euler
        r, g, b, _ = v.rgba

        core_x = v.core_size[0] * (0.92 + 0.10 * sack.fill_ratio)
        core_y = v.core_size[1] * (0.88 + 0.20 * sack.fill_ratio)
        core_z = v.core_size[2] * (0.82 + 0.14 * sack.fill_ratio)
        shell_x = v.shell_size[0] * (0.96 + 0.08 * sack.fill_ratio)
        shell_y = v.shell_size[1] * (0.90 + 0.28 * sack.fill_ratio + 0.60 * sack.side_bulge)
        shell_z = v.shell_size[2] * sack.flattening
        top_z = sack.top_offset + shell_z * 0.64 - sack.top_collapse

        side_sign = -1.0 if sack.exposed_face == "side" else 1.0
        front_lean = 0.014 if sack.exposed_face == "partial" else 0.0
        mass_core = 0.42 * sack.total_mass
        mass_lobes = 0.18 * sack.total_mass
        mass_shell = 0.08 * sack.total_mass
        rgba_text = f"{r:.3f} {g:.3f} {b:.3f}"

        return f"""
    <body name="{sack.name}" pos="{sack.pos[0]:.4f} {sack.pos[1]:.4f} {sack.pos[2]:.4f}" euler="{roll:.4f} {pitch:.4f} {yaw:.4f}">
      <freejoint name="{sack.name}_free"/>

      <geom name="{sack.name}_visual" type="mesh" mesh="{sack.name}_visual_mesh" rgba="{rgba_text} 0.92" contype="0" conaffinity="0"/>
      <site name="{sack.name}_center_site" pos="0 0 0.000" size="0.006" rgba="0 1 0 1"/>
      <site name="{sack.name}_top_site" pos="0 {0.006 * side_sign:.4f} {top_z + 0.030:.4f}" size="0.007" rgba="1 0.5 0 1"/>
      <site name="{sack.name}_side_site" pos="{front_lean:.4f} {side_sign * (shell_y - 0.010):.4f} 0.018" size="0.007" rgba="1 0.8 0 1"/>

      <geom name="{sack.name}_core_geom" type="ellipsoid" pos="0 0 {0.006 - 0.012 * sack.top_collapse:.4f}" size="{core_x:.4f} {core_y:.4f} {core_z:.4f}" mass="{mass_core:.4f}" rgba="{rgba_text} 0.12" friction="1.0 0.03 0.01"/>
      <geom name="{sack.name}_core_front" type="capsule" fromto="{-0.020 + front_lean:.4f} 0 {-0.020:.4f} {0.045 + front_lean:.4f} 0 {0.060:.4f}" size="{0.026 + 0.006 * sack.fill_ratio:.4f}" mass="{0.10 * sack.total_mass:.4f}" rgba="{rgba_text} 0.10"/>
      <geom name="{sack.name}_core_left" type="capsule" fromto="-0.008 {0.020 + sack.side_bulge:.4f} -0.010 0.018 {0.038 + sack.side_bulge:.4f} 0.060" size="{0.024 + 0.010 * sack.side_bulge:.4f}" mass="{mass_lobes:.4f}" rgba="{rgba_text} 0.10"/>
      <geom name="{sack.name}_core_right" type="capsule" fromto="-0.006 {-0.020 - sack.side_bulge:.4f} -0.008 0.020 {-0.040 - sack.side_bulge:.4f} 0.054" size="{0.022 + 0.012 * sack.side_bulge:.4f}" mass="{mass_lobes:.4f}" rgba="{rgba_text} 0.10"/>

      <body name="{sack.name}_bottom_skin" pos="0 {sack.support_bias:.4f} {-shell_z + 0.016:.4f}">
        <joint name="{sack.name}_bottom_x" type="slide" axis="1 0 0" limited="true" range="-0.012 0.012" damping="18" stiffness="{340 + 180 * sack.fill_ratio:.1f}"/>
        <joint name="{sack.name}_bottom_y" type="slide" axis="0 1 0" limited="true" range="-0.012 0.012" damping="18" stiffness="{340 + 180 * sack.fill_ratio:.1f}"/>
        <joint name="{sack.name}_bottom_z" type="slide" axis="0 0 1" limited="true" range="-0.008 0.020" damping="24" stiffness="{420 + 220 * sack.fill_ratio:.1f}"/>
        <geom name="{sack.name}_bottom_support" type="box" size="{shell_x:.4f} {shell_y - 0.004:.4f} 0.0100" mass="{mass_shell:.4f}" rgba="{rgba_text} 0.34" friction="1.55 0.05 0.01"/>
      </body>

      <body name="{sack.name}_top_skin" pos="{front_lean:.4f} {0.006 * side_sign:.4f} {top_z:.4f}">
        <joint name="{sack.name}_top_x" type="slide" axis="1 0 0" limited="true" range="-0.018 0.018" damping="12" stiffness="{130 + 70 * sack.fill_ratio:.1f}"/>
        <joint name="{sack.name}_top_y" type="slide" axis="0 1 0" limited="true" range="-0.018 0.018" damping="12" stiffness="{130 + 70 * sack.fill_ratio:.1f}"/>
        <joint name="{sack.name}_top_z" type="slide" axis="0 0 1" limited="true" range="-0.040 0.010" damping="14" stiffness="{90 + 60 * sack.fill_ratio:.1f}"/>
        <geom name="{sack.name}_top_grip" type="box" size="{shell_x - 0.016:.4f} {max(shell_y - 0.020, 0.016):.4f} 0.0150" mass="{0.06 * sack.total_mass:.4f}" rgba="{rgba_text} 0.26" friction="2.2 0.05 0.01"/>
      </body>

      <body name="{sack.name}_left_skin" pos="0 {shell_y + sack.side_bulge:.4f} 0.010">
        <joint name="{sack.name}_left_side_slide" type="slide" axis="0 -1 0" limited="true" range="-0.028 0.016" damping="16" stiffness="{150 - 40 * sack.top_collapse + 120 * sack.fill_ratio:.1f}"/>
        <geom name="{sack.name}_side_left" type="box" size="{shell_x - 0.004:.4f} 0.0090 {shell_z - 0.018:.4f}" mass="{0.06 * sack.total_mass:.4f}" rgba="{rgba_text} 0.24"/>
      </body>

      <body name="{sack.name}_right_skin" pos="0 {-shell_y + 0.004 - sack.side_bulge:.4f} 0.010">
        <joint name="{sack.name}_right_side_slide" type="slide" axis="0 1 0" limited="true" range="-0.028 0.016" damping="16" stiffness="{140 - 40 * sack.top_collapse + 110 * sack.fill_ratio:.1f}"/>
        <geom name="{sack.name}_side_right" type="box" size="{shell_x - 0.004:.4f} 0.0090 {shell_z - 0.018:.4f}" mass="{0.06 * sack.total_mass:.4f}" rgba="{rgba_text} 0.24"/>
      </body>

      <body name="{sack.name}_front_skin" pos="{shell_x - 0.010 + front_lean:.4f} 0 0.000">
        <joint name="{sack.name}_front_slide" type="slide" axis="-1 0 0" limited="true" range="-0.022 0.012" damping="15" stiffness="{160 + 90 * sack.fill_ratio:.1f}"/>
        <geom name="{sack.name}_front_panel" type="box" size="0.0090 {shell_y - 0.004:.4f} {shell_z - 0.020:.4f}" mass="{0.05 * sack.total_mass:.4f}" rgba="{rgba_text} 0.22"/>
      </body>

      <body name="{sack.name}_back_skin" pos="{-shell_x + 0.010:.4f} 0 0.000">
        <joint name="{sack.name}_back_slide" type="slide" axis="1 0 0" limited="true" range="-0.022 0.012" damping="15" stiffness="{150 + 80 * sack.fill_ratio:.1f}"/>
        <geom name="{sack.name}_back_panel" type="box" size="0.0090 {shell_y - 0.004:.4f} {shell_z - 0.020:.4f}" mass="{0.05 * sack.total_mass:.4f}" rgba="{rgba_text} 0.22"/>
      </body>

      <body name="{sack.name}_local_patch" pos="{front_lean * 0.6:.4f} {side_sign * (0.012 + sack.side_bulge):.4f} {0.020 + 0.060 * (1.0 - sack.fill_ratio):.4f}">
        <joint name="{sack.name}_bulge_y" type="slide" axis="0 1 0" limited="true" range="-0.020 0.020" damping="8" stiffness="{60 + 120 * max(0.0, 1.0 - sack.fill_ratio):.1f}"/>
        <joint name="{sack.name}_bulge_z" type="slide" axis="0 0 1" limited="true" range="-0.014 0.028" damping="8" stiffness="{70 + 90 * max(0.0, 1.0 - sack.fill_ratio):.1f}"/>
        <geom name="{sack.name}_compliant_patch" type="ellipsoid" size="{0.020 + 0.010 * sack.side_bulge:.4f} {0.016 + 0.020 * sack.side_bulge:.4f} {0.026 + 0.010 * sack.top_collapse:.4f}" mass="{0.04 * sack.total_mass:.4f}" rgba="{rgba_text} 0.28"/>
      </body>
    </body>"""
