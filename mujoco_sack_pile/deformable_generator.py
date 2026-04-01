from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DeformableSackSpec:
    """deformable sack 1개의 초기 조건."""

    name: str
    mesh_file: str
    pos: tuple[float, float, float]
    euler: tuple[float, float, float]
    scale: float
    mass: float
    radius: float
    rgba: tuple[float, float, float, float]
    variant: str


class DeformablePileGenerator:
    """MuJoCo flexcomp 기반 비정형 sack pile scene을 생성한다."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.generated_dir = self.base_dir / "mujoco_sack_pile" / "generated"
        self.generated_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, seed: int, episode_id: str, sack_count: int | None = None) -> Path:
        rng = random.Random(seed)
        count = sack_count if sack_count is not None else rng.randint(2, 3)
        sacks = self._sample_specs(rng, count)
        xml_path = self.generated_dir / f"{episode_id}_deformable.xml"
        xml_path.write_text(self._build_xml(episode_id, sacks), encoding="utf-8")
        return xml_path

    def _sample_specs(self, rng: random.Random, count: int) -> list[DeformableSackSpec]:
        families = [
            ("regular_well_filled", ("sack7.obj", "sack8.obj", "sack9.obj"), (0.066, 0.072), (0.65, 0.95), (0.0042, 0.0052), (0.84, 0.74, 0.50, 0.55)),
            ("low_fill_top_collapsed", ("sack.obj", "sack2.obj", "sack10.obj"), (0.065, 0.073), (0.45, 0.70), (0.0036, 0.0046), (0.82, 0.66, 0.44, 0.52)),
            ("side_bulged_unstable", ("sack3.obj", "sack6Apply.obj", "sack9.obj"), (0.067, 0.074), (0.55, 0.82), (0.0044, 0.0054), (0.76, 0.60, 0.40, 0.54)),
        ]
        specs: list[DeformableSackSpec] = []
        for i in range(count):
            variant, meshes, scale_range, mass_range, radius_range, rgba = families[i % len(families)]
            if i >= len(families):
                variant, meshes, scale_range, mass_range, radius_range, rgba = rng.choice(families)

            placed = False
            for _ in range(120):
                radius = rng.uniform(0.04, 0.22)
                angle = rng.uniform(-math.pi, math.pi)
                x = 0.62 + radius * math.cos(angle)
                y = 0.00 + radius * math.sin(angle)
                z = 0.17 + rng.uniform(0.00, 0.06) + (0.04 if rng.random() < 0.30 else 0.0)
                if any(self._xy_distance((x, y), (s.pos[0], s.pos[1])) < 0.12 for s in specs):
                    continue
                roll = rng.uniform(-0.75, 0.75)
                pitch = rng.uniform(-0.75, 0.75)
                yaw = rng.uniform(-math.pi, math.pi)
                specs.append(
                    DeformableSackSpec(
                        name=f"dsack_{i}",
                        mesh_file=rng.choice(meshes),
                        pos=(x, y, z),
                        euler=(roll, pitch, yaw),
                        scale=rng.uniform(*scale_range),
                        mass=rng.uniform(*mass_range),
                        radius=rng.uniform(*radius_range),
                        rgba=rgba,
                        variant=variant,
                    )
                )
                placed = True
                break
            if not placed:
                specs.append(
                    DeformableSackSpec(
                        name=f"dsack_{i}",
                        mesh_file=rng.choice(meshes),
                        pos=(0.50 + 0.06 * i, -0.12 + 0.06 * (i % 3), 0.20 + 0.02 * (i // 3)),
                        euler=(0.2 * (-1) ** i, -0.15, 0.3 * i),
                        scale=sum(scale_range) * 0.5,
                        mass=sum(mass_range) * 0.5,
                        radius=sum(radius_range) * 0.5,
                        rgba=rgba,
                        variant=variant,
                    )
                )
        return specs

    @staticmethod
    def _xy_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return math.sqrt(dx * dx + dy * dy)

    def _build_xml(self, episode_id: str, sacks: list[DeformableSackSpec]) -> str:
        sack_xml = "\n".join(self._sack_xml(spec) for spec in sacks)
        return f"""<mujoco model="{episode_id}_deformable_pile">
  <compiler angle="radian" coordinate="local"/>
  <option gravity="0 0 -9.81" timestep="0.002" iterations="120" tolerance="1e-10"/>
  <size memory="1024M" nconmax="20000"/>

  <visual>
    <global azimuth="130" elevation="-22"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.80 0.80 0.80" specular="0.15 0.15 0.15"/>
  </visual>

  <worldbody>
    <light name="key" pos="0.9 0.0 1.8" dir="-0.2 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.93 0.93 0.93 1"/>
    <geom name="work_pad" type="box" pos="0.62 0 0.010" size="0.32 0.32 0.010" rgba="0.60 0.61 0.62 1" friction="1.5 0.05 0.01"/>
    <camera name="overview" pos="1.26 0.00 0.96" xyaxes="0 1 0 -0.48 0 0.88"/>
{sack_xml}
  </worldbody>
</mujoco>
"""

    def _sack_xml(self, spec: DeformableSackSpec) -> str:
        r, g, b, a = spec.rgba
        payload_scale = spec.scale / 0.070
        # flexcomp mesh 자체가 deformable shell 역할을 하고,
        # 내부의 느슨한 payload 구체들이 자루가 눌리거나 기울 때 shape variation을 만든다.
        payload = f"""
      <body name="{spec.name}_payload0" pos="0.000 0.000 {0.010 * payload_scale:.4f}">
        <joint name="{spec.name}_payload0_x" type="slide" axis="1 0 0" limited="true" range="-0.020 0.020" damping="4"/>
        <joint name="{spec.name}_payload0_y" type="slide" axis="0 1 0" limited="true" range="-0.020 0.020" damping="4"/>
        <joint name="{spec.name}_payload0_z" type="slide" axis="0 0 1" limited="true" range="-0.030 0.030" damping="4"/>
        <geom type="sphere" size="{0.018 * payload_scale:.4f}" mass="{0.16 * spec.mass:.4f}" rgba="{r:.3f} {g:.3f} {b:.3f} 0.10" friction="0.9 0.03 0.01"/>
      </body>
      <body name="{spec.name}_payload1" pos="{0.022 * payload_scale:.4f} 0.000 {-0.012 * payload_scale:.4f}">
        <joint name="{spec.name}_payload1_x" type="slide" axis="1 0 0" limited="true" range="-0.020 0.020" damping="4"/>
        <joint name="{spec.name}_payload1_y" type="slide" axis="0 1 0" limited="true" range="-0.020 0.020" damping="4"/>
        <joint name="{spec.name}_payload1_z" type="slide" axis="0 0 1" limited="true" range="-0.030 0.030" damping="4"/>
        <geom type="sphere" size="{0.016 * payload_scale:.4f}" mass="{0.14 * spec.mass:.4f}" rgba="{r:.3f} {g:.3f} {b:.3f} 0.10" friction="0.9 0.03 0.01"/>
      </body>
      <body name="{spec.name}_payload2" pos="{-0.020 * payload_scale:.4f} {0.010 * payload_scale:.4f} 0.000">
        <joint name="{spec.name}_payload2_x" type="slide" axis="1 0 0" limited="true" range="-0.020 0.020" damping="4"/>
        <joint name="{spec.name}_payload2_y" type="slide" axis="0 1 0" limited="true" range="-0.020 0.020" damping="4"/>
        <joint name="{spec.name}_payload2_z" type="slide" axis="0 0 1" limited="true" range="-0.030 0.030" damping="4"/>
        <geom type="sphere" size="{0.015 * payload_scale:.4f}" mass="{0.12 * spec.mass:.4f}" rgba="{r:.3f} {g:.3f} {b:.3f} 0.10" friction="0.9 0.03 0.01"/>
      </body>"""

        return f"""
    <body name="{spec.name}" pos="{spec.pos[0]:.4f} {spec.pos[1]:.4f} {spec.pos[2]:.4f}" euler="{spec.euler[0]:.4f} {spec.euler[1]:.4f} {spec.euler[2]:.4f}">
      <freejoint name="{spec.name}_free"/>
      <inertial pos="0 0 0" mass="0.02" diaginertia="0.0002 0.0002 0.0002"/>
      <flexcomp name="{spec.name}_flex" type="mesh" file="../../object/{spec.mesh_file}" dim="2" mass="{spec.mass:.4f}" radius="{spec.radius:.5f}" scale="{spec.scale:.5f} {spec.scale:.5f} {spec.scale:.5f}">
        <edge damping="0.18"/>
        <contact condim="4" friction="1.4 0.05 0.01" solref="0.004 1" solimp="0.95 0.99 0.002"/>
      </flexcomp>
      {payload}
      <site name="{spec.name}_site" pos="0 0 0" size="0.006" rgba="{r:.3f} {g:.3f} {b:.3f} 1"/>
    </body>"""
