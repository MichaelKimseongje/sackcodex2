from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import mujoco
import numpy as np

# 이 모듈은 메인 benchmark 경로가 아닌 exploratory soft reference 전용이다.
# shape fidelity 실험용으로만 유지하며, support-state 성능 집계나 sim2real 주장의 본선 지표에는 포함하지 않는다.


@dataclass
class SoftPrototypeSpec:
    """mesh 기반 soft sack 비교 실험에 쓰는 단일 자루 설정."""

    variant: str
    mesh_file: str
    scale: float
    shell_mass: float
    particle_radius: float
    rgba: tuple[float, float, float, float]
    payload_offsets: tuple[tuple[float, float, float], ...]
    payload_size: float
    payload_mass: float
    euler: tuple[float, float, float]


@dataclass
class SoftPrototypeReport:
    """mesh prototype의 settle 결과를 비교하기 위한 요약 리포트."""

    mode: str
    xml_path: str
    settle_seconds: float
    stable: bool
    non_finite: bool
    tail_mean_qvel: float
    tail_max_qvel: float
    peak_contact_count: int
    final_min_geom_height: float
    final_z_span: float
    failure_tags: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


class SoftSackPrototypeGenerator:
    """experimental mesh soft reference를 생성하고 비교한다."""

    FAMILY_LIBRARY: dict[str, dict] = {
        "regular_well_filled": {
            "mesh_files": ("sack8.obj", "sack9.obj"),
            "scale_range": (0.067, 0.072),
            "shell_mass_range": (0.68, 0.88),
            "radius_range": (0.0047, 0.0053),
            "rgba": (0.84, 0.74, 0.50, 1.0),
            "payload_offsets": ((0.000, 0.000, 0.006), (0.018, -0.004, -0.010), (-0.020, 0.010, -0.004), (0.000, -0.016, 0.010)),
            "payload_size": 0.018,
            "payload_mass_scale": 0.32,
            "euler_range": ((0.04, 0.14), (-0.10, 0.10), (-0.18, 0.18)),
        },
        "low_fill_top_collapsed": {
            "mesh_files": ("sack.obj", "sack2.obj", "sack10.obj"),
            "scale_range": (0.067, 0.073),
            "shell_mass_range": (0.44, 0.62),
            "radius_range": (0.0042, 0.0048),
            "rgba": (0.82, 0.66, 0.44, 1.0),
            "payload_offsets": ((0.000, 0.000, -0.004), (0.016, 0.000, -0.012), (-0.014, 0.008, -0.008)),
            "payload_size": 0.015,
            "payload_mass_scale": 0.20,
            "euler_range": ((0.12, 0.28), (-0.16, 0.16), (-0.20, 0.20)),
        },
        "side_bulged_unstable": {
            "mesh_files": ("sack3.obj", "sack6Apply.obj", "sack9.obj"),
            "scale_range": (0.068, 0.074),
            "shell_mass_range": (0.58, 0.78),
            "radius_range": (0.0048, 0.0055),
            "rgba": (0.76, 0.60, 0.40, 1.0),
            "payload_offsets": ((0.014, 0.020, -0.006), (-0.012, -0.010, 0.008), (0.000, 0.000, -0.014), (0.012, -0.018, 0.010)),
            "payload_size": 0.017,
            "payload_mass_scale": 0.26,
            "euler_range": ((0.26, 0.46), (-0.22, 0.22), (-0.26, 0.26)),
        },
    }

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.generated_dir = self.base_dir / "mujoco_sack_pile" / "generated"
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.base_dir / "mujoco_sack_pile" / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.mesh_dir = self.base_dir / "object"

    def sample_spec(self, variant: str, seed: int, mesh_file: str | None = None) -> SoftPrototypeSpec:
        """family에 맞는 초기 shape/payload 파라미터를 샘플링한다."""

        library = self.FAMILY_LIBRARY[variant]
        rng = random.Random(seed)
        mesh_choice = mesh_file or rng.choice(library["mesh_files"])
        roll_range, pitch_range, yaw_range = library["euler_range"]
        return SoftPrototypeSpec(
            variant=variant,
            mesh_file=mesh_choice,
            scale=rng.uniform(*library["scale_range"]),
            shell_mass=rng.uniform(*library["shell_mass_range"]),
            particle_radius=rng.uniform(*library["radius_range"]),
            rgba=library["rgba"],
            payload_offsets=library["payload_offsets"],
            payload_size=library["payload_size"],
            payload_mass=rng.uniform(*library["shell_mass_range"]) * library["payload_mass_scale"],
            euler=(
                rng.uniform(*roll_range),
                rng.uniform(*pitch_range),
                rng.uniform(*yaw_range),
            ),
        )

    def generate_xml(self, spec: SoftPrototypeSpec, mode: str, episode_id: str) -> Path:
        """요청한 prototype 모드에 맞는 MJCF를 작성한다."""

        xml_path = self.generated_dir / f"{episode_id}_{mode}.xml"
        xml_path.write_text(self._build_xml(spec, mode, episode_id), encoding="utf-8")
        return xml_path

    def compare_headless(
        self,
        spec: SoftPrototypeSpec,
        episode_id: str,
        settle_seconds: float = 5.0,
    ) -> dict[str, SoftPrototypeReport]:
        """mesh-only와 mesh+payload를 각각 settle시켜 비교 리포트를 만든다."""

        reports: dict[str, SoftPrototypeReport] = {}
        for mode in ("mesh_only", "mesh_with_payload"):
            xml_path = self.generate_xml(spec, mode=mode, episode_id=episode_id)
            reports[mode] = self.simulate_settle(xml_path, settle_seconds=settle_seconds, mode=mode)

        report_path = self.log_dir / f"{episode_id}_soft_prototype_compare.json"
        report_path.write_text(
            json.dumps(
                {
                    "variant": spec.variant,
                    "mesh_file": spec.mesh_file,
                    "reports": {mode: report.to_dict() for mode, report in reports.items()},
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return reports

    @staticmethod
    def simulate_settle(xml_path: Path, settle_seconds: float, mode: str) -> SoftPrototypeReport:
        """headless settle을 수행하고 발산/축늘어짐 proxy를 요약한다."""

        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        settle_steps = max(1, int(round(settle_seconds / model.opt.timestep)))
        window_steps = max(40, int(round(0.5 / model.opt.timestep)))
        tail_qvel_norms: list[float] = []
        peak_contact_count = 0
        non_finite = False

        for step_idx in range(settle_steps):
            mujoco.mj_step(model, data)
            peak_contact_count = max(peak_contact_count, int(data.ncon))
            if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
                non_finite = True
                break
            if step_idx >= settle_steps - window_steps:
                tail_qvel_norms.append(float(np.linalg.norm(data.qvel)))

        static_geom_ids: list[int] = []
        for name in ("floor", "work_pad"):
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if geom_id >= 0:
                static_geom_ids.append(int(geom_id))
        dynamic_geom_ids = np.array([geom_id for geom_id in range(model.ngeom) if geom_id not in static_geom_ids], dtype=np.int32)
        height_samples: list[np.ndarray] = []
        if getattr(model, "nflexvert", 0) > 0 and hasattr(data, "flexvert_xpos"):
            height_samples.append(np.asarray(data.flexvert_xpos[:, 2], dtype=np.float64))
        if dynamic_geom_ids.size > 0:
            height_samples.append(np.asarray(data.geom_xpos[dynamic_geom_ids, 2], dtype=np.float64))
        if height_samples:
            merged_heights = np.concatenate(height_samples, axis=0)
            final_min_geom_height = float(np.min(merged_heights))
            final_z_span = float(np.max(merged_heights) - np.min(merged_heights))
        else:
            final_min_geom_height = float("nan")
            final_z_span = float("nan")

        tail_mean_qvel = float(np.mean(tail_qvel_norms)) if tail_qvel_norms else float("inf")
        tail_max_qvel = float(np.max(tail_qvel_norms)) if tail_qvel_norms else float("inf")

        failure_tags: list[str] = []
        if non_finite:
            failure_tags.append("non_finite_state")
        if tail_mean_qvel > 0.45:
            failure_tags.append("tail_mean_velocity_high")
        if tail_max_qvel > 2.20:
            failure_tags.append("tail_peak_velocity_high")
        if math.isfinite(final_min_geom_height) and final_min_geom_height < 0.006:
            failure_tags.append("sagged_to_ground")
        if math.isfinite(final_z_span) and final_z_span < 0.035:
            failure_tags.append("collapsed_z_span")

        return SoftPrototypeReport(
            mode=mode,
            xml_path=str(xml_path),
            settle_seconds=settle_seconds,
            stable=not failure_tags,
            non_finite=non_finite,
            tail_mean_qvel=tail_mean_qvel,
            tail_max_qvel=tail_max_qvel,
            peak_contact_count=peak_contact_count,
            final_min_geom_height=final_min_geom_height,
            final_z_span=final_z_span,
            failure_tags=failure_tags,
        )

    def _build_xml(self, spec: SoftPrototypeSpec, mode: str, episode_id: str) -> str:
        """prototype MJCF 본문을 생성한다."""

        bag_body = self._bag_body_xml(spec, mode)
        return f"""<mujoco model="{episode_id}_{mode}">
  <compiler angle="radian" coordinate="local"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" iterations="120" tolerance="1e-10"/>
  <size memory="768M" nconmax="16000"/>

  <extension>
    <plugin plugin="mujoco.elasticity.shell"/>
  </extension>

  <visual>
    <global azimuth="128" elevation="-20"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.82 0.82 0.82" specular="0.12 0.12 0.12"/>
  </visual>

  <default>
    <geom condim="4" friction="1.25 0.04 0.01" margin="0.0025"/>
  </default>

  <worldbody>
    <light name="key" pos="0.95 0.05 1.8" dir="-0.25 -0.04 -1" directional="true"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.94 0.94 0.94 1"/>
    <geom name="work_pad" type="box" pos="0.62 0 0.010" size="0.30 0.30 0.010" rgba="0.60 0.61 0.62 1" friction="1.5 0.05 0.01"/>
    <camera name="overview" pos="1.20 0.00 0.90" xyaxes="0 1 0 -0.46 0 0.88"/>
{bag_body}
  </worldbody>
</mujoco>
"""

    def _bag_body_xml(self, spec: SoftPrototypeSpec, mode: str) -> str:
        """동일 mesh를 mesh-only 또는 mesh+payload로 배치한다."""

        r, g, b, _ = spec.rgba
        payload_xml = ""
        if mode == "mesh_with_payload":
            payload_xml = "\n".join(
                self._payload_body_xml(
                    payload_id=idx,
                    offset=offset,
                    size=spec.payload_size,
                    mass=spec.payload_mass * (0.96 - 0.08 * idx),
                    rgba=(r, g, b, 0.16),
                )
                for idx, offset in enumerate(spec.payload_offsets)
            )

        # mesh 자체를 지키되, 매우 강한 제약 대신 완만한 elasticity를 둬서
        # mesh-only가 어떻게 축 처지는지와 payload가 얼마나 버텨주는지 비교한다.
        return f"""
    <body name="bag" pos="0.62 0.00 0.22" euler="{spec.euler[0]:.4f} {spec.euler[1]:.4f} {spec.euler[2]:.4f}">
      <freejoint name="bag_free"/>
      <inertial pos="0 0 0" mass="0.03" diaginertia="0.0003 0.0003 0.0003"/>
      <flexcomp
          name="bag_shell"
          type="mesh"
          file="../../object/{spec.mesh_file}"
          dim="2"
          mass="{spec.shell_mass:.4f}"
          radius="{spec.particle_radius:.5f}"
          scale="{spec.scale:.5f} {spec.scale:.5f} {spec.scale:.5f}">
        <contact condim="4" friction="1.35 0.04 0.01" solref="0.004 1" solimp="0.95 0.99 0.002"/>
        <edge equality="false" damping="0.30"/>
        <plugin plugin="mujoco.elasticity.shell">
          <config key="poisson" value="0.28"/>
          <config key="thickness" value="0.0028"/>
          <config key="young" value="8e4"/>
        </plugin>
      </flexcomp>
      <site name="bag_center_site" pos="0 0 0" size="0.006" rgba="{r:.3f} {g:.3f} {b:.3f} 1"/>
      {payload_xml}
    </body>"""

    @staticmethod
    def _payload_body_xml(
        payload_id: int,
        offset: tuple[float, float, float],
        size: float,
        mass: float,
        rgba: tuple[float, float, float, float],
    ) -> str:
        """자루 내부를 완전히 채우지는 않되 shape 붕괴를 늦추는 최소 payload를 둔다."""

        r, g, b, a = rgba
        return f"""
      <body name="payload_{payload_id}" pos="{offset[0]:.4f} {offset[1]:.4f} {offset[2]:.4f}">
        <joint name="payload_{payload_id}_x" type="slide" axis="1 0 0" limited="true" range="-0.018 0.018" damping="12"/>
        <joint name="payload_{payload_id}_y" type="slide" axis="0 1 0" limited="true" range="-0.018 0.018" damping="12"/>
        <joint name="payload_{payload_id}_z" type="slide" axis="0 0 1" limited="true" range="-0.024 0.024" damping="14"/>
        <geom type="sphere" size="{size:.4f}" mass="{mass:.4f}" rgba="{r:.3f} {g:.3f} {b:.3f} {a:.3f}" friction="0.9 0.03 0.01"/>
      </body>"""
