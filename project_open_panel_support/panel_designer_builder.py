"""GUI에서 조작 가능한 open-panel chain model XML builder.

MuJoCo 기본 viewer는 runtime body/joint 생성을 지원하지 않으므로,
GUI에서 파라미터를 바꾸면 이 builder가 XML을 다시 생성하고 viewer를 다시 연다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
GENERATED_DIR = PROJECT_DIR / "generated"
DESIGNER_XML = GENERATED_DIR / "panel_designer_model.xml"
MIN_PANELS = 4
MAX_PANELS = 12
MIN_HIDDEN_MASS_COUNT = 1
MAX_HIDDEN_MASS_COUNT = 12


def closed_panel_angles(panel_count: int) -> list[float]:
    """N개 panel이 같은 길이의 닫힌 단면을 이루도록 초기 상대각을 만든다."""

    count = int(max(MIN_PANELS, min(MAX_PANELS, panel_count)))
    turn_deg = 360.0 / count
    return [0.0] + [turn_deg] * (count - 1)


@dataclass
class PanelDesignerConfig:
    """GUI에서 수정하는 open-panel model 파라미터."""

    panel_count: int = 8
    bag_length: float = 0.28
    panel_width: float = 0.070
    panel_thickness: float = 0.003
    panel_mass: float = 0.045
    hinge_axis: str = "1 0 0"
    hinge_range_min_deg: float = -90.0
    hinge_range_max_deg: float = 90.0
    hinge_damping: float = 0.02
    hinge_stiffness: float = 1.2
    hinge_armature: float = 0.0003
    passive_hinge_mode: bool = False
    actuator_kp: float = 1.0
    bag_frame_z: float = 0.24
    hidden_mass_enabled: bool = True
    hidden_mass_count: int = 6
    hidden_mass_total: float = 0.34
    hidden_mass_size_scale: float = 0.58
    hidden_mass_ball_joint: bool = True
    hidden_mass_slide_damping: float = 0.05
    hidden_mass_slide_armature: float = 0.0001
    hidden_mass_range_x: float = 0.055
    hidden_mass_range_y: float = 0.055
    hidden_mass_range_z: float = 0.055
    close_loop_enabled: bool = True
    close_loop_solref_time: float = 0.005
    initial_angles_deg: list[float] = field(default_factory=lambda: closed_panel_angles(6))

    def normalized(self) -> "PanelDesignerConfig":
        self.panel_count = int(max(MIN_PANELS, min(MAX_PANELS, self.panel_count)))
        self.hidden_mass_count = int(max(MIN_HIDDEN_MASS_COUNT, min(MAX_HIDDEN_MASS_COUNT, self.hidden_mass_count)))
        self.hidden_mass_size_scale = float(max(0.1, min(2.0, self.hidden_mass_size_scale)))
        if len(self.initial_angles_deg) < self.panel_count:
            fill = closed_panel_angles(self.panel_count)
            self.initial_angles_deg = (self.initial_angles_deg + fill[len(self.initial_angles_deg) :])[: self.panel_count]
        else:
            self.initial_angles_deg = self.initial_angles_deg[: self.panel_count]
        return self


def _actuator_ctrlrange(min_deg: float, max_deg: float) -> str:
    return f"{min_deg:.6f} {max_deg:.6f}"


def _degree_actuator_gain(kp: float) -> float:
    """MuJoCo viewer control 입력(deg)을 내부 hinge 목표각(rad)으로 변환한다."""

    return kp * pi / 180.0


def _hinge_stiffness(cfg: PanelDesignerConfig) -> float:
    """Passive mode에서는 actuator/spring이 형상을 잡지 않도록 hinge spring을 끈다."""

    return 0.0 if cfg.passive_hinge_mode else cfg.hinge_stiffness


def _panel_chain_xml(cfg: PanelDesignerConfig, idx: int = 0) -> str:
    """중첩 body chain을 생성한다. 상위 hinge가 움직이면 하위 panel도 종속적으로 따라간다."""

    if idx >= cfg.panel_count:
        return ""

    indent = "      " + "  " * idx
    angle = cfg.initial_angles_deg[idx]
    child = _panel_chain_xml(cfg, idx + 1)
    geom_mat = "mat_top_panel" if idx == cfg.panel_count // 2 else "mat_panel"
    panel_name = f"panel_{idx}"
    hinge_name = f"hinge_panel_{idx}"
    geom_name = f"geom_panel_{idx}"
    site_name = f"site_panel_{idx}_tip"
    joint_range = f"{cfg.hinge_range_min_deg:.4f} {cfg.hinge_range_max_deg:.4f}"
    # panel_0은 root hinge에서 바로 시작하고, 이후 panel은 이전 panel tip에 붙는다.
    hinge_pos_y = 0.0 if idx == 0 else cfg.panel_width
    body = f'''{indent}<body name="{panel_name}" pos="0 {hinge_pos_y:.6f} 0" euler="{angle:.6f} 0 0">
{indent}  <joint name="{hinge_name}" type="hinge" axis="{cfg.hinge_axis}" range="{joint_range}"
{indent}         limited="true" damping="{cfg.hinge_damping:.6f}" stiffness="{_hinge_stiffness(cfg):.6f}"
{indent}         springref="0" armature="{cfg.hinge_armature:.6f}"/>
{indent}  <geom name="{geom_name}" type="box" pos="0 {cfg.panel_width * 0.5:.6f} 0"
{indent}        size="{cfg.bag_length * 0.5:.6f} {cfg.panel_width * 0.5:.6f} {cfg.panel_thickness:.6f}"
{indent}        mass="{cfg.panel_mass:.6f}" material="{geom_mat}" condim="4" friction="1.0 0.04 0.004"/>
{indent}  <site name="{site_name}" pos="0 {cfg.panel_width:.6f} 0" size="0.008" rgba="0.1 0.85 0.1 1"/>
{child}{indent}</body>
'''
    return body


def _hidden_mass_xml(cfg: PanelDesignerConfig) -> str:
    if not cfg.hidden_mass_enabled:
        return ""
    mass_count = int(max(MIN_HIDDEN_MASS_COUNT, min(MAX_HIDDEN_MASS_COUNT, cfg.hidden_mass_count)))
    mass_each = cfg.hidden_mass_total / mass_count
    rx = abs(cfg.hidden_mass_range_x)
    ry = abs(cfg.hidden_mass_range_y)
    rz = abs(cfg.hidden_mass_range_z)

    def joints(label: str) -> str:
        ball = ""
        if cfg.hidden_mass_ball_joint:
            ball = (
                f'        <joint name="hidden_mass_{label}_ball" type="ball" '
                f'damping="{cfg.hidden_mass_slide_damping:.6f}" armature="{cfg.hidden_mass_slide_armature:.6f}"/>\n'
            )
        return (
            f'        <joint name="hidden_mass_{label}_slide_x" type="slide" axis="1 0 0" '
            f'range="{-rx:.6f} {rx:.6f}" damping="{cfg.hidden_mass_slide_damping:.6f}" '
            f'armature="{cfg.hidden_mass_slide_armature:.6f}" limited="true"/>\n'
            f'        <joint name="hidden_mass_{label}_slide_y" type="slide" axis="0 1 0" '
            f'range="{-ry:.6f} {ry:.6f}" damping="{cfg.hidden_mass_slide_damping:.6f}" '
            f'armature="{cfg.hidden_mass_slide_armature:.6f}" limited="true"/>\n'
            f'        <joint name="hidden_mass_{label}_slide_z" type="slide" axis="0 0 1" '
            f'range="{-rz:.6f} {rz:.6f}" damping="{cfg.hidden_mass_slide_damping:.6f}" '
            f'armature="{cfg.hidden_mass_slide_armature:.6f}" limited="true"/>\n'
            f"{ball}"
        )

    scale = cfg.hidden_mass_size_scale
    base_size = (0.030 * scale, 0.020 * scale, 0.016 * scale)
    if mass_count == 1:
        x_positions = [0.0]
    else:
        span = cfg.bag_length * 0.58
        x_positions = [-span * 0.5 + span * i / (mass_count - 1) for i in range(mass_count)]

    bodies = []
    for idx, x in enumerate(x_positions):
        label = f"{idx:02d}"
        layer = idx % 3
        y = cfg.panel_width * (0.40 + 0.12 * layer)
        z = cfg.panel_width * (0.42 + 0.10 * ((idx + 1) % 3))
        sx = base_size[0] * (1.08 if idx % 2 == 0 else 0.92)
        sy = base_size[1] * (0.95 if idx % 3 == 0 else 1.05)
        sz = base_size[2] * (1.00 if idx % 4 else 0.90)
        bodies.append(
            f'''
      <body name="hidden_mass_{label}" pos="{x:.6f} {y:.6f} {z:.6f}">
{joints(label).rstrip()}
        <geom name="geom_hidden_mass_{label}" type="ellipsoid" size="{sx:.6f} {sy:.6f} {sz:.6f}" mass="{mass_each:.6f}"
              material="mat_hidden_mass" condim="4" friction="1.0 0.03 0.003"/>
      </body>'''
        )
    return "\n".join(bodies) + "\n"


def _close_loop_xml(cfg: PanelDesignerConfig) -> str:
    """마지막 panel tip과 첫 root site를 tendon으로 닫는다.

    MuJoCo 3.1.x는 equality/connect의 site1/site2 문법을 지원하지 않으므로,
    현재 환경에서는 spatial tendon이 가장 안전한 폐루프 근사이다.
    """

    if not cfg.close_loop_enabled:
        return ""
    last_idx = cfg.panel_count - 1
    close_stiffness = max(1.0, 14.0 / max(cfg.close_loop_solref_time, 0.005))
    close_damping = max(0.8, close_stiffness * 0.004)
    return f'''
  <tendon>
    <spatial name="close_loop_panel_{last_idx}_to_root"
             stiffness="{close_stiffness:.6f}"
             damping="{close_damping:.6f}"
             springlength="0.0"
             width="0.004"
             rgba="1 0.35 0.05 0.85">
      <site site="site_panel_{last_idx}_tip"/>
      <site site="site_close_root"/>
    </spatial>
  </tendon>
'''


def build_designer_xml(cfg: PanelDesignerConfig) -> str:
    cfg = cfg.normalized()
    actuators = ""
    if not cfg.passive_hinge_mode:
        actuators = "\n".join(
            f'    <general name="panel_{i}_angle_deg_act" joint="hinge_panel_{i}" ctrllimited="true" '
            f'ctrlrange="{_actuator_ctrlrange(cfg.hinge_range_min_deg, cfg.hinge_range_max_deg)}" '
            f'gainprm="{_degree_actuator_gain(cfg.actuator_kp):.9f} 0 0" '
            f'biasprm="0 {-cfg.actuator_kp:.9f} 0"/>'
            for i in range(cfg.panel_count)
        )
    panel_chain = _panel_chain_xml(cfg)
    hidden_mass_xml = _hidden_mass_xml(cfg)
    close_loop_xml = _close_loop_xml(cfg)
    actuator_xml = f"  <actuator>\n{actuators}\n  </actuator>" if actuators else ""

    return f'''<mujoco model="panel_designer_open_panel">
  <compiler angle="degree" inertiafromgeom="true" autolimits="true"/>
  <option timestep="0.001" solver="Newton" iterations="80" tolerance="1e-9" cone="elliptic" impratio="5"/>

  <visual>
    <global azimuth="135" elevation="-25" offwidth="1280" offheight="720"/>
  </visual>

  <asset>
    <texture name="tex_floor" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.72 0.72 0.68" rgb2="0.62 0.62 0.58"/>
    <material name="mat_floor" texture="tex_floor" texrepeat="3 3" rgba="0.75 0.75 0.70 1"/>
    <material name="mat_panel" rgba="0.75 0.62 0.39 0.88"/>
    <material name="mat_top_panel" rgba="0.94 0.72 0.33 0.93"/>
    <material name="mat_hinge" rgba="1.0 0.52 0.02 1"/>
    <material name="mat_hidden_mass" rgba="0.78 0.08 0.05 0.45"/>
  </asset>

  <worldbody>
    <light name="key" pos="0 -1 1.6" dir="0.1 0.7 -1"/>
    <camera name="designer_cam" pos="0.58 -0.88 0.50" xyaxes="0.83 0.55 0 -0.25 0.38 0.89"/>
    <camera name="front_cam" pos="0 -0.92 0.26" xyaxes="1 0 0 0 0.24 0.97"/>
    <geom name="floor" type="plane" size="1.2 1.2 0.02" material="mat_floor"/>

    <body name="bag_frame" pos="0 {-cfg.panel_width * 2.8:.6f} {cfg.bag_frame_z:.6f}">
      <freejoint name="bag_frame_freejoint"/>
      <site name="site_close_root" pos="0 0 0" size="0.010" rgba="1 0.2 0.2 1"/>
      <geom name="designer_root_hinge_bar" type="capsule" fromto="{-cfg.bag_length * 0.52:.6f} 0 0 {cfg.bag_length * 0.52:.6f} 0 0"
            size="0.007" material="mat_hinge" contype="0" conaffinity="0" group="4"/>
{panel_chain}{hidden_mass_xml}    </body>
  </worldbody>
{close_loop_xml}
{actuator_xml}
</mujoco>
'''


def write_designer_xml(cfg: PanelDesignerConfig, path: Path = DESIGNER_XML) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_designer_xml(cfg), encoding="utf-8")
    return path


def main() -> int:
    path = write_designer_xml(PanelDesignerConfig())
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
