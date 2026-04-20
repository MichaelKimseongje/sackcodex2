"""MjSpec 기반 panel designer backend.

MuJoCo 3.2+에서는 MjSpec으로 body/geom/joint/actuator를 프로그래밍 방식으로
추가한 뒤 compile/to_xml 할 수 있다. 현재 Yolov9 환경처럼 MjSpec이 없는
MuJoCo 3.1.x에서는 기존 XML-string builder로 자동 fallback한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, radians
from pathlib import Path
from typing import Any

import mujoco

from panel_designer_builder import DESIGNER_XML, PanelDesignerConfig, write_designer_xml


@dataclass
class BackendBuildResult:
    """designer model 생성 결과."""

    path: Path
    backend: str
    message: str
    mujoco_version: str
    model: mujoco.MjModel | None = None


def mjspec_available() -> bool:
    """현재 Python MuJoCo binding에서 MjSpec API를 쓸 수 있는지 확인한다."""

    return hasattr(mujoco, "MjSpec")


def _rgba(name: str) -> list[float]:
    colors = {
        "floor": [0.75, 0.75, 0.70, 1.0],
        "panel": [0.75, 0.62, 0.39, 0.88],
        "top": [0.94, 0.72, 0.33, 0.93],
        "hinge": [1.0, 0.52, 0.02, 1.0],
        "mass": [0.78, 0.08, 0.05, 0.45],
    }
    return colors[name]


def _add_designer_panel_chain_mjspec(spec: Any, parent_body: Any, cfg: PanelDesignerConfig) -> list[Any]:
    """MjSpec API로 panel body, hinge joint, box geom, site를 순차 추가한다."""

    panel_bodies: list[Any] = []
    parent = parent_body
    hinge_axis = [float(x) for x in cfg.hinge_axis.split()]
    joint_range = [radians(cfg.hinge_range_min_deg), radians(cfg.hinge_range_max_deg)]

    for idx in range(cfg.panel_count):
        angle = radians(cfg.initial_angles_deg[idx])
        body = parent.add_body(
            name=f"panel_{idx}",
            pos=[0.0, 0.0 if idx == 0 else cfg.panel_width, 0.0],
            euler=[angle, 0.0, 0.0],
        )
        joint = body.add_joint(
            name=f"hinge_panel_{idx}",
            type=mujoco.mjtJoint.mjJNT_HINGE,
            axis=hinge_axis,
        )
        # MjSpec은 radian 기준으로 저장되므로 GUI degree 값을 radian으로 변환한다.
        joint.range = joint_range
        joint.limited = True
        joint.damping = cfg.hinge_damping
        joint.stiffness = 0.0 if cfg.passive_hinge_mode else cfg.hinge_stiffness
        joint.springref = 0.0
        joint.armature = cfg.hinge_armature

        body.add_geom(
            name=f"geom_panel_{idx}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0.0, cfg.panel_width * 0.5, 0.0],
            size=[cfg.bag_length * 0.5, cfg.panel_width * 0.5, cfg.panel_thickness],
            mass=cfg.panel_mass,
            rgba=_rgba("top" if idx == cfg.panel_count // 2 else "panel"),
            condim=4,
            friction=[1.0, 0.04, 0.004],
        )
        body.add_site(
            name=f"site_panel_{idx}_tip",
            pos=[0.0, cfg.panel_width, 0.0],
            size=[0.008, 0.0, 0.0],
            rgba=[0.1, 0.85, 0.1, 1.0],
        )

        panel_bodies.append(body)
        parent = body

    return panel_bodies


def _add_hidden_mass_mjspec(bag_frame: Any, cfg: PanelDesignerConfig) -> None:
    """MjSpec API로 hidden 3-clump mass와 3축 slide joint를 추가한다."""

    if not cfg.hidden_mass_enabled:
        return

    mass_count = cfg.hidden_mass_count
    mass_each = cfg.hidden_mass_total / mass_count
    ranges = [
        [-abs(cfg.hidden_mass_range_x), abs(cfg.hidden_mass_range_x)],
        [-abs(cfg.hidden_mass_range_y), abs(cfg.hidden_mass_range_y)],
        [-abs(cfg.hidden_mass_range_z), abs(cfg.hidden_mass_range_z)],
    ]
    scale = cfg.hidden_mass_size_scale
    base_size = [0.030 * scale, 0.020 * scale, 0.016 * scale]
    if mass_count == 1:
        x_positions = [0.0]
    else:
        span = cfg.bag_length * 0.58
        x_positions = [-span * 0.5 + span * i / (mass_count - 1) for i in range(mass_count)]
    axes = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    suffixes = ("x", "y", "z")

    for idx, x in enumerate(x_positions):
        label = f"{idx:02d}"
        layer = idx % 3
        pos = [x, cfg.panel_width * (0.40 + 0.12 * layer), cfg.panel_width * (0.42 + 0.10 * ((idx + 1) % 3))]
        size = [
            base_size[0] * (1.08 if idx % 2 == 0 else 0.92),
            base_size[1] * (0.95 if idx % 3 == 0 else 1.05),
            base_size[2] * (1.00 if idx % 4 else 0.90),
        ]
        body = bag_frame.add_body(name=f"hidden_mass_{label}", pos=pos)
        for axis, suffix, joint_range in zip(axes, suffixes, ranges):
            joint = body.add_joint(
                name=f"hidden_mass_{label}_slide_{suffix}",
                type=mujoco.mjtJoint.mjJNT_SLIDE,
                axis=axis,
            )
            joint.range = joint_range
            joint.limited = True
            joint.damping = cfg.hidden_mass_slide_damping
            joint.armature = cfg.hidden_mass_slide_armature
        if cfg.hidden_mass_ball_joint:
            joint = body.add_joint(
                name=f"hidden_mass_{label}_ball",
                type=mujoco.mjtJoint.mjJNT_BALL,
            )
            joint.damping = cfg.hidden_mass_slide_damping
            joint.armature = cfg.hidden_mass_slide_armature
        body.add_geom(
            name=f"geom_hidden_mass_{label}",
            type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            size=size,
            mass=mass_each,
            rgba=_rgba("mass"),
            condim=4,
            friction=[1.0, 0.03, 0.003],
        )


def _add_position_actuators_mjspec(spec: Any, cfg: PanelDesignerConfig) -> None:
    """MjSpec API로 degree 입력을 받는 hinge servo actuator를 추가한다."""

    if cfg.passive_hinge_mode:
        return

    ctrlrange = [cfg.hinge_range_min_deg, cfg.hinge_range_max_deg]
    for idx in range(cfg.panel_count):
        actuator = spec.add_actuator(
            name=f"panel_{idx}_angle_deg_act",
            target=f"hinge_panel_{idx}",
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            ctrllimited=True,
            ctrlrange=ctrlrange,
        )
        actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        # viewer ctrl은 degree, joint 좌표는 radian이므로 gain에서 degree->radian 변환을 적용한다.
        actuator.gainprm = [cfg.actuator_kp * pi / 180.0, 0.0, 0.0]
        actuator.biasprm = [0.0, -cfg.actuator_kp, 0.0]


def _add_close_loop_mjspec(spec: Any, cfg: PanelDesignerConfig) -> None:
    """MjSpec spatial tendon으로 마지막 panel tip과 root site를 부드럽게 연결한다."""

    if not cfg.close_loop_enabled:
        return
    close_stiffness = max(1.0, 14.0 / max(cfg.close_loop_solref_time, 0.005))
    close_damping = max(0.8, close_stiffness * 0.004)
    tendon = spec.add_tendon(
        name=f"close_loop_panel_{cfg.panel_count - 1}_to_root",
        stiffness=close_stiffness,
        damping=close_damping,
        springlength=[0.0, 0.0],
        width=0.004,
        rgba=[1.0, 0.35, 0.05, 0.85],
    )
    tendon.wrap_site(f"site_panel_{cfg.panel_count - 1}_tip")
    tendon.wrap_site("site_close_root")


def build_designer_spec(cfg: PanelDesignerConfig) -> Any:
    """MjSpec으로 designer model을 처음부터 구성한다.

    이 함수는 MuJoCo 3.2+에서만 실행된다.
    """

    if not mjspec_available():
        raise RuntimeError(
            f"현재 mujoco=={mujoco.__version__}에는 MjSpec이 없습니다. MuJoCo 3.2+가 필요합니다."
        )

    cfg = cfg.normalized()
    spec = mujoco.MjSpec()
    spec.modelname = "panel_designer_mjspec"
    spec.opt.timestep = 0.001
    spec.opt.iterations = 80
    spec.opt.tolerance = 1e-9
    spec.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.opt.impratio = 5

    spec.worldbody.add_light(name="key", pos=[0.0, -1.0, 1.6], dir=[0.1, 0.7, -1.0])
    spec.worldbody.add_camera(
        name="designer_cam",
        pos=[0.58, -0.88, 0.50],
        xyaxes=[0.83, 0.55, 0.0, -0.25, 0.38, 0.89],
    )
    spec.worldbody.add_camera(
        name="front_cam",
        pos=[0.0, -0.92, 0.26],
        xyaxes=[1.0, 0.0, 0.0, 0.0, 0.24, 0.97],
    )
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[1.2, 1.2, 0.02],
        rgba=_rgba("floor"),
    )

    bag_frame = spec.worldbody.add_body(
        name="bag_frame",
        pos=[0.0, -cfg.panel_width * 2.8, cfg.bag_frame_z],
    )
    freejoint = bag_frame.add_freejoint()
    freejoint.name = "bag_frame_freejoint"
    bag_frame.add_site(
        name="site_close_root",
        pos=[0.0, 0.0, 0.0],
        size=[0.010, 0.0, 0.0],
        rgba=[1.0, 0.2, 0.2, 1.0],
    )

    bag_frame.add_geom(
        name="designer_root_hinge_bar",
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[-cfg.bag_length * 0.52, 0.0, 0.0, cfg.bag_length * 0.52, 0.0, 0.0],
        size=[0.007, 0.0, 0.0],
        rgba=_rgba("hinge"),
        contype=0,
        conaffinity=0,
    )

    _add_designer_panel_chain_mjspec(spec, bag_frame, cfg)
    _add_hidden_mass_mjspec(bag_frame, cfg)
    _add_close_loop_mjspec(spec, cfg)
    _add_position_actuators_mjspec(spec, cfg)
    return spec


def write_designer_xml_mjspec(cfg: PanelDesignerConfig, path: Path = DESIGNER_XML) -> BackendBuildResult:
    """MjSpec로 model을 compile하고 XML로 저장한다."""

    spec = build_designer_spec(cfg)
    model = spec.compile()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(spec.to_xml(), encoding="utf-8")
    return BackendBuildResult(
        path=path,
        backend="mjspec",
        message="MjSpec backend로 body/geom/joint/actuator를 추가하고 compile/to_xml 저장했습니다.",
        mujoco_version=mujoco.__version__,
        model=model,
    )


def write_designer_xml_auto(cfg: PanelDesignerConfig, path: Path = DESIGNER_XML) -> BackendBuildResult:
    """가능하면 MjSpec, 불가능하면 XML builder fallback으로 designer model을 저장한다."""

    if mjspec_available():
        try:
            return write_designer_xml_mjspec(cfg, path)
        except Exception as exc:
            fallback_path = write_designer_xml(cfg, path)
            return BackendBuildResult(
                path=fallback_path,
                backend="xml_fallback_after_mjspec_error",
                message=f"MjSpec backend 실패 후 XML fallback 사용: {exc}",
                mujoco_version=mujoco.__version__,
                model=None,
            )

    fallback_path = write_designer_xml(cfg, path)
    return BackendBuildResult(
        path=fallback_path,
        backend="xml_fallback_no_mjspec",
        message=f"현재 mujoco=={mujoco.__version__}에는 MjSpec이 없어 XML builder fallback을 사용했습니다.",
        mujoco_version=mujoco.__version__,
        model=None,
    )


def main() -> int:
    result = write_designer_xml_auto(PanelDesignerConfig())
    print(f"path={result.path}")
    print(f"backend={result.backend}")
    print(f"mujoco_version={result.mujoco_version}")
    print(result.message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
