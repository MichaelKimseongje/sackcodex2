from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from mujoco_sack_pile.benchmark_definition import PILE_DIFFICULTIES
from mujoco_sack_pile.scene_generator import SceneGenerator
from mujoco_sack_pile.scene_generator import SACK_VARIANTS


class ManualTeleop:
    """viewer key 입력으로 gripper와 scoop를 수동 조작한다."""

    def __init__(self, env):
        self.env = env
        self.selected_tool = "gripper"
        self.pos_step = 0.015
        self.rot_step = np.deg2rad(8.0)

    def handle_key(self, keycode: int):
        if keycode in (ord("1"),):
            self.selected_tool = "gripper"
            print("manual_tool=gripper")
            return
        if keycode in (ord("2"),):
            self.selected_tool = "scoop"
            print("manual_tool=scoop")
            return
        if keycode in (ord("H"), ord("h")):
            self._print_help()
            return
        if keycode in (ord("["),):
            self.env.set_gripper_width(max(0.010, self.env.data.ctrl[self.env.left_finger_act] - 0.004))
            print(f"gripper_width={self.env.data.ctrl[self.env.left_finger_act]:.3f}")
            return
        if keycode in (ord("]"),):
            self.env.set_gripper_width(min(0.040, self.env.data.ctrl[self.env.left_finger_act] + 0.004))
            print(f"gripper_width={self.env.data.ctrl[self.env.left_finger_act]:.3f}")
            return

        pos_delta = np.zeros(3, dtype=np.float64)
        if keycode in (ord("W"), ord("w")):
            pos_delta[0] += self.pos_step
        elif keycode in (ord("S"), ord("s")):
            pos_delta[0] -= self.pos_step
        elif keycode in (ord("A"), ord("a")):
            pos_delta[1] += self.pos_step
        elif keycode in (ord("D"), ord("d")):
            pos_delta[1] -= self.pos_step
        elif keycode in (ord("R"), ord("r")):
            pos_delta[2] += self.pos_step
        elif keycode in (ord("F"), ord("f")):
            pos_delta[2] -= self.pos_step
        if np.linalg.norm(pos_delta) > 0:
            self._move_tool(pos_delta=pos_delta, euler_delta=np.zeros(3, dtype=np.float64))
            return

        euler_delta = np.zeros(3, dtype=np.float64)
        if keycode in (ord("I"), ord("i")):
            euler_delta[1] += self.rot_step
        elif keycode in (ord("K"), ord("k")):
            euler_delta[1] -= self.rot_step
        elif keycode in (ord("J"), ord("j")):
            euler_delta[2] += self.rot_step
        elif keycode in (ord("L"), ord("l")):
            euler_delta[2] -= self.rot_step
        elif keycode in (ord("U"), ord("u")):
            euler_delta[0] += self.rot_step
        elif keycode in (ord("O"), ord("o")):
            euler_delta[0] -= self.rot_step
        if np.linalg.norm(euler_delta) > 0:
            self._move_tool(pos_delta=np.zeros(3, dtype=np.float64), euler_delta=euler_delta)

    def _move_tool(self, pos_delta: np.ndarray, euler_delta: np.ndarray):
        mocap_id = self.env.gripper_mocap_id if self.selected_tool == "gripper" else self.env.scoop_mocap_id
        current_pos = self.env.data.mocap_pos[mocap_id].copy()
        current_quat = self.env.data.mocap_quat[mocap_id].copy()
        target_pos = current_pos + pos_delta
        current_euler = self._quat_to_euler(current_quat)
        target_quat = self.env.euler_to_quat(current_euler + euler_delta)
        if self.selected_tool == "gripper":
            self.env.set_gripper_pose(target_pos, target_quat)
        else:
            self.env.set_scoop_pose(target_pos, target_quat)
        print(
            f"{self.selected_tool}_pose=pos{tuple(round(v, 3) for v in target_pos)} "
            f"euler{tuple(round(v, 3) for v in current_euler + euler_delta)}"
        )

    @staticmethod
    def _quat_to_euler(quat: np.ndarray) -> np.ndarray:
        w, x, y, z = quat
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            pitch = np.sign(sinp) * (np.pi / 2.0)
        else:
            pitch = np.arcsin(sinp)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return np.array([roll, pitch, yaw], dtype=np.float64)

    @staticmethod
    def _print_help():
        print("manual_help:")
        print("  1 / 2 : gripper / scoop 선택")
        print("  W S : +X / -X")
        print("  A D : +Y / -Y")
        print("  R F : +Z / -Z")
        print("  I K : pitch + / -")
        print("  J L : yaw + / -")
        print("  U O : roll + / -")
        print("  [ ] : gripper 열기 / 닫기")
        print("  H   : 도움말 다시 출력")

class ViewerControls:
    """viewer 키 입력과 마우스 perturb 도움말을 함께 관리한다."""

    def __init__(self, env, enable_manual: bool):
        self.env = env
        self.paused = False
        self.teleop = ManualTeleop(env) if enable_manual else None

    def handle_key(self, keycode: int):
        if keycode in (32,):
            self.paused = not self.paused
            print(f"viewer_paused={self.paused}")
            return
        if keycode in (ord("M"), ord("m")):
            self.print_help()
            return
        if self.teleop is not None:
            self.teleop.handle_key(keycode)

    def print_help(self):
        print("mouse_help:")
        print("  left drag / right drag / wheel : 카메라 이동")
        print("  double-click body : 객체 선택")
        print("  Ctrl + drag : 선택한 객체 perturb")
        print("  Space : 일시정지 / 재개")
        print("  M : 도움말 다시 출력")
        if self.teleop is not None:
            self.teleop._print_help()


def main():
    try:
        import mujoco
        import mujoco.viewer
    except ModuleNotFoundError as exc:
        raise SystemExit("MuJoCo Python 패키지가 설치되어 있지 않습니다. `pip install mujoco` 후 다시 실행해 주세요.") from exc

    from mujoco_sack_pile.baselines.heuristics import BASELINES
    from mujoco_sack_pile.environment import SackPileEnv

    parser = argparse.ArgumentParser(
        description="MuJoCo task-driven benchmark runner for support-state formation under shape and pile uncertainty"
    )
    parser.add_argument("--baseline", choices=sorted(BASELINES.keys()), default="top_grasp_flat_scoop_drag")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-id", type=str, default=None)
    parser.add_argument("--sack-count", type=int, default=None)
    parser.add_argument("--target-family", choices=sorted(SACK_VARIANTS.keys()), default=None)
    parser.add_argument("--pile-difficulty", choices=PILE_DIFFICULTIES, default=None)
    parser.add_argument("--settle-seconds", type=float, default=5.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--fixed-camera", action="store_true")
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--manual-control", action="store_true")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    generator = SceneGenerator(base_dir)
    episode_id = args.episode_id or f"{args.baseline}_seed{args.seed}"
    scene = generator.generate_episode(
        seed=args.seed,
        episode_id=episode_id,
        sack_count=args.sack_count,
        target_variant=args.target_family,
        pile_difficulty=args.pile_difficulty,
    )
    env = SackPileEnv(scene=scene, log_dir=base_dir / "mujoco_sack_pile" / "logs")
    settle_report = env.reset(settle_seconds=args.settle_seconds, verify_stability=True)

    runner = BASELINES[args.baseline]
    if args.headless:
        if args.preview_only:
            print(f"preview_scene={scene.xml_path}")
            print(f"benchmark={scene.benchmark_name}")
            print(f"research_question={scene.research_question}")
            if scene.target_case is not None:
                print(f"target_case={scene.target_case.case_id}")
            print(f"settle_stable={settle_report.stable}")
            print(f"settle_failure_tags={','.join(settle_report.failure_tags) if settle_report.failure_tags else 'none'}")
            print("preview_only headless 모드에서는 baseline 없이 benchmark case 생성만 수행했습니다.")
            return
        runner(env, viewer=None)
        metrics = env.finalize_metrics()
        env.save_episode_log(args.baseline, metrics)
        print(f"baseline={args.baseline}")
        print(f"scene_xml={scene.xml_path}")
        print(f"benchmark={scene.benchmark_name}")
        print(f"research_question={scene.research_question}")
        if scene.target_case is not None:
            print(f"target_case={scene.target_case.case_id}")
        print(f"settle_stable={settle_report.stable}")
        print(f"settle_failure_tags={','.join(settle_report.failure_tags) if settle_report.failure_tags else 'none'}")
        print(f"support_success={metrics.support_success}")
        print(f"support_state_score={metrics.support_state_score:.3f}")
        print(f"scoop_insertion_depth={metrics.scoop_insertion_depth:.3f}")
        print(f"micro_lift_stability={metrics.micro_lift_stability:.3f}")
        print(f"failure_tags={','.join(metrics.failure_tags) if metrics.failure_tags else 'none'}")
        return

    controls = ViewerControls(env, enable_manual=args.manual_control)
    controls.print_help()

    with mujoco.viewer.launch_passive(
        env.model,
        env.data,
        key_callback=controls.handle_key,
        show_left_ui=True,
        show_right_ui=True,
    ) as viewer:
        if args.fixed_camera:
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "overview")
        if args.preview_only:
            print(f"preview_scene={scene.xml_path}")
            print(f"benchmark={scene.benchmark_name}")
            print(f"research_question={scene.research_question}")
            if scene.target_case is not None:
                print(f"target_case={scene.target_case.case_id}")
            print(f"settle_stable={settle_report.stable}")
            print(f"settle_failure_tags={','.join(settle_report.failure_tags) if settle_report.failure_tags else 'none'}")
            print("preview_only=True, baseline은 실행하지 않고 benchmark case만 표시합니다.")
        else:
            runner(env, viewer=viewer)
            metrics = env.finalize_metrics()
            env.render_viewer(viewer)
            env.save_episode_log(args.baseline, metrics)
            print(f"support_success={metrics.support_success}")
            print(f"support_state_score={metrics.support_state_score:.3f}")
            print(f"failure_tags={metrics.failure_tags}")
        while viewer.is_running():
            if controls.paused:
                env.render_viewer(viewer)
                time.sleep(env.model.opt.timestep)
                continue
            env.step(1, viewer=viewer, sleep=True)


if __name__ == "__main__":
    main()
