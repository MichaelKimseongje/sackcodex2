"""Open-panel sack panel/joint designer GUI.

MuJoCo viewer 안에서 body/joint를 직접 추가하는 대신,
이 GUI가 XML을 재생성하고 viewer를 재시작한다.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox

import mujoco

from panel_designer_builder import DESIGNER_XML, MAX_PANELS, PROJECT_DIR, PanelDesignerConfig, closed_panel_angles
from panel_designer_mjspec_backend import mjspec_available, write_designer_xml_auto


CONFIG_JSON = PROJECT_DIR / "out" / "panel_designer_config.json"


class PanelDesignerGUI(tk.Tk):
    """패널 개수/힌지/질량을 수정하고 MuJoCo viewer를 재시작하는 GUI."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Open Panel Sack Designer")
        self.geometry("1120x720")
        self.minsize(980, 620)
        self.viewer_process: subprocess.Popen | None = None
        self.last_backend = tk.StringVar(value="last backend: none")
        defaults = PanelDesignerConfig().normalized()

        self.panel_count = tk.IntVar(value=defaults.panel_count)
        self.bag_length = tk.DoubleVar(value=defaults.bag_length)
        self.panel_width = tk.DoubleVar(value=defaults.panel_width)
        self.panel_thickness = tk.DoubleVar(value=defaults.panel_thickness)
        self.panel_mass = tk.DoubleVar(value=defaults.panel_mass)
        self.hinge_axis_label = tk.StringVar(value=self._axis_label_from_value(defaults.hinge_axis))
        self.hinge_min = tk.DoubleVar(value=defaults.hinge_range_min_deg)
        self.hinge_max = tk.DoubleVar(value=defaults.hinge_range_max_deg)
        self.hinge_damping = tk.DoubleVar(value=defaults.hinge_damping)
        self.hinge_stiffness = tk.DoubleVar(value=defaults.hinge_stiffness)
        self.hinge_armature = tk.DoubleVar(value=defaults.hinge_armature)
        self.passive_hinge_mode = tk.BooleanVar(value=defaults.passive_hinge_mode)
        self.actuator_kp = tk.DoubleVar(value=defaults.actuator_kp)
        self.bag_frame_z = tk.DoubleVar(value=defaults.bag_frame_z)
        self.hidden_mass_enabled = tk.BooleanVar(value=defaults.hidden_mass_enabled)
        self.hidden_mass_count = tk.IntVar(value=defaults.hidden_mass_count)
        self.hidden_mass_total = tk.DoubleVar(value=defaults.hidden_mass_total)
        self.hidden_mass_size_scale = tk.DoubleVar(value=defaults.hidden_mass_size_scale)
        self.hidden_mass_ball_joint = tk.BooleanVar(value=defaults.hidden_mass_ball_joint)
        self.hidden_mass_slide_damping = tk.DoubleVar(value=defaults.hidden_mass_slide_damping)
        self.hidden_mass_slide_armature = tk.DoubleVar(value=defaults.hidden_mass_slide_armature)
        self.hidden_mass_range_x = tk.DoubleVar(value=defaults.hidden_mass_range_x)
        self.hidden_mass_range_y = tk.DoubleVar(value=defaults.hidden_mass_range_y)
        self.hidden_mass_range_z = tk.DoubleVar(value=defaults.hidden_mass_range_z)
        self.close_loop_enabled = tk.BooleanVar(value=defaults.close_loop_enabled)
        self.close_loop_solref_time = tk.DoubleVar(value=defaults.close_loop_solref_time)
        self.initial_angles = [
            tk.DoubleVar(value=defaults.initial_angles_deg[i] if i < len(defaults.initial_angles_deg) else 0.0)
            for i in range(MAX_PANELS)
        ]

        self._build_ui()
        self._refresh_panel_rows()

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)

        main = ttk.Frame(root)
        main.pack(fill=tk.BOTH, expand=True)

        notebook = ttk.Notebook(main)
        notebook.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        geometry_tab = ttk.Frame(notebook, padding=8)
        hinge_tab = ttk.Frame(notebook, padding=8)
        mass_tab = ttk.Frame(notebook, padding=8)
        run_tab = ttk.Frame(notebook, padding=8)
        notebook.add(geometry_tab, text="Geometry")
        notebook.add(hinge_tab, text="Hinge")
        notebook.add(mass_tab, text="Mass / Loop")
        notebook.add(run_tab, text="Run")

        right = ttk.Frame(main, width=430)
        right.pack(side=tk.RIGHT, fill=tk.BOTH)

        self._build_geometry_panel(geometry_tab)
        self._build_joint_panel(hinge_tab)
        self._build_mass_loop_panel(mass_tab)
        self._build_backend_panel(run_tab)
        self._build_action_panel(run_tab)
        self._build_panel_table(right)
        self._build_log_panel(root)

    def _build_geometry_panel(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="1. Panel Geometry", padding=8)
        box.pack(fill=tk.X, pady=(0, 8))

        self._spinbox(box, "panel_count", self.panel_count, 4, MAX_PANELS, 1, row=0, command=self._refresh_panel_rows)
        self._entry(box, "bag_length [m]", self.bag_length, row=1)
        self._entry(box, "panel_width [m]", self.panel_width, row=2)
        self._entry(box, "panel_thickness [m]", self.panel_thickness, row=3)
        self._entry(box, "panel_mass [kg]", self.panel_mass, row=4)
        self._entry(box, "bag_frame_z [m]", self.bag_frame_z, row=5)

    def _build_mass_loop_panel(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="3. Hidden Mass / Close Loop", padding=8)
        box.pack(fill=tk.X, pady=(0, 8))

        ttk.Checkbutton(box, text="hidden_mass_C/L/R 사용", variable=self.hidden_mass_enabled).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=2
        )
        self._spinbox(box, "hidden_mass_count", self.hidden_mass_count, 1, 12, 1, row=1)
        self._entry(box, "hidden_mass_total [kg]", self.hidden_mass_total, row=2)
        self._entry(box, "hidden_mass_size_scale", self.hidden_mass_size_scale, row=3)
        ttk.Checkbutton(box, text="hidden_mass free rotation(ball joint)", variable=self.hidden_mass_ball_joint).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=2
        )
        self._entry(box, "hidden_mass_slide_damping", self.hidden_mass_slide_damping, row=5)
        self._entry(box, "hidden_mass_slide_armature", self.hidden_mass_slide_armature, row=6)
        self._entry(box, "hidden_mass_range_x [m]", self.hidden_mass_range_x, row=7)
        self._entry(box, "hidden_mass_range_y [m]", self.hidden_mass_range_y, row=8)
        self._entry(box, "hidden_mass_range_z [m]", self.hidden_mass_range_z, row=9)
        ttk.Checkbutton(box, text="마지막 panel을 첫 root에 연결(5-1 close loop)", variable=self.close_loop_enabled).grid(
            row=10, column=0, columnspan=2, sticky="w", pady=(12, 2)
        )
        self._entry(box, "close_loop_time (smaller=stronger)", self.close_loop_solref_time, row=11)

    def _build_joint_panel(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="2. Hinge / Actuator", padding=8)
        box.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(box, text="hinge_axis").grid(row=0, column=0, sticky="w", pady=2)
        axis = ttk.Combobox(box, textvariable=self.hinge_axis_label, values=["X axis", "Y axis", "Z axis"], width=12, state="readonly")
        axis.grid(row=0, column=1, sticky="ew", pady=2)
        self._entry(box, "range_min [deg]", self.hinge_min, row=1)
        self._entry(box, "range_max [deg]", self.hinge_max, row=2)
        self._entry(box, "damping", self.hinge_damping, row=3)
        self._entry(box, "stiffness", self.hinge_stiffness, row=4)
        self._entry(box, "armature", self.hinge_armature, row=5)
        ttk.Checkbutton(
            box,
            text="Passive free hinge mode (체크 시 actuator 없이 min/max 안에서 자유 운동)",
            variable=self.passive_hinge_mode,
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=2)
        self._entry(box, "actuator_kp", self.actuator_kp, row=7)

        ttk.Button(box, text="Fast passive response preset", command=self._set_fast_passive_response).grid(
            row=8, column=0, columnspan=2, sticky="ew", pady=(8, 2)
        )

        ttk.Button(box, text="Low / flat template", command=lambda: self._set_scaled_template(0.5)).grid(
            row=9, column=0, columnspan=2, sticky="ew", pady=2
        )
        ttk.Button(box, text="Closed panel template", command=lambda: self._set_closed_template()).grid(
            row=10, column=0, columnspan=2, sticky="ew", pady=2
        )
        ttk.Button(box, text="Lifted droop template", command=lambda: self._set_scaled_template(1.15, first_angle=18.0)).grid(
            row=11, column=0, columnspan=2, sticky="ew", pady=2
        )

    def _build_backend_panel(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Backend", padding=8)
        box.pack(fill=tk.X, pady=(0, 8))
        available = mjspec_available()
        msg = "MjSpec available: body/geom/joint를 MjSpec API로 생성" if available else (
            f"MjSpec unavailable in mujoco {mujoco.__version__}; XML fallback 사용"
        )
        ttk.Label(box, text=msg, wraplength=250).pack(fill=tk.X)
        ttk.Label(box, textvariable=self.last_backend).pack(fill=tk.X, pady=(4, 0))

    def _build_action_panel(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="3-4. Generate / Viewer", padding=8)
        box.pack(fill=tk.X)

        ttk.Button(box, text="Generate XML (degree actuator)", command=self.generate_xml).pack(fill=tk.X, pady=2)
        ttk.Button(box, text="Open Viewer", command=self.open_viewer).pack(fill=tk.X, pady=2)
        ttk.Button(box, text="Regenerate + Reload Viewer", command=self.regenerate_reload_viewer).pack(fill=tk.X, pady=2)
        ttk.Button(box, text="Validate Loaded Model", command=self.validate_model).pack(fill=tk.X, pady=2)
        ttk.Button(box, text="Stop Viewer", command=self.stop_viewer).pack(fill=tk.X, pady=(10, 2))

    def _build_panel_table(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Panel Initial Angles", padding=8)
        box.pack(fill=tk.X)
        ttk.Label(box, text="panel").grid(row=0, column=0, sticky="w")
        ttk.Label(box, text="initial_angle_deg").grid(row=0, column=1, sticky="w")
        ttk.Label(box, text="설명").grid(row=0, column=2, sticky="w")
        self.panel_rows = []
        for i in range(MAX_PANELS):
            label = ttk.Label(box, text=f"panel_{i}")
            entry = ttk.Entry(box, textvariable=self.initial_angles[i], width=12)
            desc = ttk.Label(box, text=self._panel_desc(i))
            label.grid(row=i + 1, column=0, sticky="w", pady=3)
            entry.grid(row=i + 1, column=1, sticky="w", pady=3)
            desc.grid(row=i + 1, column=2, sticky="w", pady=3)
            self.panel_rows.append((label, entry, desc))

    def _build_log_panel(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="Log", padding=8)
        box.pack(fill=tk.X, pady=(8, 0))
        self.log_text = tk.Text(box, height=7, wrap="word")
        self.log_text.pack(fill=tk.X)
        self.log("GUI 준비 완료. 먼저 Generate XML을 누른 뒤 Open Viewer를 누르세요.")
        self.log(f"MuJoCo version: {mujoco.__version__}, MjSpec available: {mjspec_available()}")

    def _entry(self, parent: ttk.Frame, label: str, variable: tk.DoubleVar, row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(parent, textvariable=variable, width=14).grid(row=row, column=1, sticky="ew", pady=2)

    def _spinbox(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.IntVar,
        from_: int,
        to: int,
        increment: int,
        row: int,
        command=None,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        ttk.Spinbox(parent, from_=from_, to=to, increment=increment, textvariable=variable, width=12, command=command).grid(
            row=row, column=1, sticky="ew", pady=2
        )

    def _panel_desc(self, idx: int) -> str:
        labels = {
            0: "첫 번째 rigid panel (hidden mass 아님)",
            1: "side wall",
            2: "top 또는 mid panel",
            3: "opposite side wall",
            4: "opposite lower panel",
            5: "optional extra panel",
        }
        return labels.get(idx, "extra articulated panel")

    def _refresh_panel_rows(self) -> None:
        count = int(self.panel_count.get())
        for idx, widgets in enumerate(self.panel_rows):
            state = "normal" if idx < count else "disabled"
            for widget in widgets:
                widget.configure(state=state)

    def _set_template(self, values: list[float]) -> None:
        for var, value in zip(self.initial_angles, values):
            var.set(value)
        self.log(f"template 적용: {values[: int(self.panel_count.get())]}")

    def _set_closed_template(self) -> None:
        self._set_template(closed_panel_angles(int(self.panel_count.get())))

    def _set_scaled_template(self, scale: float, first_angle: float = 0.0) -> None:
        values = closed_panel_angles(int(self.panel_count.get()))
        values = [first_angle] + [v * scale for v in values[1:]]
        self._set_template(values)

    def _set_fast_passive_response(self) -> None:
        self.passive_hinge_mode.set(True)
        self.hinge_damping.set(0.05)
        self.hinge_stiffness.set(0.0)
        self.hinge_armature.set(0.0002)
        self.panel_mass.set(0.025)
        self.close_loop_solref_time.set(0.005)
        self.log("fast passive preset: passive=ON, damping=0.05, stiffness=0, armature=0.0002, panel_mass=0.025")

    def _axis_label_from_value(self, axis: str) -> str:
        if axis == "0 1 0":
            return "Y axis"
        if axis == "0 0 1":
            return "Z axis"
        return "X axis"

    def _axis_value(self) -> str:
        label = self.hinge_axis_label.get()
        if label.startswith("Y"):
            return "0 1 0"
        if label.startswith("Z"):
            return "0 0 1"
        return "1 0 0"

    def _config(self) -> PanelDesignerConfig:
        return PanelDesignerConfig(
            panel_count=int(self.panel_count.get()),
            bag_length=float(self.bag_length.get()),
            panel_width=float(self.panel_width.get()),
            panel_thickness=float(self.panel_thickness.get()),
            panel_mass=float(self.panel_mass.get()),
            hinge_axis=self._axis_value(),
            hinge_range_min_deg=float(self.hinge_min.get()),
            hinge_range_max_deg=float(self.hinge_max.get()),
            hinge_damping=float(self.hinge_damping.get()),
            hinge_stiffness=float(self.hinge_stiffness.get()),
            hinge_armature=float(self.hinge_armature.get()),
            passive_hinge_mode=bool(self.passive_hinge_mode.get()),
            actuator_kp=float(self.actuator_kp.get()),
            bag_frame_z=float(self.bag_frame_z.get()),
            hidden_mass_enabled=bool(self.hidden_mass_enabled.get()),
            hidden_mass_count=int(self.hidden_mass_count.get()),
            hidden_mass_total=float(self.hidden_mass_total.get()),
            hidden_mass_size_scale=float(self.hidden_mass_size_scale.get()),
            hidden_mass_ball_joint=bool(self.hidden_mass_ball_joint.get()),
            hidden_mass_slide_damping=float(self.hidden_mass_slide_damping.get()),
            hidden_mass_slide_armature=float(self.hidden_mass_slide_armature.get()),
            hidden_mass_range_x=float(self.hidden_mass_range_x.get()),
            hidden_mass_range_y=float(self.hidden_mass_range_y.get()),
            hidden_mass_range_z=float(self.hidden_mass_range_z.get()),
            close_loop_enabled=bool(self.close_loop_enabled.get()),
            close_loop_solref_time=float(self.close_loop_solref_time.get()),
            initial_angles_deg=[float(v.get()) for v in self.initial_angles],
        ).normalized()

    def generate_xml(self) -> Path | None:
        try:
            cfg = self._config()
            result = write_designer_xml_auto(cfg)
            path = result.path
            self.last_backend.set(f"last backend: {result.backend}")
            CONFIG_JSON.parent.mkdir(parents=True, exist_ok=True)
            payload = cfg.__dict__.copy()
            payload["backend"] = result.backend
            payload["backend_message"] = result.message
            payload["mujoco_version"] = result.mujoco_version
            CONFIG_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            self.log(f"XML 생성 완료: {path}")
            self.log(f"backend={result.backend} | {result.message}")
            return path
        except Exception as exc:
            messagebox.showerror("Generate XML failed", str(exc))
            self.log(f"XML 생성 실패: {exc}")
            return None

    def open_viewer(self) -> None:
        path = self.generate_xml()
        if path is None:
            return
        if self.viewer_process and self.viewer_process.poll() is None:
            self.log("viewer가 이미 실행 중입니다. Reload를 사용하세요.")
            return
        cmd = [sys.executable, "-m", "mujoco.viewer", f"--mjcf={DESIGNER_XML}"]
        self.viewer_process = subprocess.Popen(cmd, cwd=str(PROJECT_DIR.parent))
        self.log(f"viewer 실행: {' '.join(cmd)}")

    def regenerate_reload_viewer(self) -> None:
        self.stop_viewer()
        self.open_viewer()

    def stop_viewer(self) -> None:
        proc = self.viewer_process
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
            self.log("viewer 종료 완료")
        self.viewer_process = None

    def validate_model(self) -> None:
        try:
            if not DESIGNER_XML.exists():
                self.generate_xml()
            model = mujoco.MjModel.from_xml_path(str(DESIGNER_XML))
            joint_names = [model.joint(i).name for i in range(model.njnt)]
            body_names = [model.body(i).name for i in range(model.nbody)]
            panel_bodies = [name for name in body_names if name.startswith("panel_")]
            hinge_joints = [name for name in joint_names if name.startswith("hinge_panel_")]
            hidden_mass_joints = [name for name in joint_names if name.startswith("hidden_mass_")]
            actuator_names = [model.actuator(i).name for i in range(model.nu)]
            self.log("loaded model validation:")
            self.log(f"  bodies={model.nbody}, joints={model.njnt}, actuators={model.nu}")
            self.log(f"  panel_bodies={panel_bodies}")
            self.log(f"  hinge_joints={hinge_joints}")
            self.log(f"  hidden_mass_joints={hidden_mass_joints}")
            self.log(f"  actuator_names={actuator_names}")
            self.log(f"  actuator_ctrlrange(deg)={model.actuator_ctrlrange.tolist()}")
            tendon_names = [model.tendon(i).name for i in range(model.ntendon)]
            self.log(f"  tendons={tendon_names}")
            self.log(f"  xml={DESIGNER_XML}")
        except Exception as exc:
            messagebox.showerror("Validation failed", str(exc))
            self.log(f"검증 실패: {exc}")

    def log(self, message: str) -> None:
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def destroy(self) -> None:
        self.stop_viewer()
        super().destroy()


def main() -> int:
    app = PanelDesignerGUI()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
