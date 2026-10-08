"""trame + PyVista front end.

    python -m mos.app [--spec model.mos] [--port 8080]

The drawer holds the model description (see mos/core/spec.py), interactive load tools,
and view controls; the 3D view is a server-side-rendered VTK scene.
"""
from __future__ import annotations

import re
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyvista as pv
from pyvista.trame.ui import plotter_ui
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import html
from trame.widgets import vuetify3 as v3

from mos import viz
from mos.core.solver import solve_case
from mos.core.spec import build, parse_spec
from mos.templates import TEMPLATES

pv.OFF_SCREEN = True


def main(argv=None):
    """Build the UI and start the trame server (blocks until stopped)."""
    server = get_server(client_type="vue3")
    state, ctrl = server.state, server.controller
    parser = server.cli
    parser.add_argument("--spec", help="model description file to open at startup")
    args, _ = parser.parse_known_args(argv)

    pl = pv.Plotter()
    S = SimpleNamespace(grid=None, scalars={}, u_pts=None, res=None, spec=None, model=None, case=None)

    # ------------------------------------------------------------ scene
    def update_scene(reset_camera=False):
        """Redraw the 3D view from the current solution and view controls."""
        is_dark = state.theme == "dark"
        bg_color = "#18181b" if is_dark else "#f8fafc"
        pl.background_color = bg_color

        if S.grid is None or state.field not in S.scalars:
            pl.clear()
            ctrl.view_update()
            return

        arr, kind = S.scalars[state.field]
        div = viz.STRESS_UNITS[state.stress_unit] if kind == "stress" else 1.0
        label = f"{state.field} [{state.stress_unit}]" if kind == "stress" else state.field
        base = S.grid
        g = base.copy()
        g.points = base.points + float(state.scale) * S.u_pts
        g.point_data["s"] = arr / div

        if state.clip_axis != "none":
            ax = "xyz".index(state.clip_axis)
            g = g.clip(normal=state.clip_axis,
                       origin=viz.clip_origin(base.bounds, ax, float(state.clip_pos)))

        lo, hi = float(arr.min() / div), float(arr.max() / div)
        if hi - lo < 1e-30:
            hi = lo + 1e-30

        pl.clear()
        sb_args = viz.scalar_bar_args(label, is_dark=is_dark)
        pl.add_mesh(g, scalars="s", cmap="turbo", clim=[lo, hi],
                    show_edges=bool(state.show_edges), scalar_bar_args=sb_args)

        # Faint outline of undeformed shape
        if float(state.scale) > 0:
            outline_color = "#71717a" if is_dark else "#94a3b8"
            pl.add_mesh(base.extract_surface().extract_feature_edges(30),
                        color=outline_color, line_width=1.5, opacity=0.45)

        bounds = base.bounds
        diag = float(np.linalg.norm(np.ptp(np.asarray(base.points), axis=0)))

        # Highlight selected face
        if bool(state.show_face_highlight) and state.selected_face and (state.active_tab == "loads" or state.load_target == "face"):
            face_mesh = viz.create_face_mesh(state.selected_face, bounds)
            if face_mesh is not None:
                pl.add_mesh(face_mesh, color="#06b6d4", opacity=0.4,
                            show_edges=True, line_width=2.5, edge_color="#22d3ee")

        # Highlight selected point marker
        sphere_fn = getattr(pv, "Sphere", None)
        if (state.load_target == "point" or bool(state.show_point_marker)) and sphere_fn is not None:
            pt_coord = (float(state.point_x), float(state.point_y), float(state.point_z))
            try:
                pt_mesh = sphere_fn(radius=max(0.02 * diag, 1e-4), center=pt_coord)
                pl.add_mesh(pt_mesh, color="#f59e0b", opacity=0.9)
            except Exception:
                pass

        # Render 3D load glyphs from solved model load case
        if bool(state.show_glyphs) and S.case is not None and S.model is not None:
            try:
                glyphs = viz.build_load_glyphs(S.case, S.model.mesh, diag)
                for g_mesh, g_color in glyphs:
                    pl.add_mesh(g_mesh, color=g_color)
            except Exception:
                pass

        # Render interactive preview arrow if in Loads tab
        if state.active_tab == "loads" and bool(state.show_preview_glyph):
            try:
                if state.load_target == "face":
                    faces = viz.face_definitions(bounds)
                    c = faces.get(state.selected_face, {}).get("centroid", np.array([0, 0, 0]))
                else:
                    c = np.array([float(state.point_x), float(state.point_y), float(state.point_z)])

                if state.load_mode == "force":
                    f_vec = _parse_vec3(state.force_vec)
                    arr_prev = viz.create_arrow_glyph(c, f_vec, diag)
                    if arr_prev is not None:
                        pl.add_mesh(arr_prev, color="#22c55e")
                elif state.load_mode == "moment":
                    m_vec = _get_moment_vector(state, bounds)
                    arr_prev = viz.create_moment_glyph(c, m_vec, diag)
                    if arr_prev is not None:
                        pl.add_mesh(arr_prev, color="#c084fc")
            except Exception:
                pass

        if reset_camera:
            pl.reset_camera()
        ctrl.view_update()

    # ------------------------------------------------------------ vector helpers
    def _parse_vec3(s: str) -> np.ndarray:
        try:
            parts = [float(x.strip()) for x in s.split(",")]
            if len(parts) == 3:
                return np.array(parts, float)
        except Exception:
            pass
        return np.array([0.0, -1.0, 0.0])

    def _get_moment_vector(st, bounds) -> np.ndarray:
        try:
            mag = float(st.moment_mag)
        except Exception:
            mag = 1e5

        if st.moment_axis_type == "normal":
            norm = viz.face_definitions(bounds).get(st.selected_face, {}).get("normal", np.array([1, 0, 0]))
            return mag * (norm / np.linalg.norm(norm))
        elif st.moment_axis_type == "x":
            return np.array([mag, 0.0, 0.0])
        elif st.moment_axis_type == "y":
            return np.array([0.0, mag, 0.0])
        elif st.moment_axis_type == "z":
            return np.array([0.0, 0.0, mag])
        else:
            axis = _parse_vec3(st.custom_axis)
            norm = np.linalg.norm(axis)
            if norm < 1e-12:
                return np.array([mag, 0.0, 0.0])
            return mag * (axis / norm)

    # ------------------------------------------------------------ solve
    def run_solve():
        """Parse the model text, solve it, refresh the plot and status panel; report errors in the UI."""
        try:
            state.status, state.status_type = "Solving ...", "info"
            state.flush()
            spec = parse_spec(state.spec_text)
            model, case = build(spec)
            res = solve_case(model, case)
            S.res, S.grid = res, viz.to_pyvista(model.mesh)
            S.spec, S.model, S.case = spec, model, case
            S.scalars = viz.compute_scalars(res)
            S.u_pts = np.ascontiguousarray(res.u_nodal.T)
            state.field_items = list(S.scalars)
            if state.field not in S.scalars:
                state.field = "von Mises"

            factor = viz.suggest_scale(S.grid.points, res.u_nodal)
            state.auto_scale = factor
            state.scale_max = max(10.0, 3.0 * factor)
            state.scale_step = max(0.001, state.scale_max / 300.0)

            # Keep deformation scale 1:1 by default
            if state.scale is None or state.scale <= 0:
                state.scale = 1.0
            _update_scale_display(state.scale)

            # Update bounds and face info
            bounds = S.grid.bounds
            state.model_bounds = list(bounds)
            _sync_face_info(bounds)

            div = viz.STRESS_UNITS[state.stress_unit]
            n_dof = res.u.size
            umax = float(S.scalars["|u|"][0].max())
            state.max_u_str = f"{umax:.4g} m ({umax*1e3:.4g} mm)"
            state.status = (
                f"{S.grid.n_cells} cells, {S.grid.n_points} vertices, {n_dof} DOFs\n"
                f"max |u| = {umax:.4g} m ({umax*1e3:.4g} mm)\n"
                f"max von Mises = {S.scalars['von Mises'][0].max() / div:.4g} {state.stress_unit}\n"
                f"reactions = {np.array2string(res.reactions, precision=4)}\n"
                f"equilibrium error = {res.equilibrium_error:.2e}")
            state.status_type = "success" if res.equilibrium_error < 1e-6 else "warning"
            _update_preview_snippet()
            update_scene(reset_camera=True)
        except Exception as e:
            traceback.print_exc()
            state.status = f"{type(e).__name__}: {e}"
            state.status_type = "error"

    def _update_scale_display(scale_val):
        s = float(scale_val)
        if abs(s - 1.0) < 1e-4:
            state.scale_display = "1:1 (True Scale)"
        elif s <= 0:
            state.scale_display = "0x (Undeformed)"
        elif abs(s - float(state.auto_scale)) < 1e-3 * max(float(state.auto_scale), 1.0):
            state.scale_display = f"Auto ({s:.2g}x)"
        else:
            state.scale_display = f"{s:.2g}x"

    def _sync_face_info(bounds):
        faces = viz.face_definitions(bounds)
        if state.selected_face in faces:
            info = faces[state.selected_face]
            c = info["centroid"]
            state.face_centroid_str = f"({c[0]:.2g}, {c[1]:.2g}, {c[2]:.2g})"
            n = info["normal"]
            state.face_normal_str = f"[{n[0]:.0f}, {n[1]:.0f}, {n[2]:.0f}]"
            if state.origin_is_centroid:
                state.point_x = float(c[0])
                state.point_y = float(c[1])
                state.point_z = float(c[2])

    def _update_preview_snippet():
        bounds = S.grid.bounds if S.grid else tuple(state.model_bounds)
        lines = []
        if state.load_target == "face":
            face = state.selected_face
            reg_id = f"face_{face.replace('+', 'pos_').replace('-', 'neg_')}"
            cond = _face_condition(face, bounds)
            lines.append(f"region   {reg_id} facets {cond}")
            if state.load_mode == "force":
                lines.append(f"traction {reg_id} {state.force_vec}")
            else:
                m_vec = _get_moment_vector(state, bounds)
                lines.append(f"moment   {reg_id} {m_vec[0]:.4g}, {m_vec[1]:.4g}, {m_vec[2]:.4g}")
        else:
            px, py, pz = float(state.point_x), float(state.point_y), float(state.point_z)
            reg_id = "pt_applied"
            lines.append(f"region   {reg_id} nearest {px:.4g}, {py:.4g}, {pz:.4g}")
            if state.load_mode == "force":
                lines.append(f"force    {reg_id} {state.force_vec}")
            else:
                m_vec = _get_moment_vector(state, bounds)
                lines.append(f"moment   {reg_id} {m_vec[0]:.4g}, {m_vec[1]:.4g}, {m_vec[2]:.4g}")
        state.load_preview_snippet = "\n".join(lines)

    ctrl.run_solve = run_solve

    # ------------------------------------------------------------ load application helpers
    def apply_current_load():
        """Apply the currently configured load (face/point, force/moment) to the model text and solve."""
        _update_preview_snippet()
        text = state.spec_text.rstrip()
        state.spec_text = text + f"\n\n# Applied load ({state.load_target} {state.load_mode})\n" + state.load_preview_snippet + "\n"
        state.active_tab = "model"
        run_solve()

    def _face_condition(face_key, bounds):
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        if face_key == "+x":
            return f"x > {xmax:.4g} - 1e-9"
        elif face_key == "-x":
            return f"x < {xmin:.4g} + 1e-9"
        elif face_key == "+y":
            return f"y > {ymax:.4g} - 1e-9"
        elif face_key == "-y":
            return f"y < {ymin:.4g} + 1e-9"
        elif face_key == "+z":
            return f"z > {zmax:.4g} - 1e-9"
        elif face_key == "-z":
            return f"z < {zmin:.4g} + 1e-9"
        return "x > 0"

    def _set_force_preset(preset_key: str):
        mag = state.force_mag if state.force_mag else "1e5"
        if preset_key == "down":
            state.force_vec = f"0, -{mag}, 0"
        elif preset_key == "up":
            state.force_vec = f"0, {mag}, 0"
        elif preset_key == "tension":
            state.force_vec = f"{mag}, 0, 0"
        elif preset_key == "compression":
            state.force_vec = f"-{mag}, 0, 0"
        elif preset_key == "lateral_z":
            state.force_vec = f"0, 0, {mag}"
        _update_preview_snippet()
        update_scene()

    ctrl.apply_current_load = apply_current_load
    ctrl.set_force_preset_down = lambda: _set_force_preset("down")
    ctrl.set_force_preset_up = lambda: _set_force_preset("up")
    ctrl.set_force_preset_tension = lambda: _set_force_preset("tension")
    ctrl.set_force_preset_compression = lambda: _set_force_preset("compression")
    ctrl.set_force_preset_lateral_z = lambda: _set_force_preset("lateral_z")

    ctrl.set_scale_1 = lambda: state.update({"scale": 1.0})
    ctrl.set_scale_auto = lambda: state.update({"scale": float(state.auto_scale)})
    ctrl.set_scale_0 = lambda: state.update({"scale": 0.0})

    # ------------------------------------------------------------ initial state
    state.update({
        "theme": "dark",
        "spec_text": TEMPLATES[next(iter(TEMPLATES))],
        "template": None, "templates": list(TEMPLATES),
        "status": "Edit the model on the left, then press Solve.", "status_type": "info",
        "field": "von Mises", "field_items": [],
        "stress_unit": "MPa", "stress_units": list(viz.STRESS_UNITS),
        "scale": 1.0, "scale_max": 3.0, "scale_step": 0.01,
        "auto_scale": 1.0, "scale_display": "1:1 (True Scale)",
        "max_u_str": "",
        "show_edges": True, "clip_axis": "none", "clip_axes": ["none", "x", "y", "z"],
        "clip_pos": 0.5,
        # Interactive load tools
        "active_tab": "model",
        "load_target": "face",
        "load_mode": "force",
        "selected_face": "+x",
        "face_options": [
            {"title": "+X (Tip / Right Face)", "value": "+x"},
            {"title": "-X (Root / Left Face)", "value": "-x"},
            {"title": "+Y (Top Face)", "value": "+y"},
            {"title": "-Y (Bottom Face)", "value": "-y"},
            {"title": "+Z (Front Face)", "value": "+z"},
            {"title": "-Z (Back Face)", "value": "-z"},
        ],
        "face_centroid_str": "(10, 0, 0)",
        "face_normal_str": "[1, 0, 0]",
        "point_x": 10.0, "point_y": 0.0, "point_z": 0.0,
        "point_preset": "Tip Center",
        "point_presets": ["Tip Center", "Root Center", "Top Midspan", "Bottom Midspan", "Origin", "Custom"],
        "origin_is_centroid": True,
        # Force settings
        "force_mag": "1e5",
        "force_vec": "0, -1e5, 0",
        # Moment settings
        "moment_axis_type": "normal",
        "moment_axis_types": [
            {"title": "Face Normal (Torsion)", "value": "normal"},
            {"title": "X Axis [1, 0, 0]", "value": "x"},
            {"title": "Y Axis [0, 1, 0]", "value": "y"},
            {"title": "Z Axis [0, 0, 1]", "value": "z"},
            {"title": "Custom Axis", "value": "custom"},
        ],
        "custom_axis": "1, 0, 0",
        "moment_mag": "1e5",
        # Visual cues
        "show_glyphs": True,
        "show_preview_glyph": True,
        "show_face_highlight": True,
        "show_point_marker": True,
        "load_preview_snippet": "traction face_pos_x 0, -1e5, 0",
        "model_bounds": [0, 10, -0.5, 0.5, -0.5, 0.5],
    })

    if args.spec:
        state.spec_text = Path(args.spec).read_text()

    @state.change("template")
    def _load_template(template, **kw):
        if template in TEMPLATES:
            state.spec_text = TEMPLATES[template]

    @state.change("scale")
    def _scale_changed(scale, **kw):
        _update_scale_display(scale)
        update_scene()

    @state.change("theme")
    def _theme_changed(theme, **kw):
        update_scene()

    @state.change("field", "stress_unit", "show_edges", "clip_axis", "clip_pos",
                   "show_glyphs", "show_face_highlight", "show_point_marker", "show_preview_glyph", "active_tab")
    def _redraw(**kw):
        update_scene()

    @state.change("selected_face")
    def _face_changed(selected_face, **kw):
        bounds = S.grid.bounds if S.grid else tuple(state.model_bounds)
        _sync_face_info(bounds)
        _update_preview_snippet()
        update_scene()

    @state.change("point_preset")
    def _preset_changed(point_preset, **kw):
        bounds = S.grid.bounds if S.grid else tuple(state.model_bounds)
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        cx, cy, cz = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)
        if point_preset == "Tip Center":
            state.point_x, state.point_y, state.point_z = float(xmax), float(cy), float(cz)
        elif point_preset == "Root Center":
            state.point_x, state.point_y, state.point_z = float(xmin), float(cy), float(cz)
        elif point_preset == "Top Midspan":
            state.point_x, state.point_y, state.point_z = float(cx), float(ymax), float(cz)
        elif point_preset == "Bottom Midspan":
            state.point_x, state.point_y, state.point_z = float(cx), float(ymin), float(cz)
        elif point_preset == "Origin":
            state.point_x, state.point_y, state.point_z = 0.0, 0.0, 0.0
        _update_preview_snippet()
        update_scene()

    @state.change("point_x", "point_y", "point_z", "force_vec", "force_mag",
                   "moment_mag", "moment_axis_type", "custom_axis", "load_target", "load_mode")
    def _load_params_changed(**kw):
        _update_preview_snippet()
        update_scene()

    # ------------------------------------------------------------ layout
    with SinglePageWithDrawerLayout(server) as layout:
        layout.root.theme = ("theme",)
        layout.title.set_text("MoS Analyzer")
        layout.drawer.width = 490

        # Toolbar controls
        with layout.toolbar:
            v3.VSpacer()
            v3.VBtn(
                text=("scale_display",),
                size="small",
                variant="outlined",
                classes="mr-2",
                click=ctrl.set_scale_1,
            )
            v3.VBtn(
                icon="mdi-theme-light-dark",
                variant="text",
                density="comfortable",
                click="theme = theme === 'dark' ? 'light' : 'dark'",
                classes="mr-2",
            )

        with layout.drawer:
            with v3.VContainer(fluid=True, classes="pa-2"):
                # Navigation tabs
                with v3.VTabs(v_model=("active_tab",), grow=True, density="compact", classes="mb-2"):
                    v3.VTab("Model", value="model")
                    v3.VTab("Apply Loads", value="loads")
                    v3.VTab("View & Results", value="view")

                # TAB 1: MODEL EDITOR
                with v3.VWindow(v_model=("active_tab",)):
                    with v3.VWindowItem(value="model"):
                        v3.VSelect(v_model=("template",), items=("templates",), label="Load example",
                                   density="compact", hide_details=True, classes="mb-2")
                        v3.VTextarea(v_model=("spec_text",), rows=17, variant="outlined",
                                     hide_details=True, no_resize=True,
                                     style="font-family: monospace; font-size: 12px;")
                        v3.VBtn("Solve", block=True, color="primary", classes="my-2", click=ctrl.run_solve)
                        html.Pre("{{ status }}", style="white-space: pre-wrap; font-size: 11px; padding: 6px; border-radius: 4px; background: rgba(128,128,128,0.1);")

                    # TAB 2: INTERACTIVE LOADS (Face & Point Selection)
                    with v3.VWindowItem(value="loads"):
                        # Target Selection: Face vs Point
                        with v3.VCard(variant="tonal", classes="pa-2 mb-2"):
                            html.Div("1. Target Location", classes="text-subtitle-2 font-weight-bold mb-1")
                            with v3.VBtnToggle(v_model=("load_target",), mandatory=True, density="compact", divided=True, color="primary", classes="mb-2"):
                                v3.VBtn("Face Target", value="face", size="small")
                                v3.VBtn("Point Target", value="point", size="small")

                            # --- FACE TARGET CONTROLS ---
                            with html.Div(v_show="load_target === 'face'"):
                                v3.VSelect(
                                    v_model=("selected_face",),
                                    items=("face_options",),
                                    item_title="title",
                                    item_value="value",
                                    label="Select Face",
                                    density="compact",
                                    hide_details=True,
                                    classes="mb-2"
                                )
                                # Quick face chips
                                with html.Div(classes="d-flex flex-wrap ga-1 mb-1"):
                                    v3.VBtn("+X (Tip)", size="x-small", variant="outlined", click="selected_face = '+x'")
                                    v3.VBtn("-X (Root)", size="x-small", variant="outlined", click="selected_face = '-x'")
                                    v3.VBtn("+Y (Top)", size="x-small", variant="outlined", click="selected_face = '+y'")
                                    v3.VBtn("-Y (Bottom)", size="x-small", variant="outlined", click="selected_face = '-y'")
                                    v3.VBtn("+Z (Front)", size="x-small", variant="outlined", click="selected_face = '+z'")
                                    v3.VBtn("-Z (Back)", size="x-small", variant="outlined", click="selected_face = '-z'")
                                html.Div("Centroid: {{ face_centroid_str }}  |  Normal: {{ face_normal_str }}",
                                         classes="text-caption text-medium-emphasis mt-1")

                            # --- POINT TARGET CONTROLS ---
                            with html.Div(v_show="load_target === 'point'"):
                                v3.VSelect(
                                    v_model=("point_preset",),
                                    items=("point_presets",),
                                    label="Point Location Preset",
                                    density="compact",
                                    hide_details=True,
                                    classes="mb-2"
                                )
                                with html.Div(classes="d-flex ga-1 mb-1"):
                                    v3.VTextField(v_model=("point_x",), label="X (m)", type="number", step=0.1,
                                                  density="compact", hide_details=True)
                                    v3.VTextField(v_model=("point_y",), label="Y (m)", type="number", step=0.1,
                                                  density="compact", hide_details=True)
                                    v3.VTextField(v_model=("point_z",), label="Z (m)", type="number", step=0.1,
                                                  density="compact", hide_details=True)

                        # Load Type Selection: Force vs Moment
                        with v3.VCard(variant="outlined", classes="pa-2 mb-2"):
                            html.Div("2. Load Type & Vector", classes="text-subtitle-2 font-weight-bold mb-1")
                            with v3.VBtnToggle(v_model=("load_mode",), mandatory=True, density="compact", divided=True, color="secondary", classes="mb-2"):
                                v3.VBtn("Force Vector", value="force", size="small")
                                v3.VBtn("Moment (Torque)", value="moment", size="small")

                            # --- FORCE CONTROLS ---
                            with html.Div(v_show="load_mode === 'force'"):
                                with html.Div(classes="d-flex flex-wrap ga-1 mb-2"):
                                    v3.VBtn("↓ Down (-Y)", size="x-small", variant="tonal", click=ctrl.set_force_preset_down)
                                    v3.VBtn("↑ Up (+Y)", size="x-small", variant="tonal", click=ctrl.set_force_preset_up)
                                    v3.VBtn("→ Pull (+X)", size="x-small", variant="tonal", click=ctrl.set_force_preset_tension)
                                    v3.VBtn("← Push (-X)", size="x-small", variant="tonal", click=ctrl.set_force_preset_compression)
                                    v3.VBtn("↗ Side (+Z)", size="x-small", variant="tonal", click=ctrl.set_force_preset_lateral_z)

                                v3.VTextField(
                                    v_model=("force_vec",),
                                    label="Force Vector [Fx, Fy, Fz] (N or Pa)",
                                    density="compact",
                                    hide_details=True,
                                    classes="mb-2"
                                )

                            # --- MOMENT CONTROLS ---
                            with html.Div(v_show="load_mode === 'moment'"):
                                v3.VSelect(
                                    v_model=("moment_axis_type",),
                                    items=("moment_axis_types",),
                                    item_title="title",
                                    item_value="value",
                                    label="Moment Axis",
                                    density="compact",
                                    hide_details=True,
                                    classes="mb-2"
                                )
                                v3.VTextField(
                                    v_model=("custom_axis",),
                                    label="Custom Axis (ax, ay, az)",
                                    density="compact",
                                    hide_details=True,
                                    classes="mb-2",
                                    v_show="moment_axis_type === 'custom'"
                                )
                                v3.VTextField(
                                    v_model=("moment_mag",),
                                    label="Magnitude M (N·m)",
                                    density="compact",
                                    hide_details=True,
                                    classes="mb-2"
                                )

                        # Live Code Preview & Apply Button
                        with v3.VCard(variant="tonal", classes="pa-2 mb-2"):
                            html.Div("3. Code Preview & Confirmation", classes="text-caption font-weight-bold mb-1")
                            html.Pre("{{ load_preview_snippet }}",
                                     style="font-family: monospace; font-size: 11px; padding: 4px; background: rgba(0,0,0,0.2); border-radius: 4px;")
                            v3.VBtn("Apply Load to Model & Solve", block=True, color="primary",
                                    density="default", classes="mt-2", click=ctrl.apply_current_load)

                    # TAB 3: VIEW & RESULTS
                    with v3.VWindowItem(value="view"):
                        v3.VSelect(v_model=("field",), items=("field_items",), label="Show Field",
                                   density="compact", hide_details=True, classes="mb-2")
                        v3.VSelect(v_model=("stress_unit",), items=("stress_units",), label="Stress Unit",
                                   density="compact", hide_details=True, classes="mb-3")

                        # Deformation scale controls with clear 1:1 default
                        with v3.VCard(variant="tonal", classes="pa-2 mb-3"):
                            with html.Div(classes="d-flex align-center justify-space-between mb-1"):
                                html.Span("Deformation Scale", classes="text-caption font-weight-bold")
                                v3.VChip("{{ scale_display }}", color="primary", size="x-small", classes="font-weight-bold")
                            with html.Div(classes="d-flex ga-1 mb-2"):
                                v3.VBtn("1:1 True", size="x-small", variant="tonal", click=ctrl.set_scale_1)
                                v3.VBtn("Auto (~10%)", size="x-small", variant="tonal", click=ctrl.set_scale_auto)
                                v3.VBtn("0 Undeformed", size="x-small", variant="tonal", click=ctrl.set_scale_0)
                            v3.VSlider(v_model=("scale",), min=0, max=("scale_max",), step=("scale_step",),
                                       hide_details=True, density="compact")
                            html.Div("{{ max_u_str }}", classes="text-caption text-medium-emphasis mt-1")

                        v3.VSwitch(v_model=("show_edges",), label="Mesh edges", hide_details=True, density="compact")
                        v3.VSwitch(v_model=("show_glyphs",), label="Show 3D load glyphs (applied loads)",
                                   hide_details=True, density="compact")
                        v3.VSwitch(v_model=("show_preview_glyph",), label="Show interactive load preview arrow",
                                   hide_details=True, density="compact")
                        v3.VSwitch(v_model=("show_face_highlight",), label="Highlight selected face",
                                   hide_details=True, density="compact")

                        v3.VDivider(classes="my-2")
                        v3.VSelect(v_model=("clip_axis",), items=("clip_axes",), label="Section plane normal",
                                   density="compact", hide_details=True, classes="mb-2")
                        v3.VSlider(v_model=("clip_pos",), min=0, max=1, step=0.01,
                                   disabled=("clip_axis === 'none'",), label="Section position",
                                   hide_details=True, density="compact")

        with layout.content:
            with v3.VContainer(fluid=True, classes="pa-0 fill-height"):
                view = plotter_ui(pl, mode="server")
                ctrl.view_update = view.update

    server.start()
