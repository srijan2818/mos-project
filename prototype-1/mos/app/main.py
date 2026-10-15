"""trame + PyVista front end.

    python -m mos.app [--spec model.mos] [--port 8080]

The drawer holds interactive geometric/CAD model builders, materials, supports,
loads, and view controls; the 3D view is a server-side-rendered VTK scene.
"""
from __future__ import annotations

import json
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
from mos.core.cad import is_step_file, load_step_mesh  # noqa: F401 (may be used by templates)
from mos.core.image_to_cad import image_to_step
from mos.core.solver import solve_case
from mos.core.regions import cad_boundary_name_for_side
from mos.core.spec import build, parse_spec
from mos.templates import TEMPLATES

pv.OFF_SCREEN = True

# ---------------------------------------------------------------------------
# Predefined libraries
# ---------------------------------------------------------------------------
MATERIAL_PRESETS = {
    "Steel":     {"E": 210e9, "nu": 0.30, "rho": 7850},
    "Aluminum":  {"E": 68.9e9, "nu": 0.33, "rho": 2700},
    "Titanium":  {"E": 113.8e9, "nu": 0.34, "rho": 4430},
    "Concrete":  {"E": 30.0e9, "nu": 0.20, "rho": 2400},
    "CastIron":  {"E": 120.0e9, "nu": 0.26, "rho": 7200},
    "Custom":    {"E": 200e9, "nu": 0.30, "rho": 7800},
}

CAD_PRESETS = [
    {"title": "Structural L-Bracket (STEP)", "value": "cad_models/bracket.step"},
    {"title": "Stepped Shaft with Keyway (STEP)", "value": "cad_models/shaft.step"},
    {"title": "Tensile Plate with Hole (STEP)", "value": "cad_models/plate_hole.step"},
    {"title": "Cantilever Beam (STEP)", "value": "cad_models/beam.step"},
    {"title": "Drawing Extrusion Profile (STEP)", "value": "cad_models/drawing_extruded.step"},
    {"title": "Custom STEP file path …", "value": "custom"},
]

SUPPORT_COMP_OPTIONS = [
    {"title": "Fixed / Clamped (XYZ)", "value": "xyz"},
    {"title": "Pinned (XYZ — no moment)", "value": "xyz"},
    {"title": "Roller — free X (YZ)", "value": "yz"},
    {"title": "Roller — free Z (XY)", "value": "xy"},
    {"title": "Symmetry — X axis (YZ)", "value": "yz"},
    {"title": "Axial only (X)", "value": "x"},
    {"title": "Vertical only (Y)", "value": "y"},
    {"title": "Lateral only (Z)", "value": "z"},
]

_COMP_LABEL = {
    "xyz": "Fixed XYZ", "yz": "Roller (YZ)", "xy": "Roller (XY)",
    "x": "Axial (X)", "y": "Vertical (Y)", "z": "Lateral (Z)",
}
_FACE_LABEL_SHORT = {"+x": "+X", "-x": "-X", "+y": "+Y", "-y": "-Y", "+z": "+Z", "-z": "-Z"}
_FACE_NAME_SAFE = {"+x": "pos_x", "-x": "neg_x", "+y": "pos_y", "-y": "neg_y", "+z": "pos_z", "-z": "neg_z"}


def _safe_region_name(face_name: str, prefix: str = "reg") -> str:
    """Convert any UI face key into a valid spec region identifier."""
    token = str(face_name or "").strip()
    if not token:
        return f"{prefix}_0"
    token = _FACE_NAME_SAFE.get(token, token)
    token = re.sub(r"[^A-Za-z0-9_]", "_", token)
    token = token.strip("_")
    if not token:
        token = prefix
    if token[0].isdigit():
        token = f"{prefix}_{token}"
    return token


def _unique_region_name(text: str, stem: str) -> str:
    """Choose a spec region name not already declared in user or generated text."""
    declared = {m.group(1) for m in re.finditer(
        r"^\s*region\s+([^\s#]+)", text, flags=re.IGNORECASE | re.MULTILINE)}
    root = _safe_region_name(stem, "ui_region")
    candidate, index = f"ui_{root}", 1
    while candidate in declared:
        candidate = f"ui_{root}_{index}"
        index += 1
    return candidate


def _resolved_face_name(face_name: str, mesh_ref=None) -> str:
    """Resolve a generic axis face to the corresponding named CAD boundary.

    The UI can still hold generic values such as ``-x`` when a STEP model is
    first selected. Once its mesh is available, choose the CAD boundary whose
    surface centroid is furthest toward that side instead of generating a
    coordinate predicate against stale/default bounds.
    """
    boundaries = getattr(mesh_ref, "boundaries", None) if mesh_ref is not None else None
    if not boundaries:
        return face_name
    if face_name in boundaries:
        return face_name
    if face_name in _FACE_NAME_SAFE:
        name = cad_boundary_name_for_side(mesh_ref, face_name)
        return name if name is not None else face_name
    # Preserve an explicit CAD face if it exists. If it became stale because a
    # different STEP model was selected, fall back deterministically rather than
    # silently inventing a coordinate-defined region.
    if face_name and face_name.startswith("cad_face_"):
        return next(iter(boundaries))
    return face_name


# ---------------------------------------------------------------------------
def main(argv=None):
    """Build the UI and start the trame server (blocks until stopped)."""
    server = get_server(client_type="vue3")
    state, ctrl = server.state, server.controller
    parser = server.cli
    parser.add_argument("--spec", help="model description file to open at startup")
    args, _ = parser.parse_known_args(argv)

    pl = pv.Plotter()
    S = SimpleNamespace(grid=None, scalars={}, u_pts=None, res=None,
                        spec=None, model=None, case=None, main_actor=None)

    # ------------------------------------------------------------------ scene
    def update_scene(reset_camera=False):
        """Redraw the 3D view from the current solution and view controls."""
        is_dark = state.theme == "dark"
        pl.background_color = "#18181b" if is_dark else "#f8fafc"

        if S.grid is None or state.field not in S.scalars:
            pl.clear()
            ctrl.view_update()
            return

        arr, kind = S.scalars[state.field]
        div = viz.STRESS_UNITS[state.stress_unit] if kind == "stress" else 1.0
        label = f"{state.field} [{state.stress_unit}]" if kind == "stress" else state.field
        g = S.grid.copy()
        g.points = S.grid.points + float(state.scale) * S.u_pts
        g.point_data["s"] = arr / div

        if state.clip_axis != "none":
            ax = "xyz".index(state.clip_axis)
            g = g.clip(normal=state.clip_axis,
                       origin=viz.clip_origin(S.grid.bounds, ax, float(state.clip_pos)))

        lo, hi = float(arr.min() / div), float(arr.max() / div)
        if hi - lo < 1e-30:
            hi = lo + 1e-30

        pl.clear()
        sb_args = viz.scalar_bar_args(label, is_dark=is_dark)
        S.main_actor = pl.add_mesh(
            g, scalars="s", cmap="turbo", clim=[lo, hi],
            show_edges=bool(state.show_edges), scalar_bar_args=sb_args,
            pickable=True,
        )

        if float(state.scale) > 0:
            outline_color = "#71717a" if is_dark else "#94a3b8"
            try:
                surf = S.grid.extract_surface()
                edges = surf.extract_feature_edges(30)
                pl.add_mesh(edges, color=outline_color, line_width=1.5, opacity=0.45, pickable=False)
            except Exception:
                pass

        bounds = S.grid.bounds
        diag = float(np.linalg.norm(np.ptp(np.asarray(S.grid.points), axis=0)))
        mesh_ref = S.model.mesh if S.model else None

        # Highlight selected face
        if bool(state.show_face_highlight) and state.selected_face:
            face_mesh = viz.create_face_mesh(state.selected_face, bounds, mesh=mesh_ref)
            if face_mesh is not None:
                pl.add_mesh(face_mesh, color="#06b6d4", opacity=0.55,
                            show_edges=True, line_width=2.5, edge_color="#22d3ee", pickable=False)

        # Highlight selected point / target marker
        if bool(state.show_point_marker):
            pt_coord = (float(state.point_x), float(state.point_y), float(state.point_z))
            try:
                sph = pv.Sphere(radius=max(0.02 * diag, 1e-4), center=pt_coord)
                pl.add_mesh(sph, color="#f59e0b", opacity=0.9, pickable=False)
            except Exception:
                pass

        # 3D load glyphs after solve
        if bool(state.show_glyphs) and S.case is not None:
            try:
                for g_mesh, g_color in viz.build_load_glyphs(S.case, S.model.mesh, diag):
                    pl.add_mesh(g_mesh, color=g_color, pickable=False)
            except Exception:
                pass

        # Interactive preview arrow / moment arc
        if bool(state.show_preview_glyph) and state.active_tab == "loads":
            try:
                faces = viz.face_definitions(bounds, mesh=mesh_ref)
                if state.load_target == "face":
                    c = faces.get(state.selected_face, {}).get("centroid", np.zeros(3))
                else:
                    c = np.array([float(state.point_x), float(state.point_y), float(state.point_z)])

                if state.load_mode == "force":
                    arr_prev = viz.create_arrow_glyph(c, _parse_vec3(state.force_vec), diag)
                    if arr_prev is not None:
                        pl.add_mesh(arr_prev, color="#22c55e", pickable=False)
                elif state.load_mode == "moment":
                    m_vec = _get_moment_vector(state, bounds, mesh_ref)
                    arr_prev = viz.create_moment_glyph(c, m_vec, diag)
                    if arr_prev is not None:
                        pl.add_mesh(arr_prev, color="#c084fc", pickable=False)
                elif state.load_mode == "pressure":
                    # Pressure arrows pointing inward normal to face
                    norm = faces.get(state.selected_face, {}).get("normal", np.array([0, 0, 1]))
                    arr_prev = viz.create_arrow_glyph(c, -norm, diag)
                    if arr_prev is not None:
                        pl.add_mesh(arr_prev, color="#f59e0b", pickable=False)
            except Exception:
                pass

        # Re-attach point picking callback on the plotter
        try:
            pl.enable_point_picking(
                callback=_on_viewport_point_picked,
                left_clicking=True,
                show_message=False,
                show_point=False,
            )
        except Exception:
            pass

        if reset_camera:
            pl.reset_camera()
        ctrl.view_update()

    def _on_viewport_point_picked(point):
        """Callback triggered when the user clicks a point on the 3D model surface."""
        if point is None:
            return
        px, py, pz = float(point[0]), float(point[1]), float(point[2])
        _pick_point_direct(px, py, pz)

    def _pick_point_direct(px: float, py: float, pz: float):
        """Update point coordinates and snap to nearest face or element centroid."""
        state.point_x = float(np.round(px, 4))
        state.point_y = float(np.round(py, 4))
        state.point_z = float(np.round(pz, 4))
        state.point_preset = "Custom (Clicked)"
        state.load_target = "point"

        # Find nearest boundary face and select it too for convenience
        bounds = S.grid.bounds if S.grid else tuple(state.model_bounds)
        mesh_ref = S.model.mesh if S.model else None
        faces = viz.face_definitions(bounds, mesh=mesh_ref)
        if faces:
            pt = np.array([px, py, pz])
            best_face = min(faces.keys(), key=lambda f: float(np.linalg.norm(faces[f]["centroid"] - pt)))
            state.selected_face = best_face
            state.support_face = best_face

        state.click_status = f"Picked location ({px:.3f}, {py:.3f}, {pz:.3f}) m"
        _update_preview_snippet()
        update_scene()

    # ------------------------------------------------------------------ helpers
    def _parse_vec3(s: str) -> np.ndarray:
        try:
            parts = [float(x.strip()) for x in s.split(",")]
            if len(parts) == 3:
                return np.array(parts, float)
        except Exception:
            pass
        return np.array([0.0, -1.0, 0.0])

    def _get_moment_vector(st, bounds, mesh_ref=None) -> np.ndarray:
        try:
            mag = float(st.moment_mag)
        except Exception:
            mag = 1e5
        faces = viz.face_definitions(bounds, mesh=mesh_ref)
        if st.moment_axis_type == "normal":
            norm = faces.get(st.selected_face, {}).get("normal", np.array([1, 0, 0]))
            n = np.linalg.norm(norm)
            return mag * (norm / (n if n > 1e-12 else 1.0))
        elif st.moment_axis_type == "x":
            return np.array([mag, 0.0, 0.0])
        elif st.moment_axis_type == "y":
            return np.array([0.0, mag, 0.0])
        elif st.moment_axis_type == "z":
            return np.array([0.0, 0.0, mag])
        else:
            axis = _parse_vec3(st.custom_axis)
            n = np.linalg.norm(axis)
            return mag * (axis / n) if n > 1e-12 else np.array([mag, 0.0, 0.0])

    def _face_condition(face_key, bounds):
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        if not all(np.isfinite([xmin, xmax, ymin, ymax, zmin, zmax])) or xmax <= xmin or ymax <= ymin or zmax <= zmin:
            if state.geom_mode == "box":
                L = float(state.box_L)
                H = float(state.box_H)
                W = float(state.box_W)
                xmin, xmax, ymin, ymax, zmin, zmax = 0.0, L, -0.5 * H, 0.5 * H, -0.5 * W, 0.5 * W
            else:
                xmin, xmax, ymin, ymax, zmin, zmax = -1.0, 1.0, -1.0, 1.0, -1.0, 1.0
        tol = "1e-6"
        m = {"+x": f"x > {xmax:.4g} - {tol}", "-x": f"x < {xmin:.4g} + {tol}",
             "+y": f"y > {ymax:.4g} - {tol}", "-y": f"y < {ymin:.4g} + {tol}",
             "+z": f"z > {zmax:.4g} - {tol}", "-z": f"z < {zmin:.4g} + {tol}"}
        return m.get(face_key, "x > 0")

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

    def _sync_face_options(mesh, bounds):
        faces = viz.face_definitions(bounds, mesh=mesh)
        options = []
        is_cad = hasattr(mesh, "boundaries") and bool(mesh.boundaries)
        if is_cad:
            for b_name in mesh.boundaries:
                if b_name in faces:
                    finfo = faces[b_name]
                    c = finfo["centroid"]
                    options.append({
                        "title": f"{finfo['name']}  ({c[0]:.2g}, {c[1]:.2g}, {c[2]:.2g})",
                        "value": b_name,
                    })
        else:
            for k, title in [("+x", "+X Face (Tip / Right)"), ("-x", "-X Face (Root / Left)"),
                             ("+y", "+Y Face (Top)"), ("-y", "-Y Face (Bottom)"),
                             ("+z", "+Z Face (Front)"), ("-z", "-Z Face (Back)")]:
                options.append({"title": title, "value": k})
        if options:
            state.face_options = options
            vals = [o["value"] for o in options]
            if state.selected_face not in vals:
                state.selected_face = options[0]["value"]
            if state.support_face not in vals:
                state.support_face = options[-1]["value"] if len(options) > 1 else options[0]["value"]

    def _sync_face_info(bounds):
        mesh_ref = S.model.mesh if S.model else None
        faces = viz.face_definitions(bounds, mesh=mesh_ref)
        if state.selected_face in faces:
            info = faces[state.selected_face]
            c = info["centroid"]
            state.face_centroid_str = f"({c[0]:.3g}, {c[1]:.3g}, {c[2]:.3g})"
            n = info["normal"]
            state.face_normal_str = f"[{n[0]:.2f}, {n[1]:.2f}, {n[2]:.2f}]"
            if state.origin_is_centroid:
                state.point_x, state.point_y, state.point_z = float(c[0]), float(c[1]), float(c[2])

    # ------------------------------------------------------------------ spec rebuild
    def _rebuild_spec_from_interactive(mesh_ref_override=None):
        """Construct a valid .mos script from the interactive UI state.

        ``mesh_ref_override`` is used while solving a newly selected STEP file:
        it is the just-imported mesh, not the mesh from the previous model.
        """
        lines = ["# MoS Model — Interactive Builder"]

        # 1. Mesh
        if state.geom_mode == "step":
            if state.cad_preset == "custom":
                step_file = str(state.cad_custom_path).strip()
            else:
                # Resolve bundled examples relative to the project, not the launch CWD.
                project_root = Path(__file__).resolve().parents[2]
                step_file = str((project_root / state.cad_preset).resolve())
            # JSON quoting preserves spaces and Windows path separators in the spec.
            step_file_token = json.dumps(step_file)
            ms = f" {state.cad_mesh_size:.3g}" if state.cad_mesh_size else ""
            lines.append(f"mesh     step {step_file_token}{ms}")
        else:
            lines.append(f"param L = {state.box_L}")
            lines.append(f"param H = {state.box_H}")
            lines.append(f"param W = {state.box_W}")
            lines.append(f"mesh     box L H W {state.box_nx} {state.box_ny} {state.box_nz} {state.box_elem}")

        # 2. Material
        lines.append(f"material E = {float(state.mat_E):.4e}, nu = {float(state.mat_nu):.4g}, rho = {float(state.mat_rho):.4g}")
        lines.append("")

        # 3. Supports
        bounds = tuple(state.model_bounds)
        mesh_ref = mesh_ref_override if mesh_ref_override is not None else (S.model.mesh if S.model else None)
        if state.supports_list:
            lines.append("# Boundary Supports")
            for i, sup in enumerate(state.supports_list):
                face = sup["face"]
                face = _resolved_face_name(face, mesh_ref)
                comps = sup["comps"]
                if state.geom_mode == "box":
                    reg = f"sup_{i}_{_safe_region_name(face, 'face')}"
                    lines.append(f"region   {reg} facets {_face_condition(face, bounds)}")
                    lines.append(f"support  {reg} {comps}")
                else:
                    reg = _safe_region_name(face, "support_face")
                    if not face.startswith("cad_face_"):
                        lines.append(f"region   {reg} facets {_face_condition(face, bounds)}")
                    lines.append(f"support  {reg} {comps}")

        # 4. Loads
        if state.loads_list:
            lines.append("")
            lines.append("# Applied Loads")
            for i, ld in enumerate(state.loads_list):
                target = ld.get("target", "face")
                mode = ld.get("mode", "force")
                face = _resolved_face_name(ld.get("face", "+x"), mesh_ref)

                if target == "face":
                    if state.geom_mode == "box":
                        reg = f"ld_{i}_{_safe_region_name(face, 'face')}"
                        lines.append(f"region   {reg} facets {_face_condition(face, bounds)}")
                    else:
                        reg = _safe_region_name(face, "face")
                        if not face.startswith("cad_face_"):
                            lines.append(f"region   {reg} facets {_face_condition(face, bounds)}")
                    if mode == "force":
                        lines.append(f"force    {reg} {ld.get('vec', '0, -1e5, 0')}")
                    elif mode == "pressure":
                        lines.append(f"pressure {reg} {ld.get('val', '1e6')}")
                    elif mode == "moment":
                        lines.append(f"moment   {reg} {ld.get('vec', '1000, 0, 0')}")
                else:
                    pt = ld.get("pt", (0, 0, 0))
                    px, py, pz = float(pt[0]), float(pt[1]), float(pt[2])
                    reg = f"pt_ld_{i}"
                    lines.append(f"region   {reg} nearest {px:.4g}, {py:.4g}, {pz:.4g}")
                    if mode == "force":
                        lines.append(f"force    {reg} {ld.get('vec', '0, -1e5, 0')}")
                    elif mode == "moment":
                        if state.geom_mode == "box":
                            freg = f"face_ld_{i}_{_safe_region_name(face, 'face')}"
                            lines.append(f"region   {freg} facets {_face_condition(face, bounds)}")
                        else:
                            freg = _safe_region_name(face, "face")
                            if not face.startswith("cad_face_"):
                                lines.append(f"region   {freg} facets {_face_condition(face, bounds)}")
                        lines.append(f"moment   {freg} {ld.get('vec', '1000, 0, 0')} about {px:.4g}, {py:.4g}, {pz:.4g}")

        if bool(state.include_gravity):
            lines.append("body     0, -rho*9.81, 0")

        state.spec_text = "\n".join(lines) + "\n"

    # ------------------------------------------------------------------ preview snippet
    def _update_preview_snippet():
        bounds = S.grid.bounds if S.grid else tuple(state.model_bounds)
        mesh_ref = S.model.mesh if S.model else None
        lines = []

        if state.load_target == "face":
            face = _resolved_face_name(state.selected_face, mesh_ref)
            reg_id = face if face.startswith("cad_face_") else _unique_region_name(
                state.spec_text, f"load_{face}_{len(state.loads_list)}")
            if not face.startswith("cad_face_"):
                lines.append(f"region   {reg_id} facets {_face_condition(face, bounds)}")
            if state.load_mode == "force":
                lines.append(f"force    {reg_id} {state.force_vec}")
            elif state.load_mode == "pressure":
                lines.append(f"pressure {reg_id} {state.pressure_val}")
            else:
                mv = _get_moment_vector(state, bounds, mesh_ref)
                lines.append(f"moment   {reg_id} {mv[0]:.4g}, {mv[1]:.4g}, {mv[2]:.4g}")
        else:
            px, py, pz = float(state.point_x), float(state.point_y), float(state.point_z)
            if state.load_mode == "force":
                pt_reg = _unique_region_name(state.spec_text, f"point_{len(state.loads_list)}")
                lines.append(f"region   {pt_reg} nearest {px:.4g}, {py:.4g}, {pz:.4g}")
                lines.append(f"force    {pt_reg} {state.force_vec}")
            elif state.load_mode == "pressure":
                face = _resolved_face_name(state.selected_face, mesh_ref)
                reg_id = face if face.startswith("cad_face_") else _unique_region_name(
                    state.spec_text, f"load_{face}_{len(state.loads_list)}")
                if not face.startswith("cad_face_"):
                    lines.append(f"region   {reg_id} facets {_face_condition(face, bounds)}")
                lines.append(f"pressure {reg_id} {state.pressure_val}")
            else:
                face = _resolved_face_name(state.selected_face, mesh_ref)
                reg_id = face if face.startswith("cad_face_") else _unique_region_name(
                    state.spec_text, f"load_{face}_{len(state.loads_list)}")
                if not face.startswith("cad_face_"):
                    lines.append(f"region   {reg_id} facets {_face_condition(face, bounds)}")
                mv = _get_moment_vector(state, bounds, mesh_ref)
                lines.append(f"moment   {reg_id} {mv[0]:.4g}, {mv[1]:.4g}, {mv[2]:.4g} about {px:.4g}, {py:.4g}, {pz:.4g}")

        state.load_preview_snippet = "\n".join(lines)

    # ------------------------------------------------------------------ solve
    def run_solve():
        """Parse the active specification, solve the finite element model, and update views."""
        try:
            state.status = "Solving finite element model …"
            state.status_type = "info"
            state.flush()

            mesh_override = None
            if state.config_mode == "interactive":
                _rebuild_spec_from_interactive()
                if state.geom_mode == "step":
                    # On the first STEP import (or after changing files), S.model
                    # is absent or still points to the previous geometry. Building
                    # face selectors against its bounds creates conditions such as
                    # ``x < 1e-6`` that can never match the newly imported CAD mesh.
                    # Build just the mesh/material part first, then generate the
                    # interactive supports and loads against its named CAD faces.
                    geometry_directives = {"param", "mesh", "material"}
                    geometry_lines = []
                    for line in state.spec_text.splitlines():
                        stripped = line.strip()
                        if (not stripped or stripped.startswith("#") or
                                stripped.split(None, 1)[0].lower() in geometry_directives):
                            geometry_lines.append(line)
                    geometry_spec = parse_spec("\n".join(geometry_lines) + "\n")
                    geometry_model, _ = build(geometry_spec)
                    mesh_override = geometry_model.mesh
                    points = np.asarray(mesh_override.p, dtype=float)
                    lo, hi = points.min(axis=1), points.max(axis=1)
                    bounds = (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
                    state.model_bounds = list(bounds)
                    _sync_face_options(mesh_override, bounds)
                    _rebuild_spec_from_interactive(mesh_ref_override=mesh_override)

            spec = parse_spec(state.spec_text)
            model, case = build(spec, mesh=mesh_override) if mesh_override is not None else build(spec)
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
            if not state.scale or state.scale <= 0:
                state.scale = 1.0
            _update_scale_display(state.scale)

            bounds = S.grid.bounds
            state.model_bounds = list(bounds)
            _sync_face_options(model.mesh, bounds)
            _sync_face_info(bounds)

            div = viz.STRESS_UNITS[state.stress_unit]
            n_dof = res.u.size
            umax = float(S.scalars["|u|"][0].max())
            max_vm = float(S.scalars["von Mises"][0].max() / div)
            rx, ry, rz = res.reactions

            state.kpi_max_vm = f"{max_vm:.3f} {state.stress_unit}"
            state.kpi_max_u = f"{umax * 1e3:.3f} mm"
            state.kpi_reactions = f"[{rx:.3g}, {ry:.3g}, {rz:.3g}] N"
            a = np.asarray(getattr(res, "translational_acceleration", np.zeros(3)), float)
            alpha = np.asarray(getattr(res, "angular_acceleration", np.zeros(3)), float)
            state.kpi_acceleration = (f"a=[{a[0]:.3g}, {a[1]:.3g}, {a[2]:.3g}] m/s²; "
                                      f"α=[{alpha[0]:.3g}, {alpha[1]:.3g}, {alpha[2]:.3g}] rad/s²"
                                      if getattr(res, "body_fixed_frame", False) else "Supported body")
            state.kpi_mesh = f"{S.grid.n_cells} cells, {S.grid.n_points} nodes ({n_dof} DOFs)"
            state.kpi_error = f"{res.equilibrium_error:.2e}"
            state.max_u_str = f"Max deflection: {umax * 1e3:.4g} mm"

            state.status = (
                f"Solved — {S.grid.n_cells} cells, {S.grid.n_points} nodes | "
                f"Max |u| = {umax * 1e3:.4g} mm | Peak von Mises = {max_vm:.4g} {state.stress_unit} | "
                f"Equilibrium error = {res.equilibrium_error:.2e}"
            )
            state.status_type = "success" if res.equilibrium_error < 1e-5 else "warning"

            _update_preview_snippet()
            update_scene(reset_camera=True)
        except Exception as e:
            traceback.print_exc()
            state.status = f"{type(e).__name__}: {e}"
            state.status_type = "error"

    ctrl.run_solve = run_solve

    # ------------------------------------------------------------------ load/support actions
    def apply_current_load():
        """Capture current load config into loads_list, rebuild spec, and solve."""
        _update_preview_snippet()
        bounds = tuple(state.model_bounds)
        mesh_ref = S.model.mesh if S.model else None
        mv = _get_moment_vector(state, bounds, mesh_ref) if state.load_mode == "moment" else None

        new_ld = {
            "mode": state.load_mode,
            "target": state.load_target,
            "face": state.selected_face,
            "pt": (float(state.point_x), float(state.point_y), float(state.point_z)),
            "vec": (state.force_vec if state.load_mode == "force" else
                    f"{mv[0]:.4g}, {mv[1]:.4g}, {mv[2]:.4g}" if mv is not None else ""),
            "val": state.pressure_val,
        }
        state.loads_list = list(state.loads_list) + [new_ld]

        if state.config_mode == "script":
            text = state.spec_text.rstrip()
            state.spec_text = text + "\n\n" + state.load_preview_snippet + "\n"
        else:
            _rebuild_spec_from_interactive()

        run_solve()

    def add_support_to_list():
        """Add the currently selected boundary support to the model."""
        new_sup = {"face": state.support_face, "comps": state.support_comps}
        state.supports_list = list(state.supports_list) + [new_sup]
        _rebuild_spec_from_interactive()
        run_solve()

    def remove_support(idx):
        """Remove a boundary support constraint by index."""
        sups = list(state.supports_list)
        if 0 <= idx < len(sups):
            sups.pop(idx)
            state.supports_list = sups
            _rebuild_spec_from_interactive()
            run_solve()

    def remove_load(idx):
        """Remove an applied load by index."""
        lds = list(state.loads_list)
        if 0 <= idx < len(lds):
            lds.pop(idx)
            state.loads_list = lds
            _rebuild_spec_from_interactive()
            run_solve()

    def clear_all_loads():
        """Remove all active boundary supports and loads."""
        state.loads_list = []
        state.supports_list = []
        _rebuild_spec_from_interactive()
        run_solve()

    def convert_image_drawing_to_cad():
        """Convert a 2D image drawing/sketch to a 3D extruded STEP model."""
        try:
            in_p = state.image_path.strip()
            if not in_p:
                in_p = "cad_models/sample_drawing.png"
            out_p = Path("cad_models/drawing_extruded.step").resolve()
            th = float(state.image_thickness)
            sx = float(state.image_scale_x)
            sy = float(state.image_scale_y)
            image_to_step(in_p, out_p, thickness=th, scale_x=sx, scale_y=sy)
            state.cad_preset = "cad_models/drawing_extruded.step"
            state.geom_mode = "step"
            state.image_status = f"Converted {Path(in_p).name} -> drawing_extruded.step"
            if state.config_mode == "interactive":
                _rebuild_spec_from_interactive()
            run_solve()
        except Exception as e:
            traceback.print_exc()
            state.image_status = f"Conversion error: {e}"

    ctrl.apply_current_load = apply_current_load
    ctrl.add_support_to_list = add_support_to_list
    ctrl.remove_support = remove_support
    ctrl.remove_load = remove_load
    ctrl.clear_all_loads = clear_all_loads
    ctrl.convert_image_drawing_to_cad = convert_image_drawing_to_cad

    # Force direction presets
    def _set_force_preset(key):
        mag = state.force_mag or "1e5"
        presets = {
            "down": f"0, -{mag}, 0", "up": f"0, {mag}, 0",
            "tension": f"{mag}, 0, 0", "compression": f"-{mag}, 0, 0",
            "lateral_z": f"0, 0, {mag}",
        }
        state.force_vec = presets.get(key, state.force_vec)
        _update_preview_snippet()
        update_scene()

    for _k in ("down", "up", "tension", "compression", "lateral_z"):
        setattr(ctrl, f"set_force_preset_{_k}", (lambda k=_k: lambda: _set_force_preset(k))())

    # Material preset
    def _set_material_preset(mat_key):
        if mat_key in MATERIAL_PRESETS:
            info = MATERIAL_PRESETS[mat_key]
            state.mat_preset = mat_key
            state.mat_E = f"{info['E']:.4g}"
            state.mat_nu = f"{info['nu']:.4g}"
            state.mat_rho = f"{info['rho']:.4g}"
            if state.config_mode == "interactive":
                _rebuild_spec_from_interactive()

    # Deformation scale shortcuts
    ctrl.set_scale_1 = lambda: state.update({"scale": 1.0})
    ctrl.set_scale_auto = lambda: state.update({"scale": float(state.auto_scale)})
    ctrl.set_scale_0 = lambda: state.update({"scale": 0.0})

    # Snapping point to centroids
    def snap_to_face_centroid():
        """Snap the active target point to the centroid of the selected boundary face."""
        bounds = S.grid.bounds if S.grid else tuple(state.model_bounds)
        mesh_ref = S.model.mesh if S.model else None
        faces = viz.face_definitions(bounds, mesh=mesh_ref)
        if state.selected_face in faces:
            c = faces[state.selected_face]["centroid"]
            _pick_point_direct(c[0], c[1], c[2])

    ctrl.snap_to_face_centroid = snap_to_face_centroid

    # ------------------------------------------------------------------ initial state
    default_template = next(iter(TEMPLATES))
    state.update({
        "theme": "dark",
        "spec_text": TEMPLATES[default_template],
        "template": None,
        "templates": list(TEMPLATES),
        "status": "Edit the model parameters or load a CAD file, then press Solve.",
        "status_type": "info",
        "active_tab": "model",
        "results_right": True,
        "config_mode": "interactive",
        "geom_mode": "box",
        # Box
        "box_L": 10.0, "box_H": 1.0, "box_W": 1.0,
        "box_nx": 20, "box_ny": 2, "box_nz": 2,
        "box_elem": "tet", "box_elems": ["tet", "hex"],
        # CAD STEP
        "cad_preset": "cad_models/bracket.step",
        "cad_presets": CAD_PRESETS,
        "cad_custom_path": "",
        "cad_mesh_size": 0.6,
        # Image -> CAD Converter
        "image_path": "cad_models/sample_drawing.png",
        "image_thickness": 0.2,
        "image_scale_x": 2.0,
        "image_scale_y": 2.0,
        "image_status": "Ready to convert image drawing to 3D STEP",
        # Material
        "mat_preset": "Steel",
        "mat_preset_items": list(MATERIAL_PRESETS),
        "mat_E": "210e9", "mat_nu": "0.3", "mat_rho": "7850",
        # Supports & Loads
        "supports_list": [{"face": "-x", "comps": "xyz"}],
        "loads_list": [{"mode": "force", "target": "face", "face": "+x",
                        "vec": "0, -1e5, 0", "val": "0", "pt": [0, 0, 0]}],
        "support_face": "-x",
        "support_comps": "xyz",
        "support_comp_options": SUPPORT_COMP_OPTIONS,
        "include_gravity": False,
        # Results
        "field": "von Mises",
        "field_items": [],
        "stress_unit": "MPa",
        "stress_units": list(viz.STRESS_UNITS),
        "scale": 1.0, "scale_max": 3.0, "scale_step": 0.01,
        "auto_scale": 1.0, "scale_display": "1:1 (True Scale)",
        "max_u_str": "",
        "kpi_max_vm": "—", "kpi_max_u": "—",
        "kpi_reactions": "—", "kpi_mesh": "—", "kpi_error": "—",
        "kpi_acceleration": "—",
        # View toggles
        "show_edges": True,
        "clip_axis": "none", "clip_axes": ["none", "x", "y", "z"],
        "clip_pos": 0.5,
        "show_glyphs": True,
        "show_preview_glyph": True,
        "show_face_highlight": True,
        "show_point_marker": True,
        # Load builder & picking
        "load_target": "face",
        "load_mode": "force",
        "selected_face": "+x",
        "face_options": [
            {"title": "+X Face (Tip / Right)", "value": "+x"},
            {"title": "-X Face (Root / Left)", "value": "-x"},
            {"title": "+Y Face (Top)", "value": "+y"},
            {"title": "-Y Face (Bottom)", "value": "-y"},
            {"title": "+Z Face (Front)", "value": "+z"},
            {"title": "-Z Face (Back)", "value": "-z"},
        ],
        "face_centroid_str": "(10, 0, 0)",
        "face_normal_str": "[1, 0, 0]",
        "point_x": 10.0, "point_y": 0.0, "point_z": 0.0,
        "point_preset": "Tip Center",
        "point_presets": ["Tip Center", "Root Center", "Top Midspan",
                          "Bottom Midspan", "Origin", "Custom (Clicked)"],
        "origin_is_centroid": True,
        "click_status": "Click anywhere on 3D mesh surface to pick target point",
        # Force
        "force_mag": "1e5",
        "force_vec": "0, -1e5, 0",
        # Moment
        "moment_axis_type": "normal",
        "moment_axis_types": [
            {"title": "Face Normal (Torsion)", "value": "normal"},
            {"title": "X Axis [1, 0, 0]", "value": "x"},
            {"title": "Y Axis [0, 1, 0]", "value": "y"},
            {"title": "Z Axis [0, 0, 1]", "value": "z"},
            {"title": "Custom Axis …", "value": "custom"},
        ],
        "custom_axis": "1, 0, 0",
        "moment_mag": "1e5",
        # Pressure
        "pressure_val": "1e6",
        # Preview
        "load_preview_snippet": "force face_pos_x 0, -1e5, 0",
        "model_bounds": [0, 10, -0.5, 0.5, -0.5, 0.5],
    })

    if args.spec:
        state.spec_text = Path(args.spec).read_text()

    # ------------------------------------------------------------------ reactive watchers
    @state.change("template")
    def _load_template(template, **_):
        if template in TEMPLATES:
            state.spec_text = TEMPLATES[template]
            state.config_mode = "script"
            run_solve()

    @state.change("scale")
    def _scale_changed(scale, **_):
        _update_scale_display(scale)
        update_scene()

    @state.change("theme")
    def _theme_changed(**_):
        update_scene()

    @state.change("field", "stress_unit", "show_edges", "clip_axis", "clip_pos",
                  "show_glyphs", "show_face_highlight", "show_point_marker",
                  "show_preview_glyph", "active_tab")
    def _redraw(**_):
        update_scene()

    @state.change("selected_face")
    def _face_changed(selected_face, **_):
        bounds = S.grid.bounds if S.grid else tuple(state.model_bounds)
        _sync_face_info(bounds)
        _update_preview_snippet()
        update_scene()

    @state.change("point_preset")
    def _preset_changed(point_preset, **_):
        bounds = S.grid.bounds if S.grid else tuple(state.model_bounds)
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        cx, cy, cz = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
        presets = {
            "Tip Center": (xmax, cy, cz),
            "Root Center": (xmin, cy, cz),
            "Top Midspan": (cx, ymax, cz),
            "Bottom Midspan": (cx, ymin, cz),
            "Origin": (0.0, 0.0, 0.0),
        }
        if point_preset in presets:
            state.point_x, state.point_y, state.point_z = [float(v) for v in presets[point_preset]]
        _update_preview_snippet()
        update_scene()

    @state.change("point_x", "point_y", "point_z", "force_vec", "force_mag",
                  "moment_mag", "moment_axis_type", "custom_axis",
                  "load_target", "load_mode", "pressure_val")
    def _load_params_changed(**_):
        _update_preview_snippet()
        update_scene()

    @state.change("geom_mode", "cad_preset", "cad_custom_path", "cad_mesh_size",
                  "box_L", "box_H", "box_W", "box_nx", "box_ny", "box_nz", "box_elem",
                  "mat_E", "mat_nu", "mat_rho", "include_gravity")
    def _interactive_param_changed(**_):
        if state.config_mode == "interactive":
            _rebuild_spec_from_interactive()

    # ================================================================ LAYOUT
    with SinglePageWithDrawerLayout(server) as layout:
        layout.root.theme = ("theme",)
        layout.title.set_text("MoS Structural Analyzer")
        layout.drawer.width = 420

        # Toolbar
        with layout.toolbar:
            v3.VSpacer()
            v3.VChip(
                "{{ scale_display }}",
                color="primary", variant="tonal", size="small", classes="mr-2",
                click=ctrl.set_scale_1,
                title="Click to reset to 1:1 true scale",
            )
            v3.VBtn(
                icon="mdi-theme-light-dark", variant="text",
                density="comfortable", classes="mr-1",
                click="theme = theme === 'dark' ? 'light' : 'dark'",
            )

        # Drawer
        with layout.drawer:
            with v3.VContainer(fluid=True, classes="pa-2"):

                # Navigation tabs
                with v3.VTabs(v_model=("active_tab",), grow=True,
                              density="compact", classes="mb-2"):
                    v3.VTab("Model", value="model")
                    v3.VTab("Loads & Supports", value="loads")
                    v3.VTab("Results", value="results")

                with v3.VWindow(v_model=("active_tab",)):

                    # ════════════════════════════════════════════════════════
                    # TAB 1: MODEL & GEOMETRY
                    # ════════════════════════════════════════════════════════
                    with v3.VWindowItem(value="model"):

                        # Mode toggle
                        with v3.VBtnToggle(
                            v_model=("config_mode",), mandatory=True,
                            density="compact", divided=True, color="primary",
                            classes="mb-3 w-100",
                        ):
                            v3.VBtn("Interactive Builder", value="interactive",
                                    size="small", classes="flex-grow-1")
                            v3.VBtn("Script Editor (.mos)", value="script",
                                    size="small", classes="flex-grow-1")

                        # INTERACTIVE CONFIG
                        with html.Div(v_show="config_mode === 'interactive'"):

                            # 1. Geometry Source
                            with v3.VCard(variant="tonal", classes="pa-3 mb-2 rounded-lg"):
                                html.Div("1. Geometry Source", classes="text-subtitle-2 font-weight-bold mb-2")

                                with v3.VBtnToggle(
                                    v_model=("geom_mode",), mandatory=True,
                                    density="compact", divided=True,
                                    color="secondary", classes="mb-2 w-100",
                                ):
                                    v3.VBtn("Parametric Box", value="box",
                                            size="small", classes="flex-grow-1")
                                    v3.VBtn("Import CAD (STEP)", value="step",
                                            size="small", classes="flex-grow-1")

                                # Parametric box
                                with html.Div(v_show="geom_mode === 'box'"):
                                    with html.Div(classes="d-flex ga-1 mb-1"):
                                        v3.VTextField(v_model=("box_L",), label="Length (m)",
                                                      type="number", step=0.5,
                                                      density="compact", hide_details=True)
                                        v3.VTextField(v_model=("box_H",), label="Height (m)",
                                                      type="number", step=0.1,
                                                      density="compact", hide_details=True)
                                        v3.VTextField(v_model=("box_W",), label="Width (m)",
                                                      type="number", step=0.1,
                                                      density="compact", hide_details=True)
                                    with html.Div(classes="d-flex ga-1"):
                                        v3.VTextField(v_model=("box_nx",), label="nx",
                                                      type="number", step=1,
                                                      density="compact", hide_details=True)
                                        v3.VTextField(v_model=("box_ny",), label="ny",
                                                      type="number", step=1,
                                                      density="compact", hide_details=True)
                                        v3.VTextField(v_model=("box_nz",), label="nz",
                                                      type="number", step=1,
                                                      density="compact", hide_details=True)
                                        v3.VSelect(v_model=("box_elem",), items=("box_elems",),
                                                   label="Type", density="compact",
                                                   hide_details=True)

                                # CAD STEP importer
                                with html.Div(v_show="geom_mode === 'step'"):
                                    v3.VSelect(
                                        v_model=("cad_preset",), items=("cad_presets",),
                                        item_title="title", item_value="value",
                                        label="CAD Model Preset",
                                        density="compact", hide_details=True, classes="mb-2",
                                    )
                                    v3.VTextField(
                                        v_model=("cad_custom_path",),
                                        label="Custom .step / .stp file path",
                                        density="compact", hide_details=True, classes="mb-2",
                                        v_show="cad_preset === 'custom'",
                                    )
                                    with html.Div(classes="d-flex align-center ga-2 mb-2"):
                                        html.Span("Mesh size:", classes="text-caption")
                                        v3.VSlider(
                                            v_model=("cad_mesh_size",),
                                            min=0.05, max=2.0, step=0.05,
                                            density="compact", hide_details=True,
                                            classes="flex-grow-1",
                                        )
                                        html.Span("{{ cad_mesh_size }} m",
                                                  classes="text-caption font-weight-bold",
                                                  style="min-width:48px; text-align:right")

                                    # Image to AutoCAD/STEP Conversion Accordion
                                    v3.VDivider(classes="my-2")
                                    with html.Div(classes="pa-1"):
                                        html.Div("Image / Drawing to CAD STEP", classes="text-caption font-weight-bold mb-1")
                                        v3.VTextField(
                                            v_model=("image_path",),
                                            label="2D Drawing Image Path (PNG / JPG)",
                                            density="compact", hide_details=True, classes="mb-1",
                                        )
                                        with html.Div(classes="d-flex ga-1 mb-1"):
                                            v3.VTextField(v_model=("image_thickness",), label="Thickness Z (m)",
                                                          type="number", step=0.05, density="compact", hide_details=True)
                                            v3.VTextField(v_model=("image_scale_x",), label="Width X (m)",
                                                          type="number", step=0.5, density="compact", hide_details=True)
                                            v3.VTextField(v_model=("image_scale_y",), label="Height Y (m)",
                                                          type="number", step=0.5, density="compact", hide_details=True)
                                        v3.VBtn("Convert Drawing to 3D STEP", size="small", block=True,
                                                variant="outlined", color="secondary", click=ctrl.convert_image_drawing_to_cad)
                                        html.Div("{{ image_status }}", classes="text-caption text-medium-emphasis mt-1")

                            # 2. Material
                            with v3.VCard(variant="outlined", classes="pa-3 mb-2 rounded-lg"):
                                html.Div("2. Material Properties", classes="text-subtitle-2 font-weight-bold mb-2")
                                v3.VSelect(
                                    v_model=("mat_preset",),
                                    items=("mat_preset_items",),
                                    label="Material Preset",
                                    density="compact", hide_details=True, classes="mb-2",
                                    update_model_value=_set_material_preset,
                                )
                                with html.Div(classes="d-flex ga-1"):
                                    v3.VTextField(v_model=("mat_E",), label="E (Pa)",
                                                  density="compact", hide_details=True)
                                    v3.VTextField(v_model=("mat_nu",), label="nu (Poisson)",
                                                  density="compact", hide_details=True)
                                    v3.VTextField(v_model=("mat_rho",), label="rho (kg/m3)",
                                                  density="compact", hide_details=True)

                        # SCRIPT EDITOR
                        with html.Div(v_show="config_mode === 'script'"):
                            v3.VSelect(
                                v_model=("template",), items=("templates",),
                                label="Load Benchmark / Template Example",
                                density="compact", hide_details=True, classes="mb-2",
                            )
                            v3.VTextarea(
                                v_model=("spec_text",), rows=14,
                                variant="outlined", hide_details=True, no_resize=True,
                                style="font-family: monospace; font-size: 11px;",
                            )

                        # Solve Button
                        v3.VBtn(
                            "Solve Model", block=True, color="primary",
                            size="large", classes="my-3 font-weight-bold",
                            click=ctrl.run_solve,
                        )

                        # Status
                        with v3.VAlert(
                            type=("status_type",), variant="tonal",
                            density="compact", classes="mt-0 mb-1",
                        ):
                            html.Pre("{{ status }}",
                                     style="white-space: pre-wrap; font-size: 11px; margin: 0;")

                    # ════════════════════════════════════════════════════════
                    # TAB 2: LOADS & SUPPORTS
                    # ════════════════════════════════════════════════════════
                    with v3.VWindowItem(value="loads"):

                        # ── SECTION A: Active Supports ──────────────────────
                        with v3.VCard(variant="tonal", classes="pa-3 mb-2 rounded-lg"):
                            with html.Div(classes="d-flex align-center justify-space-between mb-2"):
                                html.Span("Boundary Supports", classes="text-subtitle-2 font-weight-bold")
                                v3.VChip("{{ supports_list.length }}", size="x-small", color="secondary")

                            with html.Div(v_if="supports_list.length > 0", classes="mb-2"):
                                with v3.VList(density="compact", classes="pa-0 rounded"):
                                    with v3.VListItem(
                                        v_for="(sup, idx) in supports_list",
                                        key="idx", classes="px-2 py-1",
                                        style="border-bottom: 1px solid rgba(128,128,128,0.15);",
                                    ):
                                        with html.Template(v_slot_default=True):
                                            html.Span("{{ sup.face }} · {{ sup.comps }}",
                                                      classes="text-caption font-weight-medium")
                                        with html.Template(v_slot_append=True):
                                            v3.VBtn(
                                                text="Remove", size="x-small", variant="text", color="error",
                                                click=(ctrl.remove_support, "[idx]"),
                                            )

                            with html.Div(v_else=True, classes="text-caption text-medium-emphasis text-center pa-2"):
                                html.Span("No supports applied yet.")

                            v3.VDivider(classes="mb-2")
                            html.Div("Add a boundary support:", classes="text-caption mb-1")
                            with html.Div(classes="d-flex ga-2 mb-2"):
                                v3.VSelect(
                                    v_model=("support_face",),
                                    items=("face_options",),
                                    item_title="title", item_value="value",
                                    label="Target Face",
                                    density="compact", hide_details=True,
                                    classes="flex-grow-1",
                                )
                                v3.VSelect(
                                    v_model=("support_comps",),
                                    items=("support_comp_options",),
                                    item_title="title", item_value="value",
                                    label="Constraint",
                                    density="compact", hide_details=True,
                                    classes="flex-grow-1",
                                )
                            v3.VBtn(
                                "Add Support", block=True, size="small",
                                variant="tonal", color="secondary",
                                click=ctrl.add_support_to_list,
                            )

                        # ── SECTION B: Active Loads ─────────────────────────
                        with v3.VCard(variant="outlined", classes="pa-3 mb-2 rounded-lg"):
                            with html.Div(classes="d-flex align-center justify-space-between mb-2"):
                                html.Span("Applied Loads", classes="text-subtitle-2 font-weight-bold")
                                v3.VChip("{{ loads_list.length }}", size="x-small", color="primary")

                            with html.Div(v_if="loads_list.length > 0", classes="mb-2"):
                                with v3.VList(density="compact", classes="pa-0 rounded"):
                                    with v3.VListItem(
                                        v_for="(ld, idx) in loads_list",
                                        key="idx", classes="px-2 py-1",
                                        style="border-bottom: 1px solid rgba(128,128,128,0.15);",
                                    ):
                                        with html.Template(v_slot_default=True):
                                            html.Span("{{ ld.mode }} on {{ ld.face || ld.target }}",
                                                      classes="text-caption font-weight-medium")
                                            html.Div("{{ ld.vec || ld.val }}",
                                                     classes="text-caption text-medium-emphasis")
                                        with html.Template(v_slot_append=True):
                                            v3.VBtn(
                                                text="Remove", size="x-small", variant="text", color="error",
                                                click=(ctrl.remove_load, "[idx]"),
                                            )

                            with html.Div(v_else=True, classes="text-caption text-medium-emphasis text-center pa-2"):
                                html.Span("No loads applied yet.")

                            with html.Div(v_if="loads_list.length > 0 || supports_list.length > 0",
                                          classes="mb-1 text-right"):
                                v3.VBtn(
                                    "Clear All", size="x-small", variant="text",
                                    color="error", click=ctrl.clear_all_loads,
                                )

                        # ── SECTION C: Load Builder & Click to Pick ──────────
                        with v3.VCard(variant="tonal", classes="pa-3 mb-2 rounded-lg"):
                            html.Div("Apply New Force or Moment", classes="text-subtitle-2 font-weight-bold mb-2")

                            # Interactive 3D Picking Status Banner
                            with v3.VAlert(type="info", variant="tonal", density="compact", classes="mb-2 pa-1"):
                                html.Span("{{ click_status }}", classes="text-caption")

                            # Target Location Selector
                            with v3.VBtnToggle(
                                v_model=("load_target",), mandatory=True,
                                density="compact", divided=True,
                                color="primary", classes="mb-2 w-100",
                            ):
                                v3.VBtn("Boundary Face", value="face",
                                        size="small", classes="flex-grow-1")
                                v3.VBtn("Point / Element Location", value="point",
                                        size="small", classes="flex-grow-1")

                            # Face selector
                            with html.Div(v_show="load_target === 'face'"):
                                v3.VSelect(
                                    v_model=("selected_face",),
                                    items=("face_options",),
                                    item_title="title", item_value="value",
                                    label="Target Face",
                                    density="compact", hide_details=True, classes="mb-1",
                                )
                                html.Div(
                                    "Centroid: {{ face_centroid_str }} | Normal: {{ face_normal_str }}",
                                    classes="text-caption text-medium-emphasis mb-2",
                                )

                            # Point selector & picking controls
                            with html.Div(v_show="load_target === 'point'"):
                                with html.Div(classes="d-flex ga-1 mb-1"):
                                    v3.VSelect(
                                        v_model=("point_preset",), items=("point_presets",),
                                        label="Location Preset",
                                        density="compact", hide_details=True, classes="flex-grow-1",
                                    )
                                    v3.VBtn("Snap to Face", size="small", variant="outlined",
                                            click=ctrl.snap_to_face_centroid)

                                with html.Div(classes="d-flex ga-1 mb-2"):
                                    v3.VTextField(v_model=("point_x",), label="X (m)",
                                                  type="number", step=0.1,
                                                  density="compact", hide_details=True)
                                    v3.VTextField(v_model=("point_y",), label="Y (m)",
                                                  type="number", step=0.1,
                                                  density="compact", hide_details=True)
                                    v3.VTextField(v_model=("point_z",), label="Z (m)",
                                                  type="number", step=0.1,
                                                  density="compact", hide_details=True)

                            v3.VDivider(classes="mb-2")

                            # Load type toggle
                            with v3.VBtnToggle(
                                v_model=("load_mode",), mandatory=True,
                                density="compact", divided=True,
                                color="secondary", classes="mb-2 w-100",
                            ):
                                v3.VBtn("Force Vector", value="force",
                                        size="small", classes="flex-grow-1")
                                v3.VBtn("Moment (Torque)", value="moment",
                                        size="small", classes="flex-grow-1")
                                v3.VBtn("Pressure", value="pressure",
                                        size="small", classes="flex-grow-1")

                            # Force controls
                            with html.Div(v_show="load_mode === 'force'"):
                                with html.Div(classes="d-flex flex-wrap ga-1 mb-2"):
                                    v3.VBtn("Down (-Y)", size="x-small", variant="tonal",
                                            click=ctrl.set_force_preset_down)
                                    v3.VBtn("Up (+Y)", size="x-small", variant="tonal",
                                            click=ctrl.set_force_preset_up)
                                    v3.VBtn("Tension (+X)", size="x-small", variant="tonal",
                                            click=ctrl.set_force_preset_tension)
                                    v3.VBtn("Compression (-X)", size="x-small", variant="tonal",
                                            click=ctrl.set_force_preset_compression)
                                    v3.VBtn("Lateral (+Z)", size="x-small", variant="tonal",
                                            click=ctrl.set_force_preset_lateral_z)
                                with html.Div(classes="d-flex ga-1"):
                                    v3.VTextField(v_model=("force_mag",), label="Magnitude (N)",
                                                  density="compact", hide_details=True)
                                    v3.VTextField(v_model=("force_vec",), label="Vector [Fx, Fy, Fz]",
                                                  density="compact", hide_details=True)

                            # Moment controls
                            with html.Div(v_show="load_mode === 'moment'"):
                                v3.VSelect(
                                    v_model=("moment_axis_type",),
                                    items=("moment_axis_types",),
                                    item_title="title", item_value="value",
                                    label="Moment Axis",
                                    density="compact", hide_details=True, classes="mb-2",
                                )
                                v3.VTextField(
                                    v_model=("custom_axis",),
                                    label="Custom axis (ax, ay, az)",
                                    density="compact", hide_details=True, classes="mb-2",
                                    v_show="moment_axis_type === 'custom'",
                                )
                                v3.VTextField(
                                    v_model=("moment_mag",), label="Magnitude M (N*m)",
                                    density="compact", hide_details=True, classes="mb-2",
                                )
                                with v3.VAlert(type="info", variant="text",
                                               density="compact", classes="pa-1"):
                                    html.Span(
                                        "Moments are applied as equivalent zero-net-force torque couples "
                                        "and displayed as rotational arc arrows.",
                                        classes="text-caption",
                                    )

                            # Pressure controls
                            with html.Div(v_show="load_mode === 'pressure'"):
                                v3.VTextField(
                                    v_model=("pressure_val",), label="Pressure p (Pa)",
                                    density="compact", hide_details=True, classes="mb-2",
                                )

                            v3.VSwitch(
                                v_model=("include_gravity",),
                                label="Include gravity self-weight",
                                density="compact", hide_details=True, classes="mt-1",
                            )

                        # Code preview & apply
                        with v3.VCard(variant="outlined", classes="pa-3 mb-2 rounded-lg"):
                            html.Div("Generated load directive preview",
                                     classes="text-caption text-medium-emphasis mb-1")
                            html.Pre(
                                "{{ load_preview_snippet }}",
                                style=(
                                    "font-family: monospace; font-size: 11px; "
                                    "padding: 6px; border-radius: 4px; "
                                    "background: rgba(0,0,0,0.2); "
                                    "white-space: pre-wrap; margin: 0 0 8px 0;"
                                ),
                            )
                            v3.VBtn(
                                "Apply Load and Solve", block=True, color="primary",
                                density="default", click=ctrl.apply_current_load,
                            )

                    # ════════════════════════════════════════════════════════
                    # TAB 3: RESULTS
                    # ════════════════════════════════════════════════════════
                    with v3.VWindowItem(value="results"):
                        with v3.VCard(variant="tonal", classes="pa-3 rounded-lg"):
                            html.Div("Results", classes="text-subtitle-2 font-weight-bold mb-2")

                            with v3.VCard(variant="tonal", classes="pa-3 mb-3 rounded-lg"):
                                html.Div("Solution Summary", classes="text-subtitle-2 font-weight-bold mb-2")
                                for kpi_label, kpi_state, chip_color in [
                                    ("Peak von Mises Stress", "kpi_max_vm", "error"),
                                    ("Max Displacement", "kpi_max_u", "primary"),
                                ]:
                                    with html.Div(classes="d-flex justify-space-between align-center mb-2"):
                                        html.Span(kpi_label, classes="text-caption")
                                        v3.VChip(
                                            f"{{{{ {kpi_state} }}}}",
                                            color=chip_color, size="small",
                                            classes="font-weight-bold",
                                        )

                                for txt_label, txt_state in [
                                    ("Total Reactions", "kpi_reactions"),
                            ("Body-Fixed Acceleration", "kpi_acceleration"),
                                    ("Equilibrium Residual", "kpi_error"),
                                    ("Mesh Statistics", "kpi_mesh"),
                                ]:
                                    with html.Div(classes="d-flex justify-space-between align-center mb-1"):
                                        html.Span(txt_label, classes="text-caption text-medium-emphasis")
                                        html.Span(
                                            f"{{{{ {txt_state} }}}}",
                                            classes="text-caption font-weight-medium",
                                        )

                            with html.Div(classes="d-flex ga-1 mb-2"):
                                v3.VSelect(
                                    v_model=("field",), items=("field_items",),
                                    label="Plot Field",
                                    density="compact", hide_details=True, classes="flex-grow-1",
                                )
                                v3.VSelect(
                                    v_model=("stress_unit",), items=("stress_units",),
                                    label="Unit",
                                    density="compact", hide_details=True,
                                    style="max-width: 110px;",
                                )

                            with v3.VCard(variant="tonal", classes="pa-2 mb-3 rounded-lg"):
                                with html.Div(classes="d-flex align-center justify-space-between mb-1"):
                                    html.Span("Deformation Scale",
                                              classes="text-caption font-weight-bold")
                                    v3.VChip("{{ scale_display }}", color="primary",
                                             size="x-small", classes="font-weight-bold")
                                with html.Div(classes="d-flex ga-1 mb-1"):
                                    v3.VBtn("1:1 True", size="x-small", variant="tonal",
                                            click=ctrl.set_scale_1)
                                    v3.VBtn("Auto (~10%)", size="x-small", variant="tonal",
                                            click=ctrl.set_scale_auto)
                                    v3.VBtn("0 Undeformed", size="x-small", variant="tonal",
                                            click=ctrl.set_scale_0)
                                v3.VSlider(
                                    v_model=("scale",), min=0,
                                    max=("scale_max",), step=("scale_step",),
                                    hide_details=True, density="compact",
                                )
                                html.Div("{{ max_u_str }}",
                                         classes="text-caption text-medium-emphasis mt-1")

                            v3.VDivider(classes="mb-2")
                            html.Div("3D Visualization Controls", classes="text-caption font-weight-bold mb-1")
                            v3.VSwitch(v_model=("show_edges",), label="Mesh element edges",
                                       hide_details=True, density="compact")
                            v3.VSwitch(v_model=("show_glyphs",),
                                       label="Show applied load arrows and moments",
                                       hide_details=True, density="compact")
                            v3.VSwitch(v_model=("show_preview_glyph",),
                                       label="Show live load preview indicator",
                                       hide_details=True, density="compact")
                            v3.VSwitch(v_model=("show_face_highlight",),
                                       label="Highlight active boundary face",
                                       hide_details=True, density="compact")

                            v3.VDivider(classes="my-2")
                            html.Div("Section Plane (Clipping)",
                                     classes="text-caption font-weight-bold mb-1")
                            with html.Div(classes="d-flex ga-1 align-center mb-1"):
                                v3.VSelect(
                                    v_model=("clip_axis",), items=("clip_axes",),
                                    label="Normal axis",
                                    density="compact", hide_details=True,
                                    style="max-width: 100px;",
                                )
                                v3.VSlider(
                                    v_model=("clip_pos",), min=0, max=1, step=0.01,
                                    disabled=("clip_axis === 'none'",),
                                    label="Position", hide_details=True, density="compact",
                                    classes="flex-grow-1",
                                )

        # Main viewport
        with layout.content:
            with v3.VContainer(fluid=True, classes="pa-0 fill-height"):
                with html.Div(style="position: relative; width: 100%; height: 100%; padding: 8px; box-sizing: border-box;"):
                    view = plotter_ui(pl, mode="server")
                    ctrl.view_update = view.update
    server.start()
