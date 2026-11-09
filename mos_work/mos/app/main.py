"""trame + PyVista front end.

    python -m mos.app --server [--spec model.mos] [--port 8080]

The drawer holds interactive geometric/CAD model builders, materials, supports,
loads, and view controls; the 3D view is a server-side-rendered VTK scene.
"""
from __future__ import annotations

import json
import re
import threading
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyvista as pv
from trame_pyvista.ui import plotter_ui
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import html
from trame.widgets import vuetify3 as v3

from mos import viz
from mos.core.cad import is_step_file, load_step_mesh  # noqa: F401 (may be used by templates)
from mos.core.cad_parametric import generate_cad_from_parameters
from mos.core.cv_detector import analyze_image_object, cv_image_to_cad_step
from mos.core.image_to_cad import image_to_step
from mos.core.image_reconstruction import reconstruct_single_image
from mos.core.multiview import (
    calibrate_point_cloud_scale, gaussian_ply_to_point_cloud,
    nerfstudio_commands, prepare_capture_session, run_nerfstudio_preprocess,
    tool_availability,
)
from mos.core.solver import CachedLinearSolver, solve_case
from mos.core.regions import cad_boundary_name_for_side
from mos.core import post
from mos.core.spec import build, parse_spec
from mos.templates import TEMPLATES
from mos.lab import evaluate_stress as lab_evaluate_stress, evaluate_traction as lab_evaluate_traction, model_summary as lab_model_summary, tensor_matrix_html as lab_tensor_matrix_html
from mos.learning import (
    analyse_stress_tensor, mechanics_quick_check, mohr_diagram_svg,
    render_direction_rows, render_tensor_matrix, tensor_diagram_svg,
)

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

LAB_CASE_PRESETS = {
    "Combined 3D stress": (80.0, 20.0, 10.0, 30.0, 12.0, -8.0),
    "Uniaxial tension": (100.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "Pure shear": (0.0, 0.0, 0.0, 60.0, 0.0, 0.0),
    "Hydrostatic stress": (50.0, 50.0, 50.0, 0.0, 0.0, 0.0),
    "Triaxial compression": (-80.0, -50.0, -20.0, 0.0, 0.0, 0.0),
}
LAB_CASE_ITEMS = [{"title": name, "value": name} for name in LAB_CASE_PRESETS]
LAB_TOOL_ITEMS = [
    {"title": "Beam bending", "value": "beam"},
    {"title": "Shaft torsion", "value": "shaft"},
    {"title": "Thin-wall pressure vessel", "value": "vessel"},
]

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
                        preview_mesh=None,
                        spec=None, model=None, case=None, main_actor=None,
                        cached_solver=None, live_response=None,
                        cached_geometry_signature=None, solved_spec_text=None,
                        live_response_suspended=False,
                        lab_refresh_suspended=False,
                        point_picker_installed=False, analysis_actors=[])

    def _displayed_mesh_points():
        """Mesh vertex positions after the current displacement scale is applied."""
        if S.grid is None:
            return None
        reference = np.asarray(S.grid.points, float)
        if S.u_pts is None or np.asarray(S.u_pts).shape != reference.shape:
            return reference
        return reference + float(state.scale) * np.asarray(S.u_pts, float)

    def update_analysis_overlay(render=True):
        """Update only the selected-point/plane actors; never rebuild the FE mesh."""
        if S.grid is None:
            return

        # Remove only the previous analysis actors. Rebuilding the complete scene
        # for every XYZ or plane edit is expensive for dense CAD meshes and can
        # disturb the viewport camera.
        for actor in list(getattr(S, "analysis_actors", [])):
            try:
                pl.remove_actor(actor, reset_camera=False, render=False)
            except Exception:
                pass
        S.analysis_actors = []

        bounds = S.grid.bounds
        diag = float(np.linalg.norm(np.ptp(np.asarray(S.grid.points), axis=0)))
        display_points = _displayed_mesh_points()
        if display_points is None:
            return

        if bool(getattr(state, "show_point_marker", True)):
            try:
                if bool(getattr(state, "analysis_dock_open", False)):
                    node_index = int(getattr(state, "lab_node_index", -1))
                    if 0 <= node_index < len(display_points):
                        point = np.asarray(display_points[node_index], float)
                    else:
                        point = np.array([
                            float(state.lab_point_x), float(state.lab_point_y),
                            float(state.lab_point_z),
                        ])
                else:
                    point = np.array([
                        float(state.point_x), float(state.point_y), float(state.point_z),
                    ])
                marker = pv.Sphere(radius=max(0.02 * diag, 1e-4), center=point)
                actor = pl.add_mesh(marker, color="#f59e0b", opacity=0.95,
                                     pickable=False, reset_camera=False, render=False)
                S.analysis_actors.append(actor)
            except Exception:
                pass

        # Keep the analysis plane and vectors as a small overlay layer. They are
        # replaced in-place when the point, normal or load response changes.
        if (bool(getattr(state, "analysis_dock_open", False))
                and bool(getattr(state, "lab_has_result", False))
                and getattr(S, "res", None) is not None):
            try:
                node_index = int(state.lab_node_index)
                if 0 <= node_index < S.grid.n_points:
                    sample_xyz = np.asarray(display_points[node_index], float)
                    tensor = np.asarray(S.res.sig_nodal[:, :, node_index], float)
                    normal = np.asarray(_lab_plane_vector(tensor / 1.0e6), float)
                    normal_length = float(np.linalg.norm(normal))
                    if not np.isfinite(normal_length) or normal_length < 1e-14:
                        raise ValueError("Plane normal must be finite and non-zero")
                    normal /= normal_length
                    plane_size = max(0.12 * diag, 1e-3)
                    plane_mesh = pv.Plane(
                        center=sample_xyz, direction=normal,
                        i_size=plane_size, j_size=plane_size,
                        i_resolution=1, j_resolution=1,
                    )
                    actor = pl.add_mesh(
                        plane_mesh, color="#38bdf8", opacity=0.20,
                        show_edges=True, edge_color="#7dd3fc", line_width=1.25,
                        pickable=False, reset_camera=False, render=False,
                    )
                    S.analysis_actors.append(actor)

                    normal_glyph = viz.create_arrow_glyph(
                        sample_xyz, normal, diag, reference_magnitude=1.0
                    )
                    if normal_glyph is not None:
                        actor = pl.add_mesh(normal_glyph, color="#22c55e",
                                             pickable=False, reset_camera=False, render=False)
                        S.analysis_actors.append(actor)

                    traction = tensor @ normal
                    traction_reference = max(float(np.linalg.norm(tensor)), 1.0)
                    traction_glyph = viz.create_arrow_glyph(
                        sample_xyz, traction, diag,
                        reference_magnitude=traction_reference,
                    )
                    if traction_glyph is not None:
                        actor = pl.add_mesh(traction_glyph, color="#f59e0b",
                                             pickable=False, reset_camera=False, render=False)
                        S.analysis_actors.append(actor)
            except Exception:
                # Numerical values stay available even if a VTK glyph cannot be made.
                pass

        if render:
            ctrl.view_update()

    # ------------------------------------------------------------------ scene
    def update_scene(reset_camera=False):
        """Redraw the 3D view from the active preview or solved FEM result."""
        is_dark = state.theme == "dark"
        pl.background_color = "#18181b" if is_dark else "#f8fafc"

        # Single-image reconstruction previews are deliberately isolated from the
        # FEM model/result objects. Switching to this view never replaces S.grid,
        # S.model, the active spec, or the cached solver.
        if bool(getattr(state, "cv3d_preview_active", False)) and S.preview_mesh is not None:
            pl.clear()
            preview = S.preview_mesh.copy()
            color_fields = ("red", "green", "blue")
            if all(field in preview.point_data for field in color_fields):
                preview.point_data["rgb"] = np.ascontiguousarray(
                    np.column_stack([preview.point_data[field] for field in color_fields]).astype(np.uint8)
                )
            elif "RGB" in preview.point_data:
                preview.point_data["rgb"] = np.asarray(preview.point_data["RGB"], dtype=np.uint8)[:, :3]
            elif "RGBA" in preview.point_data:
                preview.point_data["rgb"] = np.asarray(preview.point_data["RGBA"], dtype=np.uint8)[:, :3]
            if "rgb" in preview.point_data:
                pl.add_mesh(
                    preview, scalars="rgb", rgb=True, show_edges=False,
                    smooth_shading=True, ambient=0.25, diffuse=0.8,
                )
            else:
                pl.add_mesh(preview, color="#94a3b8", show_edges=False, smooth_shading=True)
            try:
                pl.add_axes(line_width=2)
                pl.reset_camera()
            except Exception:
                pass
            ctrl.view_update()
            return

        if S.grid is None or state.field not in S.scalars:
            pl.clear()
            S.analysis_actors = []
            ctrl.view_update()
            return

        # pl.clear() removes actors but some VTK/PyVista versions also alter the
        # active camera when the scene is repopulated. Preserve the user's view
        # unless this is an explicit first-solve/reset-camera operation.
        saved_camera = None
        if not reset_camera:
            try:
                saved_camera = pl.camera_position
            except Exception:
                saved_camera = None

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
        S.analysis_actors = []
        sb_args = viz.scalar_bar_args(label, is_dark=is_dark)
        S.main_actor = pl.add_mesh(
            g, scalars="s", cmap="turbo", clim=[lo, hi],
            show_edges=bool(state.show_edges), scalar_bar_args=sb_args,
            pickable=True,
        )

        if float(state.scale) > 0:
            outline_color = "#71717a" if is_dark else "#94a3b8"
            try:
                surf = S.grid.extract_surface(algorithm=None)
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

        # Point/plane annotations are maintained independently of the base mesh.
        update_analysis_overlay(render=False)

        # 3D load glyphs after solve
        if bool(state.show_glyphs) and S.case is not None:
            try:
                for g_mesh, g_color in viz.build_load_glyphs(S.case, S.model.mesh, diag):
                    pl.add_mesh(g_mesh, color=g_color, pickable=False)
            except Exception:
                pass

        # Interactive preview arrow / moment arc
        if (bool(state.show_preview_glyph) and state.active_tab == "loads"
                and not bool(getattr(state, "analysis_dock_open", False))):
            try:
                faces = viz.face_definitions(bounds, mesh=mesh_ref)
                if state.load_target == "face":
                    c = faces.get(state.selected_face, {}).get("centroid", np.zeros(3))
                else:
                    c = np.array([float(state.point_x), float(state.point_y), float(state.point_z)])

                if state.load_mode == "force":
                    arr_prev = viz.create_arrow_glyph(
                        c, _parse_vec3(state.force_vec), diag, reference_magnitude=1e5
                    )
                    if arr_prev is not None:
                        pl.add_mesh(arr_prev, color="#22c55e", pickable=False)
                elif state.load_mode == "moment":
                    m_vec = _get_moment_vector(state, bounds, mesh_ref)
                    arr_prev = viz.create_moment_glyph(c, m_vec, diag, reference_magnitude=1e5)
                    if arr_prev is not None:
                        pl.add_mesh(arr_prev, color="#c084fc", pickable=False)
                elif state.load_mode == "pressure":
                    # Pressure arrows pointing inward normal to face
                    norm = faces.get(state.selected_face, {}).get("normal", np.array([0, 0, 1]))
                    try:
                        pressure_value = float(state.pressure_val)
                    except (TypeError, ValueError):
                        pressure_value = 1e6
                    arr_prev = viz.create_arrow_glyph(
                        c, -norm * pressure_value, diag, reference_magnitude=1e6
                    )
                    if arr_prev is not None:
                        pl.add_mesh(arr_prev, color="#f59e0b", pickable=False)
            except Exception:
                pass

        # Point picking is an interactor observer; registering it on every redraw
        # accumulates callbacks and progressively slows the application.
        if not S.point_picker_installed:
            try:
                pl.enable_point_picking(
                    callback=_on_viewport_point_picked,
                    left_clicking=True,
                    show_message=False,
                    show_point=False,
                )
                S.point_picker_installed = True
            except Exception:
                pass

        if reset_camera:
            pl.reset_camera()
        elif saved_camera is not None:
            try:
                pl.camera_position = saved_camera
                pl.reset_camera_clipping_range()
            except Exception:
                pass
        ctrl.view_update()

    def _on_viewport_point_picked(point):
        """Callback triggered when the user clicks a point on the 3D model surface."""
        if point is None:
            return
        px, py, pz = float(point[0]), float(point[1]), float(point[2])
        _pick_point_direct(px, py, pz)

    def _pick_point_direct(px: float, py: float, pz: float):
        """Update point coordinates and snap to nearest face or element centroid."""
        if S.grid is not None:
            # The viewport may show an amplified deformed shape. Convert the picked
            # display-space point back to the reference mesh node before sampling stress.
            picked = np.array([px, py, pz], dtype=float)
            display_points = _displayed_mesh_points()
            nearest = int(np.argmin(np.linalg.norm(display_points - picked[None, :], axis=1)))
            reference_point = np.asarray(S.grid.points[nearest], float)
            px, py, pz = (float(reference_point[0]), float(reference_point[1]),
                          float(reference_point[2]))
        state.lab_point_x = float(np.round(px, 6))
        state.lab_point_y = float(np.round(py, 6))
        state.lab_point_z = float(np.round(pz, 6))
        if not bool(getattr(state, "analysis_dock_open", False)):
            # In setup mode, a viewport pick remains a load-target operation.
            state.point_x = float(np.round(px, 4))
            state.point_y = float(np.round(py, 4))
            state.point_z = float(np.round(pz, 4))
            state.point_preset = "Custom (Clicked)"
            state.load_target = "point"

            # Keep the previous convenience behavior for load/support face selection.
            bounds = S.grid.bounds if S.grid else tuple(state.model_bounds)
            mesh_ref = S.model.mesh if S.model else None
            faces = viz.face_definitions(bounds, mesh=mesh_ref)
            if faces:
                pt = np.array([px, py, pz])
                best_face = min(faces.keys(), key=lambda f: float(np.linalg.norm(faces[f]["centroid"] - pt)))
                state.selected_face = best_face
                state.support_face = best_face
            state.click_status = f"Picked load target ({px:.3f}, {py:.3f}, {pz:.3f}) m"
            _update_preview_snippet()
        else:
            state.click_status = f"Analysis sample selected ({px:.3f}, {py:.3f}, {pz:.3f}) m"
        try:
            _lab_stress_from_result()
        except Exception:
            pass
        if bool(getattr(state, "analysis_dock_open", False)):
            update_analysis_overlay()
        else:
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
        if state.geom_mode in ("step", "cv"):
            if state.geom_mode == "cv":
                step_file = str(Path(state.cv_step_path).resolve())
            elif state.cad_preset == "custom":
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

    # ------------------------------------------------------------------ live load response
    def _geometry_signature(spec):
        """Signature of mesh-defining inputs, separate from loads/material/supports."""
        params = tuple(sorted((str(key), repr(value)) for key, value in spec.params.items()))
        return repr(spec.mesh), params

    def _refresh_live_load_items():
        """Build compact, human-readable choices for the contextual live-load editor."""
        items = []
        for i, load in enumerate(list(getattr(state, "loads_list", []))):
            mode = str(load.get("mode", "force")).capitalize()
            target = (f"face {load.get('face', '')}" if load.get("target", "face") == "face"
                      else "picked point")
            detail = load.get("vec") or load.get("val", "")
            items.append({"title": f"{i + 1}. {mode} · {target} · {detail}", "value": str(i)})
        state.live_load_items = items
        if not items:
            state.live_load_selection = ""
            state.editing_load_index = -1

    def _start_live_load_edit(index):
        """Select one applied load and prepare its unit-load response basis."""
        index = int(index)
        loads = list(getattr(state, "loads_list", []))
        if not 0 <= index < len(loads):
            state.live_load_status = "Select an applied load to explore its response."
            return
        if state.config_mode != "interactive":
            state.live_load_status = "Live load editing is available in Interactive Builder mode."
            return

        if S.solved_spec_text is not None and str(state.spec_text) != S.solved_spec_text:
            warning = (
                "The model definition has unsolved edits. The displayed contour and point values "
                "are from the last solved setup. Run Solve before exploring live loads."
            )
            state.analysis_result_warning = warning
            state.live_load_status = warning
            return

        load = loads[index]
        mode = str(load.get("mode", "force"))
        try:
            vector = _parse_vec3(str(load.get("vec", "0, 0, 0")))
            pressure = float(load.get("val", 0.0) or 0.0)
        except (TypeError, ValueError):
            state.live_load_status = "This load's values cannot be edited in the live inspector."
            return

        S.live_response_suspended = True
        try:
            state.update({
                "editing_load_index": index,
                "live_load_selection": str(index),
                "live_load_mode": mode,
                "live_load_x": float(vector[0]),
                "live_load_y": float(vector[1]),
                "live_load_z": float(vector[2]),
                "live_pressure": pressure,
                "live_load_multiplier": 1.0,
                "live_load_value": "",
                "live_load_ready": False,
                "analysis_dock_open": True,
            })
        finally:
            S.live_response_suspended = False

        # Keep the load's geometric target visible behind the dock.  Moments and
        # pressure act on a face even when their reporting origin is a picked point.
        if mode in ("moment", "pressure") or load.get("target", "face") == "face":
            state.selected_face = load.get("face", state.selected_face)

        S.live_response = None
        try:
            current_spec = parse_spec(state.spec_text)
            if (S.cached_geometry_signature is not None
                    and _geometry_signature(current_spec) != S.cached_geometry_signature):
                state.live_load_status = (
                    "Geometry parameters differ from the last solved model. Run Solve before "
                    "exploring live load changes."
                )
                return
            if S.cached_solver is not None and S.model is not None:
                current_model, _ = build(current_spec, mesh=S.model.mesh)
                if not S.cached_solver.compatible(current_model):
                    state.live_load_status = (
                        "Material or support settings differ from the last solved model. "
                        "Run Solve before exploring live load changes."
                    )
                    return
        except Exception as exc:
            state.live_load_status = f"Current model definition is invalid: {exc}. Run Solve first."
            return
        if S.cached_solver is None or S.case is None:
            state.live_load_status = (
                "Live editing requires a solved, physically supported linear-elastic model. "
                "You can still inspect the current result; use Solve after changing a load."
            )
            return
        if mode == "pressure":
            state.live_load_value = f"Current pressure: {pressure:.6g} Pa"
        else:
            unit = "N·m" if mode == "moment" else "N"
            magnitude = float(np.linalg.norm(vector))
            state.live_load_value = (
                f"Current vector: [{vector[0]:.5g}, {vector[1]:.5g}, {vector[2]:.5g}] {unit} "
                f"· magnitude {magnitude:.5g} {unit}"
            )

        try:
            state.live_load_status = "Preparing cached unit-load response fields …"
            state.flush()
            S.live_response = S.cached_solver.prepare_live_load(S.case, index)
            state.live_load_ready = True
            state.live_load_status = (
                "Cached response ready. Change the components or scale; the model and "
                "point/plane results update without another FE solve."
            )
        except Exception as exc:
            state.live_load_ready = False
            state.live_load_status = f"Live editing unavailable: {exc}. Use Solve after editing."

    def _live_load_inputs_changed(**_):
        """Compose a changed active load with its cached unit-load response fields."""
        if S.live_response_suspended or S.live_response is None or not bool(state.live_load_ready):
            return
        index = int(getattr(state, "editing_load_index", -1))
        loads = list(getattr(state, "loads_list", []))
        if not 0 <= index < len(loads) or S.model is None:
            return
        previous_loads = list(loads)
        previous_editor = {
            "force_vec": getattr(state, "force_vec", "0, 0, 0"),
            "moment_mag": getattr(state, "moment_mag", "0"),
            "pressure_val": getattr(state, "pressure_val", "0"),
        }

        def _rollback_live_load(message, *, disable=False):
            state.loads_list = previous_loads
            state.force_vec = previous_editor["force_vec"]
            state.moment_mag = previous_editor["moment_mag"]
            state.pressure_val = previous_editor["pressure_val"]
            _rebuild_spec_from_interactive()
            _refresh_live_load_items()
            state.live_load_selection = str(index)
            if disable:
                state.live_load_ready = False
                S.live_response = None
            state.live_load_status = message

        try:
            multiplier = float(state.live_load_multiplier)
            if not np.isfinite(multiplier) or multiplier < 0.0:
                raise ValueError("Load scale must be finite and non-negative")
            mode = str(state.live_load_mode)
            updated = dict(loads[index])
            if mode in ("force", "moment"):
                base = np.array([float(state.live_load_x), float(state.live_load_y),
                                 float(state.live_load_z)], dtype=float)
                if not np.isfinite(base).all():
                    raise ValueError("Load components must be finite")
                vector = base * multiplier
                updated["vec"] = ", ".join(f"{v:.9g}" for v in vector)
                unit = "N·m" if mode == "moment" else "N"
                live_value = (
                    f"Current vector: [{vector[0]:.5g}, {vector[1]:.5g}, {vector[2]:.5g}] {unit} "
                    f"· magnitude {np.linalg.norm(vector):.5g} {unit}"
                )
                if mode == "force":
                    state.force_vec = updated["vec"]
                else:
                    magnitude = float(np.linalg.norm(vector))
                    state.moment_mag = f"{magnitude:.9g}"
            elif mode == "pressure":
                pressure = float(state.live_pressure)
                if not np.isfinite(pressure):
                    raise ValueError("Pressure must be finite")
                updated["val"] = f"{pressure * multiplier:.9g}"
                state.pressure_val = updated["val"]
                live_value = f"Current pressure: {pressure * multiplier:.6g} Pa"
            else:
                raise ValueError(f"Unsupported live load type: {mode}")

            loads[index] = updated
            state.loads_list = loads
            _refresh_live_load_items()
            state.live_load_selection = str(index)
            _rebuild_spec_from_interactive()
            spec = parse_spec(state.spec_text)
            if (S.cached_geometry_signature is not None
                    and _geometry_signature(spec) != S.cached_geometry_signature):
                _rollback_live_load(
                    "Geometry parameters changed since the last solve. The edit was rolled "
                    "back; solve the updated model first.", disable=True
                )
                return
            model, case = build(spec, mesh=S.model.mesh)
            if S.cached_solver is None or not S.cached_solver.compatible(model):
                _rollback_live_load(
                    "The mesh, material or supports changed. The edit was rolled back; "
                    "run Solve to rebuild the response cache.", disable=True
                )
                return
            if (S.live_response is None or index >= len(case.loads)
                    or not S.live_response.matches(case.loads[index])):
                S.live_response = S.cached_solver.prepare_live_load(case, index)
            result = S.live_response.evaluate(case.loads[index])

            # Publish the combined result without touching the viewport camera or view options.
            S.spec, S.model, S.case, S.res = spec, model, case, result
            S.solved_spec_text = str(state.spec_text)
            state.analysis_result_warning = ""
            S.scalars = viz.compute_scalars(result)
            S.u_pts = np.ascontiguousarray(result.u_nodal.T)
            state.field_items = list(S.scalars)
            if state.field not in S.scalars:
                state.field = "von Mises"
            div = viz.STRESS_UNITS[state.stress_unit]
            umax = float(S.scalars["|u|"][0].max())
            max_vm = float(S.scalars["von Mises"][0].max() / div)
            rx, ry, rz = result.reactions
            state.update({
                "kpi_max_vm": f"{max_vm:.3f} {state.stress_unit}",
                "kpi_max_u": f"{umax * 1e3:.3f} mm",
                "kpi_reactions": f"[{rx:.3g}, {ry:.3g}, {rz:.3g}] N",
                "kpi_error": f"{result.equilibrium_error:.2e}",
                "max_u_str": f"Max deflection: {umax * 1e3:.4g} mm",
                "status": (f"Live response · peak von Mises {max_vm:.4g} {state.stress_unit} · "
                           f"max deflection {umax * 1e3:.4g} mm"),
                "status_type": "success" if result.equilibrium_error < 1e-5 else "warning",
                "live_load_status": "Live result updated from cached unit-load responses.",
                "live_load_value": live_value,
            })
            _lab_stress_from_result()
            update_scene(reset_camera=False)
        except Exception as exc:
            _rollback_live_load(f"Live update failed; previous load retained: {exc}")

    def _select_live_load(live_load_selection, **_):
        if str(live_load_selection).strip() == "":
            return
        _start_live_load_edit(int(live_load_selection))

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
                if state.geom_mode in ("step", "cv"):
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
            S.cached_solver = None
            S.live_response = None
            state.update({
                "editing_load_index": -1,
                "live_load_selection": "",
                "live_load_ready": False,
                "live_load_status": "Select an applied load to prepare its live response.",
            })
            try:
                if model.bcs:
                    # Assemble and factorise once, then use that factorisation for the
                    # initial solve as well as every later live-load response.
                    S.cached_solver = CachedLinearSolver(model)
                    res = S.cached_solver.solve(case)
                else:
                    # Free-body inertial-frame cases use the normal solver path.
                    res = solve_case(model, case)
                state.live_response_status = (
                    "Stiffness matrix and factorisation cached. Select a load to prepare "
                    "unit-load response fields for live exploration."
                    if S.cached_solver is not None else
                    "Free-body analysis uses the standard solve path; live load scaling is unavailable."
                )
            except (ValueError, RuntimeError, TypeError, np.linalg.LinAlgError) as cache_exc:
                # Preserve the regular solver as a robust fallback for configurations
                # which the reusable supported-body cache cannot factorise.
                S.cached_solver = None
                res = solve_case(model, case)
                state.live_response_status = (
                    f"Live response cache unavailable: {cache_exc}. The model was solved "
                    "normally; edit loads and use Solve to update the result."
                )
            S.res, S.grid = res, viz.to_pyvista(model.mesh)
            S.spec, S.model, S.case = spec, model, case
            S.solved_spec_text = str(state.spec_text)
            state.analysis_result_warning = ""
            S.cached_geometry_signature = _geometry_signature(spec)
            S.scalars = viz.compute_scalars(res)
            S.u_pts = np.ascontiguousarray(res.u_nodal.T)
            _refresh_live_load_items()

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
            # Refresh the read-only Lab snapshot from the newly solved result, then
            # redraw once so a principal-plane overlay follows the refreshed tensor.
            _lab_stress_from_result()
            if bool(getattr(state, "analysis_dock_open", False)):
                update_scene(reset_camera=False)
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
        state.editing_load_index = -1
        state.live_load_selection = ""
        S.live_response = None

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
            state.editing_load_index = -1
            state.live_load_selection = ""
            S.live_response = None
            _rebuild_spec_from_interactive()
            run_solve()

    def clear_all_loads():
        """Remove all active boundary supports and loads."""
        state.loads_list = []
        state.supports_list = []
        state.editing_load_index = -1
        state.live_load_selection = ""
        S.live_response = None
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

    def run_cv_analysis():
        """Analyze a real-life photo with the lightweight CV model to determine object & parameters."""
        try:
            img_source = None
            if state.cv_image_b64 and state.cv_source_mode == "webcam":
                img_source = state.cv_image_b64
            elif state.cv_source_mode == "upload" and str(state.cv_custom_path).strip():
                img_source = str(state.cv_custom_path).strip()
            else:
                preset = state.cv_preset_path or "cad_models/cv_samples/bracket_photo.png"
                project_root = Path(__file__).resolve().parents[2]
                img_source = str((project_root / preset).resolve())

            state.cv_status = "Analyzing real-life object photo..."
            ref_scale = float(state.cv_ref_scale or 0.30)
            res = analyze_image_object(img_source, reference_scale=ref_scale)

            state.cv_detected_type = res.object_type
            state.cv_friendly_name = res.friendly_name
            state.cv_confidence_pct = int(res.confidence * 100)
            state.cv_features_text = "\n".join(f"• {f}" for f in res.detected_features)
            state.cv_annotated_preview = res.annotated_image_b64

            # Map parameters to reactive state
            p = res.parameters
            if res.object_type == "bracket":
                state.cv_param_leg_x = p.get("leg_x", 0.20)
                state.cv_param_leg_y = p.get("leg_y", 0.20)
                state.cv_param_width = p.get("width", 0.06)
                state.cv_param_thickness = p.get("thickness", 0.015)
                state.cv_param_hole_radius = p.get("hole_radius", 0.008)
            elif res.object_type == "plate_hole":
                state.cv_param_length = p.get("length", 0.40)
                state.cv_param_width = p.get("width", 0.20)
                state.cv_param_thickness = p.get("thickness", 0.02)
                state.cv_param_hole_diameter = p.get("hole_diameter", 0.06)
            elif res.object_type == "beam":
                state.cv_param_length = p.get("length", 1.0)
                state.cv_param_height = p.get("height", 0.15)
                state.cv_param_width = p.get("width", 0.10)
            elif res.object_type == "ibeam":
                state.cv_param_length = p.get("length", 1.0)
                state.cv_param_height = p.get("height", 0.20)
                state.cv_param_flange_width = p.get("flange_width", 0.12)
                state.cv_param_flange_thickness = p.get("flange_thickness", 0.02)
                state.cv_param_web_thickness = p.get("web_thickness", 0.015)
            elif res.object_type == "shaft":
                state.cv_param_length = p.get("length", 0.60)
                state.cv_param_diameter_main = p.get("diameter_main", 0.08)
                state.cv_param_diameter_step = p.get("diameter_step", 0.05)
                state.cv_param_step_length = p.get("step_length", 0.20)
            elif res.object_type == "pipe":
                state.cv_param_length = p.get("length", 0.80)
                state.cv_param_outer_diameter = p.get("outer_diameter", 0.10)
                state.cv_param_inner_diameter = p.get("inner_diameter", 0.07)

            image_w, image_h = res.pixel_dimensions
            state.cv_status = (
                f"{res.friendly_name} · {int(res.confidence * 100)}% model confidence · "
                f"{image_w} × {image_h} px. Dimensions use your assumed reference scale; "
                "review the mask and edit every CAD parameter before export."
            )
        except Exception as e:
            traceback.print_exc()
            state.cv_status = f"CV analysis failed: {e}"

    def _refresh_cv_multiview_commands():
        """Regenerate the copyable NeRF / Gaussian-splatting CLI commands."""
        images_dir = str(getattr(state, "cv_mv_images_dir", "") or "").strip()
        if not images_dir:
            state.cv_mv_commands = "Prepare a multi-view capture first."
            return
        output_dir = str(getattr(state, "cv_mv_ns_data_dir", "") or "").strip()
        if not output_dir:
            output_dir = str(Path(images_dir).parent / "nerfstudio_data")
            state.cv_mv_ns_data_dir = output_dir
        state.cv_mv_commands = nerfstudio_commands(
            images_dir, output_dir, str(getattr(state, "cv_mv_method", "splatfacto"))
        )

    def set_cv_multiview_method(method):
        """Update the active NeRF / Gaussian representation and its CLI commands."""
        state.cv_mv_method = str(method or "splatfacto")
        _refresh_cv_multiview_commands()

    def prepare_cv_multiview_dataset():
        """Save browser-selected views and produce the NeRF/3DGS data commands."""
        if bool(getattr(state, "cv_mv_running", False)) or bool(getattr(state, "cv_mv_colmap_running", False)):
            state.cv_mv_status = "Wait for the active reconstruction process to finish before replacing this capture."
            return
        try:
            items = list(getattr(state, "cv_mv_images", []) or [])
            if not items:
                state.cv_mv_status = "Choose at least three different photos of the same object first."
                return
            project_root = Path(__file__).resolve().parents[2]
            output_root = project_root / "cad_models" / "reconstruction_sessions"
            session = prepare_capture_session(
                items, output_root, session_name=str(getattr(state, "cv_mv_name", "mechanical_part"))
            )
            ns_data_dir = session.session_dir / "nerfstudio_data"
            state.update({
                "cv_mv_session_dir": str(session.session_dir),
                "cv_mv_images_dir": str(session.images_dir),
                "cv_mv_ns_data_dir": str(ns_data_dir),
                "cv_mv_image_count": session.image_count,
                "cv_mv_summary": session.summary,
                "cv_mv_warnings": "\n".join(f"• {warning}" for warning in session.warnings),
                "cv_mv_status": f"Capture saved: {session.session_dir}",
                "cv_mv_preprocess_status": "Camera poses not processed yet.",
                "cv_mv_preprocess_log": "",
                "cv_mv_preprocess_ready": False,
                "cv_mv_colmap_ready": False,
                "cv_mv_colmap_running": False,
                "cv_mv_colmap_status": "Standard COLMAP dataset has not been built yet.",
                "cv_mv_colmap_log": "",
                "cv_mv_colmap_workspace": "",
                "cv_mv_colmap_model": "",
                "cv_mv_colmap_points": "",
            })
            _refresh_cv_multiview_commands()
        except Exception as exc:
            traceback.print_exc()
            state.cv_mv_status = f"Capture preparation failed: {exc}"

    def run_cv_multiview_preprocess():
        """Run COLMAP camera-pose recovery through Nerfstudio without blocking the UI."""
        images_dir = str(getattr(state, "cv_mv_images_dir", "") or "").strip()
        output_dir = str(getattr(state, "cv_mv_ns_data_dir", "") or "").strip()
        if not images_dir or not output_dir:
            state.cv_mv_preprocess_status = "Prepare the multi-view image dataset first."
            return
        if bool(getattr(state, "cv_mv_running", False)):
            return
        if not tool_availability()["ns_process_data"]:
            state.cv_mv_preprocess_status = (
                "Nerfstudio is not installed in this environment. Run the displayed "
                "ns-process-data command in the environment where Nerfstudio and COLMAP are installed."
            )
            return

        state.update({
            "cv_mv_running": True,
            "cv_mv_preprocess_ready": False,
            "cv_mv_preprocess_status": "Estimating camera poses and preparing transforms.json with COLMAP…",
        })

        def _worker():
            try:
                result = run_nerfstudio_preprocess(images_dir, output_dir)
                state.update({
                    "cv_mv_preprocess_ready": True,
                    "cv_mv_preprocess_status": (
                        f"Camera poses ready · transforms.json saved to {result['transforms_path']}"
                    ),
                    "cv_mv_preprocess_log": result.get("log_tail", ""),
                })
            except Exception as exc:
                state.update({
                    "cv_mv_preprocess_ready": False,
                    "cv_mv_preprocess_status": f"Reconstruction failed: {exc}",
                })
            finally:
                state.cv_mv_running = False

        threading.Thread(target=_worker, name="mos-multiview-preprocess", daemon=True).start()

    def run_cv_colmap_sparse():
        """Build the standard COLMAP sparse folder used by 2DGS and Graphdeco trainers."""
        images_dir = str(getattr(state, "cv_mv_images_dir", "") or "").strip()
        if not images_dir:
            state.cv_mv_colmap_status = "Build the multi-view capture dataset first."
            return
        if bool(getattr(state, "cv_mv_colmap_running", False)):
            return
        if not tool_availability()["colmap"]:
            state.cv_mv_colmap_status = "COLMAP is not installed or is not available on PATH."
            return
        state.update({
            "cv_mv_colmap_running": True,
            "cv_mv_colmap_ready": False,
            "cv_mv_colmap_status": "Running COLMAP feature extraction, matching and sparse mapping…",
        })

        def _worker():
            try:
                result = run_colmap_sparse_reconstruction(images_dir)
                state.update({
                    "cv_mv_colmap_ready": True,
                    "cv_mv_colmap_workspace": str(result["workspace_dir"]),
                    "cv_mv_colmap_model": str(result["model_dir"]),
                    "cv_mv_colmap_points": str(result["point_cloud_path"]),
                    "cv_mv_colmap_status": (
                        f"COLMAP model ready · {result['model_count']} sparse model(s) · "
                        f"model folder: {result['model_dir']} · point cloud: {result['point_cloud_path']}"
                    ),
                    "cv_mv_colmap_log": result.get("log_tail", ""),
                })
            except Exception as exc:
                state.update({
                    "cv_mv_colmap_ready": False,
                    "cv_mv_colmap_status": f"COLMAP reconstruction failed: {exc}",
                })
            finally:
                state.cv_mv_colmap_running = False

        threading.Thread(target=_worker, name="mos-colmap-sparse", daemon=True).start()

    def inspect_cv_gaussian_ply():
        """Extract the means of imported 3D Gaussian splats as an RGB point cloud."""
        try:
            source = str(getattr(state, "cv_gaussian_ply_path", "") or "").strip()
            if not source:
                state.cv_gaussian_status = "Enter the path to a trained Gaussian-splat .ply file."
                return
            summary = gaussian_ply_to_point_cloud(source)
            bounds = summary.bounds
            opacity = (
                f" · mean opacity {summary.mean_opacity:.3f}"
                if summary.mean_opacity is not None else ""
            )
            scale = (
                f" · mean splat scale {summary.mean_scale:.4g}"
                if summary.mean_scale is not None else ""
            )
            kind = "3D Gaussian attributes detected" if summary.has_gaussian_attributes else "PLY point cloud (no Gaussian attributes found)"
            state.update({
                "cv_gaussian_centers_path": str(summary.centers_path),
                "cv_gaussian_status": (
                    f"{kind} · {summary.gaussian_count:,} centers{opacity}{scale}\n"
                    f"Bounds X [{bounds[0][0]:.4g}, {bounds[0][1]:.4g}] · "
                    f"Y [{bounds[1][0]:.4g}, {bounds[1][1]:.4g}] · "
                    f"Z [{bounds[2][0]:.4g}, {bounds[2][1]:.4g}]\n"
                    f"Exported RGB point cloud: {summary.centers_path}"
                ),
            })
        except Exception as exc:
            traceback.print_exc()
            state.cv_gaussian_status = f"Gaussian PLY import failed: {exc}"

    def calibrate_cv_gaussian_cloud():
        """Set the point-cloud's global scale from one known physical dimension."""
        try:
            source = str(getattr(state, "cv_gaussian_centers_path", "") or "").strip()
            if not source:
                state.cv_gaussian_scale_status = "Extract Gaussian centers first."
                return
            known_extent = float(state.cv_gaussian_known_extent_m)
            axis = str(state.cv_gaussian_scale_axis or "x")
            result = calibrate_point_cloud_scale(source, known_extent, axis=axis)
            state.update({
                "cv_gaussian_calibrated_path": str(result["output_path"]),
                "cv_gaussian_scale_status": (
                    f"Metric scale applied · {axis.upper()} extent = {known_extent:.4g} m · "
                    f"uniform scale factor = {result['scale_factor']:.6g}\n"
                    f"Calibrated point cloud: {result['output_path']}"
                ),
            })
        except Exception as exc:
            traceback.print_exc()
            state.cv_gaussian_scale_status = f"Scale calibration failed: {exc}"

    def run_cv_universal_reconstruction():
        """Build a single-image 2.5D preview and an optional silhouette STEP solid."""
        source = str(getattr(state, "cv3d_image_b64", "") or "").strip()
        if not source.startswith("data:image"):
            state.cv3d_status = "Choose a JPG, PNG, or WEBP image first."
            return
        if bool(getattr(state, "cv3d_running", False)):
            state.cv3d_status = "A reconstruction is already running."
            return

        try:
            settings = {
                "object_width_m": float(state.cv3d_object_width_m),
                "body_thickness_m": float(state.cv3d_body_thickness_m),
                "relief_depth_m": float(state.cv3d_relief_depth_m),
                "max_resolution": int(state.cv3d_mesh_resolution),
                "geometry_mode": str(state.cv3d_geometry_mode),
                "segmentation_mode": str(state.cv3d_segmentation_mode),
            }
            if settings["object_width_m"] <= 0 or settings["body_thickness_m"] <= 0 or settings["relief_depth_m"] < 0:
                raise ValueError("Width and thickness must be positive; relief depth cannot be negative.")
        except (TypeError, ValueError) as exc:
            state.cv3d_status = f"Invalid reconstruction settings: {exc}"
            return

        S.preview_mesh = None
        state.update({
            "cv3d_running": True,
            "cv3d_preview_active": False,
            "cv3d_preview_path": "",
            "cv3d_step_path": "",
            "cv3d_warning_text": "",
            "cv3d_status": "Preparing foreground mask and reconstructing the 3D preview…",
        })
        update_scene(reset_camera=False)
        project_root = Path(__file__).resolve().parents[2]
        output_root = project_root / "cad_models" / "image_reconstruction"

        def _worker():
            try:
                result = reconstruct_single_image(
                    {"name": str(getattr(state, "cv3d_image_name", "") or "uploaded_image"), "data": source},
                    output_root,
                    object_width_m=settings["object_width_m"],
                    body_thickness_m=settings["body_thickness_m"],
                    relief_depth_m=settings["relief_depth_m"],
                    max_resolution=settings["max_resolution"],
                    geometry_mode=settings["geometry_mode"],
                    segmentation_mode=settings["segmentation_mode"],
                    export_step=True,
                )
                step_path = str(result.step_path) if result.step_path else ""
                summary = (
                    f"Preview ready · {result.vertex_count:,} vertices · {result.face_count:,} faces\n"
                    f"Foreground: {result.segmentation_method}\n"
                    f"Depth: {result.depth_method}\n"
                    f"Approximate preview envelope: {result.width_m:.3g} × {result.height_m:.3g} × "
                    f"{result.thickness_m + (settings['relief_depth_m'] if settings['geometry_mode'] == 'depth_relief' else 0.0):.3g} m"
                )
                state.update({
                    "cv3d_preview_path": str(result.preview_mesh_path),
                    "cv3d_mask_path": str(result.mask_path),
                    "cv3d_depth_path": str(result.depth_preview_path) if result.depth_preview_path else "",
                    "cv3d_step_path": step_path,
                    "cv3d_warning_text": "\n".join(f"• {warning}" for warning in result.warnings),
                    "cv3d_status": summary + ("\nSilhouette STEP is ready to load into CAD/FEM." if step_path else "\nPreview only: STEP export needs a working Gmsh installation."),
                })
            except Exception as exc:
                traceback.print_exc()
                state.update({
                    "cv3d_preview_path": "",
                    "cv3d_step_path": "",
                    "cv3d_status": f"Reconstruction failed: {type(exc).__name__}: {exc}",
                })
            finally:
                state.cv3d_running = False

        threading.Thread(target=_worker, name="mos-single-image-reconstruction", daemon=True).start()

    def show_cv3d_preview():
        """Show the generated surface in the CAD viewport without changing FEM state."""
        if S.preview_mesh is None:
            path = str(getattr(state, "cv3d_preview_path", "") or "").strip()
            if not path or not Path(path).is_file():
                state.cv3d_status = "Generate a preview mesh first."
                return
            try:
                S.preview_mesh = pv.read(path)
            except Exception as exc:
                state.cv3d_status = f"Could not load preview mesh: {exc}"
                return
        state.cv3d_preview_active = True
        update_scene(reset_camera=True)
        state.cv3d_status = "Showing the single-image 3D preview. The active FEM model is unchanged."

    def clear_cv3d_preview():
        """Return the viewport to the last solved FEM model, if present."""
        state.cv3d_preview_active = False
        update_scene(reset_camera=False)

    def load_cv3d_step_to_fem():
        """Load the silhouette-extrusion STEP into the existing CAD/FEM pipeline."""
        step_path = str(getattr(state, "cv3d_step_path", "") or "").strip()
        if not step_path or not Path(step_path).is_file():
            state.cv3d_status = "No STEP export is available. Check Gmsh and reconstruct the image again."
            return
        # Explicit user action: the STEP solid now becomes the active geometry.
        state.cv3d_preview_active = False
        S.preview_mesh = None
        state.cv_step_path = step_path
        state.geom_mode = "cv"
        state.config_mode = "interactive"
        state.cad_mesh_size = max(0.02, min(0.05, float(state.cv3d_object_width_m) / 10.0))
        state.active_tab = "model"
        _rebuild_spec_from_interactive()
        run_solve()
        if state.status_type == "error":
            state.cv3d_status = f"STEP was generated but CAD/FEM loading failed: {state.status}"
        else:
            state.cv3d_status = "Silhouette STEP loaded through the existing CAD/FEM importer. Review supports, loads, and mesh quality before interpreting results."

    def generate_cv_cad_and_load():
        """Build parametric 3D STEP solid from detected object and load into CAD engine."""
        try:
            obj_type = state.cv_detected_type
            if not obj_type:
                run_cv_analysis()
                obj_type = state.cv_detected_type
                if not obj_type:
                    state.cv_status = "Please analyze an object photo first."
                    return

            params = {}
            if obj_type == "bracket":
                params = {
                    "leg_x": float(state.cv_param_leg_x),
                    "leg_y": float(state.cv_param_leg_y),
                    "width": float(state.cv_param_width),
                    "thickness": float(state.cv_param_thickness),
                    "hole_radius": float(state.cv_param_hole_radius),
                }
            elif obj_type == "plate_hole":
                params = {
                    "length": float(state.cv_param_length),
                    "width": float(state.cv_param_width),
                    "thickness": float(state.cv_param_thickness),
                    "hole_diameter": float(state.cv_param_hole_diameter),
                }
            elif obj_type == "beam":
                params = {
                    "length": float(state.cv_param_length),
                    "height": float(state.cv_param_height),
                    "width": float(state.cv_param_width),
                }
            elif obj_type == "ibeam":
                params = {
                    "length": float(state.cv_param_length),
                    "height": float(state.cv_param_height),
                    "flange_width": float(state.cv_param_flange_width),
                    "flange_thickness": float(state.cv_param_flange_thickness),
                    "web_thickness": float(state.cv_param_web_thickness),
                }
            elif obj_type == "shaft":
                params = {
                    "length": float(state.cv_param_length),
                    "diameter_main": float(state.cv_param_diameter_main),
                    "diameter_step": float(state.cv_param_diameter_step),
                    "step_length": float(state.cv_param_step_length),
                }
            elif obj_type == "pipe":
                params = {
                    "length": float(state.cv_param_length),
                    "outer_diameter": float(state.cv_param_outer_diameter),
                    "inner_diameter": float(state.cv_param_inner_diameter),
                }

            out_step = Path(state.cv_step_path).resolve()
            out_step.parent.mkdir(parents=True, exist_ok=True)
            state.cv_status = f"Generating 3D OpenCASCADE STEP CAD solid for {obj_type}..."
            generate_cad_from_parameters(obj_type, params, out_step)

            state.cv_status = f"Generated {out_step.name}. Meshing solid and solving..."
            state.geom_mode = "cv"
            if state.config_mode == "interactive":
                _rebuild_spec_from_interactive()
            run_solve()
            state.cv_status = f"Active: {state.cv_friendly_name} loaded and meshed in CAD engine!"
        except Exception as e:
            traceback.print_exc()
            state.cv_status = f"CAD generation error: {e}"

    ctrl.apply_current_load = apply_current_load
    ctrl.add_support_to_list = add_support_to_list
    ctrl.remove_support = remove_support
    ctrl.remove_load = remove_load
    ctrl.clear_all_loads = clear_all_loads
    ctrl.start_live_load_edit = _start_live_load_edit
    ctrl.convert_image_drawing_to_cad = convert_image_drawing_to_cad
    ctrl.run_cv_analysis = run_cv_analysis
    ctrl.generate_cv_cad_and_load = generate_cv_cad_and_load
    ctrl.prepare_cv_multiview_dataset = prepare_cv_multiview_dataset
    ctrl.run_cv_multiview_preprocess = run_cv_multiview_preprocess
    ctrl.run_cv_colmap_sparse = run_cv_colmap_sparse
    ctrl.inspect_cv_gaussian_ply = inspect_cv_gaussian_ply
    ctrl.calibrate_cv_gaussian_cloud = calibrate_cv_gaussian_cloud
    ctrl.run_cv_universal_reconstruction = run_cv_universal_reconstruction
    ctrl.show_cv3d_preview = show_cv3d_preview
    ctrl.clear_cv3d_preview = clear_cv3d_preview
    ctrl.load_cv3d_step_to_fem = load_cv3d_step_to_fem

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

    # ------------------------------------------------------------------ learning lab helpers
    def _lab_float(name, fallback=0.0):
        """Read a numeric Learning Lab field without letting an empty widget crash the page."""
        try:
            value = float(getattr(state, name))
            return value if np.isfinite(value) else float(fallback)
        except (TypeError, ValueError, AttributeError):
            return float(fallback)

    def _refresh_learning_lab():
        """Recompute the tensor experiment and every linked visual from current inputs."""
        if bool(getattr(S, "lab_refresh_suspended", False)):
            return
        result = analyse_stress_tensor(
            _lab_float("lab_sigma_xx"), _lab_float("lab_sigma_yy"),
            _lab_float("lab_sigma_zz"), _lab_float("lab_tau_xy"),
            _lab_float("lab_tau_yz"), _lab_float("lab_tau_zx"),
            _lab_float("lab_azimuth"), _lab_float("lab_elevation"),
        )
        principal = result["principal"]
        normal = result["normal"]
        traction = result["traction"]
        state.update({
            "lab_tensor_svg": tensor_diagram_svg(result),
            "lab_mohr_svg": mohr_diagram_svg(principal),
            "lab_matrix_html": render_tensor_matrix(result),
            "lab_directions_html": render_direction_rows(result),
            "lab_sigma_1": f"{principal[0]:.2f}",
            "lab_sigma_2": f"{principal[1]:.2f}",
            "lab_sigma_3": f"{principal[2]:.2f}",
            "lab_max_shear": f"{result['max_shear']:.2f} MPa",
            "lab_von_mises": f"{result['von_mises']:.2f} MPa",
            "lab_tresca": f"{result['tresca']:.2f} MPa",
            "lab_normal_stress": f"{result['normal_stress']:.2f} MPa",
            "lab_plane_shear": f"{result['shear_stress']:.2f} MPa",
            "lab_plane_normal": "[" + ", ".join(f"{x:.3f}" for x in normal) + "]",
            "lab_plane_traction": "[" + ", ".join(f"{x:.2f}" for x in traction) + "] MPa",
        })
        if bool(getattr(state, "lab_snapshot_saved", False)):
            saved = _lab_float("lab_snapshot_sigma1")
            delta = float(principal[0]) - saved
            state.lab_snapshot_delta = f"{delta:+.2f} MPa"
            state.lab_snapshot_current = f"{principal[0]:.2f} MPa"

    def _load_lab_case(case_name):
        values = LAB_CASE_PRESETS.get(case_name)
        if values is None:
            return
        state.update(dict(zip(
            ("lab_sigma_xx", "lab_sigma_yy", "lab_sigma_zz",
             "lab_tau_xy", "lab_tau_yz", "lab_tau_zx"), values
        )))
        _refresh_learning_lab()

    def _refresh_mechanics_tool():
        """Update one compact textbook design check from its clearly labelled inputs."""
        try:
            result = mechanics_quick_check(
                str(state.lab_tool_mode), _lab_float("lab_q1"),
                _lab_float("lab_q2"), _lab_float("lab_q3"), _lab_float("lab_q4"),
            )
            state.update({
                "lab_quick_title": result["title"],
                "lab_quick_primary_label": result["primary_label"],
                "lab_quick_primary_value": result["primary_value"],
                "lab_quick_secondary_label": result["secondary_label"],
                "lab_quick_secondary_value": result["secondary_value"],
                "lab_quick_tertiary_label": result["tertiary_label"],
                "lab_quick_tertiary_value": result["tertiary_value"],
                "lab_quick_formula": result["formula"],
                "lab_quick_note": result["note"],
                "lab_quick_error": "",
            })
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            state.lab_quick_error = str(exc)

    def _load_mechanics_tool(mode):
        presets = {
            "beam": (500.0, 40.0, 80.0, 180.0,
                     "Bending moment M (N·m)", "Section width b (mm)",
                     "Section height h (mm)", "Allowable stress (MPa)"),
            "shaft": (150.0, 30.0, 500.0, 79.0,
                      "Torque T (N·m)", "Shaft diameter d (mm)",
                      "Shaft length L (mm)", "Shear modulus G (GPa)"),
            "vessel": (2.0, 500.0, 8.0, 150.0,
                       "Internal pressure p (MPa)", "Internal diameter d (mm)",
                       "Wall thickness t (mm)", "Allowable stress (MPa)"),
        }
        if mode not in presets:
            return
        values = presets[mode]
        state.update({
            "lab_q1": values[0], "lab_q2": values[1],
            "lab_q3": values[2], "lab_q4": values[3],
            "lab_q1_label": values[4], "lab_q2_label": values[5],
            "lab_q3_label": values[6], "lab_q4_label": values[7],
        })
        _refresh_mechanics_tool()

    def _lab_plane_vector(tensor=None):
        """Resolve a preset or custom plane normal from the active stress tensor."""
        preset = str(getattr(state, "lab_plane_preset", "custom"))
        axes = {
            "+X": np.array([1.0, 0.0, 0.0]), "-X": np.array([-1.0, 0.0, 0.0]),
            "+Y": np.array([0.0, 1.0, 0.0]), "-Y": np.array([0.0, -1.0, 0.0]),
            "+Z": np.array([0.0, 0.0, 1.0]), "-Z": np.array([0.0, 0.0, -1.0]),
        }
        if preset in axes:
            return axes[preset]
        if preset.startswith("Face: ") and S.grid is not None:
            face_key = preset.split(": ", 1)[1]
            faces = viz.face_definitions(S.grid.bounds, mesh=S.model.mesh if S.model else None)
            if face_key in faces:
                return np.asarray(faces[face_key]["normal"], dtype=float)
            raise ValueError(f"Unknown mesh face normal: {face_key}")
        if preset.startswith("Principal "):
            if tensor is None:
                tensor = np.array([
                    [_lab_float("lab_sigma_xx"), _lab_float("lab_tau_xy"), _lab_float("lab_tau_zx")],
                    [_lab_float("lab_tau_xy"), _lab_float("lab_sigma_yy"), _lab_float("lab_tau_yz")],
                    [_lab_float("lab_tau_zx"), _lab_float("lab_tau_yz"), _lab_float("lab_sigma_zz")],
                ], dtype=float)
            _, vectors = np.linalg.eigh(0.5 * (np.asarray(tensor) + np.asarray(tensor).T))
            vectors = vectors[:, ::-1]
            index = int(preset[-1]) - 1
            return vectors[:, index]
        raw = str(getattr(state, "lab_plane_custom", "0, 0, 1"))
        parts = [float(x.strip()) for x in raw.split(",")]
        if len(parts) != 3 or not np.isfinite(parts).all() or np.linalg.norm(parts) < 1e-14:
            raise ValueError("Enter a finite, non-zero plane normal as x, y, z")
        return np.asarray(parts, dtype=float)

    def _lab_stress_from_result():
        """Refresh all FEA-backed Lab readouts at the editable XYZ location."""
        if getattr(S, "res", None) is None or getattr(S, "grid", None) is None:
            state.lab_status = "No solved result. Solve a model in Analysis Workspace first."
            state.lab_model_summary = lab_model_summary(None, None, None)
            return
        try:
            xyz = np.array([_lab_float("lab_point_x"), _lab_float("lab_point_y"),
                            _lab_float("lab_point_z")], dtype=float)
            if not np.isfinite(xyz).all():
                raise ValueError("Point coordinates must be finite")
            points = np.asarray(S.grid.points, float)
            sample = lab_evaluate_stress(S.res, points, xyz)
            faces = viz.face_definitions(S.grid.bounds, mesh=S.model.mesh if S.model else None)
            plane_items = [{"title": name, "value": name} for name in
                           ["Custom", "+X", "-X", "+Y", "-Y", "+Z", "-Z",
                            "Principal 1", "Principal 2", "Principal 3"]]
            plane_items.extend({"title": f"Face normal · {key}", "value": f"Face: {key}"}
                               for key in faces)
            state.lab_plane_items = plane_items
            tensor_mpa = sample["tensor"] / 1.0e6
            traction = lab_evaluate_traction(S.res, points, xyz, _lab_plane_vector(tensor_mpa))
            # These state changes would otherwise trigger the tensor watcher once per
            # sampled point, then repeat the same SVG/matrix work explicitly below.
            S.lab_refresh_suspended = True
            try:
                # Lab tensor inputs are MPa; the solver and Results API remain in Pa.
                state.update({
                "lab_sigma_xx": float(tensor_mpa[0, 0]),
                "lab_sigma_yy": float(tensor_mpa[1, 1]),
                "lab_sigma_zz": float(tensor_mpa[2, 2]),
                "lab_tau_xy": float(tensor_mpa[0, 1]),
                "lab_tau_yz": float(tensor_mpa[1, 2]),
                "lab_tau_zx": float(tensor_mpa[0, 2]),
                "lab_node_index": str(sample["node_index"]),
                "lab_point_coords": "[" + ", ".join(f"{v:.6g}" for v in points[sample["node_index"]]) + "] m",
                "lab_status": (f"FEA result · nearest node {sample['node_index']} · "
                               f"distance {np.linalg.norm(points[sample['node_index']] - xyz):.4g} m"),
                "lab_model_summary": lab_model_summary(S.model, S.case, S.grid),
                "lab_strain_matrix_html": lab_tensor_matrix_html(
                    S.res.eps_nodal[:, :, sample["node_index"]], scale=1.0, digits=4),
                "lab_displacement": "[" + ", ".join(
                    f"{v:.5g}" for v in S.res.u_nodal[:, sample["node_index"]]) + "] m",
                "lab_has_result": True,
                })
                n_unit = traction["normal"]
                state.lab_azimuth = float(np.degrees(np.arctan2(n_unit[1], n_unit[0])))
                state.lab_elevation = float(np.degrees(np.arcsin(np.clip(n_unit[2], -1.0, 1.0))))
            finally:
                S.lab_refresh_suspended = False
            _refresh_learning_lab()
            # Use the core Results API for the selected plane, with Pa -> MPa conversion.
            state.update({
                "lab_plane_normal": "[" + ", ".join(f"{v:.4f}" for v in traction["normal"]) + "]",
                "lab_plane_traction": "[" + ", ".join(f"{v / 1e6:.4f}" for v in traction["traction"]) + "] MPa",
                "lab_normal_stress": f"{traction['normal_stress'] / 1e6:.4f} MPa",
                "lab_plane_shear": f"{traction['shear_stress'] / 1e6:.4f} MPa",
                "lab_shear_vector": "[" + ", ".join(f"{v / 1e6:.4f}" for v in traction["shear_vector"]) + "] MPa",
                "lab_fea_principal": "[" + ", ".join(f"{v / 1e6:.4f}" for v in sample["principal"]) + "] MPa",
                "lab_fea_vm": f"{sample['von_mises'] / 1e6:.4f} MPa",
                "lab_fea_tresca": f"{sample['tresca'] / 1e6:.4f} MPa",
            })
        except Exception as exc:
            state.lab_status = f"Point/plane analysis error: {exc}"

    def _lab_plane_changed(**_):
        if getattr(S, "res", None) is not None:
            _lab_stress_from_result()
        else:
            try:
                normal = _lab_plane_vector()
                normal = normal / np.linalg.norm(normal)
                state.lab_azimuth = float(np.degrees(np.arctan2(normal[1], normal[0])))
                state.lab_elevation = float(np.degrees(np.arcsin(np.clip(normal[2], -1.0, 1.0))))
                _refresh_learning_lab()
                state.lab_plane_normal = "[" + ", ".join(f"{v:.4f}" for v in normal) + "]"
            except Exception as exc:
                state.lab_status = str(exc)
        if bool(getattr(state, "analysis_dock_open", False)):
            update_analysis_overlay()


    def save_lab_snapshot():
        """Save the current principal state as comparison case A."""
        state.update({
            "lab_snapshot_saved": True,
            "lab_snapshot_label": str(state.lab_case),
            "lab_snapshot_sigma1": _lab_float("lab_sigma_1"),
            "lab_snapshot_current": f"{_lab_float('lab_sigma_1'):.2f} MPa",
            "lab_snapshot_delta": "0.00 MPa",
        })

    ctrl.save_lab_snapshot = save_lab_snapshot
    ctrl.reset_lab_tensor = lambda: _load_lab_case("Combined 3D stress")

    # ------------------------------------------------------------------ initial state
    default_template = next(iter(TEMPLATES))
    recon_tools = tool_availability()
    state.update({
        "theme": "dark",
        "spec_text": TEMPLATES[default_template],
        "template": None,
        "templates": list(TEMPLATES),
        "status": "Edit the model parameters or load a CAD file, then press Solve.",
        "status_type": "info",
        "active_tab": "model",
        "active_page": "workspace",
        "main_drawer": True,
        "results_right": True,
        "analysis_dock_open": False,
        "analysis_dock_tab": "response",
        "analysis_result_warning": "",
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
        # Real-Life Object CV Mode
        "cv_source_mode": "preset",
        "cv_preset_items": [
            {"title": "Structural L-Bracket (Real-Life Photo)", "value": "cad_models/cv_samples/bracket_photo.png"},
            {"title": "Tensile Plate with Hole (Real-Life Photo)", "value": "cad_models/cv_samples/plate_hole_photo.png"},
            {"title": "Rectangular Beam / Bar (Real-Life Photo)", "value": "cad_models/cv_samples/beam_photo.png"},
            {"title": "Structural I-Beam Profile (Real-Life Photo)", "value": "cad_models/cv_samples/ibeam_photo.png"},
            {"title": "Stepped Motor Shaft (Real-Life Photo)", "value": "cad_models/cv_samples/shaft_photo.png"},
            {"title": "Hollow Tube / Pipe (Real-Life Photo)", "value": "cad_models/cv_samples/pipe_photo.png"},
        ],
        "cv_preset_path": "cad_models/cv_samples/bracket_photo.png",
        "cv_custom_path": "",
        "cv_image_b64": "",
        "cv_ref_scale": 0.30,
        "cv_status": "Ready to snap a camera photo or select a sample image.",
        "cv_detected_type": "",
        "cv_friendly_name": "",
        "cv_confidence_pct": 0,
        "cv_features_text": "",
        "cv_annotated_preview": "",
        "cv_step_path": "cad_models/cv_detected_object.step",
        "cv_param_leg_x": 0.20, "cv_param_leg_y": 0.20,
        "cv_param_width": 0.06, "cv_param_thickness": 0.015,
        "cv_param_hole_radius": 0.008,
        "cv_param_length": 0.40, "cv_param_height": 0.15,
        "cv_param_hole_diameter": 0.06,
        "cv_param_flange_width": 0.12, "cv_param_flange_thickness": 0.02,
        "cv_param_web_thickness": 0.015,
        "cv_param_diameter_main": 0.08, "cv_param_diameter_step": 0.05,
        "cv_param_step_length": 0.20,
        "cv_param_outer_diameter": 0.10, "cv_param_inner_diameter": 0.07,
        # Multi-view NeRF / Gaussian-splatting reconstruction studio
        "cv_mv_images": [],
        "cv_mv_file_names": [],
        "cv_mv_name": "mechanical_part",
        "cv_mv_method": "splatfacto",
        "cv_mv_method_items": [
            {"title": "3D Gaussian Splatting · Splatfacto", "value": "splatfacto"},
            {"title": "NeRF · Nerfacto", "value": "nerfacto"},
            {"title": "2D Gaussian Splatting · external trainer", "value": "2dgs"},
            {"title": "3D Gaussian Splatting · Graphdeco trainer", "value": "graphdeco"},
        ],
        "cv_mv_session_dir": "",
        "cv_mv_images_dir": "",
        "cv_mv_ns_data_dir": "",
        "cv_mv_image_count": 0,
        "cv_mv_summary": "",
        "cv_mv_warnings": "",
        "cv_mv_commands": "Prepare a multi-view capture first.",
        "cv_mv_status": "Select 20–60 overlapping photos of the same object.",
        "cv_mv_tools_status": "COLMAP: {colmap} · ns-process-data: {process} · ns-train: {train} · FFmpeg: {ffmpeg}".format(
            colmap="available" if recon_tools["colmap"] else "not found",
            process="available" if recon_tools["ns_process_data"] else "not found",
            train="available" if recon_tools["ns_train"] else "not found",
            ffmpeg="available" if recon_tools["ffmpeg"] else "not found",
        ),
        "cv_mv_preprocess_status": "Camera poses have not been estimated yet.",
        "cv_mv_preprocess_log": "",
        "cv_mv_preprocess_ready": False,
        "cv_mv_running": False,
        "cv_mv_colmap_ready": False,
        "cv_mv_colmap_running": False,
        "cv_mv_colmap_status": "Standard COLMAP dataset has not been built yet.",
        "cv_mv_colmap_log": "",
        "cv_mv_colmap_workspace": "",
        "cv_mv_colmap_model": "",
        "cv_mv_colmap_points": "",
        # Universal single-image → 2.5D preview → optional silhouette STEP workflow
        "cv3d_image_b64": "",
        "cv3d_image_name": "",
        "cv3d_geometry_mode": "depth_relief",
        "cv3d_geometry_mode_items": [
            {"title": "AI relative-depth relief (Depth Anything V2)", "value": "depth_relief"},
            {"title": "Silhouette extrusion (no depth model)", "value": "silhouette"},
        ],
        "cv3d_segmentation_mode": "auto",
        "cv3d_segmentation_mode_items": [
            {"title": "Automatic foreground mask", "value": "auto"},
            {"title": "Use entire image frame", "value": "full_frame"},
        ],
        "cv3d_object_width_m": 0.20,
        "cv3d_body_thickness_m": 0.025,
        "cv3d_relief_depth_m": 0.045,
        "cv3d_mesh_resolution": 192,
        "cv3d_running": False,
        "cv3d_preview_path": "",
        "cv3d_mask_path": "",
        "cv3d_depth_path": "",
        "cv3d_step_path": "",
        "cv3d_preview_active": False,
        "cv3d_status": "Upload one image to build a colored 3D preview.",
        "cv3d_warning_text": "",
        "cv_gaussian_ply_path": "",
        "cv_gaussian_centers_path": "",
        "cv_gaussian_status": "Import a trained 3DGS .ply to inspect its Gaussian centers.",
        "cv_gaussian_known_extent_m": 0.30,
        "cv_gaussian_scale_axis": "x",
        "cv_gaussian_scale_axis_items": [
            {"title": "X extent", "value": "x"},
            {"title": "Y extent", "value": "y"},
            {"title": "Z extent", "value": "z"},
        ],
        "cv_gaussian_calibrated_path": "",
        "cv_gaussian_scale_status": "",
        # Material
        "mat_preset": "Steel",
        "mat_preset_items": list(MATERIAL_PRESETS),
        "mat_E": "210e9", "mat_nu": "0.3", "mat_rho": "7850",
        # Supports & Loads
        "supports_list": [{"face": "-x", "comps": "xyz"}],
        "loads_list": [{"mode": "force", "target": "face", "face": "+x",
                        "vec": "0, -1e5, 0", "val": "0", "pt": [0, 0, 0]}],
        "live_load_items": [],
        "live_load_selection": "",
        "editing_load_index": -1,
        "live_load_mode": "force",
        "live_load_x": 0.0, "live_load_y": -1e5, "live_load_z": 0.0,
        "live_pressure": 1e6,
        "live_load_multiplier": 1.0,
        "live_load_value": "",
        "live_load_status": "Solve a supported model, then select an applied load to enable live response.",
        "live_load_ready": False,
        "live_response_status": "Live response cache will be prepared after a supported model is solved.",
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
        # Learning Lab — an idealised point-stress experiment, separate from FEM results.
        "lab_case": "Combined 3D stress",
        "lab_case_items": LAB_CASE_ITEMS,
        "lab_sigma_xx": 80.0, "lab_sigma_yy": 20.0, "lab_sigma_zz": 10.0,
        "lab_tau_xy": 30.0, "lab_tau_yz": 12.0, "lab_tau_zx": -8.0,
        "lab_azimuth": 35.0, "lab_elevation": 20.0,
        "lab_point_x": 10.0, "lab_point_y": 0.0, "lab_point_z": 0.0,
        "lab_plane_preset": "Custom", "lab_plane_custom": "0, 0, 1",
        "lab_plane_items": [{"title": name, "value": name} for name in
                            ["Custom", "+X", "-X", "+Y", "-Y", "+Z", "-Z",
                             "Principal 1", "Principal 2", "Principal 3"]],
        "lab_node_index": "—", "lab_point_coords": "—", "lab_status": "No solved result. Solve a model in Analysis Workspace first.",
        "lab_model_summary": "No solved model is available. Solve a model in Analysis Workspace.",
        "lab_strain_matrix_html": "—", "lab_displacement": "—", "lab_shear_vector": "—",
        "lab_fea_principal": "—", "lab_fea_vm": "—", "lab_fea_tresca": "—", "lab_has_result": False,
        "lab_tool_mode": "beam", "lab_tool_items": LAB_TOOL_ITEMS,
        "lab_q1": 500.0, "lab_q2": 40.0, "lab_q3": 80.0, "lab_q4": 180.0,
        "lab_q1_label": "Bending moment M (N·m)",
        "lab_q2_label": "Section width b (mm)",
        "lab_q3_label": "Section height h (mm)",
        "lab_q4_label": "Allowable stress (MPa)",
        "lab_snapshot_saved": False,
        "lab_snapshot_label": "No case saved",
        "lab_snapshot_sigma1": 0.0,
        "lab_snapshot_current": "—", "lab_snapshot_delta": "—",
    })
    _refresh_learning_lab()
    _refresh_mechanics_tool()

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

    @state.change("lab_case")
    def _lab_case_changed(lab_case, **_):
        _load_lab_case(lab_case)

    @state.change("lab_sigma_xx", "lab_sigma_yy", "lab_sigma_zz",
                  "lab_tau_xy", "lab_tau_yz", "lab_tau_zx",
                  "lab_azimuth", "lab_elevation")
    def _lab_tensor_changed(**_):
        _refresh_learning_lab()

    @state.change("lab_point_x", "lab_point_y", "lab_point_z")
    def _lab_point_changed(**_):
        _lab_stress_from_result()
        if bool(getattr(state, "analysis_dock_open", False)):
            update_analysis_overlay()

    @state.change("lab_plane_preset", "lab_plane_custom")
    def _lab_plane_inputs_changed(**_):
        _lab_plane_changed()

    @state.change("lab_tool_mode")
    def _lab_tool_mode_changed(lab_tool_mode, **_):
        _load_mechanics_tool(lab_tool_mode)

    @state.change("lab_q1", "lab_q2", "lab_q3", "lab_q4")
    def _lab_tool_inputs_changed(**_):
        _refresh_mechanics_tool()

    @state.change("analysis_dock_open")
    def _analysis_dock_visibility_changed(analysis_dock_open, **_):
        """Toggle the analysis UI without touching the VTK scene or camera.

        Opening a UI panel must be a presentation-only state transition. The prior
        implementation refreshed PyVista actors and rendered the plotter here; on
        some Trame/VTK client combinations that render re-measured the viewport and
        caused the camera to jump while the dock was being shown.
        """
        if not analysis_dock_open:
            return

        if S.res is None or S.grid is None:
            state.analysis_result_warning = "No solved result yet. Solve the current model first."
            state.lab_has_result = False
            state.lab_status = "No solved result. Solve a model in Analysis Workspace first."
            return

        if S.solved_spec_text is not None and str(state.spec_text) != S.solved_spec_text:
            state.analysis_result_warning = (
                "The current setup has unsolved edits. Values shown are from the last solved "
                "model; run Solve before interpreting them or changing loads."
            )
        else:
            state.analysis_result_warning = ""

        # Always resample from the latest published Results object. Do this without
        # creating/removing VTK actors or calling view.update().
        _lab_stress_from_result()

    @state.change("live_load_selection")
    def _live_load_selection_changed(live_load_selection, **_):
        if S.live_response_suspended or str(live_load_selection).strip() == "":
            return
        try:
            idx = int(live_load_selection)
        except (TypeError, ValueError):
            return
        if idx == int(getattr(state, "editing_load_index", -1)):
            return
        _start_live_load_edit(idx)

    @state.change("live_load_x", "live_load_y", "live_load_z", "live_pressure",
                  "live_load_multiplier")
    def _live_load_values_changed(**_):
        _live_load_inputs_changed()

    # ================================================================ LAYOUT
    with SinglePageWithDrawerLayout(server) as layout:
        layout.root.theme = ("theme",)
        layout.title.set_text("MoS · Solid Mechanics Studio")
        layout.drawer.width = 420

        # Keep the original toolbar's labels, order and visual treatment.  The
        # Learning Lab action now opens the integrated analysis dock over the
        # existing model instead of navigating to a separate full-page Lab.
        with layout.toolbar:
            with html.Div(classes="d-flex align-center ga-2"):
                html.Div("MoS", classes="text-h6 font-weight-black text-primary")
                html.Div("SOLID MECHANICS STUDIO", classes="text-caption font-weight-bold text-medium-emphasis")
            v3.VSpacer()
            v3.VBtn(
                "Analysis workspace", prepend_icon="mdi-cube-scan",
                color="primary", variant="{{ active_page === 'workspace' ? 'flat' : 'text' }}",
                density="comfortable", size="small",
                click="active_page = 'workspace'; analysis_dock_open = false; main_drawer = true",
            )
            v3.VBtn(
                "Learning Lab", prepend_icon="mdi-school-outline",
                color="primary", variant="{{ analysis_dock_open ? 'flat' : 'text' }}",
                density="comfortable", size="small",
                click="analysis_dock_open = true",
            )
            v3.VChip(
                "{{ active_page === 'workspace' ? 'LIVE FEM' : 'CONCEPT LAB' }}",
                color="{{ active_page === 'workspace' ? 'success' : 'deep-purple' }}",
                variant="tonal", size="small", classes="ml-1 mr-2",
            )
            v3.VBtn(
                icon="mdi-theme-light-dark", variant="text",
                density="comfortable", classes="mr-1",
                click="theme = theme === 'dark' ? 'light' : 'dark'",
                title="Toggle theme",
            )

        # Drawer: model creation and FEA controls remain grouped in the analysis workspace.
        with layout.drawer:
            with v3.VContainer(fluid=True, classes="pa-3"):
                with html.Div(classes="mb-3 px-1"):
                    html.Div("ANALYSIS WORKSPACE", classes="text-overline text-primary font-weight-bold")
                    html.Div("Build a model. Apply loads. Read the response.",
                             classes="text-caption text-medium-emphasis")

                # Navigation tabs
                with v3.VTabs(v_model=("active_tab",), grow=True,
                              density="compact", classes="mb-2"):
                    v3.VTab("Model", value="model")
                    v3.VTab("Loads & Supports", value="loads")
                    v3.VTab("Results", value="results")

                with v3.VCard(variant="tonal", color="primary", classes="pa-2 mb-3 rounded-lg"):
                    with html.Div(classes="d-flex align-center ga-2"):
                        v3.VIcon(icon="mdi-cube-scan", size="small")
                        html.Div("3D CAD → mesh → linear-elastic FEA",
                                 classes="text-caption font-weight-medium")
                    html.Div("STEP import · image-to-CAD · live loads · stress contours",
                             classes="text-caption text-medium-emphasis mt-1")

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
                                    v3.VBtn("Real-Life CV Mode", value="cv",
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

                                # Real-Life CV Object Detector & Parametric CAD Mode
                                with html.Div(v_show="geom_mode === 'cv'"):
                                    html.Div(
                                        "Detect → segment → parameterize. Use a clear, well-lit photo with the whole part visible against a simple background.",
                                        classes="text-caption text-medium-emphasis mb-2",
                                    )

                                    # Input source toggle: Presets, Upload file, Webcam snapshot
                                    with v3.VBtnToggle(
                                        v_model=("cv_source_mode",), mandatory=True,
                                        density="compact", divided=True, color="primary",
                                        classes="mb-2 w-100",
                                    ):
                                        v3.VBtn("Sample Photo", value="preset", size="x-small", classes="flex-grow-1")
                                        v3.VBtn("File Path", value="upload", size="x-small", classes="flex-grow-1")
                                        v3.VBtn("Webcam / Camera", value="webcam", size="x-small", classes="flex-grow-1")

                                    with html.Div(v_show="cv_source_mode === 'preset'"):
                                        v3.VSelect(
                                            v_model=("cv_preset_path",), items=("cv_preset_items",),
                                            item_title="title", item_value="value",
                                            label="Real-Life Object Photo Preset",
                                            density="compact", hide_details=True, classes="mb-2",
                                        )

                                    with html.Div(v_show="cv_source_mode === 'upload'"):
                                        v3.VTextField(
                                            v_model=("cv_custom_path",),
                                            label="Image File Path (PNG / JPG)",
                                            density="compact", hide_details=True, classes="mb-2",
                                        )

                                    with html.Div(v_show="cv_source_mode === 'webcam'"):
                                        html.Input(
                                            id="mos_cam_input", type="file", accept="image/*", capture="environment",
                                            style="display:none;",
                                            onchange="const f=event.target.files[0]; if (f) { const r=new FileReader(); r.onload=(e)=>{ window.trame.state.set('cv_image_b64', e.target.result); }; r.readAsDataURL(f); }",
                                        )
                                        v3.VBtn("Snap Photo with Camera", icon="mdi-camera", color="primary", size="small",
                                                block=True, classes="mb-2", click="document.getElementById('mos_cam_input').click()")
                                        html.Div("Camera photo ready for analysis", classes="text-caption text-success mb-1",
                                                 v_show="cv_image_b64.length > 0")

                                    # Dimension reference calibration scale
                                    with html.Div(classes="d-flex align-center ga-1 mb-2"):
                                        v3.VTextField(
                                            v_model=("cv_ref_scale",), label="Assumed largest dimension (m)",
                                            type="number", step=0.05, density="compact", hide_details=True,
                                            classes="flex-grow-1",
                                        )
                                        v3.VBtn("Run CV Model", color="secondary", variant="tonal", size="small",
                                                click=ctrl.run_cv_analysis)

                                    # CV Detection results preview
                                    with html.Div(v_show="cv_detected_type !== ''", classes="mt-2"):
                                        with html.Div(classes="d-flex align-center justify-space-between mb-1"):
                                            v3.VChip("{{ cv_friendly_name }}", color="success", size="small", classes="font-weight-bold")
                                            v3.VChip("{{ cv_confidence_pct }}% confidence", color="info", size="x-small", variant="outlined")

                                        html.Img(
                                            src=("cv_annotated_preview",),
                                            style="width:100%; max-height:240px; object-fit:contain; border-radius:12px; background:#111; border:1px solid rgba(148,163,184,.22);",
                                            classes="mb-2",
                                        )
                                        html.Div(
                                            "{{ cv_features_text }}",
                                            style="white-space:pre-line; font-size:11px;",
                                            classes="text-medium-emphasis mb-2 pa-2 bg-surface-variant rounded",
                                        )

                                        # Dynamic Parametric Inputs
                                        html.Div("Extracted CAD Parameters (meters):", classes="text-caption font-weight-bold mb-1")

                                        # Bracket parameters
                                        with html.Div(v_show="cv_detected_type === 'bracket'", classes="d-flex flex-column ga-1 mb-2"):
                                            with html.Div(classes="d-flex ga-1"):
                                                v3.VTextField(v_model=("cv_param_leg_x",), label="Leg X (m)", type="number", step=0.01, density="compact", hide_details=True)
                                                v3.VTextField(v_model=("cv_param_leg_y",), label="Leg Y (m)", type="number", step=0.01, density="compact", hide_details=True)
                                            with html.Div(classes="d-flex ga-1"):
                                                v3.VTextField(v_model=("cv_param_width",), label="Width W (m)", type="number", step=0.01, density="compact", hide_details=True)
                                                v3.VTextField(v_model=("cv_param_thickness",), label="Thickness T (m)", type="number", step=0.005, density="compact", hide_details=True)
                                                v3.VTextField(v_model=("cv_param_hole_radius",), label="Hole Radius (m)", type="number", step=0.002, density="compact", hide_details=True)

                                        # Plate hole parameters
                                        with html.Div(v_show="cv_detected_type === 'plate_hole'", classes="d-flex flex-column ga-1 mb-2"):
                                            with html.Div(classes="d-flex ga-1"):
                                                v3.VTextField(v_model=("cv_param_length",), label="Length (m)", type="number", step=0.02, density="compact", hide_details=True)
                                                v3.VTextField(v_model=("cv_param_width",), label="Width (m)", type="number", step=0.01, density="compact", hide_details=True)
                                            with html.Div(classes="d-flex ga-1"):
                                                v3.VTextField(v_model=("cv_param_thickness",), label="Thickness (m)", type="number", step=0.005, density="compact", hide_details=True)
                                                v3.VTextField(v_model=("cv_param_hole_diameter",), label="Hole Diameter (m)", type="number", step=0.005, density="compact", hide_details=True)

                                        # Beam parameters
                                        with html.Div(v_show="cv_detected_type === 'beam'", classes="d-flex ga-1 mb-2"):
                                            v3.VTextField(v_model=("cv_param_length",), label="Length (m)", type="number", step=0.05, density="compact", hide_details=True)
                                            v3.VTextField(v_model=("cv_param_height",), label="Height (m)", type="number", step=0.01, density="compact", hide_details=True)
                                            v3.VTextField(v_model=("cv_param_width",), label="Width (m)", type="number", step=0.01, density="compact", hide_details=True)

                                        # I-Beam parameters
                                        with html.Div(v_show="cv_detected_type === 'ibeam'", classes="d-flex flex-column ga-1 mb-2"):
                                            with html.Div(classes="d-flex ga-1"):
                                                v3.VTextField(v_model=("cv_param_length",), label="Length (m)", type="number", step=0.05, density="compact", hide_details=True)
                                                v3.VTextField(v_model=("cv_param_height",), label="Height (m)", type="number", step=0.01, density="compact", hide_details=True)
                                            with html.Div(classes="d-flex ga-1"):
                                                v3.VTextField(v_model=("cv_param_flange_width",), label="Flange Width (m)", type="number", step=0.01, density="compact", hide_details=True)
                                                v3.VTextField(v_model=("cv_param_flange_thickness",), label="Flange Thick (m)", type="number", step=0.005, density="compact", hide_details=True)
                                                v3.VTextField(v_model=("cv_param_web_thickness",), label="Web Thick (m)", type="number", step=0.005, density="compact", hide_details=True)

                                        # Shaft parameters
                                        with html.Div(v_show="cv_detected_type === 'shaft'", classes="d-flex flex-column ga-1 mb-2"):
                                            with html.Div(classes="d-flex ga-1"):
                                                v3.VTextField(v_model=("cv_param_length",), label="Length (m)", type="number", step=0.05, density="compact", hide_details=True)
                                                v3.VTextField(v_model=("cv_param_diameter_main",), label="Main Diam (m)", type="number", step=0.01, density="compact", hide_details=True)
                                            with html.Div(classes="d-flex ga-1"):
                                                v3.VTextField(v_model=("cv_param_diameter_step",), label="Step Diam (m)", type="number", step=0.01, density="compact", hide_details=True)
                                                v3.VTextField(v_model=("cv_param_step_length",), label="Step Len (m)", type="number", step=0.01, density="compact", hide_details=True)

                                        # Pipe parameters
                                        with html.Div(v_show="cv_detected_type === 'pipe'", classes="d-flex flex-column ga-1 mb-2"):
                                            with html.Div(classes="d-flex ga-1"):
                                                v3.VTextField(v_model=("cv_param_length",), label="Length (m)", type="number", step=0.05, density="compact", hide_details=True)
                                                v3.VTextField(v_model=("cv_param_outer_diameter",), label="Outer Diam (m)", type="number", step=0.01, density="compact", hide_details=True)
                                                v3.VTextField(v_model=("cv_param_inner_diameter",), label="Inner Diam (m)", type="number", step=0.01, density="compact", hide_details=True)

                                        # Mesh size & Solid conversion button
                                        with html.Div(classes="d-flex align-center ga-2 mb-2"):
                                            html.Span("CAD Mesh size:", classes="text-caption")
                                            v3.VSlider(
                                                v_model=("cad_mesh_size",),
                                                min=0.02, max=0.5, step=0.01,
                                                density="compact", hide_details=True,
                                                classes="flex-grow-1",
                                            )
                                            html.Span("{{ cad_mesh_size }} m",
                                                      classes="text-caption font-weight-bold",
                                                      style="min-width:48px; text-align:right")

                                        v3.VBtn("Generate 3D CAD Solid & Solve FEA", color="primary", size="small", block=True,
                                                prepend_icon="mdi-cube-outline", click=ctrl.generate_cv_cad_and_load, classes="mt-1")

                                    html.Div("{{ cv_status }}", classes="text-caption text-medium-emphasis mt-1")

                                    # Multi-view reconstruction: camera poses -> NeRF / Gaussian splats
                                    with v3.VExpansionPanels(variant="accordion", density="compact", multiple=True, classes="mt-3"):
                                        with v3.VExpansionPanel():
                                            with v3.VExpansionPanelTitle():
                                                html.Span("Multi-view 3D Reconstruction · NeRF / Gaussian Splatting",
                                                          classes="text-caption font-weight-bold")
                                            with v3.VExpansionPanelText():
                                                html.Div(
                                                    "Capture the same rigid object from overlapping viewpoints. "
                                                    "This creates a COLMAP/Nerfstudio-ready dataset; it does not replace the single-photo parametric CAD tool above.",
                                                    classes="text-caption text-medium-emphasis mb-2",
                                                )
                                                html.Input(
                                                    id="mos_multiview_input", type="file", accept="image/*", multiple=True,
                                                    style="display:none;",
                                                    onchange="""const files = Array.from(event.target.files || []);
if (!files.length) return;
Promise.all(files.map(async (file) => {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, 1600 / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(bitmap.width * scale));
  canvas.height = Math.max(1, Math.round(bitmap.height * scale));
  canvas.getContext('2d').drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close && bitmap.close();
  return { name: file.name, data: canvas.toDataURL('image/jpeg', 0.88) };
})).then((items) => {
  window.trame.state.set('cv_mv_images', items);
  window.trame.state.set('cv_mv_file_names', items.map((item) => item.name));
  window.trame.state.set('cv_mv_status', items.length + ' photos ready. Build the capture dataset to validate and save them.');
}).catch((error) => window.trame.state.set('cv_mv_status', 'Image upload error: ' + error.message));
event.target.value = '';""",
                                                )
                                                v3.VBtn(
                                                    "Choose Multi-view Photos", icon="mdi-image-multiple", color="primary",
                                                    size="small", block=True, classes="mb-1",
                                                    click="document.getElementById('mos_multiview_input').click()",
                                                )
                                                html.Div(
                                                    "{{ cv_mv_file_names.length }} photos selected · {{ cv_mv_file_names.join(', ') }}",
                                                    classes="text-caption text-medium-emphasis mb-2",
                                                    style="overflow-wrap:anywhere;",
                                                )
                                                with html.Div(classes="d-flex ga-2 align-center mb-2"):
                                                    v3.VTextField(
                                                        v_model=("cv_mv_name",), label="Capture name",
                                                        density="compact", hide_details=True,
                                                    )
                                                    v3.VBtn(
                                                        "Build Dataset", color="secondary", size="small",
                                                        click=ctrl.prepare_cv_multiview_dataset,
                                                    )
                                                html.Div("{{ cv_mv_summary }}", classes="text-caption font-weight-medium mb-1")
                                                html.Div("Reconstruction tools · {{ cv_mv_tools_status }}",
                                                         classes="text-caption text-medium-emphasis mb-1")
                                                html.Div(
                                                    "{{ cv_mv_warnings }}", v_show="cv_mv_warnings !== ''",
                                                    classes="text-caption text-warning mb-2", style="white-space:pre-line;",
                                                )
                                                html.Div("{{ cv_mv_status }}", classes="text-caption text-medium-emphasis mb-2",
                                                         style="overflow-wrap:anywhere; white-space:pre-wrap;")
                                                v3.VSelect(
                                                    v_model=("cv_mv_method",), items=("cv_mv_method_items",),
                                                    item_title="title", item_value="value",
                                                    label="Reconstruction representation", density="compact",
                                                    hide_details=True, classes="mb-2",
                                                    update_model_value=set_cv_multiview_method,
                                                )
                                                with html.Div(v_show="cv_mv_method === 'splatfacto' || cv_mv_method === 'nerfacto'"):
                                                    v3.VBtn(
                                                        "Prepare Nerfstudio Poses · COLMAP", color="primary", variant="tonal",
                                                        size="small", block=True, disabled=("cv_mv_running",),
                                                        loading=("cv_mv_running",),
                                                        click=ctrl.run_cv_multiview_preprocess,
                                                    )
                                                html.Div(
                                                    "{{ cv_mv_preprocess_status }}", classes="text-caption text-medium-emphasis mt-2",
                                                    style="white-space:pre-wrap; overflow-wrap:anywhere;",
                                                )
                                                v3.VBtn(
                                                    "Build Standard COLMAP Dataset · 2DGS / Graphdeco",
                                                    color="secondary", variant="outlined", size="small", block=True,
                                                    disabled=("cv_mv_colmap_running",),
                                                    loading=("cv_mv_colmap_running",),
                                                    click=ctrl.run_cv_colmap_sparse, classes="mt-2",
                                                )
                                                html.Div(
                                                    "{{ cv_mv_colmap_status }}", classes="text-caption text-medium-emphasis mt-2",
                                                    style="white-space:pre-wrap; overflow-wrap:anywhere;",
                                                )
                                                html.Div(
                                                    "{{ cv_mv_colmap_log }}", v_show="cv_mv_colmap_log !== ''",
                                                    classes="text-caption text-medium-emphasis mt-1 pa-2 bg-surface-variant rounded",
                                                    style="max-height:120px; overflow:auto; white-space:pre-wrap;",
                                                )
                                                html.Div(
                                                    "{{ cv_mv_preprocess_log }}", v_show="cv_mv_preprocess_log !== ''",
                                                    classes="text-caption text-medium-emphasis mt-1 pa-2 bg-surface-variant rounded",
                                                    style="max-height:120px; overflow:auto; white-space:pre-wrap;",
                                                )
                                                html.Div("Training commands", classes="text-caption font-weight-bold mt-3 mb-1")
                                                html.Pre(
                                                    "{{ cv_mv_commands }}",
                                                    style="font-size:10px; white-space:pre-wrap; overflow-wrap:anywhere; padding:8px; border-radius:6px; background:rgba(127,127,127,.12);",
                                                )
                                                html.Div(
                                                    "Capture tip: use 20–60 sharp images, walk around the stationary part at several heights, "
                                                    "keep adjacent views overlapping, and avoid changing zoom or lighting. NeRF/Gaussian scene scale is arbitrary until calibrated.",
                                                    classes="text-caption text-medium-emphasis mt-2",
                                                )

                                        with v3.VExpansionPanel():
                                            with v3.VExpansionPanelTitle():
                                                html.Span("Gaussian PLY → CAD preparation point cloud",
                                                          classes="text-caption font-weight-bold")
                                            with v3.VExpansionPanelText():
                                                html.Div(
                                                    "Import a trained 3D Gaussian Splatting .ply. The tool extracts Gaussian centers and colors into a standard point cloud for inspection and later surface fitting.",
                                                    classes="text-caption text-medium-emphasis mb-2",
                                                )
                                                v3.VTextField(
                                                    v_model=("cv_gaussian_ply_path",),
                                                    label="Gaussian model .ply path",
                                                    density="compact", hide_details=True, classes="mb-2",
                                                )
                                                v3.VBtn(
                                                    "Extract Gaussian Centers", color="secondary", variant="tonal",
                                                    size="small", block=True, click=ctrl.inspect_cv_gaussian_ply,
                                                )
                                                html.Div(
                                                    "{{ cv_gaussian_status }}", classes="text-caption text-medium-emphasis mt-2",
                                                    style="white-space:pre-wrap; overflow-wrap:anywhere;",
                                                )
                                                with html.Div(v_show="cv_gaussian_centers_path !== ''", classes="mt-2"):
                                                    html.Div("Metric scale calibration", classes="text-caption font-weight-bold mb-1")
                                                    with html.Div(classes="d-flex ga-2 align-center"):
                                                        v3.VSelect(
                                                            v_model=("cv_gaussian_scale_axis",),
                                                            items=("cv_gaussian_scale_axis_items",),
                                                            item_title="title", item_value="value",
                                                            label="Known dimension axis", density="compact",
                                                            hide_details=True,
                                                        )
                                                        v3.VTextField(
                                                            v_model=("cv_gaussian_known_extent_m",),
                                                            label="Known extent (m)", type="number", step=0.01,
                                                            density="compact", hide_details=True,
                                                        )
                                                    v3.VBtn(
                                                        "Calibrate Point Cloud to Metres", color="secondary",
                                                        size="small", block=True, variant="tonal",
                                                        click=ctrl.calibrate_cv_gaussian_cloud,
                                                        classes="mt-1",
                                                    )
                                                    html.Div(
                                                        "{{ cv_gaussian_scale_status }}", classes="text-caption text-medium-emphasis mt-2",
                                                        style="white-space:pre-wrap; overflow-wrap:anywhere;",
                                                    )
                                                html.Div(
                                                    "Engineering note: Gaussian centers are not a watertight surface or a parametric solid. "
                                                    "A surface-reconstruction and dimension-calibration stage is still required before reliable STEP/FEM use.",
                                                    classes="text-caption text-warning mt-2",
                                                )

                                        with v3.VExpansionPanel():
                                            with v3.VExpansionPanelTitle():
                                                html.Span("Universal Image → 3D → CAD Viewer",
                                                          classes="text-caption font-weight-bold")
                                            with v3.VExpansionPanelText():
                                                html.Div(
                                                    "Upload one ordinary image. The app estimates a foreground mask and builds a colored 2.5D surface for the CAD viewport. "
                                                    "For stronger geometry, use the multi-view capture workflow above.",
                                                    classes="text-caption text-medium-emphasis mb-2",
                                                )
                                                html.Input(
                                                    id="mos_cv3d_image_input", type="file", accept="image/png,image/jpeg,image/webp,image/bmp",
                                                    style="display:none;",
                                                    onchange="""const file = event.target.files && event.target.files[0];
if (!file) return;
if (file.size > 15 * 1024 * 1024) {
  window.trame.state.set('cv3d_status', 'Image is larger than 15 MB. Resize it and try again.');
  event.target.value = '';
  return;
}
const reader = new FileReader();
reader.onload = (e) => {
  window.trame.state.set('cv3d_image_b64', e.target.result);
  window.trame.state.set('cv3d_image_name', file.name);
  window.trame.state.set('cv3d_status', 'Image ready: ' + file.name + '. Choose reconstruction settings and build the preview.');
};
reader.onerror = () => window.trame.state.set('cv3d_status', 'Could not read the selected image.');
reader.readAsDataURL(file);
event.target.value = '';""",
                                                )
                                                v3.VBtn(
                                                    "Choose Image · JPG / PNG / WEBP", icon="mdi-image-plus",
                                                    color="primary", size="small", block=True,
                                                    click="document.getElementById('mos_cv3d_image_input').click()",
                                                )
                                                html.Div(
                                                    "{{ cv3d_image_name }}", v_show="cv3d_image_b64 !== ''",
                                                    classes="text-caption text-medium-emphasis mt-1",
                                                )
                                                html.Img(
                                                    src=("cv3d_image_b64",), v_show="cv3d_image_b64 !== ''",
                                                    style="width:100%; max-height:140px; object-fit:contain; background:#111; border-radius:6px; margin-top:6px;",
                                                )
                                                v3.VSelect(
                                                    v_model=("cv3d_geometry_mode",), items=("cv3d_geometry_mode_items",),
                                                    item_title="title", item_value="value",
                                                    label="Single-image geometry mode", density="compact",
                                                    hide_details=True, classes="mt-2 mb-2",
                                                )
                                                v3.VSelect(
                                                    v_model=("cv3d_segmentation_mode",), items=("cv3d_segmentation_mode_items",),
                                                    item_title="title", item_value="value",
                                                    label="Foreground extraction", density="compact",
                                                    hide_details=True, classes="mb-2",
                                                )
                                                with html.Div(classes="d-flex ga-2"):
                                                    v3.VTextField(
                                                        v_model=("cv3d_object_width_m",), label="Approx. width (m)",
                                                        type="number", min=0.001, step=0.01,
                                                        density="compact", hide_details=True,
                                                    )
                                                    v3.VTextField(
                                                        v_model=("cv3d_body_thickness_m",), label="Extrusion depth (m)",
                                                        type="number", min=0.001, step=0.005,
                                                        density="compact", hide_details=True,
                                                    )
                                                v3.VTextField(
                                                    v_model=("cv3d_relief_depth_m",), label="Depth-relief amplitude (m)",
                                                    type="number", min=0, step=0.005,
                                                    density="compact", hide_details=True, classes="mt-1",
                                                )
                                                html.Div("Preview mesh resolution", classes="text-caption font-weight-medium mt-2")
                                                with html.Div(classes="d-flex align-center ga-2"):
                                                    v3.VSlider(
                                                        v_model=("cv3d_mesh_resolution",), min=96, max=320, step=32,
                                                        density="compact", hide_details=True, classes="flex-grow-1",
                                                    )
                                                    html.Span("{{ cv3d_mesh_resolution }} px", classes="text-caption")
                                                v3.VBtn(
                                                    "Reconstruct 3D Preview", color="primary", size="small", block=True,
                                                    prepend_icon="mdi-cube-scan", click=ctrl.run_cv_universal_reconstruction,
                                                    disabled=("cv3d_image_b64 === '' || cv3d_running",),
                                                    loading=("cv3d_running",), classes="mt-2",
                                                )
                                                html.Pre(
                                                    "{{ cv3d_status }}", classes="text-caption text-medium-emphasis mt-2",
                                                    style="white-space:pre-wrap; overflow-wrap:anywhere;",
                                                )
                                                html.Pre(
                                                    "{{ cv3d_warning_text }}", v_show="cv3d_warning_text !== ''",
                                                    classes="text-caption text-warning mt-1",
                                                    style="white-space:pre-wrap; overflow-wrap:anywhere;",
                                                )
                                                with html.Div(v_show="cv3d_preview_path !== ''", classes="mt-2"):
                                                    v3.VBtn(
                                                        "Show 3D Preview in CAD Viewer", color="secondary", variant="tonal",
                                                        size="small", block=True, click=ctrl.show_cv3d_preview,
                                                    )
                                                    v3.VBtn(
                                                        "Clear Preview · Return to FEM Model", color="secondary", variant="text",
                                                        size="small", block=True, click=ctrl.clear_cv3d_preview,
                                                    )
                                                with html.Div(v_show="cv3d_step_path !== ''", classes="mt-1"):
                                                    v3.VBtn(
                                                        "Load Silhouette STEP into CAD / FEM", color="success",
                                                        size="small", block=True, prepend_icon="mdi-cube-outline",
                                                        click=ctrl.load_cv3d_step_to_fem,
                                                    )
                                                html.Div(
                                                    "Important: the viewer preview uses an estimated front-surface depth and a simplified back. "
                                                    "The STEP export is a constant-thickness silhouette extrusion, not the same depth-relief mesh; inner holes may be omitted. "
                                                    "Single-image depth is relative and must not be treated as measured geometry.",
                                                    classes="text-caption text-warning mt-2",
                                                )

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
                            html.Div("Define where the model is restrained", classes="text-caption text-medium-emphasis mb-2")
                            with html.Div(classes="d-flex ga-2 mb-2"):
                                v3.VSelect(
                                    v_model=("support_face",),
                                    items=("face_options",),
                                    item_title="title", item_value="value",
                                    label="Constrained surface",
                                    density="compact", hide_details=True,
                                    classes="flex-grow-1",
                                )
                                v3.VSelect(
                                    v_model=("support_comps",),
                                    items=("support_comp_options",),
                                    item_title="title", item_value="value",
                                    label="Support condition",
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
                                            with html.Div(classes="d-flex align-center ga-1"):
                                                v3.VBtn(
                                                    text="Explore", size="x-small", variant="text", color="primary",
                                                    click=(ctrl.start_live_load_edit, "[idx]"),
                                                )
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
                            html.Div("Create a load", classes="text-subtitle-2 font-weight-bold mb-1")
                            html.Div("Choose where it acts, then define its physical quantity and magnitude.",
                                     classes="text-caption text-medium-emphasis mb-2")

                            # Interactive 3D Picking Status Banner
                            with v3.VAlert(type="info", variant="tonal", density="compact", classes="mb-2 pa-1"):
                                html.Span("{{ click_status }}", classes="text-caption")

                            # Target Location Selector: surface loads act on faces; concentrated
                            # loads act at a point or element location.
                            with v3.VBtnToggle(
                                v_model=("load_target",), mandatory=True,
                                density="compact", divided=True,
                                color="primary", classes="mb-2 w-100",
                            ):
                                v3.VBtn("Surface / Face", value="face",
                                        size="small", classes="flex-grow-1")
                                v3.VBtn("Point Location", value="point",
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
                                        label="Point location preset",
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
                                v3.VBtn("Force", value="force",
                                        size="small", classes="flex-grow-1")
                                v3.VBtn("Moment / Torque", value="moment",
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
                                    v3.VTextField(v_model=("force_mag",), label="Force magnitude (N)",
                                                  density="compact", hide_details=True)
                                    v3.VTextField(v_model=("force_vec",), label="Direction vector [Fx, Fy, Fz]",
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
                                    v_model=("pressure_val",), label="Pressure magnitude (Pa)",
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
                            html.Div(
                                "Inspect stress at a model point and explore how a selected load changes the response.",
                                classes="text-caption text-medium-emphasis mb-2",
                            )
                            v3.VBtn(
                                "Open point analysis", block=True, size="small", color="primary",
                                variant="tonal", prepend_icon="mdi-chart-scatter-plot",
                                click="analysis_dock_open = true",
                                classes="mb-3",
                            )

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

        # The original viewport remains the central work surface. Point analysis is
        # a contextual overlay dock. Opening it does not toggle the setup drawer: a
        # drawer transition resizes the VTK host while it is rendering and causes a
        # stale canvas size/camera crop in some Trame/PyVista client combinations.
        with layout.content:
            html.Style("""
                @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
                :root { --mos-ui-font: 'DM Sans', 'Segoe UI', sans-serif; --mos-display-font: 'Manrope', 'Segoe UI', sans-serif; }
                body, button, input, textarea, select, .v-application { font-family: var(--mos-ui-font) !important; }
                .v-toolbar-title, .v-btn, .v-card-title, .v-expansion-panel-title { font-family: var(--mos-display-font) !important; }
                .mos-analysis-dock {
                    /* Keep the dock out of the VTK host's layout/stacking context.
                       It overlays the model without resizing or reflowing the canvas. */
                    position: fixed !important; z-index: 10000 !important;
                    top: 76px; right: 14px; bottom: 14px;
                    box-sizing: border-box; width: min(460px, calc(100vw - 28px));
                    display: flex; flex-direction: column; min-height: 0;
                    overflow: hidden; border: 1px solid rgba(148,163,184,.26);
                    border-radius: 14px; background: rgba(20,22,27,.985);
                    box-shadow: 0 18px 55px rgba(0,0,0,.48); color: #e5e7eb;
                    pointer-events: auto;
                }
                .mos-analysis-dock .v-card { color: #e5e7eb; }
                .mos-analysis-dock .v-expansion-panel, .mos-analysis-dock .v-expansion-panel-text, .mos-analysis-dock .v-field { background-color: #222630 !important; color: #e5e7eb !important; }
                .mos-analysis-dock .v-field__input, .mos-analysis-dock .v-label { color: #cbd5e1 !important; }
                .mos-analysis-dock { font-family: var(--mos-ui-font); letter-spacing: .005em; }
                .mos-analysis-dock .v-card .text-subtitle-1,
                .mos-analysis-dock .dock-label { font-family: var(--mos-display-font); }
                .mos-analysis-dock .text-subtitle-1 { letter-spacing: -.02em; }
                .mos-analysis-dock .v-card-text { min-height: 0; }
                .mos-analysis-dock .text-medium-emphasis { color: #a1a1aa !important; }
                .mos-analysis-dock .text-caption { color: #a1a1aa; }
                .mos-analysis-dock .v-field--variant-outlined .v-field__outline { opacity: .55; }
                .mos-analysis-dock .v-slider { margin-top: 4px; }
                .mos-analysis-dock svg { display: block; width: 100%; max-width: 100%; height: auto; }
                .mos-analysis-dock table th { padding: 6px 5px; }
                .mos-analysis-dock table td { padding: 6px 5px; border-top: 1px solid #343840; }
                .mos-analysis-dock .dock-section {
                    border: 1px solid rgba(148,163,184,.16); border-radius: 12px;
                    background: linear-gradient(145deg, rgba(39,42,49,.84), rgba(28,30,36,.76));
                    padding: 12px; margin-bottom: 10px;
                }
                .mos-analysis-dock .dock-label {
                    font-size: 10px; letter-spacing: .11em; text-transform: uppercase;
                    font-weight: 800; color: #9cc9ff; margin-bottom: 8px;
                }
                .mos-analysis-dock .mos-lab-formula {
                    font-family: 'STIX Two Math', 'Cambria Math', Georgia, serif;
                    font-size: 14px; line-height: 1.7; letter-spacing: .01em;
                    color: #dbeafe; padding: 10px 12px; border-radius: 9px;
                    border: 1px solid rgba(96,165,250,.2); background: rgba(37,99,235,.10);
                    overflow-wrap: anywhere;
                }
                .mos-analysis-dock .v-expansion-panel { background: rgba(39,42,49,.55); }
                .mos-analysis-dock .v-expansion-panel-title { font-weight: 650; }
                .mos-analysis-dock .v-field { border-radius: 9px; }
                .mos-analysis-dock .v-btn { letter-spacing: .01em; }
                @media (max-width: 700px) {
                    .mos-analysis-dock { left: 8px; right: 8px; width: auto; top: 72px; bottom: 8px; }
                }
            """)
            with html.Div(style="position:relative;width:100%;height:100%;overflow:hidden;"):
                with html.Div(v_show="active_page === 'workspace'", style="width:100%;height:100%;"):
                    with v3.VContainer(fluid=True, classes="pa-0 fill-height"):
                        with html.Div(style="position: relative; width: 100%; height: 100%; padding: 8px; box-sizing: border-box;"):
                            view = plotter_ui(pl, mode="server")
                            ctrl.view_update = view.update

            # Analysis dock: the selected point, plane and load stay in one context.
            with html.Div(v_show="analysis_dock_open", classes="mos-analysis-dock"):
                with v3.VCard(theme="dark", variant="flat", classes="fill-height d-flex flex-column", style="background:#14161b;color:#e5e7eb;"):
                    with html.Div(classes="d-flex align-center justify-space-between px-3 py-2",
                                  style="border-bottom:1px solid rgba(148,163,184,.18);flex:0 0 auto;"):
                        with html.Div():
                            html.Div("LEARNING LAB · LIVE MODEL", classes="text-caption font-weight-bold text-info")
                            html.Div("Explore the stress state", classes="text-subtitle-1 font-weight-bold")
                        v3.VBtn(icon="mdi-close", variant="text", density="comfortable",
                                title="Close analysis", click="analysis_dock_open = false")
                    with html.Div(classes="pa-3", style="overflow-y:auto;min-height:0;flex:1 1 auto;"):
                        with html.Div(v_show="analysis_result_warning.length > 0", classes="mb-2"):
                            v3.VAlert("{{ analysis_result_warning }}", type="warning",
                                      variant="tonal", density="compact")
                        with html.Div(classes="dock-section"):
                            html.Div("Live load", classes="dock-label")
                            v3.VSelect(
                                v_model=("live_load_selection",), items=("live_load_items",),
                                item_title="title", item_value="value", label="Applied load to edit",
                                density="compact", hide_details=True, variant="outlined",
                                disabled=("live_load_items.length === 0",), classes="mb-2",
                            )
                            with html.Div(v_show="editing_load_index >= 0"):
                                html.Div("{{ live_load_mode === 'pressure' ? 'Pressure (Pa)' : (live_load_mode === 'moment' ? 'Moment components (N·m)' : 'Force components (N)') }}",
                                         classes="text-caption font-weight-medium mb-1")
                                with html.Div(v_show="live_load_mode !== 'pressure'",
                                              style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;"):
                                    v3.VTextField(v_model=("live_load_x",), label="X", type="number",
                                                  density="compact", hide_details=True, variant="outlined",
                                                  disabled=("!live_load_ready",))
                                    v3.VTextField(v_model=("live_load_y",), label="Y", type="number",
                                                  density="compact", hide_details=True, variant="outlined",
                                                  disabled=("!live_load_ready",))
                                    v3.VTextField(v_model=("live_load_z",), label="Z", type="number",
                                                  density="compact", hide_details=True, variant="outlined",
                                                  disabled=("!live_load_ready",))
                                v3.VTextField(v_model=("live_pressure",), label="Uniform pressure (Pa)",
                                              type="number", density="compact", hide_details=True,
                                              variant="outlined", v_show="live_load_mode === 'pressure'",
                                              disabled=("!live_load_ready",))
                                with html.Div(classes="d-flex align-center justify-space-between"):
                                    html.Span("Magnitude scale", classes="text-caption text-medium-emphasis")
                                    v3.VBtn("Reset", size="x-small", variant="text", color="secondary",
                                            disabled=("!live_load_ready",), click="live_load_multiplier = 1.0")
                                v3.VSlider(
                                    v_model=("live_load_multiplier",), min=0, max=5.0, step=0.01,
                                    thumb_label=True, color="primary",
                                    density="compact", hide_details=True,
                                    disabled=("!live_load_ready",),
                                )
                                html.Div("{{ (live_load_multiplier * 100).toFixed(0) }}% of the selected load's base value",
                                         classes="text-caption text-info")
                                html.Div("{{ live_load_value }}", classes="text-caption font-weight-medium mt-1")
                            html.Div("{{ live_load_status }}", classes="text-caption text-medium-emphasis mt-2")
                            html.Div("{{ live_response_status }}", classes="text-caption text-medium-emphasis mt-2")
                            html.Div("Support or material edits change the stiffness operator and require Solve. Load changes use cached unit-load responses.",
                                     classes="text-caption text-medium-emphasis mt-2")

                        with html.Div(classes="dock-section"):
                            with html.Div(classes="d-flex justify-space-between align-center mb-2"):
                                html.Div("Sample location", classes="dock-label mb-0")
                                v3.VChip("CLICK MODEL TO PICK", color="info", variant="tonal", size="x-small")
                            with html.Div(style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;"):
                                v3.VTextField(v_model=("lab_point_x",), label="X (m)", type="number",
                                              density="compact", hide_details=True, variant="outlined")
                                v3.VTextField(v_model=("lab_point_y",), label="Y (m)", type="number",
                                              density="compact", hide_details=True, variant="outlined")
                                v3.VTextField(v_model=("lab_point_z",), label="Z (m)", type="number",
                                              density="compact", hide_details=True, variant="outlined")
                            html.Div("{{ lab_status }}", classes="text-caption text-info mt-2")
                            html.Div("Nearest node {{ lab_node_index }} · {{ lab_point_coords }}",
                                     classes="text-caption text-medium-emphasis")

                        with html.Div(v_if="lab_has_result", classes="dock-section"):
                            html.Div("Stress at selected point", classes="dock-label")
                            with html.Div(style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;"):
                                with html.Div(classes="pa-2 rounded", style="background:rgba(14,165,233,.12)"):
                                    html.Div("von Mises", classes="text-caption")
                                    html.Div("{{ lab_fea_vm }}", classes="text-subtitle-2 font-weight-bold")
                                with html.Div(classes="pa-2 rounded", style="background:rgba(139,92,246,.12)"):
                                    html.Div("σ₁", classes="text-caption")
                                    html.Div("{{ lab_sigma_1 }} MPa", classes="text-subtitle-2 font-weight-bold")
                                with html.Div(classes="pa-2 rounded", style="background:rgba(245,158,11,.12)"):
                                    html.Div("Max shear", classes="text-caption")
                                    html.Div("{{ lab_max_shear }}", classes="text-subtitle-2 font-weight-bold")
                            html.Div("Stress tensor · MPa", classes="text-caption font-weight-bold mt-3 mb-1")
                            html.Div(v_html=("lab_matrix_html",), style="overflow-x:auto;")

                        with html.Div(v_if="!lab_has_result", classes="dock-section"):
                            with v3.VAlert(type="info", variant="tonal", density="compact"):
                                html.Span("Solve a model, then click a point on the mesh to inspect its FEA stress state.")

                        with html.Div(classes="dock-section"):
                            html.Div("Traction on selected plane", classes="dock-label")
                            v3.VSelect(
                                v_model=("lab_plane_preset",), items=("lab_plane_items",),
                                label="Plane normal", density="compact", hide_details=True,
                                variant="outlined", classes="mb-2",
                            )
                            v3.VTextField(
                                v_model=("lab_plane_custom",), label="Custom normal · nx, ny, nz",
                                density="compact", hide_details=True, variant="outlined", classes="mb-2",
                            )
                            with html.Div(classes="d-flex justify-space-between align-center mb-1"):
                                html.Span("Traction vector", classes="text-caption text-medium-emphasis")
                                html.Span("{{ lab_plane_traction }}", classes="text-caption font-weight-bold")
                            with html.Div(style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px;"):
                                with html.Div(classes="pa-2 rounded", style="background:rgba(14,165,233,.10)"):
                                    html.Div("Normal stress σₙ", classes="text-caption")
                                    html.Div("{{ lab_normal_stress }}", classes="text-subtitle-2 font-weight-bold")
                                with html.Div(classes="pa-2 rounded", style="background:rgba(245,158,11,.10)"):
                                    html.Div("Shear magnitude τ", classes="text-caption")
                                    html.Div("{{ lab_plane_shear }}", classes="text-subtitle-2 font-weight-bold")

                        with html.Div(classes="dock-section"):
                            html.Div("Mohr’s circles", classes="dock-label")
                            html.Div(v_html=("lab_mohr_svg",), style="width:100%;overflow:hidden;")
                            html.Div("Principal stresses: {{ lab_fea_principal }} · Tresca: {{ lab_fea_tresca }}",
                                     classes="text-caption text-medium-emphasis mt-1")

                        with v3.VExpansionPanels(variant="accordion", density="compact", multiple=True):
                            with v3.VExpansionPanel():
                                with v3.VExpansionPanelTitle():
                                    html.Span("Tensor diagram, directions and strain/displacement", classes="text-caption font-weight-bold")
                                with v3.VExpansionPanelText():
                                    html.Div(v_html=("lab_tensor_svg",), style="width:100%;overflow:hidden;")
                                    html.Div(v_html=("lab_directions_html",), style="overflow-x:auto;")
                                    html.Div("Strain tensor", classes="text-caption font-weight-bold mt-2")
                                    html.Div(v_html=("lab_strain_matrix_html",), style="overflow-x:auto;")
                                    html.Div("Displacement at point: {{ lab_displacement }}", classes="text-caption mt-2")
                            with v3.VExpansionPanel():
                                with v3.VExpansionPanelTitle():
                                    html.Span("Solved model information", classes="text-caption font-weight-bold")
                                with v3.VExpansionPanelText():
                                    html.Div(v_html=("lab_model_summary",), style="overflow-x:auto;")
                            with v3.VExpansionPanel():
                                with v3.VExpansionPanelTitle():
                                    html.Span("Idealised mechanics quick checks", classes="text-caption font-weight-bold")
                                with v3.VExpansionPanelText():
                                    html.Div("Independent textbook checks; these do not modify the FEA model.",
                                             classes="text-caption text-medium-emphasis mb-2")
                                    v3.VSelect(v_model=("lab_tool_mode",), items=("lab_tool_items",),
                                               label="Check type", density="compact", hide_details=True,
                                               variant="outlined", classes="mb-2")
                                    v3.VTextField(v_model=("lab_q1",), label=("lab_q1_label",), type="number",
                                                  density="compact", hide_details=True, variant="outlined", classes="mb-1")
                                    v3.VTextField(v_model=("lab_q2",), label=("lab_q2_label",), type="number",
                                                  density="compact", hide_details=True, variant="outlined", classes="mb-1")
                                    v3.VTextField(v_model=("lab_q3",), label=("lab_q3_label",), type="number",
                                                  density="compact", hide_details=True, variant="outlined", classes="mb-1")
                                    v3.VTextField(v_model=("lab_q4",), label=("lab_q4_label",), type="number",
                                                  density="compact", hide_details=True, variant="outlined")
                                    html.Div("{{ lab_quick_title }}", classes="text-caption font-weight-bold mt-2")
                                    html.Div("{{ lab_quick_primary_label }}: {{ lab_quick_primary_value }}",
                                             classes="text-caption")
                                    html.Div("{{ lab_quick_secondary_label }}: {{ lab_quick_secondary_value }}",
                                             classes="text-caption")
                                    html.Div("{{ lab_quick_tertiary_label }}: {{ lab_quick_tertiary_value }}",
                                             classes="text-caption")
                                    html.Div("{{ lab_quick_formula }}", classes="mos-lab-formula mt-2")
                                    html.Div("{{ lab_quick_note }}", classes="text-caption text-medium-emphasis mt-1")
                                    html.Div("{{ lab_quick_error }}", classes="text-caption text-error")
    server.start()
