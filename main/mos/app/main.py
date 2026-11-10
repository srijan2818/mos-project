"""trame + PyVista front end.

    python -m mos.app --server [--spec model.mos] [--port 8080]

The drawer holds interactive geometric/CAD model builders, materials, supports,
loads, and view controls; the 3D view is a server-side-rendered VTK scene.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import time
import re
import threading
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyvista as pv
from PIL import Image
from trame_pyvista.ui import plotter_ui
from trame.app import get_server
from trame.ui.vuetify3 import SinglePageWithDrawerLayout
from trame.widgets import html
from trame.widgets import vuetify3 as v3

from mos import viz

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Everything the app writes (meshes, masks, STEP exports, capture sessions) goes here,
# never next to the bundled examples; the folder is git-ignored and safe to delete.
GENERATED_DIR = PROJECT_ROOT / "cad_models" / "_generated"


def _click_to_image(ox, oy, cw, ch, nw, nh):
    """Map a click on an ``object-fit: contain`` <img> to image pixels; None if on the letterbox."""
    scale = min(cw / nw, ch / nh)
    x = (ox - (cw - nw * scale) / 2.0) / scale
    y = (oy - (ch - nh * scale) / 2.0) / scale
    return (x, y) if 0 <= x < nw and 0 <= y < nh else None


def _project_path(path) -> Path:
    """A user/preset path: absolute as given, relative ones resolved against the project."""
    path = Path(str(path)).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
from mos.core.cad_parametric import generate_cad_from_parameters
from mos.core.cv_detector import analyze_image_object
from mos.core.image_to_cad import image_to_step
from mos.core.image_reconstruction import reconstruct_single_image
from mos.core.image_to_mesh import load_mask_image, save_mask_image
from mos.core.multiview import (
    calibrate_point_cloud_scale, gaussian_ply_to_point_cloud,
    nerfstudio_commands, prepare_capture_session, run_colmap_sparse_reconstruction,
    run_nerfstudio_preprocess, tool_availability,
)
from mos.core.solver import (DIRECT_SOLVE_MAX_DOFS, FREE_BODY_MAX_DOFS, CachedLinearSolver,
                             count_dofs, solve_case)
from mos.core.regions import cad_boundary_name_for_side
from mos.core.spec import build, coarsen_mesh_text, parse_spec
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
# Proportional limits (MPa), sources in docs/theory.md "Proportional limit". Published
# proportional limits are scarce, so these are specified minima used as stand-ins:
#   metals  sigma_pl = minimum yield (A36 Fy; 6061-T6 and annealed Ti-6Al-4V 0.2 % offset),
#           checked with von Mises; the true limit of Al/Ti alloys lies somewhat lower;
#   brittle sigma_pl / sigma_pl_c = tension / compression limits, Coulomb-Mohr check:
#           concrete f'c = 30 MPa: 0.1 f'c (cracking) / 0.4 f'c (end of near-linear range);
#           gray iron (ASTM A48 class 30) has no linear range in tension, so its tensile and
#           compressive *strengths* are used.
MATERIAL_PRESETS = {
    "Steel":     {"E": 210e9, "nu": 0.30, "rho": 7850, "sigma_pl": 250, "sigma_pl_c": None},
    "Aluminum":  {"E": 68.9e9, "nu": 0.33, "rho": 2700, "sigma_pl": 240, "sigma_pl_c": None},
    "Titanium":  {"E": 113.8e9, "nu": 0.34, "rho": 4430, "sigma_pl": 825, "sigma_pl_c": None},
    "Concrete":  {"E": 30.0e9, "nu": 0.20, "rho": 2400, "sigma_pl": 3, "sigma_pl_c": 12},
    "CastIron":  {"E": 120.0e9, "nu": 0.26, "rho": 7200, "sigma_pl": 214, "sigma_pl_c": 752},
    "Custom":    None,  # keeps whatever the user typed
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

# Solid elements carry no rotational DOFs, so a "pinned" face is the same as a fixed one.
# Values must be unique: a v-select displays every item whose value matches.
# Three-photo solid (mos.core.multiview_hull): slot order, what to photograph, image axes.
ORTHO_VIEWS = [
    {"axis": "front", "title": "Front · level with the part, looking straight at it",
     "axes": "→ length x   ↑ height z"},
    {"axis": "top", "title": "Top · straight down, the front edge at the bottom of the photo",
     "axes": "→ length x   ↑ depth y (away from you)"},
    {"axis": "side", "title": "Right side · level, from the part's right",
     "axes": "→ depth y (front → back)   ↑ height z"},
]
ORTHO_HELP = ("Plain background, part in the middle of each photo. Shoot from at least 5× the "
              "part's size and zoom in: parts at different depths are magnified differently "
              "(measured on an L-bracket at 5×: sizes within 6 %, volume +7 %; at 10×: 3 %, +4 %). "
              "Check that each outline below shows the stated axes; flip or rotate it if not: a "
              "mirrored photo can still give a consistent (wrong) solid.")

SUPPORT_COMP_OPTIONS = [
    {"title": "Fixed / Clamped (X, Y, Z)", "value": "xyz"},
    {"title": "Roller — free X (fix Y, Z)", "value": "yz"},
    {"title": "Roller — free Y (fix X, Z)", "value": "xz"},
    {"title": "Roller — free Z (fix X, Y)", "value": "xy"},
    {"title": "Fix X only · symmetry plane ⊥ X", "value": "x"},
    {"title": "Fix Y only · symmetry plane ⊥ Y", "value": "y"},
    {"title": "Fix Z only · symmetry plane ⊥ Z", "value": "z"},
]

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
        # Silently substituting another face put supports/loads on the wrong surface.
        raise ValueError(
            f"A support or load refers to {face_name.replace('cad_face_', 'Face ')}, which does not "
            "exist on the current geometry. Re-select its face (or clear it) after changing the model.")
    return face_name


# ---------------------------------------------------------------------------
def main(argv=None):
    """Build the UI and start the trame server (blocks until stopped)."""
    server = get_server(client_type="vue3")
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    server.enable_module({"serve": {"mos_static": str(Path(__file__).with_name("static")),
                                    "mos_generated": str(GENERATED_DIR)},
                          "scripts": ["mos_static/upload.js"], "styles": ["mos_static/app.css"]})
    state, ctrl = server.state, server.controller
    parser = server.cli
    parser.add_argument("--spec", help="model description file to open at startup")
    args, _ = parser.parse_known_args(argv)

    pl = pv.Plotter()
    S = SimpleNamespace(grid=None, scalars={}, u_pts=None, res=None,
                        preview_mesh=None,
                        spec=None, model=None, case=None, main_actor=None,
                        cached_solver=None, live_response=None, limit_nodes=None,
                        cached_geometry_signature=None, solved_spec_text=None,
                        lab_refreshed_inputs=None,
                        point_picker_installed=False, analysis_actors=[],
                        geometry_identity=None,
                        cv_source=None, cv_auto_points=[], cv_auto_bbox=None,
                        mv3_masks={}, mv3_sig=None,
                        cv_mask=None,
                        cancel_event=threading.Event())

    def cancel_current_operation():
        """Globally cancel any running background solver, CV analysis, or reconstruction task."""
        S.cancel_event.set()
        state.is_busy = False
        state.busy_task_name = ""
        state.busy_task_desc = ""
        state.solve_running = False
        state.cv_running = False
        state.cv3d_running = False
        state.mv3_running = False
        state.status = "Operation cancelled by user."
        state.status_type = "warning"
        state.cv_status = "Cancelled."
        state.cv3d_status = "Cancelled."
        ctrl.view_update()

    def _spawn(worker, name):
        """Run ``worker(ui)`` off the UI thread; ``ui(fn)`` applies state changes on it.

        trame state is not thread-safe and changes made from another thread are never
        pushed to the browser, so every mutation from a worker goes through ``ui``,
        which schedules it on the server's event loop inside ``with state:`` (flush).
        Without a running loop (offline smoke test, scripting) the worker runs inline.
        """
        S.cancel_event.clear()
        state.is_busy = True
        state.busy_task_name = name.replace("mos-", "").replace("-", " ").title()
        state.busy_task_desc = "Processing... Click Cancel to stop anytime."

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            try:
                worker(lambda fn: fn())
            finally:
                state.is_busy = False
                state.busy_task_name = ""
                state.busy_task_desc = ""
            return

        def ui(fn):
            """Apply ``fn`` (which mutates state) on the event loop and flush it."""
            def apply():
                """Run ``fn`` and flush the resulting changes to the browser."""
                with state:
                    fn()
            loop.call_soon_threadsafe(apply)

        def _wrapped_worker(ui_cb):
            try:
                worker(ui_cb)
            finally:
                def _finish_busy():
                    state.is_busy = False
                    state.busy_task_name = ""
                    state.busy_task_desc = ""
                ui_cb(_finish_busy)

        threading.Thread(target=_wrapped_worker, args=(ui,), name=name, daemon=True).start()

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
            except Exception as exc:
                log.warning("update_analysis_overlay: %s: %s", type(exc).__name__, exc)
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
                # High-visibility target probe (central beacon + outer target halo)
                beacon_r = max(0.016 * diag, 1e-4)
                marker = pv.Sphere(radius=beacon_r, center=point)
                actor = pl.add_mesh(marker, color="#f59e0b", opacity=1.0,
                                     pickable=False, reset_camera=False, render=False)
                S.analysis_actors.append(actor)
                halo = pv.Sphere(radius=beacon_r * 2.2, center=point)
                actor_halo = pl.add_mesh(halo, color="#fbbf24", opacity=0.30,
                                          pickable=False, reset_camera=False, render=False)
                S.analysis_actors.append(actor_halo)
            except Exception as exc:
                log.warning("update_analysis_overlay: %s: %s", type(exc).__name__, exc)

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
                    plane_size = max(0.24 * diag, 2e-3)
                    plane_mesh = pv.Plane(
                        center=sample_xyz, direction=normal,
                        i_size=plane_size, j_size=plane_size,
                        i_resolution=1, j_resolution=1,
                    )
                    actor = pl.add_mesh(
                        plane_mesh, color="#0284c7", opacity=0.45,
                        show_edges=True, edge_color="#38bdf8", line_width=3.0,
                        culling=False, pickable=False, reset_camera=False, render=False,
                    )
                    S.analysis_actors.append(actor)

                    disc_fn = getattr(pv, "Disc", None)
                    if disc_fn is not None:
                        try:
                            disc_mesh = disc_fn(
                                center=sample_xyz, inner=0, outer=plane_size * 0.48,
                                normal=normal, c_res=36
                            )
                            actor_disc = pl.add_mesh(
                                disc_mesh, color="#38bdf8", opacity=0.30,
                                show_edges=True, edge_color="#e0f2fe", line_width=1.5,
                                culling=False, pickable=False, reset_camera=False, render=False,
                            )
                            S.analysis_actors.append(actor_disc)
                        except Exception:
                            pass

                    normal_glyph = viz.create_arrow_glyph(
                        sample_xyz, normal, diag, reference_magnitude=1.0
                    )
                    if normal_glyph is not None:
                        actor = pl.add_mesh(normal_glyph, color="#10b981",
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
            except Exception as exc:
                log.warning("update_scene: %s: %s", type(exc).__name__, exc)
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
            except Exception as exc:
                log.warning("update_scene: %s: %s", type(exc).__name__, exc)

        bounds = S.grid.bounds
        diag = float(np.linalg.norm(np.ptp(np.asarray(S.grid.points), axis=0)))
        mesh_ref = S.model.mesh if S.model else None

        # Nodes at/over the proportional limit: red halo + red dot at the deformed position
        if S.limit_nodes is not None and len(S.limit_nodes):
            pts = (S.grid.points[S.limit_nodes] + float(state.scale) * S.u_pts[S.limit_nodes])
            cloud = pv.PolyData(np.ascontiguousarray(pts, dtype=float))
            # Screen-space discs keep their size at any zoom. Not in S.analysis_actors: the probe
            # overlay removes those on every redraw; pl.clear() above drops these instead.
            pl.add_points(cloud, color="#ef4444", point_size=26, opacity=0.35,
                          render_points_as_spheres=True, pickable=False)
            pl.add_points(cloud, color="#b91c1c", point_size=9,
                          render_points_as_spheres=True, pickable=False)

        # Highlight support and load faces for clear visual feedback
        if bool(state.show_face_highlight):
            # 1. Render all active fixed support surfaces (Vibrant Crimson Red)
            supported_faces = set()
            for sup in getattr(state, "supports_list", []):
                face_key = sup.get("face") if isinstance(sup, dict) else None
                if not face_key or face_key in supported_faces:
                    continue
                supported_faces.add(face_key)
                sup_mesh = viz.create_face_mesh(face_key, bounds, mesh=mesh_ref)
                if sup_mesh is not None:
                    pl.add_mesh(
                        sup_mesh, color="#ef4444", opacity=0.60,
                        show_edges=True, line_width=3.0, edge_color="#b91c1c", pickable=False,
                    )

            # 2. Highlight candidate support face if in loads/supports tab (Warm Coral/Orange)
            if state.active_tab == "loads" and getattr(state, "support_face", None):
                sup_candidate = state.support_face
                if sup_candidate not in supported_faces:
                    cand_mesh = viz.create_face_mesh(sup_candidate, bounds, mesh=mesh_ref)
                    if cand_mesh is not None:
                        pl.add_mesh(
                            cand_mesh, color="#f97316", opacity=0.45,
                            show_edges=True, line_width=2.5, edge_color="#ea580c", pickable=False,
                        )

            # 3. Highlight target load face (Cyan)
            if state.selected_face and state.selected_face not in supported_faces:
                face_mesh = viz.create_face_mesh(state.selected_face, bounds, mesh=mesh_ref)
                if face_mesh is not None:
                    pl.add_mesh(
                        face_mesh, color="#06b6d4", opacity=0.55,
                        show_edges=True, line_width=2.5, edge_color="#22d3ee", pickable=False,
                    )

        # Point/plane annotations are maintained independently of the base mesh.
        update_analysis_overlay(render=False)

        # 3D load glyphs after solve
        if bool(state.show_glyphs) and S.case is not None:
            try:
                for g_mesh, g_color in viz.build_load_glyphs(S.case, S.model.mesh, diag):
                    pl.add_mesh(g_mesh, color=g_color, pickable=False)
            except Exception as exc:
                log.warning("update_scene: %s: %s", type(exc).__name__, exc)

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
            except Exception as exc:
                log.warning("update_scene: %s: %s", type(exc).__name__, exc)

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
            except Exception as exc:
                log.warning("update_scene: %s: %s", type(exc).__name__, exc)

        if reset_camera:
            pl.reset_camera()
        elif saved_camera is not None:
            try:
                pl.camera_position = saved_camera
                pl.reset_camera_clipping_range()
            except Exception as exc:
                log.warning("update_scene: %s: %s", type(exc).__name__, exc)
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
        except Exception as exc:
            log.warning("_pick_point_direct: %s: %s", type(exc).__name__, exc)
        if bool(getattr(state, "analysis_dock_open", False)):
            update_analysis_overlay()
        else:
            update_scene()

    # ------------------------------------------------------------------ helpers
    def _parse_vec3(s: str) -> np.ndarray:
        """Three comma-separated numbers; anything else is a ValueError, never a default."""
        try:
            parts = [float(x) for x in str(s).split(",")]
        except ValueError:
            parts = []
        if len(parts) != 3:
            raise ValueError(f"'{s}' is not a vector of three numbers 'x, y, z'")
        return np.array(parts, float)

    def _check_vec3_text(s: str, what: str):
        """Spec vectors may be expressions; only require three non-empty components."""
        parts = [p.strip() for p in str(s).split(",")]
        if len(parts) != 3 or not all(parts):
            raise ValueError(f"{what} '{s}' needs three comma-separated components 'x, y, z'")

    def _get_moment_vector(st, bounds, mesh_ref=None) -> np.ndarray:
        try:
            mag = float(st.moment_mag)
        except (TypeError, ValueError):
            raise ValueError(f"Moment magnitude '{st.moment_mag}' is not a number (N·m)") from None
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

    _BOX_FACE_CONDITIONS = {"+x": "x > L - 1e-6", "-x": "x < 1e-6",
                            "+y": "y > H/2 - 1e-6", "-y": "y < -H/2 + 1e-6",
                            "+z": "z > W/2 - 1e-6", "-z": "z < -W/2 + 1e-6"}

    def _unknown_face(face_key):
        return ValueError(f"Face '{face_key}' is not an outer face of the current box "
                          "(+x, -x, +y, -y, +z, -z). Re-select it after changing the model.")

    def _box_face_condition(face_key):
        """Face predicate in terms of the builder's `param L/H/W`, so it can't go stale."""
        if face_key not in _BOX_FACE_CONDITIONS:
            raise _unknown_face(face_key)
        return _BOX_FACE_CONDITIONS[face_key]

    def _face_condition(face_key, bounds):
        """Numeric face predicate from mesh bounds (for scripts without box params)."""
        xmin, xmax, ymin, ymax, zmin, zmax = (float(b) for b in bounds)
        if not np.all(np.isfinite(bounds)) or xmax <= xmin or ymax <= ymin or zmax <= zmin:
            raise ValueError(f"No valid model bounds for face '{face_key}'; solve the model first.")
        m = {"+x": f"x > {xmax!r} - 1e-6", "-x": f"x < {xmin!r} + 1e-6",
             "+y": f"y > {ymax!r} - 1e-6", "-y": f"y < {ymin!r} + 1e-6",
             "+z": f"z > {zmax!r} - 1e-6", "-z": f"z < {zmin!r} + 1e-6"}
        if face_key not in m:
            raise _unknown_face(face_key)
        return m[face_key]

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
            # A face load acts at the face; a point load keeps the picked/typed point.
            if state.origin_is_centroid and state.load_target == "face":
                state.point_x, state.point_y, state.point_z = float(c[0]), float(c[1]), float(c[2])

    def _geometry_identity():
        return (state.geom_mode, state.cad_preset, str(state.cad_custom_path),
                str(getattr(state, "cv_mesh_source", "")), str(getattr(state, "cv_mask_path", "")),
                str(getattr(state, "cv_step_path", "")))

    def _drop_stale_face_refs():
        """After the geometry source changes, `cad_face_N` names refer to another model.

        Such supports/loads are removed (a mask mesh always has faces 1 and 2, so keeping them
        would silently move them). Box-face keys (+x, -x, ...) are kept: they are re-mapped.
        Returns the number of removed entries.
        """
        identity = _geometry_identity()
        if identity == S.geometry_identity:
            return 0
        S.geometry_identity = identity
        keep = lambda items: [d for d in items if not str(d.get("face", "")).startswith("cad_face_")]
        sups, lds = keep(state.supports_list), keep(state.loads_list)
        dropped = len(state.supports_list) - len(sups) + len(state.loads_list) - len(lds)
        if dropped:
            state.supports_list, state.loads_list = sups, lds
            state.editing_load_index, state.live_load_selection = -1, ""
            S.live_response = None
            state.status = (f"Geometry changed: removed {dropped} support(s)/load(s) on the previous "
                            "model's faces. Re-select them on the new model.")
            state.status_type = "warning"
        return dropped

    # ------------------------------------------------------------------ spec rebuild
    def _limit_terms():
        """`, sigma_pl = …[, sigma_pl_c = …]` (Pa) from the MPa inputs; '' when left blank."""
        spl, splc = str(state.mat_spl or "").strip(), str(state.mat_splc or "").strip()
        if not spl:
            return ""
        out = f", sigma_pl = {float(spl) * 1e6:.4g}"
        if bool(state.mat_brittle):
            if not splc:
                raise ValueError("brittle material: enter the compression limit (MPa)")
            out += f", sigma_pl_c = {float(splc) * 1e6:.4g}"
        return out

    def _rebuild_spec_from_interactive(mesh_ref_override=None):
        """Construct a valid .mos script from the interactive UI state.

        ``mesh_ref_override`` is used while solving a newly selected STEP file:
        it is the just-imported mesh, not the mesh from the previous model.
        """
        lines = ["# MoS Model — Interactive Builder"]

        # 1. Mesh
        if state.geom_mode == "cv" and str(getattr(state, "cv_mesh_source", "step")) == "hull":
            # Solid carved from the three-photo session (outlines + orientation + scale).
            ms = f" {float(state.cad_mesh_size):.3g}" if state.cad_mesh_size else ""
            lines.append(f"mesh     hull {json.dumps(str(state.mv3_session))}{ms}")
        elif state.geom_mode == "cv" and str(getattr(state, "cv_mesh_source", "step")) == "mask":
            # Photographed outline meshed directly (no template, no STEP round trip).
            ms = f" {float(state.cad_mesh_size):.3g}" if state.cad_mesh_size else ""
            lines.append(f"mesh     mask {json.dumps(str(state.cv_mask_path))} "
                         f"{float(state.cv_mask_size):.4g} {float(state.cv_mask_thickness):.4g}{ms}")
        elif state.geom_mode in ("step", "cv"):
            # Relative paths resolve against the project, not the launch directory.
            if state.geom_mode == "cv":
                step_file = str(_project_path(state.cv_step_path))
            elif state.cad_preset == "custom":
                step_file = str(_project_path(str(state.cad_custom_path).strip()))
            else:
                step_file = str(_project_path(state.cad_preset))
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
        lines.append(f"material E = {float(state.mat_E):.4e}, nu = {float(state.mat_nu):.4g}, "
                     f"rho = {float(state.mat_rho):.4g}{_limit_terms()}")
        lines.append("")

        # 3. Supports
        bounds = tuple(state.model_bounds)
        mesh_ref = mesh_ref_override if mesh_ref_override is not None else (S.model.mesh if S.model else None)
        if state.geom_mode == "box":
            mesh_ref = None   # the last solved mesh may be a STEP model; box faces are +x, -x, ...
        if state.supports_list:
            lines.append("# Boundary Supports")
            for i, sup in enumerate(state.supports_list):
                face = sup["face"]
                face = _resolved_face_name(face, mesh_ref)
                comps = sup["comps"]
                if state.geom_mode == "box":
                    reg = f"sup_{i}_{_safe_region_name(face, 'face')}"
                    lines.append(f"region   {reg} facets {_box_face_condition(face)}")
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
                        lines.append(f"region   {reg} facets {_box_face_condition(face)}")
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
                    if mode == "pressure":
                        raise ValueError(f"Load {i + 1}: pressure acts on a face, not a point")
                    reg = f"pt_ld_{i}"
                    lines.append(f"region   {reg} nearest {px:.4g}, {py:.4g}, {pz:.4g}")
                    if mode == "force":
                        lines.append(f"force    {reg} {ld.get('vec', '0, -1e5, 0')}")
                    elif mode == "moment":
                        if state.geom_mode == "box":
                            freg = f"face_ld_{i}_{_safe_region_name(face, 'face')}"
                            lines.append(f"region   {freg} facets {_box_face_condition(face)}")
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

        # Watchers fired by these changes run later; the editing_load_index and
        # live_load_ready guards make them no-ops.
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
        if S.live_response is None or not bool(state.live_load_ready):
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
            S.scalars = viz.compute_scalars(result, model.material)
            _check_limit(model.material)
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

    # ------------------------------------------------------------------ solve
    def _ui_wait(ui, fn):
        """From a worker: run ``fn`` on the UI thread, wait for it and return its result."""
        box, done = {}, threading.Event()

        def call():
            """Run fn, keeping its result or exception for the waiting worker."""
            try:
                box["value"] = fn()
            except BaseException as exc:  # re-raised in the worker below
                box["error"] = exc
            finally:
                done.set()
        ui(call)
        done.wait()
        if "error" in box:
            raise box["error"]
        return box.get("value")

    def _cad_spec_for_mesh(mesh):
        """UI thread: point face choices at a freshly meshed STEP/mask model and rebuild the spec."""
        points = np.asarray(mesh.p, dtype=float)
        lo, hi = points.min(axis=1), points.max(axis=1)
        bounds = (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
        state.model_bounds = list(bounds)
        _sync_face_options(mesh, bounds)
        _rebuild_spec_from_interactive(mesh_ref_override=mesh)
        return str(state.spec_text)

    def _geometry_text(text):
        """The geometry-defining lines (param / mesh / material) of a spec text."""
        keep = {"param", "mesh", "material"}
        return "\n".join(line for line in text.splitlines()
                         if not line.strip() or line.strip().startswith("#")
                         or line.strip().split(None, 1)[0].lower() in keep) + "\n"

    FREE_BODY_COARSEN_TRIES = 4

    def _coarsen_for_free_body(mesh, n_dofs, spec_text):
        """UI thread: ask for a coarser mesh so a free-body model fits the direct solver.

        The target is the comfortable direct-solve size (``DIRECT_SOLVE_MAX_DOFS``), not just
        under the hard limit: the indefinite free-body solve grows ~ n^1.9 (sphere: 21k DOFs
        24 s, 45k 98 s, ~100k still running after 10 min at 7 GB). DOFs ~ h^-3 for the
        quadratic basis, so element size grows by (n / 0.9 target)^(1/3). Interactive mode
        changes the builder's own fields (mesh size / box counts) so the rebuilt spec stays in
        sync; script mode rewrites the mesh line. Returns (spec_text, geometry_text or None, note).
        """
        target = 0.9 * min(DIRECT_SOLVE_MAX_DOFS, FREE_BODY_MAX_DOFS)
        factor = (n_dofs / target) ** (1.0 / 3.0)
        if state.config_mode != "interactive":
            text = coarsen_mesh_text(spec_text, factor, mesh)
            state.spec_text = text
            return text, None, f"mesh line now: {next(l for l in text.splitlines() if l.strip().lower().startswith('mesh'))}"
        if state.geom_mode == "box":
            old = (int(state.box_nx), int(state.box_ny), int(state.box_nz))
            new = [max(1, int(n / factor)) for n in old]
            state.box_nx, state.box_ny, state.box_nz = new
            note = f"elements {old[0]}×{old[1]}×{old[2]} → {new[0]}×{new[1]}×{new[2]}"
        else:
            from mos.core.meshes import mean_edge_length
            old = float(state.cad_mesh_size) if state.cad_mesh_size else mean_edge_length(mesh)
            state.cad_mesh_size = float(f"{old * factor:.3g}")
            note = f"mesh size {old:.3g} → {state.cad_mesh_size:.3g} m"
        _rebuild_spec_from_interactive()
        text = str(state.spec_text)
        return text, (_geometry_text(text) if state.geom_mode in ("step", "cv") else None), note

    def run_solve(on_done=None):
        """Parse the active specification, solve the finite element model, and update views.

        Meshing and solving run in a worker thread (``_spawn``) so the UI stays responsive;
        the result is applied on the UI thread. ``on_done(ok)`` is called there afterwards.
        A second Solve while one is running is refused.
        """
        if bool(getattr(state, "solve_running", False)):
            state.status = "A solve is already running; wait for it to finish."
            state.status_type = "warning"
            return
        try:
            state.status = "Solving finite element model …"
            state.status_type = "info"
            state.flush()
            dropped, geometry_text = 0, None
            if state.config_mode == "interactive":
                dropped = _drop_stale_face_refs()
                _rebuild_spec_from_interactive()
                if state.geom_mode in ("step", "cv"):
                    # The new CAD/mask mesh decides which named faces exist, so supports and
                    # loads are generated after meshing (on the UI thread, see the worker).
                    geometry_text = _geometry_text(state.spec_text)
            spec_text = str(state.spec_text)
        except Exception as e:
            _solve_failed(e, on_done)
            return
        state.solve_running = True

        def _worker(ui):
            if S.cancel_event.is_set():
                ui(lambda: state.update({"solve_running": False}))
                return
            try:
                cur_text, cur_geometry, coarsened, last_n = spec_text, geometry_text, [], None
                for attempt in range(FREE_BODY_COARSEN_TRIES + 1):
                    text, mesh = cur_text, None
                    if cur_geometry is not None:
                        mesh = build(parse_spec(cur_geometry))[0].mesh       # Gmsh: slow
                        text = _ui_wait(ui, lambda: _cad_spec_for_mesh(mesh))
                    if S.cancel_event.is_set():
                        ui(lambda: state.update({"solve_running": False}))
                        return
                    spec = parse_spec(text)
                    model, case = build(spec, mesh=mesh) if mesh is not None else build(spec)
                    if S.cancel_event.is_set():
                        ui(lambda: state.update({"solve_running": False}))
                        return
                    n = count_dofs(model.mesh)
                    if model.bcs or n <= FREE_BODY_MAX_DOFS:
                        break
                    # A free body needs the indefinite direct solve: coarsen instead of failing.
                    if attempt == FREE_BODY_COARSEN_TRIES or (last_n is not None and n > 0.95 * last_n):
                        raise ValueError(
                            f"free-body model has {n:,} DOFs (limit {FREE_BODY_MAX_DOFS:,}) and "
                            f"automatic coarsening stopped reducing it ({len(coarsened)} attempt(s)); "
                            "use a coarser mesh or add supports")
                    m_now, n_now, t_now = model.mesh, n, cur_text
                    cur_text, cur_geometry, note = _ui_wait(
                        ui, lambda: _coarsen_for_free_body(m_now, n_now, t_now))
                    coarsened.append((n, note))
                    last_n = n
                if coarsened:
                    coarse_note = (f"Free-body mesh coarsened automatically: {coarsened[0][0]:,} → {n:,} "
                                   f"DOFs (the free-body direct solve allows {FREE_BODY_MAX_DOFS:,}; "
                                   f"~{DIRECT_SOLVE_MAX_DOFS // 1000}k solves in about a minute); "
                                   + "; ".join(note for _, note in coarsened) + ". ")
                    log.info(coarse_note)
                else:
                    coarse_note = ""
                cached, res, live_status = None, None, None
                try:
                    if model.bcs:
                        # Assemble and factorise once; reused for live-load responses.
                        cached = CachedLinearSolver(model)
                        res = cached.solve(case)
                    else:
                        res = solve_case(model, case)   # free-body inertial-frame path
                    live_status = (
                        "Stiffness matrix and factorisation cached. Select a load to prepare "
                        "unit-load response fields for live exploration."
                        if cached is not None else
                        "Free-body analysis uses the standard solve path; live load scaling is unavailable.")
                except (ValueError, RuntimeError, TypeError, np.linalg.LinAlgError) as cache_exc:
                    # Fall back to the regular solver for setups the cache can't factorise.
                    cached, res = None, solve_case(model, case)
                    live_status = (f"Live response cache unavailable: {cache_exc}. The model was solved "
                                   "normally; edit loads and use Solve to update the result.")
            except Exception as e:
                if S.cancel_event.is_set():
                    ui(lambda: state.update({"solve_running": False, "status": "Solve cancelled."}))
                    return
                traceback.print_exc()
                error = e
                ui(lambda: _solve_failed(error, on_done))
                return
            if S.cancel_event.is_set():
                ui(lambda: state.update({"solve_running": False}))
                return
            ui(lambda: _apply_solution(spec, text, model, case, res, cached, live_status,
                                       dropped, on_done, coarse_note))

        _spawn(_worker, "mos-solve")

    def _solve_failed(e, on_done=None):
        """UI thread: report a failed solve and keep the view consistent with the last result."""
        state.solve_running = False
        state.status = f"{type(e).__name__}: {e}"
        state.status_type = "error"
        if S.grid is not None and S.model is not None:
            # Meshing a new geometry may have switched bounds/faces before the failure;
            # the viewport still shows the last solved model, so they must match it.
            state.model_bounds = list(S.grid.bounds)
            _sync_face_options(S.model.mesh, S.grid.bounds)
            state.analysis_result_warning = "The last Solve failed; results are from the previous model."
        if on_done:
            on_done(False)

    LIMIT_MARKERS_MAX = 2000   # red circles drawn at most (highest utilisation first)

    def _check_limit(material):
        """Flag nodes at or beyond the proportional limit; open the warning dialog if any.

        Reads the utilisation field of ``S.scalars`` (element-vertex nodal stresses). Linear
        elasticity makes U proportional to the loads, so the result is valid up to
        1 / max(U) times the applied loads.
        """
        S.limit_nodes = None
        state.update({"limit_count": 0, "limit_text": "", "limit_dialog": False})
        if viz.UTILISATION not in S.scalars:
            return
        U = S.scalars[viz.UTILISATION][0]
        u_max = float(np.max(U))
        hit = np.flatnonzero(U >= 1.0)
        if not hit.size:
            return
        S.limit_nodes = hit[np.argsort(-U[hit])][:LIMIT_MARKERS_MAX]
        div = viz.STRESS_UNITS[state.stress_unit]
        if material.sigma_pl_c is None:
            what = (f"peak von Mises {u_max * material.sigma_pl / div:.4g} {state.stress_unit} vs "
                    f"σ_pl = {material.sigma_pl / div:.4g} {state.stress_unit}")
        else:
            what = (f"peak Coulomb–Mohr utilisation {u_max:.3g} (tension limit "
                    f"{material.sigma_pl / div:.4g}, compression limit {material.sigma_pl_c / div:.4g} "
                    f"{state.stress_unit})")
        capped = (f" Only the {LIMIT_MARKERS_MAX} most stressed are circled."
                  if hit.size > LIMIT_MARKERS_MAX else "")
        text = (f"{hit.size} node(s) have reached the proportional limit ({what}). The FEM "
                f"calculation is linear-elastic and is not valid beyond this point: the linear "
                f"result holds only up to {1.0 / u_max:.3g} × the applied loads.{capped} "
                "Very local peaks at clamped corners or point loads depend on the mesh.")
        state.update({"limit_count": int(hit.size), "limit_text": text, "limit_dialog": True})
        state.analysis_result_warning = _limit_warning()

    def _limit_warning():
        """Dock warning for a result with nodes beyond the proportional limit ('' if none)."""
        n = int(state.limit_count or 0)
        return (f"Proportional limit exceeded at {n} node(s): linear FEM results are not valid "
                "at this load." if n else "")

    def _apply_solution(spec, text, model, case, res, cached, live_status, dropped, on_done,
                        coarse_note=""):
        """UI thread: install a finished solve and refresh every view."""
        try:
            S.cached_solver = cached
            S.live_response = None
            state.update({
                "editing_load_index": -1,
                "live_load_selection": "",
                "live_load_ready": False,
                "live_load_status": "Select an applied load to prepare its live response.",
                "live_response_status": live_status,
            })
            S.res, S.grid = res, viz.to_pyvista(model.mesh)
            S.spec, S.model, S.case = spec, model, case
            S.solved_spec_text = text
            state.analysis_result_warning = ""
            S.cached_geometry_signature = _geometry_signature(spec)
            S.scalars = viz.compute_scalars(res, model.material)
            _check_limit(model.material)
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
            if dropped:
                state.status = (f"Removed {dropped} support(s)/load(s) on the previous model's faces; "
                                f"re-select them. {state.status}")
                state.status_type = "warning"
            if coarse_note:
                state.status = coarse_note + state.status
                state.status_type = "warning"
            state.cv3d_preview_active = False   # the new result replaces any preview

            _update_preview_snippet()
            update_scene(reset_camera=True)
            # Refresh the read-only Lab snapshot from the newly solved result, then
            # redraw once so a principal-plane overlay follows the refreshed tensor.
            _lab_stress_from_result()
            if bool(getattr(state, "analysis_dock_open", False)):
                update_scene(reset_camera=False)
        except Exception as e:
            traceback.print_exc()
            _solve_failed(e, on_done)
            return
        state.solve_running = False
        if on_done:
            on_done(state.status_type != "error")

    ctrl.run_solve = run_solve

    # ------------------------------------------------------------------ load/support actions
    def _report_error(e):
        log.warning("%s: %s", type(e).__name__, e)
        state.status = f"{type(e).__name__}: {e}"
        state.status_type = "error"

    def apply_current_load():
        """Capture current load config into loads_list, rebuild spec, and solve."""
        try:
            if state.load_target != "face" and state.load_mode == "pressure":
                raise ValueError("Pressure acts on a face: choose Surface / Face as the target")
            if state.load_mode == "force":
                _check_vec3_text(state.force_vec, "Force vector")
            elif state.load_mode == "pressure" and not str(state.pressure_val or "").strip():
                raise ValueError("enter a pressure value (Pa)")   # an expression is fine
            bounds = tuple(state.model_bounds)
            mesh_ref = S.model.mesh if S.model else None
            mv = _get_moment_vector(state, bounds, mesh_ref) if state.load_mode == "moment" else None
            _update_preview_snippet()
        except ValueError as e:
            _report_error(e)
            return

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

    def _script_support_lines(face, comps):
        """Support lines appended to a hand-written script (numeric bounds, unique region)."""
        mesh_ref = S.model.mesh if S.model else None
        face = _resolved_face_name(face, mesh_ref)
        if face.startswith("cad_face_"):
            return [f"support  {face} {comps}"]
        bounds = S.grid.bounds if S.grid else tuple(state.model_bounds)
        reg = _unique_region_name(state.spec_text, f"sup_{_safe_region_name(face, 'face')}")
        return [f"region   {reg} facets {_face_condition(face, bounds)}", f"support  {reg} {comps}"]

    def _script_mode_edit_refused():
        """Removing list entries can't remove the matching script lines; say so instead."""
        if state.config_mode != "script":
            return False
        state.status = "Script mode: delete the support/load lines in the script, then Solve."
        state.status_type = "warning"
        return True

    def add_support_to_list():
        """Add the currently selected boundary support to the model."""
        new_sup = {"face": state.support_face, "comps": state.support_comps}
        try:
            if state.config_mode == "script":
                lines = _script_support_lines(new_sup["face"], new_sup["comps"])
                state.spec_text = state.spec_text.rstrip() + "\n" + "\n".join(lines) + "\n"
            state.supports_list = list(state.supports_list) + [new_sup]
            if state.config_mode != "script":
                _rebuild_spec_from_interactive()
        except ValueError as e:
            _report_error(e)
            return
        run_solve()

    def remove_support(idx):
        """Remove a boundary support constraint by index."""
        if _script_mode_edit_refused():
            return
        sups = list(state.supports_list)
        if 0 <= idx < len(sups):
            sups.pop(idx)
            state.supports_list = sups
            _rebuild_spec_from_interactive()
            run_solve()

    def remove_load(idx):
        """Remove an applied load by index."""
        if _script_mode_edit_refused():
            return
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
        if _script_mode_edit_refused():
            return
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
            in_p = _project_path(state.image_path.strip() or "cad_models/sample_drawing.png")
            out_p = GENERATED_DIR / "drawing_extruded.step"   # never the bundled example
            th = float(state.image_thickness)
            sx = float(state.image_scale_x)
            sy = float(state.image_scale_y)
            image_to_step(in_p, out_p, thickness=th, scale_x=sx, scale_y=sy)
            state.cad_preset, state.cad_custom_path = "custom", str(out_p)
            state.geom_mode = "step"
            state.image_status = f"Converted {Path(in_p).name} -> drawing_extruded.step"
            if state.config_mode == "interactive":
                _rebuild_spec_from_interactive()
            run_solve()
        except Exception as e:
            traceback.print_exc()
            state.image_status = f"Conversion error: {e}"

    def run_cv_analysis():
        """Analyze a new photo (in the background): guess the part, measure it, show the mask."""
        if bool(getattr(state, "cv_running", False)):
            return
        _clear_cv_result()
        state.cv_points, state.cv_candidate = [], None
        state.mv3_active = -1
        try:
            if state.cv_source_mode == "webcam":
                if not state.cv_image_b64:
                    raise ValueError("snap a photo with the camera first")
                img_source = state.cv_image_b64
            elif state.cv_source_mode == "upload":
                img_source = str(state.cv_custom_path or "").strip()
                if not img_source:
                    raise ValueError("enter the image file path first")
            else:
                img_source = str(_project_path(state.cv_preset_path or "cad_models/cv_samples/bracket_photo.png"))
            ref_scale = float(state.cv_ref_scale or 0.30)
            if not np.isfinite(ref_scale) or ref_scale <= 0:
                raise ValueError("the assumed largest dimension must be a positive length")
        except (TypeError, ValueError) as e:
            state.cv_status = f"CV analysis failed: {e}"
            return

        S.cv_source = (img_source, ref_scale)
        _run_cv("Analyzing photo… (the first cluttered photo also loads the detection models)")

    def _run_cv(message, points=None, labels=None, candidate=None):
        """Segment + measure ``S.cv_source`` in a worker, with optional clicks / mask choice."""
        img_source, ref_scale = S.cv_source
        state.update({"cv_running": True, "cv_status": message, "cv_mask_accepted": False})

        def _worker(ui):
            t0 = time.perf_counter()
            if S.cancel_event.is_set():
                ui(lambda: state.update({"cv_running": False}))
                return
            try:
                res = analyze_image_object(img_source, reference_scale=ref_scale,
                                           points=points, labels=labels, candidate=candidate)
            except Exception as e:
                if S.cancel_event.is_set():
                    ui(lambda: state.update({"cv_running": False, "cv_status": "CV analysis cancelled."}))
                    return
                traceback.print_exc()
                message = f"CV analysis failed: {e}"
                ui(lambda: state.update({"cv_running": False, "cv_status": message}))
                return
            if S.cancel_event.is_set():
                ui(lambda: state.update({"cv_running": False}))
                return
            elapsed = time.perf_counter() - t0

            def apply():
                """Show the result on the UI thread; never leave the Run button disabled."""
                try:
                    _apply_cv_result(res, elapsed)
                except Exception as e:
                    log.exception("applying the CV result failed")
                    _clear_cv_result()
                    state.cv_status = f"CV analysis failed while reading the result: {e}"
                finally:
                    state.cv_running = False
            ui(apply)

        _spawn(_worker, "mos-cv-analysis")

    def _clear_cv_result():
        """Forget the previous photo's mask, category and preview (they belong to another image)."""
        S.cv_mask = None
        state.update({"cv_detected_type": "", "cv_friendly_name": "", "cv_confidence_pct": None,
                      "cv_features_text": "", "cv_annotated_preview": "", "cv_mask_ready": False,
                      "cv_mask_accepted": False})

    # --- choosing the object: clicks, nested masks, acceptance -------------------------
    def _prompts():
        """The user's clicks, or (before any) the automatic guess, as points + labels."""
        pts = list(state.cv_points) or [list(p) for p in S.cv_auto_points]
        return [(p[0], p[1]) for p in pts], [int(p[2]) for p in pts]

    @ctrl.trigger("cv_click")
    def cv_click(ox, oy, cw, ch, nw, nh, shift=False):
        """A click on the preview: mark the part (or, in Remove mode / shift, not the part)."""
        if S.cv_source is None or bool(state.cv_running):
            return
        xy = _click_to_image(float(ox), float(oy), float(cw), float(ch), float(nw), float(nh))
        if xy is None:
            return
        positive = (state.cv_click_mode == "add") != bool(shift)
        if not state.cv_points and not positive:
            state.cv_status = "First click on the part itself; then mark what is not the part."
            return
        base = list(state.cv_points)
        if not base and S.cv_auto_bbox is not None:
            # First click near the guessed object extends it (a missed piece); a click far
            # away means the guess was the wrong object, so the click replaces it.
            x0, y0, x1, y1 = S.cv_auto_bbox
            gx, gy = 0.25 * (x1 - x0), 0.25 * (y1 - y0)
            if x0 - gx <= xy[0] <= x1 + gx and y0 - gy <= xy[1] <= y1 + gy:
                base = [list(p) for p in S.cv_auto_points]
        state.cv_points = base + [[round(xy[0], 1), round(xy[1], 1), int(positive)]]
        state.cv_candidate = None
        pts, labels = _prompts()
        _run_cv("Updating the mask from your clicks…", pts, labels)

    def cv_mask_step(step):
        """Show the next smaller (-1) or larger (+1) of SAM's nested masks for the same clicks."""
        if S.cv_source is None or bool(state.cv_running) or not state.cv_mask_ready:
            return
        n = int(state.cv_n_candidates or 1)
        cur = state.cv_candidate if state.cv_candidate is not None else n - 1
        new = min(max(int(cur) + int(step), 0), n - 1)
        if new == cur:
            return
        state.cv_candidate = new
        pts, labels = _prompts()
        _run_cv("Switching mask…", pts, labels, new)

    def cv_clear_clicks():
        """Drop the clicks and go back to the automatic guess."""
        if S.cv_source is None or bool(state.cv_running):
            return
        state.cv_points, state.cv_candidate = [], None
        _run_cv("Back to the automatic guess…")

    def accept_cv_mask():
        """Confirm that the outlined object is the part; meshing is allowed only after this."""
        if not state.cv_mask_ready:
            state.cv_status = "Run CV Model on a photo first."
            return
        state.cv_mask_accepted = True
        state.cv_status = "Mask accepted: this outline will be meshed."
        k = int(state.mv3_active)
        if 0 <= k < len(state.mv3_mask_ok):
            S.mv3_masks[k] = S.cv_mask.copy()
            ok = list(state.mv3_mask_ok)
            ok[k] = True
            state.mv3_mask_ok = ok
            _mv_outline(k)
            state.cv_status = f"Part accepted in {_mv_label(k)}."

    ctrl.cv_mask_smaller = lambda: cv_mask_step(-1)
    ctrl.cv_mask_larger = lambda: cv_mask_step(+1)
    ctrl.cv_clear_clicks = cv_clear_clicks
    ctrl.accept_cv_mask = accept_cv_mask

    # --- three-photo solid (mos.core.multiview_hull) ------------------------------------
    def _mv_label(k):
        """Human name of photo k."""
        return f"photo {k + 1} ({ORTHO_VIEWS[k]['axis']})"

    def _mv_view(k):
        """The OrthoView of accepted photo k, with the user's flip / rotation."""
        from mos.core.multiview_hull import OrthoView
        return OrthoView(ORTHO_VIEWS[k]["axis"], S.mv3_masks[k], ORTHO_VIEWS[k]["axis"],
                         bool(state.mv3_flip[k]), int(state.mv3_rot[k]))

    def _mv_outline(k):
        """Thumbnail of photo k's accepted outline as it will be used (oriented, cropped)."""
        from mos.core.multiview_hull import crop_part
        outlines = list(state.mv3_outline)
        if k not in S.mv3_masks:
            outlines[k] = ""
        else:
            crop = crop_part(_mv_view(k).oriented())
            part = np.pad(crop, max(4, int(0.08 * max(crop.shape))))   # margin: the outline shows
            img = Image.fromarray(np.where(part[..., None], [37, 99, 235], [226, 232, 240]).astype(np.uint8))
            img.thumbnail((160, 120))
            buf = io.BytesIO()
            img.save(buf, "PNG")
            outlines[k] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        state.mv3_outline = outlines

    def mv3_orient(k, what):
        """Mirror ('flip') or turn ('rot', 90° counter-clockwise) photo k's outline."""
        k = int(k)
        key = {"flip": "mv3_flip", "rot": "mv3_rot"}[what]
        vals = list(getattr(state, key))
        vals[k] = (not vals[k]) if what == "flip" else (int(vals[k]) + 1) % 4
        setattr(state, key, vals)
        _mv_outline(k)

    def mv3_select(k):
        """Load photo k into the preview above to click-select and accept the part in it."""
        k = int(k)
        src = getattr(state, f"mv3_img_{k}")
        if not src:
            state.mv3_msg = f"{_mv_label(k).capitalize()} is missing."
            return
        _clear_cv_result()
        state.mv3_active = k
        state.cv_points, state.cv_candidate = [], None
        S.cv_source = (src, float(state.cv_ref_scale or 0.3))
        _run_cv(f"Finding the part in {_mv_label(k)}…")

    def mv3_build_and_solve():
        """Carve the solid from the three accepted outlines, report its size, mesh + solve."""
        if not all(state.mv3_mask_ok):
            state.mv3_msg = "Select and accept the part in all three photos first."
            return
        try:
            known_m = float(state.mv3_known_mm) / 1000.0
            if not (np.isfinite(known_m) and known_m > 0):
                raise ValueError
        except (TypeError, ValueError):
            state.mv3_msg = "Enter one measured dimension of the part (mm) for the scale."
            return
        if bool(state.mv3_running):
            return
        from mos.core.multiview_hull import SIZE_MISMATCH_WARN, carve_ortho, save_session
        views = [_mv_view(k) for k in range(len(ORTHO_VIEWS))]
        axis = str(state.mv3_known_axis)
        session = GENERATED_DIR / "hull_sessions" / time.strftime("%Y%m%d_%H%M%S")
        state.update({"mv3_running": True, "mv3_msg": "Carving the solid from 3 photos…"})

        def _worker(ui):
            try:
                hull = carve_ortho(views, axis, known_m)
                save_session(session, views, axis, known_m)
                err = None
            except Exception as exc:
                hull, err = None, exc

            def apply():
                """Show the solid's size and the photos' agreement, then mesh + solve it."""
                state.mv3_running = False
                if err is not None:
                    state.mv3_msg = f"Could not build the solid: {err}"
                    return
                ext = hull.extent() * 1000
                warn = (f" The photos disagree on the proportions by {hull.size_mismatch:.0%}: a "
                        "hand or a second object in one outline, a tilted photo, or a wrong "
                        "flip/rotation. Check the outlines." if hull.size_mismatch > SIZE_MISMATCH_WARN
                        else "")
                state.mv3_msg = (f"Solid {ext[0]:.1f} × {ext[1]:.1f} × {ext[2]:.1f} mm "
                                 f"(length × depth × height), {hull.volume * 1e6:.2f} cm³; "
                                 f"proportions agree within {hull.size_mismatch:.1%}.{warn}")
                state.mv3_session = str(session)
                state.cad_mesh_size = round(float(np.max(hull.extent())) / 15.0, 4)
                state.update({"geom_mode": "cv", "cv_mesh_source": "hull", "config_mode": "interactive"})
                _rebuild_spec_from_interactive()
                run_solve()
            ui(apply)
        _spawn(_worker, "mos-mv3-hull")

    ctrl.mv3_select = mv3_select
    ctrl.mv3_orient = mv3_orient
    ctrl.mv3_build_and_solve = mv3_build_and_solve

    @state.change("mv3_img_0", "mv3_img_1", "mv3_img_2")
    def _mv3_photo_changed(**_):
        """A replaced photo loses its accepted outline."""
        sig = tuple(hash(str(getattr(state, f"mv3_img_{k}"))) for k in range(len(ORTHO_VIEWS)))
        old = S.mv3_sig or sig
        ok = list(state.mv3_mask_ok)
        for k, (a, b) in enumerate(zip(old, sig)):
            if a != b:
                ok[k] = False
                S.mv3_masks.pop(k, None)
                _mv_outline(k)
        S.mv3_sig = sig
        state.mv3_mask_ok = ok

    def _apply_cv_result(res, elapsed):
        """Copy a CVAnalysisResult into the UI state (runs on the UI thread)."""
        S.cv_mask = res.mask
        if not state.cv_points:   # remember the automatic guess (prompt + extent of its mask)
            S.cv_auto_points = [list(p) for p in res.prompt_points]
            ys, xs = np.nonzero(res.mask) if res.mask is not None else ([], [])
            S.cv_auto_bbox = (xs.min(), ys.min(), xs.max(), ys.max()) if len(xs) else None
        state.cv_candidate = res.candidate_index
        state.cv_n_candidates = res.n_candidates
        state.cv_mask_accepted = False
        state.cv_detected_type = res.object_type
        state.cv_friendly_name = res.friendly_name
        state.cv_confidence_pct = int(res.confidence * 100)
        state.cv_features_text = "\n".join(f"• {f}" for f in res.detected_features)
        state.cv_annotated_preview = res.annotated_image_b64
        state.cv_mask_ready = res.mask is not None and bool(np.any(res.mask))
        p = res.parameters
        # Depth normal to the photo for direct outline meshing: a cross-section seen
        # end-on (I-beam, tube) extrudes by the member length, a plate by its thickness,
        # side views (bracket, bar, shaft) by their out-of-plane width.
        depth_key = {"ibeam": "length", "pipe": "length", "plate_hole": "thickness",
                     "bracket": "width", "beam": "width", "shaft": "diameter_main"}.get(res.object_type)
        state.cv_mask_thickness = round(float(p.get(depth_key, 0.1 * float(state.cv_ref_scale or 0.3))), 4)
        state.cv_mask_size = float(state.cv_ref_scale or 0.3)
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
        state.cv_running = False
        state.cv_status = (
            f"{res.friendly_name} · {int(res.confidence * 100)}% confidence · "
            f"{image_w} × {image_h} px · {elapsed:.1f} s. Dimensions use your assumed reference scale; "
            "review the mask, then mesh the outline directly or edit the template parameters."
        )

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

    @state.change("cv_mv_method")
    def _cv_multiview_method_changed(**_):
        """Regenerate the CLI commands for the selected NeRF / Gaussian representation."""
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
            output_root = GENERATED_DIR / "reconstruction_sessions"
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

        def _worker(ui):
            updates = {}
            try:
                result = run_nerfstudio_preprocess(images_dir, output_dir)
                updates.update({
                    "cv_mv_preprocess_ready": True,
                    "cv_mv_preprocess_status": (
                        f"Camera poses ready · transforms.json saved to {result['transforms_path']}"
                    ),
                    "cv_mv_preprocess_log": result.get("log_tail", ""),
                })
            except Exception as exc:
                updates.update({
                    "cv_mv_preprocess_ready": False,
                    "cv_mv_preprocess_status": f"Reconstruction failed: {exc}",
                })
            finally:
                updates["cv_mv_running"] = False
                ui(lambda: state.update(updates))

        _spawn(_worker, "mos-multiview-preprocess")

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

        def _worker(ui):
            updates = {}
            try:
                result = run_colmap_sparse_reconstruction(images_dir)
                updates.update({
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
                updates.update({
                    "cv_mv_colmap_ready": False,
                    "cv_mv_colmap_status": f"COLMAP reconstruction failed: {exc}",
                })
            finally:
                updates["cv_mv_colmap_running"] = False
                ui(lambda: state.update(updates))

        _spawn(_worker, "mos-colmap-sparse")

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
            "cv3d_mask_path": "",
            "cv3d_depth_path": "",
            "cv3d_warning_text": "",
            "cv3d_status": "Preparing foreground mask and reconstructing the 3D preview…",
        })
        update_scene(reset_camera=False)
        output_root = GENERATED_DIR / "image_reconstruction"

        def _worker(ui):
            updates = {}
            if S.cancel_event.is_set():
                updates["cv3d_status"] = "Reconstruction cancelled."
                updates["cv3d_running"] = False
                ui(lambda: state.update(updates))
                return
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
                if S.cancel_event.is_set():
                    updates["cv3d_status"] = "Reconstruction cancelled."
                    updates["cv3d_running"] = False
                    ui(lambda: state.update(updates))
                    return
                step_path = str(result.step_path) if result.step_path else ""
                summary = (
                    f"Preview ready · {result.vertex_count:,} vertices · {result.face_count:,} faces\n"
                    f"Foreground: {result.segmentation_method}\n"
                    f"Depth: {result.depth_method}\n"
                    f"Approximate preview envelope: {result.width_m:.3g} × {result.height_m:.3g} × "
                    f"{result.thickness_m + (settings['relief_depth_m'] if settings['geometry_mode'] == 'depth_relief' else 0.0):.3g} m"
                )
                updates.update({
                    "cv3d_preview_path": str(result.preview_mesh_path),
                    "cv3d_mask_path": str(result.mask_path),
                    "cv3d_depth_path": str(result.depth_preview_path) if result.depth_preview_path else "",
                    "cv3d_step_path": step_path,
                    "cv3d_warning_text": "\n".join(f"• {warning}" for warning in result.warnings),
                    "cv3d_status": summary + ("\nSilhouette STEP is ready to load into CAD/FEM." if step_path else "\nPreview only: STEP export needs a working Gmsh installation."),
                })
            except Exception as exc:
                traceback.print_exc()
                updates.update({
                    "cv3d_preview_path": "",
                    "cv3d_step_path": "",
                    "cv3d_status": f"Reconstruction failed: {type(exc).__name__}: {exc}",
                })
            finally:
                updates["cv3d_running"] = False
                ui(lambda: state.update(updates))

        _spawn(_worker, "mos-single-image-reconstruction")

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
        state.cv_mesh_source = "step"
        state.geom_mode = "cv"
        state.config_mode = "interactive"
        state.cad_mesh_size = max(0.02, min(0.05, float(state.cv3d_object_width_m) / 10.0))
        state.active_tab = "model"
        _rebuild_spec_from_interactive()

        def done(ok):
            """Report the STEP load outcome in the reconstruction panel."""
            state.cv3d_status = (
                "Silhouette STEP loaded through the existing CAD/FEM importer. Review supports, "
                "loads, and mesh quality before interpreting results." if ok else
                f"STEP was generated but CAD/FEM loading failed: {state.status}")
        run_solve(on_done=done)

    def _load_mask_geometry(mask_path, size_m, thickness_m, status_key):
        """Make a saved mask the active geometry (``mesh mask``) and solve it."""
        state.update({
            "cv_mask_path": str(mask_path), "cv_mask_size": float(size_m),
            "cv_mask_thickness": float(thickness_m), "cv_mesh_source": "mask",
            "geom_mode": "cv", "config_mode": "interactive", "cv3d_preview_active": False,
        })
        S.preview_mesh = None
        # ~20 elements across the part; the STEP slider default is far too coarse here.
        state.cad_mesh_size = round(max(0.005, float(size_m) / 20.0), 4)
        _rebuild_spec_from_interactive()

        def done(ok):
            """Report the outline-mesh outcome in the panel that started it."""
            state.update({status_key: (
                f"Photographed outline meshed directly ({state.kpi_mesh}). Faces: 1 = bottom, 2 = top, "
                "then the side walls and hole walls. Check supports/loads before trusting results."
                if ok else f"Outline mesh could not be solved: {state.status}")})
        run_solve(on_done=done)

    def mesh_cv_outline_and_load():
        """Mesh the analysed photo's mask directly (outline + holes, extruded) and solve."""
        if S.cv_mask is None or not np.any(S.cv_mask):
            state.cv_status = "Run the CV model on a photo first; its mask is what gets meshed."
            return
        if not state.cv_mask_accepted:
            state.cv_status = "Check the outlined object and press Accept mask before meshing."
            return
        try:
            size = float(state.cv_mask_size)
            thickness = float(state.cv_mask_thickness)
            if not (np.isfinite(size) and size > 0 and np.isfinite(thickness) and thickness > 0):
                raise ValueError("size and depth must be positive lengths")
            out = GENERATED_DIR / "cv_meshes"
            mask_path = save_mask_image(S.cv_mask, out / f"cv_mask_{time.strftime('%Y%m%d_%H%M%S')}.png")
        except Exception as exc:
            state.cv_status = f"Could not prepare the outline mesh: {exc}"
            return
        _load_mask_geometry(mask_path, size, thickness, "cv_status")

    def mesh_cv3d_mask_and_load():
        """Mesh the single-image reconstruction's foreground mask directly and solve."""
        mask_path = str(getattr(state, "cv3d_mask_path", "") or "").strip()
        if not mask_path or not Path(mask_path).is_file():
            state.cv3d_status = "Reconstruct an image first; its foreground mask is what gets meshed."
            return
        try:
            mask = load_mask_image(mask_path)
            ys, xs = np.nonzero(mask)
            if xs.size == 0:
                raise ValueError("the mask is empty")
            # The reconstruction width spans the whole image; convert to the part's extent.
            extent_px = max(xs.max() - xs.min(), ys.max() - ys.min()) + 1
            size = float(state.cv3d_object_width_m) * extent_px / max(mask.shape[1], 1)
            thickness = float(state.cv3d_body_thickness_m)
        except Exception as exc:
            state.cv3d_status = f"Could not read the reconstruction mask: {exc}"
            return
        _load_mask_geometry(mask_path, size, thickness, "cv3d_status")

    def generate_cv_cad_and_load():
        """Build parametric 3D STEP solid from detected object and load into CAD engine."""
        try:
            if not state.cv_mask_ready:
                state.cv_status = "Run CV Model on a photo first."
                return
            if not state.cv_mask_accepted:
                state.cv_status = "Check the outlined object and press Accept mask first."
                return
            obj_type = state.cv_detected_type
            if not obj_type:
                state.cv_status = ("The outline matches no template part; use Mesh Photo Outline instead."
                                   if state.cv_mask_ready else "Run CV Model on a photo first.")
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

            step_path = str(state.cv_step_path or "")
            if not step_path or step_path == str(getattr(state, "cv3d_step_path", "")):
                # Never overwrite the silhouette STEP produced by the reconstruction panel.
                step_path = str(GENERATED_DIR / "cv_detected_object.step")
                state.cv_step_path = step_path
            out_step = _project_path(step_path)
            out_step.parent.mkdir(parents=True, exist_ok=True)
            generate_cad_from_parameters(obj_type, params, out_step)

            state.geom_mode = "cv"
            state.cv_mesh_source = "step"
            if state.config_mode == "interactive":
                _rebuild_spec_from_interactive()
            def done(ok):
                """Report the template solid's mesh/solve outcome."""
                state.cv_status = (
                    f"Active: {state.cv_friendly_name} template solid ({out_step.name}) meshed and solved."
                    if ok else f"Generated {out_step.name}, but meshing/solving failed: {state.status}")
            run_solve(on_done=done)
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
    ctrl.mesh_cv_outline_and_load = mesh_cv_outline_and_load
    ctrl.mesh_cv3d_mask_and_load = mesh_cv3d_mask_and_load
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
    @state.change("mat_preset")
    def _set_material_preset(mat_preset, **_):
        if mat_preset in MATERIAL_PRESETS and mat_preset != "Custom":
            info = MATERIAL_PRESETS[mat_preset]
            state.mat_E = f"{info['E']:.4g}"
            state.mat_nu = f"{info['nu']:.4g}"
            state.mat_rho = f"{info['rho']:.4g}"
            state.mat_brittle = info["sigma_pl_c"] is not None
            state.mat_spl = f"{info['sigma_pl']:g}"
            state.mat_splc = "" if info["sigma_pl_c"] is None else f"{info['sigma_pl_c']:g}"
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

    def set_lab_plane(preset_name):
        """Quick preset selector for plane orientation in Learning Lab."""
        state.lab_plane_preset = str(preset_name)
        _lab_plane_changed()

    def snap_probe_to(location):
        """Quick snap of analysis probe point to key geometric features."""
        if S.grid is None:
            return
        bounds = S.grid.bounds
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        cx, cy, cz = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0
        loc = str(location).lower()
        if loc == "center":
            _pick_point_direct(cx, cy, cz)
        elif loc == "top":
            _pick_point_direct(cx, ymax, cz)
        elif loc == "bottom":
            _pick_point_direct(cx, ymin, cz)
        elif loc == "tip":
            _pick_point_direct(xmax, cy, cz)
        elif loc == "root":
            _pick_point_direct(xmin, cy, cz)
        elif loc == "origin":
            _pick_point_direct(0.0, 0.0, 0.0)

    ctrl.set_lab_plane = set_lab_plane
    ctrl.snap_probe_to = snap_probe_to
    ctrl.cancel_current_operation = cancel_current_operation

    # ------------------------------------------------------------------ learning lab helpers
    def _lab_float(name, fallback=0.0):
        """Read a numeric Learning Lab field without letting an empty widget crash the page."""
        try:
            value = float(getattr(state, name))
            return value if np.isfinite(value) else float(fallback)
        except (TypeError, ValueError, AttributeError):
            return float(fallback)

    def _refresh_learning_lab():
        """Recompute the tensor experiment and every linked visual from current inputs.

        trame runs @state.change watchers after the handler that changed the inputs has
        returned, so a "suspend" flag set inside that handler is already cleared. Instead
        the inputs of the last refresh are kept and an identical request is skipped (an
        exact cache-key match on the values the app itself stored, not a numeric test).
        """
        inputs = tuple(_lab_float(k) for k in (
            "lab_sigma_xx", "lab_sigma_yy", "lab_sigma_zz", "lab_tau_xy",
            "lab_tau_yz", "lab_tau_zx", "lab_azimuth", "lab_elevation"))
        if inputs == S.lab_refreshed_inputs:
            return
        S.lab_refreshed_inputs = inputs
        result = analyse_stress_tensor(*inputs)
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
            try:   # an empty/invalid coordinate is an error, not the origin
                xyz = np.array([float(state.lab_point_x), float(state.lab_point_y),
                                float(state.lab_point_z)], dtype=float)
            except (TypeError, ValueError):
                raise ValueError("enter numeric X, Y and Z coordinates for the sample point") from None
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
            if state.lab_plane_preset not in [item["value"] for item in plane_items]:
                state.lab_plane_preset = "Custom"   # a face of a previous model
            tensor_mpa = sample["tensor"] / 1.0e6
            traction = lab_evaluate_traction(S.res, points, xyz, _lab_plane_vector(tensor_mpa))
            # The watchers these changes trigger find the inputs already refreshed and skip.
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
        "analysis_dock_open": False,
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
        "cv_confidence_pct": None,
        "solve_running": False,
        "cv_points": [], "cv_candidate": None, "cv_n_candidates": 0,
        "cv_mask_accepted": False, "cv_click_mode": "add",
        "mv3_img_0": "", "mv3_img_1": "", "mv3_img_2": "",
        "mv3_mask_ok": [False] * 3, "mv3_outline": [""] * 3, "mv3_flip": [False] * 3,
        "mv3_rot": [0] * 3, "mv3_active": -1, "mv3_running": False, "mv3_msg": "",
        "mv3_session": "", "mv3_known_axis": "x", "mv3_known_mm": "",
        "mv3_axis_items": [{"title": "Length x", "value": "x"}, {"title": "Depth y", "value": "y"},
                           {"title": "Height z", "value": "z"}],
        "cv_features_text": "",
        "cv_annotated_preview": "",
        "cv_step_path": str(GENERATED_DIR / "cv_detected_object.step"),
        "cv_running": False,
        "cv_mask_ready": False,
        "cv_mesh_source": "step",
        "cv_mask_path": "",
        "cv_mask_size": 0.30,
        "cv_mask_thickness": 0.02,
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
        # Global busy & cancel state
        "is_busy": False,
        "busy_task_name": "",
        "busy_task_desc": "",
        # Material
        "mat_preset": "Steel",
        "mat_preset_items": list(MATERIAL_PRESETS),
        "mat_E": "210e9", "mat_nu": "0.3", "mat_rho": "7850",
        "mat_spl": "250", "mat_splc": "", "mat_brittle": False,
        "limit_dialog": False, "limit_text": "", "limit_count": 0,
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
    S.geometry_identity = _geometry_identity()

    if args.spec:
        state.spec_text = Path(args.spec).read_text()
        state.config_mode = "script"  # otherwise the first Solve rebuilds it from the builder

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
                  "show_preview_glyph", "active_tab", "support_face", "supports_list")
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
                  "mat_E", "mat_nu", "mat_rho", "mat_spl", "mat_splc", "mat_brittle",
                  "include_gravity")
    def _interactive_param_changed(**_):
        if state.config_mode == "interactive":
            _drop_stale_face_refs()
            try:
                _rebuild_spec_from_interactive()
            except ValueError as e:
                _report_error(e)

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
            state.analysis_result_warning = _limit_warning()

        # Always resample from the latest published Results object. Do this without
        # creating/removing VTK actors or calling view.update().
        _lab_stress_from_result()

    @state.change("live_load_selection")
    def _live_load_selection_changed(live_load_selection, **_):
        if str(live_load_selection).strip() == "":
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
            with html.Div(v_if="is_busy", classes="d-flex align-center ga-2 mr-3 px-3 py-1 rounded-pill",
                          style="background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4);"):
                v3.VProgressCircular(indeterminate=True, size=16, width=2, color="error")
                html.Span("{{ busy_task_name }}: {{ busy_task_desc }}", classes="text-caption font-weight-bold text-error")
                v3.VBtn(
                    "Cancel", prepend_icon="mdi-close-circle",
                    color="error", size="x-small", variant="flat",
                    click=ctrl.cancel_current_operation,
                )
            v3.VSpacer()
            v3.VBtn(
                "Analysis workspace", prepend_icon="mdi-cube-scan",
                color="primary", variant=("analysis_dock_open ? 'text' : 'flat'",),
                density="comfortable", size="small",
                click="active_page = 'workspace'; analysis_dock_open = false; main_drawer = true",
            )
            v3.VBtn(
                "Learning Lab", prepend_icon="mdi-school-outline",
                color="primary", variant=("analysis_dock_open ? 'flat' : 'text'",),
                density="comfortable", size="small",
                click="analysis_dock_open = true",
            )
            v3.VChip("LIVE FEM", color="success", variant="tonal", size="small", classes="ml-1 mr-2")
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
                                    v3.VBtn("Box", value="box", prepend_icon="mdi-cube-outline",
                                            size="small", classes="flex-grow-1")
                                    v3.VBtn("STEP file", value="step", prepend_icon="mdi-file-cad",
                                            size="small", classes="flex-grow-1")
                                    v3.VBtn("Photo (CV)", value="cv", prepend_icon="mdi-camera",
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
                                                   label="Element", density="compact",
                                                   hide_details=True, style="min-width:96px;max-width:96px")

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
                                            id="mos_cam_input", type="file", accept="image/*",
                                            raw_attrs=['capture="environment"'], style="display:none;",
                                            change="window.mosUpload.single($event, {data: 'cv_image_b64', status: 'cv_status'})",
                                        )
                                        v3.VBtn("Snap Photo with Camera", prepend_icon="mdi-camera", color="primary", size="small",
                                                block=True, classes="mb-2", click="window.mosUpload.camera({data: 'cv_image_b64', status: 'cv_status'}, 'mos_cam_input')")
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
                                                loading=("cv_running",), disabled=("cv_running",),
                                                click=ctrl.run_cv_analysis)

                                    # CV Detection results preview
                                    with html.Div(v_show="cv_mask_ready", classes="mt-2"):
                                        with html.Div(classes="d-flex align-center justify-space-between mb-1"):
                                            v3.VChip("{{ cv_friendly_name }}", color="success", size="small", classes="font-weight-bold")
                                            v3.VChip("{{ cv_confidence_pct }}% confidence", color="info", size="x-small", variant="outlined",
                                                     v_show="cv_detected_type")

                                        html.Div(
                                            "Is the green outline your part? Click the part to select it; "
                                            "in Remove mode (or shift-click) mark the hand or background.",
                                            classes="text-caption mb-1")
                                        html.Img(
                                            src=("cv_annotated_preview",),
                                            style="width:100%; height:240px; object-fit:contain; border-radius:12px; background:#111; border:1px solid rgba(148,163,184,.22); cursor:crosshair;",
                                            classes="mb-1",
                                            click="trigger('cv_click', [$event.offsetX, $event.offsetY, $event.target.clientWidth, "
                                                  "$event.target.clientHeight, $event.target.naturalWidth, "
                                                  "$event.target.naturalHeight, $event.shiftKey])",
                                        )
                                        with html.Div(classes="d-flex flex-wrap align-center ga-1 mb-1"):
                                            with v3.VBtnToggle(v_model=("cv_click_mode",), mandatory=True,
                                                               density="compact", divided=True, color="secondary"):
                                                v3.VBtn("Add", value="add", size="x-small", prepend_icon="mdi-plus-circle")
                                                v3.VBtn("Remove", value="remove", size="x-small", prepend_icon="mdi-minus-circle")
                                            v3.VBtn("Smaller", size="x-small", variant="tonal", click=ctrl.cv_mask_smaller,
                                                    disabled=("cv_running || cv_candidate === 0",))
                                            v3.VBtn("Larger", size="x-small", variant="tonal", click=ctrl.cv_mask_larger,
                                                    disabled=("cv_running || cv_candidate === null || cv_candidate >= cv_n_candidates - 1",))
                                            v3.VBtn("Clear clicks", size="x-small", variant="text", click=ctrl.cv_clear_clicks,
                                                    disabled=("cv_running || cv_points.length === 0",))
                                        v3.VBtn("{{ cv_mask_accepted ? 'Mask accepted' : 'Accept mask' }}",
                                                color=("cv_mask_accepted ? 'success' : 'warning'",), size="small", block=True,
                                                prepend_icon=("cv_mask_accepted ? 'mdi-check-circle' : 'mdi-check'",),
                                                click=ctrl.accept_cv_mask, disabled=("cv_running",), classes="mb-2")
                                        html.Div(
                                            "{{ cv_features_text }}",
                                            style="white-space:pre-line; font-size:11px;",
                                            classes="text-medium-emphasis mb-2 pa-2 bg-surface-variant rounded",
                                        )

                                        # Direct mesh of the measured outline (recommended path)
                                        html.Div("Mesh the photographed outline directly (recommended):",
                                                 classes="text-caption font-weight-bold mb-1")
                                        html.Div(
                                            "Meshes exactly the segmented silhouette, holes included, as a prism of the "
                                            "given depth. No template guess, no STEP round trip. For an end-on photo of a "
                                            "beam or tube, the depth is the member length.",
                                            classes="text-caption text-medium-emphasis mb-1",
                                        )
                                        with html.Div(classes="d-flex ga-1 mb-1"):
                                            v3.VTextField(v_model=("cv_mask_size",), label="Largest extent (m)", type="number",
                                                          step=0.01, density="compact", hide_details=True)
                                            v3.VTextField(v_model=("cv_mask_thickness",), label="Depth / extrusion (m)", type="number",
                                                          step=0.005, density="compact", hide_details=True)
                                        v3.VBtn("Mesh Photo Outline & Solve FEA", color="primary", size="small", block=True,
                                                prepend_icon="mdi-vector-polygon", click=ctrl.mesh_cv_outline_and_load,
                                                disabled=("!cv_mask_accepted || cv_running",), classes="mb-3")

                                        # Dynamic Parametric Inputs
                                        html.Div("Or build an idealised template solid from the extracted parameters (meters):",
                                                 classes="text-caption font-weight-bold mb-1")

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
                                                min=0.005, max=0.5, step=0.005,
                                                density="compact", hide_details=True,
                                                classes="flex-grow-1",
                                            )
                                            html.Span("{{ cad_mesh_size }} m",
                                                      classes="text-caption font-weight-bold",
                                                      style="min-width:48px; text-align:right")

                                        v3.VBtn("Generate Template CAD Solid & Solve FEA", color="secondary", variant="tonal", size="small", block=True,
                                                prepend_icon="mdi-cube-outline", click=ctrl.generate_cv_cad_and_load,
                                                disabled=("!cv_mask_accepted || cv_running",), classes="mt-1")

                                    html.Div("{{ cv_status }}", classes="text-caption text-medium-emphasis mt-1")

                                    # Multi-view reconstruction: camera poses -> NeRF / Gaussian splats
                                    with v3.VExpansionPanels(variant="accordion", density="compact", multiple=True, classes="mt-3"):
                                        with v3.VExpansionPanel():
                                            with v3.VExpansionPanelTitle():
                                                html.Span("3-photo solid · real 3D (recommended)",
                                                          classes="text-caption font-weight-bold")
                                            with v3.VExpansionPanelText():
                                                html.Div("One photo cannot show depth or hidden sides. Three "
                                                         "photos along the part's axes give its solid "
                                                         "(intersection of the three outlines).",
                                                         classes="text-caption mb-1")
                                                html.Div(ORTHO_HELP, classes="text-caption text-warning mb-2")
                                                for k, v in enumerate(ORTHO_VIEWS):
                                                    with html.Div(classes="mb-2 pa-2 rounded",
                                                                  style="border:1px solid rgba(148,163,184,.25)"):
                                                        html.Div(f"{k + 1}. {v['title']}", classes="text-caption font-weight-bold")
                                                        html.Input(id=f"mos_mv3_input_{k}", type="file", accept="image/*",
                                                                   style="display:none;",
                                                                   change=f"window.mosUpload.single($event, {{data: 'mv3_img_{k}', status: 'mv3_msg'}}, 25)")
                                                        with html.Div(classes="d-flex flex-wrap align-center ga-1"):
                                                            v3.VBtn("Camera", size="x-small", prepend_icon="mdi-camera",
                                                                    click=f"window.mosUpload.camera({{data: 'mv3_img_{k}', status: 'mv3_msg'}}, 'mos_mv3_input_{k}')")
                                                            v3.VBtn("File", size="x-small", variant="tonal", prepend_icon="mdi-image",
                                                                    click=f"window.mosUpload.pick('mos_mv3_input_{k}')")
                                                            v3.VBtn("Select part", size="x-small", variant="tonal", color="primary",
                                                                    click=(ctrl.mv3_select, f"[{k}]"),
                                                                    disabled=(f"!mv3_img_{k} || cv_running",))
                                                            v3.VIcon("mdi-image-check", color="success", size="small", v_show=f"mv3_img_{k}")
                                                            v3.VChip("part ✓", color="success", size="x-small", v_show=f"mv3_mask_ok[{k}]")
                                                        with html.Div(v_show=f"mv3_outline[{k}]", classes="d-flex align-center ga-2 mt-1"):
                                                            html.Img(src=(f"mv3_outline[{k}]",),
                                                                     style="max-width:160px; max-height:120px; border-radius:4px;")
                                                            with html.Div():
                                                                html.Div(v["axes"], classes="text-caption")
                                                                with html.Div(classes="d-flex ga-1 mt-1"):
                                                                    v3.VBtn("Flip", size="x-small", variant="tonal",
                                                                            prepend_icon="mdi-flip-horizontal",
                                                                            click=(ctrl.mv3_orient, f"[{k}, 'flip']"))
                                                                    v3.VBtn("Rotate 90°", size="x-small", variant="tonal",
                                                                            prepend_icon="mdi-rotate-left",
                                                                            click=(ctrl.mv3_orient, f"[{k}, 'rot']"))
                                                with html.Div(classes="d-flex ga-1 mb-2"):
                                                    v3.VSelect(v_model=("mv3_known_axis",), items=("mv3_axis_items",),
                                                               item_title="title", item_value="value",
                                                               label="Measured dimension", density="compact",
                                                               hide_details=True)
                                                    v3.VTextField(v_model=("mv3_known_mm",), label="Length (mm)",
                                                                  placeholder="e.g. 60", density="compact",
                                                                  hide_details=True)
                                                v3.VBtn("Build solid & solve FEA", block=True, size="small", color="primary",
                                                        prepend_icon="mdi-cube-scan", click=ctrl.mv3_build_and_solve,
                                                        loading=("mv3_running",),
                                                        disabled=("mv3_running || !mv3_mask_ok.every(Boolean) || !mv3_known_mm",))
                                                html.Div("{{ mv3_msg }}", classes="text-caption text-medium-emphasis mt-1",
                                                         style="white-space:pre-line")
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
                                                    change="window.mosUpload.multiview($event)",
                                                )
                                                v3.VBtn(
                                                    "Choose Multi-view Photos", prepend_icon="mdi-image-multiple", color="primary",
                                                    size="small", block=True, classes="mb-1",
                                                    click="window.mosUpload.pick('mos_multiview_input')",
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
                                                    "COLMAP Dataset · 2DGS / Graphdeco",
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
                                                    change="window.mosUpload.single($event, {data: 'cv3d_image_b64', name: 'cv3d_image_name', status: 'cv3d_status'}, 15)",
                                                )
                                                v3.VBtn(
                                                    "Choose Image · JPG / PNG / WEBP", prepend_icon="mdi-image-plus",
                                                    color="primary", size="small", block=True,
                                                    click="window.mosUpload.pick('mos_cv3d_image_input')",
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
                                                with html.Div(v_show="cv3d_mask_path !== ''", classes="mt-1"):
                                                    v3.VBtn(
                                                        "Mesh Foreground Mask Directly & Solve", color="primary",
                                                        size="small", block=True, prepend_icon="mdi-vector-polygon",
                                                        click=ctrl.mesh_cv3d_mask_and_load,
                                                    )
                                                html.Div(
                                                    "Important: the viewer preview uses an estimated front-surface depth and a simplified back. "
                                                    "The STEP export and the direct mask mesh are constant-thickness silhouette extrusions (through-holes kept), not the depth-relief mesh. "
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
                                )
                                with html.Div(classes="d-flex ga-1"):
                                    v3.VTextField(v_model=("mat_E",), label="E (Pa)",
                                                  density="compact", hide_details=True)
                                    v3.VTextField(v_model=("mat_nu",), label="nu (Poisson)",
                                                  density="compact", hide_details=True)
                                    v3.VTextField(v_model=("mat_rho",), label="rho (kg/m3)",
                                                  density="compact", hide_details=True)
                                with html.Div(classes="d-flex ga-1 mt-2 align-center"):
                                    v3.VTextField(v_model=("mat_spl",),
                                                  label=("mat_brittle ? 'Tension limit (MPa)' : "
                                                         "'Proportional limit σ_pl (MPa)'",),
                                                  placeholder="blank = no check",
                                                  density="compact", hide_details=True)
                                    v3.VTextField(v_show="mat_brittle", v_model=("mat_splc",),
                                                  label="Compression limit (MPa)",
                                                  density="compact", hide_details=True)
                                v3.VCheckbox(v_model=("mat_brittle",),
                                             label="Brittle (Coulomb–Mohr: separate tension / compression limits)",
                                             density="compact", hide_details=True)
                                html.Div("Ductile: nodes with von Mises ≥ σ_pl are flagged. Brittle: "
                                         "max(σ1,0)/St + max(−σ3,0)/Sc ≥ 1 is flagged. Beyond this the linear "
                                         "model is not valid.", classes="text-caption text-medium-emphasis")

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
                            loading=("solve_running",), disabled=("solve_running",),
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
                            with html.Div(classes="d-flex align-center justify-space-between mb-1"):
                                with html.Div(classes="d-flex align-center ga-1"):
                                    v3.VIcon("mdi-shield-lock", size="small", color="error")
                                    html.Span("Boundary Supports (Restraints)", classes="text-subtitle-2 font-weight-bold")
                                v3.VChip("{{ supports_list.length }} fixed", size="x-small", color="error")

                            html.Div("Fixed surfaces (shown in red on the 3D model) anchor the object in space.",
                                     classes="text-caption text-medium-emphasis mb-2")

                            with html.Div(v_if="supports_list.length > 0", classes="mb-2"):
                                with v3.VList(density="compact", classes="pa-0 rounded"):
                                    with v3.VListItem(
                                        v_for="(sup, idx) in supports_list",
                                        key="idx", classes="px-2 py-1",
                                        style="border-bottom: 1px solid rgba(128,128,128,0.15);",
                                    ):
                                        with html.Template(v_slot_default=True):
                                            with html.Div(classes="d-flex align-center ga-1"):
                                                html.Span("■", style="color:#ef4444;font-size:14px;")
                                                html.Span("Face {{ sup.face }}", classes="text-caption font-weight-bold")
                                                v3.VChip("{{ sup.comps.toUpperCase() }}", size="x-small", variant="outlined", color="error", classes="ml-1")
                                        with html.Template(v_slot_append=True):
                                            v3.VBtn(
                                                text="Remove", size="x-small", variant="text", color="error",
                                                click=(ctrl.remove_support, "[idx]"),
                                            )

                            with html.Div(v_else=True, classes="text-caption text-medium-emphasis text-center pa-2"):
                                html.Span("No supports applied yet. Without supports, the solver treats the body as free-floating.")

                            v3.VDivider(classes="my-2")
                            html.Div("Select a surface to anchor / clamp:", classes="text-caption font-weight-medium mb-1")
                            with html.Div(classes="d-flex flex-column ga-2 mb-2"):
                                v3.VSelect(
                                    v_model=("support_face",),
                                    items=("face_options",),
                                    item_title="title", item_value="value",
                                    label="Constrained surface (Orange preview in 3D)",
                                    density="compact", hide_details=True,
                                    classes="flex-grow-1",
                                )
                                v3.VSelect(
                                    v_model=("support_comps",),
                                    items=("support_comp_options",),
                                    item_title="title", item_value="value",
                                    label="Restraint type",
                                    density="compact", hide_details=True,
                                    classes="flex-grow-1",
                                )
                            v3.VBtn(
                                "Clamp / Support Selected Surface", block=True, size="small",
                                variant="flat", color="error", prepend_icon="mdi-pin",
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
            with html.Div(style="position:relative;width:100%;height:100%;overflow:hidden;"):
                with html.Div(v_show="active_page === 'workspace'", style="width:100%;height:100%;"):
                    with v3.VContainer(fluid=True, classes="pa-0 fill-height"):
                        with html.Div(style="position: relative; width: 100%; height: 100%; padding: 8px; box-sizing: border-box;"):
                            # 3D Viewport Controls HUD (User Guidance)
                            with html.Div(classes="mos-viewport-hud"):
                                with html.Span(classes="d-flex align-center ga-1"):
                                    v3.VIcon("mdi-mouse", size="x-small")
                                    html.Span("Rotate: Left-drag")
                                with html.Span(classes="d-flex align-center ga-1"):
                                    v3.VIcon("mdi-magnify-plus-outline", size="x-small")
                                    html.Span("Zoom: Right-drag / Scroll")
                                with html.Span(classes="d-flex align-center ga-1"):
                                    v3.VIcon("mdi-hand-back-left", size="x-small")
                                    html.Span("Pan: Shift+Left-drag")
                                with html.Span(classes="d-flex align-center ga-1"):
                                    v3.VIcon("mdi-crosshairs-gps", size="x-small")
                                    html.Span("Inspect: Click model")

                            # 3D Scene Visual Legend HUD
                            with html.Div(classes="mos-viewport-legend"):
                                with html.Span(classes="mos-legend-chip"):
                                    html.Span("■", style="color:#ef4444;font-size:14px;")
                                    html.Span("Fixed Restraint")
                                with html.Span(classes="mos-legend-chip"):
                                    html.Span("■", style="color:#06b6d4;font-size:14px;")
                                    html.Span("Target Face")
                                with html.Span(classes="mos-legend-chip"):
                                    html.Span("▲", style="color:#22c55e;font-size:12px;")
                                    html.Span("Applied Force")
                                with html.Span(classes="mos-legend-chip"):
                                    html.Span("●", style="color:#f59e0b;font-size:12px;")
                                    html.Span("Inspection Probe")
                                with html.Span(v_show="limit_count > 0", classes="mos-legend-chip"):
                                    html.Span("◯", style="color:#dc2626;font-size:14px;font-weight:bold;")
                                    html.Span("Proportional limit reached ({{ limit_count }})")

                            view = plotter_ui(pl, mode="server")
                            ctrl.view_update = view.update

                            with v3.VDialog(v_model=("limit_dialog",), max_width=560):
                                with v3.VCard(classes="pa-2"):
                                    with v3.VCardTitle(classes="d-flex align-center ga-2 text-error"):
                                        v3.VIcon("mdi-alert-octagon", color="error")
                                        html.Span("Proportional limit reached")
                                    v3.VCardText("{{ limit_text }}")
                                    v3.VCardText("The affected nodes are circled in red. Reduce the "
                                                 "loads, change the material or geometry, or treat "
                                                 "the result as invalid beyond the first yield.",
                                                 classes="text-caption")
                                    with v3.VCardActions():
                                        v3.VSpacer()
                                        v3.VBtn("OK", color="primary", variant="flat",
                                                click="limit_dialog = false")

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
                                placeholder="Choose a load to explore its live response",
                                persistent_placeholder=True,
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
                            with html.Div(classes="d-flex justify-space-between align-center mb-1"):
                                html.Div("Inspection Probe Point", classes="dock-label mb-0")
                                v3.VChip("CLICK 3D MODEL TO PICK", color="warning", variant="flat", size="x-small")
                            html.Div("Click anywhere on the 3D model surface to place the inspection probe.",
                                     classes="text-caption text-medium-emphasis mb-2")
                            with html.Div(style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;"):
                                v3.VTextField(v_model=("lab_point_x",), label="X (m)", type="number",
                                              density="compact", hide_details=True, variant="outlined")
                                v3.VTextField(v_model=("lab_point_y",), label="Y (m)", type="number",
                                              density="compact", hide_details=True, variant="outlined")
                                v3.VTextField(v_model=("lab_point_z",), label="Z (m)", type="number",
                                              density="compact", hide_details=True, variant="outlined")
                            with html.Div(classes="d-flex flex-wrap ga-1 mt-2"):
                                v3.VBtn("Center", size="x-small", variant="tonal", classes="probe-preset-btn",
                                        click=(ctrl.snap_probe_to, "['center']"))
                                v3.VBtn("Top (+Y)", size="x-small", variant="tonal", classes="probe-preset-btn",
                                        click=(ctrl.snap_probe_to, "['top']"))
                                v3.VBtn("Tip (+X)", size="x-small", variant="tonal", classes="probe-preset-btn",
                                        click=(ctrl.snap_probe_to, "['tip']"))
                                v3.VBtn("Root (-X)", size="x-small", variant="tonal", classes="probe-preset-btn",
                                        click=(ctrl.snap_probe_to, "['root']"))
                            html.Div("{{ lab_status }}", classes="text-caption text-info mt-2")
                            html.Div("Nearest node {{ lab_node_index }} · {{ lab_point_coords }}",
                                     classes="text-caption text-medium-emphasis")

                        with html.Div(v_if="lab_has_result", classes="dock-section"):
                            html.Div("Stress at selected point", classes="dock-label")
                            with html.Div(style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;"):
                                with html.Div(classes="pa-2 rounded", style="background:rgba(14,165,233,.12)"):
                                    html.Div("von Mises (Yield)", classes="text-caption")
                                    html.Div("{{ lab_fea_vm }}", classes="text-subtitle-2 font-weight-bold")
                                with html.Div(classes="pa-2 rounded", style="background:rgba(139,92,246,.12)"):
                                    html.Div("σ₁ (Max Tension)", classes="text-caption")
                                    html.Div("{{ lab_sigma_1 }} MPa", classes="text-subtitle-2 font-weight-bold")
                                with html.Div(classes="pa-2 rounded", style="background:rgba(245,158,11,.12)"):
                                    html.Div("Max Shear", classes="text-caption")
                                    html.Div("{{ lab_max_shear }}", classes="text-subtitle-2 font-weight-bold")
                            html.Div(v_html=("lab_matrix_html",), style="overflow-x:auto;", classes="mt-3")

                        with html.Div(v_if="!lab_has_result", classes="dock-section"):
                            with v3.VAlert(type="info", variant="tonal", density="compact"):
                                html.Span("Solve a model, then click a point on the mesh to inspect its FEA stress state.")

                        with html.Div(classes="dock-section"):
                            html.Div("Traction on selected plane", classes="dock-label")
                            html.Div("A virtual cut through the material at the probe point. Normal stress (σₙ) acts perpendicular; shear stress (τ) acts parallel along the slice.",
                                     classes="text-caption text-medium-emphasis mb-2")
                            # Visual guide for 3D elements
                            with html.Div(classes="d-flex align-center ga-3 mb-2 px-2 py-1 rounded",
                                          style="background:rgba(15,23,42,.6);border:1px solid rgba(148,163,184,.15);font-size:11px;"):
                                with html.Span(classes="d-flex align-center ga-1"):
                                    html.Span("■", style="color:#0284c7;font-size:13px;")
                                    html.Span("Cut Plane", classes="text-caption")
                                with html.Span(classes="d-flex align-center ga-1"):
                                    html.Span("▲", style="color:#10b981;font-size:11px;")
                                    html.Span("Normal n", classes="text-caption font-weight-bold")
                                with html.Span(classes="d-flex align-center ga-1"):
                                    html.Span("▲", style="color:#f59e0b;font-size:11px;")
                                    html.Span("Traction t", classes="text-caption font-weight-bold")
                            # Quick Preset Pills
                            with html.Div(classes="d-flex flex-wrap ga-1 mb-2"):
                                v3.VBtn("+X", size="x-small", variant="tonal", classes="plane-preset-btn", click=(ctrl.set_lab_plane, "['+X']"))
                                v3.VBtn("-X", size="x-small", variant="tonal", classes="plane-preset-btn", click=(ctrl.set_lab_plane, "['-X']"))
                                v3.VBtn("+Y", size="x-small", variant="tonal", classes="plane-preset-btn", click=(ctrl.set_lab_plane, "['+Y']"))
                                v3.VBtn("-Y", size="x-small", variant="tonal", classes="plane-preset-btn", click=(ctrl.set_lab_plane, "['-Y']"))
                                v3.VBtn("+Z", size="x-small", variant="tonal", classes="plane-preset-btn", click=(ctrl.set_lab_plane, "['+Z']"))
                                v3.VBtn("-Z", size="x-small", variant="tonal", classes="plane-preset-btn", click=(ctrl.set_lab_plane, "['-Z']"))
                                v3.VBtn("Principal 1", size="x-small", variant="tonal", classes="plane-preset-btn", click=(ctrl.set_lab_plane, "['Principal 1']"))
                                v3.VBtn("Principal 2", size="x-small", variant="tonal", classes="plane-preset-btn", click=(ctrl.set_lab_plane, "['Principal 2']"))
                                v3.VBtn("Principal 3", size="x-small", variant="tonal", classes="plane-preset-btn", click=(ctrl.set_lab_plane, "['Principal 3']"))
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
                                    html.Div("Normal stress σₙ (Perpendicular)", classes="text-caption")
                                    html.Div("{{ lab_normal_stress }}", classes="text-subtitle-2 font-weight-bold")
                                with html.Div(classes="pa-2 rounded", style="background:rgba(245,158,11,.10)"):
                                    html.Div("Shear magnitude τ (Parallel)", classes="text-caption")
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
