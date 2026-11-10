"""App controller logic against the offline fakes from ``tools/smoke_app.py``.

These tests drive the real handlers in ``mos.app.main`` (spec building, state resets,
input parsing) with stand-ins for trame, PyVista and the solver. ``sys.modules`` and the
attributes of already-imported project modules are restored afterwards so the fakes never
leak into the FEM tests.
"""
import sys

import numpy as np
import pytest

_PROJECT_ROOTS = ("mos", "tests", "tools")


@pytest.fixture
def make_env():
    """Factory starting the app against fakes: ``make_env(spec_path=None)``."""
    saved = dict(sys.modules)
    saved_vars = {n: dict(vars(m)) for n, m in saved.items()
                  if n.split(".")[0] in _PROJECT_ROOTS and m is not None}
    for name in [n for n in sys.modules if n == "mos.viz" or n.startswith("mos.app")]:
        del sys.modules[name]
    from tools.smoke_app import FAKE_MODULES, make_fake_app
    try:
        yield make_fake_app
    finally:
        # Only project modules and the fakes are reset: dropping third-party C extensions
        # that were first imported meanwhile would make their re-import fail.
        for name in [n for n in sys.modules
                     if n.split(".")[0] in _PROJECT_ROOTS or n in FAKE_MODULES]:
            if saved.get(name) is not sys.modules[name]:
                if name in saved:
                    sys.modules[name] = saved[name]
                else:
                    del sys.modules[name]
        for name, d in saved_vars.items():
            v = vars(saved[name])
            for k in [k for k in v if k not in d]:
                delattr(saved[name], k)
            for k, val in d.items():
                setattr(saved[name], k, val)


@pytest.fixture
def env(make_env):
    """A freshly started app wired to fakes; yields the smoke-test namespace."""
    return make_env()


def _box_interactive(state, L=10.0):
    state.config_mode = "interactive"
    state.geom_mode = "box"
    state.box_L, state.box_H, state.box_W = L, 1.0, 1.0
    state.box_nx, state.box_ny, state.box_nz, state.box_elem = 4, 1, 1, "tet"
    state.supports_list = [{"face": "-x", "comps": "xyz"}]
    state.loads_list = [{"mode": "force", "target": "face", "face": "+x",
                         "vec": "0, -100, 0", "val": "0", "pt": [L, 0, 0]}]


def _face_x(case, mesh):
    """x coordinates of every vertex of the first load's facets."""
    return mesh.p[0, mesh.facets[:, case.loads[0].region.ids]]


def test_box_length_change_uses_new_bounds_on_first_solve(env):
    s, c = env.state, env.ctrl
    _box_interactive(s, L=10.0)
    c.run_solve()
    assert s.status_type == "success", s.status
    s.box_L = 20.0
    c.run_solve()
    assert s.status_type == "success", s.status
    case = env.solved_cases[-1]
    mesh = env.solved_models[-1].mesh
    assert np.allclose(_face_x(case, mesh), 20.0), "tip load must sit on x = L after L changes"


def test_box_length_shrink_does_not_empty_regions(env):
    s, c = env.state, env.ctrl
    _box_interactive(s, L=10.0)
    c.run_solve()
    s.box_L = 5.0
    c.run_solve()
    assert s.status_type == "success", s.status
    assert np.allclose(_face_x(env.solved_cases[-1], env.solved_models[-1].mesh), 5.0)


def test_box_unknown_face_key_is_an_error_not_x_gt_0(env):
    s, c = env.state, env.ctrl
    _box_interactive(s)
    s.supports_list = [{"face": "cad_face_3", "comps": "xyz"}]
    c.run_solve()
    assert s.status_type == "error", s.status
    assert "x > 0" not in s.spec_text


def test_material_preset_sets_constants(env):
    s = env.state
    s.mat_preset = "Aluminum"
    s.fire("mat_preset")
    assert float(s.mat_E) == pytest.approx(70e9, rel=0.1)
    assert float(s.mat_rho) == pytest.approx(2700, rel=0.05)


def test_material_preset_sets_proportional_limits(env):
    s = env.state
    s.mat_preset = "Concrete"
    s.fire("mat_preset")
    assert s.mat_brittle and float(s.mat_spl) == pytest.approx(3) and float(s.mat_splc) == pytest.approx(12)
    s.mat_preset = "Steel"
    s.fire("mat_preset")
    assert not s.mat_brittle and float(s.mat_spl) == pytest.approx(250) and s.mat_splc == ""


def _solve_with_limit(env, spl, splc="", brittle=False):
    s, c = env.state, env.ctrl
    _box_interactive(s)
    s.mat_spl, s.mat_splc, s.mat_brittle = spl, splc, brittle
    c.run_solve()
    assert s.status_type == "success", s.status
    return env.solved_models[-1].material


def test_limit_check_flags_nodes_and_opens_the_dialog(env):
    from mos.core import post
    s = env.state
    mat = _solve_with_limit(env, "0.5")                  # synthetic stresses are ~1 MPa
    assert mat.sigma_pl == pytest.approx(0.5e6) and mat.sigma_pl_c is None
    U = post.von_mises(env.res.sig_nodal) / 0.5e6
    assert s.limit_count == int((U >= 1).sum()) > 0 and s.limit_dialog
    assert f"{1 / U.max():.3g} × the applied loads" in s.limit_text
    assert "Proportional limit exceeded" in s.analysis_result_warning
    assert "Utilisation (σ/σ_pl)" in s.field_items


def test_limit_check_quiet_below_the_limit_and_without_one(env):
    s = env.state
    _solve_with_limit(env, "250")
    assert s.limit_count == 0 and not s.limit_dialog and s.analysis_result_warning == ""
    mat = _solve_with_limit(env, "")
    assert mat.sigma_pl is None and "Utilisation (σ/σ_pl)" not in s.field_items


def test_brittle_limit_uses_coulomb_mohr_and_needs_both_limits(env):
    s, c = env.state, env.ctrl
    mat = _solve_with_limit(env, "0.2", "5", brittle=True)
    assert mat.sigma_pl_c == pytest.approx(5e6) and s.limit_count > 0
    assert "Coulomb–Mohr" in s.limit_text
    s.mat_splc = ""
    c.run_solve()
    assert s.status_type == "error" and "compression limit" in s.status, s.status


def test_point_pressure_is_rejected_not_dropped(env):
    s, c = env.state, env.ctrl
    _box_interactive(s)
    before = list(s.loads_list)
    s.load_target, s.load_mode, s.pressure_val = "point", "pressure", "1e6"
    c.apply_current_load()
    assert s.status_type == "error" and "face" in s.status.lower(), s.status
    assert s.loads_list == before


def test_bad_moment_magnitude_is_an_error_not_1e5(env):
    s, c = env.state, env.ctrl
    _box_interactive(s)
    before = list(s.loads_list)
    s.load_target, s.load_mode, s.selected_face = "face", "moment", "+x"
    s.moment_axis_type, s.moment_mag = "x", "2e4 Nm"
    c.apply_current_load()
    assert s.status_type == "error" and "2e4 Nm" in s.status, s.status
    assert s.loads_list == before


def test_malformed_force_vector_is_an_error(env):
    s, c = env.state, env.ctrl
    _box_interactive(s)
    before = list(s.loads_list)
    s.load_target, s.load_mode, s.selected_face, s.force_vec = "face", "force", "+x", "0, -1e5"
    c.apply_current_load()
    assert s.status_type == "error", s.status
    assert s.loads_list == before


def test_script_mode_add_support_keeps_the_script(env):
    s, c = env.state, env.ctrl
    from mos.templates import TEMPLATES
    script = TEMPLATES["Cantilever - tip load"]
    s.config_mode, s.spec_text = "script", script
    c.run_solve()
    s.support_face, s.support_comps = "+x", "y"
    c.add_support_to_list()
    assert s.spec_text.startswith(script.rstrip()), s.spec_text
    assert s.spec_text.rstrip().endswith("y"), s.spec_text
    assert s.status_type == "success", s.status


def test_script_mode_remove_and_clear_leave_the_script_alone(env):
    s, c = env.state, env.ctrl
    from mos.templates import TEMPLATES
    script = TEMPLATES["Cantilever - tip load"]
    s.config_mode, s.spec_text = "script", script
    s.supports_list = [{"face": "-x", "comps": "xyz"}]
    s.loads_list = [{"mode": "force", "target": "face", "face": "+x",
                     "vec": "0, -1, 0", "val": "0", "pt": [10, 0, 0]}]
    for action in (lambda: c.remove_load(0), lambda: c.remove_support(0), c.clear_all_loads):
        action()
        assert s.spec_text == script
        assert s.status_type == "warning", s.status


def test_cli_spec_file_is_used_by_solve(make_env, tmp_path):
    from mos.templates import TEMPLATES
    f = tmp_path / "model.mos"
    f.write_text(TEMPLATES["Bar - torsion"])
    env = make_env(spec_path=str(f))
    assert env.state.config_mode == "script"
    env.ctrl.run_solve()
    assert env.state.spec_text == f.read_text()
    assert env.state.status_type == "success", env.state.status


def test_upload_inputs_use_vue_change_handlers():
    """The file pickers must bind `change` (not the dropped `onchange`) and avoid bare `document`."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "mos/app/main.py").read_text()
    assert "onchange=" not in src
    assert "document.getElementById" not in src
    assert src.count("window.mosUpload.") >= 6
    js = (Path(__file__).resolve().parents[1] / "mos/app/static/upload.js").read_text()
    assert "window.mosUpload" in js
    # Desktop browsers ignore <input capture>; the camera button must use getUserMedia.
    assert "window.mosUpload.camera(" in src and "getUserMedia" in js


def test_colmap_button_reaches_the_runner(env, tmp_path):
    s, c = env.state, env.ctrl
    calls = []
    env.app.tool_availability = lambda: {"colmap": True, "ns-process-data": False, "ns-train": False}
    s.cv_mv_images_dir, s.cv_mv_colmap_running = str(tmp_path), False
    c.run_cv_colmap_sparse()
    assert "is not defined" not in s.cv_mv_colmap_status, s.cv_mv_colmap_status


def test_multiview_method_select_updates_commands(env, tmp_path):
    s = env.state
    s.cv_mv_images_dir, s.cv_mv_ns_data_dir = str(tmp_path / "images"), str(tmp_path / "ns")
    s.cv_mv_method = "graphdeco"
    s.fire("cv_mv_method")
    assert "graphdeco" in str(s.cv_mv_commands).lower(), s.cv_mv_commands


def _analyse_sample(s, c):
    s.geom_mode, s.cv_source_mode = "cv", "preset"
    s.cv_preset_path, s.cv_ref_scale = "cad_models/cv_samples/bracket_photo.png", 0.25
    c.run_cv_analysis()
    assert s.cv_detected_type == "bracket" and s.cv_mask_ready, s.cv_status


@pytest.mark.parametrize("mode", ["webcam", "upload"])
def test_cv_without_an_image_does_not_analyse_the_sample(env, mode):
    s, c = env.state, env.ctrl
    s.cv_source_mode, s.cv_image_b64, s.cv_custom_path = mode, "", ""
    c.run_cv_analysis()
    assert not s.cv_detected_type and "failed" in s.cv_status.lower(), s.cv_status


def test_failed_cv_analysis_clears_the_previous_result(env):
    s, c = env.state, env.ctrl
    _analyse_sample(s, c)
    s.cv_source_mode, s.cv_custom_path = "upload", "/nonexistent/photo.png"
    c.run_cv_analysis()
    assert "failed" in s.cv_status.lower(), s.cv_status
    assert not s.cv_detected_type and not s.cv_mask_ready and not s.cv_annotated_preview


def test_cv_running_is_cleared_when_applying_the_result_fails(env):
    s, c = env.state, env.ctrl
    env.app.analyze_image_object = lambda *a, **k: object()   # no result attributes
    s.cv_source_mode = "preset"
    c.run_cv_analysis()
    assert s.cv_running is False
    assert "failed" in s.cv_status.lower(), s.cv_status


def test_template_cad_needs_a_finished_analysis(env):
    s, c = env.state, env.ctrl
    s.cv_detected_type, s.cv_mask_ready, s.cv_running = "", False, False
    c.generate_cv_cad_and_load()
    assert not s.cv_running, "must not start an analysis it can't wait for"
    assert "run cv model" in s.cv_status.lower(), s.cv_status


def _solve_step(s, c):
    s.config_mode, s.geom_mode, s.cad_preset = "interactive", "step", "custom"
    s.cad_custom_path, s.cad_mesh_size = "/tmp/model.step", 0.6
    s.fire("geom_mode", "cad_custom_path")   # as the UI does when the file is chosen
    s.supports_list = [{"face": "cad_face_1", "comps": "xyz"}]
    s.loads_list = [{"mode": "force", "target": "face", "face": "cad_face_2",
                     "vec": "0, -100, 0", "val": "0", "pt": [10, 0, 0]}]
    s.show_face_highlight = False
    c.run_solve()
    assert s.status_type == "success", s.status


def test_failed_solve_restores_the_solved_models_faces(env):
    s, c = env.state, env.ctrl
    _box_interactive(s)
    c.run_solve()
    bounds, options = list(s.model_bounds), list(s.face_options)
    s.geom_mode, s.cad_preset, s.cad_custom_path = "step", "custom", "/tmp/model.step"
    s.loads_list = [{"mode": "force", "target": "face", "face": "+x",
                     "vec": "0, -1", "val": "0", "pt": [0, 0, 0]}]   # fails after meshing
    c.run_solve()
    assert s.status_type == "error", s.status
    assert s.model_bounds == bounds and s.face_options == options


def test_switching_geometry_drops_cad_face_references(env):
    s, c = env.state, env.ctrl
    _solve_step(s, c)
    s.supports_list = s.supports_list + [{"face": "-x", "comps": "y"}]
    s.geom_mode = "box"
    s.fire("geom_mode")
    assert [sp["face"] for sp in s.supports_list] == ["-x"]
    assert s.loads_list == []
    assert s.status_type == "warning" and "re-select" in s.status.lower(), s.status


def test_stale_lab_face_plane_is_reset_on_the_new_model(env):
    s, c = env.state, env.ctrl
    _solve_step(s, c)
    s.lab_plane_preset = "Face: cad_face_5"
    _box_interactive(s)
    c.run_solve()
    assert s.lab_plane_preset == "Custom"
    assert s.lab_status.startswith("FEA result"), s.lab_status


def test_solve_replaces_the_reconstruction_preview(env):
    s, c = env.state, env.ctrl
    _box_interactive(s)
    s.cv3d_preview_active = True
    c.run_solve()
    assert s.cv3d_preview_active is False


def test_support_options_have_unique_values_and_correct_symmetry(env):
    opts = env.app.SUPPORT_COMP_OPTIONS
    values = [o["value"] for o in opts]
    assert len(values) == len(set(values)), "a v-select shows every item matching the value"
    for o in opts:
        if "symmetry" in o["title"].lower():
            # a symmetry plane normal to an axis fixes only that displacement component
            axis = o["title"].lower().split("⊥")[1].strip()[0]
            assert o["value"] == axis, o


def test_picked_point_is_not_snapped_to_the_face_centroid(env):
    s, c = env.state, env.ctrl
    _box_interactive(s)
    c.run_solve()
    s.analysis_dock_open, s.scale = False, 0.0
    pick = env.plotter.enable_point_picking.call_args.kwargs["callback"]
    target = tuple(np.round(np.random.default_rng(1).random((5, 3))[3], 4))  # a fake-grid node
    pick(target)
    s.fire("selected_face")          # trame runs the face watcher after the pick handler
    assert s.load_target == "point"
    assert np.allclose([s.point_x, s.point_y, s.point_z], target, atol=1e-4)


def test_late_tensor_watcher_keeps_the_fea_traction_readout(env):
    """trame fires @state.change watchers after the handler returns, so a 'suspended' flag
    is already cleared by then; the deferred refresh must not overwrite the FEA values."""
    s, c = env.state, env.ctrl
    _box_interactive(s)
    c.run_solve()
    s.lab_point_x, s.lab_point_y, s.lab_point_z = 5.0, 0.0, 0.0
    s.fire("lab_point_x")
    fea = s.lab_plane_traction
    assert fea.count(".") and len(fea.split(",")[0].split(".")[1].split()[0]) == 4, fea
    s.fire("lab_sigma_xx", "lab_azimuth")   # the deferred watcher flush
    assert s.lab_plane_traction == fea


def test_app_css_is_a_served_stylesheet_not_a_template_style_tag():
    """Vue drops <style> inside templates: the dock CSS never applied and the dock was
    appended below the viewport, stretching the render canvas."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    assert "html.Style(" not in (root / "mos/app/main.py").read_text()
    css = (root / "mos/app/static/app.css").read_text()
    assert ".mos-analysis-dock" in css and "position: fixed" in css


def test_solve_runs_off_the_ui_loop_and_refuses_a_second_solve(env):
    """With a running event loop, Solve returns at once; the result is applied later."""
    import asyncio
    import threading
    s, c, app = env.state, env.ctrl, env.app
    _box_interactive(s)
    gate, base = threading.Event(), app.CachedLinearSolver

    class SlowSolver(base):
        def solve(self, case):
            assert gate.wait(10), "test gate never opened"
            return super().solve(case)
    app.CachedLinearSolver = SlowSolver
    n_before = len(env.solved_cases)

    async def scenario():
        c.run_solve()
        assert s.solve_running is True            # returned while the worker is blocked
        c.run_solve()                             # a second click is refused, not queued
        assert "already running" in s.status.lower(), s.status
        gate.set()
        for _ in range(200):
            await asyncio.sleep(0.02)
            if not s.solve_running:
                break
    asyncio.run(scenario())
    assert s.solve_running is False
    assert s.status_type == "success", s.status
    assert len(env.solved_cases) == n_before + 1


def test_empty_lab_coordinate_is_an_error_not_the_origin(env):
    s, c = env.state, env.ctrl
    _box_interactive(s)
    c.run_solve()
    s.lab_point_x, s.lab_point_y, s.lab_point_z = "", 0.0, 0.0
    s.fire("lab_point_x")
    assert "error" in s.lab_status.lower(), s.lab_status


def test_face_pressure_accepts_an_expression(env):
    """Spec values are expressions; the builder must not demand a bare float."""
    s, c = env.state, env.ctrl
    _box_interactive(s)
    s.load_target, s.load_mode, s.selected_face, s.pressure_val = "face", "pressure", "+y", "2*1e6"
    c.apply_current_load()
    assert s.status_type == "success", s.status
    assert "pressure" in s.spec_text and "2*1e6" in s.spec_text


def test_click_mapping_handles_the_letterbox(env):
    # 1600x900 image shown in a 400x240 box with object-fit: contain -> 400x225, 7.5 px bars
    from mos.app.main import _click_to_image
    assert _click_to_image(200, 120, 400, 240, 1600, 900) == pytest.approx((800, 450))
    assert _click_to_image(0, 7.5, 400, 240, 1600, 900) == pytest.approx((0, 0))
    assert _click_to_image(200, 3, 400, 240, 1600, 900) is None          # on the bar


def _recording_analyser(env):
    """Real classical analysis of the sample photo, recording the prompts the app passes."""
    from mos.core.cv_detector import analyze_image_object as real
    calls = []

    def fake(src, reference_scale=0.3, points=None, labels=None, candidate=None):
        calls.append({"points": points, "labels": labels, "candidate": candidate})
        res = real(src, reference_scale=reference_scale, use_models=False)
        res.mask = np.zeros_like(res.mask)
        res.mask[200:281, 200:281] = True          # a known 80 px object: its bbox decides extend/replace
        res.prompt_points = ([(x, y, l) for (x, y), l in zip(points, labels)] if points
                             else [(240.0, 240.0, 1)])
        res.n_candidates = 3
        res.candidate_index = 2 if candidate is None else candidate
        return res
    env.app.analyze_image_object = fake
    return calls


def test_click_select_flow_and_accept_gate(env):
    s, c = env.state, env.ctrl
    calls = _recording_analyser(env)
    s.geom_mode, s.cv_source_mode = "cv", "preset"
    s.cv_preset_path, s.cv_ref_scale = "cad_models/cv_samples/bracket_photo.png", 0.25
    c.run_cv_analysis()
    assert calls[-1]["points"] is None and s.cv_mask_ready and not s.cv_mask_accepted
    # meshing is refused until the mask is accepted
    c.mesh_cv_outline_and_load()
    assert "accept" in s.cv_status.lower()
    # a negative first click is refused; a positive one replaces the automatic guess
    s.cv_click_mode = "remove"
    c.cv_click(100, 100, 480, 480, 480, 480)
    assert s.cv_points == [] and "first click" in s.cv_status.lower()
    s.cv_click_mode = "add"
    # a first click near the guessed object extends it (keeps the guess's point)...
    c.cv_click(240, 250, 480, 480, 480, 480)
    assert calls[-1]["points"] == [(240.0, 240.0), (240.0, 250.0)], calls[-1]
    c.cv_clear_clicks()
    # ...a first click far away replaces it (the guess was the wrong object)
    c.cv_click(2, 2, 480, 480, 480, 480)
    assert calls[-1]["points"] == [(2.0, 2.0)] and calls[-1]["labels"] == [1]
    c.cv_click(300, 50, 480, 480, 480, 480, True)                    # shift = not the part
    assert calls[-1]["labels"] == [1, 0]
    c.cv_mask_smaller()
    assert calls[-1]["candidate"] == 1 and calls[-1]["points"] == [(2.0, 2.0), (300.0, 50.0)]
    c.cv_clear_clicks()
    assert s.cv_points == [] and calls[-1]["points"] is None
    c.accept_cv_mask()
    assert s.cv_mask_accepted
    c.cv_click(100, 100, 480, 480, 480, 480)                          # any change needs re-accepting
    assert not s.cv_mask_accepted


def test_three_photo_flow_builds_a_hull_session_and_solves(env):
    """3 synthetic photos (front/top/side) -> accepted outlines -> orientation -> hull -> solve."""
    import base64
    import io
    from PIL import Image
    from mos.core.cv_detector import CVAnalysisResult
    from tests import synthetic_views as sv
    s, c, app = env.state, env.ctrl, env.app
    box = [(np.zeros(3), np.array([0.060, 0.040, 0.030]))]
    masks = {}
    for k, (name, m) in enumerate(sv.three_masks(box, [0.03, 0.02, 0.015], (3000.0, 2500.0, 3500.0)).items()):
        assert name == app.ORTHO_VIEWS[k]["axis"]
        if name == "side":
            m = np.rot90(m, -1)                                  # taken in portrait
        buf = io.BytesIO()
        Image.fromarray(np.where(m, 60, 200).astype(np.uint8)).convert("RGB").save(buf, "PNG")
        url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        setattr(s, f"mv3_img_{k}", url)
        s.fire(f"mv3_img_{k}")
        masks[url] = m

    def fake(src, reference_scale=0.3, points=None, labels=None, candidate=None):
        m = masks[src]
        return CVAnalysisResult(object_type=None, friendly_name="Unclassified outline", confidence=0.0,
                                detected_features=[], parameters={}, pixel_dimensions=m.shape[::-1],
                                annotated_image_b64="data:image/png;base64,", mask=m,
                                prompt_points=[(1.0, 1.0, 1)], candidate_index=0, n_candidates=1)
    app.analyze_image_object = fake
    c.mv3_build_and_solve()
    assert "all three photos" in s.mv3_msg
    for k in range(3):
        c.mv3_select(k)
        assert s.mv3_active == k and s.cv_mask_ready
        c.accept_cv_mask()
        assert s.mv3_outline[k].startswith("data:image/png;base64,")
    assert all(s.mv3_mask_ok)
    c.mv3_build_and_solve()
    assert "measured dimension" in s.mv3_msg
    s.mv3_known_axis, s.mv3_known_mm = "x", "60"
    c.mv3_build_and_solve()
    assert "disagree" in s.mv3_msg, s.mv3_msg                   # side photo still rotated
    c.mv3_orient(2, "rot")
    assert s.mv3_rot == [0, 0, 1]
    c.mv3_build_and_solve()
    import re
    dims = [float(v) for v in re.search(r"Solid ([\d.]+) × ([\d.]+) × ([\d.]+) mm", s.mv3_msg).groups()]
    assert dims == pytest.approx([60, 40, 30], abs=0.5) and "disagree" not in s.mv3_msg, s.mv3_msg  # 1 voxel
    assert "mesh     hull" in s.spec_text and s.cv_mesh_source == "hull"
    from mos.core.multiview_hull import load_session
    views, axis, known = load_session(s.mv3_session)
    assert [v.axis for v in views] == ["front", "top", "side"] and views[2].rot90 == 1
    assert axis == "x" and known == pytest.approx(0.060)
    # replacing a photo drops its outline only
    s.mv3_img_1 = s.mv3_img_0
    s.fire("mv3_img_1")
    assert s.mv3_mask_ok == [True, False, True] and s.mv3_outline[1] == ""


def _free_body_box(env, counts=(20, 4, 4)):
    from mos.core.solver import count_dofs
    from tests import fakemesh
    s = env.state
    _box_interactive(s)
    s.supports_list = []                                  # free body: indefinite direct solve
    s.box_nx, s.box_ny, s.box_nz = counts
    return count_dofs(fakemesh.box(10.0, 1.0, 1.0, *counts))


def test_free_body_over_the_limit_is_coarsened_not_refused(env):
    from mos.core.solver import count_dofs
    s, c = env.state, env.ctrl
    n0 = _free_body_box(env)
    env.app.FREE_BODY_MAX_DOFS = n0 // 3
    c.run_solve()
    assert "coarsened automatically" in s.status, s.status
    n = count_dofs(env.solved_models[-1].mesh)
    assert n <= n0 // 3 < n0
    assert (s.box_nx, s.box_ny, s.box_nz) != (20, 4, 4)       # the builder's own fields changed
    assert f"mesh     box L H W {s.box_nx} {s.box_ny} {s.box_nz}" in s.spec_text
    c.run_solve()                                             # now within the limit: no note
    assert "coarsened" not in s.status


def test_supported_models_are_not_coarsened(env):
    s, c = env.state, env.ctrl
    n0 = _free_body_box(env)
    s.supports_list = [{"face": "-x", "comps": "xyz"}]
    env.app.FREE_BODY_MAX_DOFS = n0 // 3                      # only the free-body solve has the limit
    c.run_solve()
    assert "coarsened" not in s.status and s.box_nx == 20


def test_script_mode_rewrites_the_mesh_line(env):
    s, c = env.state, env.ctrl
    s.config_mode = "script"
    s.spec_text = ("mesh box 10 1 1 20 4 4   # beam\nmaterial E = 210e9, nu = 0.3, rho = 7850\n"
                   "region tip facets x > 10 - 1e-6\nforce tip 0, -100, 0\n")
    from mos.core.solver import count_dofs
    from tests import fakemesh
    env.app.FREE_BODY_MAX_DOFS = count_dofs(fakemesh.box(10, 1, 1, 20, 4, 4)) // 3
    c.run_solve()
    assert "coarsened automatically" in s.status, s.status
    mesh_line = next(l for l in s.spec_text.splitlines() if l.startswith("mesh"))
    assert mesh_line.startswith("mesh box 10 1 1 ") and "# beam" in mesh_line and " 20 4 4" not in mesh_line


def test_coarsening_that_cannot_reach_the_limit_is_an_error(env):
    s, c = env.state, env.ctrl
    _free_body_box(env)
    env.app.FREE_BODY_MAX_DOFS = 10                           # below even a 1 x 1 x 1 box
    c.run_solve()
    assert s.status_type == "error" and "stopped reducing" in s.status, s.status
