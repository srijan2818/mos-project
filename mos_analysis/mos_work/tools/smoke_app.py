"""Offline check of the app's own logic with stand-ins for trame, PyVista and scikit-fem.

    python tools/smoke_app.py

Runs mos.app.main end to end (solve button, redraw handlers, template menu, error reporting)
against fake UI objects and a fake solver result. It catches typos and wiring mistakes only;
it says nothing about whether the real trame / PyVista / scikit-fem calls are right.
"""
import pathlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


class State:
    """Minimal stand-in for trame's reactive state."""

    def __init__(self):
        self.__dict__["_handlers"] = []

    def update(self, d):
        """Set many keys at once."""
        self.__dict__.update(d)

    def flush(self):
        """No-op (the real one pushes changes to the browser)."""

    def change(self, *names):
        """Decorator registering a change handler, like `@state.change(...)`."""
        def deco(fn):
            """Register fn and return it unchanged."""
            self._handlers.append((names, fn))
            return fn
        return deco

    def fire(self, *changed):
        """Call the handlers registered for any of the changed keys."""
        for names, fn in self._handlers:
            if set(names) & set(changed):
                fn(**{n: self.__dict__.get(n) for n in names})


class Grid:
    """Minimal stand-in for a pyvista UnstructuredGrid."""

    def __init__(self, n=5):
        self.points = np.random.default_rng(1).random((n, 3))
        self.point_data = {}
        self.bounds = (0, 10, -.5, .5, -.5, .5)
        self.n_cells, self.n_points = 7, n

    def copy(self):
        """Deep-ish copy (points only)."""
        g = Grid(self.n_points)
        g.points = self.points.copy()
        return g

    def clip(self, **kw):
        """Pretend to clip."""
        return self

    def extract_surface(self, algorithm=None):
        """Pretend to extract the surface."""
        return MagicMock()


def main():
    """Run the checks; raises AssertionError on the first failure."""
    state, ctrl = State(), SimpleNamespace()
    cli = MagicMock()
    cli.parse_known_args.return_value = (SimpleNamespace(spec=None), [])
    server = SimpleNamespace(state=state, controller=ctrl, cli=cli, start=lambda: None)
    plotter = MagicMock()
    pv = types.ModuleType("pyvista")
    pv.Plotter, pv.OFF_SCREEN = (lambda: plotter), False
    sys.modules.update({
        "pyvista": pv, "trame_pyvista": MagicMock(), "trame_pyvista.ui": MagicMock(),
        "trame": MagicMock(), "trame.app": MagicMock(get_server=lambda **k: server),
        "trame.ui": MagicMock(), "trame.ui.vuetify3": MagicMock(), "trame.widgets": MagicMock(),
        "trame.widgets.html": MagicMock(), "trame.widgets.vuetify3": MagicMock(),
        "skfem": MagicMock(), "skfem.element": MagicMock(), "skfem.io": MagicMock(),
        "skfem.io.meshio": MagicMock()})

    from mos.app import main as app
    from mos.core import spec as spec_mod
    from tests import fakemesh

    n, rng = 5, np.random.default_rng(2)
    sig, eps = rng.random((3, 3, n)) * 1e6, rng.random((3, 3, n)) * 1e-3
    from mos.core import post
    sig_nodal = 0.5 * (sig + sig.transpose(1, 0, 2))
    sig_cell = sig_nodal[:, :, :2]
    res = SimpleNamespace(u=np.zeros(30), u_nodal=rng.random((3, n)) * 1e-3,
                          reactions=np.array([0, 1e6, 0.]), equilibrium_error=1e-12,
                          sig_nodal=sig_nodal, sig_cell=sig_cell,
                          sig_q=sig_cell[:, :, :, None],
                          eps_nodal=0.5 * (eps + eps.transpose(1, 0, 2)),
                          traction_on_plane=lambda xyz, normal, points, where="nodal":
                              post.traction_on_plane({"nodal": sig_nodal, "cell": sig_cell,
                                                      "q": sig_cell[:, :, :, None]}[where],
                                                     points, xyz, normal))
    def fake_cad_mesh():
        """A box-like tetra mesh with named CAD surfaces for STEP-path smoke tests."""
        mesh = fakemesh.box(10.0, 1.0, 1.0, 2, 1, 1)
        boundary_ids = np.asarray(mesh.boundary_facets(), dtype=int)
        boundary_vertices = mesh.facets[:, boundary_ids]
        boundaries = {}
        for axis, axis_letter in enumerate("xyz"):
            lo, hi = float(mesh.p[axis].min()), float(mesh.p[axis].max())
            coords = mesh.p[axis, boundary_vertices]
            boundaries[f"cad_face_{2 * axis + 1}"] = boundary_ids[
                np.all(np.isclose(coords, lo, rtol=0.0, atol=1e-9), axis=0)]
            boundaries[f"cad_face_{2 * axis + 2}"] = boundary_ids[
                np.all(np.isclose(coords, hi, rtol=0.0, atol=1e-9), axis=0)]
        mesh.boundaries = boundaries
        return mesh

    def fake_build(spec, mesh=None):
        """Build with stand-in meshes so the smoke test stays offline."""
        if mesh is None:
            if spec.mesh[0] == "step":
                mesh = fake_cad_mesh()
            else:
                mesh = fakemesh.box(*spec.mesh[1:7])
        return spec_mod.build(spec, mesh=mesh)

    app.build = fake_build

    def fake_generate_cad_from_parameters(obj_type, params, output_path):
        """Stand in for OpenCASCADE; the mesh builder above supplies the test mesh."""
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f"offline smoke STEP placeholder: {obj_type}\n")
        return output_path

    app.generate_cad_from_parameters = fake_generate_cad_from_parameters
    solved_cases = []

    def fake_solve(model, case):
        """Record the case under test and return the synthetic result."""
        solved_cases.append(case)
        return res

    app.solve_case = fake_solve

    class FakeLiveResponse:
        """Minimal live-response stand-in for validating controller wiring."""

        def matches(self, load):
            """Accept the already-selected load signature."""
            return True

        def evaluate(self, load):
            """Return a stable synthetic result without solving."""
            return res

    class FakeCachedLinearSolver:
        """Small cache double that verifies the UI live-edit path offline."""
        def __init__(self, model):
            self.model = model

        def solve(self, case):
            """Record the initial solve and return the synthetic result."""
            solved_cases.append(case)
            return res

        def compatible(self, model):
            """Require the same mesh object for the cached stiffness operator."""
            return model.mesh is self.model.mesh

        def prepare_live_load(self, case, index):
            """Return a response object without invoking the full solver."""
            assert 0 <= index < len(case.loads)
            return FakeLiveResponse()

    app.CachedLinearSolver = FakeCachedLinearSolver
    app.viz.to_pyvista = lambda mesh: Grid(n)

    app.main([])
    assert state.status.startswith("Edit"), state.status

    # Learning Lab: initial visuals, case presets, reactive principal stresses,
    # plane orientation, snapshot comparison, and analytical teaching checks.
    assert state.active_page == "workspace"
    assert state.lab_tensor_svg.startswith("<svg")
    assert state.lab_mohr_svg.startswith("<svg")
    state.lab_case = "Uniaxial tension"
    state.fire("lab_case")
    assert state.lab_sigma_1 == "100.00"
    state.lab_case = "Pure shear"
    state.fire("lab_case")
    assert (state.lab_sigma_1, state.lab_sigma_2, state.lab_sigma_3) == ("60.00", "0.00", "-60.00")
    state.lab_case = "Uniaxial tension"
    state.fire("lab_case")
    ctrl.save_lab_snapshot()
    state.lab_sigma_xx = 120.0
    state.fire("lab_sigma_xx")
    assert state.lab_snapshot_delta == "+20.00 MPa", state.lab_snapshot_delta
    state.lab_tool_mode = "shaft"
    state.fire("lab_tool_mode")
    assert state.lab_quick_primary_value == "28.29 MPa"

    ctrl.run_solve()
    assert state.theme == "dark", "theme should default to dark"
    assert state.field_items and abs(float(state.scale) - 1.0) < 1e-12 and state.scale_max > 0
    assert plotter.add_mesh.called and plotter.clear.called
    assert state.lab_has_result is True
    assert state.lab_node_index != "—"
    assert "nodes" in state.lab_model_summary

    # Opening the integrated Lab is an overlay-only operation: it must not clear
    # and rebuild the FE mesh, move the camera, or register another VTK picker.
    scene_clears_before_lab = plotter.clear.call_count
    actor_additions_before_lab = plotter.add_mesh.call_count
    view_updates_before_lab = ctrl.view_update.call_count
    picker_registrations_before_lab = plotter.enable_point_picking.call_count
    drawer_before_lab = state.main_drawer
    state.analysis_dock_open = True
    state.fire("analysis_dock_open")
    assert plotter.clear.call_count == scene_clears_before_lab
    assert plotter.add_mesh.call_count == actor_additions_before_lab, (
        "opening the analysis dock must not add VTK actors"
    )
    assert ctrl.view_update.call_count == view_updates_before_lab, (
        "opening the analysis dock must not render or resize the VTK viewport"
    )
    assert plotter.enable_point_picking.call_count == picker_registrations_before_lab == 1
    assert state.lab_has_result is True
    assert state.lab_status.startswith("FEA result"), state.lab_status
    # Opening the Lab must not collapse the model drawer: that changes the
    # plotter host width during a VTK render and caused a clipped/zoomed canvas.
    assert state.main_drawer == drawer_before_lab

    # Manual sample-point changes refresh the numeric analysis and only the small
    # marker/plane overlay, not the complete VTK scene.
    state.lab_point_x, state.lab_point_y, state.lab_point_z = 0.25, 0.0, 0.0
    state.fire("lab_point_x", "lab_point_y", "lab_point_z")
    assert state.lab_node_index != "—"
    assert plotter.clear.call_count == scene_clears_before_lab
    state.analysis_dock_open = False
    state.fire("analysis_dock_open")
    assert plotter.clear.call_count == scene_clears_before_lab
    assert state.main_drawer == drawer_before_lab
    state.lab_plane_preset = "+X"
    state.fire("lab_plane_preset")
    assert state.lab_plane_normal == "[1.0000, 0.0000, 0.0000]"
    state.lab_plane_preset = "Custom"
    state.lab_plane_custom = "0, 3, 4"
    state.fire("lab_plane_preset", "lab_plane_custom")
    assert state.lab_plane_normal == "[0.0000, 0.6000, 0.8000]"

    # Live analysis: pick an existing load, then scale it without invoking solve_case.
    solve_count_before_live_edit = len(solved_cases)
    ctrl.start_live_load_edit(0)
    assert state.analysis_dock_open and state.live_load_ready
    state.live_load_multiplier = 1.2
    state.fire("live_load_multiplier")
    assert len(solved_cases) == solve_count_before_live_edit
    assert np.allclose(
        [float(v) for v in state.loads_list[0]["vec"].split(",")],
        [0.0, -120000.0, 0.0],
    )
    assert "Live response" in state.status

    assert plotter.add_mesh.call_args_list[0].kwargs["scalar_bar_args"]["title"] == "von Mises [MPa]"
    assert "color" in plotter.add_mesh.call_args_list[0].kwargs["scalar_bar_args"]
    state.template = "Bar - torsion"
    state.fire("template")
    assert "moment" in state.spec_text
    state.clip_axis, state.field, state.stress_unit = "x", "σxx", "kPa"
    state.fire("field")
    assert plotter.enable_point_picking.call_count == 1
    assert plotter.add_mesh.call_args_list[-2].kwargs["scalar_bar_args"]["title"] == "σxx [kPa]"
    state.spec_text = "mesh box 1 1 1 2 2 2\nbogus\n"
    ctrl.run_solve()
    assert state.status_type == "error" and "line 2" in state.status, state.status

    # Interactive: repeated loads on the same face must remain separate and additive.
    from mos.core import Moment, SurfaceForce
    state.config_mode = "interactive"
    state.geom_mode = "box"
    state.supports_list = [{"face": "-x", "comps": "xyz"}]
    state.loads_list = []
    state.selected_face = "+x"
    state.load_target = "face"
    state.load_mode = "force"
    state.moment_mag = "not a number"  # inactive controls must not block a force load
    state.force_vec = "100, 0, 0"
    ctrl.apply_current_load()
    state.force_vec = "0, 200, 0"
    ctrl.apply_current_load()
    assert state.status_type == "success", state.status
    interactive_case = solved_cases[-1]
    assert len(interactive_case.loads) == 2
    assert all(isinstance(load, SurfaceForce) for load in interactive_case.loads)
    assert np.allclose(interactive_case.loads[0].force, [100, 0, 0])
    assert np.allclose(interactive_case.loads[1].force, [0, 200, 0])
    regions = [load.region.name for load in interactive_case.loads]
    assert len(set(regions)) == 2, regions

    # Interactive: overlapping moments on the same face are retained independently.
    state.loads_list = []
    state.load_mode = "moment"
    state.moment_mag = "250"
    ctrl.apply_current_load()
    state.moment_mag = "750"
    ctrl.apply_current_load()
    assert state.status_type == "success", state.status
    moment_case = solved_cases[-1]
    assert len(moment_case.loads) == 2
    assert all(isinstance(load, Moment) for load in moment_case.loads)
    assert np.allclose(moment_case.loads[0].M, [250, 0, 0])
    assert np.allclose(moment_case.loads[1].M, [750, 0, 0])
    assert len({load.region.name for load in moment_case.loads}) == 2

    # Script mode: repeated clicks append uniquely named regions, rather than redefining one.
    from mos.templates import TEMPLATES
    state.config_mode = "script"
    state.spec_text = TEMPLATES["Cantilever - tip load"]
    state.loads_list = []
    state.load_mode = "force"
    state.force_vec = "0, -10, 0"
    ctrl.apply_current_load()
    state.force_vec = "0, -20, 0"
    ctrl.apply_current_load()
    assert state.status_type == "success", state.status
    script_case = solved_cases[-1]
    assert len(script_case.loads) == 3  # original template load plus both additions
    assert len({load.region.name for load in script_case.loads}) == len(script_case.loads)

    # Interactive first-load of STEP: generic -x/+x choices must be mapped to
    # the imported mesh's named boundary faces, not stale default x=0 bounds.
    state.config_mode = "interactive"
    state.geom_mode = "step"
    state.cad_preset = "custom"
    state.cad_custom_path = "/tmp/model with spaces.step"
    state.cad_mesh_size = 0.6
    state.supports_list = [{"face": "-x", "comps": "xyz"}]
    state.loads_list = [{"mode": "force", "target": "face", "face": "+x",
                         "vec": "0, -100, 0", "val": "0", "pt": [10, 0, 0]}]
    state.selected_face, state.support_face = "+x", "-x"
    state.show_face_highlight = False  # the smoke-test PyVista stub has no PolyData
    ctrl.run_solve()
    assert state.status_type in ("success", "warning"), state.status
    assert "region   neg_x facets" not in state.spec_text, state.spec_text
    assert "support  cad_face_1 xyz" in state.spec_text, state.spec_text
    assert "force    cad_face_2 0, -100, 0" in state.spec_text, state.spec_text
    assert len(solved_cases[-1].loads) == 1
    assert solved_cases[-1].loads[0].region.name == "cad_face_2"

    # CV Mode check: analyze sample real-life object and generate CAD spec
    state.geom_mode = "cv"
    state.cv_source_mode = "preset"
    state.cv_preset_path = "cad_models/cv_samples/bracket_photo.png"
    state.cv_ref_scale = 0.25
    ctrl.run_cv_analysis()
    assert state.cv_detected_type == "bracket", state.cv_detected_type
    assert state.cv_confidence_pct >= 70, state.cv_confidence_pct
    assert state.cv_param_leg_x > 0
    assert state.cv_annotated_preview.startswith("data:image/png;base64,")

    # Generate CAD & load into interactive model. The OpenCASCADE operation is
    # stubbed above; this checks the controller/spec/solve wiring offline.
    state.cv_step_path = "/tmp/mos_smoke_cv_detected_object.step"
    ctrl.generate_cv_cad_and_load()
    assert state.status_type in ("success", "warning"), state.status
    assert "mesh     step" in state.spec_text, state.spec_text
    assert "cv_detected_object.step" in state.spec_text, state.spec_text
    print("app smoke test OK")


if __name__ == "__main__":
    main()
