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

    def extract_surface(self):
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
        "pyvista": pv, "pyvista.trame": MagicMock(), "pyvista.trame.ui": MagicMock(),
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
    res = SimpleNamespace(u=np.zeros(30), u_nodal=rng.random((3, n)) * 1e-3,
                          reactions=np.array([0, 1e6, 0.]), equilibrium_error=1e-12,
                          sig_nodal=0.5 * (sig + sig.transpose(1, 0, 2)),
                          eps_nodal=0.5 * (eps + eps.transpose(1, 0, 2)))
    app.build = lambda spec: spec_mod.build(spec, fakemesh.box(*spec.mesh[1:7]))
    solved_cases = []

    def fake_solve(model, case):
        """Record the case under test and return the synthetic result."""
        solved_cases.append(case)
        return res

    app.solve_case = fake_solve
    app.viz.to_pyvista = lambda mesh: Grid(n)

    app.main([])
    assert state.status.startswith("Edit"), state.status
    ctrl.run_solve()
    assert state.theme == "dark", "theme should default to dark"
    assert state.field_items and abs(float(state.scale) - 1.0) < 1e-12 and state.scale_max > 0
    assert plotter.add_mesh.called and plotter.clear.called
    assert plotter.add_mesh.call_args_list[0].kwargs["scalar_bar_args"]["title"] == "von Mises [MPa]"
    assert "color" in plotter.add_mesh.call_args_list[0].kwargs["scalar_bar_args"]
    state.template = "Bar - torsion"
    state.fire("template")
    assert "moment" in state.spec_text
    state.clip_axis, state.field, state.stress_unit = "x", "σxx", "kPa"
    state.fire("field")
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
    print("app smoke test OK")


if __name__ == "__main__":
    main()
