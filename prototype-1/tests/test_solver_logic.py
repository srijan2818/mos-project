"""DOF bookkeeping in mos.core.solver, tested with a fake basis (no scikit-fem needed).

If scikit-fem is missing, a stub is injected ONLY while importing the solver module and removed
afterwards, so it cannot leak into test_fem (which skips cleanly without scikit-fem).
"""
import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from mos.core import BC, LoadCase, Material, Model, PointForce, Region


def _import_solver():
    try:
        import skfem  # noqa: F401  (real library present: nothing to stub)
        return importlib.import_module("mos.core.solver")
    except ImportError:
        names = ["skfem", "skfem.element"]
        saved = {n: sys.modules.get(n) for n in names}
        sys.modules.update({n: MagicMock() for n in names})
        sys.modules.pop("mos.core.solver", None)
        try:
            return importlib.import_module("mos.core.solver")
        finally:
            for n, m in saved.items():
                if m is None:
                    sys.modules.pop(n, None)
                else:
                    sys.modules[n] = m
            sys.modules.pop("mos.core.solver", None)
            import mos.core
            if hasattr(mos.core, "solver"):
                delattr(mos.core, "solver")


solver = _import_solver()


class FakeBasis:
    """4 nodes + 2 edges, 3 components each; dof of (node i, comp c) = 3 i + c, edge j: 12 + 3 j + c."""
    N = 18
    mesh = None
    nodal_dofs = np.arange(12).reshape(4, 3).T
    edge_dofs = 12 + np.arange(6).reshape(2, 3).T
    facet_dofs = np.zeros((0, 0), int)
    interior_dofs = np.zeros((0, 0), int)
    doflocs = np.random.default_rng(0).random((3, 18))

    def zeros(self):
        return np.zeros(self.N)

    def get_dofs(self, nodes=None, facets=None):
        nd = np.asarray(nodes) if nodes is not None else np.array([0, 1, 2])   # a facet touches nodes 0-2
        ids = list(self.nodal_dofs[:, nd].ravel())
        if facets is not None:
            ids += list(self.edge_dofs[:, 0])                                   # ... and edge 0
        return SimpleNamespace(all=lambda: np.array(sorted(ids)))


MAT = Material.isotropic(1.0, 0.3)


def _model(*bcs):
    return Model(None, MAT, list(bcs))


def test_component_dofs():
    b = FakeBasis()
    assert solver.component_dofs(b, 1).tolist() == [1, 4, 7, 10, 13, 16]
    assert sorted(np.concatenate([solver.component_dofs(b, c) for c in range(3)])) == list(range(18))


def test_constraints_node_region_single_component():
    D, x0, per = solver.constraints(_model(BC(Region("n", "nodes", [0, 1]), (1,), 0.5)), FakeBasis())
    assert D.tolist() == [1, 4]
    assert x0[[1, 4]].tolist() == [0.5, 0.5] and np.count_nonzero(x0) == 2
    assert [p.tolist() for p in per] == [[], [1, 4], []]


def test_constraints_facet_region_includes_edge_dofs_and_union():
    fixed = BC(Region("f", "facets", [7]), (0, 2))
    extra = BC(Region("n", "nodes", [0]), (0, 1, 2))                 # overlaps node 0
    D, x0, per = solver.constraints(_model(fixed, extra), FakeBasis())
    assert D.tolist() == sorted(set(D.tolist()))                       # no duplicates
    assert per[0].tolist() == [0, 3, 6, 12] and per[2].tolist() == [2, 5, 8, 14]
    assert per[1].tolist() == [1]                                      # only from the node BC


def test_constraints_callable_value_uses_dof_locations():
    b = FakeBasis()
    bc = BC(Region("n", "nodes", [2]), (0, 1, 2), lambda X: np.stack([X[0] * 0 + 1, X[1] * 0 + 2, X[2] * 0 + 3]))
    D, x0, _ = solver.constraints(_model(bc), b)
    assert x0[6] == 1 and x0[7] == 2 and x0[8] == 3                    # node 2: dofs 6,7,8


def test_point_force_is_shared_equally_and_totals_preserved():
    case = LoadCase("p", [PointForce(Region("n", "nodes", [0, 2]), [3.0, 6.0, 9.0])])
    f = solver.assemble_loads(_model(), case, FakeBasis(), None)
    assert f[[0, 6]].tolist() == [1.5, 1.5] and f[[1, 7]].tolist() == [3.0, 3.0] and f[[2, 8]].tolist() == [4.5, 4.5]
    assert [f[solver.component_dofs(FakeBasis(), c)].sum() for c in range(3)] == [3.0, 6.0, 9.0]


def test_non_finite_load_vector_is_rejected():
    case = LoadCase("p", [PointForce(Region("n", "nodes", [0]), [np.nan, 0.0, 0.0])])
    try:
        solver.assemble_loads(_model(), case, FakeBasis(), None)
    except ValueError as e:
        assert "NaN or inf" in str(e)
        return
    raise AssertionError("expected ValueError")


def test_equilibrium_error():
    f = solver.equilibrium_error
    assert f([0, -10, 0], [0, 10, 0], [10.0]) == 0.0                       # balanced
    assert np.isclose(f([0, -10, 0], [0, 9, 0], [9.0]), 1 / 10)            # 10 % off
    # displacement-only model: applied = 0, reactions self-equilibrate -> small, not 1e291
    assert f([0, 0, 0], [1e-9, 0, 0], [5e6, -5e6]) < 1e-12
    assert f([0, 0, 0], [0, 0, 0], [0.0]) == 0.0                           # nothing at all


def test_results_stress_accessors():
    S = np.zeros((3, 3, 2)); S[0, 0] = [100.0, 40.0]
    r = solver.Results(basis=None, u=np.zeros(1), u_nodal=np.zeros((3, 2)), reactions=np.zeros(3),
                       applied=np.zeros(3), equilibrium_error=0.0, eps_q=None, sig_q=S[..., None],
                       eps_cell=None, sig_cell=S, eps_nodal=None, sig_nodal=S)
    assert np.allclose(r.von_mises(), [100, 40]) and np.allclose(r.von_mises("cell"), [100, 40])
    assert np.allclose(r.principal()[0][0], [100, 40]) and np.allclose(r.von_mises("q")[:, 0], [100, 40])
