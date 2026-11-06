"""DOF bookkeeping in mos.core.solver, tested with a fake basis (no scikit-fem needed).

If scikit-fem is missing, a stub is injected ONLY while importing the solver module and removed
afterwards, so it cannot leak into test_fem (which skips cleanly without scikit-fem).
"""
import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
from scipy import sparse

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


def test_conflicting_overlapping_supports_are_rejected():
    bc_a = BC(Region("a", "nodes", [0]), (0,), 0.0)
    bc_b = BC(Region("b", "nodes", [0]), (0,), 0.01)
    with np.testing.assert_raises_regex(ValueError, "conflicting prescribed displacements"):
        solver.constraints(_model(bc_a, bc_b), FakeBasis())


def test_point_force_is_shared_equally_and_totals_preserved():
    case = LoadCase("p", [PointForce(Region("n", "nodes", [0, 2]), [3.0, 6.0, 9.0])])
    f = solver.assemble_loads(_model(), case, FakeBasis(), None)
    assert f[[0, 6]].tolist() == [1.5, 1.5] and f[[1, 7]].tolist() == [3.0, 3.0] and f[[2, 8]].tolist() == [4.5, 4.5]
    assert [f[solver.component_dofs(FakeBasis(), c)].sum() for c in range(3)] == [3.0, 6.0, 9.0]


def test_overlapping_point_loads_superpose_and_duplicate_region_ids_are_set_membership():
    region = Region("n", "nodes", [2, 0, 2])
    assert region.ids.tolist() == [0, 2]
    case = LoadCase("overlap", [
        PointForce(region, [2.0, -4.0, 6.0]),
        PointForce(Region("one", "nodes", [2]), [3.0, 1.0, -2.0]),
    ])
    f = solver.assemble_loads(_model(), case, FakeBasis(), None)
    dofs = [solver.component_dofs(FakeBasis(), c) for c in range(3)]
    total = np.array([f[d].sum() for d in dofs])
    assert np.allclose(total, [5.0, -3.0, 4.0])
    assert np.allclose(f[[0, 6]], [1.0, 4.0])


def test_non_finite_point_force_is_rejected_at_construction():
    with np.testing.assert_raises_regex(ValueError, "must be finite"):
        PointForce(Region("n", "nodes", [0]), [np.nan, 0.0, 0.0])


def test_equilibrium_error():
    f = solver.equilibrium_error
    assert abs(f([0, -10, 0], [0, 10, 0], [10.0]) - 0.0) < 1e-12       # balanced
    assert np.isclose(f([0, -10, 0], [0, 9, 0], [9.0]), 1 / 10)            # 10 % off
    # displacement-only model: applied = 0, reactions self-equilibrate -> small, not 1e291
    assert f([0, 0, 0], [1e-9, 0, 0], [5e6, -5e6]) < 1e-12
    assert abs(f([0, 0, 0], [0, 0, 0], [0.0]) - 0.0) < 1e-12               # nothing at all


def test_load_resultants_preserve_force_and_moment_about_origin():
    b = FakeBasis()
    points = np.array([[0., 1., 0., 0., 0., .5],
                       [0., 0., 1., 0., 0., .5],
                       [0., 0., 0., 1., 0., .5]])
    b.doflocs = np.zeros((3, b.N))
    dofs = [solver.component_dofs(b, c) for c in range(3)]
    for ids in dofs:
        b.doflocs[:, ids] = points
    f = np.zeros(b.N)
    force = np.array([2., -3., 4.])
    for c in range(3):
        f[dofs[c][1]] = force[c]
    origin = np.array([.2, -.1, .4])
    F, M = solver.load_resultants(f, b, origin)
    expected_M = np.cross(points[:, 1] - origin, force)
    assert np.allclose(F, force)
    assert np.allclose(M, expected_M)


def test_body_frame_newton_euler_and_dalembert_resultants():
    # Symmetric 3-D material points with equal quadrature volumes, centered at the COM.
    corners = np.array([[x, y, z] for x in (-1., 1.)
                        for y in (-2., 2.) for z in (-3., 3.)]).T
    X = corners[:, None, :]
    dv = np.ones((1, corners.shape[1]))
    rho = 2.0
    mass = rho * dv.sum()
    center = np.zeros(3)
    r = corners
    I = np.eye(3) * (rho * np.sum(r * r)) - (r * rho) @ r.T
    applied_F = np.array([12., -6., 4.])
    applied_M = np.array([2., 3., -5.])
    a, alpha = solver.body_frame_acceleration(applied_F, applied_M, mass, I)
    q_inertial = solver.inertial_body_force(X, rho, center, a, alpha)
    dF = (q_inertial * dv[None, ...]).sum(axis=(1, 2))
    dM = (np.cross(np.moveaxis(X - center[:, None, None], 0, -1),
                   np.moveaxis(q_inertial, 0, -1)) * dv[..., None]).sum(axis=(0, 1))
    assert np.allclose(a, applied_F / mass)
    assert np.allclose(I @ alpha, applied_M)
    assert np.allclose(dF, -applied_F)
    assert np.allclose(dM, -applied_M)


def test_body_frame_acceleration_tolerates_roundoff_scale_inertia_asymmetry():
    F, M = np.array([1., 2., 3.]), np.array([1., 0., -1.])
    # The skew here is far below the tensor's scale and should be projected away.
    I = np.array([[5., 0.5 + 1e-13, 0.], [0.5, 4., 0.], [0., 0., 3.]])
    I_symmetric = 0.5 * (I + I.T)
    _, alpha = solver.body_frame_acceleration(F, M, 2.0, I)
    assert np.allclose(I_symmetric @ alpha, M, rtol=1e-12, atol=1e-12)


def test_body_frame_acceleration_rejects_nonphysical_inertia_tensors():
    F, M = np.array([1., 2., 3.]), np.array([1., 0., -1.])
    with np.testing.assert_raises_regex(ValueError, "materially asymmetric"):
        solver.body_frame_acceleration(F, M, 2.0, np.array([[1., 1., 0.], [0., 1., 0.], [0., 0., 1.]]))
    with np.testing.assert_raises_regex(ValueError, "positive definite"):
        solver.body_frame_acceleration(F, M, 2.0, np.diag([1., 1., -1.]))
    with np.testing.assert_raises_regex(ValueError, "positive material density"):
        solver.body_frame_acceleration(F, M, 0.0, np.eye(3))


def test_free_body_gauge_does_not_pin_physical_dofs_and_rejects_unbalanced_rhs():
    # Six zero-energy rigid modes, six unit-stiffness deformation modes.
    K = sparse.diags([0.] * 6 + [2.] * 6, format="csc")
    modes = np.eye(12)[:, :6]
    f = np.r_[np.zeros(6), np.array([2., 4., 6., 8., 10., 12.])]
    u = solver.solve_free_body_system(K, f, modes)
    assert np.allclose(u[:6], 0.0)
    assert np.allclose(u[6:], f[6:] / 2.0)
    with np.testing.assert_raises_regex(ValueError, "not in rigid-body equilibrium"):
        solver.solve_free_body_system(K, f + np.r_[1., np.zeros(11)], modes)


def test_results_stress_accessors():
    S = np.zeros((3, 3, 2)); S[0, 0] = [100.0, 40.0]
    r = solver.Results(basis=None, u=np.zeros(1), u_nodal=np.zeros((3, 2)), reactions=np.zeros(3),
                       applied=np.zeros(3), equilibrium_error=0.0, eps_q=None, sig_q=S[..., None],
                       eps_cell=None, sig_cell=S, eps_nodal=None, sig_nodal=S)
    assert np.allclose(r.von_mises(), [100, 40]) and np.allclose(r.von_mises("cell"), [100, 40])
    assert np.allclose(r.principal()[0][0], [100, 40]) and np.allclose(r.von_mises("q")[:, 0], [100, 40])


def test_mass_properties_accepts_finite_signed_quadrature_weights():
    # Quadrature coefficients may be signed even though the integrated volume is positive.
    # Rejecting any negative coefficient is not a valid general quadrature check.
    points = np.array([[x, y, z] for x in (-1.0, 1.0)
                       for y in (-1.0, 1.0) for z in (-1.0, 1.0)]).T
    weights = np.ones((1, points.shape[1]))
    weights[0, 0] = -0.1
    basis = SimpleNamespace(
        global_coordinates=lambda: points[:, None, :],
        dx=weights,
    )
    volume, center, mass, inertia = solver.mass_properties(basis, 2.0)
    assert abs(volume - np.sum(weights)) < 1e-12
    assert np.isfinite(center).all() and np.isfinite(mass)
    assert np.isfinite(inertia).all()
    assert np.array_equal(inertia, inertia.T)  # explicitly symmetrized by construction
    assert mass > 0.0 and np.linalg.eigvalsh(inertia).min() > 0.0


def test_mass_properties_is_translation_invariant_for_far_from_origin_mesh():
    points = np.array([[x, y, z] for x in (-1.0, 1.0)
                       for y in (-2.0, 2.0) for z in (-3.0, 3.0)]).T
    weights = np.ones((1, points.shape[1]))
    origin = np.array([1e9, -2e9, 3e9])
    def make_basis(coords):
        return SimpleNamespace(
            global_coordinates=lambda: coords[:, None, :],
            dx=weights,
        )
    _, c0, _, I0 = solver.mass_properties(make_basis(points), 2.0)
    _, c1, _, I1 = solver.mass_properties(make_basis(points + origin[:, None]), 2.0)
    assert np.allclose(c1 - origin, c0, rtol=0.0, atol=2e-7)
    assert np.allclose(I1, I0, rtol=1e-7, atol=1e-7)


def test_inertia_symmetry_guard_rejects_material_skew_before_cleaning():
    raw = np.array([[5.0, 0.5, 0.0], [0.5001, 4.0, 0.0], [0.0, 0.0, 3.0]])
    with np.testing.assert_raises_regex(ValueError, "materially asymmetric.*not being silently symmetrized"):
        solver._check_and_clean_inertia_symmetry(raw, source="test integration")


def test_inertia_symmetry_guard_only_cleans_roundoff_scale_skew():
    raw = np.array([[5.0, 0.5 + 1e-14, 0.0], [0.5, 4.0, 0.0], [0.0, 0.0, 3.0]])
    clean = solver._check_and_clean_inertia_symmetry(raw, source="test integration")
    assert np.array_equal(clean, clean.T)
    assert np.allclose(clean, 0.5 * (raw + raw.T), rtol=0.0, atol=0.0)


def test_mass_properties_reports_non_finite_quadrature_weights_precisely():
    basis = SimpleNamespace(
        global_coordinates=lambda: np.zeros((3, 1, 1)),
        dx=np.array([[np.nan]]),
    )
    with np.testing.assert_raises_regex(ValueError, "quadrature weights contain NaN or infinity"):
        solver.mass_properties(basis, 1.0)


def test_live_load_response_uses_unit_fields_without_backsolving_on_evaluate(monkeypatch):
    """Changing an already-prepared point force only composes cached responses."""
    region = Region("tip", "nodes", [0])
    other = PointForce(Region("other", "nodes", [1]), [7.0, 8.0, 9.0])
    active = PointForce(region, [1.0, 2.0, 3.0])
    case = LoadCase("live", [other, active])
    calls = {"solve": 0}

    def assemble(_model, load_case, _basis, _elem):
        total = np.zeros(3)
        for load in load_case.loads:
            total += np.asarray(load.force, float)
        return total

    def solve_displacement(force, *, prescribed):
        calls["solve"] += 1
        offset = np.array([10.0, 20.0, 30.0]) if prescribed else np.zeros(3)
        return offset + 2.0 * np.asarray(force, float)

    fake_solver = SimpleNamespace(
        model=SimpleNamespace(mesh=SimpleNamespace(p=np.zeros((3, 2)))),
        basis=object(), elem=None,
        _solve_displacement=solve_displacement,
        _recover=lambda displacement, force: (displacement.copy(), force.copy()),
    )
    monkeypatch.setattr(solver, "assemble_loads", assemble)

    response = solver.LiveLoadResponse(fake_solver, case, 1)
    solves_after_prepare = calls["solve"]
    changed = PointForce(region, [4.0, 5.0, 6.0])
    displacement, force = response.evaluate(changed)

    assert calls["solve"] == solves_after_prepare
    assert np.allclose(force, [11.0, 13.0, 15.0])
    assert np.allclose(displacement, [32.0, 46.0, 60.0])
    assert response.matches(changed)
    assert not response.matches(PointForce(Region("other-target", "nodes", [1]), [4, 5, 6]))


def test_cached_linear_solver_compatibility_includes_density():
    """Density changes must invalidate live-load baselines when gravity is present."""
    mesh = object()
    C = np.eye(6)
    model = SimpleNamespace(
        mesh=mesh,
        material=SimpleNamespace(C=C.copy(), rho=7850.0),
        bcs=[],
    )
    cache = object.__new__(solver.CachedLinearSolver)
    cache._mesh_identity = id(mesh)
    cache._material_matrix = C.copy()
    cache._material_density = 7850.0
    cache._support_signature = solver.CachedLinearSolver._make_support_signature(model)

    assert cache.compatible(model)
    changed_density = SimpleNamespace(
        mesh=mesh,
        material=SimpleNamespace(C=C.copy(), rho=7800.0),
        bcs=[],
    )
    assert not cache.compatible(changed_density)

    changed_stiffness = SimpleNamespace(
        mesh=mesh,
        material=SimpleNamespace(C=2.0 * C, rho=7850.0),
        bcs=[],
    )
    assert not cache.compatible(changed_stiffness)
