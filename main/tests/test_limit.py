"""Proportional-limit check: utilisation criteria, material/spec inputs, linear load factor."""
import numpy as np
import pytest

from mos.core import post
from mos.core.material import Material
from mos.core.spec import SpecError, parse_spec


def _tensor(s):
    return np.asarray(s, float).reshape(3, 3, 1)


def test_von_mises_uniaxial_and_shear():
    assert post.utilisation(_tensor([[200e6, 0, 0], [0, 0, 0], [0, 0, 0]]), 250e6)[0] == pytest.approx(0.8)
    tau = 100e6   # pure shear: sigma_vM = sqrt(3) tau
    U = post.utilisation(_tensor([[0, tau, 0], [tau, 0, 0], [0, 0, 0]]), 250e6)[0]
    assert U == pytest.approx(np.sqrt(3) * tau / 250e6)


@pytest.mark.parametrize("S, expected", [
    ([[3e6, 0, 0], [0, 0, 0], [0, 0, 0]], 3e6 / 3e6),            # pure tension: s1 / St
    ([[0, 0, 0], [0, -6e6, 0], [0, 0, 0]], 6e6 / 12e6),          # pure compression: -s3 / Sc
    ([[0, 1e6, 0], [1e6, 0, 0], [0, 0, 0]], 1e6 / 3e6 + 1e6 / 12e6),   # shear: tau (1/St + 1/Sc)
])
def test_coulomb_mohr(S, expected):
    assert post.utilisation(_tensor(S), 3e6, 12e6)[0] == pytest.approx(expected)


def test_material_limits_validated():
    m = Material.isotropic(210e9, 0.3, 7850, 240e6)
    assert m.sigma_pl == 240e6 and m.sigma_pl_c is None
    with pytest.raises(ValueError):
        Material.isotropic(210e9, 0.3, 7850, -1.0)
    with pytest.raises(ValueError):
        Material.isotropic(30e9, 0.2, 2400, None, 12e6)     # compression limit alone


def test_spec_material_limits():
    spec = parse_spec("mesh box 1 1 1 2 2 2\nmaterial E = 30e9, nu = 0.2, sigma_pl = 3e6, sigma_pl_c = 12e6")
    assert spec.material.sigma_pl == pytest.approx(3e6)
    assert spec.material.sigma_pl_c == pytest.approx(12e6)
    assert spec.params["sigma_pl"] == pytest.approx(3e6)
    assert parse_spec("mesh box 1 1 1 2 2 2\nmaterial E = 1e9, nu = 0.3").material.sigma_pl is None
    with pytest.raises(SpecError):
        parse_spec("mesh box 1 1 1 2 2 2\nmaterial E = 1e9, nu = 0.3, sigma_pl = 0")



@pytest.mark.parametrize("limits", [(240e6, None), (3e6, 12e6)])
def test_load_factor_is_exact_for_linear_solve(limits):
    """Scaling the loads by 1 / max(U) brings max(U) to 1 (linear solve, round-off only)."""
    pytest.importorskip("skfem")
    from mos.core import BC, LoadCase, Model, Traction, facets_where
    from mos.core.meshes import box_mesh
    from mos.core.solver import solve_case
    mesh = box_mesh(10.0, 1.0, 1.0, 10, 2, 2)
    fixed = facets_where(mesh, lambda x, y, z: np.isclose(x, 0), "fixed")
    tip = facets_where(mesh, lambda x, y, z: np.isclose(x, 10.0), "tip")
    mat = Material.isotropic(210e9, 0.3, 0.0, *limits)
    model = Model(mesh, mat, [BC(fixed)])
    U = solve_case(model, LoadCase("a", [Traction(tip, [0, -1e5, 0])])).utilisation(mat)
    lam = 1.0 / U.max()
    U2 = solve_case(model, LoadCase("b", [Traction(tip, [0, -1e5 * lam, 0])])).utilisation(mat)
    assert np.isclose(U2.max(), 1.0, rtol=1e-9, atol=0.0)
