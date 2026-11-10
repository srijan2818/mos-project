"""Visualization helpers (pure numpy)."""
from types import SimpleNamespace

import numpy as np

from mos import viz


def _res(n=3):
    S = np.zeros((3, 3, n)); S[0, 0] = [100.0, 0.0, 50.0]; S[0, 1] = S[1, 0] = [0.0, 30.0, 0.0]
    E = np.zeros((3, 3, n)); E[0, 1] = E[1, 0] = 1e-3; E[0, 0] = 2e-3
    u = np.array([[3.0, 0, 0], [4.0, 0, 0], [0, 0, 0]])
    return SimpleNamespace(u_nodal=u, sig_nodal=S, eps_nodal=E)


def test_compute_scalars():
    sc = viz.compute_scalars(_res())
    assert np.allclose(sc["|u|"][0], [5, 0, 0]) and sc["|u|"][1] == "disp"
    vm, kind = sc["von Mises"]
    assert kind == "stress" and np.allclose(vm, [100, 30 * np.sqrt(3), 50])
    assert np.allclose(sc["σ1 (max principal)"][0], [100, 30, 50])
    assert np.allclose(sc["σ3 (min principal)"][0], [0, -30, 0])
    assert np.allclose(sc["Tresca"][0], [100, 60, 50])
    assert np.allclose(sc["τxy"][0], [0, 30, 0])
    assert np.allclose(sc["γxy"][0], 2e-3) and sc["γxy"][1] == "strain"   # engineering shear
    assert sc["ε1 (max principal)"][0].shape == (3,)
    assert all(v[0].shape == (3,) for v in sc.values())
    assert viz.UTILISATION not in sc                      # no material limit, no field


def test_compute_scalars_utilisation():
    from mos.core.material import Material
    sc = viz.compute_scalars(_res(), Material.isotropic(1e9, 0.3, 0.0, 100.0))
    U, kind = sc[viz.UTILISATION]
    assert kind == "ratio" and np.allclose(U, np.array([100, 30 * np.sqrt(3), 50]) / 100.0)


def test_suggest_scale_and_clip():
    pts = np.array([[0, 0, 0], [1, 1, 1.0]])
    u = np.array([[0.01, 0], [0, 0], [0, 0]])
    assert np.isclose(viz.suggest_scale(pts, u), 0.1 * np.sqrt(3) / 0.01)
    assert abs(viz.suggest_scale(pts, np.zeros((3, 2))) - 1.0) < 1e-12
    assert np.allclose(viz.clip_origin((0, 10, -1, 1, -2, 2), 0, 0.25), [2.5, 0.0, 0.0], atol=1e-12, rtol=0.0)
    assert np.allclose(viz.clip_origin((0, 10, -1, 1, -2, 2), 2, 1.0), [5.0, 0.0, 2.0], atol=1e-12, rtol=0.0)


def test_load_glyph_scale_tracks_magnitude_with_safe_bounds():
    assert np.isclose(viz.glyph_magnitude_scale(1e5, 1e5), 1.0)
    assert np.isclose(viz.glyph_magnitude_scale(4e5, 1e5), 2.0)
    assert viz.glyph_magnitude_scale(0.0, 1e5) == 0.12
    assert viz.glyph_magnitude_scale(1e12, 1e5) == 2.2
