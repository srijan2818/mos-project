"""post.strain_stress_q / cell_average with a fake basis that has a known constant gradient."""
from types import SimpleNamespace

import numpy as np

from mos.core import Material, post


def test_strain_stress_from_known_gradient():
    A = np.array([[1e-3, 2e-3, 0.0], [0.0, -5e-4, 1e-3], [3e-3, 0.0, 2e-4]])   # u = A x
    grad = np.broadcast_to(A[:, :, None, None], (3, 3, 2, 4)).copy()
    basis = SimpleNamespace(interpolate=lambda u: SimpleNamespace(grad=grad))
    E, nu = 200e9, 0.3
    mat = Material.isotropic(E, nu)
    eps, sig = post.strain_stress_q(basis, None, mat.C4)
    eps0 = 0.5 * (A + A.T)
    assert eps.shape == sig.shape == (3, 3, 2, 4)
    assert np.allclose(eps[:, :, 1, 3], eps0)
    lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
    assert np.allclose(sig[:, :, 0, 0], lam * np.trace(eps0) * np.eye(3) + 2 * mu * eps0)   # Hooke
    assert np.allclose(sig, np.swapaxes(sig, 0, 1))


def test_cell_average_is_quadrature_weighted():
    T = np.zeros((3, 3, 2, 2)); T[0, 0, :, 1] = 1.0                   # value 1 at 2nd qp only
    dx = np.array([[1.0, 3.0], [2.0, 2.0]])
    avg = post.cell_average(T, dx)
    assert avg.shape == (3, 3, 2) and np.allclose(avg[0, 0], [0.75, 0.5])
