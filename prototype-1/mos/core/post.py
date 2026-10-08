"""Post-processing on plain arrays. Tensors have shape (3, 3, ...)."""
from __future__ import annotations

import numpy as np


def strain_stress_q(basis, u, C4):
    """Strain and stress tensors at quadrature points: (3, 3, nel, nqp)."""
    g = basis.interpolate(u).grad                  # g[i, j] = d u_i / d x_j
    eps = 0.5 * (g + np.swapaxes(g, 0, 1))
    sig = np.einsum("ijkl,kl...->ij...", C4, eps)
    return eps, sig


def cell_average(T, dx):
    """Quadrature-weighted average per cell: (3,3,nel,nqp) -> (3,3,nel)."""
    return (T * dx).sum(axis=-1) / dx.sum(axis=-1)


def nodal_average(Tc, t, vol, nnodes):
    """Volume-weighted average of cell values onto vertices (for smooth plots).
    Tc: (..., nel); t: (nverts_per_cell, nel) connectivity; vol: (nel,)."""
    flat = Tc.reshape(-1, Tc.shape[-1])
    num = np.zeros((flat.shape[0], nnodes))
    den = np.zeros(nnodes)
    for k in range(t.shape[0]):
        np.add.at(num, (slice(None), t[k]), flat * vol)
        np.add.at(den, t[k], vol)
    return (num / den).reshape(Tc.shape[:-1] + (nnodes,))


def von_mises(S):
    """Von Mises equivalent stress of a tensor field S with shape (3, 3, ...)."""
    sxx, syy, szz = S[0, 0], S[1, 1], S[2, 2]
    sxy, syz, sxz = S[0, 1], S[1, 2], S[0, 2]
    return np.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                   + 3.0 * (sxy ** 2 + syz ** 2 + sxz ** 2))


def principal(S):
    """Principal values s1 >= s2 >= s3, shape (3, ...); vectors shape (3, 3, ...)
    with vectors[:, k] the direction of s_{k+1}."""
    shape = S.shape[2:]
    A = np.moveaxis(S.reshape(3, 3, -1), -1, 0)
    A = 0.5 * (A + np.swapaxes(A, 1, 2))
    vals, vecs = np.linalg.eigh(A)
    vals, vecs = vals[:, ::-1], vecs[:, :, ::-1]
    return (np.moveaxis(vals, 0, -1).reshape((3,) + shape),
            np.moveaxis(vecs, 0, -1).reshape((3, 3) + shape))


def tresca(S):
    """Tresca equivalent stress (max shear * 2 = s1 - s3) of S with shape (3, 3, ...)."""
    s, _ = principal(S)
    return s[0] - s[2]
