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
    t = np.asarray(t)
    flat = Tc.reshape(-1, Tc.shape[-1])
    nv = t.shape[0]
    # One sparse scatter (cell -> vertex) instead of np.add.at, which is unbuffered and slow.
    idx = t.ravel()                                   # rows of t concatenated
    w = np.tile(np.asarray(vol, float), nv)
    den = np.bincount(idx, weights=w, minlength=nnodes)
    weighted = np.tile(flat * vol, (1, nv))           # (ncomp, nv*nel), same order as idx
    order = np.argsort(idx, kind="stable")
    sorted_idx = idx[order]
    starts = np.flatnonzero(np.r_[True, sorted_idx[1:] != sorted_idx[:-1]])
    num = np.zeros((flat.shape[0], nnodes))
    num[:, sorted_idx[starts]] = np.add.reduceat(weighted[:, order], starts, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return out.reshape(Tc.shape[:-1] + (nnodes,))


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


def stress_at_point(sig_nodal, points, xyz):
    """Return the (3,3) stress tensor at spatial point xyz by nearest-node lookup.

    sig_nodal: (3,3,nnodes); points: (nnodes,3) or (3,nnodes).
    Returns (tensor (3,3), node_index).
    """
    pts = np.asarray(points, float)
    if pts.shape[0] == 3 and pts.shape[1] != 3:
        pts = pts.T
    xyz = np.asarray(xyz, float).ravel()
    d2 = np.sum((pts - xyz[None, :]) ** 2, axis=1)
    idx = int(np.argmin(d2))
    return np.asarray(sig_nodal[:, :, idx], float), idx


def traction_on_plane(sig, xyz_points, xyz, normal):
    """Stress tensor + traction on a plane at spatial point.

    Returns dict: tensor(3,3), node_index, normal(3,), traction(3,),
    normal_stress(float), shear_stress(float), shear_vector(3,).
    """
    tensor, idx = stress_at_point(sig, xyz_points, xyz)
    n = np.asarray(normal, float).ravel()
    nn = np.linalg.norm(n)
    if not np.isfinite(nn) or nn < 1e-30:
        raise ValueError(f"plane normal {n.tolist()} has no direction")
    n = n / nn
    t = tensor @ n
    sn = float(n @ t)
    tau_vec = t - sn * n
    tau = float(np.linalg.norm(tau_vec))
    return {"tensor": tensor, "node_index": idx, "normal": n,
            "traction": t, "normal_stress": sn, "shear_stress": tau,
            "shear_vector": tau_vec}
