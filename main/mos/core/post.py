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


def _scatter_mean(values, idx, w, nnodes):
    """Weighted mean of ``values`` (ncomp, n) onto ``nnodes`` bins given bin ids ``idx`` (n,).
    One sorted ``reduceat`` instead of np.add.at, which is unbuffered and slow."""
    den = np.bincount(idx, weights=w, minlength=nnodes)
    order = np.argsort(idx, kind="stable")
    sorted_idx = idx[order]
    starts = np.flatnonzero(np.r_[True, sorted_idx[1:] != sorted_idx[:-1]])
    num = np.zeros((values.shape[0], nnodes))
    num[:, sorted_idx[starts]] = np.add.reduceat((values * w)[:, order], starts, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def nodal_average(Tc, t, vol, nnodes):
    """Volume-weighted average of cell values onto vertices.
    Tc: (..., nel); t: (nverts_per_cell, nel) connectivity; vol: (nel,)."""
    t = np.asarray(t)
    flat = Tc.reshape(-1, Tc.shape[-1])
    nv = t.shape[0]
    out = _scatter_mean(np.tile(flat, (1, nv)), t.ravel(), np.tile(np.asarray(vol, float), nv), nnodes)
    return out.reshape(Tc.shape[:-1] + (nnodes,))


def vertex_values(basis, u, C4):
    """Strain and stress evaluated inside each element at its own vertices: (3, 3, nel, nverts).

    The quadrature points of a copy of ``basis`` are put at the reference vertices
    (``mesh.refdom.p``, same order as the rows of ``mesh.t``). For P2/Q2 the stress is linear /
    bilinear inside an element, so this is exact per element; the jump between elements is the
    discretisation error.
    """
    from skfem import Basis
    X = np.asarray(basis.mesh.refdom.p, float)
    vb = Basis(basis.mesh, basis.elem, quadrature=(X, np.ones(X.shape[1])))
    return strain_stress_q(vb, u, C4)


def vertex_average(Tv, t, vol, nnodes):
    """Volume-weighted average of element-vertex values onto mesh vertices.

    Tv: (..., nel, nverts) from :func:`vertex_values`; t: (nverts, nel); vol: (nel,).
    Unlike :func:`nodal_average` of cell means this keeps the linear variation inside each
    element: on the default 20 x 2 x 2 cantilever the mid-span surface stress is within ~1 %
    of beam theory instead of -41 %.
    """
    t = np.asarray(t)
    nel, nv = Tv.shape[-2:]
    flat = np.swapaxes(Tv, -1, -2).reshape(-1, nv * nel)   # vertex-major, matches t.ravel()
    out = _scatter_mean(flat, t.ravel(), np.tile(np.asarray(vol, float), nv), nnodes)
    return out.reshape(Tv.shape[:-2] + (nnodes,))


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


def utilisation(S, sigma_pl, sigma_pl_c=None):
    """Fraction of the proportional limit used, per point, for stresses S (3, 3, ...).

    * ``sigma_pl_c`` is None (ductile): von Mises, ``U = sigma_vM / sigma_pl``.
    * ``sigma_pl_c`` given (brittle Coulomb-Mohr, tension limit St = ``sigma_pl``, compression
      limit Sc = ``sigma_pl_c``): ``U = max(s1, 0) / St + max(-s3, 0) / Sc``. This is s1/St in
      pure tension, -s3/Sc in pure compression and s1/St - s3/Sc in mixed states (pure shear
      fails at tau = St Sc / (St + Sc)).

    Both are positively homogeneous of degree 1 in S, so in a linear solve all loads may be
    scaled by ``1 / max(U)`` before any point reaches U = 1.
    """
    if sigma_pl_c is None:
        return von_mises(S) / sigma_pl
    s, _ = principal(S)
    return np.maximum(s[0], 0.0) / sigma_pl + np.maximum(-s[2], 0.0) / sigma_pl_c


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
