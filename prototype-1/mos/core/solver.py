"""Static linear-elastic 3D solver on scikit-fem.

Written for scikit-fem >= 8. NOT yet run against the real library (it could not be installed
where this was written): the DOF bookkeeping is unit-tested with a fake basis, but the assembly
calls are first exercised by tests/test_fem.py. See docs/testing.md.

Pipeline: stiffness K -> load vector f -> constraints -> solve -> strain/stress/reactions.
Everything scikit-fem-specific lives in this file.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skfem import (Basis, BilinearForm, ElementHex1, ElementHex2, ElementTetP1,
                   ElementTetP2, ElementVector, FacetBasis, LinearForm, MeshHex,
                   MeshTet, asm, condense, solve)

try:
    from skfem import DiscreteField
except ImportError:  # older layouts
    from skfem.element import DiscreteField

from . import post
from .loads import (BodyForce, Moment, PointForce, Pressure, Traction, cross0,
                    moment_coefficients)
from .model import LoadCase, Model


# ----------------------------------------------------------------- helpers
def _v(a):
    """Raw array of a DiscreteField (or the array itself)."""
    return getattr(a, "value", a)


def _dot(a, b):
    return np.einsum("i...,i...->...", _v(a), _v(b))


def make_basis(mesh, order: int = 2):
    """Vector P2 (default) on tets, or Q2 on hexes.  P1/Q1 are too stiff for bending."""
    if isinstance(mesh, MeshTet):
        base = {1: ElementTetP1, 2: ElementTetP2}[order]
    elif isinstance(mesh, MeshHex):
        base = {1: ElementHex1, 2: ElementHex2}[order]
    else:
        raise TypeError("mesh must be a scikit-fem MeshTet or MeshHex")
    elem = ElementVector(base())
    return Basis(mesh, elem), elem


def component_dofs(basis, c: int) -> np.ndarray:
    """All DOF indices belonging to displacement component c (0, 1, 2)."""
    src = basis if hasattr(basis, "nodal_dofs") else basis.dofs
    parts = []
    for name in ("nodal_dofs", "edge_dofs", "facet_dofs", "interior_dofs"):
        arr = getattr(src, name, None)
        if arr is not None and np.size(arr):
            parts.append(np.asarray(arr)[c::3].ravel())
    return np.concatenate(parts)


# ----------------------------------------------------------------- assembly
def assemble_stiffness(basis, C4):
    """Global stiffness matrix K (sparse) for elasticity tensor C4 on `basis`."""
    def sym(f):
        """Symmetric gradient (small strain) of a vector DiscreteField."""
        g = f.grad
        return 0.5 * (g + np.swapaxes(g, 0, 1))

    @BilinearForm
    def stiff(u, v, w):
        """Weak form: integral of sigma(u) : eps(v)."""
        s = np.einsum("ijkl,kl...->ij...", C4, sym(u))
        return np.einsum("ij...,ij...->...", s, sym(v))

    return asm(stiff, basis)


@LinearForm
def _load_form(v, w):
    return _dot(w["q"], v)


def assemble_loads(model: Model, case: LoadCase, basis, elem) -> np.ndarray:
    """Global load vector f for all loads of `case` (point, surface and volume integrals)."""
    mesh = basis.mesh
    f = np.zeros(basis.N)
    for load in case.loads:
        if isinstance(load, PointForce):
            ids = load.region.ids
            for c in range(3):
                f[np.asarray(basis.nodal_dofs)[c, ids]] += load.force[c] / len(ids)
        elif isinstance(load, BodyForce):
            q = load.field(_v(basis.global_coordinates()))
            f += asm(_load_form, basis, q=DiscreteField(q))
        else:
            fb = FacetBasis(mesh, elem, facets=load.region.ids)
            X = _v(fb.global_coordinates())
            if isinstance(load, Traction):
                t = load.field(X)
            elif isinstance(load, Pressure):
                t = -load.p(X) * _v(fb.normals)
            elif isinstance(load, Moment):
                omega, o = moment_coefficients(X.reshape(3, -1), np.asarray(fb.dx).ravel(),
                                               load.M, load.origin)
                t = cross0(omega, X - o.reshape(3, 1, 1))
            else:
                raise TypeError(f"unknown load type {type(load).__name__}")
            f += asm(_load_form, fb, q=DiscreteField(t))
    if not np.isfinite(f).all():
        raise ValueError("the load vector contains NaN or inf - check load expressions "
                         "for division by zero or log/sqrt of negative values")
    return f


def constraints(model: Model, basis):
    """Returns (D, x0, per_component_sets): constrained DOFs, prescribed values."""
    x0 = basis.zeros()
    per_comp = [set(), set(), set()]
    for bc in model.bcs:
        reg = bc.region
        dofs = basis.get_dofs(**({"nodes": reg.ids} if reg.kind == "nodes"
                                 else {"facets": reg.ids}))
        region_dofs = np.asarray(dofs.all())
        for c in bc.components:
            d = np.intersect1d(region_dofs, component_dofs(basis, c))
            if callable(bc.value):
                x0[d] = bc.value(basis.doflocs[:, d])[c]
            else:
                x0[d] = float(bc.value)
            per_comp[c].update(d.tolist())
    D = np.array(sorted(set().union(*per_comp)), dtype=int)
    return D, x0, [np.array(sorted(s), dtype=int) for s in per_comp]


# ----------------------------------------------------------------- results
@dataclass
class Results:
    """Everything produced by one solve.

    Attributes (nel = cells, nqp = quadrature points per cell, nn = mesh vertices):

    basis              the scikit-fem Basis used (P2/Q2 vector element by default)
    u                  all DOF values, shape (ndofs,)
    u_nodal            displacement at mesh vertices, (3, nn)
    reactions          total support reaction force per component, (3,)
    applied            total applied force per component, (3,)
    equilibrium_error  relative force-equilibrium error (see `equilibrium_error`); ~1e-10 or smaller
    eps_q, sig_q       strain / stress tensors at quadrature points, (3, 3, nel, nqp)
    eps_cell, sig_cell quadrature-weighted cell averages, (3, 3, nel)
    eps_nodal, sig_nodal  volume-weighted averages of the cell values at vertices, (3, 3, nn);
                       smooth, for plotting (peaks are underestimated, see docs/theory.md)
    """

    basis: object
    u: np.ndarray               # all DOFs
    u_nodal: np.ndarray         # (3, nnodes) vertex displacements
    reactions: np.ndarray       # (3,) total support reaction
    applied: np.ndarray         # (3,) total applied force
    equilibrium_error: float    # see equilibrium_error()
    eps_q: np.ndarray           # (3, 3, nel, nqp) strain at quadrature points
    sig_q: np.ndarray           # (3, 3, nel, nqp) stress at quadrature points
    eps_cell: np.ndarray        # (3, 3, nel)
    sig_cell: np.ndarray
    eps_nodal: np.ndarray       # (3, 3, nnodes) smoothed, for plotting
    sig_nodal: np.ndarray

    def von_mises(self, where="nodal"):
        """Von Mises stress at 'nodal' (smoothed), 'cell' (averaged) or 'q' (quadrature) points."""
        return post.von_mises({"nodal": self.sig_nodal, "cell": self.sig_cell,
                               "q": self.sig_q}[where])

    def principal(self, where="nodal"):
        """Principal stresses (values, vectors) at 'nodal', 'cell' or 'q' points."""
        return post.principal({"nodal": self.sig_nodal, "cell": self.sig_cell,
                               "q": self.sig_q}[where])


def equilibrium_error(applied, reactions, r_constrained) -> float:
    """Relative global force-equilibrium error |applied + reactions| / scale.

    scale is the larger of |applied| and the norm of the individual reaction values, so a model
    driven only by prescribed displacements (applied = 0) is still judged sensibly.
    """
    scale = max(np.linalg.norm(applied), np.linalg.norm(r_constrained), 1e-300)
    return float(np.linalg.norm(np.asarray(applied) + np.asarray(reactions)) / scale)


def solve_case(model: Model, case: LoadCase, order: int = 2) -> Results:
    """Solve one load case: assemble, constrain, solve, then recover strain, stress, reactions.

    order=2 (default) uses P2 tets / Q2 hexes; order=1 is available but far too stiff in bending.
    """
    basis, elem = make_basis(model.mesh, order)
    C4 = model.material.C4
    K = assemble_stiffness(basis, C4)
    f = assemble_loads(model, case, basis, elem)
    D, x0, per_comp = constraints(model, basis)
    if len(D) == 0:
        raise ValueError("no boundary conditions: the body is free to move (singular K)")

    u = solve(*condense(K, f, x=x0, D=D))

    r = K @ u - f                                   # support reactions live on D
    applied = np.array([f[component_dofs(basis, c)].sum() for c in range(3)])
    reactions = np.array([r[per_comp[c]].sum() for c in range(3)])
    err = equilibrium_error(applied, reactions, r[D])

    eps_q, sig_q = post.strain_stress_q(basis, u, C4)
    dx = np.asarray(basis.dx)
    eps_c, sig_c = post.cell_average(eps_q, dx), post.cell_average(sig_q, dx)
    mesh, nn = model.mesh, model.mesh.p.shape[1]
    vol = dx.sum(axis=1)
    return Results(
        basis=basis, u=u, u_nodal=u[np.asarray(basis.nodal_dofs)],
        reactions=reactions, applied=applied, equilibrium_error=float(err),
        eps_q=eps_q, sig_q=sig_q, eps_cell=eps_c, sig_cell=sig_c,
        eps_nodal=post.nodal_average(eps_c, mesh.t, vol, nn),
        sig_nodal=post.nodal_average(sig_c, mesh.t, vol, nn))
