"""Linear-elastic 3D solver on scikit-fem.

Supported bodies use static elasticity. Support-free bodies use a quasi-static accelerating
body-fixed frame at an instant of zero spin, with Newton-Euler acceleration and D'Alembert
inertia loads. The real scikit-fem path still requires integration validation in an environment
where the dependency is installed; see docs/testing.md.

Pipeline: stiffness K -> loads -> physical constraints or free-body gauge -> solve ->
strain/stress/resultants. Everything scikit-fem-specific lives in this file.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from skfem import (Basis, BilinearForm, ElementHex1, ElementHex2, ElementTetP1,
                   ElementTetP2, ElementVector, FacetBasis, LinearForm, MeshHex,
                   MeshTet, asm, condense, solve)

try:
    from skfem import DiscreteField
except ImportError:  # older layouts
    from skfem.element import DiscreteField

from . import post
from .loads import (BodyForce, Moment, PointForce, Pressure, SurfaceForce,
                    Traction, cross0, moment_coefficients)
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
            arr = np.asarray(arr)
            parts.append((arr[c] if arr.ndim > 1 and arr.shape[0] == 3
                          else arr[c::3]).ravel())
    return np.concatenate(parts)


def rigid_body_modes(basis, center=None) -> np.ndarray:
    """Return six independent rigid translations/rotations sampled at vector DOFs.

    Rotational modes are scaled by the mesh extent before an orthonormal basis is
    formed.  These modes are used to verify support sufficiency and, for a free body,
    to choose a numerical body-fixed gauge without pinning physical displacement DOFs.
    """
    X = np.asarray(_v(basis.doflocs), float)
    if X.shape != (3, basis.N):
        raise ValueError("basis.doflocs must have shape (3, basis.N)")
    if center is None:
        p = np.asarray(basis.mesh.p, float)
        center = p.mean(axis=1)
    center = np.asarray(center, float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("rigid-mode center must be a finite 3-vector")
    p = np.asarray(basis.mesh.p, float)
    length = float(np.linalg.norm(np.ptp(p, axis=1)))
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("mesh must have a non-zero finite extent")

    modes = np.zeros((basis.N, 6), dtype=float)
    dofs = [component_dofs(basis, c) for c in range(3)]
    for c, ids in enumerate(dofs):
        modes[ids, c] = 1.0
    r = X - center[:, None]
    axes = np.eye(3)
    for j, axis in enumerate(axes):
        rotation = cross0(axis, r) / length
        for c, ids in enumerate(dofs):
            modes[ids, 3 + j] = rotation[c, ids]
    q, triangular = np.linalg.qr(modes, mode="reduced")
    if np.linalg.matrix_rank(triangular) != 6:
        raise ValueError("mesh DOFs do not span all six rigid-body modes")
    return q


def load_resultants(f, basis, origin=None):
    """Return the force and moment resultants represented by an assembled FE vector.

    The moment is evaluated about ``origin`` (global origin by default).  Summing each
    component's scalar shape-function loads at its DOF coordinates exactly preserves
    resultants for the P1/P2 and Q1/Q2 Lagrange bases used by this solver.
    """
    f = np.asarray(f, float)
    X = np.asarray(_v(basis.doflocs), float)
    if f.shape != (basis.N,) or X.shape != (3, basis.N):
        raise ValueError("load vector and basis DOF coordinates have incompatible shapes")
    origin = np.zeros(3) if origin is None else np.asarray(origin, float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("moment origin must be a finite 3-vector")
    dofs = [component_dofs(basis, c) for c in range(3)]
    vals = [f[d] for d in dofs]
    coords = [X[:, d] - origin[:, None] for d in dofs]
    force = np.array([v.sum() for v in vals], dtype=float)
    moment = np.array([
        np.dot(coords[2][1], vals[2]) - np.dot(coords[1][2], vals[1]),
        np.dot(coords[0][2], vals[0]) - np.dot(coords[2][0], vals[2]),
        np.dot(coords[1][0], vals[1]) - np.dot(coords[0][1], vals[0]),
    ], dtype=float)
    return force, moment


def mass_properties(basis, density: float):
    """Return volume, volume centroid, mass and inertia tensor for uniform density."""
    X = np.asarray(_v(basis.global_coordinates()), float)
    dx = np.asarray(basis.dx, float)
    if X.shape[0] != 3 or X.shape[1:] != dx.shape:
        raise ValueError("basis coordinates and quadrature weights have incompatible shapes")
    # Quadrature weights are signed coefficients, not physical pointwise volumes.
    # Some valid quadrature rules use negative weights; rejecting every negative dx
    # incorrectly prevents an otherwise valid mesh from being solved.  What matters
    # here is that coordinates and weights are finite and that the *integrated* volume
    # below is positive.  For free-body dynamics, mass/inertia are checked again for
    # physical validity before accelerations are computed.
    if not np.isfinite(X).all():
        raise ValueError("mesh quadrature coordinates contain NaN or infinity")
    if not np.isfinite(dx).all():
        raise ValueError("mesh quadrature weights contain NaN or infinity")
    volume = float(dx.sum())
    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError("mesh must have positive finite volume")
    center = (X * dx[None, ...]).sum(axis=(1, 2)) / volume
    density = float(density)
    if not np.isfinite(density) or density < 0.0:
        raise ValueError("density must be finite and non-negative")
    mass = density * volume
    r = X - center[:, None, None]
    rflat = r.reshape(3, -1)
    dm = (density * dx).ravel()
    inertia = np.eye(3) * np.dot(dm, np.sum(rflat * rflat, axis=0)) - (rflat * dm) @ rflat.T
    return volume, center, mass, inertia


def body_frame_acceleration(force, moment, mass, inertia):
    """Newton-Euler translational/angular acceleration at an instant of zero spin."""
    force, moment = np.asarray(force, float), np.asarray(moment, float)
    if force.shape != (3,) or moment.shape != (3,):
        raise ValueError("force and moment must be 3-vectors")
    if not np.isfinite(force).all() or not np.isfinite(moment).all():
        raise ValueError("force and moment resultants must be finite")
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("a positive material density is required for an unbalanced free body")
    inertia = np.asarray(inertia, float)
    if inertia.shape != (3, 3) or not np.isfinite(inertia).all():
        raise ValueError("inertia tensor must be a finite 3x3 matrix")
    if not np.allclose(inertia, inertia.T, rtol=1e-10, atol=1e-12):
        raise ValueError("inertia tensor must be symmetric")
    eigenvalues = np.linalg.eigvalsh(inertia)
    if eigenvalues[0] <= 0.0:
        raise ValueError("inertia tensor must be positive definite")
    if eigenvalues[-1] / eigenvalues[0] > 1e12:
        raise ValueError("body inertia tensor is ill-conditioned")
    return force / mass, np.linalg.solve(inertia, moment)


def inertial_body_force(X, density, center, acceleration, angular_acceleration):
    """D'Alembert force density for an instantaneously non-rotating body-fixed frame.

    The omitted centrifugal/Coriolis terms are zero under the explicit zero-spin,
    quasi-static assumption.  Euler acceleration is represented by the distributed
    force ``-rho * (a + alpha x r)``, whose force/moment resultants cancel the applied
    Newton-Euler resultants.
    """
    X = np.asarray(X, float)
    center = np.asarray(center, float)
    acceleration = np.asarray(acceleration, float)
    alpha = np.asarray(angular_acceleration, float)
    density = float(density)
    if X.ndim != 3 or X.shape[0] != 3 or not np.isfinite(X).all():
        raise ValueError("X must be a finite (3, cells, quadrature-points) array")
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("center must be a finite 3-vector")
    if acceleration.shape != (3,) or not np.isfinite(acceleration).all():
        raise ValueError("acceleration must be a finite 3-vector")
    if alpha.shape != (3,) or not np.isfinite(alpha).all():
        raise ValueError("angular acceleration must be a finite 3-vector")
    if not np.isfinite(density) or density < 0.0:
        raise ValueError("density must be finite and non-negative")
    r = X - center[:, None, None]
    return -density * (acceleration[:, None, None] + cross0(alpha, r))


def solve_free_body_system(K, f, modes, equilibrium_scale=None):
    """Solve ``K u = f`` with rigid modes gauged, without fixing physical DOFs."""
    f = np.asarray(f, float)
    modes = np.asarray(modes, float)
    if modes.ndim != 2 or modes.shape[0] != len(f) or modes.shape[1] != 6:
        raise ValueError("rigid-mode matrix must have shape (ndofs, 6)")
    if not np.isfinite(f).all() or not np.isfinite(modes).all():
        raise ValueError("free-body system contains non-finite values")
    if np.linalg.matrix_rank(modes) != 6:
        raise ValueError("rigid-mode matrix must have rank six")
    R = sparse.csc_matrix(modes)
    augmented = sparse.bmat([[K.tocsc(), R], [R.T, None]], format="csc")
    rhs = np.concatenate([f, np.zeros(6)])
    solution = np.asarray(spsolve(augmented, rhs), float)
    if not np.isfinite(solution).all():
        raise ValueError("free-body solve failed; check mesh quality and load equilibrium")
    u = solution[:len(f)]
    multipliers = solution[len(f):]
    scale = max(float(np.linalg.norm(f) if equilibrium_scale is None else equilibrium_scale),
                1e-300)
    if np.linalg.norm(multipliers) > 1e-8 * scale:
        raise ValueError("effective free-body loads are not in rigid-body equilibrium; inertial loads failed to balance them")
    return u


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
    """Global load vector for point, resultant-force, surface and volume loads.

    ``force`` on nodes is shared equally over those nodes; ``force`` on facets is a
    total resultant distributed as uniform traction over the selected surface.
    Explicit ``traction`` remains force per area.
    """
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
            if isinstance(load, SurfaceForce):
                area = float(np.asarray(fb.dx).sum())
                if not np.isfinite(area) or area <= 0.0:
                    raise ValueError("surface force region must have positive finite area")
                t = np.broadcast_to(load.force.reshape(3, 1, 1) / area, X.shape)
            elif isinstance(load, Traction):
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
    """Return constrained DOFs, prescribed values, and per-component DOF sets.

    Overlapping supports may prescribe the same value, but contradictory values on the same
    DOF are physically incompatible and are rejected instead of depending on directive order.
    """
    x0 = basis.zeros()
    assigned = np.zeros(basis.N, dtype=bool)
    per_comp = [set(), set(), set()]
    for bc in model.bcs:
        reg = bc.region
        dofs = basis.get_dofs(**({"nodes": reg.ids} if reg.kind == "nodes"
                                 else {"facets": reg.ids}))
        region_dofs = np.asarray(dofs.all())
        for c in bc.components:
            d = np.intersect1d(region_dofs, component_dofs(basis, c))
            if not d.size:
                continue
            if callable(bc.value):
                values = np.asarray(bc.value(basis.doflocs[:, d])[c], dtype=float)
                values = np.broadcast_to(values, d.shape)
            else:
                values = np.full(d.shape, float(bc.value), dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("prescribed displacements must be finite")
            overlap = assigned[d]
            if overlap.any() and not np.allclose(x0[d[overlap]], values[overlap], rtol=1e-9, atol=1e-12):
                raise ValueError(f"conflicting prescribed displacements on component {c} at DOFs {d[overlap].tolist()}")
            new = ~overlap
            x0[d[new]] = values[new]
            assigned[d] = True
            per_comp[c].update(d.tolist())
    D = np.flatnonzero(assigned)
    return D, x0, [np.array(sorted(s), dtype=int) for s in per_comp]


# ----------------------------------------------------------------- results
@dataclass
class Results:
    """Everything produced by one solve.

    Attributes (nel = cells, nqp = quadrature points per cell, nn = mesh vertices):

    basis              the scikit-fem Basis used (P2/Q2 vector element by default)
    u                  all DOF values, shape (ndofs,)
    u_nodal            displacement at mesh vertices, (3, nn)
    applied            resultant of user-applied external forces, (3,)
    reactions          resultant of physical support reactions; gauge constraints are not reactions
    applied_moment     external moment about the body's volume centroid, (3,)
    reaction_moment    support-reaction moment about the same centroid, (3,)
    inertial_force     resultant of D'Alembert force density, (3,)
    inertial_moment    moment of the D'Alembert force density, (3,)
    translational_acceleration, angular_acceleration  Newton-Euler acceleration used for free bodies
    equilibrium_error  maximum of relative resultant force and moment residuals
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
    equilibrium_error: float    # maximum normalized force and moment resultant residual
    eps_q: np.ndarray           # (3, 3, nel, nqp) strain at quadrature points
    sig_q: np.ndarray           # (3, 3, nel, nqp) stress at quadrature points
    eps_cell: np.ndarray        # (3, 3, nel)
    sig_cell: np.ndarray
    eps_nodal: np.ndarray       # (3, 3, nnodes) smoothed, for plotting
    sig_nodal: np.ndarray
    applied_moment: np.ndarray = field(default_factory=lambda: np.zeros(3))
    reaction_moment: np.ndarray = field(default_factory=lambda: np.zeros(3))
    inertial_force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    inertial_moment: np.ndarray = field(default_factory=lambda: np.zeros(3))
    translational_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    moment_equilibrium_error: float = 0.0
    body_fixed_frame: bool = False

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
    """Solve a load case and recover small-strain stress and strain.

    Supported bodies use ordinary static elasticity and physical support reactions.
    A body with no prescribed-displacement supports is treated as a free body in a
    body-fixed frame at an instant of zero spin: Newton-Euler acceleration is computed
    from the external resultant force/moment, the matching D'Alembert body-force density
    is assembled, and six global gauge constraints remove only rigid translation/rotation.
    This is a quasi-static acceleration-corrected solve, not transient dynamics.  A model
    with some but insufficient supports is rejected instead of being silently pinned.

    order=2 (default) uses P2 tets / Q2 hexes; order=1 is available but far too stiff in bending.
    """
    basis, elem = make_basis(model.mesh, order)
    C4 = model.material.C4
    K = assemble_stiffness(basis, C4)
    f_external = assemble_loads(model, case, basis, elem)
    D, x0, per_comp = constraints(model, basis)
    if model.bcs and not D.size:
        raise ValueError("supports are defined but constrain no displacement DOFs")
    _, center, mass, inertia = mass_properties(basis, model.material.rho)
    applied, applied_moment = load_resultants(f_external, basis, center)
    inertial = np.zeros_like(f_external)
    acceleration = np.zeros(3)
    angular_acceleration = np.zeros(3)
    body_fixed = not model.bcs

    if body_fixed:
        length = float(np.linalg.norm(np.ptp(np.asarray(model.mesh.p, float), axis=1)))
        load_scale = max(float(np.linalg.norm(f_external, ord=1)), np.finfo(float).tiny)
        force_tol, moment_tol = 1e-12 * load_scale, 1e-12 * load_scale * length
        if np.linalg.norm(applied) > force_tol or np.linalg.norm(applied_moment) > moment_tol:
            acceleration, angular_acceleration = body_frame_acceleration(
                applied, applied_moment, mass, inertia)
            X = np.asarray(_v(basis.global_coordinates()), float)
            q_inertial = inertial_body_force(
                X, model.material.rho, center, acceleration, angular_acceleration)
            inertial = asm(_load_form, basis, q=DiscreteField(q_inertial))
        f = f_external + inertial
        # K u = f is singular for a free body.  This saddle-point gauge enforces
        # orthogonality to the six rigid modes without fixing any physical DOF.
        original_scale = np.linalg.norm(f_external) + np.linalg.norm(inertial)
        u = solve_free_body_system(K, f, rigid_body_modes(basis, center), original_scale)
        reactions = np.zeros(3)
        reaction_moment = np.zeros(3)
        r_support = np.zeros_like(f)
    else:
        Rrigid = rigid_body_modes(basis, center)
        if np.linalg.matrix_rank(Rrigid[D, :]) < 6:
            raise ValueError("supports do not remove all six rigid-body modes; add constraints or remove all supports to use the free-body frame")
        f = f_external
        u = solve(*condense(K, f, x=x0, D=D))
        r = np.asarray(K @ u - f, float)
        r_support = np.zeros_like(r)
        r_support[D] = r[D]
        reactions = np.array([r[per_comp[c]].sum() for c in range(3)])
        _, reaction_moment = load_resultants(r_support, basis, center)

    inertial_force, inertial_moment = load_resultants(inertial, basis, center)
    effective_applied, effective_moment = load_resultants(f, basis, center)
    length = float(np.linalg.norm(np.ptp(np.asarray(model.mesh.p, float), axis=1)))
    # A pure couple (or a self-equilibrated load set) can have almost-zero resultant
    # force. Scale resultant residuals by the assembled load-vector norm as well, so
    # harmless round-off is not divided by a near-zero net force. Multiplying that
    # force scale by the body length gives the corresponding moment scale.
    load_vector_scale = float(np.linalg.norm(f_external) + np.linalg.norm(inertial))
    force_scale = max(float(np.linalg.norm(applied) + np.linalg.norm(inertial_force)),
                      float(np.linalg.norm(reactions)), float(np.linalg.norm(r_support[D])),
                      load_vector_scale, 1e-300)
    moment_scale = max(float(np.linalg.norm(applied_moment) + np.linalg.norm(inertial_moment)),
                       float(np.linalg.norm(reaction_moment)),
                       float(np.linalg.norm(r_support[D]) * length),
                       load_vector_scale * length, 1e-300)
    force_error = float(np.linalg.norm(effective_applied + reactions) / force_scale)
    moment_error = float(np.linalg.norm(effective_moment + reaction_moment) / moment_scale)
    err = max(force_error, moment_error)

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
        sig_nodal=post.nodal_average(sig_c, mesh.t, vol, nn),
        applied_moment=applied_moment, reaction_moment=reaction_moment,
        inertial_force=inertial_force, inertial_moment=inertial_moment,
        translational_acceleration=acceleration, angular_acceleration=angular_acceleration,
        moment_equilibrium_error=float(moment_error), body_fixed_frame=body_fixed)
