# Theory and conventions

What the solver computes, and the exact formulas behind it. Equations are written in plain text:
`:` is the double contraction, `x` between vectors is the cross product, `^T` is the transpose.

## 1. The problem

Static, linear-elastic, small-strain equilibrium of a body Omega:

```text
div(sigma) + b = 0        in Omega        (b = body force per volume)
u = u_bar                 on Gamma_D      (supports / prescribed displacement)
sigma n = t               on Gamma_N      (traction; n is the outward normal)
sigma = C : eps,   eps = sym(grad u) = (grad u + grad u^T) / 2
```

Weak form (what the finite-element method solves): find `u` such that for every test function `v`

```text
integral_Omega  sigma(u) : eps(v)  dV  =  integral_Omega b . v dV
                                        + integral_Gamma_N t . v dS
                                        + sum_i F_i . v(x_i)
```

The three right-hand terms are exactly the loads: `body`, `traction`/`pressure`/`moment`, and `force`.

## 2. Material: the C matrix

`C` is the 6x6 stiffness in **Voigt order** `xx, yy, zz, yz, xz, xy` with **engineering shear
strains** (`gamma = 2 eps`), so `sigma_voigt = C @ eps_voigt`. `Material` checks that `C` is
symmetric and positive definite. The 3x3x3x3 tensor `C4` used in the code satisfies
`sigma_ij = C4_ijkl eps_kl`, with `C4[i,j,k,l] = C[v(i,j), v(k,l)]` for all index orders.

Isotropic material (`Material.isotropic(E, nu)`):

```text
lambda = E nu / ((1 + nu)(1 - 2 nu)),      mu = E / (2 (1 + nu))
C[0:3, 0:3] = lambda,   C[i, i] = lambda + 2 mu  (i < 3),   C[3,3] = C[4,4] = C[5,5] = mu
```

so `sigma = lambda tr(eps) I + 2 mu eps`. The text format only creates isotropic materials; the
Python `Material(C, rho)` accepts any valid `C` (orthotropic, ...).

## 3. Discretisation

The displacement is a vector finite-element field. By default **quadratic** elements: P2 on
tetrahedra, Q2 on hexahedra (`solve_case(..., order=2)`). Linear elements (P1/Q1) have constant
strain per tetrahedron, so they cannot represent the linear bending strain through a beam's
depth and come out too stiff on coarse meshes (a well-known behaviour; not measured here). P2 is
the default for that reason.

* Stiffness matrix: `K_ij = integral sigma(phi_j) : eps(phi_i) dV`.
* `body`: `f_i = integral b . phi_i dV`, evaluating the field at the quadrature points.
* `traction`: `f_i = integral_facets t . phi_i dS`. `pressure` is the same with `t = -p n`.
* `force` on a node region: the total force is shared equally among the selected vertex nodes.
  `force` on a facet region: the total force is converted to uniform traction `F / area` and
  integrated over the full facet set. These represent the same resultant but different load
  distributions; the latter avoids dependence on how many mesh nodes happen to lie on the face.
  `SurfaceForce` stores that total-force surface load in the Python API.

Solution vector `u` holds all DOFs; `Results.u_nodal` is its restriction to the mesh vertices.
(Mid-edge DOFs of P2 are not shown in the viewer.)

## 4. Moments and torque

A solid mesh has no rotational DOFs, so a moment `M` is represented as a traction on a set of faces.
The traction is centred at the area centroid `c` of the selected face set:

```text
t(x) = omega x (x - c),    r = x - c
A omega = M,               A = integral ( |r|^2 I - r r^T ) dA
```

`A` is the area-inertia tensor of the face set about `c`, and `omega` is `A^-1 M`. Then
`integral r x t dA = A omega = M`, so the **net moment is `M`** (to quadrature accuracy; the same
quadrature that applies the load is used to build `A`). The **net force is zero** because
`integral t dA = omega x integral r dA = 0`. The optional `about x, y, z` point only specifies a
reference point for reporting the moment. Since a pure couple has zero resultant force, its moment
is independent of reference point; specifying `about` must not introduce a fictitious net force.

Two special cases, for a planar face with outward normal `+x` and principal axes `y`, `z`:

```text
torsion  M = (T, 0, 0):   t = (T / J) * (0, -z, y),   J = integral (y^2 + z^2) dA   (tangential, linear in radius)
bending  M = (0, 0, Mz):  t_x = -(Mz / Izz) y,        Izz = integral y^2 dA          (normal, linear in y)
```

The bending case reproduces the beam-theory stress `sigma = -M y / I` with the sign convention
that a positive `Mz` on the `+x` end compresses the `+y` side.

**Caveat (Saint-Venant).** The traction is *statically equivalent* to the real section stress, not
the same. For a circular shaft in torsion it is exact; for other sections the true torsional
traction is not linear in the radius because of warping. Stresses within about one section depth
of the loaded face differ; further away they agree. A degenerate face set (zero area, or lying on a
line) cannot carry a moment and raises an error.

## 5. Supports, reactions and the equilibrium check

Constrained DOFs `D` are eliminated: with `I` the free DOFs,
`K_II u_I = f_I - K_ID u_D`, solved with a sparse direct solver. Afterwards

```text
r = K u - f        (zero on free DOFs; the support reaction on D)
```

The solver reports the total applied force and support reaction, plus their moments about the body's
volume centroid. `equilibrium_error` is the maximum of normalized force and moment residuals.
Each component's basis functions sum to 1, so `sum f` preserves the force resultant, and a rigid
translation lies in the null space of `K`. Moment balance is evaluated from the forces at their DOF
coordinates. With sufficient constraints, the residual should be near numerical round-off; a large
violation signals a load-assembly, support, or conditioning issue rather than discretisation error.


## 6. Free bodies in an accelerating body-fixed frame

A model with **no supports** is interpreted as a free elastic body. For a quasi-static stress solve
at an instant when the body has zero angular velocity, the rigid-body acceleration is computed from
the applied force and moment resultants about the center of mass `C`:

```text
m a_C = F_external
I_C alpha = M_external,C
b_inertial(x) = -rho [a_C + alpha x (x - C)]
```

The last term is the D'Alembert inertial force per volume. Its integrated force and moment cancel
the external resultants, leaving a self-equilibrated load for the elastic solve. Six global rigid modes
are then removed with a numerical gauge constraint; this does not pin physical DOFs and its Lagrange
multipliers are not reported as support reactions. The density `rho` must be positive whenever a
nonzero external resultant requires an acceleration.

This is deliberately **not transient dynamics**. It ignores elastic vibration/inertia, and assumes
zero angular velocity at the instant being analysed; centrifugal and Coriolis terms are therefore
zero at that instant. It does not model the evolution of spin, acceleration, or deformation over time.
A partially constrained body is still a static-support problem and must have enough supports to
remove all six rigid modes; insufficient supports are rejected rather than silently pinned.

## 7. Strain and stress recovery

* At every quadrature point: `eps = sym(grad u)`, `sigma = C : eps` (`Results.eps_q`, `sig_q`,
  shape `(3, 3, cells, points)`). These are the raw results.
* **Cell values**: quadrature-weighted average per cell (`eps_cell`, `sig_cell`).
* **Nodal values** (`eps_nodal`, `sig_nodal`, used for plotting and the proportional-limit
  check): each element's strain/stress is evaluated **at its own vertices** (exact inside the
  element: linear for P2 tets, bilinear-type for Q2 hexes), then the values from all elements
  sharing a vertex are volume-weighted averaged (`post.vertex_values`, `post.vertex_average`).

Measured on the 20 × 2 × 2 cantilever (L/H = 10, tip shear), mid-span surface `sigma_xx` vs
`M y / I`: P2 tets +0.8 %, Q2 hexes exact to round-off. Averaging cell means instead (the earlier
method) gave −39 % / −50 %: on a mesh two elements deep a cell mean sits a quarter of the depth
inside the surface. The averaged field is still smooth across elements and does not honour
traction-free surfaces exactly; at real stress concentrations refine the mesh and watch the peak
converge, or read `sig_q`.

## 8. Derived quantities

For a stress tensor `S` with principal values `s1 >= s2 >= s3` (eigenvalues of the symmetric `S`):

```text
von Mises  = sqrt( 1/2 [(sxx-syy)^2 + (syy-szz)^2 + (szz-sxx)^2] + 3 (sxy^2 + syz^2 + sxz^2) )
Tresca     = s1 - s3
```

Uniaxial stress `s` gives von Mises `= |s|`; pure shear `tau` gives von Mises `= sqrt(3) tau` and
principal values `(tau, 0, -tau)`. The same eigen-decomposition gives principal strains.

### Proportional-limit check

Everything above is linear elasticity, which only holds while the material stays below its
proportional limit. With `sigma_pl` set on the material (`material ..., sigma_pl = ...`, or the
app's material panel) every solve computes a utilisation `U` per node from the nodal stresses
(`post.utilisation`) and flags nodes with `U >= 1`:

```text
ductile (sigma_pl only)            U = von Mises / sigma_pl
brittle (sigma_pl = St, sigma_pl_c = Sc), Coulomb-Mohr:
                                   U = max(s1, 0) / St + max(-s3, 0) / Sc
```

The Coulomb-Mohr form is `s1/St` in pure tension, `-s3/Sc` in pure compression and
`s1/St - s3/Sc` in mixed states; pure shear (torsion) reaches the limit at
`tau = St Sc / (St + Sc)` (von Mises: `tau = sigma_pl / sqrt(3)`), so tension, compression,
torsion and any combination are covered by one number. Checking `sigma_xx` against a tension limit
instead would depend on the axes, and checking max/min principal separately (Rankine) allows
`tau = St` in torsion: unconservative for brittle materials.

Both criteria are positively homogeneous of degree 1 in the stress, and the solve is linear, so
`U` scales with the loads: the linear result is valid up to exactly `1 / max(U)` times the applied
loads (tested to round-off). Beyond it the solution is not physical (plastic flow, cracking).
Stress singularities at clamped edges, re-entrant corners and point loads grow with mesh
refinement, so a flagged node there may be a modelling artefact rather than real yielding.

Preset limits (app `MATERIAL_PRESETS`). Published *proportional* limits are scarce, so specified
minima are used as stand-ins; enter test data where it matters.

| Preset | Check | Limit(s), MPa | Basis |
|---|---|---|---|
| Steel | von Mises | 250 | ASTM A36 minimum yield Fy = 36 ksi; for structural steel the proportional limit, elastic limit and yield are treated as one point |
| Aluminum | von Mises | 240 | 6061-T6 minimum 0.2 % offset yield (typical 276); the true proportional limit of Al alloys is somewhat lower |
| Titanium | von Mises | 825 | Ti-6Al-4V annealed minimum 0.2 % offset yield (AMS 4911 / ASTM B265 ~825–828); true proportional limit lower |
| Concrete | Coulomb–Mohr | St 3, Sc 12 | f′c = 30 MPa: tension ≈ 0.1 f′c (cracking), compression ≈ 0.4 f′c (end of the near-linear range, Eurocode 2 practice) |
| CastIron | Coulomb–Mohr | St 214, Sc 752 | gray iron ASTM A48 class 30 tensile / compressive *strengths*: gray iron has no linear range in tension, so this is a fracture check, not end-of-linearity |

## 9. Reference solutions used by the benchmarks

Cantilever, length `L`, square section `H x W`, `I = W H^3 / 12`, `G = E / (2 (1 + nu))`, fixed at x = 0:

| Load at the free end | Tip deflection (Euler-Bernoulli) |
|---|---|
| shear force `P` | `P L^3 / (3 E I)`, plus a shear contribution `P L / (kappa G A)`, `kappa ~ 5/6` (~0.8 % at L/H = 10) |
| moment `M` | `M L^2 / (2 E I)` |
| self-weight, `q = rho g W H` | `q L^4 / 8 E I` |

Bending stress at section `x`: `sigma_xx = M(x) y / I` with `M(x) = P (L - x)`. Torsion of a square
bar of side `a`: twist `theta = T L / (G J)`, `J ~ 0.1406 a^4`. The finite-element results differ
from these by the fixed-end constraint (which stiffens the root), shear deformation, and mesh
coarseness: a few percent on the coarse test meshes. See [testing.md](testing.md).
