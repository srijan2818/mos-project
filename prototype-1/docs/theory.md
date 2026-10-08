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
* `force`: the total force, divided equally among the region's nodes, is added to those nodes'
  DOFs.

Solution vector `u` holds all DOFs; `Results.u_nodal` is its restriction to the mesh vertices.
(Mid-edge DOFs of P2 are not shown in the viewer.)

## 4. Moments and torque

A solid mesh has no rotational DOFs, so a moment `M` about a point `o` is applied as a traction
on a set of faces:

```text
t(x) = omega x (x - o),    r = x - o
A omega = M,               A = integral ( |r|^2 I - r r^T ) dA
```

`A` is the area-inertia tensor of the face set about `o`, and `omega` is `A^-1 M`. Then
`integral r x t dA = A omega = M`, so the **net moment about `o` is exactly `M`** (to quadrature
accuracy; the same quadrature that applies the load is used to build `A`). With `o` at the area
centroid (the default), the **net force is exactly zero**, because it equals `omega x integral r dA`.
With another origin there is a net force: the load is then "a moment `M` about `o`", not a pure
couple.

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

The solver reports, per component, the total applied force `sum f` and total reaction `sum r`, and

```text
equilibrium_error = |applied + reactions| / max(|applied|, |r_D|)
```

Why this check is meaningful: each component's basis functions sum to 1, so `sum f` equals the
total applied force exactly, and a rigid translation lies in the null space of `K`, so
`sum (K u) = 0`. A properly assembled and solved model therefore satisfies the check to round-off
(typically ~1e-10 or smaller; the tests allow 1e-8). A violation means a bug or an ill-conditioned model, not discretisation error.
Moment equilibrium is not checked.

## 6. Strain and stress recovery

* At every quadrature point: `eps = sym(grad u)`, `sigma = C : eps` (`Results.eps_q`, `sig_q`,
  shape `(3, 3, cells, points)`). These are the raw results.
* **Cell values**: quadrature-weighted average per cell (`eps_cell`, `sig_cell`).
* **Nodal values** (`eps_nodal`, `sig_nodal`, used for plotting): the volume-weighted average of
  the cell values of all cells touching a vertex.

The nodal field is smooth and cheap but it is an average, so it **smears and underestimates
peaks** at stress concentrations and does not honour traction-free surfaces exactly. For a peak
value, refine the mesh and watch it converge, or read `sig_q`.

## 7. Derived quantities

For a stress tensor `S` with principal values `s1 >= s2 >= s3` (eigenvalues of the symmetric `S`):

```text
von Mises  = sqrt( 1/2 [(sxx-syy)^2 + (syy-szz)^2 + (szz-sxx)^2] + 3 (sxy^2 + syz^2 + sxz^2) )
Tresca     = s1 - s3
```

Uniaxial stress `s` gives von Mises `= |s|`; pure shear `tau` gives von Mises `= sqrt(3) tau` and
principal values `(tau, 0, -tau)`. The same eigen-decomposition gives principal strains.

## 8. Reference solutions used by the benchmarks

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
