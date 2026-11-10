# API reference

Generated from the source by `python tools/gen_api_docs.py`; do not edit by hand
(`tests/test_docs.py` fails if this file is out of date).
Only public names are listed. Conventions used throughout: SI units, arrays with the component
axis first, e.g. positions `(3, ...)`, tensors `(3, 3, ...)`. See [theory.md](theory.md).

## `mos.core.fields`

```text
Scalar / vector fields of position (loads, body forces, prescribed displacements)
and region conditions, written as text expressions.

Convention: positions have shape (3, ...); a field returns shape (...) for a scalar
field or (3, ...) for a vector field. This is scikit-fem's layout, so a field can be
evaluated directly at quadrature points.

Expressions are parsed with `ast` and run by a small whitelisted interpreter; nothing
is ever passed to eval()/exec(). Supported:
  numbers, x y z, named parameters, pi, e, + - * / ** (or ^), unary +/-
  functions   sin cos tan asin acos atan atan2 sinh cosh tanh exp log sqrt abs
              step(t)          0 for t < 0, 1 for t >= 0
              where(c, a, b)   a where condition c holds, else b
  conditions  < <= > >= == !=   (chains like  0 < x < 1  work),  and  or  not
```

### `eval_constant(expr: str, **params: float) -> float`

Evaluate an expression that must not depend on x, y, z (e.g. '2*L').

### `compile_predicate(expr: str, **params: float) -> Callable`

Compile a position predicate to a boolean array.

Prefer tolerance bands (e.g. ``abs(x - L) < 1e-6``) over exact floating-point
coordinate equality (``x == L``), especially for imported CAD meshes.

### `split_top_level(s: str, sep: str=',') -> list`

Split on `sep` outside brackets: 'where(x>1, 2, 3), 0, -g' -> 3 parts.

### class `Field`

A scalar or vector function of position, evaluated on arrays of points.

Build with `constant`, `from_expr` (text) or `from_function` (Python callable).
Call it with positions of shape (3, ...); it returns (...) if scalar, (3, ...) if vector.

#### `Field.constant(value) -> 'Field'`  _(@classmethod)_

Field with the same value everywhere: a number (scalar) or a 3-vector.

#### `Field.from_function(fn: Callable, vector: bool=False) -> 'Field'`  _(@classmethod)_

fn(x, y, z) -> array (scalar) or a sequence of 3 arrays (vector).

#### `Field.from_expr(expr: str | Sequence[str], **params: float) -> 'Field'`  _(@classmethod)_

```text
Field.from_expr("-rho*g", rho=7850, g=9.81)                      scalar
Field.from_expr(["0", "where(x < L/2, -p, 0)", "0"], p=1e3, L=10)  vector
```

### `as_field(value, vector: bool) -> Field`

Return `value` if it is already a Field of the right rank, else wrap it as a constant.

## `mos.core.material`

Linear elastic material, defined by its 6x6 stiffness matrix C (SI units).

Voigt order: xx, yy, zz, yz, xz, xy with *engineering* shear strains (gamma = 2*eps),
so that  sigma_voigt = C @ eps_voigt.

### `voigt_to_c4(C: np.ndarray) -> np.ndarray`

Expand a 6x6 Voigt matrix into the 3x3x3x3 tensor C_ijkl (sigma_ij = C_ijkl eps_kl).

### class `Material`  _(@dataclass(frozen=True, eq=False))_

Linear elastic material: 6x6 stiffness `C` (Pa, Voigt order) and density `rho`.

Optional proportional limits (Pa) bound the range where linear elasticity holds:
``sigma_pl`` alone is checked against von Mises (ductile metals); with ``sigma_pl_c`` as
well, ``sigma_pl`` is the tension and ``sigma_pl_c`` the compression limit of the brittle
Coulomb-Mohr check (see ``post.utilisation``). None means no check.

Fields: `C: np.ndarray`, `rho: float = 0.0`, `sigma_pl: Optional[float] = None`, `sigma_pl_c: Optional[float] = None`

#### `Material.C4() -> np.ndarray`  _(@property)_

The 3x3x3x3 tensor form of C (sigma_ij = C4_ijkl eps_kl).

#### `Material.isotropic(E: float, nu: float, rho: float=0.0, sigma_pl: Optional[float]=None, sigma_pl_c: Optional[float]=None) -> 'Material'`  _(@classmethod)_

Isotropic material from Young's modulus E, Poisson ratio nu, density rho and the
optional proportional limits (see the class docstring).

## `mos.core.regions`

```text
Regions: named labels pointing at a subset of ONE mesh (the mesh itself is never split).

kind = "nodes"  -> node indices   (point forces, supports)
kind = "facets" -> boundary facet indices (tractions, pressure, moments, supports)

Selectors take any scikit-fem-like mesh: mesh.p (3, nnodes), mesh.facets (nv, nfacets),
mesh.boundary_facets().  Predicates receive coordinate arrays: pred(x, y, z) -> bool array.
```

### class `Region`  _(@dataclass)_

```text
A named subset of ONE mesh (the mesh is never split).

name  label used by supports and loads
kind  'nodes' (node indices) or 'facets' (boundary-facet indices)
ids   integer index array into the mesh's nodes or facets
```

Fields: `name: str`, `kind: str`, `ids: np.ndarray`

### `nodes_where(mesh, pred: Callable, name: str) -> Region`

Node region: all nodes where pred(x, y, z) is true.

### `nearest_node(mesh, point, name: str) -> Region`

Node region holding the single node closest to `point`.

### `outward_facet_normals(mesh, facet_ids) -> np.ndarray`

Area-weighted outward normals (3, n) of boundary triangle facets.

``mesh.facets`` stores sorted vertex indices, so the raw cross product has an
arbitrary sign per facet. Each normal is flipped to point away from the centroid of
the element that owns the facet. The vector length is twice the facet area.

### `boundary_normal(mesh, facet_ids) -> np.ndarray`

Unit area-weighted mean outward normal of a boundary facet set ((1,0,0) if degenerate).

### `cad_boundary_name_for_side(mesh, side: str) -> str | None`

Return the named CAD boundary furthest toward an axis-aligned side.

This is a compatibility mapping for UI aliases such as ``neg_x``/``-x``.
It uses each named boundary's facet centroid, so it works when a STEP
model's coordinates are offset from the origin. Returns ``None`` when the
mesh has no usable named boundaries or ``side`` is not a recognized alias.

### `facets_where(mesh, pred: Callable, name: str) -> Region`

Boundary facets whose vertices ALL satisfy pred.

## `mos.core.loads`

```text
Declarative loads. They are plain data; the solver turns them into force vectors.

PointForce  node region    total force F (N), shared equally among the region's nodes
SurfaceForce facet region  total force F (N), distributed uniformly over the face set
Traction    facet region   force per area (Pa), constant or Field
Pressure    facet region   p (Pa) acting INTO the body (t = -p n), constant or Field
BodyForce   whole body     force per volume (N/m^3), constant or Field
Moment      facet region   moment M (N m), applied as a pure couple with zero net force;
                           optional `origin` is a reference point for reporting / visualization.
                           Torsion is just a Moment along the face normal.
```

### `cross0(a, b) -> np.ndarray`

Cross product along axis 0. a: (3,) or (3, ...); b: (3, ...).

### `moment_coefficients(X, w, M, origin=None)`

Return a pure-couple traction field for the requested resultant moment.

X: (3, N) quadrature points on the face set, w: (N,) area weights.

The traction is ``t(x) = omega x (x - c)``, where ``c`` is the area centroid and
``A omega = M`` for ``A = integral(|r|^2 I - r r^T) dA``.  Centering at the area
centroid makes the resultant force zero.  A pure couple's resultant moment is
independent of the point about which it is reported, so an explicit ``origin``
must not shift the traction field or introduce a spurious net force.

### class `PointForce`  _(@dataclass)_

Total force `force` (N) on a node region, shared equally among its nodes.

Fields: `region: Region`, `force: object`

### class `SurfaceForce`  _(@dataclass)_

Total force `force` (N), distributed as uniform traction over a facet region.

Fields: `region: Region`, `force: object`

### class `Traction`  _(@dataclass)_

Force per area (Pa) on a facet region: a 3-vector or a vector Field.

Fields: `region: Region`, `field: object`

### class `Pressure`  _(@dataclass)_

Pressure p (Pa) on a facet region, acting into the body (traction = -p n).

Fields: `region: Region`, `p: object`

### class `BodyForce`  _(@dataclass)_

Force per volume (N/m^3) on the whole body: a 3-vector or a vector Field.

Fields: `field: object`

### class `Moment`  _(@dataclass)_

Pure-couple moment M (N m) on a facet region.

``origin`` is an optional reference point for reporting/visualization only: a pure couple's
resultant moment is independent of reference point and must have zero resultant force.
Torsion is a moment along the face normal.

Fields: `region: Region`, `M: object`, `origin: Optional[object] = None`

## `mos.core.model`

Model = mesh + one material + boundary conditions; LoadCases are solved against it.

### class `BC`  _(@dataclass)_

Prescribe displacement components on a node or facet region.
components: subset of (0, 1, 2) for x, y, z.  value: number, or a vector Field.

Fields: `region: Region`, `components: tuple = (0, 1, 2)`, `value: object = 0.0`

### class `LoadCase`  _(@dataclass)_

A named list of loads (PointForce, SurfaceForce, Traction, Pressure, BodyForce, Moment).

Fields: `name: str = 'case'`, `loads: List[object] = field(default_factory=list)`

### class `Model`  _(@dataclass)_

Mesh + one material + boundary conditions. Solve it with mos.core.solver.solve_case.

Fields: `mesh: object`, `material: Material`, `bcs: List[BC] = field(default_factory=list)`

## `mos.core.spec`

```text
Plain-text model description -> (Model, LoadCase).

One directive per line, '#' starts a comment. Units are whatever you use consistently
(SI by default: m, N, Pa). Expressions follow mos.core.fields.

  param    NAME = expr                          named constant, usable below
  mesh     box L H W nx ny nz [tet|hex]         beam along x, y,z centred on 0
  mesh     file PATH                            volume mesh via meshio (.msh .vtk .vtu .xdmf ...)
  material E = ..., nu = ..., rho = ...         isotropic; E, nu, rho also become params
           [, sigma_pl = ...[, sigma_pl_c = ...]]  proportional limit(s), Pa (checked after solve)

  region   NAME nodes  <condition>              nodes where the condition holds
  region   NAME facets <condition>              boundary facets whose vertices all satisfy it
  region   NAME nearest x, y, z                 the single node closest to a point

  support  REGION [xyz] [= value]               fix displacement components (default xyz = 0)
  force    REGION fx, fy, fz                    total force (nodes: shared nodal force; facets: uniform resultant)
  traction REGION tx, ty, tz                    force per area; components may depend on x, y, z
  pressure REGION p                             pushes into the body; may depend on x, y, z
  body     bx, by, bz                           force per volume; may depend on x, y, z
  moment   REGION mx, my, mz [about x, y, z]    bending / torque as a zero-net-force traction
```

### class `SpecError`

A problem in a model description; the message names the offending line.

### class `Spec`  _(@dataclass)_

```text
A parsed model description (see parse_spec). Regions are resolved later, by `build`.

params     named constants (from `param` lines plus E, nu, rho from `material`)
mesh       ("box", L, H, W, nx, ny, nz, "tet"|"hex")  or  ("file", path)
material   the Material
regions    list of (line, name, kind, arg); kind is nodes / facets / nearest
supports   list of (line, region name, component tuple, value)
loads      list of (line, kind, region name or None, payload)
```

Fields: `params: dict = field(default_factory=dict)`, `mesh: tuple | None = None`, `material: Material | None = None`, `regions: list = field(default_factory=list)`, `supports: list = field(default_factory=list)`, `loads: list = field(default_factory=list)`

### `parse_spec(text: str) -> Spec`

Parse model-description text into a Spec; raises SpecError with the line number.

### `coarsen_mesh_text(text: str, factor: float, mesh=None) -> str`

``text`` with its ``mesh`` line asking for elements ``factor`` (> 1) times larger.

``box``: element counts divided by ``factor`` (at least 1). ``step`` / ``mask`` /
``hull``: mesh size multiplied by ``factor``; without a size in the line, the current
``mesh``'s mean edge length is the starting size. A ``file`` mesh cannot be re-meshed.
The quadratic basis has ~ (1 / h)^3 DOFs, so ``factor = (n / n_target)^(1/3)``
reaches about ``n_target``; Gmsh size bounds and minimum resolutions make it approximate.

### `make_mesh(spec: Spec)`

Create the scikit-fem mesh a Spec asks for (generated box, or loaded from file).

### `build(spec: Spec, mesh=None)`

Resolve regions on the mesh and create the Model and LoadCase.

## `mos.core.solver`

Linear-elastic 3D solver on scikit-fem.

Supported bodies use static elasticity. Support-free bodies use a quasi-static accelerating
body-fixed frame at an instant of zero spin, with Newton-Euler acceleration and D'Alembert
inertia loads. The real scikit-fem path still requires integration validation in an environment
where the dependency is installed; see docs/testing.md.

Pipeline: stiffness K -> loads -> physical constraints or free-body gauge -> solve ->
strain/stress/resultants. Everything scikit-fem-specific lives in this file.

### `make_basis(mesh, order: int=2)`

Vector P2 (default) on tets, or Q2 on hexes.  P1/Q1 are too stiff for bending.

### `component_dofs(basis, c: int) -> np.ndarray`

All DOF indices belonging to displacement component c (0, 1, 2).

### `rigid_body_modes(basis, center=None) -> np.ndarray`

Return six independent rigid translations/rotations sampled at vector DOFs.

Rotational modes are scaled by the mesh extent before an orthonormal basis is
formed.  These modes are used to verify support sufficiency and, for a free body,
to choose a numerical body-fixed gauge without pinning physical displacement DOFs.

### `load_resultants(f, basis, origin=None)`

Return the force and moment resultants represented by an assembled FE vector.

The moment is evaluated about ``origin`` (global origin by default).  Summing each
component's scalar shape-function loads at its DOF coordinates exactly preserves
resultants for the P1/P2 and Q1/Q2 Lagrange bases used by this solver.

### `mass_properties(basis, density: float)`

Return volume, volume centroid, mass and inertia tensor for uniform density.

### `body_frame_acceleration(force, moment, mass, inertia)`

Newton-Euler translational/angular acceleration at an instant of zero spin.

### `inertial_body_force(X, density, center, acceleration, angular_acceleration)`

D'Alembert force density for an instantaneously non-rotating body-fixed frame.

The omitted centrifugal/Coriolis terms are zero under the explicit zero-spin,
quasi-static assumption.  Euler acceleration is represented by the distributed
force ``-rho * (a + alpha x r)``, whose force/moment resultants cancel the applied
Newton-Euler resultants.

### `solve_free_body_system(K, f, modes, equilibrium_scale=None)`

Solve ``K u = f`` with rigid modes gauged, without fixing physical DOFs.

### `count_dofs(mesh) -> int`

Displacement DOFs of the default quadratic basis, from the mesh topology alone.

P2 tetrahedra carry a node per vertex and per edge; Q2 hexahedra (27 nodes) also one per
face and per cell; three components each. Pure NumPy, so it is cheap to call before the
basis is built.

### `solve_spd(A, b, rtol: float=1e-10) -> np.ndarray`

Solve a symmetric positive-definite stiffness system ``A x = b``.

Small systems use sparse LU. Large ones use CG on the symmetrically diagonal-scaled
system, preconditioned with smoothed-aggregation AMG when ``pyamg`` is installed
(Jacobi otherwise). The true residual is checked before returning.

### `assemble_stiffness(basis, C4)`

Global stiffness matrix K (sparse) for elasticity tensor C4 on `basis`.

Isotropic materials use ``lambda tr(e_u) tr(e_v) + 2 mu e_u : e_v``, about twice as
fast as the general 81-term contraction (assembly dominates solve time for P2 tets).

### `assemble_loads(model: Model, case: LoadCase, basis, elem) -> np.ndarray`

Global load vector for point, resultant-force, surface and volume loads.

``force`` on nodes is shared equally over those nodes; ``force`` on facets is a
total resultant distributed as uniform traction over the selected surface.
Explicit ``traction`` remains force per area.

### `node_region_dofs(basis, ids) -> np.ndarray`

DOFs of a node region: its vertices plus every edge/facet whose vertices all lie in it.

For P1/Q1 this is just the vertex DOFs. For P2/Q2 the mid-edge (and Q2 mid-face) DOFs of
entities spanned by the region are included, so clamping a face given as a node set fixes
the whole face, as it does for the equivalent facet region.

### `constraints(model: Model, basis)`

Return constrained DOFs, prescribed values, and per-component DOF sets.

Overlapping supports may prescribe the same value, but contradictory values on the same
DOF are physically incompatible and are rejected instead of depending on directive order.

### class `Results`  _(@dataclass)_

```text
Everything produced by one solve.

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
eps_nodal, sig_nodal  each element's strain/stress evaluated at its own vertices, then
                   volume-weighted averaged per vertex, (3, 3, nn) (see docs/theory.md)
```

Fields: `basis: object`, `u: np.ndarray`, `u_nodal: np.ndarray`, `reactions: np.ndarray`, `applied: np.ndarray`, `equilibrium_error: float`, `eps_q: np.ndarray`, `sig_q: np.ndarray`, `eps_cell: np.ndarray`, `sig_cell: np.ndarray`, `eps_nodal: np.ndarray`, `sig_nodal: np.ndarray`, `applied_moment: np.ndarray = field(default_factory=lambda: np.zeros(3))`, `reaction_moment: np.ndarray = field(default_factory=lambda: np.zeros(3))`, `inertial_force: np.ndarray = field(default_factory=lambda: np.zeros(3))`, `inertial_moment: np.ndarray = field(default_factory=lambda: np.zeros(3))`, `translational_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))`, `angular_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))`, `moment_equilibrium_error: float = 0.0`, `body_fixed_frame: bool = False`

#### `Results.von_mises(where='nodal')`

Von Mises stress at 'nodal' (smoothed), 'cell' (averaged) or 'q' (quadrature) points.

#### `Results.principal(where='nodal')`

Principal stresses (values, vectors) at 'nodal', 'cell' or 'q' points.

#### `Results.utilisation(material, where='nodal')`

Fraction of ``material``'s proportional limit used (see post.utilisation), or None
when the material has no limit.

#### `Results.stress_at_point(xyz, points, where='nodal')`

Stress tensor (3,3) at spatial point xyz (nearest node in `where` field).

#### `Results.traction_on_plane(xyz, normal, points, where='nodal')`

Traction on a plane at spatial point xyz. See post.traction_on_plane.

### `equilibrium_error(applied, reactions, r_constrained) -> float`

Relative global force-equilibrium error |applied + reactions| / scale.

scale is the larger of |applied| and the norm of the individual reaction values, so a model
driven only by prescribed displacements (applied = 0) is still judged sensibly.

### `solve_case(model: Model, case: LoadCase, order: int=2) -> Results`

Solve a load case and recover small-strain stress and strain.

Supported bodies use ordinary static elasticity and physical support reactions.
A body with no prescribed-displacement supports is treated as a free body in a
body-fixed frame at an instant of zero spin: Newton-Euler acceleration is computed
from the external resultant force/moment, the matching D'Alembert body-force density
is assembled, and six global gauge constraints remove only rigid translation/rotation.
This is a quasi-static acceleration-corrected solve, not transient dynamics.  A model
with some but insufficient supports is rejected instead of being silently pinned.

order=2 (default) uses P2 tets / Q2 hexes; order=1 is available but far too stiff in bending.

### class `CachedLinearSolver`

Reusable supported-body stiffness system for interactive load exploration.

The mesh, material and supports define the stiffness operator.  They are assembled
and factorised once.  ``prepare_live_load`` then prepares unit-load displacement
fields for one selected force, moment or uniform-pressure entry.  Changing that
entry's components or magnitude is evaluated by linear superposition; no FE solve
or stiffness factorisation is performed while the user moves the controls.

This cache deliberately supports physically supported, linear-static models only.
Free-body inertial-frame cases, geometry/material edits and support edits must use
the ordinary ``solve_case`` path and establish a fresh cache afterwards.

#### `CachedLinearSolver.compatible(model: Model) -> bool`

Whether a new case still uses the cached mesh/material/support operator.

#### `CachedLinearSolver.solve(case: LoadCase) -> Results`

Solve a changed load case with cached K and a reusable sparse LU factorisation.

#### `CachedLinearSolver.prepare_live_load(case: LoadCase, index: int)`

Prepare a no-solve, superposed response for one selected interactive load.

### class `LiveLoadResponse`

Unit-load response basis for one editable linear-static load.

The active load is removed from the baseline case.  Three unit displacement fields
are prepared for force/moment components, or one for uniform pressure.  Thereafter
every magnitude/direction update is just ``u_base + Σ coefficient * u_basis`` plus
inexpensive strain/stress recovery.  No matrix assembly, factorisation or backsolve
occurs during slider movement.

#### `LiveLoadResponse.matches(load) -> bool`

Return whether ``load`` uses the same type and target as this basis.

#### `LiveLoadResponse.coefficients(load) -> np.ndarray`

Return the physical component coefficients for a load matching this basis.

#### `LiveLoadResponse.evaluate(load) -> Results`

Evaluate updated load coefficients without another finite-element solve.

## `mos.core.post`

Post-processing on plain arrays. Tensors have shape (3, 3, ...).

### `strain_stress_q(basis, u, C4)`

Strain and stress tensors at quadrature points: (3, 3, nel, nqp).

### `cell_average(T, dx)`

Quadrature-weighted average per cell: (3,3,nel,nqp) -> (3,3,nel).

### `nodal_average(Tc, t, vol, nnodes)`

Volume-weighted average of cell values onto vertices.
Tc: (..., nel); t: (nverts_per_cell, nel) connectivity; vol: (nel,).

### `vertex_values(basis, u, C4)`

Strain and stress evaluated inside each element at its own vertices: (3, 3, nel, nverts).

The quadrature points of a copy of ``basis`` are put at the reference vertices
(``mesh.refdom.p``, same order as the rows of ``mesh.t``). For P2/Q2 the stress is linear /
bilinear inside an element, so this is exact per element; the jump between elements is the
discretisation error.

### `vertex_average(Tv, t, vol, nnodes)`

Volume-weighted average of element-vertex values onto mesh vertices.

Tv: (..., nel, nverts) from :func:`vertex_values`; t: (nverts, nel); vol: (nel,).
Unlike :func:`nodal_average` of cell means this keeps the linear variation inside each
element: on the default 20 x 2 x 2 cantilever the mid-span surface stress is within ~1 %
of beam theory instead of -41 %.

### `von_mises(S)`

Von Mises equivalent stress of a tensor field S with shape (3, 3, ...).

### `principal(S)`

Principal values s1 >= s2 >= s3, shape (3, ...); vectors shape (3, 3, ...)
with vectors[:, k] the direction of s_{k+1}.

### `utilisation(S, sigma_pl, sigma_pl_c=None)`

```text
Fraction of the proportional limit used, per point, for stresses S (3, 3, ...).

* ``sigma_pl_c`` is None (ductile): von Mises, ``U = sigma_vM / sigma_pl``.
* ``sigma_pl_c`` given (brittle Coulomb-Mohr, tension limit St = ``sigma_pl``, compression
  limit Sc = ``sigma_pl_c``): ``U = max(s1, 0) / St + max(-s3, 0) / Sc``. This is s1/St in
  pure tension, -s3/Sc in pure compression and s1/St - s3/Sc in mixed states (pure shear
  fails at tau = St Sc / (St + Sc)).

Both are positively homogeneous of degree 1 in S, so in a linear solve all loads may be
scaled by ``1 / max(U)`` before any point reaches U = 1.
```

### `tresca(S)`

Tresca equivalent stress (max shear * 2 = s1 - s3) of S with shape (3, 3, ...).

### `stress_at_point(sig_nodal, points, xyz)`

Return the (3,3) stress tensor at spatial point xyz by nearest-node lookup.

sig_nodal: (3,3,nnodes); points: (nnodes,3) or (3,nnodes).
Returns (tensor (3,3), node_index).

### `traction_on_plane(sig, xyz_points, xyz, normal)`

Stress tensor + traction on a plane at spatial point.

Returns dict: tensor(3,3), node_index, normal(3,), traction(3,),
normal_stress(float), shear_stress(float), shear_vector(3,).

## `mos.core.meshes`

Mesh helpers (imports scikit-fem lazily).

### `box_mesh(L, H, W, nx, ny, nz, kind='tet')`

Beam along x: x in [0, L], y in [-H/2, H/2], z in [-W/2, W/2].

### `mean_edge_length(mesh) -> float`

Mean length of the cells' vertex-to-vertex edges (tets: 6 per cell, hexes: the
12 box edges), the mesh's characteristic element size.

## `mos.core.gmsh_session`

One process-wide Gmsh session shared by the CAD importer, generators and image meshers.

Gmsh keeps global state and is not thread-safe, so every caller that builds or meshes a
model must hold :data:`GMSH_LOCK`. ``gmsh.initialize()`` by default installs a SIGINT
handler, which raises ``ValueError`` outside the main thread; :func:`ensure_gmsh`
initialises without it so background workers in the app can use Gmsh too.

### `ensure_gmsh() -> None`

Initialise Gmsh once (thread-safe, no signal handler) and silence its terminal output.

## `mos.core.image_to_mesh`

```text
Direct image -> FE mesh: foreground mask -> outline + holes -> layered tetrahedral extrusion.

This is the robust single-photo geometry path. It meshes exactly the silhouette that was
segmented (including through-holes) instead of classifying the part into one of a few
templates and guessing template dimensions, and it never writes or re-imports a STEP file.

    mask (dark/alpha = material) --contours--> outer polygon + hole polygons (pixels)
        --scale--> metres --Gmsh built-in kernel, in memory--> 2D triangles
        --extrude in structured layers--> conforming tetrahedra --> scikit-fem MeshTet

Named boundaries (usable directly as support/load regions):
    cad_face_1   bottom (z = 0)          cad_face_2   top (z = thickness)
    cad_face_3.. outer side walls, one per roughly straight run of the outline
    then         one face per hole wall

A single view cannot reveal depth, so the part is assumed prismatic (constant thickness
normal to the photo). That is exact for plates, brackets photographed side-on and
cross-sections photographed end-on (I-beams, tubes, channels) once a length is entered.
```

### `load_mask_image(path: Union[str, Path]) -> np.ndarray`

Read a mask image as a bool array: True = material.

Transparent pixels are background when the image has an alpha channel; otherwise
dark pixels (< 128) are material, matching the convention of the drawing importer
and of the masks written by single-image reconstruction.

### `mask_contours(mask: np.ndarray, simplify_px: float=1.0, min_hole_px: int=12) -> Tuple[np.ndarray, List[np.ndarray]]`

Outer outline and hole outlines of a single-part mask, in pixel units with y up.

Loops follow pixel edges (not centres), so a w x h pixel block has area w * h.

### `mesh_from_mask(mask: np.ndarray, size_m: float, thickness_m: float, mesh_size: Optional[float]=None, max_elements: int=12000) -> Tuple[Any, Dict[str, Any]]`

Mesh the largest part in ``mask`` as a prism of the given thickness.

``size_m`` is the real length of the part's largest in-plane extent (the scale
reference); ``thickness_m`` is the extrusion depth normal to the image. Returns the
scikit-fem ``MeshTet`` and an info dict (face names/roles, dimensions, counts).
``mesh_size`` (default ``size_m / 20``) is coarsened when the estimated tetrahedron
count would exceed ``max_elements``; ``info["mesh_size"]`` reports the size used.

### `load_mask_mesh(path: Union[str, Path], size_m: float, thickness_m: float, mesh_size: Optional[float]=None) -> Tuple[Any, Dict[str, Any]]`

``mesh_from_mask`` for a mask image file, cached on (path, mtime, parameters).

The app re-builds the model on every solve; the cache makes that free for an
unchanged mask and settings.

### `save_mask_image(mask: np.ndarray, path: Union[str, Path]) -> Path`

Write a mask in the convention :func:`load_mask_image` reads (black = material).

## `mos.core.multiview`

Multi-view capture, NeRF/Gaussian-splatting dataset preparation, and PLY tools.

This module deliberately keeps neural reconstruction dependencies optional.  It
can package and assess image captures with the base project dependencies, then
optionally delegate camera-pose recovery to Nerfstudio/COLMAP when installed.
It also extracts the Gaussian means from common 3D Gaussian Splatting PLY files
as a conventional colored point cloud; those means are not a watertight CAD
surface by themselves.

### class `CaptureSession`  _(@dataclass)_

Description of a saved multi-view image capture.

Fields: `session_dir: Path`, `images_dir: Path`, `manifest_path: Path`, `image_count: int`, `summary: str`, `warnings: List[str]`

### class `GaussianPlySummary`  _(@dataclass)_

Statistics for a Gaussian PLY and its exported center point cloud.

Fields: `source_path: Path`, `centers_path: Path`, `gaussian_count: int`, `bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]`, `centroid: Tuple[float, float, float]`, `mean_opacity: Optional[float]`, `mean_scale: Optional[float]`, `has_gaussian_attributes: bool`

### `prepare_capture_session(images: Sequence[Any], output_root: Union[str, Path], session_name: str='mechanical_part', max_images: int=300) -> CaptureSession`

Validate and save uploaded views into a Nerfstudio-ready raw image folder.

Input items may be image paths, bytes, PIL images, data URIs, or mappings
containing ``name`` and ``data``. Exact duplicate pixel images are skipped.
At least three unique views are required for a meaningful multi-view job.

### `nerfstudio_commands(images_dir: Union[str, Path], output_dir: Union[str, Path], method: str='splatfacto') -> str`

Build copyable processing/training commands for supported recon methods.

### `tool_availability() -> Dict[str, bool]`

Report whether optional reconstruction CLIs are available on PATH.

### `run_nerfstudio_preprocess(images_dir: Union[str, Path], output_dir: Union[str, Path], timeout: int=1800) -> Dict[str, Any]`

Run Nerfstudio's image parser, which estimates camera poses using COLMAP.

### `gaussian_ply_to_point_cloud(gaussian_ply_path: Union[str, Path], output_path: Optional[Union[str, Path]]=None) -> GaussianPlySummary`

Extract Gaussian means from a 3DGS PLY into a conventional RGB point cloud.

Common 3DGS fields ``f_dc_0..2`` are converted from SH DC color to RGB.
Standard ``red/green/blue`` fields are preserved. The export intentionally
discards covariance/orientation and view-dependent SH terms: it is a point
cloud inspection bridge, not a surface or parametric CAD reconstruction.

### `calibrate_point_cloud_scale(point_cloud_path: Union[str, Path], known_extent_m: float, axis: str='x', output_path: Optional[Union[str, Path]]=None) -> Dict[str, Any]`

Apply a single metric scale from one known bounding-box dimension.

Coordinates are centered at the point-cloud centroid and uniformly scaled,
preserving the recovered shape proportions. This only fixes global scale;
it does not repair incomplete geometry or turn points into a CAD solid.

### `run_colmap_sparse_reconstruction(images_dir: Union[str, Path], workspace_dir: Optional[Union[str, Path]]=None, timeout_per_step: int=1800) -> Dict[str, Any]`

Create a standard COLMAP sparse model and PLY point cloud.

The workspace layout is compatible with the usual Graphdeco/2DGS COLMAP
loaders: ``images/`` and ``sparse/0/{cameras,images,points3D}.bin``. Feature
extraction and matching use the COLMAP defaults so it can select available
acceleration; the operation is intended to run in a background UI thread.

## `mos.core.image_reconstruction`

Single-image 2.5D reconstruction for the MoS CAD preview workflow.

A single photograph cannot reveal hidden/back-side geometry.  This module therefore
creates a colored, closed relief mesh from a foreground mask and (optionally) a
monocular *relative* depth estimate.  It can also export a separate silhouette-
extrusion STEP approximation when Gmsh is available.  The result is suitable for
visual inspection and rough shape studies, not dimensional inspection or direct
engineering validation without manual review.

The depth model and background-removal model are optional and loaded lazily.

### class `ImageReconstructionResult`  _(@dataclass)_

Generated files and diagnostics for one image reconstruction.

Fields: `source_name: str`, `preview_mesh_path: Path`, `mask_path: Path`, `depth_preview_path: Optional[Path]`, `step_path: Optional[Path]`, `width_m: float`, `height_m: float`, `thickness_m: float`, `vertex_count: int`, `face_count: int`, `segmentation_method: str`, `depth_method: str`, `warnings: list[str]`

### `reconstruct_single_image(image_source: Any, output_dir: Union[str, Path], *, object_width_m: float=0.2, body_thickness_m: float=0.025, relief_depth_m: float=0.045, max_resolution: int=192, geometry_mode: str='depth_relief', segmentation_mode: str='auto', export_step: bool=True) -> ImageReconstructionResult`

Create an inspectable single-image 2.5D mesh and optional STEP silhouette.

``geometry_mode`` may be ``depth_relief`` (uses Depth Anything V2 relative
depth) or ``silhouette`` (flat extruded relief).  STEP export is always a
silhouette extrusion and is intentionally separate from the relief preview.
``object_width_m`` only sets approximate scene scale; monocular depth is not
metric depth.

## `mos.core.multiview_hull`

```text
Solid from three orthogonal photos (top, front, right side) -> tetrahedral FE mesh.

Each photo contributes the part's silhouette. Treating each photo as an orthographic
projection along its axis, a point belongs to the part only if it lies inside the silhouette
in all three views: the solid is the intersection of the three extruded silhouettes (the
**visual hull** of three orthogonal views). No printed markers are needed: the views are
registered by their bounding boxes, and one dimension typed by the user sets the scale.

Frame (z up): x = length (left -> right in the front photo), y = depth (front -> back),
z = height. Image directions of the three photos:

    front   camera in front, level      columns -> +x, rows (down) -> -z
    top     camera above, front edge at the bottom of the photo
                                        columns -> +x, rows (down) -> -y
    side    camera on the right, level  columns -> +y, rows (down) -> -z

Each axis is seen by two photos, which gives each photo's scale up to one common factor
(:func:`fit_scales`); the third pairing is a consistency check.

What it can and cannot represent, and its errors:

* exact (to the voxel size) for parts whose shape is an intersection of three extrusions
  (blocks, brackets, plates with through-holes seen along an axis); concavities that no view
  shows (blind pockets, recesses, anything only visible obliquely) are filled in, so volume and
  stiffness are over-estimated there;
* perspective: a phone photo is not orthographic. A magnification common to a whole photo
  cancels (each photo's bounding box is rescaled, the typed dimension fixes the scale), so a
  box comes out exact from any distance. Parts of one photo at different depths are
  magnified differently, by a relative ~ (depth difference) / d at camera distance d.
  Measured on an 80 x 30 x 50 mm L-bracket: d = 5x its size: extents <= 6 %, volume +7 %,
  the 10 mm upright 12.5 mm thick; d = 10x: <= 3 %, +4 %. Shoot from far, zoomed in;
* tilt: a view turned by theta from its axis adds about D sin(theta) of the depth to the width;
* checking: the only reliable consistency check is the bounding-box proportions
  (``size_mismatch``): an extra object or a hand in one photo changes its proportions.
  Silhouette agreement (projecting the hull back into each photo) is *not* used: with three
  orthogonal views the hull reproduces a view almost whenever the other two are solid
  outlines, so a mirrored photo or a hand inside the outline usually still "agrees";
* discretisation: the hull is carved on a voxel grid (``resolution`` voxels along the longest
  side) and surfaced by marching cubes on a slightly smoothed occupancy: O(voxel) boundary error.

The FE mesh uses the app's convention, y up: ``(X, Y, Z) = (x, z, -y)``, table at Y = 0.
```

### class `OrthoView`  _(@dataclass)_

One photo: which view it is, the part's silhouette (H, W) and how to orient it.

``rot90`` quarter turns counter-clockwise, then ``flip_h`` mirrors left-right; both fix a
photo taken in portrait or from the wrong side without retaking it.

Fields: `name: str`, `mask: np.ndarray`, `axis: str`, `flip_h: bool = False`, `rot90: int = 0`

#### `OrthoView.oriented() -> np.ndarray`

The mask after rotation and mirroring.

### class `Hull`  _(@dataclass)_

Carved occupancy grid (voxel centres at origin + (i, j, k) * voxel), hull frame.

Fields: `occupied: np.ndarray`, `origin: np.ndarray`, `voxel: float`, `size_mismatch: Optional[float] = None`

#### `Hull.volume() -> float`  _(@property)_

Carved volume (m^3), voxel-counted.

#### `Hull.extent() -> np.ndarray`

Bounding-box size (m) of the occupied voxels, centre to centre plus one voxel.

### `crop_part(mask: np.ndarray) -> np.ndarray`

Largest connected component of ``mask``, cropped to its bounding box.

### `fit_scales(views: List[OrthoView], known_axis: str, known_m: float)`

Part size (m, along x, y, z) and each photo's scale (m/px) from the bounding boxes.

Each world axis spans ``n_a`` pixels in the two photos that see it, so their scales k
satisfy ``k_i n_i = k_j n_j``: three equations for three unknowns, one of which is the common
factor fixed by ``known_m`` (the true size along ``known_axis``). In logs the equations
form a cycle; least squares spreads its closure residual r equally over the three, and
``|e^r - 1|`` (returned as ``mismatch``) is how much the three photos disagree about the
proportions: 0 for perfect orthographic silhouettes; perspective, tilt or a wrong
selection make it grow.

### `carve_ortho(views: List[OrthoView], known_axis: str='x', known_m: float=0.1, resolution: int=128) -> Hull`

Intersection of the three extruded silhouettes; ``resolution`` voxels on the longest side.

Each silhouette's bounding box is stretched onto the common size from :func:`fit_scales`
(that registers the photos); ``size_mismatch`` reports how well the boxes agreed.

### `surface(hull: Hull, smooth_voxels: float=1.0, step: int=1) -> Tuple[np.ndarray, np.ndarray]`

Closed triangle surface (vertices in the hull frame, faces) by marching cubes.

``step`` > 1 samples every step-th voxel: fewer, larger triangles. Gmsh could not fill such
surfaces at steps 3-4; for coarser meshes carve at a lower resolution instead.

### `name_faces_by_normal(mesh, max_dev_deg: float=35.0, min_area_frac: float=0.01) -> Dict[str, np.ndarray]`

Group a mesh's boundary facets into faces for supports and loads.

Region growing over edge-adjacent boundary triangles: a triangle joins a face while its
outward normal stays within ``max_dev_deg`` of the face's area-weighted mean normal (seeds
in order of area). Unlike a bend-angle split, a rounded edge (marching cubes on a smoothed
grid) is cut along its middle instead of merging its two faces. Faces under
``min_area_frac`` of the surface join the adjacent face with the closest normal.
Returns ``{"cad_face_N": facet ids}``, N = 1 for the largest face.

### `mesh_surface(verts: np.ndarray, faces: np.ndarray)`

Tetrahedral MeshTet (app frame, y up, table at Y = 0) inside a closed triangle surface.

The marching-cubes triangles are kept as the boundary (a watertight, non-self-intersecting
surface) and Gmsh fills the volume. Re-meshing the surface through Gmsh's
classify/reparametrise step was tried first and failed on a real session ("a segment and a
facet intersect"); element size is set by the carving resolution instead (see
:func:`load_hull_mesh`). Faces are named by :func:`name_faces_by_normal`.

### `save_session(directory: Union[str, Path], views: List[OrthoView], known_axis: str, known_m: float) -> Path`

Write masks (PNG) and ``ortho.json`` so ``mesh hull DIR`` can rebuild the solid.

### `load_session(directory: Union[str, Path])`

(views, known_axis, known_m) saved by :func:`save_session`.

### `load_hull_mesh(directory: Union[str, Path], mesh_size: Optional[float]=None, resolution: int=128)`

Session directory -> carved hull -> tetrahedral mesh (the ``mesh hull`` directive).

## `mos.lab.__init__`

Composable Learning Lab feature registry.

Feature modules consume solved Results/Model data and never invoke the solver.
Add future analysis modules by registering a small adapter here.

### `run_feature(name, *args, **kwargs)`

Dispatch a registered post-processing feature by its stable key.

## `mos.lab.stress`

Point-driven stress analysis adapters for solved FEA results.

### `evaluate_stress(results, points, xyz, where='nodal')`

Sample a stress tensor and invariants at the nearest mesh node.

Results remain in SI (Pa); the returned tensor and scalar stresses are Pa.

## `mos.lab.traction`

Arbitrary-plane traction adapters for solved FEA results.

### `evaluate_traction(results, points, xyz, normal, where='nodal')`

Evaluate traction decomposition at a nearest mesh node, in SI units.

## `mos.lab.summary`

Read-only model and result summaries for the Learning Lab.

### `model_summary(model, case, grid)`

Return compact HTML readouts derived from the current solved model.

### `tensor_matrix_html(tensor, scale=1.0, digits=4)`

Render a compact escaped 3x3 matrix.

## `mos.viz`

Results -> things to draw. The array helpers are pure numpy; to_pyvista needs pyvista + meshio.

### `compute_scalars(res, material=None) -> dict`

name -> (nodal array (nnodes,), kind) with kind in {'disp', 'stress', 'strain', 'ratio'}.
`res` needs u_nodal (3, n), sig_nodal (3, 3, n), eps_nodal (3, 3, n). With a ``material``
that has a proportional limit, the dimensionless :data:`UTILISATION` field is added.

### `suggest_scale(points, u_nodal) -> float`

Deformation factor that makes the largest displacement ~10 % of the model size.

### `clip_origin(bounds, axis: int, frac: float)`

Point on the clip plane at fraction `frac` of the bounding box along `axis`.

### `to_pyvista(mesh)`

scikit-fem mesh -> pyvista UnstructuredGrid (same node order as mesh.p).

### `face_definitions(bounds, mesh=None) -> dict`

Dictionary of standard bounding faces: key -> {name, normal, centroid, center, direction, i_size, j_size}.

### `create_face_mesh(face_key: str, bounds, mesh=None)`

PyVista geometry representing the specified bounding face.

### `scalar_bar_args(title: str, is_dark: bool=True) -> dict`

Return scalar_bar_args dict themed for dark or light mode.

### `glyph_magnitude_scale(magnitude: float, reference_magnitude: float=1.0) -> float`

Map a physical load magnitude to a bounded square-root glyph scale.

### `create_arrow_glyph(origin, vector, model_size: float, reference_magnitude=None)`

Create an arrow aligned with ``vector``.

When ``reference_magnitude`` is supplied, arrow length changes with the square
root of the physical magnitude. This preserves a useful visible range while making
live load scaling apparent in the viewport. The reference uses the vector's units.

### `create_moment_glyph(origin, M_vec, model_size: float, reference_magnitude=None)`

Visualize a moment with a curved arc whose radius can reflect its magnitude.

### `build_load_glyphs(case, mesh, model_size: float)`

Return (glyph, color) pairs for loads in a case.

Arrow direction indicates the load direction; arrow length responds to magnitude,
so changing a load in the live inspector also produces immediate viewport feedback.
Reference magnitudes are type-specific because N, N·m and Pa are not comparable.

## `mos.templates`

Example model descriptions shown in the app's 'Load example' menu.

`TEMPLATES` maps each menu name to a model description: *Cantilever - tip load*, *Cantilever - self weight*, *Bar - torsion*, *Cantilever - partial pressure*, *Simply supported - center load*, *Cantilever - pure end moment*, *Bar - axial tension*, *Fixed-fixed beam - uniform load*, *Cantilever - point force and custom axis moment*, *Overhanging beam - tip and span loads*, *CAD - Structural L-Bracket (STEP)*, *CAD - Stepped Shaft Torsion (STEP)*, *CAD - Plate with Hole (STEP)*, *CAD - Image to STEP Drawing Profile*.

## `mos.app.main`

```text
trame + PyVista front end.

    python -m mos.app --server [--spec model.mos] [--port 8080]

The drawer holds interactive geometric/CAD model builders, materials, supports,
loads, and view controls; the 3D view is a server-side-rendered VTK scene.
```

### `main(argv=None)`

Build the UI and start the trame server (blocks until stopped).
