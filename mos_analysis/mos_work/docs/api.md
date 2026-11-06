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

Fields: `C: np.ndarray`, `rho: float = 0.0`

#### `Material.C4() -> np.ndarray`  _(@property)_

The 3x3x3x3 tensor form of C (sigma_ij = C4_ijkl eps_kl).

#### `Material.isotropic(E: float, nu: float, rho: float=0.0) -> 'Material'`  _(@classmethod)_

Isotropic material from Young's modulus E, Poisson ratio nu and density rho.

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

### `assemble_stiffness(basis, C4)`

Global stiffness matrix K (sparse) for elasticity tensor C4 on `basis`.

### `assemble_loads(model: Model, case: LoadCase, basis, elem) -> np.ndarray`

Global load vector for point, resultant-force, surface and volume loads.

``force`` on nodes is shared equally over those nodes; ``force`` on facets is a
total resultant distributed as uniform traction over the selected surface.
Explicit ``traction`` remains force per area.

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
eps_nodal, sig_nodal  volume-weighted averages of the cell values at vertices, (3, 3, nn);
                   smooth, for plotting (peaks are underestimated, see docs/theory.md)
```

Fields: `basis: object`, `u: np.ndarray`, `u_nodal: np.ndarray`, `reactions: np.ndarray`, `applied: np.ndarray`, `equilibrium_error: float`, `eps_q: np.ndarray`, `sig_q: np.ndarray`, `eps_cell: np.ndarray`, `sig_cell: np.ndarray`, `eps_nodal: np.ndarray`, `sig_nodal: np.ndarray`, `applied_moment: np.ndarray = field(default_factory=lambda: np.zeros(3))`, `reaction_moment: np.ndarray = field(default_factory=lambda: np.zeros(3))`, `inertial_force: np.ndarray = field(default_factory=lambda: np.zeros(3))`, `inertial_moment: np.ndarray = field(default_factory=lambda: np.zeros(3))`, `translational_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))`, `angular_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))`, `moment_equilibrium_error: float = 0.0`, `body_fixed_frame: bool = False`

#### `Results.von_mises(where='nodal')`

Von Mises stress at 'nodal' (smoothed), 'cell' (averaged) or 'q' (quadrature) points.

#### `Results.principal(where='nodal')`

Principal stresses (values, vectors) at 'nodal', 'cell' or 'q' points.

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

Volume-weighted average of cell values onto vertices (for smooth plots).
Tc: (..., nel); t: (nverts_per_cell, nel) connectivity; vol: (nel,).

### `von_mises(S)`

Von Mises equivalent stress of a tensor field S with shape (3, 3, ...).

### `principal(S)`

Principal values s1 >= s2 >= s3, shape (3, ...); vectors shape (3, 3, ...)
with vectors[:, k] the direction of s_{k+1}.

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

### `compute_scalars(res) -> dict`

name -> (nodal array (nnodes,), kind) with kind in {'disp', 'stress', 'strain'}.
`res` needs u_nodal (3, n), sig_nodal (3, 3, n), eps_nodal (3, 3, n).

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
