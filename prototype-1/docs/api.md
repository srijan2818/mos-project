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

Condition on position, e.g. 'abs(x - L) < 1e-9 and y > 0'.  Returns pred(x, y, z) -> bool array.

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

### `facets_where(mesh, pred: Callable, name: str) -> Region`

Boundary facets whose vertices ALL satisfy pred.

## `mos.core.loads`

```text
Declarative loads. They are plain data; the solver turns them into force vectors.

PointForce  node region    total force F (N), shared equally among the region's nodes
Traction    facet region   force per area (Pa), constant or Field
Pressure    facet region   p (Pa) acting INTO the body (t = -p n), constant or Field
BodyForce   whole body     force per volume (N/m^3), constant or Field
Moment      facet region   moment M (N m) about `origin` (default: facet-set centroid),
                           applied as a traction field with zero net force.
                           Torsion is just a Moment along the face normal.
```

### `cross0(a, b) -> np.ndarray`

Cross product along axis 0. a: (3,) or (3, ...); b: (3, ...).

### `moment_coefficients(X, w, M, origin=None)`

Traction t(x) = omega x (x - o) with  A omega = M,  A = sum w (|r|^2 I - r r^T).

X: (3, N) quadrature points on the face set, w: (N,) area weights.
Returns (omega, origin).  Net moment about `origin` is exactly M; net force is zero
when origin is the area centroid (the default).

### class `PointForce`  _(@dataclass)_

Total force `force` (N) on a node region, shared equally among its nodes.

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

Moment M (N m) about `origin` (default: facet-set centroid) on a facet region,
applied as a zero-net-force traction field. Torsion = a moment along the face normal.

Fields: `region: Region`, `M: object`, `origin: Optional[object] = None`

## `mos.core.model`

Model = mesh + one material + boundary conditions; LoadCases are solved against it.

### class `BC`  _(@dataclass)_

Prescribe displacement components on a node or facet region.
components: subset of (0, 1, 2) for x, y, z.  value: number, or a vector Field.

Fields: `region: Region`, `components: tuple = (0, 1, 2)`, `value: object = 0.0`

### class `LoadCase`  _(@dataclass)_

A named list of loads (PointForce, Traction, Pressure, BodyForce, Moment).

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
  force    REGION fx, fy, fz                    total point force, shared among region nodes
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

Static linear-elastic 3D solver on scikit-fem.

Written for scikit-fem >= 8. NOT yet run against the real library (it could not be installed
where this was written): the DOF bookkeeping is unit-tested with a fake basis, but the assembly
calls are first exercised by tests/test_fem.py. See docs/testing.md.

Pipeline: stiffness K -> load vector f -> constraints -> solve -> strain/stress/reactions.
Everything scikit-fem-specific lives in this file.

### `make_basis(mesh, order: int=2)`

Vector P2 (default) on tets, or Q2 on hexes.  P1/Q1 are too stiff for bending.

### `component_dofs(basis, c: int) -> np.ndarray`

All DOF indices belonging to displacement component c (0, 1, 2).

### `assemble_stiffness(basis, C4)`

Global stiffness matrix K (sparse) for elasticity tensor C4 on `basis`.

### `assemble_loads(model: Model, case: LoadCase, basis, elem) -> np.ndarray`

Global load vector f for all loads of `case` (point, surface and volume integrals).

### `constraints(model: Model, basis)`

Returns (D, x0, per_component_sets): constrained DOFs, prescribed values.

### class `Results`  _(@dataclass)_

```text
Everything produced by one solve.

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
```

Fields: `basis: object`, `u: np.ndarray`, `u_nodal: np.ndarray`, `reactions: np.ndarray`, `applied: np.ndarray`, `equilibrium_error: float`, `eps_q: np.ndarray`, `sig_q: np.ndarray`, `eps_cell: np.ndarray`, `sig_cell: np.ndarray`, `eps_nodal: np.ndarray`, `sig_nodal: np.ndarray`

#### `Results.von_mises(where='nodal')`

Von Mises stress at 'nodal' (smoothed), 'cell' (averaged) or 'q' (quadrature) points.

#### `Results.principal(where='nodal')`

Principal stresses (values, vectors) at 'nodal', 'cell' or 'q' points.

### `equilibrium_error(applied, reactions, r_constrained) -> float`

Relative global force-equilibrium error |applied + reactions| / scale.

scale is the larger of |applied| and the norm of the individual reaction values, so a model
driven only by prescribed displacements (applied = 0) is still judged sensibly.

### `solve_case(model: Model, case: LoadCase, order: int=2) -> Results`

Solve one load case: assemble, constrain, solve, then recover strain, stress, reactions.

order=2 (default) uses P2 tets / Q2 hexes; order=1 is available but far too stiff in bending.

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

## `mos.core.meshes`

Mesh helpers (imports scikit-fem lazily).

### `box_mesh(L, H, W, nx, ny, nz, kind='tet')`

Beam along x: x in [0, L], y in [-H/2, H/2], z in [-W/2, W/2].

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

### `create_arrow_glyph(origin, vector, model_size: float)`

PyVista Arrow mesh aligned with vector and scaled relative to model_size.

### `create_moment_glyph(origin, M_vec, model_size: float)`

PyVista mesh visualizing a moment about an axis at origin with curved arc and arrow.

### `build_load_glyphs(case, mesh, model_size: float)`

```text
Return list of (pyvista_mesh, color_hex) for all loads in a LoadCase.

Draws:
  - PointForce  -> red arrow at node-region centroid
  - Traction    -> bright orange arrow at facet-region centroid
  - Pressure    -> amber inward arrow at facet centroid
  - Moment      -> purple moment arrow
  - BodyForce   -> cyan arrow at model centre
```

## `mos.templates`

Example model descriptions shown in the app's 'Load example' menu.

`TEMPLATES` maps each menu name to a model description: *Cantilever - tip load*, *Cantilever - self weight*, *Bar - torsion*, *Cantilever - partial pressure*, *Simply supported - center load*, *Cantilever - pure end moment*, *Bar - axial tension*, *Fixed-fixed beam - uniform load*, *Cantilever - point force and custom axis moment*, *Overhanging beam - tip and span loads*, *CAD - Structural L-Bracket (STEP)*, *CAD - Stepped Shaft Torsion (STEP)*, *CAD - Plate with Hole (STEP)*, *CAD - Image to STEP Drawing Profile*.

## `mos.app.main`

```text
trame + PyVista front end.

    python -m mos.app [--spec model.mos] [--port 8080]

The drawer holds interactive geometric/CAD model builders, materials, supports,
loads, and view controls; the 3D view is a server-side-rendered VTK scene.
```

### `main(argv=None)`

Build the UI and start the trame server (blocks until stopped).
