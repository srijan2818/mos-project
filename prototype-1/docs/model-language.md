# The model language

A model is a text file (or the editor in the app). One directive per line; everything after `#`
is a comment; blank lines are ignored; directive names are not case sensitive. Order matters:
a `param` must come before it is used, a `region` before the supports and loads that name it, and
`material` before an expression that uses `rho`.

A model needs exactly these: one `mesh`, one `material`, and at least one `support`.

## Directives

### `param`

```text
param NAME = expression
```

A named constant usable in every later expression. `NAME` must be an identifier and must not be
one of the reserved names (`x y z pi e` and the function names below). The expression may use
earlier params but not `x y z`, and must be finite (`1/0` is an error).

### `mesh`

```text
mesh box L H W nx ny nz [tet|hex]
mesh file PATH
mesh step PATH [mesh_size]
```

`box` generates a beam along x: x in [0, L], y in [-H/2, H/2], z in [-W/2, W/2], with `nx`, `ny`,
`nz` cells in each direction (positive integers). `tet` (default) splits each cell into
tetrahedra; `hex` keeps hexahedra. Numbers may be expressions without spaces (`L/2`).

`step` imports a general CAD STEP file (`.step` or `.stp`) using Gmsh (OpenCASCADE) and generates a 3D
tetrahedral volume mesh. Each boundary CAD surface is automatically tagged as a named boundary
(`cad_face_1`, `cad_face_2`, ...), directly usable in supports, tractions, pressures, and moments.
Optional `mesh_size` controls the target characteristic element size.

`file` loads a volume mesh through scikit-fem's meshio reader (e.g. `.msh`, `.vtk`, `.vtu`,
`.xdmf`); if the file is a `.step` or `.stp` file, it is automatically routed through the CAD STEP mesher.
Surface-only meshes (an STL) are not enough: remesh the volume first. The mesh is solved with quadratic
elements (see [theory.md](theory.md)).

### `material`

```text
material E = 210e9, nu = 0.3, rho = 7850
```

Isotropic. `E` and `nu` are required, `rho` (default 0) is only needed for self-weight-type
loads. Afterwards `E`, `nu` and `rho` are also available as params in expressions.

### `region`

```text
region NAME nodes   <condition>
region NAME facets  <condition>
region NAME nearest x, y, z
```

A region is a **named label on the one mesh**. Nothing is cut or split.

* `nodes`: every node where the condition holds (interior nodes too).
* `facets`: **boundary** facets whose vertices *all* satisfy the condition. Interior facets are
  never selected.
* `nearest`: the single node closest to the point. Use it for point forces.

A condition is a comparison on `x y z`, e.g. `x < 1e-9`, `abs(x - L) < 1e-6 and y > 0`. Avoid `==`
on coordinates; use a tolerance. A region that matches nothing is an error that names the line.
Region names are unique.

### `support`

```text
support REGION [xyz] [= value]
```

Fixes displacement components on a region. With no components all of `x y z` are fixed to zero.
`support right y = -0.002` prescribes a displacement. Components are any letters of `xyz`; a
value requires them to be named.

The supports must remove all six rigid-body motions of the body, or the stiffness matrix is
singular. Symptoms of an under-constrained model: enormous displacements or a failed solve.
Fixing one whole face is the easy way to be safe.

### Loads

| Directive | Syntax | Region | Meaning |
|---|---|---|---|
| `force` | `force REGION fx, fy, fz` | nodes | **total** point force (N), shared equally among the region's nodes |
| `traction` | `traction REGION tx, ty, tz` | facets | force per area (Pa), global axes; may depend on `x y z` |
| `pressure` | `pressure REGION p` | facets | pressure (Pa) acting **into** the body; may depend on `x y z` |
| `body` | `body bx, by, bz` | whole body | force per **volume** (N/m^3); may depend on `x y z` |
| `moment` | `moment REGION mx, my, mz [about x, y, z]`<br>or `moment REGION axis ax, ay, az mag M [about x, y, z]` | facets | moment (N m), right-hand rule or specified axis & magnitude |

Vector arguments are three comma-separated expressions; commas inside brackets are fine:
`traction tip 0, where(y > 0, -1, -2), 0`.

* **Gravity** is a body force per volume, not an acceleration: `body 0, -rho*9.81, 0`.
* **A moment** is applied as a traction field on the facets with zero net force and exactly the
  requested moment about `about` (default: the centroid of the facets). You can specify the moment vector
  directly as `mx, my, mz` or by axis direction and magnitude: `moment tip axis 1, 0, 0 mag 1e5`.
  A moment along the face normal is **torsion**; one in the face plane is **bending**. It is statically equivalent to a
  real section stress, not identical to it, so expect local differences near the loaded face
  (Saint-Venant). Details in [theory.md](theory.md).
* Point forces on a solid mesh create a stress singularity at the node: stresses at that node
  grow without bound as the mesh is refined. Compare away from it, or spread the load with a
  `traction` instead.

## Expressions

Used for `param`, region conditions, and the components of tractions, pressures and body forces.
They are parsed with Python's `ast` module and run by a small whitelisted interpreter, never
`eval`.

| Kind | Allowed |
|---|---|
| numbers | `3`, `0.5`, `1e-9` |
| names | `x y z`, your params, `pi`, `e` |
| arithmetic | `+ - * / **`, and `^` as a synonym for `**`; unary `+ -`; brackets |
| functions | `sin cos tan asin acos atan atan2 sinh cosh tanh exp log sqrt abs` |
| `step(t)` | 0 where `t < 0`, 1 where `t >= 0` |
| `where(c, a, b)` | `a` where condition `c` holds, else `b` |
| conditions | `< <= > >= == !=`, chains such as `0 < x < 1`, and `and`, `or`, `not` |

Precedence follows Python (`-x**2` is `-(x**2)`). A condition is a 0/1 value in arithmetic:
`(x > 5) * 100` is 100 where `x > 5`. Anything else (attribute access, indexing, lambdas,
`if ... else`, unknown functions) is rejected when the line is read, not at solve time. A region
condition must be a true/false expression: `x + 1` is rejected.

## Worked examples

Every block below is parsed and built by the test suite.

### Pressure on a patch

```mos
param L = 10
param H = 1
param W = 1
mesh     box L H W 20 2 2
material E = 70e9, nu = 0.33
region   fixed facets x < 1e-9
region   patch facets y > H/2 - 1e-9 and x > 4 and x < 6
support  fixed xyz
pressure patch 2e6
```

### Pressure only on the outer half, with `step`

```mos
param L = 10
param p0 = 5e6
mesh     box L 1 1 30 3 3
material E = 210e9, nu = 0.3
region   fixed facets x < 1e-9
region   top   facets y > 0.5 - 1e-9
support  fixed xyz
pressure top step(x - L/2) * p0
```

### A load that varies in space, with `where`

```mos
param L = 10
mesh     box L 1 1 20 2 2
material E = 210e9, nu = 0.3
region   fixed facets x < 1e-9
region   tip   facets x > L - 1e-9
support  fixed xyz
traction tip 0, where(y > 0, -2e6, -1e6), 0
```

### A prescribed displacement (support settlement)

```mos
mesh     box 10 1 1 20 2 2
material E = 210e9, nu = 0.3
region   fixed facets x < 1e-9
region   tip   facets x > 10 - 1e-9
support  fixed xyz
support  tip y = -0.01
```

### A general body-force field (centrifugal-type)

```mos
param w = 300
mesh     box 10 1 1 20 2 2
material E = 210e9, nu = 0.3, rho = 7850
region   root facets x < 1e-9
support  root xyz
body     rho*w^2*x, rho*w^2*y, 0
```

### A point force and a moment about a chosen point

```mos
param L = 10
mesh     box L 1 1 20 2 2
material E = 210e9, nu = 0.3
region   fixed facets x < 1e-9
region   tip   facets x > L - 1e-9
region   pt    nearest L, 0, 0
support  fixed xyz
force    pt 0, -1e5, 0
moment   tip 0, 0, 2e5 about L, 0, 0
```

## Errors

Every error names the line. Common ones:

| Message contains | Cause and fix |
|---|---|
| `unknown directive 'foo'` | misspelt directive; the message lists the valid ones |
| `unknown region 'name'` | used before it is defined, or misspelt; the message lists defined regions |
| `region 'name' is empty` | the condition matched nothing; use a tolerance (`abs(x - L) < 1e-6`) and check the coordinates |
| `is not a condition` | a region expression that is not a comparison, e.g. `x + 1` |
| `unknown name 'foo'` | undefined param, or `x y z` used where only constants are allowed |
| `unsupported syntax` | something outside the whitelist (indexing, attribute access, ...) |
| `reserved` | a param named like `x`, `pi`, `step`, ... |
| `needs a node region` | `force` on a facet region; use `region ... nearest` or `nodes` |
| `needs 3 comma-separated components` | a vector with the wrong number of parts |
| `positive integer` | element counts must be whole numbers >= 1 |
| `not finite` | a constant that is `inf` or `nan` (division by zero, `log(-1)`, ...) |
| `no 'support' line` | nothing stops rigid-body motion |
| `load vector contains NaN or inf` | a load expression blew up at some point (`1/x` at `x = 0`, ...) |
