# Architecture

## Modules

| Module | Responsibility | Needs |
|---|---|---|
| `mos/core/fields.py` | text expressions to fields and conditions (safe interpreter); `Field` | numpy |
| `mos/core/material.py` | `Material`: 6x6 `C`, `C4`, isotropic constructor | numpy |
| `mos/core/regions.py` | `Region` and selectors (`nodes_where`, `facets_where`, `nearest_node`) | numpy |
| `mos/core/loads.py` | declarative loads; `moment_coefficients`; `cross0` | numpy |
| `mos/core/model.py` | `BC`, `LoadCase`, `Model` containers | none |
| `mos/core/spec.py` | text model description: `parse_spec`, `build`, `make_mesh` | numpy (+ skfem to make meshes) |
| `mos/core/solver.py` | scikit-fem assembly, constraints, solve, `Results`, `solve_case` | **scikit-fem** |
| `mos/core/post.py` | strain/stress recovery (element-vertex values averaged per vertex), von Mises, principal values, proportional-limit utilisation | numpy (+ skfem for `vertex_values`) |
| `mos/core/multiview.py` | multi-view capture validation, Nerfstudio/COLMAP reconstruction datasets, Gaussian PLY center extraction and metric scaling | numpy, scipy, Pillow; optional Nerfstudio/COLMAP |
| `mos/core/meshes.py` | `box_mesh` | scikit-fem |
| `mos/core/gmsh_session.py` | one process-wide Gmsh lock and thread-safe initialisation (no SIGINT handler) | gmsh |
| `mos/core/cad.py` | STEP import; `gmsh_to_meshtet` (vectorised Gmsh -> `MeshTet` with named faces) | gmsh, scikit-fem |
| `mos/core/image_to_mesh.py` | mask -> outline + holes -> layered tetrahedral extrusion (`mesh mask`) | gmsh, scikit-fem, Pillow |
| `mos/core/multiview_hull.py` | three orthogonal photo outlines -> scale fit -> voxel hull -> marching cubes -> tetrahedral mesh (`mesh hull`) | numpy, scipy, scikit-image, gmsh, scikit-fem |
| `mos/core/cv_detector.py` | photo -> part mask, category and template parameters | numpy, scipy, Pillow; optional torch/transformers |
| `mos/viz.py` | result fields for plotting, deformation scale, clip plane, pyvista conversion | numpy (+ pyvista, meshio) |
| `mos/templates.py` | example models for the app's menu | none |
| `mos/learning.py` | idealised tensor transformations, principal directions, Mohr SVGs and quick-check formulae for the Learning Lab | numpy |
| `mos/app/` | trame application and the Analysis Workspace / Learning Lab screens | pyvista, trame |

Only the FEM mesh/solver and visualization modules require scikit-fem or PyVista. The multi-view
module uses the base NumPy/SciPy/Pillow stack for capture validation and PLY conversion; camera-pose
recovery and neural training are optional external Nerfstudio/COLMAP tools.

## Data flow

```text
text ─parse_spec─▶ Spec ─build(spec, mesh)─▶ (Model, LoadCase) ─solve_case─▶ Results
                                                                     │
                                          viz.compute_scalars ◀──────┘──▶ viz.to_pyvista ─▶ app scene
```

`parse_spec` validates everything that does not need a mesh: syntax, parameter values, expression
names, region conditions (compiled), numbers of components. `build` resolves regions on a mesh
and constructs the load objects, so a bad condition and an empty region are both reported with
the line they came from.

## Design decisions

* **Regions are named labels, never mesh splits.** Supports and loads refer to regions by name,
  so solver, file format and GUI never share node indices. A region is a node set or a boundary
  facet set. A point load is just a one-node region.
* **Loads are declarative data.** `PointForce`, `SurfaceForce`, `Traction`, `Pressure`, `BodyForce` and `Moment` are plain dataclasses. `force` means a total resultant in newtons: it is shared across a node region or converted to uniform traction across a facet region. `traction` and `pressure` are force per area. Every load contributes to the same assembled vector, so overlapping regions obey superposition.
* **Free-body convention is explicit.** With no supports, `solve_case` computes translational and angular acceleration from the external resultants and adds the corresponding D'Alembert force density for a quasi-static solution at an instant of zero spin. Six global rigid modes are removed by a gauge constraint, not artificial supports. This is not transient dynamics; see `theory.md`.
* **One text format, many front ends.** The GUI edits and runs text; the same text works headless
  (`--spec`) and in tests. A form-based editor can later generate it.
* **Safe expressions.** Field and condition text is parsed with `ast` and interpreted from a
  whitelist of node types and functions. SymPy's parser was rejected because it calls `eval`.
* **Quadratic elements by default** (P2/Q2), because linear elements are too stiff in bending.
* **Linear solver by size.** Supported (SPD) systems up to `DIRECT_SOLVE_MAX_DOFS` (40k) use sparse
  LU; larger ones use diagonally scaled CG (AMG-preconditioned if `pyamg` is installed) with a
  residual check. SuperLU fill on 3D P2 elasticity grows super-linearly (~3 GB at 60k DOFs) and
  exhausted a 16 GB machine at ~190k. Isotropic materials assemble with the Lame form, about twice
  as fast as the general C4 contraction. Image meshes carry an element budget (12k tets default).
* **Solver isolation.** Everything scikit-fem specific is in `solver.py`; its DOF bookkeeping
  (`component_dofs`, `constraints`, point-force assembly, `equilibrium_error`) is written to be
  testable with a fake basis.
* **Fields use scikit-fem's array layout**, so a field is evaluated directly on the quadrature
  points of a basis: `(3, nel, nqp)` in, `(3, nel, nqp)` out.

## Using the library without the text format

```python
import numpy as np
from mos.core import BC, LoadCase, Material, Model, Traction, facets_where
from mos.core.meshes import box_mesh
from mos.core.solver import solve_case

mesh = box_mesh(10, 1, 1, 20, 2, 2)
fixed = facets_where(mesh, lambda x, y, z: x < 1e-9, "fixed")
tip = facets_where(mesh, lambda x, y, z: x > 10 - 1e-9, "tip")
model = Model(mesh, Material.isotropic(210e9, 0.3), [BC(fixed)])
res = solve_case(model, LoadCase("tip", [Traction(tip, [0, -1e6, 0])]))
print(res.u_nodal[1].min(), res.equilibrium_error)
```

## Extending

* **A new load type**: add a dataclass in `loads.py` (validate the region kind in
  `__post_init__`), add a branch in `solver.assemble_loads` that produces nodal values, add a
  directive handler in `spec.py` (`_h_...` plus an entry in `_HANDLERS`), then document it in
  `docs/model-language.md`. A test fails if a directive is undocumented.
* **A new plotted quantity**: add it to `viz.compute_scalars` with its kind (`disp`, `stress`
  or `strain`); the app's menu and unit scaling pick it up.
* **Another element type**: extend `solver.make_basis`.
* **Another material**: pass any symmetric positive-definite `C` to `Material`; to expose it in
  text, extend `_h_material`.
* **Regenerate the API reference** after changing a signature or docstring:
  `python tools/gen_api_docs.py`.

## Known limits and next steps

3D only (no plane stress or strain); one material per model; no click-to-pick regions; no
nonlinear or dynamic analysis; nodal stresses are averaged rather than recovered with a
higher-order patch method; `mesh file` is intended for first-order tet/hex volume meshes and has not yet been exercised
against real mesh files.


## Computer-vision reconstruction path

Photo to FE model, in order of preference:

```text
photo --classical mask (uniform backdrop) or Grounding DINO + SAM (clutter)--> mask
  mask --image_to_mesh (mesh mask)-------------------------------------> MeshTet   (recommended)
  mask --geometric classifier --> template parameters --> STEP --> Gmsh --> MeshTet   (idealised)
```

* **Cheapest segmenter first.** When the image border is a near-uniform backdrop the classical
  mask is exact and the foundation models are never loaded (~0.2 s instead of ~20 s on CPU).
* **No fake masks.** If SAM cannot run, the part is segmented classically inside the detector box;
  the box itself is never measured as if it were the part.
* **Geometry over zero-shot labels.** Grounding DINO often returns several prompt phrases glued
  together for these categories; such labels are treated as ambiguous and a clean mask's geometric
  classification decides.
* **Gmsh from worker threads.** Every Gmsh caller holds `GMSH_LOCK`, and initialisation skips the
  SIGINT handler that made `gmsh.initialize()` fail outside the main thread.
* **UI threading.** Long CV/reconstruction jobs run through `_spawn` in `mos/app/main.py`: work on a
  thread, every state change marshalled back to the trame event loop inside `with state:`.

The single-photo detector in `mos/core/cv_detector.py` estimates a supported mechanical-part
category and feeds editable parameters to the existing OpenCASCADE STEP generators. The separate
`mos/core/multiview.py` path handles multi-view data: it validates and stores distinct images,
records image-quality metadata, can run Nerfstudio image preprocessing for `splatfacto` (3D Gaussian
Splatting) or `nerfacto` (NeRF), and can build a standard COLMAP `images/` + `sparse/0/` dataset
for the Graphdeco 3DGS and external 2DGS trainers. Gaussian PLY import extracts colored splat
centers and supports one-dimension metric scaling; it does not claim to recover a watertight CAD
surface from the render representation.

A trained Gaussian PLY can be reduced to a colored point cloud containing the Gaussian means. A
known physical bounding-box dimension can then set a uniform metric scale. This is only a geometry
preparation step: Gaussian centers are not a watertight boundary representation, so reliable STEP
export still requires surface reconstruction, cleanup and engineering dimension checks.
