# Testing

## Running

```text
pytest                              # everything; test_fem.py is skipped if scikit-fem is missing
python tools/smoke_app.py           # the app's own logic against stand-ins for trame/PyVista/scikit-fem
python tools/gen_api_docs.py --check   # exit 1 if docs/api.md is stale (a test checks this too)
```

## What exists

| File | Tests | Covers |
|---|---|---|
| `tests/test_core.py` | 17 | Hooke's law and `C4` symmetries; text fields; `step`, `where`, comparisons and boolean logic; rejection of unsafe or malformed expressions; region selection; moment/torsion/bending traction statics; degenerate faces; von Mises, principal values, Tresca; nodal averaging |
| `tests/test_loads_material.py` | 4 | loads built from raw numbers, load/region/field validation, material validation, a general (non-isotropic) `C` |
| `tests/test_spec.py` | 10 | every directive and load kind, region sizes, prescribed displacements, about 30 error cases checked for their line numbers and wording, parameter ordering, building all four example models |
| `tests/test_post_fake_basis.py` | 2 | strain/stress recovery against Hooke's law for a known gradient; weighted cell averages |
| `tests/test_solver_logic.py` | 8 | the solver's DOF bookkeeping with a fake basis: component DOFs, constraints (components, values, unions, callable values), point-force splitting, non-finite loads, the equilibrium-error formula, `Results` accessors |
| `tests/test_viz.py` | 2 | plottable scalar fields, deformation scale, clip-plane origin |
| `tests/test_docs.py` | 5 | every `mos` example in the docs parses and builds; `api.md` is current; doc links resolve; every directive, function and example model is documented; dependency list matches |
| `tests/test_docstrings.py` | 1 | every module, public class, function and method has a docstring |
| `tests/test_fem.py` | 7 | **needs scikit-fem**: beam-theory benchmarks (below) |
| `tools/smoke_app.py` | script | solve button, redraw handlers, template menu and error reporting in `mos.app.main` |

`tests/fakemesh.py` is a pure-numpy tetrahedral box mesh with the attributes the region code uses; it
lets spec building be tested with no FE library.

## Status: what has and has not been run

Run, with all 49 tests (plus the app smoke script) passing:

* everything in the table above except `test_fem.py`.

Line coverage of that suite, measured with the standard library's `trace`:

| Module | Lines executed |
|---|---|
| `fields`, `material`, `regions`, `loads`, `model`, `post`, `templates` | 100 % |
| `spec` | 96 % (the rest is `make_mesh`, which needs scikit-fem) |
| `viz` | 91 % (the rest is `to_pyvista`, which needs pyvista) |
| `solver` | 56 % (the rest calls scikit-fem directly) |

**Never run against the real libraries** (they could not be installed where this code was written):

* `solver.make_basis`, `assemble_stiffness`, the facet-load branches of `assemble_loads` and `solve_case`:
  everything that calls scikit-fem. The code is written for scikit-fem >= 8 from knowledge of its
  API and may need small corrections.
* `meshes.box_mesh`, `spec.make_mesh` and so the `mesh file` directive.
* `viz.to_pyvista`, and the whole trame/PyVista user interface, including hexahedral meshes.

The FE benchmarks and the smoke script exist to find exactly these problems quickly.

## The FE benchmarks (`tests/test_fem.py`)

Cantilever, L = 10, H = W = 1 (L/H = 10), E = 210 GPa, nu = 0.3, a coarse 20 x 2 x 2 P2 mesh, fixed at x = 0.
Formulas are in [theory.md](theory.md), section 8.

| Test | Checks | Tolerance |
|---|---|---|
| tip shear load | tip deflection vs `P L^3 / 3EI`; reaction = P; equilibrium error < 1e-8 | 5 % |
| tip moment | tip deflection vs `M L^2 / 2EI`; net reaction ~ 0 | 5 % |
| self-weight | tip deflection vs `q L^4 / 8EI`; reaction = weight | 5 % |
| point force | reaction = applied force; equilibrium | 1e-8 |
| midspan bending stress | smoothed nodal `sigma_xx` on the top surface vs `M y / I` | 15 % |
| text-spec cantilever | the same tip-load problem built from the example text | 5 % |
| text-spec torsion bar | twist of the square bar vs `T L / G J` | 10 % |

The tolerances are first guesses, not tuned. Reading a failure:

* **Equilibrium or reaction wrong by a lot** (more than ~1e-6): a bug in assembly, constraints or
  the load vector. Not a mesh issue.
* **Deflection off by tens of percent or more**: suspect the code (units, a wrong element, a
  missing load term).
* **Deflection a few percent off**: refine the mesh (e.g. 40 x 4 x 4) before suspecting the code.
  The fixed-end constraint stiffens the beam and shear deformation adds about 1 %, so Euler-Bernoulli
  is only an approximation.

## First run checklist

1. `pip install -r requirements.txt`, then `pytest tests/test_solver_logic.py` (fast; no solve).
2. `pytest tests/test_fem.py -x -q`. If the first test errors out, the likely culprits, in order:
   `basis.nodal_dofs` / `edge_dofs` attribute names; `basis.get_dofs(...).all()`; passing
   `DiscreteField(...)` as a keyword to `asm` and reading `w["q"]`; `fb.normals`;
   `basis.interpolate(u).grad`; `basis.doflocs`; the `ElementHex2` / `MeshTet` imports.
   Paste the traceback; each is a one-line fix.
3. `python -m mos.app`, press Solve on the default example. Likely culprits: `plotter_ui(pl, mode="server")`,
   `layout.drawer.width`, `view.update`, the slider `disabled` expression.

## What the tests cannot tell you

* Accuracy on geometries other than a beam, or at stress concentrations (the smoothed nodal field
  underestimates peaks by construction; see theory.md section 6).
* The local accuracy of a moment load near the loaded face (statically equivalent, not identical).
* Anisotropic materials through the FE solver (only the `C` handling is tested), hexahedral meshes,
  performance on large meshes, or how the viewer looks in a browser.

## Defects the tests found while this suite was written

Recorded because they show what the suite is for, and they are fixed:

* `param 3 = x` was accepted as a parameter named `3`.
* `1/0` and `log(-1)` evaluated silently to `inf`/`nan` instead of raising; a non-finite number
  could reach a load vector. Constants are now rejected and the assembled load vector is checked.
* The equilibrium error divided by the applied force, which is about zero for a model driven only
  by a prescribed displacement, so it would have reported a huge false error. It now divides by the
  larger of the applied-force and reaction magnitudes.
* Several first-draft assertions were vacuous (`... or True`) and a region test had a wrong
  expectation; both were replaced by real checks.
