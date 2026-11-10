# Testing

## Running

```text
pip install -r requirements.txt
pytest -q
python tools/smoke_app.py
python tools/gen_api_docs.py --check
```

`scikit-fem` is required for the real elasticity assembly and FEM benchmarks. Gmsh is required
for STEP import/meshing. Tests that need unavailable optional system libraries skip rather than
preventing parser, load, solver-helper, documentation, and mocked-UI tests from running.

## Coverage map

| Test file | Main coverage |
|---|---|
| `tests/test_core.py` | Elasticity tensor symmetries, field expressions and region selectors, moment statics, degenerate patches, stress invariants, nodal averaging |
| `tests/test_loads_material.py` | Load and region validation, unique region-index semantics, material finite/symmetry/positive-definite checks, anisotropic stiffness input |
| `tests/test_spec.py` | Text parser directives and errors, force/traction distinction, model building, optional-support parsing and CAD-stack skip behavior |
| `tests/test_solver_logic.py` | DOF bookkeeping, overlapping point-load superposition, incompatible support checks, resultants, Newton–Euler acceleration, d'Alembert force/moment resultants, free-body rigid-mode gauge |
| `tests/test_post_fake_basis.py` | Strain/stress recovery against Hooke's law and weighted cell averages |
| `tests/test_viz.py` | Scalar fields, deformation scale, clip-plane origin and load glyph handling |
| `tests/test_learning.py` | 3D tensor eigenvalues, traction decomposition, Mohr-circle SVG, and beam/shaft/vessel checks |
| `tests/test_docs.py`, `tests/test_docstrings.py` | Documentation examples/links, generated API freshness, documented dependencies and public docstrings |
| `tests/test_fem.py` | Real scikit-fem beam benchmarks, multiple overlapping loads, free-body acceleration/inertia regression cases |
| `tests/test_image_to_mesh.py` | Mask contours with holes, direct mask meshing (volume, face roles, outward normals), `mesh mask` cantilever vs beam theory, Gmsh from a worker thread, STEP face normals, vectorised nodal averaging and RDP, detector label normalisation, halo-free classical masks |
| `tests/test_cad.py`, `tests/test_cad_fem.py`, `tests/test_image_to_cad.py` | STEP parsing, Gmsh meshing, CAD boundary mapping and image-assisted meshing; CAD/FEM portions require their dependencies |
| `tests/test_object_selection.py`, `tests/test_multiview_hull.py` | Choosing the part (clicks, candidate rule, automatic guess; SAM tests skip without weights); three-photo hull from synthetic orthographic and pinhole silhouettes (`tests/synthetic_views.py`): per-photo scales, box and L-bracket size/volume, through-hole, portrait rotation, proportion mismatch from a wrong object, a mirrored photo that silhouettes cannot detect, measured perspective error, tet mesh faces, `mesh hull` cantilever |
| `tests/test_limit.py` | Proportional-limit check: von Mises and Coulomb–Mohr utilisation (uniaxial, pure shear), material/spec validation, load factor 1 / max(U) exact for a linear solve |
| `tests/test_app_handlers.py` | Real app handlers against the `tools/smoke_app.make_fake_app` stand-ins: box face predicates after resizing, stale CAD-face references, material presets, input validation, script-mode edits, `--spec`, upload wiring, CV result/state resets, point picking, deferred Lab watchers, async Solve with a real event loop, served CSS. `sys.modules` is restored so the fakes never reach the FEM tests |
| `tools/smoke_app.py` | App handlers, Learning Lab reactivity, solve-state behavior, template menu, CV-to-CAD controller wiring and surfaced input errors using stand-ins for runtime UI dependencies |

`tests/fakemesh.py` is a pure-NumPy tetrahedral box mesh that supports region/spec tests without
scikit-fem. The solver-helper tests use a fake basis to exercise bookkeeping and mechanics formulas,
not the actual scikit-fem assembly implementation.

## Validation recorded for this patch

Environment: miniforge `FEM` env, Python 3.11, numpy 2.4, scipy 1.17, scikit-fem 12.0, gmsh 4.15,
pyvista 0.49, trame 4.0, torch 2.15 + torchvision 0.29 (cu132, RTX 5050 under WSL), transformers 5.22.
With CUDA the detector uses Grounding DINO-base + SAM 2.1-large (cluttered test scene: ~1.1 s warm,
mask IoU 0.99); on CPU the tiny variants are used.

* `pytest -q`: **150 passed, 0 skipped** (37 s with the GPU) (real scikit-fem and Gmsh; nothing mocked except where a
  test says so). `python tools/smoke_app.py` and `python tools/gen_api_docs.py --check` pass.
* The app handlers were also driven against the real trame state, off-screen PyVista and the real
  solver: default solve, Learning Lab sampling, CV analysis of all sample photos (~0.2 s each, all
  classified correctly), direct outline mesh + solve (4-22 s up to ~70k DOFs) and template CAD
  (~1 s); equilibrium errors <= 3e-11. This is not a visual browser check.

The CV detection tests use synthetic images; they show the pipeline is consistent, not that it is
accurate on real workshop photos.

## Physics regression cases expected with the full dependencies

The FEM tests include ordinary cantilever checks (tip force, applied moment, self-weight, axial tension
and stress), plus targeted load-frame regressions:

* two surface forces on the same facet set add to the same resultant as one equivalent force;
* two moments on the same patch add as a pure couple without introducing net force;
* a freely falling body under a uniform body-force field has matching rigid-body acceleration and
  negligible elastic stress in this quasi-static model;
* a free body under a nonzero pure moment receives angular acceleration and a distributed Euler
  pseudo-load whose resultant moment cancels the applied moment.

Tolerances are regression thresholds, not universal accuracy guarantees. A failure in resultant
force/moment balance points to load assembly, constraints or conditioning; a deflection error of a few
percent may need mesh refinement before the formulation is blamed. Peak stress at singular point loads
or geometric corners should be assessed by convergence, not one mesh's smoothed nodal values.

## Important model limits

The free-body feature is not transient dynamics. It assumes an instantaneously non-rotating frame and
uses `b_inertial = -rho * (a + alpha x r)`; centrifugal and Coriolis terms vanish only under the
zero-angular-velocity assumption. It does not calculate time evolution, elastic vibration, changing
angular velocity or large deformation. Supported models still use ordinary static linear elasticity.
