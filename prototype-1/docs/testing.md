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
| `tests/test_docs.py`, `tests/test_docstrings.py` | Documentation examples/links, generated API freshness, documented dependencies and public docstrings |
| `tests/test_fem.py` | Real scikit-fem beam benchmarks, multiple overlapping loads, free-body acceleration/inertia regression cases |
| `tests/test_cad.py`, `tests/test_cad_fem.py`, `tests/test_image_to_cad.py` | STEP parsing, Gmsh meshing, CAD boundary mapping and image-assisted meshing; CAD/FEM portions require their dependencies |
| `tools/smoke_app.py` | App handlers, solve-state behavior, template menu and surfaced input errors using stand-ins for runtime UI dependencies |

`tests/fakemesh.py` is a pure-NumPy tetrahedral box mesh that supports region/spec tests without
scikit-fem. The solver-helper tests use a fake basis to exercise bookkeeping and mechanics formulas,
not the actual scikit-fem assembly implementation.

## Validation recorded for this patch

In the current environment, the latest complete offline run reports **62 passed, 9 skipped**;
the app smoke script exits successfully, and source compilation succeeds. The skipped tests require
`scikit-fem` and/or Gmsh. Attempts to install them with pip and the system package manager failed
because the environment could not resolve package hosts; neither package is installed here.

Therefore these checks validate pure-Python logic and test doubles, **not** a complete real-library
FEM/CAD run. Before relying on engineering results, install `requirements.txt` in a network-enabled
environment and run `pytest -q`, paying particular attention to `tests/test_fem.py` and CAD integration
tests. In particular, the scikit-fem assembly APIs, element definitions, and real mesh facet quadrature
still need that integration run. Do not interpret mocked smoke tests as visual browser validation.

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
