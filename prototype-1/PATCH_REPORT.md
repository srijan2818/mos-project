# MoS Analyzer patch report

## Summary

This patch addresses load superposition, force/moment load semantics, support-free quasi-static analysis, and regression coverage. It is a source-level and offline-test patch; the real `scikit-fem` assembly and Gmsh/CAD paths could not be executed in the current environment because those dependencies could not be installed.

## Changes

### Loads and regions

- A `force` applied to a node region is represented as a total point resultant shared across the selected nodes.
- A `force` applied to a facet region is represented as a total force in newtons, distributed as uniform traction over the selected facets. `traction` remains force per unit area in pascals; `pressure` remains inward pressure.
- Each load is assembled additively into the global right-hand side, so overlapping forces, tractions, pressures, body loads, and moments superpose.
- Region indices are canonicalized as a set (sorted and deduplicated), avoiding repeated node/facet IDs causing lost indexed updates or duplicate facet integration.
- A `moment` is assembled as a pure couple with the requested resultant moment and zero resultant force. Its optional `about` point is a reporting/visualization reference, not a physical point of force application.
- The interactive builder creates unique generated region names for repeated loads in box meshes and appends unique regions when repeated loads are applied in script mode. CAD mesh boundary labels can be reused directly.
- Inactive moment controls no longer prevent a force from being applied.

### Supported and support-free models

- Models with effective supports continue to use static linear elasticity and report physical support reactions. Contradictory overlapping prescribed displacements are rejected; insufficient constraints are not silently completed with arbitrary pins.
- Models with no supports use a quasi-static accelerating body-fixed frame at an instant of zero angular velocity. The solver computes mass, volume centroid and inertia, then calculates translational acceleration `a = F/m` and angular acceleration `alpha = I_C^-1 M_C` from the external resultants about the center of mass.
- The D'Alembert body-force density `-rho * (a + alpha x r)` is added to the applied load before solving. A six-mode global gauge removes rigid translation/rotation numerically without pretending those gauge constraints are physical supports.
- This is **not transient dynamics**. Centrifugal and Coriolis terms are omitted because the stated reference instant has zero angular velocity. It does not model elastic vibration, large deformation, or time evolution. An unbalanced load on a support-free body requires a positive density and a valid positive-definite inertia tensor.
- The result object includes applied/reaction moments, inertial resultants, acceleration values, frame flag and a normalized force/moment equilibrium indicator.

### Validation and tests

- Added pure-Python tests for resultant force/moment calculations, Newton-Euler accelerations, D'Alembert resultants, gauge constraints, duplicate region IDs, overlapping point loads, material/load validation, and conflicting supports.
- Quadrature validation accepts finite signed quadrature coefficients (some valid rules use negative weights) while still requiring a positive integrated volume; non-finite coordinates and weights now produce distinct diagnostics.
- Added real-FEM regression tests for same-face surface-force superposition, same-patch moment superposition, free fall, and angular acceleration under a pure moment. They are currently skipped because `scikit-fem` is unavailable in this runtime.
- Extended the mocked app smoke test to apply repeated forces and moments in interactive mode, to append repeated loads in script mode, and to ensure invalid inactive moment input does not block force entry.
- Updated theory, language, architecture, testing and API documentation, as well as the README, to describe the updated semantics and limitations.

## Verification in this environment

Latest expected offline check set:

- `pytest -q -rs`: pure-Python tests pass; FEM/CAD tests requiring missing dependencies skip.
- `python -m compileall -q mos tests tools`: syntax compilation.
- `python tools/gen_api_docs.py --check`: generated API reference freshness.
- `python tools/smoke_app.py`: mocked UI load-entry regression.

The mocked smoke test is independent of pytest. Its invalid-input branch intentionally exercises app error reporting, so a traceback is printed while the overall smoke test still passes.

## Dependency blocker and required final validation

`scikit-fem`, `gmsh`, `meshio`, and `pyvista` are not installed in this runtime. Attempts to install via pip/uv and to update the OS package lists failed because external package hosts could not be resolved; no local cached wheels were found. Thus this environment cannot validate the actual scikit-fem assembly APIs, finite-element quadrature/load integration, Gmsh STEP meshing or browser rendering.

Before relying on results for engineering work, install the dependencies from `requirements.txt` in a network-enabled environment and run `pytest -q` with no FEM/CAD skips. Specifically inspect the beam theory benchmarks, zero-resultant moment tests, free-body acceleration regressions, STEP meshing tests and visual behavior for multiple load glyphs. A passing mocked test is not a substitute for that run.


## Floating-point comparison audit

Reviewed Python comparisons in the application. Structural comparisons (shapes,
strings, IDs, counts, enums) intentionally remain exact. Exact comparisons of
computed floating-point results in tests were replaced with explicit absolute
tolerances, and the coordinate-predicate test now uses a tolerance band. The
expression DSL still supports `==`/`!=` for deliberate exact comparisons, but its
documentation recommends `abs(x - target) < tolerance` for spatial coordinates,
especially on imported CAD meshes.


## CAD loading fix (2026-10-13)

- `load_step_mesh()` now creates a Gmsh model after `gmsh.clear()` and before STEP import.
- Backend errors are surfaced as `CadImportError` with the STEP filename attached.
- `mesh step` syntax now supports quoted paths and paths with spaces; the interactive builder quotes paths and resolves built-in preset files relative to the project directory.
- Added offline regression tests for the Gmsh model lifecycle and STEP paths with spaces. Real Gmsh/OpenCASCADE meshing still requires the optional runtime dependencies.
