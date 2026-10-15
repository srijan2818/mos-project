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

## CAD face-region regression (2026-10-14)

- Fixed the first-load path for STEP files in the interactive builder. Previously, it could generate a selector such as `region neg_x facets x < 1e-6` using the default or previous model's bounds before the new STEP mesh existed; the selector then matched no facets on the imported model.
- The app now builds the selected STEP mesh first, gets its actual bounds and named CAD face sets, resolves generic `-x`/`+x`/etc. face choices against those named faces, then builds the full model using the already imported mesh.
- Generic side mapping now chooses the available CAD boundary whose facet centroid is furthest toward that side, instead of arbitrarily choosing the first face. The named boundary set is used directly, avoiding fragile coordinate predicates for CAD faces.
- Extended the mocked app smoke test to reproduce the first-load case with default `-x` support and `+x` load on a STEP path containing spaces. It asserts that no `neg_x` coordinate region is generated and that the support/load resolve to the named CAD boundaries.
- The mocked regression passes, but true STEP-import/mesh behavior still needs confirmation in an environment with Gmsh/OpenCASCADE and scikit-fem installed.

## Inertia-tensor numerical root-cause audit (2026-10-15)

- The prior version symmetrized the tensor inside `mass_properties()` before checking the raw skew. That could hide a tensor-construction or quadrature coordinate/weight alignment bug. This is corrected: raw asymmetry is measured **before** cleanup. Relative skew above `1e-12` now raises a diagnostic at the integration point; only skew at round-off scale is averaged with the transpose.
- The previous `body_frame_acceleration()` check was not enough to audit the production path, because `solve_case()` receives the tensor from `mass_properties()`, which had already symmetrized it. The production path now checks at the source as well as the Newton-Euler boundary.
- Coordinates are shifted to a local reference before computing the centroid and inertia. This reduces cancellation for CAD parts located far from the global origin and makes inertia translation-invariant up to the precision available in the input coordinates.
- Added regressions for translation invariance, rejection of significant raw skew, round-off-only cleanup, and non-finite quadrature values.
- Offline result: `pytest -q` reports 72 passed, 9 skipped. Numerical helper tests pass, but the real `scikit-fem`/Gmsh integration remains unverified here because these dependencies are unavailable in this runtime.
