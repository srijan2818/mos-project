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


## Learning Lab UI refresh (2026-11-03)

- Added the point-stress tensor explorer, arbitrary-plane traction, normal/shear decomposition, principal stresses and directions, and 3D Mohr circles.
- Added a simple A/B snapshot comparison for maximum principal stress and independent idealised checks for beam bending, shaft torsion, and thin-wall pressure vessels.
- Kept idealised analytical checks separate from FEM solution data and documented their assumptions in `docs/learning-lab.md`.
- Added pure-Python regression tests in `tests/test_learning.py`.

## Analysis workspace and live-response integration (2026-11-03)

- Integrated point analysis into a contextual dock over the existing 3D viewport. The established model-definition workflow, global toolbar appearance, camera, axes, and view controls are retained. The dock does not create a third side-by-side panel.
- Added an `Explore` action to applied-load rows and an entry point in Results. Point selection, plane selection, the active load, stress tensor, Mohr circles, and traction remain in one context.
- Added a reusable supported-body linear solver cache: the constrained stiffness matrix and sparse LU factorisation are reused for the initial solve and prepared load-response bases. Force/moment components use X/Y/Z unit responses; uniform pressure uses one unit-pressure response. Slider and component changes compose those fields by linear superposition without another FE backsolve.
- Live preview is limited to supported linear-elastic models, selected point/surface forces, moments, and uniform numeric pressure. Geometry, mesh, material (including density), and support changes invalidate the cached state and require Solve. Unsolved model edits are labelled as stale in the analysis dock.
- The viewport shows load glyphs whose size responds to load magnitude and, while point analysis is open, the sampled node, selected plane, plane normal, and traction direction.

Validation in this edit runtime: `pytest -q` reports **85 passed, 10 skipped**; `python tools/smoke_app.py`, `python tools/gen_api_docs.py --check`, and `python -m compileall -q mos tests tools` pass. The smoke test intentionally exercises an invalid-script error branch before reporting success. Real scikit-fem assembly, Gmsh meshing, PyVista rendering, and browser interaction could not be exercised here because scikit-fem, Gmsh, meshio, and PyVista are not installed.

## Learning Lab viewport stability fix (2026-11-03)

- Opening/closing the integrated Lab no longer clears and reconstructs the complete PyVista scene. The base mesh and active camera remain intact; only the selected-point/plane annotation actors are refreshed.
- Point and plane changes update a small dedicated overlay layer rather than re-clipping, re-uploading and re-rendering the full FE mesh.
- PyVista point picking is registered once. Previously each full scene redraw attached another interactor observer, which could accumulate callbacks and progressively degrade interaction responsiveness.
- Full scene redraws preserve the current camera position unless a deliberate `reset_camera=True` solve/reset is requested.
- Added smoke regressions asserting that Lab navigation and sample-coordinate changes do not clear the model scene or register duplicate point-picking observers.

## Learning Lab navigation/render regression fix (2026-11-03)

- Lab navigation no longer toggles the main setup drawer. Previously, opening the overlay collapsed the drawer in the same state update that opened the dock. This resized the server-rendered VTK host during the transition and could leave the canvas cropped/zoomed toward a corner in the browser. The user's existing drawer state is now preserved on open, close, and load-exploration entry.
- The analysis overlay batches sphere/plane/vector actor additions with `render=False`, followed by one view update, instead of allowing each actor addition to trigger its own render.
- FEA point sampling suppresses the tensor watcher while writing the sampled tensor/plane values into state, then refreshes the Mohr/tensor SVGs once. This removes duplicate analysis rendering caused by the state watcher and the explicit refresh both running for the same sample.
- The app smoke test now asserts that Lab open/close preserves the drawer state in addition to preserving the VTK scene and single point-picker registration.


## Learning Lab visibility and camera-jump fix (2026-11-03)

- The Lab-open state transition no longer calls the PyVista overlay updater or the
  Trame view-update callback. Opening/closing the analysis UI is presentation-only;
  it cannot add VTK actors, render the viewport, or disturb the camera.
- Opening the dock always samples the latest published `Results` object and refreshes
  point-stress/traction state before showing the tensor readouts. Unsolved models show
  an explicit solve-first state.
- The dock was moved out of the VTK host wrapper and is now a fixed, high-z-index
  viewport overlay. It no longer participates in the canvas' sizing/stacking context,
  so it cannot shrink, reflow or be clipped by the plotter container.
- Smoke coverage now asserts that opening the dock causes no VTK actor additions,
  no view update, no scene clear, no picker re-registration, and that solved stress
  values are populated when the dock opens.


## Universal single-image image-to-3D workflow (2026-11-06)

- Added a separate **Universal Image → 3D → CAD Viewer** panel. It accepts JPG, PNG, WEBP, and BMP uploads (browser upload limit: 15 MB) and offers estimated depth-relief or silhouette-only preview modes.
- Added foreground extraction from transparency, optional `rembg[cpu]`, or a plain-background border-color estimate, with a full-frame mode for deliberately rectangular reliefs.
- Added lazy Depth Anything V2 Small inference using the Transformers depth-estimation pipeline. The result is normalized relative depth, not metric depth; a user-entered width sets only an approximate scene scale.
- Generates a colored, closed PLY relief mesh with a planar back. The viewport preview is isolated from the active FEM model and solver state. Transparent inner cutouts are preserved in the PLY preview and covered by a closed-edge-incidence regression.
- Optionally exports a separate constant-thickness STEP silhouette through the existing image-to-STEP/Gmsh code and offers an explicit action to load that STEP through the existing CAD/FEM importer. The current STEP contour conversion may omit inner holes/cutouts; the UI warns about this and the STEP is not the depth-relief mesh.
- Added `docs/image-reconstruction.md`, updated app/API documentation, and added `requirements-image3d-optional.txt` for optional higher-quality background removal.
- Validation in this edit runtime: full unit tests, API-doc consistency, and Python compilation pass. Depth model download/inference, actual Gmsh STEP generation, and live PyVista/browser rendering were not exercised in this environment; the app smoke script did not complete within its execution window.
