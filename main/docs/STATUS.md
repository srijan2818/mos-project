# Status

Working log for whoever (person or Claude) picks this project up next. Update it at the end of
every work session: move finished items to **Done**, record choices in **Decisions**, and leave
**Next** as concrete steps.

_Last updated: 2026-11-09_

## Done

### Three-view solid on an L-shaped marker mat, Step 2 (2026-11-09, night)

`pytest -q` **226 passed**. Installed with your OK: `opencv-contrib-python` 5.0 and
`scikit-image` 0.26 (numpy unchanged).

* `mos/core/marker.py`:
  * FLOOR + BACK ChArUco sheets (two-page PDF).
  * `solve_rig`: per-photo poses, the BACK pose from a photo seeing both sheets, joint
    least-squares refinement, and the focal length from the photos.
  * `sheet_mask` (projected sheet outlines).
* `mos/core/multiview_hull.py`:
  * Coarse-to-fine voxel carving and per-view agreement (IoU).
  * `project_hull`; marching cubes; Gmsh volume filled directly on the marching-cubes triangles
    (re-meshing failed on a real session).
  * Faces grouped by normal; `mesh hull DIR [h]` spec directive.
* **Selection on the mat:**
  * Sheet outlines are excluded; without that, the sheets were chosen as the part.
  * Once two other photos are accepted, their hull projected into this photo picks the prompt and
    the mask. Synthetic top view: IoU 0.54 → 0.96.
* **App panel** "Three-view solid on a marker mat": sheets PDF, 4 guided photos (camera/file),
  Check marker mat, Select part / Accept per photo, Build solid & solve. Browser-verified with
  synthetic photos: photo 1 flagged (0.58), re-selected (0.98), solid 64 × 42 × 30 mm for a
  60 × 40 × 30 box, solved.
* **Measured** (box, exact silhouettes):
  * Flat mat at 35° / 20°: height +78 % / +39 %.
  * L-mat: height within 1 mm, volume +5–7 %.
  * Plan size +5–7 % from perspective in every setup.
  * Bar cantilever: 21.8 vs 20.6 µm (beam theory).

### CV object selection, Step 1 of the multi-view plan (2026-11-09, evening)

Plan: `~/.claude/plans/analyse-the-repo-and-encapsulated-simon.md` (choose the right object, then
reconstruct from 3 views or a short video on a printed marker mat). `pytest -q` **212 passed**.

* **Cause of the whiteboard failure:** the object was chosen by Grounding DINO from six category names;
  an unknown part matched none, so the best box was the bright board.
* **The object is now chosen by clicks** (`cv_detector.segment_from_points`, SAM 2 point prompts,
  ~0.8 s per click on the GPU). Without clicks, `suggest_object_point` guesses: a 5×3 prompt grid,
  scored on centrality, border contact, skin colour and size. DINO only suggests a category.
  On your photo the guess outlines the cube; one click on the pink cap adds it.
* `default_candidate`: the largest SAM candidate that is not background-like and scores ≥ 70 %
  of SAM's best. Measured: whole part 0.39 vs one face 0.41 on the real photo; junk 0.40 vs 0.99
  on a synthetic scene.
* **App:**
  * Click the preview to add points; Remove mode or shift-click for negative points.
  * Smaller/Larger steps through SAM's masks; Clear clicks returns to the guess.
  * A first click near the guess extends it, a far one replaces it.
  * **Accept mask** gates both mesh buttons, in the UI and on the server.
  * The camera button uses `getUserMedia` (live preview + Capture).
  * Browser-verified with your photo.
* Tests: `tests/test_object_selection.py` (whiteboard/hand/part scene, click selection, candidate
  rule; SAM tests skip if the weights can't load) and app click-flow tests.
* Seen, not fixed: the mask mesh of the 7 cm cube at the default size had 18.6k tets / 95k DOFs
  (the 12k-tet budget estimate under-counts) and took 138 s to solve (CG).

### Audit + fixes session (2026-11-09, later)

`pytest -q` **204 passed** (was 150); `tools/smoke_app.py`, `gen_api_docs.py --check` and
`compileall` pass; all 6 CV sample photos still classify correctly. UI checked in a real browser
(Playwright + Chromium; screenshots were in the session scratchpad, not the repo).

* **Silently wrong results fixed** (each with a test that failed first):
  * Box mode used the *previous* solve's bounds (L 10→20: tip load on half the beam, 9.8 mm instead of 15.2).
    Face predicates are now symbolic (`x > L - 1e-6`).
  * An unknown face key became `x > 0`.
  * The material preset dropdown did nothing (always steel).
  * A `nodes` support only fixed vertex DOFs on P2 (11 % too soft). Now `solver.node_region_dofs`.
  * The Stepped Shaft template clamped the side surface (now ends `cad_face_3`/`cad_face_5`).
  * Point pressure was dropped.
  * Bad moment/vector input was replaced by defaults.
  * Script text was overwritten by Add/Remove.
  * `--spec` was ignored.
* **Broken features fixed:**
  * All file uploads (Vue `change`/`$event` + `mos/app/static/upload.js`).
  * COLMAP button `NameError`.
  * Multi-view method select.
* **CV detector:**
  * `_count_concave` counted convex corners (now 45° turn threshold).
  * The symmetry rotation sign was wrong.
  * Fake measurements are now `None`/errors; an unclassified outline keeps its mask.
  * Webcam/file mode with no image analysed the sample.
  * Stale results were not cleared.
  * `cv_running` could stick.
* **Stale state:**
  * `cad_face_N` supports/loads are dropped when the geometry source changes.
  * A failed solve restores faces/bounds.
  * The Lab face plane resets.
  * The reconstruction preview is cleared by Solve.
  * Box rebuild no longer maps ±x through a previous STEP mesh.
* **UI:**
  * Root cause of the Lab dock stretching the canvas: Vue drops `<style>` in templates, so *no* app CSS
    ever applied. It's now `mos/app/static/app.css`, served by the trame module.
  * Colour-bar title overlap, Mohr label overlap, clipped toggles/selects, duplicate titles,
    non-reactive toolbar bindings, oval icon buttons, overflowing COLMAP label.
  * Support options had duplicate values ("Pinned" ≡ "Fixed"); symmetry fixed the wrong DOFs.
* **`run_solve` is non-blocking**: worker + `_ui_wait` + `on_done`. A 41k-DOF solve (32 s) now
  keeps UI round-trips ≤ 0.6 s.
* **Core:**
  * Gmsh lock leak in `image_to_step`.
  * PLY preview walls inward.
  * Preview aspect distortion.
  * Outline traced from the first fragment instead of the largest.
  * Zero plane normal silently replaced.
  * Interior faces of multi-solid STEP files offered as load faces.
* **Spec:** a duplicate `mesh`/`material` line, `param E/nu/rho` after `material`, and very deep
  nesting are now errors. The `neg_x`… aliases are documented as reserved, and an override is logged.
* **Rules:** 15 swallowed exceptions now log; float `==` replaced; 0-instead-of-None fixed;
  `requirements.txt` adds Pillow and `scipy>=1.12`; generated files go to the git-ignored
  `cad_models/_generated/`.
* `tools/smoke_app.py` exposes `make_fake_app()`; the new `tests/test_app_handlers.py` drives the real
  handlers with it.

### Earlier sessions

Verified in the miniforge `FEM` env (Python 3.11, scikit-fem 12, gmsh 4.15, torch 2.15 +
torchvision 0.29 cu132, transformers 5.22, RTX 5050 visible in WSL): `pytest -q` **150 passed, 0
skipped**; `python tools/smoke_app.py` and `python tools/gen_api_docs.py --check` pass. App handlers
were also driven headlessly against the real trame state, off-screen PyVista and the real solver
(not a visual browser check).

* Removed 149 Windows `*:Zone.Identifier` files.
* **Direct image → mesh** (`mos/core/image_to_mesh.py`, directive `mesh mask PATH size thickness
  [mesh_size]`): mask → outline + holes → in-memory Gmsh → layered tet extrusion with named faces.
  Image-meshed cantilever is within 6 % of Euler–Bernoulli; meshing takes < 1 s.
* **CV detector** (`mos/core/cv_detector.py`): classical segmentation first (≈0.2 s on a plain
  backdrop, models never loaded); Grounding DINO + SAM only for cluttered scenes (≈1.1 s warm on
  GPU, mask IoU 0.99 on the synthetic test). All 6 sample photos classified correctly (was 4/6).
  Fixed: box used as a fake mask when SAM failed, SAM 2 post-processing under transformers 5,
  glued DINO labels, mask halo (+37 px per side), upside-down annotation overlay, pipe/I-beam and
  shaft/beam confusion, `labels` → `text_labels`.
* **Gmsh threading** (`mos/core/gmsh_session.py`): one `GMSH_LOCK`; initialise without the SIGINT
  handler, so Gmsh works from worker threads (reconstruction STEP export previously always failed).
* **Solver** (`mos/core/solver.py`): `solve_spd` uses LU ≤ 40k DOFs, scaled CG above (AMG if
  `pyamg` is installed); free-body solve refuses > 120k DOFs. Fast isotropic assembly (~2×). This
  fixes the WSL out-of-memory crash (~190k-DOF direct solve).
* **Logic fixes**: outward CAD face normals (`regions.boundary_normal`; were wrong on 5/6 beam faces);
  stale `cad_face_N` supports/loads now raise instead of silently moving to face 1; STEP import
  drops unused nodes and fragments multi-solid files; `image_to_step` keeps holes and picks the
  largest outline by area; template CAD no longer overwrites the reconstruction STEP and reports
  solve failures.
* **Performance**: vectorised nodal averaging, Gmsh→MeshTet conversion, RDP, PLY writer.
* **UI** (`mos/app/main.py`): CV analysis, reconstruction, Nerfstudio and COLMAP jobs run through
  `_spawn` (state changes marshalled to the trame loop); new buttons *Mesh Photo Outline & Solve
  FEA* and *Mesh Foreground Mask Directly & Solve*; mask meshes sized from the part (~20 elements
  across, 12k-tet budget).
* `CLAUDE.md` imports this file (`@docs/STATUS.md`) and says never to commit or push unless asked.
  Tested with fresh headless sessions with all tools off. Started in this folder, the session
  quoted STATUS.md verbatim, and a control file that isn't imported was correctly absent. Started at
  the repo root, it loaded neither CLAUDE.md nor STATUS.md.
* Docs updated: `model-language.md`, `architecture.md`, `app.md`, `image-reconstruction.md`,
  `testing.md`, regenerated `api.md`; `torchvision` added to `requirements.txt`.

## In progress

Nothing is half-edited. Open state of the repository:

* Nothing from these sessions is committed. The project folder was renamed by you to `main/`
  (was `mos_work_detection_darklab_fix/`); the previously tracked
  `mos_analysis/mos_work/`, the zips and `prototype-*` folders show as deleted in `git status`
  (deleted outside these sessions).
* Still synchronous on the UI thread (small or rare): drawing → STEP conversion (Gmsh), template-CAD
  generation, PLY inspect/calibrate, the live-load prepare step.

## Decisions

* **Stress recovery** is unchanged for now (cell means averaged to vertices), *pending your call*.
  Measured on the default 20×2×2 cantilever: mid-span surface σxx 1.78 vs 3.00 MPa (−41 %),
  peak von Mises 3.83 vs 6.0 MPa. Evaluating each element's stress at its own vertices, then
  averaging, gave 3.03 MPa (+0.8 %) on the same mesh. Tip deflection is already within 0.3 %.
* **App CSS/JS are served files** (`mos/app/static/`): Vue drops `<style>` inside templates.
  The old font overrides were not revived; they never applied.
* **`neg_x`/`pos_x`/… are reserved region names** on meshes with named boundaries: they override
  the condition, because legacy offset-STEP predicates can match the whole boundary. The override is logged.
* **Concave corner = turn sharper than 45°.** Measured: real corners ~90°, RDP artefacts ≤ 27°.
  A corner chamfered into two turns of ≤ 45° is not counted.
* **Changing the geometry source drops `cad_face_N` supports/loads** (± box keys are kept and
  re-mapped). Keeping them would silently move them, since mask meshes always have faces 1 and 2.
* **Support options**: no separate "Pinned" (identical to Fixed for solids); "Fix X only ·
  symmetry plane ⊥ X" fixes x only.
* Kept as-is after review: the live-load per-keystroke recompute (it composes cached unit fields),
  the mobile layout, and the `force` preset with a negative magnitude (`--1e5` evaluates to +1e5).

* **Direct mask meshing is the recommended photo → FE path**; template CAD stays as an optional
  idealised part. Reason: it meshes what was segmented, can't pick the wrong category and keeps
  arbitrary holes. Its one assumption is constant depth normal to the photo.
* **Cheapest segmenter first; geometry outranks zero-shot labels.** DINO returns glued phrases for
  these categories. A glued label is treated as no label, and a clean mask's geometric class decides.
* **Never measure the detector box as if it were the part.** If SAM fails, segment classically
  inside the box and record why.
* **Outlines follow pixel edges** (0.5 px outward offset of Moore-traced centres). Without it a
  30-px section loses ~10 % bending stiffness.
* **Mask convention**: alpha channel if present, else dark (< 128) = material. This matches
  `image_to_step` and the reconstruction masks.
* **Face names**: mask meshes use `cad_face_N` (1 bottom, 2 top, then outer walls, then hole walls)
  so the existing UI and `_resolved_face_name` treat them like STEP faces.
* **Linear solver by size**, thresholds 40k (LU) / 120k (free body). Measured: LU 48 s / 2.7 GB at
  61k DOFs vs Jacobi-CG 13 s.
* **Rejected after measurement**: Voigt-form assembly (inconsistent: 3× faster small, slower large),
  ILU-preconditioned CG (no convergence in 5000 iterations), `asm(nthreads=…)` (1.4×, GIL-bound).
* **Model defaults**: DINO-base + SAM 2.1-large with CUDA, DINO-tiny + SAM 2.1-tiny on CPU
  (env vars override).
* **No automatic git.** Never commit or push unless explicitly asked in the current session
  (user instruction).
* **Memory-capped experiments**: `ulimit -v` protects WSL on CPU runs but disables CUDA, so don't
  use it for GPU timing.

## Next

* **In progress (capture modes):**
  * Done in `mos/app/main.py`: a mode selector in the multi-view panel (`MV_MODES`: `lmat4`,
    `flat3`, `video`), the backend generalised to N views, and video frames taken in the browser
    (`upload.js` `recordFrames` / `videoFrames`).
  * Done: `mvv_auto_select`, which masks the remaining frames guided by the hull of the
    accepted ones, and video builds that drop frames with agreement < 0.85.
  * Done: app tests and smoke pass for `lmat4`.
  * **Not yet done:** tests for `flat3` and `video` (synthetic floor-only rig + frame list), a
    browser check of both modes, and docs (`image-reconstruction.md`, `app.md`).
* **Material checks** (asked to look at them too, not started): stress averaging fix (element-vertex
  recovery: −41 % → +1 % on the default beam), then a yield check / safety factor per material
  preset.
0. **Benchmark on real parts (Step 0):** print the sheets, photograph 4–6 parts with caliper
   dimensions, and record dimension error, agreement and time per stage. Thresholds still untuned
   on real photos: sheet IoU 0.5, agreement flag 0.85, 70 % SAM-score rule.
1. **Short-video mode (Step 3):** many frames on the same L-mat; SAM 2 video propagation
   (`Sam2VideoModel` is installed). Synthetic: video views at 15–30° alone gave height +14 %, so
   keep a table-level frame.
2. Mask meshes exceed the 12k-tet budget (95k DOFs, 138 s solve for a 7 cm cube); check the
   estimate in `image_to_mesh.mesh_from_mask`.
1. **Decide on stress recovery** (see Decisions): switch `post`/`solver` nodal stresses to
   element-vertex evaluation + averaging. Expected effect: the default-mesh surface stress error goes
   from −41 % to about +1 %; `test_fem` stress tolerances and docs/theory.md §6 would need updating.
2. Validate the CV path on real workshop photos. The tests use synthetic images, and the "trusted
   classical mask" thresholds (backdrop spread < 30, area 1–85 %) are untuned.
3. Move the remaining UI-thread Gmsh jobs (drawing → STEP, template CAD) onto `_spawn` like `run_solve`.
4. Optional: install `pyamg` (ask first) for faster CG on large models (a 41k-DOF box takes 32 s with Jacobi-CG).
5. Optional: a *revolve* mode in `image_to_mesh` for axisymmetric parts photographed side-on
   (shafts currently extrude to a flat bar).
6. Open question: how should the repository be laid out? Commit `main/`
   as the project root and drop the deleted copies?
7. Start sessions **in `main/`**, and run the app and tests from there (`python -m mos.app --server`); from the repo
   root, CLAUDE.md and this file are not loaded.
8. Housekeeping: delete the stray files `~/=2.0`, `~/=4.45`, `~/=8` (left by an unquoted
   `pip install torch>=2.0`).
9. Browser checks: Playwright's Chromium is in `~/.cache/ms-playwright`, but WSL lacks
   libnss3/libnspr4/libasound2. Either `sudo apt-get install libnss3 libnspr4 libasound2t64` in a real
   terminal, or unpack them with `apt-get download` + `dpkg -x` and set `LD_LIBRARY_PATH` (what this
   session did). The Playwright MCP plugin expects Google Chrome at `/opt/google/chrome`.
