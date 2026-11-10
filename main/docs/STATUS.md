# Status

Working log for whoever (person or Claude) picks this project up next. Update it at the end of
every work session: move finished items to **Done**, record choices in **Decisions**, and leave
**Next** as concrete steps.

_Last updated: 2026-11-10_

## Done

### 3-photo solid without markers + proportional-limit check (2026-11-10)

Plan: `~/.claude/plans/look-up-fix-object-detection-multiview-3-snappy-lerdorf.md`.
`pytest -q` **241 tests: 240 passed, 1 flaky** (`test_cv_noisy_harsh_lighting_bracket`, see Next);
`tools/smoke_app.py`, `gen_api_docs.py --check` and `compileall` pass. Browser-checked with headless
Chromium against the real server (synthetic photos, screenshots in the session scratchpad).

* **Marker mat removed** (your call): `mos/core/marker.py`, `tests/test_marker.py`, the L-mat /
  flat-mat / video modes, the sheet PDF, the video helpers in `upload.js`, and the now-unused
  `exclude` / `prior` hooks of `cv_detector.analyze_image_object` (and their two SAM tests).
* **3-photo solid** (`mos/core/multiview_hull.py`, rewritten):
  * Front / top / right-side outlines, each with *Flip* / *Rotate 90°*.
  * `fit_scales`: per-photo scales from the bounding boxes, with the cycle residual reported as a
    proportion mismatch (warning above 5 %).
  * `carve_ortho`: the intersection of the three extrusions.
  * Sessions are `ortho.json` + masks, still read by `mesh hull DIR`.
  * App panel "3-photo solid": camera/file per photo, *Select part* / *Accept*, an oriented-outline
    thumbnail with axis labels, a measured dimension (axis + mm), *Build solid & solve FEA*.
  * Measured on synthetic silhouettes:
    * Box: within one voxel at any zoom, and perspective cancels for a box.
    * L-bracket volume: −0.6 %.
    * Pinhole at 5× / 10× the part size: extents ≤ 6 % / 3 %, volume +7 % / +4 %.
    * Bar cantilever: +5 % vs beam theory.
  * Browser: 60.0 × 40.3 × 30.0 mm for a 60 × 40 × 30 box, solved (17.6k cells).
* **Hull meshing bug fixed** (found in the browser): at the default element size, marching cubes
  ran with step 3–4 on the fine grid and Gmsh generated no tetrahedra. It was latent in the old mat
  flow too. Now the hull is carved at ~2 voxels per element and surfaced at step 1 with 0.5-voxel
  smoothing.
* **Stress recovery switched** (your call): each element's stress is evaluated at its own vertices
  (`post.vertex_values`), then volume-averaged (`post.vertex_average`). Default beam mid-span
  surface stress: tets −39 % → +0.8 %, hexes −50 % → exact. This applies to plots, Lab and the
  check.
* **Proportional limit:**
  * `Material.sigma_pl` / `sigma_pl_c`; spec `material …, sigma_pl = …, sigma_pl_c = …`.
  * `post.utilisation`: von Mises / σ_pl, or brittle Coulomb–Mohr max(σ1,0)/St + max(−σ3,0)/Sc.
  * `viz` field "Utilisation (σ/σ_pl)".
  * App:
    * σ_pl input (MPa, blank = off), a *Brittle* toggle with a compression limit, and presets
      for all five materials.
    * After every solve, including live-load edits, nodes with U ≥ 1 get red circles (screen-space
      discs, max 2000 drawn).
    * A dialog says the linear FEM result is invalid beyond this point and gives the load factor
      1 / max U; a legend chip and the dock warning appear too.
  * Browser: Steel default load → none; σ_pl = 1 MPa → 103 nodes, λ = 0.167 (= 1 / 5.98);
    Concrete → Coulomb–Mohr, 33 nodes, λ = 0.466.
* **Tests:**
  * New: `tests/test_limit.py`, hex beam stress, viz utilisation, 4 app limit tests,
    `tests/test_multiview_hull.py` (14 tests), the app three-photo flow.
  * Rewritten: `tests/synthetic_views.py`.
* **Docs:**
  * Updated: `theory.md` (§7 recovery; proportional-limit check with preset sources),
    `model-language.md`, `app.md`, `image-reconstruction.md` (3-photo section),
    `architecture.md`, `testing.md`.
  * `api.md` regenerated.

### UI User-Friendliness, Global Operation Cancellation & Universal Packaging (2026-11-09)

* **UI Enhancements (Strictly non-breaking UI/visual additions):**
  * **Support surfaces visualization**: All active restraints (`state.supports_list`) now render directly on the 3D model in vibrant crimson red (`#ef4444`) with bold boundary outlines (`#b91c1c`). Candidate support faces chosen in the dropdown are highlighted in warm coral (`#f97316`) before being added.
  * **Plane selection & inspection slice**: The analysis plane is enlarged to 24% of the model diagonal and rendered with high-contrast, double-sided blue shading (`#0284c7`), a smooth circular disc overlay, an emerald green normal arrow $\mathbf{n}$, and an amber traction response arrow $\mathbf{t}$. Added one-click quick preset buttons (`+X`, `-X`, `+Y`, `-Y`, `+Z`, `-Z`, `Principal 1–3`) and friendly explanations.
  * **Point selection probe**: Replaced single-point dot with a high-visibility inspection probe (core beacon + outer translucent target halo), quick snap buttons (`Center`, `Top`, `Tip`, `Root`), and click hints.
  * **Viewport controls HUD & Scene Legend**: Floating glassmorphic HUD badge guiding non-technical users on mouse navigation (rotate, zoom, pan, pick) and color-coded scene legend.
* **Global Operation Cancellation**:
  * Cooperative `S.cancel_event` wired into background workers (`run_solve`, `_run_cv`, `run_cv_universal_reconstruction`).
  * Floating top-bar progress and cancellation banner with spinning indicator and one-click "Cancel Operation" button. Resets running flags and statuses cleanly.
* **Universal Cross-Platform Packaging**:
  * `pixi.toml`: Cross-platform configuration for Windows, macOS (Intel & Apple Silicon), and Linux with pre-compiled native libraries (`scikit-fem`, `gmsh`, `pyvista`, `torch`, `transformers`).
  * `run_app.sh` (Linux/macOS) & `run_app.bat` (Windows): 1-click self-bootstrapping launchers that set up the environment and open `http://localhost:8080`.
  * `Dockerfile` & `docker-compose.yml`: Containerized deployment with pre-baked dependencies and persistent Hugging Face cache.
  * `docs/packaging.md`: Comprehensive documentation on packaging and offline model pre-caching.
* Verified: 112 passed, 1 skipped in pytest; `tools/smoke_app.py`, `gen_api_docs.py --check`, and `test_docstrings.py` all pass.

### Three-view solid on an L-shaped marker mat, Step 2 (2026-11-09, night)

_Superseded 2026-11-10: the marker mat and `marker.py` were removed in favour of the 3-photo solid._

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

* Nothing from these sessions is committed. The project folder is `main/`; the previously tracked
  `mos_work/` and the zips show as deleted in `git status` (deleted outside these sessions).
* Still synchronous on the UI thread (small or rare): drawing → STEP conversion (Gmsh), template-CAD
  generation, PLY inspect/calibrate, the live-load prepare step.

## Decisions

* **Nodal stresses = element-vertex values averaged per vertex** (your call, 2026-11-10). Measured:
  default beam surface stress +0.8 % (tets), exact (hexes); cell means gave −39 % / −50 %.
* **Proportional-limit criteria** (your call):
  * Ductile: von Mises ≥ σ_pl.
  * Brittle: Coulomb–Mohr U = max(σ1,0)/St + max(−σ3,0)/Sc ≥ 1. It covers tension, compression,
    torsion and combinations with one number. Rankine allows τ = St in torsion (unconservative);
    per-axis checks depend on the axes.
  * Both are degree-1 homogeneous, so the linear result holds up to exactly 1 / max U × the loads.
* **Preset limits are stand-ins** (published proportional limits are scarce):
  * Steel: A36 Fy 250 MPa.
  * 6061-T6: 240 MPa (0.2 % offset minimum).
  * Ti-6Al-4V annealed: 825 MPa.
  * Concrete f′c 30: St 3, Sc 12 MPa (0.1 / 0.4 f′c).
  * Gray iron class 30: St 214, Sc 752 MPa. These are fracture strengths: gray iron has no linear
    range in tension.
  * Sources are in `theory.md`.
* **Marker mat replaced by three orthogonal photos** (your call). Printing sheets was too much effort.
* **No silhouette-agreement (IoU) check for the 3-photo solid.** Measured: with three orthogonal
  views the hull reproduces a view almost whenever the other two are solid outlines. A mirrored
  side photo of a stepped block gives a different, fully consistent solid. The check is the
  bounding-box proportion mismatch (catches a hand / second object; 66 % in the test), plus the
  oriented outlines shown for checking by eye.
* **Perspective model:** a whole-photo magnification cancels, so only depth differences within a
  photo matter (~Δdepth / distance). The plan's D/(2d) estimate was wrong; a box is exact at any
  distance.
* **Hull FE mesh: carve at ~2 voxels per element, marching cubes at step 1, smoothing 0.5 voxel.**
  Steps 3–4 left Gmsh with no tetrahedra. Smoothing 1 voxel: bar +11 % too soft vs +5 %.
  Cost: rounded edges can add small extra faces.
* **Limit markers are screen-space discs drawn by `update_scene`**, not in `S.analysis_actors`: the
  probe overlay removes those on every redraw (they vanished in the first browser check).
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

1. **Real-photo benchmark of the 3-photo solid:** 4–6 parts with caliper dimensions, phone at ~5×
   and ~10× the part size. Record the dimension error, proportion mismatch and time per stage.
   Still untuned on real photos: the 5 % mismatch warning, the 70 % SAM-score rule, lens
   distortion and tilt.
2. **Flaky test:** `tests/test_cv_detector.py::test_cv_noisy_harsh_lighting_bracket` uses unseeded
   `np.random` noise. It failed once in a full run and passed alone and with the CV files. Seed it
   (`np.random.default_rng(0)`) after checking the seeded case still passes.
3. **Dependency to drop (ask first):** `opencv-contrib-python-headless` in `pixi.toml` was only
   needed by the deleted `marker.py`. No module imports `cv2` now. It is also installed in the FEM
   env.
4. Mask meshes exceed the 12k-tet budget (95k DOFs, 138 s solve for a 7 cm cube); check the
   estimate in `image_to_mesh.mesh_from_mask`. Hull meshes at the default size are ~13–18k tets.
5. Validate the single-photo CV path on real workshop photos. The "trusted classical mask"
   thresholds (backdrop spread < 30, area 1–85 %) are untuned.
6. Move the remaining UI-thread Gmsh jobs (drawing → STEP, template CAD) onto `_spawn` like
   `run_solve`.
7. Optional: install `pyamg` (ask first) for faster CG on large models (a 41k-DOF box takes 32 s
   with Jacobi-CG).
8. Optional: a *revolve* mode in `image_to_mesh` for axisymmetric parts photographed side-on
   (shafts currently extrude to a flat bar).
9. Open question: how should the repository be laid out? Commit `main/` as the project root and
   drop the deleted copies?
10. Start sessions **in `main/`**, and run the app and tests from there
    (`python -m mos.app --server`); from the repo root, CLAUDE.md and this file are not loaded.
    `main/CLAUDE.md` line 11 still names the folder `mos_work_detection_darklab_fix/`.
11. Housekeeping: delete the stray files `~/=2.0`, `~/=4.45`, `~/=8` (left by an unquoted
    `pip install torch>=2.0`).
12. **Browser checks without installing anything:** use `playwright-core` from
    `~/.npm/_npx/e41f203b7505f1fb/node_modules/playwright-core` with
    `~/.cache/ms-playwright/chromium_headless_shell-1254/.../chrome-headless-shell` as
    `executablePath`. Unpack libnss3 / libnspr4 / libasound2t64 with `apt-get download` +
    `dpkg -x` into the scratchpad and set `LD_LIBRARY_PATH`. The Playwright MCP plugin fails: it
    expects Google Chrome at `/opt/google/chrome`. The trame state is shared by all clients, so
    restart the server between scripted runs.
