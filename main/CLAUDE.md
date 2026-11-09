# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MoS Analyzer: a 3D linear-elastic FEM studio (scikit-fem solver, trame + PyVista browser UI) with a
Learning Lab (stress tensors, traction, Mohr circles) and a computer-vision path (photo → parametric
STEP via Gmsh/OpenCASCADE, single-image 2.5D reconstruction, multi-view NeRF/3DGS dataset prep).

Repo layout note: this directory (`mos_work_detection_darklab_fix/`) is the current working copy and
sits inside the `mos-project` git repo. The parent also holds zipped snapshots of earlier versions
(`mos_analysis.zip`, `prototype-*.zip`, …); the previously tracked `mos_analysis/mos_work/` is an older
copy of this same project. Run all commands from this directory (pytest `pythonpath = ["."]`).


## Environment
- Always use FEM (virtual environment) from miniforge as your python Environment
- If a package that doesn't exist in said environment and is necessary happens to be missing, inform me and ask for permission before installing it.
- The current Hardware contains a Ryzen 7 CPU, Nvidia RTX5050, and 24GB of System RAM, However the environment runs on wsl which has the following limits as per is config: "memory=16GB,processors=16,swap=2GB,autoMemoryReclaim=gradual"
- If you think these limits must be lifted for you to proceed efficiently or if you find a discrepancy between your search and what is written here, inform me 

## Session status
Read docs/STATUS.md at the start of every session. It is imported here:

@docs/STATUS.md

- Don't reopen anything under **Decisions** unless new evidence contradicts the stated reason.
- Update it at the end of every work session, keeping its four sections: **Done** (finished and
  verified), **In progress** (with file names), **Decisions** (choice + why), **Next** (concrete
  steps, open questions). Refresh the date.

## Git
- Never commit or push on your own. Only do it when I explicitly ask in the current session; an
  earlier approval does not carry over.

## Reasoning
- Check reasoning and code thoroughly before presenting either. Run the code; don't just read it.
- Justify algorithmic choices mathematically. State the assumptions behind every guarantee and when it stops holding.
- Never call an approximate result exact. Report these separately:
  - whether the requested targets are feasible,
  - what was actually attained,
  - the error introduced by rounding or discretisation.
- Before adding a feature, model or heuristic, ask whether it can be measured reliably and whether it's worth optimising. If either is unclear, ask me first.

## Code
- Write compact, readable code. Skip defensive boilerplate, redundant validation, and wrappers that add nothing.
- Never compare floats with `==`. Use `isclose` with an explicit tolerance whose size you can justify.
- Never swallow exceptions (`except: pass`).
- A measurement that wasn't taken is `None`, never `0`. A failure must be visible as `None` plus a logged reason.
- No secrets in code or notebooks. Read tokens from environment variables.


## Commands

```bash
pip install -r requirements.txt            # optional extra: requirements-image3d-optional.txt (rembg)
pytest -q                                  # FEM/CAD tests skip if scikit-fem / gmsh are missing
pytest tests/test_spec.py::test_name -q    # single test
python tools/smoke_app.py                  # app wiring smoke test with fake trame/PyVista/solver
python tools/gen_api_docs.py               # regenerate docs/api.md (use --check to verify)
python -m compileall -q mos tests tools
python -m mos.app --server [--spec model.mos]
```

`smoke_app.py` intentionally prints a traceback from its invalid-input branch while still passing.
It validates controller wiring only, not real rendering, CAD generation or FEM.

## Tests that enforce documentation (easy to break)

- `tests/test_docs.py`: `docs/api.md` must equal `gen_api_docs.generate()` — rerun
  `python tools/gen_api_docs.py` after changing any public signature/docstring in the modules listed
  in `tools/gen_api_docs.py:MODULES` (add new modules there too). Every ```` ```mos ```` fence in
  `docs/*.md` and `README.md` must parse/build. Every directive in `spec._HANDLERS` and function in
  `fields._FUNCS` must appear in `docs/model-language.md`; every template in `mos.templates.TEMPLATES`
  must appear in `docs/app.md`. Relative `.md` links must resolve.
- `tests/test_docstrings.py`: every module and every public (non-underscore) class/function/method
  in `mos/` and `tools/`, including nested ones, needs a docstring.

## Architecture

Data flow: `text ─parse_spec→ Spec ─build(spec, mesh)→ (Model, LoadCase) ─solve_case→ Results → viz → app`.

- `mos/core/spec.py` — the `.mos` text language. `parse_spec` validates everything mesh-independent;
  `build` resolves regions against a mesh. Directives are `_h_<name>` handlers registered in `_HANDLERS`.
- `mos/core/fields.py` — safe `ast`-whitelist expression interpreter (no `eval`). Fields use
  scikit-fem layout: `(3, nel, nqp)` in/out.
- `mos/core/regions.py`, `loads.py`, `model.py`, `material.py` — plain data. Regions are named node or
  facet sets (never mesh splits); loads are dataclasses referencing regions. `force` = total resultant
  (N), `traction`/`pressure` = per area; `moment` is a pure couple. All loads superpose.
- `mos/core/solver.py` — the only scikit-fem-specific module (assembly, constraints, `Results`,
  `solve_case`). P2/Q2 elements by default. With no supports it does a quasi-static Newton–Euler /
  D'Alembert solve (zero angular velocity, six-mode gauge constraint) — not transient dynamics.
  DOF bookkeeping is tested with a fake basis (`tests/test_solver_logic.py`, `tests/test_post_fake_basis.py`).
- `mos/core/post.py` — strain/stress recovery, nodal averaging, von Mises, principal values.
- `mos/core/cad.py` (STEP import; `gmsh_to_meshtet` converts any Gmsh model to MeshTet with named
  facets), `cad_parametric.py` (template STEP generators), `image_to_cad.py`, `image_reconstruction.py`,
  `multiview.py` (Nerfstudio/COLMAP datasets, Gaussian PLY centers + metric scaling).
- `mos/core/gmsh_session.py` — every Gmsh caller must hold `GMSH_LOCK` and call `ensure_gmsh()`
  (initialises without the SIGINT handler; plain `gmsh.initialize()` fails off the main thread).
- CV: `cv_detector.py` segments classically first (uniform backdrop → models never load), else
  Grounding DINO + SAM (SAM 2 → SAM 1 fallback); geometry on a real mask outranks DINO labels.
  `image_to_mesh.py` meshes a mask directly (outline + holes → layered tet extrusion) and backs the
  `mesh mask PATH size thickness [h]` directive — the recommended photo → FE path.
- Linear solves: `solver.solve_spd` uses LU up to 40k DOFs, else Jacobi/AMG-CG. Large direct solves
  previously OOM-crashed WSL. `ulimit -v` caps memory for CPU-only experiments but also makes CUDA
  fail to initialise (torch silently falls back to CPU), so don't use it for GPU/CV timing.
- App: long jobs go through `_spawn(worker, name)` in `mos/app/main.py`; workers must mutate state only
  via the `ui(fn)` callback (trame state is not thread-safe). `run_solve` also meshes/solves in a
  worker (`_ui_wait` runs UI-thread steps from it) and reports through `on_done(ok)`; don't read
  `state.status` right after calling it. App CSS/JS live in `mos/app/static/` (Vue drops `<style>`
  in templates); generated files go to `cad_models/_generated/`.
- `mos/viz.py` — plotted scalars (`compute_scalars`; new quantities go here with kind
  `disp`/`stress`/`strain`), deformation scale, pyvista conversion.
- `mos/learning.py` — idealised tensor/Mohr/quick-check maths; `mos/lab/` — feature registry
  (`FEATURES`, `run_feature`) of adapters that consume solved `Results` and never call the solver.
- `mos/app/main.py` — the single large trame app (Analysis Workspace + Learning Lab + CV screens).
- `tests/fakemesh.py` — pure-NumPy tet box mesh so parser/region tests run without scikit-fem.

Optional dependencies are guarded (`HAS_GMSH`, `pytest.importorskip("skfem")`); keep imports of
gmsh/skfem/pyvista/torch lazy or guarded so pure-Python tests still run without them.

Adding a load type: dataclass in `loads.py` (validate region kind in `__post_init__`) → branch in
`solver.assemble_loads` → `_h_...` handler + `_HANDLERS` entry in `spec.py` → document in
`docs/model-language.md` → regenerate API docs.

See `docs/architecture.md`, `docs/theory.md` and `docs/testing.md` for details and known limits.
