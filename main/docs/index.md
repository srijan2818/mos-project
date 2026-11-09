# MoS Analyzer: documentation

Linear-elastic 3D finite-element analysis for mechanics of solids. You describe a
body (mesh, material, supports, loads) in a short text format, the solver computes
displacement, strain and stress, and a browser viewer shows the result. Supported bodies use static
analysis; support-free bodies use a quasi-static accelerating body-fixed frame at an instant of zero
angular velocity, not transient dynamics.

## Where to start

| I want to... | Read |
|---|---|
| install it and run the app | [Quick start](#quick-start) below, then [app.md](app.md) |
| write or edit a model | [model-language.md](model-language.md) |
| understand the maths and the sign conventions | [theory.md](theory.md) |
| change or extend the code | [architecture.md](architecture.md), then [api.md](api.md) |
| know what is verified and what is not | [testing.md](testing.md) |
| inspect solved point stresses and plane tractions | [learning-lab.md](learning-lab.md) |
| turn one image into a colored 3D preview and optional STEP silhouette | [image-reconstruction.md](image-reconstruction.md) |
| turn multiple object photos into NeRF / Gaussian-splat datasets | [app.md](app.md#multi-view-nerf--gaussian-splatting-and-gaussian-to-cad-preparation) |

## Quick start

```text
pip install -r requirements.txt
pytest                       # unit tests; real FEM/CAD tests skip if scikit-fem/Gmsh are unavailable
python -m mos.app --server   # starts without opening a browser; add --spec model.mos to start from a file
```

A model is plain text:

```mos
param L = 10
mesh     box L 1 1 20 2 2
material E = 210e9, nu = 0.3, rho = 7850
region   fixed facets x < 1e-9
region   tip   facets abs(x - L) < 1e-9
support  fixed xyz
traction tip 0, -1e6, 0
```

This is a 10 m cantilever with a 1 m x 1 m section, clamped at x = 0 and loaded by a uniform
shear traction at x = L.

## How the pieces fit

```text
text ──parse_spec──▶ Spec ──build(mesh)──▶ Model + LoadCase ──solve_case──▶ Results ──viz──▶ 3D view
        (spec.py)            (spec.py)                          (solver.py)    (viz.py, app/)
```

* **Spec**: the parsed text. Nothing is resolved against a mesh yet.
* **Model**: a mesh, one material and boundary conditions. **LoadCase**: a list of loads.
* **Results**: displacements, strain and stress tensors, reactions and an equilibrium check.

You can skip the text format and build `Model` / `LoadCase` objects directly in Python; see
[architecture.md](architecture.md).

## Conventions (used everywhere)

* **Units**: whatever consistent set you choose; SI (m, N, Pa, kg/m^3) in all examples. The app's
  stress-unit menu changes the display only.
* **Axes**: right-handed x, y, z. A generated `box` runs along x: x in [0, L],
  y in [-H/2, H/2], z in [-W/2, W/2].
* **Arrays**: component axis first. Positions `(3, ...)`, vectors `(3, ...)`, tensors `(3, 3, ...)`.
* **Stress sign**: tension positive. A pressure pushes *into* the body.
* **Shear strain**: tensor components `eps_xy` in the solver's tensors; the viewer's `gamma`
  fields are engineering shear strains (2 eps_xy).

## Status and limits

Linear-elastic, small-strain, 3D only; one isotropic material per model in the text
format (the Python model accepts any symmetric positive-definite 6x6 `C`); tetrahedral or
hexahedral meshes. Supported bodies use static analysis. Support-free bodies use a quasi-static,
body-fixed acceleration correction at an instant of zero spin—not transient dynamics. There is no
plane stress/strain, contact, plasticity, or thermal load.
Click-to-pick regions in the viewer are not implemented: regions come from coordinate
conditions or a nearest-node lookup. See [testing.md](testing.md) for exactly which parts have
been run against the real libraries.
