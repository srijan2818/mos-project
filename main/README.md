# MoS Analyzer

A solid-mechanics studio with a 3D linear-elastic finite-element Analysis Workspace and a
dedicated interactive Learning Lab. Build or import a body, apply supports and loads, solve for
displacement/strain/stress, and inspect the result in a browser viewer (trame + PyVista). The
computer-vision workspace supports single-image parametric CAD, a universal single-image
2.5D reconstruction preview with optional Depth Anything V2, multi-view NeRF / 2DGS / 3DGS
dataset preparation (Nerfstudio, COLMAP, Graphdeco), and Gaussian PLY point-cloud extraction /
metric calibration. In the
Learning Lab, explore live 3D stress tensors, arbitrary-plane traction, principal planes, 3D Mohr
circles, and compact beam/shaft/pressure-vessel teaching checks.

Supported bodies use static linear elasticity. A model without supports is instead
analysed quasi-statically in an accelerating body-fixed frame at an instant of zero
angular velocity, using Newton–Euler acceleration and D’Alembert inertia loads. This
is not transient dynamics; see [docs/theory.md](docs/theory.md) for the assumptions.

<img width="1200" height="675" alt="Truss Deflection" src="https://github.com/user-attachments/assets/eec55afe-c551-448a-8058-0753dab0fa04" />

<img width="1200" height="675" alt="Cantilever Beam Deflection" src="https://github.com/user-attachments/assets/667d16a5-7ddb-4b82-995d-d331968b8251" />

```text
pip install -r requirements.txt
pytest                       # real FEM/CAD tests skip if scikit-fem/Gmsh are unavailable
python tools/smoke_app.py    # app logic smoke test with UI/FEM stand-ins
python -m mos.app --server   # add --spec model.mos to open a file without launching a browser
```

```mos
param L = 10
mesh     box L 1 1 20 2 2
material E = 210e9, nu = 0.3, rho = 7850
region   fixed facets x < 1e-9
region   tip   facets abs(x - L) < 1e-9
support  fixed xyz
traction tip 0, -1e6, 0
body     0, -rho*9.81, 0
```

## Documentation

| | |
|---|---|
| [docs/index.md](docs/index.md) | overview, conventions, status |
| [docs/model-language.md](docs/model-language.md) | the text format: every directive, expressions, examples, errors |
| [docs/theory.md](docs/theory.md) | the maths: formulation, moments/torsion, reactions, stress recovery, benchmark formulas |
| [docs/architecture.md](docs/architecture.md) | modules, data flow, design decisions, how to extend |
| [docs/app.md](docs/app.md) | using the app, troubleshooting |
| [docs/image-reconstruction.md](docs/image-reconstruction.md) | single-image reconstruction, masking, depth relief, and CAD export limits |
| [docs/learning-lab.md](docs/learning-lab.md) | stress tensor explorer, principal planes, Mohr circles and quick checks |
| [docs/testing.md](docs/testing.md) | what is tested, what is not, first-run checklist |
| [docs/packaging.md](docs/packaging.md) | packaging and 1-click deployment on Linux, macOS, and Windows |
| [docs/api.md](docs/api.md) | API reference (generated from the source) |
