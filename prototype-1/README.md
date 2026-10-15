# MoS Analyzer

Linear-elastic 3D finite-element analysis for mechanics of solids: describe a body in
a short text format, solve for displacement, strain and stress, and explore the result in a
browser viewer (trame + PyVista).

Supported bodies use static linear elasticity. A model without supports is instead
analysed quasi-statically in an accelerating body-fixed frame at an instant of zero
angular velocity, using Newton–Euler acceleration and D’Alembert inertia loads. This
is not transient dynamics; see [docs/theory.md](docs/theory.md) for the assumptions.

```text
pip install -r requirements.txt
pytest                       # real FEM/CAD tests skip if scikit-fem/Gmsh are unavailable
python tools/smoke_app.py    # app logic smoke test with UI/FEM stand-ins
python -m mos.app            # add --spec model.mos to open a file
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
| [docs/testing.md](docs/testing.md) | what is tested, what is not, first-run checklist |
| [docs/api.md](docs/api.md) | API reference (generated from the source) |
