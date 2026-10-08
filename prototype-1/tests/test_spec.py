"""Spec parsing and model building, using a numpy mesh (no scikit-fem needed)."""
import numpy as np

from mos.core.loads import BodyForce, Moment, PointForce, Pressure, Traction
from mos.core.spec import SpecError, build, parse_spec
from mos.templates import TEMPLATES
from tests import fakemesh

BASE = """\
param L = 4
mesh box L 1 1 4 2 2
material E = 1e9, nu = 0.25, rho = 1000
region fixed facets x < 1e-9
region tip facets abs(x - L) < 1e-9
support fixed xyz
"""


def _mesh(spec):
    _, L, H, W, nx, ny, nz, _ = spec.mesh
    return fakemesh.box(L, H, W, nx, ny, nz)


def _fails(text, fragment, build_it=False):
    try:
        spec = parse_spec(text)
        if build_it:
            build(spec, _mesh(spec))
    except SpecError as e:
        assert fragment in str(e), f"{fragment!r} not in {e}"
        return
    raise AssertionError(f"expected SpecError containing {fragment!r}")


def test_fake_mesh_is_a_valid_tet_mesh():
    m = fakemesh.box(4, 1, 1, 4, 2, 2)
    assert m.t.shape[1] == 6 * 4 * 2 * 2
    assert len(m.boundary_facets()) == 2 * (2 * 4 * 2 + 2 * 4 * 2 + 2 * 2 * 2)   # 2 tris per quad


def test_all_templates_parse_and_build():
    for name, text in TEMPLATES.items():
        spec = parse_spec(text)
        model, case = build(spec, _mesh(spec))
        assert case.loads and model.bcs, name


def test_region_sizes_and_load_types():
    spec = parse_spec(TEMPLATES["Cantilever - tip load"])
    model, case = build(spec, _mesh(spec))
    _, L, H, W, nx, ny, nz, _ = spec.mesh
    assert len(model.bcs[0].region) == 2 * ny * nz          # 2 triangles per end-face quad
    tr = case.loads[0]
    assert isinstance(tr, Traction) and len(tr.region) == 2 * ny * nz
    assert np.allclose(tr.field(np.zeros((3, 1)))[:, 0], [0, -1e6, 0])
    assert np.isclose(model.material.C[0, 0], 210e9 * 0.7 / (1.3 * 0.4))


def test_every_load_kind():
    text = BASE + """\
region pt nearest L, 0, 0
region top facets y > 0.5 - 1e-9
force pt 0, -10, 0
traction tip 0, where(y > 0, -1, -2), 0
pressure top step(x - L/2) * 100
body 0, -rho*9.81, 0
    moment tip 0, 0, 50 about L, 0, 0
    moment tip axis 1, 0, 0 mag 100
    moment tip mag 200 axis 0, 1, 0 about L, 0, 0
"""
    spec = parse_spec(text)
    _, case = build(spec, _mesh(spec))
    kinds = [type(l) for l in case.loads]
    assert kinds == [PointForce, Traction, Pressure, BodyForce, Moment, Moment, Moment]
    assert np.allclose(case.loads[3].field(np.zeros((3, 1)))[:, 0], [0, -9810, 0])   # rho from material
    assert np.allclose(case.loads[4].origin, [4, 0, 0])
    assert np.allclose(case.loads[5].M, [100, 0, 0])
    assert np.allclose(case.loads[6].M, [0, 200, 0])
    assert np.allclose(case.loads[6].origin, [4, 0, 0])
    assert case.loads[2].p(np.array([[1.0, 3.0], [0, 0], [0, 0]])).tolist() == [0.0, 100.0]


def test_support_components_and_values():
    spec = parse_spec(BASE + "region pt nearest L, 0, 0\nsupport pt y = -0.5\nsupport pt xz\n")
    model, _ = build(spec, _mesh(spec))
    assert model.bcs[1].components == (1,) and model.bcs[1].value == -0.5
    assert model.bcs[2].components == (0, 2) and model.bcs[2].value == 0.0


def test_error_messages():
    _fails(BASE + "bogus 1 2 3\n", "line 7: unknown directive 'bogus'")
    _fails(BASE + "traction nowhere 0, 0, 0\n", "unknown region 'nowhere'", True)
    _fails(BASE + "region mid facets abs(x - 100) < 1e-9\n", "region 'mid' is empty", True)
    _fails(BASE + "region bad facets x + 1\n", "not a condition")
    _fails(BASE + "traction tip 0, 0\n", "needs 3")
    _fails(BASE + "force tip 0, -1, 0\n", "needs a node region", True)
    _fails(BASE + "region tip facets x > 0\n", "already defined")
    _fails(BASE + "param x = 3\n", "reserved")
    _fails(BASE + "support tip y = 1 \nsupport tip = 2\n", "which components")
    _fails(BASE + "moment tip 0, 0, 1 about 1, 2\n", "needs 3")
    _fails(BASE + "moment tip axis 0, 0, 0 mag 10\n", "zero vector")
    _fails("mesh box 1 1 1 2 2 2\nmaterial E=1, nu=0.3\n", "no 'support'")
    _fails("material E=1, nu=0.3\nsupport a xyz\n", "no 'mesh'")
    _fails("mesh box 1 1 1 2.5 2 2\n", "positive integer")
    _fails("mesh box 1 1 1 2 2 2\nmaterial E=1, nu=0.7\n", "nu")
    _fails(BASE + "traction tip 0, __import__('os'), 0\n", "line 7")


def test_comments_blank_lines_and_params_in_order():
    spec = parse_spec("# c\n\nparam a = 2\nparam b = a*3   # six\n" + BASE.replace("param L = 4\n", "param L = b - 2\n"))
    assert spec.params["b"] == 6 and spec.mesh[1] == 4.0


def test_more_parse_errors():
    for text, frag in [("param 3 = x\n", "param NAME"), ("param a = x\n", "unknown name"),
                       ("mesh file\n", "mesh file PATH"), ("mesh box 1 2 3\n", "mesh box L H W"),
                       ("mesh box -1 1 1 2 2 2\n", "positive"), ("mesh sphere 1\n", "mesh box"),
                       ("material E = 1\n", "at least E and nu"), ("material G = 3, E = 1\n", "material E"),
                       ("region r cells x > 0\n", "region NAME"), ("support\n", "support REGION"),
                       ("force\n", "force REGION"), ("traction tip\n", "traction REGION"),
                       ("body 0, 0\n", "needs 3"), ("param q = 1/0\n", "line 1")]:
        _fails(text, frag)


def test_mesh_directive_variants_and_node_regions():
    s = parse_spec("mesh file C:/my models/part one.msh\nmaterial E=1, nu=0.3\nsupport a xyz\n")
    assert s.mesh == ("file", "C:/my models/part one.msh")
    s = parse_spec(BASE.replace("mesh box L 1 1 4 2 2", "mesh box L 1 1 4 2 2 hex"))
    assert s.mesh[-1] == "hex"
    spec = parse_spec(BASE + "region left nodes x < 1e-9\nsupport left y\n")
    model, _ = build(spec, _mesh(spec))
    assert model.bcs[1].region.kind == "nodes" and len(model.bcs[1].region) == 3 * 3     # 3x3 nodes at x=0


def test_missing_material_and_region_selection_failures():
    _fails("mesh box 1 1 1 2 2 2\nsupport a xyz\n", "no 'material'")
    spec = parse_spec(BASE)
    def boom(x, y, z):
        raise ValueError("boom")
    spec.regions.append((99, "bad", "nodes", boom))
    try:
        build(spec, _mesh(spec))
    except SpecError as e:
        assert "line 99" in str(e) and "boom" in str(e)
    else:
        raise AssertionError("expected SpecError")
