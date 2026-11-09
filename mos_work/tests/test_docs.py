"""The documentation must stay true: examples run, links resolve, every feature is described."""
import pathlib
import re

from mos.core import fields, spec as spec_mod
from mos.core.spec import build, parse_spec
from mos.templates import TEMPLATES
from tests import fakemesh
from tools import gen_api_docs

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = sorted((ROOT / "docs").glob("*.md")) + [ROOT / "README.md"]
FENCE = re.compile(r"```mos\n(.*?)```", re.S)


def _text(name):
    return (ROOT / "docs" / name).read_text()


def test_every_mos_example_in_the_docs_parses_and_builds():
    n = 0
    for path in DOCS:
        for block in FENCE.findall(path.read_text()):
            spec = parse_spec(block)
            if spec.mesh[0] == "box" and spec.mesh[-1] == "tet":
                _, L, H, W, nx, ny, nz, _ = spec.mesh
                build(spec, fakemesh.box(L, H, W, nx, ny, nz))
            n += 1
    assert n >= 8, f"only {n} examples found - the fences moved?"


def test_api_reference_is_up_to_date():
    assert (ROOT / "docs" / "api.md").read_text() == gen_api_docs.generate(), \
        "docs/api.md is stale: run  python tools/gen_api_docs.py"


def test_relative_links_resolve():
    for path in DOCS:
        for target in re.findall(r"\]\(([^)#:]+\.md)(?:#[^)]*)?\)", path.read_text()):
            assert (path.parent / target).exists(), f"{path.name} links to missing {target}"


def test_every_directive_function_and_template_is_documented():
    guide = _text("model-language.md")
    for name in spec_mod._HANDLERS:
        assert f"`{name}`" in guide, f"directive '{name}' is not documented in model-language.md"
    for name in fields._FUNCS:
        assert name in guide, f"function '{name}' is not documented in model-language.md"
    app = _text("app.md")
    for name in TEMPLATES:
        assert name in app, f"template '{name}' is not mentioned in app.md"


def test_documented_dependencies_match_requirements():
    reqs = (ROOT / "requirements.txt").read_text()
    for pkg in ("scikit-fem", "meshio", "gmsh", "pyvista", "trame", "trame-vtk", "trame-vuetify"):
        assert pkg in reqs, pkg
