"""Tests for CAD STEP importer, OpenCASCADE meshing, and boundary tagging."""
import os
from pathlib import Path
import numpy as np
import pytest

import mos.core.cad as cad_module
from mos.core.cad import HAS_GMSH, is_step_file, load_step_mesh, CadImportError
from mos.core.spec import parse_spec, build
try:
    from skfem import MeshTet
except ImportError:
    MeshTet = None

HAS_CAD_STACK = HAS_GMSH and MeshTet is not None

CAD_DIR = Path(__file__).resolve().parents[1] / "cad_models"


def test_is_step_file_detection():
    assert is_step_file("part.step")
    assert is_step_file("PART.STP")
    assert is_step_file(Path("model.step"))
    assert not is_step_file("mesh.msh")
    assert not is_step_file("model.vtk")


def test_spec_mesh_step_path_with_spaces():
    text = r'''
mesh step "C:\Users\test user\models\part.step" 0.25
material E = 200e9, nu = 0.3
'''
    spec = parse_spec(text)
    assert spec.mesh == ("step", r"C:\Users\test user\models\part.step", 0.25)


def test_spec_mesh_step_unquoted_path_with_spaces():
    spec = parse_spec(
        "mesh step /tmp/test CAD/part.step 0.4\n"
        "material E = 200e9, nu = 0.3\n"
    )
    assert spec.mesh == ("step", "/tmp/test CAD/part.step", 0.4)


def test_gmsh_model_is_created_before_step_import(tmp_path, monkeypatch):
    """Regression: gmsh.clear() removes models; import must create one first."""
    from types import SimpleNamespace
    import sys

    path = tmp_path / "minimal.step"
    path.write_text("placeholder")
    calls = []

    class FakeOcc:
        def importShapes(self, filename):
            calls.append(("import", filename))
            assert calls[0] == ("clear", None)
            assert calls[1][0] == "add"
            raise RuntimeError("stop after checking model lifecycle")

    class FakeModel:
        def __init__(self):
            self.occ = FakeOcc()

        def add(self, name):
            calls.append(("add", name))

    class FakeGmsh:
        def __init__(self):
            self.model = FakeModel()
            self.occ = self.model.occ
            self.option = SimpleNamespace(setNumber=lambda *args: None)

        def isInitialized(self):
            return True

        def clear(self):
            calls.append(("clear", None))

    fake_skfem = SimpleNamespace(MeshTet=object)
    monkeypatch.setitem(sys.modules, "skfem", fake_skfem)
    monkeypatch.setattr(cad_module, "HAS_GMSH", True)
    monkeypatch.setattr(cad_module, "gmsh", FakeGmsh(), raising=False)

    with pytest.raises(CadImportError, match="Failed to import or mesh CAD file"):
        cad_module.load_step_mesh(path)
    assert calls[0][0] == "clear"
    assert calls[1][0] == "add"
    assert calls[2][0] == "import"


@pytest.mark.skipif(not HAS_CAD_STACK, reason="gmsh and scikit-fem are required")
def test_load_sample_beam_step():
    beam_path = CAD_DIR / "beam.step"
    assert beam_path.is_file(), f"Missing {beam_path}"
    mesh, face_info, stats = load_step_mesh(beam_path, mesh_size=0.5)

    assert isinstance(mesh, MeshTet)
    assert mesh.nelements > 100
    assert mesh.nvertices > 50
    assert stats["n_faces"] == 6
    assert np.isclose(stats["volume"], 10.0, rtol=0.01)
    assert len(mesh.boundaries) == 6
    for tag in range(1, 7):
        assert f"cad_face_{tag}" in mesh.boundaries


@pytest.mark.skipif(not HAS_CAD_STACK, reason="gmsh and scikit-fem are required")
def test_load_sample_bracket_step():
    bracket_path = CAD_DIR / "bracket.step"
    assert bracket_path.is_file()
    mesh, face_info, stats = load_step_mesh(bracket_path, mesh_size=0.6)

    assert isinstance(mesh, MeshTet)
    assert mesh.nelements > 200
    assert stats["n_faces"] >= 8
    assert len(mesh.boundaries) >= 8
    # Centroid check
    com = np.mean(mesh.p, axis=1)
    assert 0.0 < com[0] < 8.0
    assert 0.0 < com[1] < 8.0


@pytest.mark.skipif(not HAS_CAD_STACK, reason="gmsh and scikit-fem are required")
def test_load_sample_shaft_step():
    shaft_path = CAD_DIR / "shaft.step"
    assert shaft_path.is_file()
    mesh, face_info, stats = load_step_mesh(shaft_path, mesh_size=0.5)

    assert isinstance(mesh, MeshTet)
    assert mesh.nelements > 150
    # Shaft has cylinder ends along X axis from 0 to 10
    assert np.isclose(stats["bounds"][0], 0.0, atol=1e-3)
    assert np.isclose(stats["bounds"][1], 10.0, atol=1e-3)


@pytest.mark.skipif(not HAS_CAD_STACK, reason="gmsh and scikit-fem are required")
def test_cad_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_step_mesh(CAD_DIR / "non_existent.step")


@pytest.mark.skipif(not HAS_CAD_STACK, reason="gmsh and scikit-fem are required")
def test_spec_mesh_step_directive():
    bracket_path = CAD_DIR / "bracket.step"
    text = f"""
mesh step {bracket_path} 0.6
material E = 200e9, nu = 0.3
support cad_face_1 xyz
"""
    spec = parse_spec(text)
    assert spec.mesh[0] == "step"
    model, case = build(spec)
    assert isinstance(model.mesh, MeshTet)
    assert len(model.bcs) == 1
    assert model.bcs[0].region.name == "cad_face_1"
