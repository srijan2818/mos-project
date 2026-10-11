"""Tests for CAD STEP importer, OpenCASCADE meshing, and boundary tagging."""
import os
from pathlib import Path
import numpy as np
import pytest

from mos.core.cad import HAS_GMSH, is_step_file, load_step_mesh, CadImportError
from mos.core.spec import parse_spec, build
from skfem import MeshTet

CAD_DIR = Path(__file__).resolve().parents[1] / "cad_models"


@pytest.mark.skipif(not HAS_GMSH, reason="gmsh not available")
def test_is_step_file_detection():
    assert is_step_file("part.step")
    assert is_step_file("PART.STP")
    assert is_step_file(Path("model.step"))
    assert not is_step_file("mesh.msh")
    assert not is_step_file("model.vtk")


@pytest.mark.skipif(not HAS_GMSH, reason="gmsh not available")
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


@pytest.mark.skipif(not HAS_GMSH, reason="gmsh not available")
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


@pytest.mark.skipif(not HAS_GMSH, reason="gmsh not available")
def test_load_sample_shaft_step():
    shaft_path = CAD_DIR / "shaft.step"
    assert shaft_path.is_file()
    mesh, face_info, stats = load_step_mesh(shaft_path, mesh_size=0.5)

    assert isinstance(mesh, MeshTet)
    assert mesh.nelements > 150
    # Shaft has cylinder ends along X axis from 0 to 10
    assert np.isclose(stats["bounds"][0], 0.0, atol=1e-3)
    assert np.isclose(stats["bounds"][1], 10.0, atol=1e-3)


@pytest.mark.skipif(not HAS_GMSH, reason="gmsh not available")
def test_cad_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_step_mesh(CAD_DIR / "non_existent.step")


@pytest.mark.skipif(not HAS_GMSH, reason="gmsh not available")
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
