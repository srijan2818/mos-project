"""Tests for image to CAD STEP pipeline."""
from pathlib import Path
import numpy as np
import pytest
from PIL import Image, ImageDraw

from mos.core.cad import HAS_GMSH, load_step_mesh
from mos.core.image_to_cad import image_to_polygons, image_to_step

try:
    import skfem  # noqa: F401
    HAS_SKFEM = True
except ImportError:
    HAS_SKFEM = False


def test_image_to_polygons(tmp_path: Path):
    """Verify contour detection and RDP polygon simplification on an image."""
    img_path = tmp_path / "rect.png"
    img = Image.new("L", (200, 200), color=255)
    draw = ImageDraw.Draw(img)
    draw.rectangle([(40, 40), (160, 160)], fill=0)
    img.save(img_path)

    polys, dims = image_to_polygons(img_path, threshold=128, simplify_tol=3.0)
    assert len(polys) >= 1
    assert dims == (200, 200)
    poly = polys[0]
    assert len(poly) >= 4  # A simplified rectangle has around 4-5 vertices


@pytest.mark.skipif(not (HAS_GMSH and HAS_SKFEM), reason="gmsh and scikit-fem are required")
def test_image_to_step_and_mesh(tmp_path: Path):
    """Verify end-to-end conversion of an image sketch to a meshed 3D STEP solid."""
    img_path = tmp_path / "bracket.png"
    img = Image.new("L", (300, 300), color=255)
    draw = ImageDraw.Draw(img)
    # L-bracket shape
    draw.polygon([(40, 40), (260, 40), (260, 120), (120, 120), (120, 260), (40, 260)], fill=0)
    img.save(img_path)

    step_path = tmp_path / "bracket.step"
    out = image_to_step(img_path, step_path, thickness=0.15, scale_x=1.5, scale_y=1.5, simplify_tol=3.0)
    assert out.is_file()

    # Load and tetrahedralize through the CAD engine
    mesh, faces, stats = load_step_mesh(step_path, mesh_size=0.25)
    assert stats["n_elements"] > 0
    assert stats["n_vertices"] > 0
    assert len(faces) >= 6
