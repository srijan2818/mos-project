"""Unit and integration tests for CV mode object detection and parametric CAD generation."""
from pathlib import Path
import numpy as np
import pytest
from PIL import Image, ImageDraw

# The module also contains solver-backed CV→CAD integration cases.
pytest.importorskip("skfem")

from mos.core.cad import HAS_GMSH, load_step_mesh
from mos.core.cad_parametric import (
    generate_bracket_step,
    generate_plate_hole_step,
    generate_beam_step,
    generate_ibeam_step,
    generate_shaft_step,
    generate_pipe_step,
    generate_cad_from_parameters,
)
from mos.core.cv_detector import analyze_image_object, cv_image_to_cad_step
from mos.core.model import Model, Material, LoadCase
from mos.core.loads import SurfaceForce
from mos.core.solver import solve_case
from mos.core.regions import cad_boundary_name_for_side


def test_cv_detection_bracket(tmp_path: Path):
    """Verify CV model determines an L-bracket and extracts parameters."""
    img_p = tmp_path / "bracket.png"
    img = Image.new("RGB", (400, 400), color=(230, 230, 230))
    d = ImageDraw.Draw(img)
    d.polygon([(60, 60), (340, 60), (340, 140), (140, 140), (140, 340), (60, 340)], fill=(40, 45, 55))
    d.ellipse([(220, 85), (260, 125)], fill=(230, 230, 230))
    img.save(img_p)

    res = analyze_image_object(img_p, reference_scale=0.25)
    assert res.object_type == "bracket"
    assert res.confidence >= 0.70
    assert "leg_x" in res.parameters
    assert "leg_y" in res.parameters
    assert "thickness" in res.parameters
    assert res.annotated_image_b64.startswith("data:image/png;base64,")


def test_cv_noisy_harsh_lighting_bracket(tmp_path: Path):
    """Verify the deep model correctly determines an object despite severe lighting gradients, shadows, and noise."""
    w, h = 400, 400
    y_grad, x_grad = np.mgrid[0:h, 0:w]
    # Harsh directional lighting gradient (dark bottom right)
    illum = 180 * (1.0 - (x_grad + y_grad) / (w + h))
    # Workbench texture + noise
    bg_noise = 20 * np.sin(x_grad / 10.0) * np.cos(y_grad / 15.0) + np.random.normal(0, 10.0, (h, w))
    bg = np.clip(illum + bg_noise + 50, 15, 240)

    # L-bracket shape
    mask = np.zeros((h, w), dtype=bool)
    mask[80:320, 80:160] = True
    mask[240:320, 80:320] = True
    hole_mask = ((x_grad - 240)**2 + (y_grad - 280)**2) < 20**2
    mask[hole_mask] = False

    obj = np.full((h, w), 70.0) + 0.3 * illum
    combined = np.where(mask, obj, bg)
    noisy_arr = np.clip(combined + np.random.normal(0, 6.0, (h, w)), 0, 255).astype(np.uint8)
    noisy_img = Image.fromarray(noisy_arr).convert("RGB")
    noisy_path = tmp_path / "harsh_bracket.png"
    noisy_img.save(noisy_path)

    res = analyze_image_object(noisy_path, reference_scale=0.25)
    assert res.object_type == "bracket", f"Expected bracket, got {res.object_type} (scores: {res.all_scores})"
    assert res.confidence >= 0.60
    assert res.parameters["leg_x"] > 0


def test_cv_detection_plate_hole(tmp_path: Path):
    """Verify CV model determines a tensile plate with a circular through-hole."""
    img_p = tmp_path / "plate.png"
    img = Image.new("RGB", (500, 300), color=(240, 240, 240))
    d = ImageDraw.Draw(img)
    d.rectangle([(50, 50), (450, 250)], fill=(50, 60, 70))
    d.ellipse([(220, 120), (280, 180)], fill=(240, 240, 240))
    img.save(img_p)

    res = analyze_image_object(img_p, reference_scale=0.40)
    assert res.object_type == "plate_hole"
    assert res.confidence >= 0.75
    assert "hole_diameter" in res.parameters
    assert res.parameters["hole_diameter"] > 0


def test_cv_detection_beam(tmp_path: Path):
    """Verify CV model determines an elongated rectangular solid beam."""
    img_p = tmp_path / "beam.png"
    img = Image.new("RGB", (600, 200), color=(240, 240, 240))
    d = ImageDraw.Draw(img)
    d.rectangle([(50, 60), (550, 140)], fill=(45, 55, 65))
    img.save(img_p)

    res = analyze_image_object(img_p, reference_scale=1.0)
    assert res.object_type == "beam"
    assert res.confidence >= 0.75
    assert res.parameters["length"] == pytest.approx(1.0, rel=0.1)


def test_cv_detection_ibeam(tmp_path: Path):
    """Verify CV model determines a doubly-symmetric flanged I-beam."""
    img_p = tmp_path / "ibeam.png"
    img = Image.new("RGB", (400, 400), color=(240, 240, 240))
    d = ImageDraw.Draw(img)
    d.rectangle([(100, 60), (300, 110)], fill=(45, 55, 65))
    d.rectangle([(175, 110), (225, 290)], fill=(45, 55, 65))
    d.rectangle([(100, 290), (300, 340)], fill=(45, 55, 65))
    img.save(img_p)

    res = analyze_image_object(img_p, reference_scale=0.30)
    assert res.object_type == "ibeam"
    assert "flange_width" in res.parameters
    assert "web_thickness" in res.parameters


@pytest.mark.skipif(not HAS_GMSH, reason="Gmsh is required for 3D CAD STEP solid generation")
def test_parametric_cad_solids(tmp_path: Path):
    """Verify all parametric shapes generate valid STEP solids that load and mesh."""
    # 1. Bracket
    b_step = generate_bracket_step(tmp_path / "b.step", leg_x=0.2, leg_y=0.2, width=0.05, thickness=0.015)
    m, faces, stats = load_step_mesh(b_step, mesh_size=0.04)
    assert stats["n_elements"] > 0
    assert len(faces) >= 6

    # 2. Plate with hole
    p_step = generate_plate_hole_step(tmp_path / "p.step", length=0.3, width=0.15, thickness=0.02, hole_diameter=0.04)
    m, faces, stats = load_step_mesh(p_step, mesh_size=0.04)
    assert stats["n_elements"] > 0
    assert len(faces) >= 7

    # 3. Beam
    bm_step = generate_beam_step(tmp_path / "bm.step", length=0.5, height=0.1, width=0.08)
    m, faces, stats = load_step_mesh(bm_step, mesh_size=0.04)
    assert stats["n_elements"] > 0
    assert len(faces) == 6

    # 4. I-Beam
    ib_step = generate_ibeam_step(tmp_path / "ib.step", length=0.4, height=0.15, flange_width=0.1, flange_thickness=0.02, web_thickness=0.015)
    m, faces, stats = load_step_mesh(ib_step, mesh_size=0.04)
    assert stats["n_elements"] > 0
    assert len(faces) >= 12

    # 5. Pipe
    pip_step = generate_pipe_step(tmp_path / "pip.step", length=0.3, outer_diameter=0.08, inner_diameter=0.06)
    m, faces, stats = load_step_mesh(pip_step, mesh_size=0.03)
    assert stats["n_elements"] > 0
    assert len(faces) >= 4


@pytest.mark.skipif(not HAS_GMSH, reason="Gmsh is required for end-to-end FEA solve")
def test_end_to_end_cv_to_cad_fem_solve(tmp_path: Path):
    """End-to-end pipeline: Image -> CV Parameterization -> 3D STEP -> Meshing -> FEA Solve."""
    # Image
    img_p = tmp_path / "photo.png"
    img = Image.new("RGB", (400, 400), color=(240, 240, 240))
    d = ImageDraw.Draw(img)
    d.polygon([(60, 60), (340, 60), (340, 140), (140, 140), (140, 340), (60, 340)], fill=(40, 45, 55))
    img.save(img_p)

    step_p = tmp_path / "model.step"
    out_step, cv_res = cv_image_to_cad_step(img_p, step_p, reference_scale=0.20)
    assert out_step.is_file()

    # Load into existing CAD modules
    mesh, faces, stats = load_step_mesh(out_step, mesh_size=0.04)
    assert stats["n_elements"] > 0

    # Resolve CAD boundary faces for supports and loads
    root_face = cad_boundary_name_for_side(mesh, "-x")
    tip_face = cad_boundary_name_for_side(mesh, "+x")
    assert root_face in mesh.boundaries
    assert tip_face in mesh.boundaries

    # Setup linear-elastic FEA model
    from mos.core import BC
    from mos.core.regions import Region
    mat = Material.isotropic(210e9, 0.3, 7850)
    bcs = [BC(Region("root", "facets", mesh.boundaries[root_face]), (0, 1, 2))]
    loads = [SurfaceForce(Region("tip", "facets", mesh.boundaries[tip_face]), [0, -1000, 0])]
    model = Model(mesh, mat, bcs)

    # Solve
    res = solve_case(model, LoadCase("cv_cad_case", loads))
    assert res.u_nodal is not None
    assert np.max(np.abs(res.u_nodal)) > 0
    assert np.allclose(res.reactions[1], 1000.0, rtol=1e-3)
