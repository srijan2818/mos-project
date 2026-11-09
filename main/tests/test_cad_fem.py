"""Finite element solving benchmarks on CAD STEP geometries."""
from pathlib import Path
import numpy as np
import pytest

pytest.importorskip("skfem")

from mos.core.cad import HAS_GMSH
from mos.core.spec import parse_spec, build
from mos.core.solver import solve_case
from mos import viz

CAD_DIR = Path(__file__).resolve().parents[1] / "cad_models"


@pytest.mark.skipif(not HAS_GMSH, reason="gmsh not available")
def test_cad_beam_analytical_benchmark():
    """Verify that a beam imported from STEP matches Euler-Bernoulli analytical tip deflection."""
    beam_path = CAD_DIR / "beam.step"
    L, H, W = 10.0, 1.0, 1.0
    E, nu = 210e9, 0.3
    P = 1e6
    traction = P / (H * W)

    text = f"""
mesh step {beam_path} 0.5
material E = {E}, nu = {nu}
region fixed facets x < 1e-5
region tip   facets x > {L} - 1e-5
support fixed xyz
traction tip 0, -{traction}, 0
"""
    spec = parse_spec(text)
    model, case = build(spec)
    res = solve_case(model, case)

    # Analytical Euler-Bernoulli deflection: v = P * L^3 / (3 * E * I)
    I = (W * H**3) / 12.0
    v_analytical = P * (L**3) / (3.0 * E * I)
    v_fem = float(np.max(np.abs(res.u_nodal[1])))

    # With quadratic tets (P2 basis), deflection should match within ~0.5%
    rel_diff = abs(v_fem - v_analytical) / v_analytical
    assert rel_diff < 0.01, f"FEM tip deflection {v_fem:.5g} m differs from analytical {v_analytical:.5g} m by {rel_diff*100:.2f}%"

    # Static reaction in Y should balance applied shear load P = 1e6 N
    assert np.isclose(res.reactions[1], P, rtol=1e-3)
    assert res.equilibrium_error < 1e-9


@pytest.mark.skipif(not HAS_GMSH, reason="gmsh not available")
def test_cad_shaft_pure_torsion():
    """Verify pure torque applied to a CAD stepped shaft produces zero net reaction force."""
    shaft_path = CAD_DIR / "shaft.step"
    T = 1000.0  # N*m

    text = f"""
mesh step {shaft_path} 0.4
material E = 210e9, nu = 0.3
support cad_face_3 xyz
moment cad_face_5 {T}, 0, 0
"""
    spec = parse_spec(text)
    model, case = build(spec)
    res = solve_case(model, case)

    # Pure moment should produce near-zero net reaction force
    max_reaction_force = float(np.max(np.abs(res.reactions)))
    assert max_reaction_force < 1e-3, f"Unexpected net reaction force under pure moment: {res.reactions}"
    assert res.equilibrium_error < 1e-9

    scalars = viz.compute_scalars(res)
    assert "von Mises" in scalars
    assert float(scalars["von Mises"][0].max()) > 0.0


@pytest.mark.skipif(not HAS_GMSH, reason="gmsh not available")
def test_cad_bracket_solve():
    """Verify solving the L-bracket with mounting supports and tip traction."""
    bracket_path = CAD_DIR / "bracket.step"
    text = f"""
mesh step {bracket_path} 0.6
material E = 200e9, nu = 0.3, rho = 7850
support cad_face_1 xyz
traction cad_face_6 0, -1e6, 0
"""
    spec = parse_spec(text)
    model, case = build(spec)
    res = solve_case(model, case)

    assert res.u.size > 2000
    assert res.equilibrium_error < 1e-8
    assert float(np.max(np.abs(res.u_nodal))) > 0.0


@pytest.mark.skipif(not HAS_GMSH, reason="gmsh not available")
def test_shaft_template_clamps_root_end_and_twists_tip_end():
    """The template comment says root clamped, torque at the tip: check the faces really are the ends."""
    from mos.templates import TEMPLATES
    text = TEMPLATES["CAD - Stepped Shaft Torsion (STEP)"].replace(
        "cad_models/shaft.step", str(CAD_DIR / "shaft.step"))
    model, case = build(parse_spec(text))
    mesh = model.mesh
    centroid_x = lambda region: float(mesh.p[0, mesh.facets[:, region.ids]].mean())
    assert np.isclose(centroid_x(model.bcs[0].region), 0.0, atol=1e-6)
    assert np.isclose(centroid_x(case.loads[0].region), float(mesh.p[0].max()), atol=1e-6)
