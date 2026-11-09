import re

import numpy as np

from mos.learning import (
    analyse_stress_tensor,
    mechanics_quick_check,
    mohr_diagram_svg,
    render_tensor_matrix,
    tensor_diagram_svg,
)


def test_uniaxial_stress_principal_values_and_traction():
    result = analyse_stress_tensor(100, 0, 0, 0, 0, 0, 0, 0)
    assert np.allclose(result["principal"], [100, 0, 0])
    assert np.isclose(result["normal_stress"], 100)
    assert np.isclose(result["shear_stress"], 0)
    assert np.isclose(result["von_mises"], 100)


def test_pure_shear_principal_values_and_maximum_shear():
    result = analyse_stress_tensor(0, 0, 0, 60, 0, 0, 0, 0)
    assert np.allclose(result["principal"], [60, 0, -60])
    assert np.isclose(result["max_shear"], 60)
    assert np.isclose(result["von_mises"], np.sqrt(3) * 60)


def test_hydrostatic_stress_has_no_shear():
    result = analyse_stress_tensor(50, 50, 50, 0, 0, 0, 32, -17)
    assert np.allclose(result["principal"], [50, 50, 50])
    assert np.isclose(result["shear_stress"], 0)
    assert np.isclose(result["von_mises"], 0)


def test_learning_visuals_are_well_formed_svg_or_html():
    result = analyse_stress_tensor(80, 20, 10, 30, 12, -8, 35, 20)
    assert tensor_diagram_svg(result).startswith("<svg")
    mohr_svg = mohr_diagram_svg(result["principal"])
    assert mohr_svg.startswith("<svg")
    circle_tags = re.findall(r"<circle[^>]*>", mohr_svg)
    assert len(circle_tags) == 6  # three Mohr circles and three principal-stress points
    assert all(' r="' in tag and ' rx="' not in tag and ' ry="' not in tag for tag in circle_tags)
    matrix_html = render_tensor_matrix(result)
    assert matrix_html.count("<tr>") == 3
    assert matrix_html.count("<td") == 9


def test_mechanics_quick_check_formulas():
    beam = mechanics_quick_check("beam", 500, 40, 80, 180)
    assert beam["primary_value"] == "11.72 MPa"
    shaft = mechanics_quick_check("shaft", 150, 30, 500, 79)
    assert shaft["primary_value"] == "28.29 MPa"
    vessel = mechanics_quick_check("vessel", 2, 500, 8, 150)
    assert vessel["primary_value"] == "62.50 MPa"
