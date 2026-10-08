"""Benchmarks against beam theory. Need scikit-fem:  pip install -r requirements.txt

Cantilever, L=10, H=W=1 (L/H = 10), E=210 GPa, nu=0.3, coarse 20x2x2 P2 mesh.
Tolerances are first guesses: if one fails by a few percent, refine the mesh before
suspecting the code; if it fails by tens of percent, suspect the code.
"""
import numpy as np
import pytest

pytest.importorskip("skfem")

from mos.core import (BC, BodyForce, LoadCase, Material, Model, Moment, PointForce,
                      Traction, facets_where, nearest_node, nodes_where)
from mos.core.meshes import box_mesh
from mos.core.solver import solve_case

L, H, W = 10.0, 1.0, 1.0
E, NU, RHO, G = 210e9, 0.3, 7850.0, 9.81
I = W * H**3 / 12


@pytest.fixture(scope="module")
def beam():
    mesh = box_mesh(L, H, W, 20, 2, 2)
    fixed = facets_where(mesh, lambda x, y, z: np.isclose(x, 0), "fixed")
    tip = facets_where(mesh, lambda x, y, z: np.isclose(x, L), "tip")
    model = Model(mesh, Material.isotropic(E, NU, RHO), [BC(fixed)])
    tip_nodes = nodes_where(mesh, lambda x, y, z: np.isclose(x, L), "tip_nodes")
    return model, tip, tip_nodes


def test_tip_shear_load(beam):
    model, tip, tip_nodes = beam
    P = 1e6
    res = solve_case(model, LoadCase("tip", [Traction(tip, [0, -P / (H * W), 0])]))
    delta = -res.u_nodal[1, tip_nodes.ids].mean()
    expected = P * L**3 / (3 * E * I)                  # Euler-Bernoulli (shear adds ~0.8 %)
    assert delta == pytest.approx(expected, rel=0.05)
    assert res.reactions[1] == pytest.approx(P, rel=1e-8)
    assert res.equilibrium_error < 1e-8


def test_tip_moment(beam):
    model, tip, tip_nodes = beam
    M = 1e6
    res = solve_case(model, LoadCase("moment", [Moment(tip, [0, 0, M])]))
    delta = res.u_nodal[1, tip_nodes.ids].mean()
    assert delta == pytest.approx(M * L**2 / (2 * E * I), rel=0.05)
    assert np.abs(res.reactions).max() < 1e-6 * M / H   # pure moment: zero net force


def test_self_weight(beam):
    model, tip, tip_nodes = beam
    res = solve_case(model, LoadCase("gravity", [BodyForce([0, -RHO * G, 0])]))
    q = RHO * G * W * H
    delta = -res.u_nodal[1, tip_nodes.ids].mean()
    assert delta == pytest.approx(q * L**4 / (8 * E * I), rel=0.05)
    assert res.reactions[1] == pytest.approx(q * L, rel=1e-8)


def test_point_force_equilibrium(beam):
    model, _, _ = beam
    node = nearest_node(model.mesh, (L, 0, 0), "tip_center")
    res = solve_case(model, LoadCase("point", [PointForce(node, [0, -1e5, 0])]))
    assert res.reactions[1] == pytest.approx(1e5, rel=1e-8)
    assert res.equilibrium_error < 1e-8


def test_bending_stress_at_midspan(beam):
    model, tip, _ = beam
    P = 1e6
    res = solve_case(model, LoadCase("tip", [Traction(tip, [0, -P / (H * W), 0])]))
    top_mid = nodes_where(model.mesh, lambda x, y, z: np.isclose(x, L / 2) & np.isclose(y, H / 2), "m")
    sigma = res.sig_nodal[0, 0, top_mid.ids].mean()
    # Nodal averaging from cell centroids on a coarse 2-element-high mesh underestimates surface peak
    assert sigma == pytest.approx(P * (L / 2) * (H / 2) / I, rel=0.40)   # M y / I


# ---- through the text-spec path (what the app does) -------------------------------------
from mos.core.spec import build, parse_spec
from mos.templates import TEMPLATES


def test_spec_cantilever_template_end_to_end():
    model, case = build(parse_spec(TEMPLATES["Cantilever - tip load"]))
    res = solve_case(model, case)
    tip = nodes_where(model.mesh, lambda x, y, z: np.isclose(x, 10.0), "t")
    delta = -res.u_nodal[1, tip.ids].mean()
    assert delta == pytest.approx(1e6 * 10**3 / (3 * 210e9 * (1 / 12)), rel=0.05)


def test_spec_torsion_template_twist():
    model, case = build(parse_spec(TEMPLATES["Bar - torsion"]))
    res = solve_case(model, case)
    node = nearest_node(model.mesh, (10.0, 0.0, 0.5), "corner_mid")
    theta = -2 * res.u_nodal[1, node.ids[0]]            # u_y = -theta * z, z = a/2
    G = 210e9 / (2 * 1.3)
    assert theta == pytest.approx(1e5 * 10 / (G * 0.1406), rel=0.10)   # J = 0.1406 a^4


def test_spec_simply_supported_template():
    model, case = build(parse_spec(TEMPLATES["Simply supported - center load"]))
    res = solve_case(model, case)
    mid = np.isclose(model.mesh.p[0], 5.0) & np.isclose(model.mesh.p[1], -0.5)
    delta = -res.u_nodal[1, mid].mean()
    expected = 1e6 * 10**3 / (48 * 210e9 * (1 / 12))   # P L^3 / (48 E I)
    assert delta == pytest.approx(expected, rel=0.08)
    assert res.equilibrium_error < 1e-8


def test_spec_pure_moment_template():
    model, case = build(parse_spec(TEMPLATES["Cantilever - pure end moment"]))
    res = solve_case(model, case)
    tip = nodes_where(model.mesh, lambda x, y, z: np.isclose(x, 10.0), "t")
    delta = res.u_nodal[1, tip.ids].mean()
    expected = 1e6 * 10**2 / (2 * 210e9 * (1 / 12))    # M L^2 / (2 E I)
    assert delta == pytest.approx(expected, rel=0.05)
    assert np.abs(res.reactions).max() < 1e-6 * 1e6 / 1.0


def test_spec_axial_tension_template():
    model, case = build(parse_spec(TEMPLATES["Bar - axial tension"]))
    res = solve_case(model, case)
    tip = nodes_where(model.mesh, lambda x, y, z: np.isclose(x, 10.0), "t")
    elongation = res.u_nodal[0, tip.ids].mean()
    expected = 2.1e7 * 10.0 / (210e9 * 1.0 * 1.0)      # F L / (E A)
    assert elongation == pytest.approx(expected, rel=0.02)
    assert res.equilibrium_error < 1e-8


def test_spec_fixed_fixed_template():
    model, case = build(parse_spec(TEMPLATES["Fixed-fixed beam - uniform load"]))
    res = solve_case(model, case)
    mid = np.isclose(model.mesh.p[0], 5.0) & np.isclose(model.mesh.p[1], 0.0)
    delta = -res.u_nodal[1, mid].mean()
    q = 1e6 * 1.0
    expected = q * 10**4 / (384 * 210e9 * (1 / 12))    # q L^4 / (384 E I)
    assert delta == pytest.approx(expected, rel=0.15)
    assert res.equilibrium_error < 1e-6


def test_spec_point_force_and_custom_moment():
    model, case = build(parse_spec(TEMPLATES["Cantilever - point force and custom axis moment"]))
    res = solve_case(model, case)
    assert res.equilibrium_error < 1e-6
    assert res.reactions[1] == pytest.approx(5e5, rel=1e-8)

