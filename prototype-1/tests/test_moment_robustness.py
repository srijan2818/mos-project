"""Tests verifying robustness of moment and load applications."""
import numpy as np
from mos.core.spec import parse_spec, build
from mos.core.solver import solve_case


def test_moment_on_nearest_node_region_succeeds_without_error():
    """Verify that applying a moment on a nearest-node region adapts to an adjacent boundary patch."""
    text = """
param L = 10
mesh box L 1 1 10 2 2
material E = 210e9, nu = 0.3
region fixed facets x < 1e-9
region pt_applied nearest L, 0, 0
support fixed xyz
moment pt_applied 0, 0, 1e5
"""
    spec = parse_spec(text)
    model, case = build(spec)
    res = solve_case(model, case)

    assert res.equilibrium_error < 1e-8
    assert float(np.max(np.abs(res.u_nodal))) > 0.0


def test_moment_on_face_with_custom_pivot_point():
    """Verify applying a moment about an explicit pivot point."""
    text = """
param L = 10
mesh box L 1 1 10 2 2
material E = 210e9, nu = 0.3
region fixed facets x < 1e-9
region tip   facets x > L - 1e-9
support fixed xyz
moment tip 0, 0, 1e5 about L, 0.5, 0
"""
    spec = parse_spec(text)
    model, case = build(spec)
    res = solve_case(model, case)

    assert res.equilibrium_error < 1e-8
    assert float(np.max(np.abs(res.u_nodal))) > 0.0
