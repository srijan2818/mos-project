"""Automatic coarsening of free-body models: DOF count, mesh-line rewriting."""
import numpy as np
import pytest

from mos.core.spec import SpecError, coarsen_mesh_text, parse_spec
from tests import fakemesh

MAT = "material E = 1e9, nu = 0.3\n"


@pytest.mark.parametrize("kind", ["tet", "hex"])
def test_dof_count_matches_the_basis(kind):
    pytest.importorskip("skfem")
    from mos.core.meshes import box_mesh
    from mos.core.solver import count_dofs, make_basis
    for n in ((5, 4, 3), (7, 2, 9)):
        m = box_mesh(3, 2, 1, *n, kind)
        assert count_dofs(m) == make_basis(m)[0].N


def test_dof_count_needs_no_scikit_fem_for_tets():
    from mos.core.solver import count_dofs
    m = fakemesh.box(2, 1, 1, 2, 1, 1)
    edges = {tuple(sorted((int(t[a]), int(t[b])))) for t in m.t.T
             for a, b in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]}
    assert count_dofs(m) == 3 * (m.p.shape[1] + len(edges))


def test_box_counts_are_divided_and_the_comment_kept():
    text = "param L = 10\nmesh box L 1 1 20 4 4 hex   # beam\n" + MAT
    out = coarsen_mesh_text(text, 2.0)
    assert parse_spec(out).mesh == ("box", 10.0, 1.0, 1.0, 10, 2, 2, "hex")
    assert "# beam" in out and out.endswith("\n")
    assert parse_spec(coarsen_mesh_text(text, 50.0)).mesh[4:7] == (1, 1, 1)     # never below 1


@pytest.mark.parametrize("line, expected", [
    ('mesh step "/a b/part.step" 0.002', ("step", "/a b/part.step", 0.003)),
    ('mesh hull "/s/1" 0.004', ("hull", "/s/1", 0.006)),
    ('mesh mask "m.png" 0.1 0.01 0.003', ("mask", "m.png", 0.1, 0.01, 0.0045)),
])
def test_mesh_size_is_scaled(line, expected):
    got = parse_spec(coarsen_mesh_text(line + "\n" + MAT, 1.5)).mesh
    assert got[:-1] == expected[:-1] and got[-1] == pytest.approx(expected[-1], rel=1e-3)


def test_missing_size_starts_from_the_current_mesh():
    mesh = fakemesh.box(2, 1, 1, 4, 2, 2)
    from mos.core.meshes import mean_edge_length
    h = mean_edge_length(mesh)
    got = parse_spec(coarsen_mesh_text('mesh step "p.step"\n' + MAT, 2.0, mesh)).mesh
    assert got[2] == pytest.approx(2.0 * h, rel=1e-3)
    with pytest.raises(ValueError):
        coarsen_mesh_text('mesh step "p.step"\n' + MAT, 2.0)


def test_file_meshes_and_bad_factors_are_refused():
    with pytest.raises(SpecError):
        coarsen_mesh_text("mesh file part.msh\n" + MAT, 2.0)
    with pytest.raises(ValueError):
        coarsen_mesh_text("mesh box 1 1 1 2 2 2\n" + MAT, 1.0)


def test_mean_edge_length_of_unit_cells():
    pytest.importorskip("skfem")
    from mos.core.meshes import box_mesh, mean_edge_length
    assert mean_edge_length(box_mesh(4, 2, 2, 4, 2, 2, "hex")) == pytest.approx(1.0)
    # tets of a unit cube: 3 axis edges, 2 face diagonals and 1 body diagonal per tet on average
    tet = mean_edge_length(box_mesh(4, 2, 2, 4, 2, 2, "tet"))
    assert 1.0 < tet < np.sqrt(3)
