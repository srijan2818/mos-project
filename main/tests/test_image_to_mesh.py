"""Direct image -> mesh path, outward face normals, and the vectorised helpers it relies on."""
import threading

import numpy as np
import pytest
from PIL import Image

from mos.core.cad import HAS_GMSH
from mos.core.image_to_cad import _rdp_simplify, polygon_area
from mos.core.image_to_mesh import mask_contours
from mos.core.post import nodal_average


def _plate_with_hole(w=300, h=150, r=30):
    yy, xx = np.mgrid[:h + 40, :w + 40]
    mask = (xx >= 20) & (xx < 20 + w) & (yy >= 20) & (yy < 20 + h)
    mask &= (xx - (20 + w / 2)) ** 2 + (yy - (20 + h / 2)) ** 2 > r ** 2
    return mask


def test_mask_contours_finds_outline_and_hole():
    outer, holes = mask_contours(_plate_with_hole(), simplify_px=1.0)
    assert len(holes) == 1
    assert 4 <= len(outer) <= 8                           # a rectangle survives RDP as ~4 corners
    assert polygon_area(outer) == pytest.approx(300 * 150, rel=0.03)
    assert polygon_area(holes[0]) == pytest.approx(np.pi * 30 ** 2, rel=0.08)


def test_rdp_keeps_corners_and_drops_collinear_points():
    line = np.column_stack([np.arange(100.0), np.zeros(100)])
    corner = np.vstack([line, np.column_stack([np.full(50, 99.0), np.arange(1.0, 51.0)])])
    out = _rdp_simplify(corner, 0.5)
    assert len(out) == 3
    assert np.allclose(out[1], [99.0, 0.0])


def test_vectorised_nodal_average_matches_reference():
    rng = np.random.default_rng(0)
    nel, nn = 40, 25
    t = rng.integers(0, nn, size=(4, nel))
    t[:, 0] = np.arange(4)
    Tc = rng.random((3, 3, nel))
    vol = rng.random(nel) + 0.1
    flat = Tc.reshape(-1, nel)
    num, den = np.zeros((9, nn)), np.zeros(nn)
    for k in range(4):
        np.add.at(num, (slice(None), t[k]), flat * vol)
        np.add.at(den, t[k], vol)
    with np.errstate(invalid="ignore"):
        ref = (num / den).reshape(3, 3, nn)
    out = nodal_average(Tc, t, vol, nn)
    used = den > 0
    assert np.allclose(out[:, :, used], ref[:, :, used])


skfem = pytest.importorskip("skfem")
needs_gmsh = pytest.mark.skipif(not HAS_GMSH, reason="gmsh not available")


@needs_gmsh
def test_mesh_from_mask_volume_faces_and_scale():
    from mos.core.image_to_mesh import mesh_from_mask
    from mos.core.regions import boundary_normal

    mesh, info = mesh_from_mask(_plate_with_hole(), size_m=0.30, thickness_m=0.01)
    assert info["n_holes"] == 1
    roles = info["faces"]
    assert roles["cad_face_1"].startswith("bottom") and roles["cad_face_2"].startswith("top")
    assert sum(r == "outer side wall" for r in roles.values()) == 4
    assert set(roles) == set(mesh.boundaries)
    dims = np.ptp(mesh.p, axis=1)
    assert dims == pytest.approx([0.30, 0.15, 0.01], rel=0.02)
    vol = skfem.Basis(mesh, skfem.ElementTetP1()).dx.sum()
    expected = (0.30 * 0.15 - np.pi * 0.03 ** 2) * 0.01
    assert vol == pytest.approx(expected, rel=0.03)
    assert boundary_normal(mesh, mesh.boundaries["cad_face_1"]) == pytest.approx([0, 0, -1], abs=1e-9)
    assert boundary_normal(mesh, mesh.boundaries["cad_face_2"]) == pytest.approx([0, 0, 1], abs=1e-9)
    for name, role in roles.items():
        if role == "outer side wall":
            n = boundary_normal(mesh, mesh.boundaries[name])
            c = mesh.p[:, mesh.facets[:, mesh.boundaries[name]]].mean(axis=(1, 2))
            # Each wall's outward normal points away from the plate centre.
            assert np.dot(n, c - np.array([0.15, 0.075, 0.005])) > 0


@needs_gmsh
def test_mask_directive_builds_and_cantilever_matches_beam_theory(tmp_path):
    from mos.core.image_to_mesh import save_mask_image
    from mos.core.solver import solve_case
    from mos.core.spec import build, parse_spec

    mask = np.zeros((60, 420), dtype=bool)
    mask[10:50, 10:410] = True                       # 400 x 40 px bar
    path = save_mask_image(mask, tmp_path / "bar mask.png")
    L, H, B = 1.0, 0.1, 0.05
    spec = parse_spec(f"""
mesh     mask "{path}" {L} {B} 0.025
material E = 200e9, nu = 0.0
region   root facets x < 1e-9
region   tip  facets x > {L} - 1e-9
support  root xyz
force    tip 0, -1000, 0
""")
    model, case = build(spec)
    res = solve_case(model, case)
    delta = 1000 * L ** 3 / (3 * 200e9 * B * H ** 3 / 12)
    tip = np.abs(model.mesh.p[0] - L) < 1e-9
    assert abs(res.u_nodal[1, tip].mean()) == pytest.approx(delta, rel=0.06)
    assert res.equilibrium_error < 1e-8


@needs_gmsh
def test_gmsh_works_from_a_worker_thread():
    from mos.core.image_to_mesh import mesh_from_mask
    out = []

    def work():
        try:
            out.append(mesh_from_mask(_plate_with_hole(), 0.3, 0.01)[0].nelements)
        except Exception as exc:          # pragma: no cover - reported below
            out.append(exc)

    th = threading.Thread(target=work)
    th.start()
    th.join()
    assert isinstance(out[0], int) and out[0] > 0, out


@needs_gmsh
def test_cad_face_normals_point_outward():
    from pathlib import Path

    from mos import viz
    from mos.core.cad import load_step_mesh

    root = Path(__file__).resolve().parents[1]
    mesh, info, _ = load_step_mesh(root / "cad_models" / "beam.step")
    faces = viz.face_definitions((0, 1, 0, 1, 0, 1), mesh=mesh)
    lo, hi = mesh.p.min(axis=1), mesh.p.max(axis=1)
    for name, ids in mesh.boundaries.items():
        n = np.asarray(faces[name]["normal"])
        assert np.allclose(n, info[name]["normal"])
        c = mesh.p[:, mesh.facets[:, ids]].mean(axis=(1, 2))
        axis = int(np.argmax(np.abs(n)))
        # A box face's normal points toward the side of the box it lies on.
        assert (c[axis] > 0.5 * (lo[axis] + hi[axis])) == (n[axis] > 0)


def test_detector_label_normalisation_rejects_glued_phrases():
    from mos.core.cv_detector import _map_label

    assert _map_label("structural l - bracket") == "bracket"
    assert _map_label("i - beam") == "ibeam"
    assert _map_label("plate with hole") == "plate_hole"
    assert _map_label("structural l - bracket rectangular beam i - beam stepped shaft") is None


def test_cv_result_exposes_the_measured_mask(tmp_path):
    from PIL import ImageDraw

    from mos.core.cv_detector import analyze_image_object

    img = Image.new("RGB", (400, 300), (235, 235, 235))
    ImageDraw.Draw(img).rectangle([(60, 100), (340, 180)], fill=(40, 45, 55))
    res = analyze_image_object(img, reference_scale=0.5, use_models=False)
    assert res.mask is not None and res.mask.shape == (300, 400)
    ys, xs = np.nonzero(res.mask)
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (60, 340, 100, 180)   # no halo
    assert "classical" in res.segmentation


def test_iterative_solver_path_matches_direct(monkeypatch):
    from mos.core import solver
    from mos.core.spec import build, parse_spec

    text = """
mesh     box 2 0.2 0.2 16 2 2
material E = 200e9, nu = 0.3
region   root facets x < 1e-9
region   tip  facets x > 2 - 1e-9
support  root xyz
force    tip 0, -1000, 0
"""
    model, case = build(parse_spec(text))
    direct = solver.solve_case(model, case)
    monkeypatch.setattr(solver, "DIRECT_SOLVE_MAX_DOFS", 0)        # force preconditioned CG
    iterative = solver.solve_case(model, case)
    assert np.allclose(iterative.u, direct.u, rtol=1e-6, atol=1e-6 * np.abs(direct.u).max())
    assert iterative.equilibrium_error < 1e-6


def test_isotropic_fast_assembly_equals_general_form():
    from mos.core import Material
    from mos.core.meshes import box_mesh
    from mos.core.solver import _isotropic_lame, assemble_stiffness, make_basis

    basis, _ = make_basis(box_mesh(1, 0.2, 0.2, 3, 1, 1))
    C4 = Material.isotropic(70e9, 0.33).C4
    assert _isotropic_lame(C4) is not None
    fast = assemble_stiffness(basis, C4)
    # A negligible anisotropic perturbation forces the general 81-term form.
    C_aniso = C4.copy()
    C_aniso[0, 0, 0, 0] *= 1.0 + 1e-9
    assert _isotropic_lame(C_aniso) is None
    general = assemble_stiffness(basis, C_aniso)
    assert abs(fast - general).max() <= 1e-8 * abs(general).max()


def test_outline_is_traced_from_the_largest_piece():
    """A small fragment first in raster order must not become the outer outline."""
    from mos.core.image_to_mesh import mask_contours, polygon_area
    mask = np.zeros((120, 160), bool)
    mask[2:8, 2:8] = True            # 6 x 6 fragment, top-left
    mask[40:100, 20:140] = True      # 60 x 120 part
    outer, _ = mask_contours(mask)
    assert abs(polygon_area(outer)) == pytest.approx(60 * 120, rel=0.02)
