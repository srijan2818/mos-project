"""Three orthogonal photos -> visual hull: scale fit, dimensions, holes, wrong views, FE mesh."""
import numpy as np
import pytest

from mos.core.multiview_hull import OrthoView, carve_ortho, fit_scales
from tests import synthetic_views as sv

BOX = [(np.array([0.0, 0.0, 0.0]), np.array([0.060, 0.040, 0.030]))]     # 60 x 40 x 30 mm
CENTRE = np.array([0.030, 0.020, 0.015])


def _views(masks, **orient):
    return [OrthoView(name, m, name, **orient.get(name, {})) for name, m in masks.items()]


def _inside(hull, P):
    idx = np.floor((np.asarray(P) - hull.origin) / hull.voxel + 0.5).astype(int)
    ok = np.all((idx >= 0) & (idx < hull.occupied.shape), axis=1)
    out = np.zeros(len(P), bool)
    out[ok] = hull.occupied[tuple(idx[ok].T)]
    return out


def test_scales_recovered_from_photos_at_different_zoom():
    masks = sv.three_masks(BOX, CENTRE, px_per_m=(3100.0, 2400.0, 4000.0))
    size, k, mismatch = fit_scales(_views(masks), "x", 0.060)
    # 1 px in the bounding box is ~0.3-0.4 mm at these zooms: allow 1.5 px
    assert size == pytest.approx([0.060, 0.040, 0.030], abs=1.5 / 2400)
    assert k["front"] == pytest.approx(1 / 3100, rel=0.01) and k["top"] == pytest.approx(1 / 2400, rel=0.01)
    assert mismatch < 0.02                                # bounding-box pixel rounding only


def test_box_dimensions_and_volume():
    masks = sv.three_masks(BOX, CENTRE, px_per_m=(3100.0, 2400.0, 4000.0))
    hull = carve_ortho(_views(masks), "x", 0.060, resolution=120)
    assert hull.extent() == pytest.approx([0.060, 0.040, 0.030], abs=hull.voxel)
    assert hull.volume == pytest.approx(0.060 * 0.040 * 0.030, rel=0.03)


def test_l_bracket_matches_its_true_volume():
    """An L section extruded along y: the intersection of three extrusions is exact for it."""
    L = [(np.array([0, 0, 0.0]), np.array([0.080, 0.030, 0.010])),     # foot 80 x 30 x 10
         (np.array([0, 0, 0.0]), np.array([0.010, 0.030, 0.050]))]     # upright 10 x 30 x 50
    true_vol = 0.080 * 0.030 * 0.010 + 0.010 * 0.030 * 0.040
    hull = carve_ortho(_views(sv.three_masks(L, [0.040, 0.015, 0.025])), "z", 0.050, resolution=128)
    print(f"L-bracket volume {hull.volume * 1e6:.2f} cm3 vs {true_vol * 1e6:.2f}")
    assert hull.volume == pytest.approx(true_vol, rel=0.03)
    assert not _inside(hull, [[0.050, 0.015, 0.030]])[0]          # above the foot: empty
    assert _inside(hull, [[0.005, 0.015, 0.040]])[0]              # in the upright


def test_through_hole_seen_from_above_is_kept():
    masks = sv.three_masks(BOX, CENTRE)
    yy, xx = np.mgrid[:sv.H, :sv.W]
    masks["top"] &= (xx - sv.W / 2) ** 2 + (yy - sv.H / 2) ** 2 > (0.008 * 4000) ** 2   # r = 8 mm
    hull = carve_ortho(_views(masks), "x", 0.060, resolution=96)
    assert not _inside(hull, [CENTRE])[0]                         # hole axis is empty
    assert _inside(hull, [CENTRE + [0.018, 0, 0]])[0]             # material beside it


def test_mirrored_photo_cannot_be_detected_from_silhouettes():
    """Why the app shows each photo's oriented outline instead of trusting an agreement score:
    a stepped block with its side photo mirrored gives a different solid that reproduces all
    three photos exactly (so orientation must be checked by eye)."""
    step = [(np.array([0, 0, 0.0]), np.array([0.060, 0.020, 0.030])),      # tall front half
            (np.array([0, 0.020, 0.0]), np.array([0.060, 0.040, 0.010]))]  # low back half
    masks = sv.three_masks(step, CENTRE)
    masks["side"] = masks["side"][:, ::-1]
    hull = carve_ortho(_views(masks), "x", 0.060, resolution=96)
    proj = hull.occupied.any(axis=0)                                  # (y, z) seen from the side
    assert _inside(hull, [[0.03, 0.03, 0.02]])[0] and not _inside(hull, [[0.03, 0.01, 0.02]])[0]
    assert proj[:, -1].sum() == pytest.approx(proj.shape[0] / 2, abs=2)   # still a clean step
    assert hull.size_mismatch < 0.02


def test_wrong_object_in_one_photo_is_flagged():
    masks = sv.three_masks(BOX, CENTRE)
    masks["front"] = sv.box_mask(BOX + [(np.array([0.0, 0, 0.03]), np.array([0.015, 0.04, 0.05]))],
                                 "front", CENTRE)           # a hand / another object on top
    hull = carve_ortho(_views(masks), "x", 0.060, resolution=96)
    from mos.core.multiview_hull import SIZE_MISMATCH_WARN
    assert hull.size_mismatch > 4 * SIZE_MISMATCH_WARN, hull.size_mismatch


def test_rotated_portrait_photo():
    masks = sv.three_masks(BOX, CENTRE)
    masks["top"] = np.rot90(masks["top"], -1)              # phone held in portrait
    hull = carve_ortho(_views(masks, top={"rot90": 1}), "x", 0.060, resolution=96)
    assert hull.extent() == pytest.approx([0.060, 0.040, 0.030], abs=hull.voxel)


def test_perspective_cancels_for_a_box():
    """Every photo of a box outlines its near face, magnified uniformly; the typed dimension
    removes that common factor, so a box comes out exact even from close up."""
    hull = carve_ortho(_views(sv.three_masks(BOX, CENTRE, distance=0.18)), "x", 0.060, resolution=120)
    assert hull.extent() == pytest.approx([0.060, 0.040, 0.030], abs=hull.voxel)


L_BRACKET = [(np.array([0, 0, 0.0]), np.array([0.080, 0.030, 0.010])),
             (np.array([0, 0, 0.0]), np.array([0.010, 0.030, 0.050]))]


@pytest.mark.parametrize("ratio, tol", [(5.0, 0.10), (10.0, 0.06)])
def test_perspective_error_on_features_at_different_depths(ratio, tol):
    """Pinhole photos at d = ratio x the largest size: parts of one photo at different depths
    are magnified differently (relative error ~ depth difference / d). Measured on the
    L-bracket: d = 5x: extents <= 6 %, volume +7 %; d = 10x: <= 3 %, +4 %."""
    true_vol = 0.080 * 0.030 * 0.010 + 0.010 * 0.030 * 0.040
    hull = carve_ortho(_views(sv.three_masks(L_BRACKET, [0.040, 0.015, 0.025], distance=ratio * 0.080)),
                       "x", 0.080, resolution=128)
    err = hull.extent() / np.array([0.080, 0.030, 0.050]) - 1
    vol = hull.volume / true_vol - 1
    print(f"d = {ratio:.0f} x size: extent error {np.round(err * 100, 1)} %, volume {vol:+.1%}, "
          f"size mismatch {hull.size_mismatch:.1%}")
    assert np.all(np.abs(err) < tol) and abs(vol) < tol + 0.02


def test_surface_and_tet_mesh_in_app_frame():
    pytest.importorskip("skfem")
    pytest.importorskip("skimage")
    from mos.core.cad import HAS_GMSH
    if not HAS_GMSH:
        pytest.skip("gmsh not available")
    from mos.core.multiview_hull import mesh_surface, surface
    hull = carve_ortho(_views(sv.three_masks(BOX, CENTRE)), "x", 0.060, resolution=64)
    mesh = mesh_surface(*surface(hull))
    p = mesh.p
    # app frame: X = x (60 mm), Y = z (30 mm, up), Z = -y (40 mm); surface smoothing ~1 voxel
    assert p.max(axis=1) - p.min(axis=1) == pytest.approx([0.060, 0.030, 0.040], abs=2 * hull.voxel)
    assert p[1].min() == pytest.approx(0.0, abs=1e-12)
    assert len(mesh.boundaries) >= 6


def test_mesh_hull_directive_solves_like_a_box(tmp_path):
    """Saved session -> `mesh hull` spec -> FE solve; compare with `mesh box` of the hull's size."""
    pytest.importorskip("skfem")
    pytest.importorskip("skimage")
    from mos.core.cad import HAS_GMSH
    if not HAS_GMSH:
        pytest.skip("gmsh not available")
    from mos.core.multiview_hull import load_session, save_session
    from mos.core.solver import solve_case
    from mos.core.spec import build, make_mesh, parse_spec
    bar = [(np.array([0, 0, 0.0]), np.array([0.120, 0.020, 0.020]))]          # 120 x 20 x 20 mm
    save_session(tmp_path / "bar", _views(sv.three_masks(bar, [0.06, 0.01, 0.01], (2000.0,) * 3)),
                 "x", 0.120)
    assert load_session(tmp_path / "bar")[1:] == ("x", 0.120)

    def tip_deflection(mesh_line):
        mesh = make_mesh(parse_spec(f"{mesh_line}\nmaterial E = 210e9, nu = 0.3\n"))
        if getattr(mesh, "boundaries", None):     # hull: the named end faces, as a user would pick
            cx = {k: mesh.p[0, mesh.facets[:, ids]].mean() for k, ids in mesh.boundaries.items()}
            root, tip, regions = min(cx, key=cx.get), max(cx, key=cx.get), ""
        else:
            x0, x1 = mesh.p[0].min(), mesh.p[0].max()
            root, tip = "root", "tip"
            regions = f"region root facets x < {x0} + 1e-6\nregion tip facets x > {x1} - 1e-6\n"
        text = (f"{mesh_line}\nmaterial E = 210e9, nu = 0.3\n{regions}"
                f"support {root}\nforce {tip} 0, -100, 0\n")
        model, case = build(parse_spec(text), mesh=mesh)
        res = solve_case(model, case)
        assert res.equilibrium_error < 1e-8
        return -res.u_nodal[1].min(), mesh
    d_hull, mesh = tip_deflection(f'mesh hull "{tmp_path / "bar"}" 0.004')
    L, Hh, Wd = (mesh.p.max(axis=1) - mesh.p.min(axis=1))
    d_box, _ = tip_deflection(f"mesh box {L} {Hh} {Wd} 24 4 4")
    E, I = 210e9, 0.020 ** 4 / 12
    print(f"hull bar {L*1000:.1f} x {Hh*1000:.1f} x {Wd*1000:.1f} mm: tip {d_hull*1e6:.2f} um vs "
          f"box {d_box*1e6:.2f} um vs beam theory {100 * 0.120**3 / (3 * E * I) * 1e6:.2f} um")
    assert d_hull == pytest.approx(d_box, rel=0.15)


@pytest.mark.parametrize("mesh_size", [None, 0.002])
def test_mesh_hull_at_the_apps_default_element_size(tmp_path, mesh_size):
    """Regression: the app's default size (1/15 of the part) used marching-cubes steps 3-4 on
    the fine grid, and Gmsh generated no tetrahedra. Now the hull is carved at ~2 voxels per
    element and surfaced at step 1."""
    pytest.importorskip("skfem")
    pytest.importorskip("skimage")
    from mos.core.cad import HAS_GMSH
    if not HAS_GMSH:
        pytest.skip("gmsh not available")
    from mos.core.multiview_hull import load_hull_mesh, save_session
    save_session(tmp_path / "box", _views(sv.three_masks(BOX, CENTRE, (3000.0, 2500.0, 3500.0))), "x", 0.060)
    mesh = load_hull_mesh(tmp_path / "box", mesh_size)
    ext = mesh.p.max(axis=1) - mesh.p.min(axis=1)                 # app frame: x, z (up), -y
    assert ext == pytest.approx([0.060, 0.030, 0.040], abs=0.0025)    # within ~1 coarse voxel
    assert mesh.t.shape[1] > 1000 and len(mesh.boundaries) >= 6
