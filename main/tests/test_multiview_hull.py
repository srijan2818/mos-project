"""Visual hull from known cameras: dimensions, holes, wrong-view detection, FE mesh."""
import numpy as np
import pytest

pytest.importorskip("skimage")

from mos.core.marker import CameraPose
from mos.core.multiview_hull import HullView, carve, surface, view_consistency
from tests import synthetic_views as sv

LO, HI = np.array([0.105, -0.11, 0.0]), np.array([0.165, -0.07, 0.03])   # 60 x 40 x 30 mm box
CENTRE = (LO + HI) / 2


def _views(mask_fn, cams=None):
    cams = cams or sv.lmat_views(CENTRE)
    return [HullView(name, mask_fn(R, t), CameraPose(sv.K, R, t, (sv.W, sv.H)))
            for name, (R, t) in cams.items()]


def _inside(hull, P):
    idx = np.round((np.asarray(P) - hull.origin) / hull.voxel).astype(int)
    ok = np.all((idx >= 0) & (idx < hull.occupied.shape), axis=1)
    out = np.zeros(len(P), bool)
    out[ok] = hull.occupied[tuple(idx[ok].T)]
    return out


def test_l_mat_hull_of_a_box_contains_it_and_measures_it():
    hull = carve(_views(lambda R, t: sv.box_mask(LO, HI, R, t)), resolution=96)
    true = HI - LO
    # the hull contains the box (sample interior points away from the boundary voxels)
    g = np.stack(np.meshgrid(*[np.linspace(LO[i] + 2 * hull.voxel, HI[i] - 2 * hull.voxel, 6)
                               for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
    assert _inside(hull, g).all()
    # extents: never smaller than the box; plan size larger by perspective (cones widen
    # behind the part), height fixed by the table-level photo
    ext = hull.extent()
    assert np.all(ext >= true - 2 * hull.voxel), (ext, true)
    assert ext[2] == pytest.approx(true[2], abs=2 * hull.voxel)
    excess = hull.volume / np.prod(true) - 1
    print(f"L-mat box (exact poses): extent {ext * 1000} mm vs {true * 1000}, volume excess {excess:.1%}")
    assert 0 <= excess < 0.10
    assert min(hull.view_iou.values()) > 0.9, hull.view_iou


def test_through_hole_seen_by_one_view_is_kept():
    r, c = 0.008, CENTRE[:2]
    circle = lambda z: np.array([[c[0] + r * np.cos(a), c[1] + r * np.sin(a), z]
                                 for a in np.linspace(0, 2 * np.pi, 48, endpoint=False)])

    def mask(R, t):
        m = sv.box_mask(LO, HI, R, t)
        if R[2] @ np.array([0, 0, -1.0]) > 0.99:                 # the top view looks through it
            m &= ~(sv.silhouette([circle(HI[2])], R, t) & sv.silhouette([circle(0.0)], R, t))
        return m
    hull = carve(_views(mask), resolution=96)
    assert not _inside(hull, [[c[0], c[1], 0.015]])[0]            # hole axis is empty
    assert _inside(hull, [[c[0] + 0.02, c[1], 0.015]])[0]         # material beside it


def test_a_view_of_the_wrong_object_is_flagged():
    other_lo, other_hi = LO + [0.0, 0.05, 0.0], HI + [0.0, 0.05, 0.0]   # something else behind
    views = _views(lambda R, t: sv.box_mask(LO, HI, R, t))
    R, t = views[1].pose.R, views[1].pose.t
    views[1].mask = sv.box_mask(LO, HI, R, t) | sv.box_mask(other_lo, other_hi, R, t)
    hull = carve(views, resolution=64)
    iou = view_consistency(views, hull)
    assert iou["front"] < 0.85 and min(iou["top"], iou["side"]) > 0.9, iou


def test_surface_and_tet_mesh_in_app_frame():
    pytest.importorskip("skfem")
    from mos.core.cad import HAS_GMSH
    if not HAS_GMSH:
        pytest.skip("gmsh not available")
    from mos.core.multiview_hull import mesh_surface
    ring = {f"r{k}": sv.look_at(CENTRE + 0.45 * np.array([np.cos(a) * np.cos(0.6), np.sin(a) * np.cos(0.6),
                                                          np.sin(0.6)]), CENTRE)
            for k, a in enumerate(np.linspace(0, 2 * np.pi, 8, endpoint=False))}
    ring.update(sv.lmat_views(CENTRE))
    hull = carve(_views(lambda R, t: sv.box_mask(LO, HI, R, t), ring), resolution=64)
    verts, faces = surface(hull)
    mesh = mesh_surface(verts, faces)
    p = mesh.p
    ext = p.max(axis=1) - p.min(axis=1)
    # app frame: X = x (60 mm), Y = z (30 mm, up), Z = -y (40 mm)
    assert ext == pytest.approx([0.060, 0.030, 0.040], rel=0.10)   # plan size: perspective excess
    assert p[1].min() == pytest.approx(0.0, abs=1e-12)
    # the six largest faces are the box's sides (normal grouping despite rounded edges); the hull
    # may add small slanted strips where perspective widens it along the bottom edges
    assert len(mesh.boundaries) >= 6
    P = mesh.p.T
    for name in [f"cad_face_{i}" for i in range(1, 7)]:
        tri = P[mesh.facets[:, mesh.boundaries[name]].T]
        ns = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])       # facets are unoriented:
        n = (ns * np.sign(ns @ ns[np.argmax(np.linalg.norm(ns, axis=1))])[:, None]).sum(axis=0)
        assert np.max(np.abs(n)) / np.linalg.norm(n) > 0.95, name        # close to an axis
    normals_y = {k: p[1, mesh.facets[:, ids]].mean() for k, ids in mesh.boundaries.items()}
    assert min(normals_y.values()) == pytest.approx(0.0, abs=2 * hull.voxel)      # bottom face


def test_l_mat_end_to_end_with_estimated_poses():
    """Photos of the L-mat -> poses from the markers -> hull: height is what the L-mat fixes."""
    pytest.importorskip("cv2")
    from mos.core.marker import BACK, FLOOR, solve_rig
    T = sv.back_sheet_pose(BACK)
    cams = sv.lmat_views(CENTRE)
    rig = solve_rig([sv.render_lmat(R, t, FLOOR, BACK, T) for R, t in cams.values()])
    views = [HullView(n, sv.box_mask(LO, HI, R, t), p)
             for n, (R, t), p in zip(cams, cams.values(), rig.poses)]
    hull = carve(views, resolution=96)
    ext = hull.extent()
    excess = hull.volume / np.prod(HI - LO) - 1
    print(f"L-mat box: extent {np.round(ext * 1000, 1)} mm, volume excess {excess:.1%}, iou {hull.view_iou}")
    assert ext[2] == pytest.approx(0.030, abs=0.0015)          # height within 1.5 mm
    assert np.all(ext[:2] < (HI - LO)[:2] * 1.10)               # plan dimensions within +10 %
    assert -0.02 < excess < 0.10
    assert min(hull.view_iou.values()) > 0.85


def test_mesh_hull_directive_solves_like_a_box(tmp_path):
    """Saved session -> `mesh hull` spec -> FE solve; compare with `mesh box` of the hull's size."""
    pytest.importorskip("skfem")
    from mos.core.cad import HAS_GMSH
    if not HAS_GMSH:
        pytest.skip("gmsh not available")
    from mos.core.multiview_hull import save_session
    from mos.core.solver import solve_case
    from mos.core.spec import build, parse_spec
    lo, hi = np.array([0.075, -0.10, 0.0]), np.array([0.195, -0.08, 0.02])    # 120 x 20 x 20 mm bar
    views = _views(lambda R, t: sv.box_mask(lo, hi, R, t), sv.lmat_views((lo + hi) / 2))
    save_session(tmp_path / "bar", views)

    def tip_deflection(mesh_line, mesh=None):
        spec = parse_spec(f"{mesh_line}\nmaterial E = 210e9, nu = 0.3\n")
        from mos.core.spec import make_mesh
        mesh = make_mesh(spec) if mesh is None else mesh
        if getattr(mesh, "boundaries", None):     # hull: the named end faces, as a user would pick
            cx = {k: mesh.p[0, mesh.facets[:, ids]].mean() for k, ids in mesh.boundaries.items()}
            root, tip = min(cx, key=cx.get), max(cx, key=cx.get)
            regions = ""
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
    print(f"hull bar {L*1000:.1f} x {Hh*1000:.1f} x {Wd*1000:.1f} mm: tip {d_hull*1e6:.2f} um vs box {d_box*1e6:.2f} um")
    assert d_hull == pytest.approx(d_box, rel=0.15)
