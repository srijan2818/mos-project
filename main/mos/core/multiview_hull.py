"""Solid from several photos: visual hull carving on the marker mat -> tetrahedral FE mesh.

Each view contributes a silhouette mask and a camera pose (``mos.core.marker``). A point of
space belongs to the part only if it projects inside the part's mask in *every* view; the
set of such points is the **visual hull**: the largest solid consistent with the silhouettes.

What it can and cannot represent:

* exact for convex parts seen from enough directions; through-holes are kept when some view
  looks through them;
* concavities that never show in any silhouette (blind pockets, recessed faces) are filled in,
  so the hull over-estimates volume and stiffness there;
* with only three views, regions no view separates (e.g. edges between two viewing directions)
  are over-estimated as well; more views (short video) tighten the hull.

Discretisation: the hull is carved on a voxel grid, then surfaced by marching cubes on a
slightly smoothed occupancy. Boundary error is O(voxel size); ``resolution`` voxels span the
part's largest extent. Units are metres throughout.

Coordinates: carving happens in the mat frame (z up). The returned FE mesh uses the app's
convention, y up: ``(X, Y, Z) = (x, z, -y)`` (a proper rotation), table plane at Y = 0.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import ndimage as ndi

from .marker import CameraPose

log = logging.getLogger(__name__)

# mat frame (z up) -> app frame (y up); orthogonal, det = +1
TO_APP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


@dataclass
class HullView:
    """One photo: its name, the part's silhouette (H, W) and the camera that took it."""
    name: str
    mask: np.ndarray
    pose: CameraPose


@dataclass
class Hull:
    """Carved occupancy grid in the mat frame (voxel centres at origin + (i, j, k) * voxel)."""
    occupied: np.ndarray
    origin: np.ndarray
    voxel: float
    view_iou: Dict[str, float] = field(default_factory=dict)

    @property
    def volume(self) -> float:
        """Carved volume (m^3), voxel-counted."""
        return float(self.occupied.sum()) * self.voxel ** 3

    def extent(self) -> np.ndarray:
        """Bounding-box size (m) of the occupied voxels, centre to centre plus one voxel."""
        idx = np.argwhere(self.occupied)
        return (idx.max(axis=0) - idx.min(axis=0) + 1) * self.voxel


def _inside_all(views: List[HullView], P: np.ndarray) -> np.ndarray:
    """For world points (N, 3): inside the mask, in front of the camera, in every view."""
    keep = np.ones(len(P), bool)
    for v in views:
        x = P[keep] @ v.pose.R.T + v.pose.t
        front = x[:, 2] > 1e-6
        uv = x[:, :2] / np.where(front, x[:, 2], 1.0)[:, None] @ np.diag(v.pose.K[[0, 1], [0, 1]]) \
            + v.pose.K[:2, 2]
        u = np.floor(uv[:, 0]).astype(np.int64)
        w = np.floor(uv[:, 1]).astype(np.int64)
        h, wd = v.mask.shape
        ok = front & (u >= 0) & (u < wd) & (w >= 0) & (w < h)
        hit = np.zeros(len(x), bool)
        hit[ok] = v.mask[w[ok], u[ok]]
        sub = np.flatnonzero(keep)
        keep[sub[~hit]] = False
        if not keep.any():
            break
    return keep


def _grid(lo: np.ndarray, hi: np.ndarray, voxel: float):
    """Voxel-centre coordinates (N, 3) and the grid shape covering [lo, hi]."""
    n = np.maximum(1, np.ceil((hi - lo) / voxel).astype(int))
    axes = [lo[i] + (np.arange(n[i]) + 0.5) * voxel for i in range(3)]
    P = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return P, tuple(n)


def carve(views: List[HullView], resolution: int = 128, search_box=None) -> Hull:
    """Visual hull of the part above the mat (z >= 0).

    A coarse pass (48 voxels over ``search_box``) finds the part; a fine pass puts
    ``resolution`` voxels across its largest extent. ``search_box`` = (lo, hi) in the mat
    frame; default: the area around the cameras' common view, z from 0 up.
    """
    if len(views) < 2:
        raise ValueError("a solid needs at least two views; use three or more")
    if search_box is None:
        centres = np.array([v.pose.centre for v in views])
        span = float(np.max(np.linalg.norm(centres - centres.mean(axis=0), axis=1))) + 0.05
        mid = centres.mean(axis=0)
        lo = np.array([mid[0] - span, mid[1] - span, 0.0])
        hi = np.array([mid[0] + span, mid[1] + span, span])
    else:
        lo, hi = (np.asarray(b, float) for b in search_box)
    coarse = float(np.max(hi - lo)) / 48.0
    P, shape = _grid(lo, hi, coarse)
    occ = _inside_all(views, P).reshape(shape)
    if not occ.any():
        raise ValueError("no point of space is inside every silhouette: check that each view's "
                         "mask is the same part and that the mat was detected correctly")
    idx = np.argwhere(occ)
    flo = lo + (idx.min(axis=0) - 1.5) * coarse
    fhi = lo + (idx.max(axis=0) + 2.5) * coarse
    flo[2] = max(flo[2], 0.0)
    voxel = float(np.max(fhi - flo)) / resolution
    P, shape = _grid(flo, fhi, voxel)
    occ = _inside_all(views, P).reshape(shape)
    occ = ndi.binary_opening(occ, iterations=1) if occ.sum() > 64 else occ
    labels, n = ndi.label(occ)
    if n > 1:   # keep the part; carving leaves small ghost pieces where views barely overlap
        occ = labels == 1 + int(np.argmax(np.bincount(labels.ravel())[1:]))
    hull = Hull(occupied=occ, origin=flo + 0.5 * voxel, voxel=voxel)
    hull.view_iou = view_consistency(views, hull)
    return hull


def view_consistency(views: List[HullView], hull: Hull) -> Dict[str, float]:
    """IoU between each view's mask and the hull projected back into that view.

    The hull is the intersection of all silhouette cones, so it can only *shrink* a correct
    mask. A view whose mask is much larger than the reprojection (IoU well below 1) shows
    something the other views rule out: usually the wrong object or a merged hand.
    """
    out = {}
    for v in views:
        proj = project_hull(hull, v.pose)
        out[v.name] = float((proj & v.mask).sum() / max(1, (proj | v.mask).sum()))
    return out


def project_hull(hull: Hull, pose: CameraPose) -> np.ndarray:
    """(H, W) bool silhouette of the carved hull as seen by ``pose``."""
    pts = hull.origin + np.argwhere(hull.occupied) * hull.voxel
    x = pts @ pose.R.T + pose.t
    x = x[x[:, 2] > 1e-6]
    w, h = pose.image_size
    proj = np.zeros((h, w), bool)
    if not len(x):
        return proj
    uv = x[:, :2] / x[:, 2:3] @ np.diag(pose.K[[0, 1], [0, 1]]) + pose.K[:2, 2]
    u = np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1)
    r = np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1)
    proj[r, u] = True
    # voxel centres are sparse in the image: close the gaps of ~1 voxel's footprint
    px = int(np.ceil(np.median(pose.K[0, 0] * hull.voxel / x[:, 2]))) + 1
    proj = ndi.binary_closing(ndi.binary_dilation(proj, iterations=px), iterations=px)
    return ndi.binary_erosion(proj, iterations=px)


def surface(hull: Hull, smooth_voxels: float = 1.0, step: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Closed triangle surface (vertices in the mat frame, faces) by marching cubes.

    ``step`` > 1 samples every step-th voxel: fewer, larger triangles (and tets).
    """
    from skimage.measure import marching_cubes
    field_ = np.pad(hull.occupied.astype(float), 2)
    if smooth_voxels > 0:
        field_ = ndi.gaussian_filter(field_, smooth_voxels)
    verts, faces, _, _ = marching_cubes(field_, level=0.5, step_size=max(1, int(step)))
    verts = hull.origin + (verts - 2.0) * hull.voxel
    return verts, faces.astype(np.int64)


def name_faces_by_normal(mesh, max_dev_deg: float = 35.0, min_area_frac: float = 0.01) -> Dict[str, np.ndarray]:
    """Group a mesh's boundary facets into faces for supports and loads.

    Region growing over edge-adjacent boundary triangles: a triangle joins a face while its
    outward normal stays within ``max_dev_deg`` of the face's area-weighted mean normal (seeds
    in order of area). Unlike a bend-angle split, a rounded edge (marching cubes on a smoothed
    grid) is cut along its middle instead of merging its two faces. Faces under
    ``min_area_frac`` of the surface join the adjacent face with the closest normal.
    Returns ``{"cad_face_N": facet ids}``, N = 1 for the largest face.
    """
    bf = np.asarray(mesh.boundary_facets())
    tri = mesh.facets[:, bf].T                                  # (n, 3) vertex ids
    P = mesh.p.T
    nrm = np.cross(P[tri[:, 1]] - P[tri[:, 0]], P[tri[:, 2]] - P[tri[:, 0]])
    area = 0.5 * np.linalg.norm(nrm, axis=1)
    nrm /= np.maximum(2 * area, 1e-300)[:, None]
    # orient outward: away from the owning tetrahedron's fourth vertex
    owner = mesh.f2t[0, bf]
    cent = P[tri].mean(axis=1)
    inner = P[mesh.t[:, owner].T].mean(axis=1)
    nrm *= np.sign(np.einsum("ij,ij->i", nrm, cent - inner))[:, None]
    # edge adjacency
    edges = np.sort(np.stack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]], axis=1).reshape(-1, 2), axis=1)
    owner_of = np.repeat(np.arange(len(tri)), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    e, o = edges[order], owner_of[order]
    same = np.all(e[1:] == e[:-1], axis=1)
    nbrs = [[] for _ in range(len(tri))]
    for a, b in zip(o[:-1][same], o[1:][same]):
        nbrs[a].append(b)
        nbrs[b].append(a)
    cos_max = np.cos(np.radians(max_dev_deg))
    label = np.full(len(tri), -1)
    groups = []
    for seed in np.argsort(-area):
        if label[seed] >= 0:
            continue
        g = len(groups)
        label[seed] = g
        members, mean, stack = [seed], nrm[seed] * area[seed], [seed]
        while stack:
            cur = stack.pop()
            for nb in nbrs[cur]:
                if label[nb] < 0 and nrm[nb] @ mean / np.linalg.norm(mean) >= cos_max:
                    label[nb] = g
                    members.append(nb)
                    mean = mean + nrm[nb] * area[nb]
                    stack.append(nb)
        groups.append(np.array(members))
    total = area.sum()
    big = [g for g in range(len(groups)) if area[groups[g]].sum() >= min_area_frac * total]
    for g in range(len(groups)):          # small slivers -> best-aligned adjacent big face
        if g in big:
            continue
        mean = (nrm[groups[g]] * area[groups[g]][:, None]).sum(axis=0)
        adj = {label[nb] for i in groups[g] for nb in nbrs[i]} & set(big)
        cand = adj or set(big)
        best = max(cand, key=lambda h: (nrm[groups[h]] * area[groups[h]][:, None]).sum(axis=0) @ mean)
        label[groups[g]] = best
    faces = sorted({int(l) for l in label}, key=lambda l: -area[label == l].sum())
    return {f"cad_face_{i + 1}": np.sort(bf[label == l]) for i, l in enumerate(faces)}


def mesh_surface(verts: np.ndarray, faces: np.ndarray):
    """Tetrahedral MeshTet (app frame, y up, table at Y = 0) inside a closed triangle surface.

    The marching-cubes triangles are kept as the boundary (a watertight, non-self-intersecting
    surface) and Gmsh fills the volume. Re-meshing the surface through Gmsh's
    classify/reparametrise step was tried first and failed on a real session ("a segment and a
    facet intersect"); element size is set by the marching-cubes ``step`` instead (see
    :func:`load_hull_mesh`). Faces are named by :func:`name_faces_by_normal`.
    """
    from skfem import MeshTet
    from .cad import gmsh_to_meshtet
    from .gmsh_session import GMSH_LOCK, ensure_gmsh, gmsh
    v_app = verts @ TO_APP.T
    v_app[:, 1] -= v_app[:, 1].min()
    with GMSH_LOCK:
        ensure_gmsh()
        gmsh.clear()
        try:
            gmsh.model.add("visual_hull")
            s = gmsh.model.addDiscreteEntity(2)
            gmsh.model.mesh.addNodes(2, s, np.arange(1, len(v_app) + 1), v_app.ravel())
            gmsh.model.mesh.addElementsByType(s, 2, [], (faces + 1).ravel())
            loop = gmsh.model.geo.addSurfaceLoop([s])
            gmsh.model.geo.addVolume([loop])
            gmsh.model.geo.synchronize()
            gmsh.option.setNumber("Mesh.ElementOrder", 1)
            gmsh.model.mesh.generate(3)
            mesh = gmsh_to_meshtet({"surface": [s]})
        finally:
            gmsh.clear()
    return MeshTet(mesh.p, mesh.t, _boundaries=name_faces_by_normal(mesh))


# ----------------------------------------------------------------- sessions (spec directive)
def save_session(directory: Union[str, Path], views: List[HullView]) -> Path:
    """Write masks (PNG) and cameras (JSON) so ``mesh hull DIR`` can rebuild the solid."""
    from PIL import Image
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    meta = []
    for v in views:
        Image.fromarray(v.mask.astype(np.uint8) * 255).save(d / f"{v.name}_mask.png")
        meta.append({"name": v.name, "K": v.pose.K.tolist(), "R": v.pose.R.tolist(),
                     "t": v.pose.t.tolist(), "image_size": list(v.pose.image_size),
                     "corners": v.pose.corners, "rms_px": v.pose.rms_px})
    (d / "cameras.json").write_text(json.dumps({"views": meta}, indent=1))
    return d


def load_session(directory: Union[str, Path]) -> List[HullView]:
    """Views saved by :func:`save_session`."""
    from PIL import Image
    d = Path(directory)
    meta = json.loads((d / "cameras.json").read_text())["views"]
    return [HullView(m["name"], np.asarray(Image.open(d / f"{m['name']}_mask.png").convert("L")) > 127,
                     CameraPose(np.array(m["K"]), np.array(m["R"]), np.array(m["t"]),
                                tuple(m["image_size"]), m.get("corners", 0), m.get("rms_px", float("nan"))))
            for m in meta]


def load_hull_mesh(directory: Union[str, Path], mesh_size: Optional[float] = None,
                   resolution: int = 128):
    """Session directory -> carved hull -> tetrahedral mesh (the ``mesh hull`` directive)."""
    views = load_session(directory)
    hull = carve(views, resolution=resolution)
    bad = {k: round(v, 2) for k, v in hull.view_iou.items() if v < 0.85}
    if bad:
        log.warning("views disagree with the hull (IoU < 0.85): %s - check those masks", bad)
    # element size ~ marching-cubes triangles ~ 2 * step voxels; default ~ 1/15 of the part
    target = float(mesh_size) if mesh_size and mesh_size > 0 else float(np.max(hull.extent())) / 15.0
    step = int(np.clip(round(target / (2.0 * hull.voxel)), 1, 4))
    verts, faces = surface(hull, step=step)
    return mesh_surface(verts, faces)
