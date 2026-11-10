"""Solid from three orthogonal photos (top, front, right side) -> tetrahedral FE mesh.

Each photo contributes the part's silhouette. Treating each photo as an orthographic
projection along its axis, a point belongs to the part only if it lies inside the silhouette
in all three views: the solid is the intersection of the three extruded silhouettes (the
**visual hull** of three orthogonal views). No printed markers are needed: the views are
registered by their bounding boxes, and one dimension typed by the user sets the scale.

Frame (z up): x = length (left -> right in the front photo), y = depth (front -> back),
z = height. Image directions of the three photos:

    front   camera in front, level      columns -> +x, rows (down) -> -z
    top     camera above, front edge at the bottom of the photo
                                        columns -> +x, rows (down) -> -y
    side    camera on the right, level  columns -> +y, rows (down) -> -z

Each axis is seen by two photos, which gives each photo's scale up to one common factor
(:func:`fit_scales`); the third pairing is a consistency check.

What it can and cannot represent, and its errors:

* exact (to the voxel size) for parts whose shape is an intersection of three extrusions
  (blocks, brackets, plates with through-holes seen along an axis); concavities that no view
  shows (blind pockets, recesses, anything only visible obliquely) are filled in, so volume and
  stiffness are over-estimated there;
* perspective: a phone photo is not orthographic. A magnification common to a whole photo
  cancels (each photo's bounding box is rescaled, the typed dimension fixes the scale), so a
  box comes out exact from any distance. Parts of one photo at different depths are
  magnified differently, by a relative ~ (depth difference) / d at camera distance d.
  Measured on an 80 x 30 x 50 mm L-bracket: d = 5x its size: extents <= 6 %, volume +7 %,
  the 10 mm upright 12.5 mm thick; d = 10x: <= 3 %, +4 %. Shoot from far, zoomed in;
* tilt: a view turned by theta from its axis adds about D sin(theta) of the depth to the width;
* checking: the only reliable consistency check is the bounding-box proportions
  (``size_mismatch``): an extra object or a hand in one photo changes its proportions.
  Silhouette agreement (projecting the hull back into each photo) is *not* used: with three
  orthogonal views the hull reproduces a view almost whenever the other two are solid
  outlines, so a mirrored photo or a hand inside the outline usually still "agrees";
* discretisation: the hull is carved on a voxel grid (``resolution`` voxels along the longest
  side) and surfaced by marching cubes on a slightly smoothed occupancy: O(voxel) boundary error.

The FE mesh uses the app's convention, y up: ``(X, Y, Z) = (x, z, -y)``, table at Y = 0.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import ndimage as ndi

log = logging.getLogger(__name__)

# hull frame (z up) -> app frame (y up); orthogonal, det = +1
TO_APP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])

# per photo: the world axes of (image columns, image rows); rows always run against the axis
VIEW_AXES = {"front": ("x", "z"), "top": ("x", "y"), "side": ("y", "z")}
AXES = "xyz"
# Proportion disagreement worth a warning. Exact silhouettes measure ~1 % (bounding-box pixel
# rounding); a camera at 5x the part size ~1-4 %; a hand or second object easily > 20 %.
SIZE_MISMATCH_WARN = 0.05


@dataclass
class OrthoView:
    """One photo: which view it is, the part's silhouette (H, W) and how to orient it.

    ``rot90`` quarter turns counter-clockwise, then ``flip_h`` mirrors left-right; both fix a
    photo taken in portrait or from the wrong side without retaking it.
    """
    name: str
    mask: np.ndarray
    axis: str                     # "front", "top" or "side"
    flip_h: bool = False
    rot90: int = 0

    def oriented(self) -> np.ndarray:
        """The mask after rotation and mirroring."""
        m = np.rot90(np.asarray(self.mask, bool), int(self.rot90) % 4)
        return m[:, ::-1] if self.flip_h else m


@dataclass
class Hull:
    """Carved occupancy grid (voxel centres at origin + (i, j, k) * voxel), hull frame."""
    occupied: np.ndarray
    origin: np.ndarray
    voxel: float
    size_mismatch: Optional[float] = None    # |e^r - 1| of the scale loop, see fit_scales

    @property
    def volume(self) -> float:
        """Carved volume (m^3), voxel-counted."""
        return float(self.occupied.sum()) * self.voxel ** 3

    def extent(self) -> np.ndarray:
        """Bounding-box size (m) of the occupied voxels, centre to centre plus one voxel."""
        idx = np.argwhere(self.occupied)
        return (idx.max(axis=0) - idx.min(axis=0) + 1) * self.voxel


def crop_part(mask: np.ndarray) -> np.ndarray:
    """Largest connected component of ``mask``, cropped to its bounding box."""
    lab, n = ndi.label(np.asarray(mask, bool))
    if n == 0:
        raise ValueError("the mask is empty")
    part = lab == 1 + int(np.argmax(np.bincount(lab.ravel())[1:]))
    rows, cols = np.flatnonzero(part.any(axis=1)), np.flatnonzero(part.any(axis=0))
    return part[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def _by_axis(views: List[OrthoView]) -> Dict[str, np.ndarray]:
    found = {v.axis: crop_part(v.oriented()) for v in views}
    if sorted(found) != sorted(VIEW_AXES) or len(views) != 3:
        raise ValueError("need exactly one top, one front and one side view")
    return found


def fit_scales(views: List[OrthoView], known_axis: str, known_m: float):
    """Part size (m, along x, y, z) and each photo's scale (m/px) from the bounding boxes.

    Each world axis spans ``n_a`` pixels in the two photos that see it, so their scales k
    satisfy ``k_i n_i = k_j n_j``: three equations for three unknowns, one of which is the common
    factor fixed by ``known_m`` (the true size along ``known_axis``). In logs the equations
    form a cycle; least squares spreads its closure residual r equally over the three, and
    ``|e^r - 1|`` (returned as ``mismatch``) is how much the three photos disagree about the
    proportions: 0 for perfect orthographic silhouettes; perspective, tilt or a wrong
    selection make it grow.
    """
    if known_axis not in AXES or not (np.isfinite(known_m) and known_m > 0):
        raise ValueError("known dimension: axis x, y or z and a positive length")
    crops = _by_axis(views)
    px = {name: dict(zip(VIEW_AXES[name], (m.shape[1], m.shape[0]))) for name, m in crops.items()}
    names = list(VIEW_AXES)                       # unknowns: log scale of front, top, side
    rows, rhs = [], []
    for a in AXES:
        i, j = [names.index(n) for n in names if a in VIEW_AXES[n]]
        row = np.zeros(3)
        row[i], row[j] = 1.0, -1.0                # log k_i - log k_j = log(n_j / n_i)
        rows.append(row)
        rhs.append(np.log(px[names[j]][a] / px[names[i]][a]))
    A, b = np.array(rows), np.array(rhs)
    l, *_ = np.linalg.lstsq(np.vstack([A, [1.0, 0.0, 0.0]]), np.r_[b, 0.0], rcond=None)
    resid = A @ l - b
    mismatch = float(abs(np.expm1(np.abs(resid).sum())))
    k = dict(zip(names, np.exp(l)))
    size = np.array([np.sqrt(np.prod([k[n] * px[n][a] for n in names if a in VIEW_AXES[n]]))
                     for a in AXES])
    g = known_m / size[AXES.index(known_axis)]
    return size * g, {n: k[n] * g for n in names}, mismatch


def _sample(crop: np.ndarray, n_col: int, n_row: int) -> np.ndarray:
    """Nearest-pixel resampling of a cropped silhouette onto (n_col, n_row) voxel columns
    (columns along the first axis, rows flipped so index 0 is the axis minimum)."""
    h, w = crop.shape
    ci = np.minimum(((np.arange(n_col) + 0.5) * w / n_col).astype(int), w - 1)
    ri = np.minimum(((np.arange(n_row) + 0.5) * h / n_row).astype(int), h - 1)
    return crop[::-1][ri][:, ci].T


def carve_ortho(views: List[OrthoView], known_axis: str = "x", known_m: float = 0.1,
                resolution: int = 128) -> Hull:
    """Intersection of the three extruded silhouettes; ``resolution`` voxels on the longest side.

    Each silhouette's bounding box is stretched onto the common size from :func:`fit_scales`
    (that registers the photos); ``size_mismatch`` reports how well the boxes agreed.
    """
    size, _, mismatch = fit_scales(views, known_axis, known_m)
    crops = _by_axis(views)
    voxel = float(size.max()) / int(resolution)
    n = np.maximum(1, np.round(size / voxel).astype(int))
    nx, ny, nz = n
    F = _sample(crops["front"], nx, nz)          # (x, z)
    T = _sample(crops["top"], nx, ny)            # (x, y)
    Sd = _sample(crops["side"], ny, nz)          # (y, z)
    occ = T[:, :, None] & F[:, None, :] & Sd[None, :, :]
    lab, k = ndi.label(occ)
    if k == 0:
        raise ValueError("no point is inside all three silhouettes")
    if k > 1:   # keep the part; report what was cut away
        sizes = np.bincount(lab.ravel())[1:]
        occ = lab == 1 + int(np.argmax(sizes))
        log.info("hull: dropped %d separate piece(s), %.1f %% of the volume",
                 k - 1, 100.0 * (sizes.sum() - sizes.max()) / sizes.sum())
    return Hull(occupied=occ, origin=np.full(3, 0.5 * voxel), voxel=voxel, size_mismatch=mismatch)


def surface(hull: Hull, smooth_voxels: float = 1.0, step: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Closed triangle surface (vertices in the hull frame, faces) by marching cubes.

    ``step`` > 1 samples every step-th voxel: fewer, larger triangles. Gmsh could not fill such
    surfaces at steps 3-4; for coarser meshes carve at a lower resolution instead.
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
    facet intersect"); element size is set by the carving resolution instead (see
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
def save_session(directory: Union[str, Path], views: List[OrthoView], known_axis: str,
                 known_m: float) -> Path:
    """Write masks (PNG) and ``ortho.json`` so ``mesh hull DIR`` can rebuild the solid."""
    from PIL import Image
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    meta = []
    for v in views:
        Image.fromarray(np.asarray(v.mask, bool).astype(np.uint8) * 255).save(d / f"{v.name}_mask.png")
        meta.append({"name": v.name, "axis": v.axis, "flip_h": bool(v.flip_h), "rot90": int(v.rot90)})
    (d / "ortho.json").write_text(json.dumps({"views": meta, "known_axis": known_axis,
                                              "known_m": float(known_m)}, indent=1))
    return d


def load_session(directory: Union[str, Path]):
    """(views, known_axis, known_m) saved by :func:`save_session`."""
    from PIL import Image
    d = Path(directory)
    meta = json.loads((d / "ortho.json").read_text())
    views = [OrthoView(m["name"], np.asarray(Image.open(d / f"{m['name']}_mask.png").convert("L")) > 127,
                       m["axis"], m.get("flip_h", False), m.get("rot90", 0)) for m in meta["views"]]
    return views, meta["known_axis"], float(meta["known_m"])


def load_hull_mesh(directory: Union[str, Path], mesh_size: Optional[float] = None,
                   resolution: int = 128):
    """Session directory -> carved hull -> tetrahedral mesh (the ``mesh hull`` directive)."""
    views, axis, known = load_session(directory)
    size, _, mismatch = fit_scales(views, axis, known)
    if mismatch > SIZE_MISMATCH_WARN:
        log.warning("the three photos disagree on the proportions by %.0f %%: check the masks",
                    100 * mismatch)
    # Element size ~ 2 voxels: carve at that resolution and surface every voxel. Marching cubes
    # with step_size > 1 on a fine grid was tried first: at steps 3-4 Gmsh generated no
    # tetrahedra inside its surface (60 x 40 x 30 mm box from the app, default size).
    target = float(mesh_size) if mesh_size and mesh_size > 0 else float(size.max()) / 15.0
    # Smoothing 0.5 voxel: measured on a 120 x 20 x 20 mm bar cantilever at the default size,
    # tip deflection +5 % vs beam theory (1 voxel rounded the section's edges: +11 %); the
    # rounded edges may then appear as extra small faces.
    res = int(np.clip(round(2.0 * float(size.max()) / target), 24, resolution))
    return mesh_surface(*surface(carve_ortho(views, axis, known, resolution=res), smooth_voxels=0.5))
