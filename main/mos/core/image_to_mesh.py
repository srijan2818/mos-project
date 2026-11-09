"""Direct image -> FE mesh: foreground mask -> outline + holes -> layered tetrahedral extrusion.

This is the robust single-photo geometry path. It meshes exactly the silhouette that was
segmented (including through-holes) instead of classifying the part into one of a few
templates and guessing template dimensions, and it never writes or re-imports a STEP file.

    mask (dark/alpha = material) --contours--> outer polygon + hole polygons (pixels)
        --scale--> metres --Gmsh built-in kernel, in memory--> 2D triangles
        --extrude in structured layers--> conforming tetrahedra --> scikit-fem MeshTet

Named boundaries (usable directly as support/load regions):
    cad_face_1   bottom (z = 0)          cad_face_2   top (z = thickness)
    cad_face_3.. outer side walls, one per roughly straight run of the outline
    then         one face per hole wall

A single view cannot reveal depth, so the part is assumed prismatic (constant thickness
normal to the photo). That is exact for plates, brackets photographed side-on and
cross-sections photographed end-on (I-beams, tubes, channels) once a length is entered.
"""
from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from .image_to_cad import _rdp_simplify, _trace_moore_neighbor, polygon_area

MAX_MASK_PIXELS = 1024         # longest mask edge used for contouring (RDP bounds the vertex count)
_CACHE: Dict[tuple, Any] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_SIZE = 6


# ----------------------------------------------------------------------------- masks
def load_mask_image(path: Union[str, Path]) -> np.ndarray:
    """Read a mask image as a bool array: True = material.

    Transparent pixels are background when the image has an alpha channel; otherwise
    dark pixels (< 128) are material, matching the convention of the drawing importer
    and of the masks written by single-image reconstruction.
    """
    image = Image.open(Path(path).expanduser())
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        alpha = np.asarray(image.convert("RGBA").getchannel("A"))
        if alpha.min() < 250:
            return alpha > 127
    return np.asarray(image.convert("L")) < 128


def _clean_mask(mask: np.ndarray, max_pixels: int = MAX_MASK_PIXELS) -> np.ndarray:
    """Downsample to a bounded size, keep the largest part, remove 1-px spurs and specks."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        raise ValueError("The mask has no material pixels.")
    ys, xs = np.nonzero(mask)
    pad = 2
    mask = mask[max(ys.min() - pad, 0):ys.max() + pad + 1, max(xs.min() - pad, 0):xs.max() + pad + 1]
    scale = min(1.0, max_pixels / max(mask.shape))
    if scale < 1.0:
        size = (max(8, int(round(mask.shape[1] * scale))), max(8, int(round(mask.shape[0] * scale))))
        small = Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.Resampling.BILINEAR)
        mask = np.asarray(small) > 127
    mask = np.pad(mask, 2)
    mask = ndi.binary_opening(mask, structure=np.ones((3, 3), bool))
    labels, count = ndi.label(mask)
    if count == 0:
        raise ValueError("The mask is too thin or fragmented to mesh.")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def _simplify_closed(contour: np.ndarray, epsilon: float) -> np.ndarray:
    """RDP for a closed loop: split at two far-apart vertices so neither end is degenerate."""
    if len(contour) < 8:
        return contour
    i0 = 0
    i1 = int(np.argmax(np.hypot(*(contour - contour[0]).T)))
    a = _rdp_simplify(contour[i0:i1 + 1], epsilon)
    b = _rdp_simplify(np.vstack([contour[i1:], contour[:1]]), epsilon)
    return np.vstack([a[:-1], b[:-1]])


def _offset_outward(poly: np.ndarray, distance: float) -> np.ndarray:
    """Offset a closed polygon away from its own interior by ``distance`` (mitred corners).

    The Moore tracer returns boundary *pixel centres*, half a pixel inside the region it
    traces. Without this correction a part loses one pixel per dimension (and each hole
    shrinks by one) -- for a 30-px-deep section that is a 10 % error in bending stiffness.
    """
    x, y = poly[:, 0], poly[:, 1]
    signed = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    edge = np.roll(poly, -1, axis=0) - poly
    length = np.hypot(edge[:, 0], edge[:, 1])
    length[length < 1e-12] = 1.0   # repeated vertex (pixel units: real edges are >= ~0.5)
    # Right-hand normal is outward for a counter-clockwise polygon.
    normal = np.column_stack([edge[:, 1], -edge[:, 0]]) / length[:, None]
    if signed < 0:
        normal = -normal
    prev = np.roll(normal, 1, axis=0)                  # normal of the edge ending at vertex i
    bisector = prev + normal
    denom = 1.0 + np.einsum("ij,ij->i", prev, normal)
    miter = bisector / np.maximum(denom, 1.0 / 3.0)[:, None]   # cap the miter at 3x
    return poly + distance * miter


def mask_contours(mask: np.ndarray, simplify_px: float = 1.0,
                  min_hole_px: int = 12) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Outer outline and hole outlines of a single-part mask, in pixel units with y up.

    Loops follow pixel edges (not centres), so a w x h pixel block has area w * h.
    """
    mask = np.asarray(mask, dtype=bool)
    labels, count = ndi.label(mask)
    if count > 1:   # trace the part, not whichever fragment comes first in raster order
        mask = labels == 1 + int(np.argmax(np.bincount(labels.ravel())[1:]))
    outer = _offset_outward(_simplify_closed(_trace_moore_neighbor(mask), simplify_px), 0.5)
    if len(outer) < 3 or polygon_area(outer) < 4:
        raise ValueError("Could not trace a usable outline from the mask.")
    holes_mask = ndi.binary_fill_holes(mask) & ~mask
    labels, count = ndi.label(holes_mask)
    holes = []
    for k in range(1, count + 1):
        hole = labels == k
        if hole.sum() < min_hole_px:
            continue
        loop = _offset_outward(_simplify_closed(_trace_moore_neighbor(hole), simplify_px), 0.5)
        if len(loop) >= 3 and polygon_area(loop) >= 4:
            holes.append(loop)
    return outer, holes


def _group_segments(poly: np.ndarray, angle_deg: float = 25.0) -> List[List[int]]:
    """Group consecutive polygon edges whose direction turns less than ``angle_deg``."""
    n = len(poly)
    d = np.roll(poly, -1, axis=0) - poly
    ang = np.arctan2(d[:, 1], d[:, 0])
    turn = np.abs((np.diff(np.r_[ang, ang[0]]) + np.pi) % (2 * np.pi) - np.pi)
    breaks = np.flatnonzero(np.degrees(turn) >= angle_deg)   # a corner after edge i
    if breaks.size == 0:
        return [list(range(n))]
    groups, start = [], (breaks[-1] + 1) % n
    current, corner = [], set(breaks.tolist())
    for step in range(n):
        i = (start + step) % n
        current.append(i)
        if i in corner:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


# ----------------------------------------------------------------------------- meshing
def mesh_from_mask(mask: np.ndarray, size_m: float, thickness_m: float,
                   mesh_size: Optional[float] = None,
                   max_elements: int = 12000) -> Tuple[Any, Dict[str, Any]]:
    """Mesh the largest part in ``mask`` as a prism of the given thickness.

    ``size_m`` is the real length of the part's largest in-plane extent (the scale
    reference); ``thickness_m`` is the extrusion depth normal to the image. Returns the
    scikit-fem ``MeshTet`` and an info dict (face names/roles, dimensions, counts).
    ``mesh_size`` (default ``size_m / 20``) is coarsened when the estimated tetrahedron
    count would exceed ``max_elements``; ``info["mesh_size"]`` reports the size used.
    """
    from .cad import CadImportError, gmsh_to_meshtet
    from .gmsh_session import GMSH_LOCK, ensure_gmsh, gmsh

    size_m, thickness_m = float(size_m), float(thickness_m)
    if not (math.isfinite(size_m) and size_m > 0 and math.isfinite(thickness_m) and thickness_m > 0):
        raise ValueError("Part size and thickness must be positive.")

    attempts = [(1.0, 1), (1.6, 2), (2.5, 3)]       # (RDP tolerance px, smoothing passes)
    last_error: Exception | None = None
    for simplify_px, smooth in attempts:
        clean = _clean_mask(mask)
        if smooth > 1:
            st = np.ones((3, 3), bool)
            clean = ndi.binary_opening(ndi.binary_closing(clean, st, iterations=smooth - 1), st)
        outer, holes = mask_contours(clean, simplify_px)
        lo = outer.min(axis=0)
        extent = float((outer.max(axis=0) - lo).max())
        if extent <= 0:
            raise ValueError("Degenerate outline.")
        s = size_m / extent
        outer_m = (outer - lo) * s
        holes_m = [(h - lo) * s for h in holes]
        h = float(mesh_size) if mesh_size and mesh_size > 0 else size_m / 20.0
        h = max(h, 2.0 * simplify_px * s)                  # no element smaller than the pixel noise
        # Element budget: a long extrusion (end-on photo of a beam) at an in-plane size
        # would otherwise create 10^5+ P2 tets and stall the direct solver. Tets scale
        # ~ h^-3 (triangles ~ area / h^2, layers ~ depth / h), so coarsen h uniformly.
        area = polygon_area(outer_m) - sum(polygon_area(p) for p in holes_m)
        requested_h = h
        estimate = 3.0 * 2.31 * area / h ** 2 * max(2, thickness_m / h)
        if estimate > max_elements:
            h *= (estimate / max_elements) ** (1.0 / 3.0)

        def build(h, layers):
            """Mesh the outline with in-plane size ``h`` and ``layers`` through the depth."""
            with GMSH_LOCK:
                ensure_gmsh()
                gmsh.clear()
                try:
                    gmsh.model.add("image_mesh")
                    geo = gmsh.model.geo

                    def add_loop(poly):
                        """Built-in-kernel curve loop through the polygon; returns (loop, lines)."""
                        pts = [geo.addPoint(float(x), float(y), 0.0, h) for x, y in poly]
                        lines = [geo.addLine(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
                        return geo.addCurveLoop(lines), lines

                    outer_loop, outer_lines = add_loop(outer_m)
                    hole_loops = [add_loop(p) for p in holes_m]
                    surface = geo.addPlaneSurface([outer_loop] + [lp for lp, _ in hole_loops])
                    out = geo.extrude([(2, surface)], 0.0, 0.0, thickness_m, numElements=[layers])
                    geo.synchronize()
                    top = out[0][1]
                    laterals = [tag for dim, tag in out[2:] if dim == 2]
                    n_outer = len(outer_lines)
                    groups: Dict[str, list] = {"cad_face_1": [surface], "cad_face_2": [top]}
                    roles = {"cad_face_1": "bottom (z = 0)", "cad_face_2": f"top (z = {thickness_m:.4g} m)"}
                    k = 3
                    for seg in _group_segments(outer_m):
                        groups[f"cad_face_{k}"] = [laterals[i] for i in seg]
                        roles[f"cad_face_{k}"] = "outer side wall"
                        k += 1
                    offset = n_outer
                    for j, (_, lines) in enumerate(hole_loops):
                        groups[f"cad_face_{k}"] = laterals[offset:offset + len(lines)]
                        roles[f"cad_face_{k}"] = f"hole {j + 1} wall"
                        offset += len(lines)
                        k += 1
                    gmsh.option.setNumber("Mesh.ElementOrder", 1)
                    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
                    gmsh.model.mesh.generate(3)
                    return gmsh_to_meshtet(groups), roles
                finally:
                    gmsh.clear()

        try:
            for _ in range(3):
                layers = max(2, int(round(thickness_m / h)))
                mesh, roles = build(h, layers)
                if mesh.nelements <= 1.25 * max_elements:
                    break
                # Thin features refine below h; correct with the real count and remesh.
                h *= (mesh.nelements / max_elements) ** (1.0 / 3.0)
        except Exception as exc:          # self-touching outline etc.: smooth more and retry
            last_error = exc
            continue
        width, height = (outer_m.max(axis=0) - outer_m.min(axis=0))
        info = {
            "faces": roles, "n_holes": len(holes_m),
            "dimensions": [float(width), float(height), thickness_m],
            "n_vertices": int(mesh.nvertices), "n_elements": int(mesh.nelements),
            "layers": layers, "mesh_size": h, "requested_mesh_size": requested_h,
            "outline_vertices": len(outer_m),
            "simplify_px": simplify_px,
        }
        return mesh, info
    raise CadImportError(f"Could not mesh the image outline: {last_error}")


def load_mask_mesh(path: Union[str, Path], size_m: float, thickness_m: float,
                   mesh_size: Optional[float] = None) -> Tuple[Any, Dict[str, Any]]:
    """``mesh_from_mask`` for a mask image file, cached on (path, mtime, parameters).

    The app re-builds the model on every solve; the cache makes that free for an
    unchanged mask and settings.
    """
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Mask image not found: {p}")
    key = (str(p), p.stat().st_mtime_ns, float(size_m), float(thickness_m),
           None if mesh_size is None else float(mesh_size))
    with _CACHE_LOCK:
        if key in _CACHE:
            return _CACHE[key]
    result = mesh_from_mask(load_mask_image(p), size_m, thickness_m, mesh_size)
    with _CACHE_LOCK:
        if len(_CACHE) >= _CACHE_SIZE:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = result
    return result


def save_mask_image(mask: np.ndarray, path: Union[str, Path]) -> Path:
    """Write a mask in the convention :func:`load_mask_image` reads (black = material)."""
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(np.asarray(mask, bool), 0, 255).astype(np.uint8), mode="L").save(p)
    return p
