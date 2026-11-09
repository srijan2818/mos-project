"""Pipeline for converting 2D drawings or image sketches into 3D CAD STEP models.

Image -> Binarization -> Moore-Neighbor Contour Tracing -> RDP Simplification -> Gmsh OCC Extrusion -> STEP
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from .gmsh_session import GMSH_ERROR, GMSH_LOCK, HAS_GMSH, ensure_gmsh, gmsh


def _trace_moore_neighbor(binary: np.ndarray) -> np.ndarray:
    """Trace the outer boundary contour of a binary mask using Moore-Neighbor algorithm."""
    h, w = binary.shape
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    indices = np.argwhere(padded)
    if len(indices) == 0:
        return np.empty((0, 2), dtype=float)

    start_r, start_c = indices[0]
    # 8-neighborhood directions: (row, col) clockwise starting West
    dirs = [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)]

    contour = []
    curr = (int(start_r), int(start_c))
    # Flip Y so bottom-left is origin
    contour.append([float(curr[1] - 1), float((h - 1) - (curr[0] - 1))])
    backtrack = 0

    max_steps = max(2 * (w + h) * 4, 100000)
    for _ in range(max_steps):
        found = False
        start_dir = (backtrack + 1) % 8
        for i in range(8):
            d_idx = (start_dir + i) % 8
            nr, nc = curr[0] + dirs[d_idx][0], curr[1] + dirs[d_idx][1]
            if padded[nr, nc]:
                curr = (nr, nc)
                contour.append([float(curr[1] - 1), float((h - 1) - (curr[0] - 1))])
                backtrack = (d_idx + 4) % 8
                found = True
                break
        if not found or (curr == (start_r, start_c) and len(contour) > 2):
            break

    if len(contour) > 1 and contour[0] == contour[-1]:
        contour.pop()

    return np.asarray(contour, dtype=float)


def _rdp_simplify(pts: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer-Douglas-Peucker polyline simplification (iterative, vectorised distances).

    The recursive per-point Python loop it replaces was O(n^2) interpreted work and could
    hit the recursion limit on long contours from high-resolution masks.
    """
    pts = np.asarray(pts, dtype=float)
    n = len(pts)
    if n <= 3 or epsilon <= 0:
        return pts
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 - i0 < 2:
            continue
        a, b = pts[i0], pts[i1]
        seg = pts[i0 + 1:i1]
        line = b - a
        norm = float(np.hypot(line[0], line[1]))
        if norm > 1e-12:
            d = np.abs(line[0] * (a[1] - seg[:, 1]) - line[1] * (a[0] - seg[:, 0])) / norm
        else:
            d = np.hypot(seg[:, 0] - a[0], seg[:, 1] - a[1])
        k = int(np.argmax(d))
        if d[k] > epsilon:
            mid = i0 + 1 + k
            keep[mid] = True
            stack.append((i0, mid))
            stack.append((mid, i1))
    return pts[keep]


def polygon_area(poly: np.ndarray) -> float:
    """Unsigned shoelace area of a closed polygon given as (n, 2) vertices."""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def image_to_polygons(
    image_path: Union[str, Path],
    threshold: int = 128,
    invert: bool = False,
    simplify_tol: float = 3.0,
    min_area: float = 100.0,
) -> Tuple[List[np.ndarray], Tuple[int, int]]:
    """Extract ordered, simplified closed polygon contours from a 2D image."""
    img = Image.open(image_path).convert("L")
    w, h = img.size
    arr = np.array(img, dtype=np.uint8)

    binary = arr > threshold if invert else arr < threshold

    from scipy.ndimage import label
    labeled, num_features = label(binary)

    polygons = []
    for feat_idx in range(1, num_features + 1):
        feat_mask = (labeled == feat_idx)
        area = np.count_nonzero(feat_mask)
        if area < min_area:
            continue

        raw_contour = _trace_moore_neighbor(feat_mask)
        if len(raw_contour) >= 4:
            # Simplify closed loop
            simplified = _rdp_simplify(raw_contour, simplify_tol)
            if len(simplified) >= 3:
                # Ensure closed without exact duplicate end
                if np.linalg.norm(simplified[0] - simplified[-1]) < 1e-4:
                    simplified = simplified[:-1]
                if len(simplified) >= 3:
                    polygons.append(simplified)

    return polygons, (w, h)


def image_to_step(
    image_path: Union[str, Path],
    output_step_path: Union[str, Path],
    thickness: float = 0.1,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    simplify_tol: float = 3.0,
    threshold: int = 128,
) -> Path:
    """Convert a 2D drawing image profile into an extruded 3D solid STEP CAD model."""
    if not HAS_GMSH:
        raise RuntimeError("Gmsh is required for STEP export but could not be loaded"
                           + (f" ({GMSH_ERROR})" if GMSH_ERROR else "") + ".")

    from scipy.ndimage import label as _label
    from .image_to_mesh import mask_contours

    img = Image.open(image_path).convert("L")
    img_w, img_h = img.size
    binary = np.asarray(img, dtype=np.uint8) < threshold
    labeled, count = _label(binary)
    if count == 0:
        raise ValueError(f"No valid profiles detected in image '{image_path}'.")
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    # The largest part by area, with its interior holes (previously holes were dropped
    # and the "largest" outline was the one with the most vertices).
    try:
        outer, holes = mask_contours(labeled == int(np.argmax(sizes)), simplify_tol)
    except ValueError:
        raise ValueError(f"No valid profiles detected in image '{image_path}'.") from None

    out_p = Path(output_step_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with GMSH_LOCK:
        ensure_gmsh()
        gmsh.clear()
        try:
            gmsh.model.add("extruded_image")

            def add_loop(poly):
                """OCC curve loop through polygon vertices scaled to metres."""
                xs = (poly[:, 0] / max(img_w, 1)) * scale_x
                ys = (poly[:, 1] / max(img_h, 1)) * scale_y
                pts = [gmsh.model.occ.addPoint(float(x), float(y), 0.0) for x, y in zip(xs, ys)]
                lines = [gmsh.model.occ.addLine(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
                return gmsh.model.occ.addCurveLoop(lines)

            face = gmsh.model.occ.addPlaneSurface([add_loop(outer)] + [add_loop(h) for h in holes])
            gmsh.model.occ.synchronize()

            # Extrude along Z
            gmsh.model.occ.extrude([(2, face)], 0.0, 0.0, float(thickness))
            gmsh.model.occ.synchronize()

            gmsh.write(str(out_p))
            return out_p
        finally:
            gmsh.clear()
