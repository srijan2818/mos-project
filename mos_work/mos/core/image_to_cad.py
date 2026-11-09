"""Pipeline for converting 2D drawings or image sketches into 3D CAD STEP models.

Image -> Binarization -> Moore-Neighbor Contour Tracing -> RDP Simplification -> Gmsh OCC Extrusion -> STEP
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    import gmsh
    HAS_GMSH = True
    GMSH_ERROR = ""
except (ImportError, OSError) as _exc:   # OSError: gmsh is installed but its shared library won't load
    HAS_GMSH = False
    GMSH_ERROR = f"{type(_exc).__name__}: {_exc}"


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
    """Ramer-Douglas-Peucker polygon simplification."""
    if len(pts) <= 3 or epsilon <= 0:
        return pts

    dmax = 0.0
    index = 0
    end = len(pts) - 1
    line = pts[end] - pts[0]
    norm = float(np.linalg.norm(line))

    for i in range(1, end):
        if norm > 1e-12:
            d = abs(line[0] * (pts[0, 1] - pts[i, 1]) - line[1] * (pts[0, 0] - pts[i, 0])) / norm
        else:
            d = float(np.linalg.norm(pts[i] - pts[0]))
        if d > dmax:
            index = i
            dmax = d

    if dmax > epsilon:
        rec1 = _rdp_simplify(pts[:index + 1], epsilon)
        rec2 = _rdp_simplify(pts[index:], epsilon)
        return np.vstack((rec1[:-1], rec2))
    else:
        return np.array([pts[0], pts[end]])


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

    polygons, (img_w, img_h) = image_to_polygons(
        image_path,
        threshold=threshold,
        simplify_tol=simplify_tol,
    )

    if not polygons:
        raise ValueError(f"No valid profiles detected in image '{image_path}'.")

    # Select largest polygon
    poly = max(polygons, key=len)

    # Normalize coordinates to meters
    norm_x = (poly[:, 0] / max(img_w, 1)) * scale_x
    norm_y = (poly[:, 1] / max(img_h, 1)) * scale_y

    out_p = Path(output_step_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.clear()

    try:
        gmsh.model.add("extruded_image")

        n_pts = len(norm_x)
        occ_pts = [
            gmsh.model.occ.addPoint(float(norm_x[i]), float(norm_y[i]), 0.0)
            for i in range(n_pts)
        ]

        lines = [
            gmsh.model.occ.addLine(occ_pts[i], occ_pts[(i + 1) % n_pts])
            for i in range(n_pts)
        ]

        wire = gmsh.model.occ.addCurveLoop(lines)
        face = gmsh.model.occ.addPlaneSurface([wire])
        gmsh.model.occ.synchronize()

        # Extrude along Z
        gmsh.model.occ.extrude([(2, face)], 0.0, 0.0, float(thickness))
        gmsh.model.occ.synchronize()

        gmsh.write(str(out_p))
        return out_p
    finally:
        try:
            gmsh.clear()
        except Exception:
            pass