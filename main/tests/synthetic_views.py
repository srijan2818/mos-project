"""Synthetic silhouettes of boxes (and unions of boxes) for the three-photo hull tests.

Three views as the app asks for them (front, top, right side), rendered either
orthographically (exact for the hull's model) or with a pinhole camera at a finite distance
(what a phone does), so the hull's perspective error can be measured against true geometry.
"""
import numpy as np

W, H = 1280, 960

# per view: (world axis of image columns, world axis of image rows (down = decreasing),
#            viewing axis, side the camera sits on: +1 = on the + side of the axis)
VIEWS = {"front": (0, 2, 1, -1), "top": (0, 1, 2, +1), "side": (1, 2, 0, +1)}


def box_corners(lo, hi):
    """The 8 corners of an axis-aligned box."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])


def _pixels(P, view, centre, px_per_m, distance=None):
    """Image coordinates (N, 2) of world points; orthographic when ``distance`` is None."""
    cu, cr, ax, side = VIEWS[view]
    P = np.asarray(P, float) - np.asarray(centre, float)
    if distance is None:
        mag = np.ones(len(P))
    else:     # pinhole at `distance` from the centre; px_per_m holds at the centre's depth
        mag = distance / (distance - side * P[:, ax])
    u = W / 2 + px_per_m * mag * P[:, cu]
    v = H / 2 - px_per_m * mag * P[:, cr]
    return np.column_stack([u, v])


def box_mask(boxes, view, centre, px_per_m=4000.0, distance=None):
    """(H, W) silhouette of a union of boxes [(lo, hi), ...] seen in ``view``."""
    from scipy.spatial import ConvexHull
    from skimage.draw import polygon
    m = np.zeros((H, W), bool)
    for lo, hi in boxes:
        uv = _pixels(box_corners(lo, hi), view, centre, px_per_m, distance)
        hull = uv[ConvexHull(uv).vertices]
        rr, cc = polygon(hull[:, 1], hull[:, 0], shape=m.shape)
        m[rr, cc] = True
    return m


def three_masks(boxes, centre, px_per_m=(4000.0, 4000.0, 4000.0), distance=None):
    """{"front": mask, "top": mask, "side": mask} of a union of boxes."""
    return {v: box_mask(boxes, v, centre, s, distance) for v, s in zip(VIEWS, px_per_m)}
