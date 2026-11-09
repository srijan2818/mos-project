"""Synthetic photos of the marker mat with an object on it, from cameras with known poses.

Used by the multi-view tests: everything is rendered analytically (mat by homography, object
silhouette by projected polygons), so poses, scale and true geometry are exact.
"""
import numpy as np

W, H, F = 1280, 960, 1100.0
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])


def look_at(eye, target, up=(0, 0, 1)):
    """World->camera (R, t) for a camera at ``eye`` looking at ``target`` (OpenCV axes)."""
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-9:              # looking straight down: pick image-x = world x
        x = np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ eye


def three_views(centre, distance=0.45, elevation_deg=35.0):
    """Top, front and side cameras around ``centre`` (world, m), like the guided capture."""
    c = np.asarray(centre, float)
    e = np.radians(elevation_deg)
    top = look_at(c + [0.001, 0.001, distance], c, up=(0, 1, 0))
    front = look_at(c + distance * np.array([0, -np.cos(e), np.sin(e)]), c)
    side = look_at(c + distance * np.array([np.cos(e), 0, np.sin(e)]), c)
    return {"top": top, "front": front, "side": side}


def project(P, R, t):
    """World points (N, 3) -> pixels (N, 2)."""
    x = (R @ np.asarray(P, float).T).T + t
    uv = (K @ x.T).T
    return uv[:, :2] / uv[:, 2:3]


def render_mat(R, t, spec, background=200):
    """Grayscale photo of the printed mat (lying at z = 0) seen by camera (R, t)."""
    import cv2
    from mos.core.marker import board_image
    ppm = 8.0                                             # board image pixels per mm
    img = board_image(spec, px_per_mm=ppm)
    # board pixel (u, v) -> board metres (u/ppm/1000, v/ppm/1000, 0) -> world (x, -y, 0)
    S = np.array([[1 / (ppm * 1000), 0, 0], [0, -1 / (ppm * 1000), 0], [0, 0, 1]])
    Hm = K @ np.column_stack([R[:, 0], R[:, 1], t]) @ S   # homography: board px -> photo px
    return cv2.warpPerspective(img, Hm, (W, H), borderValue=background, flags=cv2.INTER_AREA)


def box_corners(lo, hi):
    """The 8 corners of an axis-aligned box."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])


def silhouette(faces, R, t):
    """Bool mask of the union of projected convex polygons (each a list of world points)."""
    from PIL import Image, ImageDraw
    from scipy.spatial import ConvexHull
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    for pts in faces:
        uv = project(pts, R, t)
        hull = uv[ConvexHull(uv).vertices]
        d.polygon([tuple(p) for p in hull], fill=255)
    return np.asarray(img) > 0


def box_mask(lo, hi, R, t):
    """Silhouette of a convex box: the convex hull of its projected corners."""
    return silhouette([box_corners(lo, hi)], R, t)


# ------------------------------------------------------------------ L-mat (floor + back sheet)
def back_sheet_pose(spec, y=0.004, z_bottom=0.004):
    """4x4 back-sheet -> world: upright along the floor sheet's far edge, facing -y."""
    T = np.eye(4)
    T[:3, :3] = np.column_stack([[1, 0, 0], [0, 0, 1], [0, -1, 0]])   # local x, y(up page), z(out)
    T[:3, 3] = [0.0, y, z_bottom + spec.size_m[1]]                     # printed top edge at the top
    return T


def render_board(img, R, t, spec, T_wb=None):
    """Draw one sheet (pose T_wb, default = floor) into grayscale ``img`` as seen by (R, t)."""
    import cv2
    from mos.core.marker import board_image
    T_wb = np.eye(4) if T_wb is None else T_wb
    ppm = 8.0
    board = board_image(spec, px_per_mm=ppm)
    S = np.array([[1 / (ppm * 1000), 0, 0], [0, -1 / (ppm * 1000), 0], [0, 0, 1]])
    Rw, tw = T_wb[:3, :3], T_wb[:3, 3]
    Hm = K @ np.column_stack([R @ Rw[:, 0], R @ Rw[:, 1], R @ tw + t]) @ S
    # only draw a sheet whose front (printed) side faces the camera and lies in front of it
    if (R @ Rw[:, 2])[2] >= 0:
        return img
    warped = cv2.warpPerspective(board, Hm, (W, H), flags=cv2.INTER_AREA)
    cover = cv2.warpPerspective(np.ones_like(board), Hm, (W, H), flags=cv2.INTER_NEAREST) > 0
    img[cover] = warped[cover]
    return img


def render_lmat(R, t, floor, back, T_back):
    """Photo of the L-mat: floor sheet at z = 0 plus the standing back sheet."""
    img = np.full((H, W), 200, np.uint8)
    render_board(img, R, t, floor)
    return render_board(img, R, t, back, T_back)


def lmat_views(centre, distance=0.45):
    """The guided L-mat capture: top, front at table level (turned ~20 deg: a sheet seen exactly
    head-on gives the least certain pose), side at 20 deg elevation, 3/4 link view."""
    c = np.asarray(centre, float)

    def at(az, el):
        a, e = np.radians(az), np.radians(el)
        return look_at(c + distance * np.array([np.cos(e) * np.sin(a), -np.cos(e) * np.cos(a), np.sin(e)]), c)
    return {"top": look_at(c + [0.001, 0.001, distance], c, up=(0, 1, 0)),
            "front": at(20, 2.0), "side": at(90, 20), "link": at(40, 32)}
