"""Printed ChArUco marker mat: camera pose and metric scale for multi-view reconstruction.

The part stands on the printed mat. Every photo that sees enough of the mat gets its camera
pose in a common world frame, in metres, so no structure-from-motion and no "assumed size"
are needed.

World frame: origin at the mat's printed top-left corner. OpenCV's board frame has x to the
right, y down the page and z into the table; the world frame flips y and z so that ``z``
points **up out of the table** (x right, y up the page): ``P_world = F @ P_board`` with
``F = diag(1, -1, -1)``. The mat therefore lies in the plane z = 0 with y <= 0.

Accuracy depends on printing at 100 % ("actual size"); measure one square after printing and
pass the measured size as ``square_m`` if it differs.

**L-mat.** A flat mat alone only allows downward-looking photos (below ~15 deg elevation the mat
is no longer detected), and then the part's height is badly over-estimated (+40 % on a test box
at 20 deg). A second sheet standing behind the part (``BACK``, its own marker ids) is seen
face-on by a camera at table level. Its pose relative to the floor sheet is estimated from
any photo that sees both (:func:`solve_rig`), so the two sheets need no precise placement.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

log = logging.getLogger(__name__)

_FLIP = np.diag([1.0, -1.0, -1.0])     # board frame <-> world frame (its own inverse)


@dataclass(frozen=True)
class BoardSpec:
    """Geometry of the printed mat (defaults fit A4 landscape with ~14 mm margins)."""
    squares: Tuple[int, int] = (9, 6)
    square_m: float = 0.030
    marker_m: float = 0.022
    dictionary: str = "DICT_5X5_100"
    first_id: int = 0               # marker ids first_id .. first_id + n_markers - 1
    label: str = "FLOOR"

    @property
    def n_markers(self) -> int:
        """ArUco markers on the board (the white squares of the chessboard)."""
        return self.squares[0] * self.squares[1] // 2

    @property
    def size_m(self) -> Tuple[float, float]:
        """Printed board width and height in metres."""
        return self.squares[0] * self.square_m, self.squares[1] * self.square_m


@dataclass
class CameraPose:
    """Pinhole camera seeing the mat: ``x_cam = R @ X_world + t``, pixels ``~ K @ x_cam``."""
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    image_size: Tuple[int, int]     # (width, height)
    corners: int = 0                # ChArUco corners used
    rms_px: float = float("nan")    # reprojection RMS of those corners

    @property
    def centre(self) -> np.ndarray:
        """Camera position in world coordinates (m)."""
        return -self.R.T @ self.t

    def elevation_deg(self) -> float:
        """Angle of the viewing direction below the table plane (90 = straight down)."""
        view = self.R.T @ np.array([0.0, 0.0, 1.0])      # camera axis in world
        return float(np.degrees(np.arcsin(np.clip(-view[2], -1.0, 1.0))))


def _cv2():
    import cv2   # optional dependency (opencv-contrib-python); only the multi-view path needs it
    return cv2


FLOOR = BoardSpec()
BACK = BoardSpec(first_id=27, label="BACK")


def make_board(spec: BoardSpec = BoardSpec()):
    """The OpenCV ChArUco board for ``spec``."""
    cv2 = _cv2()
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec.dictionary))
    ids = np.arange(spec.first_id, spec.first_id + spec.n_markers)
    return cv2.aruco.CharucoBoard(spec.squares, spec.square_m, spec.marker_m, dictionary, ids)


def board_image(spec: BoardSpec = BoardSpec(), px_per_mm: float = 300 / 25.4) -> np.ndarray:
    """The mat as a grayscale image covering exactly the board (no margin)."""
    w_m, h_m = spec.size_m
    size = (int(round(w_m * 1000 * px_per_mm)), int(round(h_m * 1000 * px_per_mm)))
    return make_board(spec).generateImage(size, marginSize=0)


def _sheet(spec: BoardSpec, px_per_mm: float, note: str):
    from PIL import Image, ImageDraw
    page = Image.new("L", (int(round(297 * px_per_mm)), int(round(210 * px_per_mm))), 255)
    board = Image.fromarray(board_image(spec, px_per_mm))
    x0, y0 = (page.width - board.width) // 2, (page.height - board.height) // 2
    page.paste(board, (x0, y0))
    ImageDraw.Draw(page).text((x0, max(10, y0 - 40)), note, fill=0)
    return page


def board_pdf(path: Union[str, Path], specs=(FLOOR, BACK)) -> Path:
    """A4-landscape PDF, one page per sheet, 300 dpi. Print at 100 % / actual size."""
    dpi = 300
    specs = (specs,) if isinstance(specs, BoardSpec) else tuple(specs)
    pages = [_sheet(sp, dpi / 25.4,
                    f"MoS {sp.label} sheet: {sp.squares[0]} x {sp.squares[1]} squares of "
                    f"{sp.square_m * 1000:.0f} mm. Print at 100 % (actual size) and check one square "
                    "with a ruler. FLOOR lies flat under the part; BACK stands upright behind it.")
             for sp in specs]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(path, "PDF", resolution=dpi, save_all=True, append_images=pages[1:])
    return path


def detect(image: np.ndarray, spec: BoardSpec = BoardSpec(), min_corners: int = 8
           ) -> Tuple[np.ndarray, np.ndarray]:
    """ChArUco corners in an RGB/gray image: (world points (N, 3) in m, image points (N, 2)).

    Raises ValueError when fewer than ``min_corners`` corners are found (mat not visible
    enough for a pose).
    """
    cv2 = _cv2()
    gray = image if image.ndim == 2 else cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2GRAY)
    board = make_board(spec)
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray)
    n = 0 if ids is None else len(ids)
    if n < min_corners:
        raise ValueError(f"marker mat not found well enough ({n} corners, need {min_corners}); "
                         "keep more of the mat in view and in focus")
    board_pts = np.asarray(board.getChessboardCorners(), float)[ids.ravel()]
    return board_pts @ _FLIP.T, corners.reshape(-1, 2).astype(float)


def _focal_guess(image_size) -> float:
    """Typical webcam/phone field of view (~64 deg horizontal on the longer side)."""
    return 0.8 * max(image_size)


def calibrate_focal(detections: List[Tuple[np.ndarray, np.ndarray]], image_size) -> np.ndarray:
    """Intrinsics from the views themselves: one focal length, principal point at the centre,
    no lens distortion. Few, mostly oblique views constrain these 1-2 numbers well; the full
    distortion model would be over-fitted. Falls back to a typical field of view if the
    estimate is implausible (e.g. only fronto-parallel views, where f and distance trade off).
    """
    cv2 = _cv2()
    w, h = image_size
    K0 = np.array([[_focal_guess(image_size), 0, w / 2.0], [0, _focal_guess(image_size), h / 2.0],
                   [0, 0, 1.0]])
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
             | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
    obj = [o.astype(np.float32) for o, _ in detections]
    img = [i.astype(np.float32) for _, i in detections]
    try:
        rms, K, _, _, _ = cv2.calibrateCamera(obj, img, (int(w), int(h)), K0.copy(), np.zeros(5), flags=flags)
    except cv2.error as exc:
        log.warning("focal calibration failed (%s); using a typical field of view", exc)
        return K0
    f = float(K[0, 0])
    if not (0.3 * max(w, h) < f < 5.0 * max(w, h)) or not np.isfinite(rms):
        log.warning("implausible focal length %.0f px (rms %.2f); using a typical field of view", f, rms)
        return K0
    return K


def solve_pose(world: np.ndarray, image_pts: np.ndarray, K: np.ndarray, image_size) -> CameraPose:
    """Camera pose from the mat corners (iterative PnP; planar target)."""
    cv2 = _cv2()
    ok, rvec, tvec = cv2.solvePnP(world.astype(np.float64), image_pts.astype(np.float64), K, None,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError("could not compute the camera pose from the marker mat")
    R = cv2.Rodrigues(rvec)[0]
    proj, _ = cv2.projectPoints(world, rvec, tvec, K, None)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - image_pts) ** 2, axis=1))))
    return CameraPose(K=K, R=R, t=tvec.ravel(), image_size=tuple(image_size),
                      corners=len(world), rms_px=rms)


def _rt(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


@dataclass
class Rig:
    """Result of :func:`solve_rig`: a pose per photo (None if no sheet seen) and the back sheet."""
    poses: List[Optional[CameraPose]]
    back_to_world: Optional[np.ndarray]     # 4x4, BACK sheet frame -> world (FLOOR) frame
    messages: List[str]


def solve_rig(images: List[np.ndarray], floor: BoardSpec = FLOOR, back: BoardSpec = BACK) -> Rig:
    """Poses of all photos on an L-mat (floor sheet + standing back sheet).

    1. Detect each sheet in each photo; one shared focal length from all detections.
    2. The back sheet's pose in the world: from the photo that sees both sheets with the most
       corners (a photo must show both, e.g. a 3/4 view from above).
    3. Each photo's pose from the corners of every sheet it sees, in world coordinates.
    """
    cv2 = _cv2()
    sizes = {(im.shape[1], im.shape[0]) for im in images}
    if len(sizes) != 1:
        raise ValueError(f"all views must come from one camera at one resolution (got {sorted(sizes)})")
    size = sizes.pop()
    dets, msgs = [], []
    for i, im in enumerate(images):
        d = {}
        for sp in (floor, back):
            try:
                d[sp.label] = detect(im, sp)
            except ValueError:
                pass
        dets.append(d)
    singles = [pair for d in dets for pair in d.values()]
    if not singles:
        raise ValueError("no photo shows the marker mat")
    K = calibrate_focal(singles, size)
    local = [{k: solve_pose(w, p, K, size) for k, (w, p) in d.items()} for d in dets]
    both = [i for i, d in enumerate(dets) if floor.label in d and back.label in d]
    back_to_world = None
    if both:
        i = max(both, key=lambda j: min(len(dets[j][floor.label][0]), len(dets[j][back.label][0])))
        f, b = local[i][floor.label], local[i][back.label]
        back_to_world = np.linalg.inv(_rt(f.R, f.t)) @ _rt(b.R, b.t)
        msgs.append(f"back sheet located from photo {i + 1}")
    elif any(back.label in d for d in dets):
        msgs.append("no photo shows both sheets: views that only see the back sheet are skipped; "
                    "add a 3/4 view from above that shows both")
    if back_to_world is not None:
        K, back_to_world = _refine(dets, local, K, back_to_world, floor.label, back.label, size)
        msgs.append("poses refined jointly (all photos, both sheets)")
    poses: List[Optional[CameraPose]] = []
    for i, d in enumerate(dets):
        world, img = [], []
        if floor.label in d:
            world.append(d[floor.label][0]); img.append(d[floor.label][1])
        if back.label in d and back_to_world is not None:
            w = d[back.label][0] @ back_to_world[:3, :3].T + back_to_world[:3, 3]
            world.append(w); img.append(d[back.label][1])
        if not world:
            poses.append(None)
            msgs.append(f"photo {i + 1}: marker mat not usable; it is left out")
            continue
        poses.append(solve_pose(np.vstack(world), np.vstack(img), K, size))
    return Rig(poses, back_to_world, msgs)


def _refine(dets, local, K, back_to_world, floor_label, back_label, size):
    """Joint least-squares refinement (bundle adjustment of a planar rig).

    Unknowns: focal length, the back sheet's pose in the world and every photo's pose.
    Residuals: reprojection errors of every detected corner of either sheet. A single oblique
    photo fixes the back sheet only roughly; every photo that sees the back sheet adds
    constraints. Returns (K, back_to_world).
    """
    cv2 = _cv2()
    from scipy.optimize import least_squares
    views = [i for i, d in enumerate(dets) if d]
    cx, cy = K[0, 2], K[1, 2]

    def pose0(i):
        """Initial world->camera pose of photo i (via the back sheet if the floor isn't seen)."""
        p = local[i].get(floor_label)
        if p is not None:
            return p.R, p.t
        b = local[i][back_label]                  # camera <- back; camera <- world = that @ inv(T)
        T = _rt(b.R, b.t) @ np.linalg.inv(back_to_world)
        return T[:3, :3], T[:3, 3]
    x0 = [np.log(K[0, 0])]
    x0 += list(cv2.Rodrigues(back_to_world[:3, :3])[0].ravel()) + list(back_to_world[:3, 3])
    for i in views:
        R, t = pose0(i)
        x0 += list(cv2.Rodrigues(R)[0].ravel()) + list(t)
    x0 = np.array(x0)

    def unpack(x):
        """Parameter vector -> focal, back-sheet rotation/translation, per-photo poses."""
        f = np.exp(x[0])
        Rb = cv2.Rodrigues(x[1:4])[0]
        return f, Rb, x[4:7], x[7:].reshape(-1, 6)

    def residuals(x):
        """Reprojection errors (px) of every detected corner of either sheet."""
        f, Rb, tb, cams = unpack(x)
        Kx = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        out = []
        for c, i in zip(cams, views):
            for label, (w, uv) in dets[i].items():
                pts = w if label == floor_label else w @ Rb.T + tb
                proj, _ = cv2.projectPoints(pts, c[:3], c[3:], Kx, None)
                out.append((proj.reshape(-1, 2) - uv).ravel())
        return np.concatenate(out)
    sol = least_squares(residuals, x0, loss="huber", f_scale=2.0)
    f, Rb, tb, _ = unpack(sol.x)
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    return K, _rt(Rb, tb)


def sheet_mask(pose: CameraPose, spec: BoardSpec = FLOOR, sheet_to_world: Optional[np.ndarray] = None
               ) -> np.ndarray:
    """(H, W) bool: where the sheet ``spec`` (placed by ``sheet_to_world``, default the floor)
    appears in this photo. Used to stop the part selection from picking the mat itself."""
    from PIL import Image, ImageDraw
    w, h = spec.size_m
    corners = np.array([[0, 0, 0], [w, 0, 0], [w, -h, 0], [0, -h, 0]], float)
    if sheet_to_world is not None:
        corners = corners @ sheet_to_world[:3, :3].T + sheet_to_world[:3, 3]
    x = corners @ pose.R.T + pose.t
    img = Image.new("L", tuple(pose.image_size), 0)
    if np.all(x[:, 2] > 1e-6):
        uv = (x @ pose.K.T)[:, :2] / x[:, 2:3]
        ImageDraw.Draw(img).polygon([tuple(q) for q in uv], fill=255)
    return np.asarray(img) > 0


def poses_from_images(images: List[np.ndarray], spec: BoardSpec = BoardSpec()) -> List[CameraPose]:
    """Detect the mat in every image, estimate one shared focal length, return each pose.

    All images must come from the same camera at the same zoom (one session).
    """
    sizes = {(im.shape[1], im.shape[0]) for im in images}
    if len(sizes) != 1:
        raise ValueError(f"all views must have the same resolution (got {sorted(sizes)})")
    size = sizes.pop()
    dets = [detect(im, spec) for im in images]
    K = calibrate_focal(dets, size)
    return [solve_pose(w, p, K, size) for w, p in dets]
