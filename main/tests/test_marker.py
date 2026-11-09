"""Marker mat: pose and focal length recovered from synthetic photos with known cameras."""
import numpy as np
import pytest

pytest.importorskip("cv2")

from mos.core import marker
from tests import synthetic_views as sv

SPEC = marker.BoardSpec()
CENTRE = (0.135, -0.09, 0.0)          # middle of the 270 x 180 mm mat


def test_poses_and_focal_from_three_views():
    cams = sv.three_views(CENTRE)
    images = [sv.render_mat(R, t, SPEC) for R, t in cams.values()]
    poses = marker.poses_from_images(images, SPEC)
    assert poses[0].K[0, 0] == pytest.approx(sv.F, rel=0.02)
    for (R, t), pose in zip(cams.values(), poses):
        assert np.linalg.norm(pose.centre - (-R.T @ t)) < 0.003       # camera within 3 mm
        angle = np.degrees(np.arccos(np.clip((np.trace(pose.R @ R.T) - 1) / 2, -1, 1)))
        assert angle < 0.5
        assert pose.rms_px < 1.0
    assert poses[0].elevation_deg() == pytest.approx(90, abs=1)        # top view
    assert poses[1].elevation_deg() == pytest.approx(35, abs=1)


def test_mat_not_visible_is_an_error():
    with pytest.raises(ValueError, match="marker mat"):
        marker.detect(np.full((480, 640), 200, np.uint8), SPEC)


def test_pdf_is_written(tmp_path):
    path = marker.board_pdf(tmp_path / "mat.pdf", SPEC)
    assert path.read_bytes()[:4] == b"%PDF"


def test_l_mat_rig_locates_back_sheet_and_all_four_views():
    from mos.core.marker import BACK, FLOOR, solve_rig
    T_back = sv.back_sheet_pose(BACK)
    cams = sv.lmat_views(CENTRE)
    rig = solve_rig([sv.render_lmat(R, t, FLOOR, BACK, T_back) for R, t in cams.values()])
    assert rig.back_to_world is not None, rig.messages
    assert np.abs(rig.back_to_world[:3, 3] - T_back[:3, 3]).max() < 0.003
    angle = np.degrees(np.arccos(np.clip((np.trace(rig.back_to_world[:3, :3] @ T_back[:3, :3].T) - 1) / 2, -1, 1)))
    assert angle < 0.5
    # What carving needs is where rays cross the *part*: a face-on planar target lets a small
    # camera shift trade off against a small tilt, which barely moves rays near the mat.
    part = sv.box_corners([0.105, -0.11, 0.0], [0.165, -0.07, 0.03])
    for name, (R, t), pose in zip(cams, cams.values(), rig.poses):
        assert pose is not None, (name, rig.messages)
        est = (pose.K @ (part @ pose.R.T + pose.t).T).T
        err = np.linalg.norm(est[:, :2] / est[:, 2:] - sv.project(part, R, t), axis=1).max()
        assert err < 2.0, (name, err)                 # 1 px ~ 0.4 mm on the part here
    assert rig.poses[1].elevation_deg() < 5          # the front photo really is at table level


def test_back_sheet_without_a_linking_photo_is_reported():
    from mos.core.marker import BACK, FLOOR, solve_rig
    T_back = sv.back_sheet_pose(BACK)
    cams = sv.lmat_views(CENTRE)
    imgs = [sv.render_lmat(*cams[k], FLOOR, BACK, T_back) for k in ("top", "front", "side")]
    rig = solve_rig(imgs)
    assert rig.poses[1] is None and any("both sheets" in m for m in rig.messages)
