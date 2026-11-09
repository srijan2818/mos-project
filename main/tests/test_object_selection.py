"""Choosing *which* object in a photo is the part (clicks, candidate choice, automatic guess).

Regression for a real photo: a part held in a hand in front of a whiteboard was ignored and the
whiteboard was meshed. The SAM-backed tests are skipped when the model can't be loaded.
"""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from mos.core import cv_detector as cd

SKIN = (205, 150, 125)   # YCrCb Cr ~152, Cb ~110: inside the skin range


def _scene(size=(640, 480)):
    """Whiteboard touching two borders, a hand from the right edge, a blue part in the middle."""
    img = Image.new("RGB", size, (190, 185, 175))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, 250, 330), fill=(250, 250, 250))             # whiteboard (top-left)
    d.rectangle((420, 230, 639, 330), fill=SKIN)                     # hand from the right
    d.rectangle((280, 190, 380, 290), fill=(40, 90, 200))            # the part
    truth = np.zeros((size[1], size[0]), bool)
    truth[190:291, 280:381] = True
    return img, truth


def _iou(a, b):
    return (a & b).sum() / max(1, (a | b).sum())


def _sam_or_skip():
    if not cd.HAS_FOUNDATION:
        pytest.skip("torch/transformers SAM not installed")
    try:
        cd._get_segmenter()
    except cd.CVModelUnavailableError as exc:
        pytest.skip(f"SAM weights unavailable: {exc}")


def test_default_candidate_prefers_the_whole_part_over_background():
    h, w = 100, 100
    face = np.zeros((h, w), bool); face[40:50, 40:50] = True          # one face of the part
    part = np.zeros((h, w), bool); part[30:60, 30:60] = True          # the whole part
    wall = np.zeros((h, w), bool); wall[:, :70] = True                # background touching 3 sides
    assert cd.default_candidate([face, part, wall], [0.9, 0.8, 0.95]) == 1
    junk = np.zeros((h, w), bool); junk[5:95:7, 5:95] = True          # scattered, low score
    assert cd.default_candidate([face, part, junk], [0.96, 0.99, 0.40]) == 1
    # only background-like candidates: fall back to SAM's best score
    assert cd.default_candidate([wall, wall | part], [0.2, 0.7]) == 1


def test_border_and_skin_measures():
    m = np.zeros((10, 10), bool); m[0, 5:] = True; m[:5, -1] = True     # top + right only
    assert cd._touched_sides(m) == 2
    rgb = np.zeros((10, 10, 3), np.uint8); rgb[:5] = SKIN; rgb[5:] = (40, 90, 200)
    assert cd._skin_fraction(rgb, np.ones((10, 10), bool)) == pytest.approx(0.5)


def test_classical_click_selects_the_clicked_component():
    img = Image.new("RGB", (300, 200), (230, 230, 230))
    d = ImageDraw.Draw(img)
    d.rectangle((30, 60, 90, 140), fill=(30, 30, 30))
    d.rectangle((180, 50, 260, 150), fill=(30, 30, 30))
    (mask,) = cd._classical_point_masks(img, [(220, 100)], [1])
    assert mask[100, 220] and not mask[100, 60]
    with pytest.raises(ValueError):
        cd._classical_point_masks(img, [(150, 20)], [1])               # background click


def test_clicks_are_required_to_be_positive():
    with pytest.raises(ValueError):
        cd.segment_from_points(Image.new("RGB", (100, 100)), [(50, 50)], [0])


def test_automatic_guess_picks_the_part_not_whiteboard_or_hand():
    _sam_or_skip()
    img, truth = _scene()
    x, y = cd.suggest_object_point(img)
    assert truth[int(y), int(x)], (x, y)
    res = cd.analyze_image_object(img, reference_scale=0.1)
    assert _iou(res.mask, truth) > 0.85
    assert res.prompt_points and res.n_candidates >= 1


def test_click_selects_that_object_and_negative_click_excludes():
    _sam_or_skip()
    img, truth = _scene()
    res = cd.analyze_image_object(img, reference_scale=0.1, points=[(330, 240)], labels=[1])
    assert _iou(res.mask, truth) > 0.85
    assert res.prompt_points == [(330.0, 240.0, 1)]
    other = cd.analyze_image_object(img, reference_scale=0.1, points=[(120, 150)], labels=[1])
    assert _iou(other.mask, truth) < 0.1                              # the user chose the board


def test_marker_sheets_are_never_chosen_as_the_part():
    """On the L-mat the sheets are the most prominent 'objects'; their projected outlines
    (known from the pose) must keep the selection on the part."""
    _sam_or_skip()
    pytest.importorskip("cv2")
    from mos.core.marker import BACK, FLOOR, sheet_mask, solve_rig
    from tests import synthetic_views as sv
    lo, hi = np.array([0.105, -0.11, 0.0]), np.array([0.165, -0.07, 0.03])
    cams = list(sv.lmat_views((lo + hi) / 2).values())
    imgs, truths = [], []
    for R, t in cams:
        g = sv.render_lmat(R, t, FLOOR, BACK, sv.back_sheet_pose(BACK))
        m = sv.box_mask(lo, hi, R, t)
        rgb = np.stack([g] * 3, -1).astype(np.uint8)
        rgb[m] = (40, 90, 200)
        imgs.append(rgb)
        truths.append(m)
    rig = solve_rig(imgs)
    k = 1                                                        # the table-level front photo
    ex = [sheet_mask(rig.poses[k], FLOOR), sheet_mask(rig.poses[k], BACK, rig.back_to_world)]
    without = cd.analyze_image_object(imgs[k], 0.1)
    with_ex = cd.analyze_image_object(imgs[k], 0.1, exclude=ex)
    assert _iou(without.mask, truths[k]) < 0.5                    # the sheets win without help
    assert _iou(with_ex.mask, truths[k]) > 0.9


def test_other_views_fix_a_wrong_top_view_selection():
    """From above, SAM's larger mask merges the part with adjacent mat squares; the solid from
    the other photos, projected into this one, picks the right mask and prompt."""
    _sam_or_skip()
    pytest.importorskip("cv2")
    pytest.importorskip("skimage")
    from mos.core.marker import BACK, FLOOR, sheet_mask, solve_rig
    from mos.core.multiview_hull import HullView, carve, project_hull
    from tests import synthetic_views as sv
    lo, hi = np.array([0.105, -0.11, 0.0]), np.array([0.165, -0.07, 0.03])
    cams = list(sv.lmat_views((lo + hi) / 2).values())
    imgs, truths = [], []
    for R, t in cams:
        g = sv.render_lmat(R, t, FLOOR, BACK, sv.back_sheet_pose(BACK))
        m = sv.box_mask(lo, hi, R, t)
        rgb = np.stack([g] * 3, -1).astype(np.uint8)
        rgb[m] = (40, 90, 200)
        imgs.append(rgb)
        truths.append(m)
    rig = solve_rig(imgs)
    ex = [sheet_mask(rig.poses[0], FLOOR), sheet_mask(rig.poses[0], BACK, rig.back_to_world)]
    prior = project_hull(carve([HullView(str(j), truths[j], rig.poses[j]) for j in (1, 2, 3)],
                               resolution=48), rig.poses[0])
    res = cd.analyze_image_object(imgs[0], 0.1, exclude=ex, prior=prior)
    assert _iou(res.mask, truths[0]) > 0.9
