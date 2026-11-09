"""Real-image CV detector for mechanical object parameterization.

Uses pretrained vision foundation models for real-world robustness:

  1. Grounding DINO  -- open-vocabulary detection (text-prompted, zero-shot)
  2. SAM 2 (or SAM)  -- promptable segmentation (box prompt -> clean mask)
  3. Geometry        -- classical parameter extraction on the CLEAN mask

Classical routines (Moore-neighbour tracing, RDP, RANSAC circle fit, PCA)
are retained but now operate on pixel-clean SAM masks rather than raw noisy
photographs, which is where they are reliable. This is the standard
industrial paradigm: learn the hard part (segmentation) on massive real-
world data, then apply geometry to the result.

When `transformers` + `torch` are unavailable (offline / CI), the module
falls back to a multi-cue geometric baseline so tests and downstream code
remain functional. For production real-image use, install:

    pip install transformers torch pillow scipy

Model weights are downloaded on first use and cached by transformers.
Override the default model IDs via the env vars ``MOS_DETECTOR_MODEL``,
``MOS_SAM2_MODEL``, ``MOS_SAM_MODEL`` (e.g. use ``sam2.1-hiera-tiny`` for
faster CPU inference).
"""
from __future__ import annotations

import base64
import inspect
import io
import math
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import scipy.ndimage as ndi

from mos.core.cad_parametric import generate_cad_from_parameters


# ===========================================================================
# Foundation-model availability (import-only; models are loaded lazily)
# ===========================================================================
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import (
        AutoProcessor,
        AutoModelForZeroShotObjectDetection,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from transformers import Sam2Model, Sam2Processor
    HAS_SAM2 = True
except ImportError:
    HAS_SAM2 = False

try:
    from transformers import SamModel, SamProcessor
    HAS_SAM1 = True
except ImportError:
    HAS_SAM1 = False

HAS_FOUNDATION = HAS_TORCH and HAS_TRANSFORMERS and (HAS_SAM2 or HAS_SAM1)


# ===========================================================================
# Configuration
# ===========================================================================
DETECTOR_MODEL_ID = os.environ.get(
    "MOS_DETECTOR_MODEL", "IDEA-Research/grounding-dino-base"
)
SAM2_MODEL_ID = os.environ.get(
    "MOS_SAM2_MODEL", "facebook/sam2.1-hiera-large"
)
SAM1_MODEL_ID = os.environ.get(
    "MOS_SAM_MODEL", "facebook/sam-vit-base"
)

# Grounding DINO prompt: period-separated noun phrases covering all target types.
DETECTION_PROMPT = (
    "structural l-bracket. plate with hole. rectangular beam. i-beam. "
    "stepped shaft. hollow pipe. mechanical part."
)

_FRIENDLY_NAMES = {
    "bracket": "Structural L-Bracket",
    "plate_hole": "Tensile Plate with Hole",
    "beam": "Rectangular Beam / Bar",
    "ibeam": "Structural I-Beam",
    "shaft": "Stepped Cylindrical Shaft",
    "pipe": "Hollow Tube / Pipe",
}


# ===========================================================================
# Result dataclass (unchanged public interface)
# ===========================================================================
@dataclass
class CVAnalysisResult:
    """Structured result from real-life object detection and parameterization.

    Attributes
    ----------
    object_type       One of bracket / plate_hole / beam / ibeam / shaft / pipe.
    friendly_name     Human-readable label (e.g. "Structural L-Bracket").
    confidence        Classifier confidence in [0, 1].
    detected_features Human-readable evidence strings from the detector + geometry.
    parameters        Metric CAD dimensions in metres, keys match
                      :func:`mos.core.cad_parametric.generate_cad_from_parameters`.
    pixel_dimensions  (width, height) of the input image in pixels.
    annotated_image_b64 Base64 data URI of the overlay image.
    all_scores        Raw detector label -> score mapping (top candidates).
    """
    object_type: str
    friendly_name: str
    confidence: float
    detected_features: List[str]
    parameters: Dict[str, float]
    pixel_dimensions: Tuple[int, int]
    annotated_image_b64: str
    all_scores: Dict[str, float] = field(default_factory=dict)


class CVModelUnavailableError(RuntimeError):
    """Raised when pretrained foundation models cannot be loaded."""


# ===========================================================================
# Lazy model loading (cached, thread-safe)
# ===========================================================================
_MODELS: Dict[str, Any] = {}
_MODELS_LOCK = threading.Lock()


def _device_str() -> str:
    """Return torch device string, preferring CUDA."""
    return "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"


def _get_detector() -> Tuple[Any, Any, str]:
    """Lazy-load Grounding DINO. Returns (processor, model, device)."""
    if "detector" not in _MODELS:
        with _MODELS_LOCK:
            if "detector" not in _MODELS:
                try:
                    proc = AutoProcessor.from_pretrained(DETECTOR_MODEL_ID)
                    mdl = AutoModelForZeroShotObjectDetection.from_pretrained(
                        DETECTOR_MODEL_ID
                    )
                    dev = _device_str()
                    mdl.to(dev).eval()
                    _MODELS["detector"] = (proc, mdl, dev)
                except Exception as exc:
                    raise CVModelUnavailableError(
                        f"Could not load Grounding DINO ('{DETECTOR_MODEL_ID}'): {exc}. "
                        "Install with `pip install transformers torch` and ensure "
                        "network access is available on first run."
                    ) from exc
    return _MODELS["detector"]


def _get_segmenter() -> Tuple[Any, Any, str, str]:
    """Lazy-load SAM 2 (preferred) or SAM 1. Returns (processor, model, device, kind)."""
    if "segmenter" not in _MODELS:
        with _MODELS_LOCK:
            if "segmenter" not in _MODELS:
                try:
                    if HAS_SAM2:
                        proc = Sam2Processor.from_pretrained(SAM2_MODEL_ID)
                        mdl = Sam2Model.from_pretrained(SAM2_MODEL_ID)
                        kind = "sam2"
                    elif HAS_SAM1:
                        proc = SamProcessor.from_pretrained(SAM1_MODEL_ID)
                        mdl = SamModel.from_pretrained(SAM1_MODEL_ID)
                        kind = "sam1"
                    else:
                        raise RuntimeError(
                            "Neither Sam2Model nor SamModel is importable. "
                            "Install transformers>=4.45 (SAM 2) or >=4.30 (SAM 1)."
                        )
                    dev = _device_str()
                    mdl.to(dev).eval()
                    _MODELS["segmenter"] = (proc, mdl, dev, kind)
                except Exception as exc:
                    raise CVModelUnavailableError(
                        f"Could not load SAM segmenter: {exc}."
                    ) from exc
    return _MODELS["segmenter"]


# ===========================================================================
# Image input normalisation
# ===========================================================================
def _load_image(img_input: Union[str, Path, Image.Image, np.ndarray, bytes]) -> Image.Image:
    """Normalize orientation/color and bound image cost for CV inference."""
    if isinstance(img_input, Image.Image):
        image = ImageOps.exif_transpose(img_input).convert("RGB")
    elif isinstance(img_input, np.ndarray):
        if img_input.ndim == 2:
            image = Image.fromarray(img_input).convert("RGB")
        else:
            image = Image.fromarray(img_input.astype(np.uint8)).convert("RGB")
    elif isinstance(img_input, (str, Path)):
        s = str(img_input).strip()
        if s.startswith("data:image"):
            _, encoded = s.split(",", 1)
            image = Image.open(io.BytesIO(base64.b64decode(encoded)))
        else:
            p = Path(s).resolve()
            if not p.is_file():
                raise FileNotFoundError(f"Image file not found: {p}")
            image = Image.open(p)
        image = ImageOps.exif_transpose(image).convert("RGB")
    elif isinstance(img_input, (bytes, bytearray)):
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(img_input))).convert("RGB")
    else:
        raise TypeError(f"Unsupported image input type: {type(img_input)}")

    if image.width < 64 or image.height < 64:
        raise ValueError("Image must be at least 64 × 64 pixels.")
    if image.width * image.height > 60_000_000:
        raise ValueError("Image exceeds 60 megapixels. Resize it before analysis.")
    # DINO/SAM gain little from oversized phone images but consume much more
    # memory. Keep detail for small parts while bounding inference to 2048 px.
    image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    return image


# ===========================================================================
# Detection & segmentation
# ===========================================================================
def _post_process_detections(
    proc: Any,
    outputs: Any,
    input_ids: Any,
    target_sizes: List[Tuple[int, int]],
    threshold: float,
    text_threshold: float,
) -> Any:
    """Call `post_process_grounded_object_detection` across transformers versions.

    transformers < 4.39.0 exposed ``box_threshold`` and ``text_threshold``.
    transformers >= 4.39.0 replaced ``box_threshold`` with ``threshold`` while
    keeping ``text_threshold``. Both names refer to the detection-box score cut.
    """
    sig = inspect.signature(proc.post_process_grounded_object_detection)
    params = sig.parameters
    kwargs: Dict[str, Any] = {
        "text_threshold": text_threshold,
        "target_sizes": target_sizes,
    }
    if "box_threshold" in params:
        kwargs["box_threshold"] = threshold
    else:
        kwargs["threshold"] = threshold
    return proc.post_process_grounded_object_detection(outputs, input_ids, **kwargs)


def _detect_parts(img: Image.Image) -> List[Dict[str, Any]]:
    """Run Grounding DINO; return candidate mechanical parts, largest first."""
    proc, mdl, dev = _get_detector()
    inputs = proc(
        images=img.convert("RGB"), text=DETECTION_PROMPT, return_tensors="pt"
    ).to(dev)
    with torch.no_grad():
        outputs = mdl(**inputs)

    results = _post_process_detections(
        proc,
        outputs,
        inputs.input_ids,
        target_sizes=[img.size[::-1]],
        threshold=0.15,
        text_threshold=0.15,
    )[0]

    detections: List[Dict[str, Any]] = []
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        x1, y1, x2, y2 = (float(v) for v in box)
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        detections.append({
            "label": str(label).strip().lower(),
            "score": float(score),
            "box": [x1, y1, x2, y2],
        })
    # Prefer confident, reasonably large, centrally located part detections.
    # Multiplying by raw box area made a low-score background-sized box win over
    # the actual part in cluttered workshop photos.
    iw, ih = img.size
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        area_fraction = max(0.0, (x2 - x1) * (y2 - y1) / (iw * ih))
        cx, cy = (x1 + x2) / (2 * iw), (y1 + y2) / (2 * ih)
        center_distance = math.hypot(cx - 0.5, cy - 0.5) / math.sqrt(0.5)
        area_factor = min(1.0, math.sqrt(area_fraction / 0.18))
        center_factor = 1.0 - 0.18 * min(1.0, center_distance)
        detection["rank_score"] = detection["score"] * area_factor * center_factor
    detections.sort(key=lambda d: d["rank_score"], reverse=True)
    return detections


def _segment_with_sam(img: Image.Image, box: List[float]) -> np.ndarray:
    """Segment the object inside `box` using SAM 2 or SAM 1; returns (H, W) bool mask."""
    proc, mdl, dev, kind = _get_segmenter()
    inputs = proc(
        images=img.convert("RGB"), input_boxes=[[box]], return_tensors="pt"
    ).to(dev)
    with torch.no_grad():
        outputs = mdl(**inputs)

    # Post-process masks back to original resolution (API differs slightly between versions).
    try:
        if kind == "sam2":
            masks = proc.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )
        else:
            masks = proc.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )
    except (AttributeError, TypeError) as exc:
        raise CVModelUnavailableError(
            f"SAM mask post-processing API not recognised: {exc}"
        ) from exc

    if isinstance(masks, (list, tuple)):
        m = masks[0]
    else:
        m = masks
    if isinstance(m, torch.Tensor):
        m = m.cpu().numpy()
    m = np.asarray(m)
    while m.ndim > 3:
        m = m[0]

    # Pick the highest-IoU candidate among SAM's multi-scale outputs.
    iou = getattr(outputs, "iou_scores", None)
    if iou is not None:
        iou_np = iou.cpu().numpy()
        if iou_np.ndim > 1:
            iou_flat = iou_np.reshape(-1, iou_np.shape[-1])[0]
        else:
            iou_flat = iou_np.reshape(-1)
        best = int(np.argmax(iou_flat))
        best = min(best, m.shape[0] - 1)
    else:
        best = 0
    return m[best].astype(bool)


# ===========================================================================
# Geometric helpers (applied to CLEAN masks)
# ===========================================================================
def _otsu(values: np.ndarray) -> float:
    """Otsu threshold for a 1-D array of scalar values."""
    vals = np.asarray(values, dtype=float).ravel()
    if len(vals) < 2 or vals.min() == vals.max():
        return float(vals.mean()) if len(vals) else 128.0
    hist, edges = np.histogram(vals, bins=256)
    centers = 0.5 * (edges[:-1] + edges[1:])
    total = hist.sum()
    sum_all = float((centers * hist).sum())
    w_b, sum_b, best_var, best_thr = 0.0, 0.0, -1.0, float(vals.mean())
    for i in range(len(hist)):
        w_b += hist[i]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += centers[i] * hist[i]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        between = w_b * w_f * (m_b - m_f) ** 2
        if between > best_var:
            best_var, best_thr = between, centers[i]
    return float(best_thr)


def _trace_contour(mask: np.ndarray) -> np.ndarray:
    """Moore-neighbour contour trace. Returns (N, 2) array of (x, y) pixels."""
    from mos.core.image_to_cad import _trace_moore_neighbor
    return _trace_moore_neighbor(mask)


def _rdp_simplify(pts: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer-Douglas-Peucker polygon simplification."""
    if len(pts) <= 3 or epsilon <= 0:
        return pts
    dmax, index = 0.0, 0
    end = len(pts) - 1
    line = pts[end] - pts[0]
    norm = float(np.linalg.norm(line))
    for i in range(1, end):
        if norm > 1e-12:
            d = abs(line[0] * (pts[0, 1] - pts[i, 1])
                    - line[1] * (pts[0, 0] - pts[i, 0])) / norm
        else:
            d = float(np.linalg.norm(pts[i] - pts[0]))
        if d > dmax:
            index, dmax = i, d
    if dmax > epsilon:
        r1 = _rdp_simplify(pts[:index + 1], epsilon)
        r2 = _rdp_simplify(pts[index:], epsilon)
        return np.vstack((r1[:-1], r2))
    return np.array([pts[0], pts[end]])


def _count_concave(pts: np.ndarray) -> int:
    """Count re-entrant (concave) corners of a closed polygon."""
    n = len(pts)
    if n < 4:
        return 0
    area2 = sum(
        (pts[(i + 1) % n][0] - pts[i][0]) * (pts[(i + 1) % n][1] + pts[i][1])
        for i in range(n)
    )
    ccw = area2 < 0
    count = 0
    for i in range(n):
        p_prev, p_curr, p_next = pts[(i - 1) % n], pts[i], pts[(i + 1) % n]
        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if (ccw and cross < -1e-5) or ((not ccw) and cross > 1e-5):
            count += 1
    return count


def _fit_circle_ransac(
    points: np.ndarray, max_iters: int = 120, dist_threshold: float = 3.0
) -> Optional[Tuple[float, float, float, float]]:
    """Fit a circle with RANSAC + algebraic refinement.

    Returns (cx, cy, r, inlier_ratio) or None when no stable circle exists.
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n < 6:
        return None
    rng = np.random.default_rng(0)
    best_circle, best_inliers, best_mask = None, 0, None
    for _ in range(max_iters):
        idx = rng.choice(n, 3, replace=False)
        p1, p2, p3 = pts[idx]
        d = 2.0 * (p1[0] * (p2[1] - p3[1])
                   + p2[0] * (p3[1] - p1[1])
                   + p3[0] * (p1[1] - p2[1]))
        if abs(d) < 1e-6:
            continue
        ux = ((p1[0] ** 2 + p1[1] ** 2) * (p2[1] - p3[1])
              + (p2[0] ** 2 + p2[1] ** 2) * (p3[1] - p1[1])
              + (p3[0] ** 2 + p3[1] ** 2) * (p1[1] - p2[1])) / d
        uy = ((p1[0] ** 2 + p1[1] ** 2) * (p3[0] - p2[0])
              + (p2[0] ** 2 + p2[1] ** 2) * (p1[0] - p3[0])
              + (p3[0] ** 2 + p3[1] ** 2) * (p2[0] - p1[0])) / d
        r = math.hypot(p1[0] - ux, p1[1] - uy)
        if r < 3.0 or r > 2000.0:
            continue
        dists = np.abs(np.hypot(pts[:, 0] - ux, pts[:, 1] - uy) - r)
        inl_mask = dists < dist_threshold
        inl = int(inl_mask.sum())
        if inl > best_inliers:
            best_inliers, best_circle, best_mask = inl, (ux, uy, r), inl_mask
    if best_circle is None or best_inliers < 6:
        return None
    xi, yi = pts[best_mask, 0], pts[best_mask, 1]
    A = np.column_stack([xi, yi, np.ones_like(xi)])
    b = xi ** 2 + yi ** 2
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx = sol[0] / 2.0
        cy = sol[1] / 2.0
        r = math.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 1.0))
        return float(cx), float(cy), float(r), float(best_inliers / n)
    except Exception:
        return (*best_circle, float(best_inliers / n))


def _geometric_invariants(mask: np.ndarray) -> Dict[str, Any]:
    """Compute scale-invariant geometric descriptors from a clean binary mask."""
    if not mask.any():
        return {
            "area": 0.0, "aspect": 1.0, "solidity": 0.0, "circularity": 0.0,
            "sym_h": 0.5, "sym_v": 0.5, "n_concave": 0, "has_hole": 0,
            "holes": [], "len_major": 1.0, "len_minor": 1.0,
            "centroid": (0.0, 0.0), "contour": np.empty((0, 2)),
        }

    ys, xs = np.nonzero(mask)
    area = float(mask.sum())
    cy, cx = float(ys.mean()), float(xs.mean())

    cov = np.cov(np.vstack([xs - cx, ys - cy]))
    eigvecs = np.linalg.eigh(cov)[1]
    major = eigvecs[:, 1]
    angle = math.atan2(major[1], major[0])
    ca, sa = math.cos(angle), math.sin(angle)
    proj_major = (xs - cx) * ca + (ys - cy) * sa
    proj_minor = -(xs - cx) * sa + (ys - cy) * ca
    len_major = float(np.percentile(proj_major, 98) - np.percentile(proj_major, 2))
    len_minor = float(np.percentile(proj_minor, 98) - np.percentile(proj_minor, 2))
    aspect = max(len_major, len_minor) / max(min(len_major, len_minor), 1.0)

    contour = _trace_contour(mask)
    if len(contour) < 4:
        contour = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)

    perim = float(np.sum(np.linalg.norm(
        np.diff(np.vstack([contour, contour[0]]), axis=0), axis=1)))
    circularity = (4.0 * math.pi * area / (perim ** 2)) if perim > 1e-3 else 0.0

    from scipy.spatial import ConvexHull
    try:
        hull_area = float(ConvexHull(contour).volume)
    except Exception:
        hull_area = 0.0
    solidity = (area / hull_area) if hull_area > 1e-3 else 1.0

    # Bilateral symmetry in the principal-axis frame
    if abs(angle) > 1e-3:
        rot = ndi.rotate(mask.astype(float), -math.degrees(angle),
                         reshape=False, order=0) > 0.5
    else:
        rot = mask.astype(bool)
    coords = np.argwhere(rot)
    if len(coords) > 20:
        r0, r1 = coords[:, 0].min(), coords[:, 0].max()
        c0, c1 = coords[:, 1].min(), coords[:, 1].max()
        crop = rot[r0:r1 + 1, c0:c1 + 1]
        sym_h = float(np.mean(crop == np.fliplr(crop)))
        sym_v = float(np.mean(crop == np.flipud(crop)))
    else:
        sym_h = sym_v = 0.5

    diag = math.hypot(len_major, len_minor)
    simplified = _rdp_simplify(contour, epsilon=max(diag * 0.03, 1.0))
    n_concave = _count_concave(simplified)

    # Interior holes (verified by RANSAC circle fit)
    filled = ndi.binary_fill_holes(mask)
    hole_mask = filled & ~mask
    h_labeled, n_holes = ndi.label(hole_mask)
    holes_info: List[Dict[str, float]] = []
    min_hole = max(int(area * 0.005), 16)
    for i in range(1, n_holes + 1):
        single = h_labeled == i
        h_area = float(single.sum())
        if h_area < min_hole:
            continue
        h_cont = _trace_contour(single)
        if len(h_cont) < 4:
            continue
        h_perim = float(np.sum(np.linalg.norm(
            np.diff(np.vstack([h_cont, h_cont[0]]), axis=0), axis=1)))
        h_circ = 4.0 * math.pi * h_area / (h_perim ** 2) if h_perim > 1e-3 else 0.0
        fit = _fit_circle_ransac(h_cont, dist_threshold=2.5)
        if fit is not None:
            hcx, hcy, hr, inlier = fit
            if inlier > 0.4 and hr > 2.0:
                holes_info.append({
                    "cx": hcx, "cy": hcy, "r": hr, "diameter": 2.0 * hr,
                    "inliers": inlier, "circularity": h_circ,
                })

    return {
        "area": area,
        "aspect": aspect,
        "solidity": solidity,
        "circularity": circularity,
        "sym_h": sym_h,
        "sym_v": sym_v,
        "n_concave": n_concave,
        "has_hole": len(holes_info),
        "holes": holes_info,
        "len_major": len_major,
        "len_minor": len_minor,
        "centroid": (cx, cy),
        "contour": contour,
    }


def _geometric_label(g: Dict[str, Any]) -> Tuple[str, float]:
    """Heuristic label from geometric invariants (cross-check / fallback)."""
    scores = {k: 0.0 for k in ("bracket", "plate_hole", "beam", "ibeam", "shaft", "pipe")}
    circ = g["circularity"]
    sol = g["solidity"]
    asp = g["aspect"]
    nh = g["has_hole"]
    nc = g["n_concave"]
    hole_circ_max = max(
        (h["circularity"] for h in g.get("holes", [])), default=0.0
    )

    if circ > 0.75 and nh >= 1 and hole_circ_max > 0.55:
        scores["pipe"] += 1.0
    if g["sym_h"] > 0.82 and g["sym_v"] > 0.82 and nc >= 4 and sol < 0.92:
        scores["ibeam"] += 1.0
    if (g["sym_h"] < 0.80 or g["sym_v"] < 0.80) and nc >= 1 and 0.45 < sol < 0.95:
        scores["bracket"] += 0.9
    if nh >= 1 and circ < 0.85 and 1.0 <= asp <= 3.5:
        scores["plate_hole"] += 0.9
    if asp >= 2.0 and nh == 0 and sol > 0.9:
        scores["shaft"] += 0.7
    if asp >= 2.0 and nh == 0 and sol > 0.88:
        scores["beam"] += 0.9

    best = max(scores, key=scores.get)
    return best, min(scores[best], 1.0)


# ===========================================================================
# Detector-label -> canonical type mapping
# ===========================================================================
_LABEL_TO_TYPE = {
    "structural l-bracket": "bracket",
    "l-bracket": "bracket",
    "l bracket": "bracket",
    "bracket": "bracket",
    "plate with hole": "plate_hole",
    "plate": "plate_hole",
    "tensile plate": "plate_hole",
    "rectangular beam": "beam",
    "beam": "beam",
    "bar": "beam",
    "i-beam": "ibeam",
    "i beam": "ibeam",
    "ibeam": "ibeam",
    "h-beam": "ibeam",
    "stepped shaft": "shaft",
    "shaft": "shaft",
    "cylinder": "shaft",
    "hollow pipe": "pipe",
    "pipe": "pipe",
    "tube": "pipe",
}


def _map_label(raw: str) -> Optional[str]:
    """Map a Grounding DINO label string to a canonical type, or None."""
    raw = raw.lower().strip()
    if raw in _LABEL_TO_TYPE:
        return _LABEL_TO_TYPE[raw]
    for key, val in _LABEL_TO_TYPE.items():
        if key in raw:
            return val
    return None


# ===========================================================================
# Metric parameter extraction from clean masks
# ===========================================================================
def _extract_parameters(
    obj_type: str,
    mask: np.ndarray,
    inv: Dict[str, Any],
    reference_scale: float,
) -> Dict[str, float]:
    """Convert pixel measurements from a clean mask into metric CAD parameters."""
    len_major = inv["len_major"]
    len_minor = inv["len_minor"]
    m_per_px = reference_scale / max(len_major, len_minor, 1.0)
    holes = inv.get("holes", [])

    if obj_type == "bracket":
        leg_x = round(len_major * m_per_px, 3)
        leg_y = round(len_minor * m_per_px, 3)
        arm_t = round(min(leg_x, leg_y) * 0.18, 3)
        depth_w = round(min(leg_x, leg_y) * 0.40, 3)
        hole_r = round(holes[0]["r"] * m_per_px, 3) if holes else round(arm_t * 0.35, 3)
        return {
            "leg_x": max(leg_x, 0.05),
            "leg_y": max(leg_y, 0.05),
            "width": max(depth_w, 0.02),
            "thickness": max(arm_t, 0.008),
            "hole_radius": max(hole_r, 0.003),
        }

    if obj_type == "plate_hole":
        L = round(len_major * m_per_px, 3)
        W = round(len_minor * m_per_px, 3)
        T = round(W * 0.12, 3)
        hole_d = (round(holes[0]["diameter"] * m_per_px, 3)
                  if holes else round(W * 0.25, 3))
        return {
            "length": max(L, 0.10),
            "width": max(W, 0.05),
            "thickness": max(T, 0.005),
            "hole_diameter": min(max(hole_d, 0.01), W * 0.6),
        }

    if obj_type == "beam":
        L = round(len_major * m_per_px, 3)
        H = round(len_minor * m_per_px, 3)
        return {
            "length": max(L, 0.10),
            "height": max(H, 0.02),
            "width": max(round(H * 0.65, 3), 0.01),
        }

    if obj_type == "ibeam":
        H = round(len_major * m_per_px, 3)
        Bf = round(len_minor * m_per_px, 3)
        L = round(max(H * 2.0, reference_scale * 2.0), 3)
        return {
            "length": max(L, 0.30),
            "height": max(H, 0.10),
            "flange_width": max(Bf, 0.06),
            "flange_thickness": round(max(H, 0.10) * 0.10, 3),
            "web_thickness": round(max(H, 0.10) * 0.08, 3),
        }

    if obj_type == "shaft":
        L = round(len_major * m_per_px, 3)
        D1 = round(len_minor * m_per_px, 3)
        return {
            "length": max(L, 0.15),
            "diameter_main": max(D1, 0.03),
            "diameter_step": max(round(D1 * 0.65, 3), 0.02),
            "step_length": round(L * 0.35, 3),
        }

    if obj_type == "pipe":
        Do = round(max(len_major, len_minor) * m_per_px, 3)
        Di = (round(holes[0]["diameter"] * m_per_px, 3)
              if holes else round(Do * 0.70, 3))
        return {
            "length": round(reference_scale * 2.5, 3),
            "outer_diameter": max(Do, 0.04),
            "inner_diameter": min(max(Di, 0.02), Do * 0.85),
        }

    return {"length": reference_scale}


# ===========================================================================
# Annotation rendering
# ===========================================================================
def _render_annotation(
    img: Image.Image,
    mask: np.ndarray,
    box: List[float],
    obj_type: str,
    confidence: float,
) -> str:
    """Return a base64 PNG data URI with mask outline + box + label badge."""
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated, "RGBA")

    contour = _trace_contour(mask)
    if len(contour) > 2:
        pts = [(int(p[0]), int(p[1])) for p in contour]
        draw.polygon(pts, outline=(0, 255, 120, 240), width=3)

    x1, y1, x2, y2 = (int(v) for v in box)
    draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 180, 0, 220), width=2)

    badge = f"{_FRIENDLY_NAMES.get(obj_type, obj_type)} | {int(confidence * 100)}%"
    bw = len(badge) * 7 + 12
    bx = max(x1, 4)
    by = max(y1 - 26, 4)
    draw.rectangle([(bx, by), (bx + bw, by + 22)], fill=(15, 23, 42, 230))
    draw.text((bx + 6, by + 4), badge, fill=(255, 255, 255, 255))

    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ===========================================================================
# Geometric fallback (used only when foundation models are unavailable)
# ===========================================================================
def _fallback_segment(img: Image.Image) -> np.ndarray:
    """Multi-cue geometric segmentation (fallback when models unavailable).

    Estimates the background color from the image border, then combines robust
    color distance with illumination-normalized local contrast. This handles
    common light/dark studio backdrops and moderate lighting gradients without
    allowing a whole-image Otsu split to absorb the background.
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    h, w = gray.shape

    # Border samples estimate a dominant, mildly textured background. Use a
    # robust MAD threshold so a few border pixels touching the object do not
    # change the foreground decision for the full image.
    band = max(2, int(round(min(h, w) * 0.035)))
    border = np.concatenate((
        rgb[:band].reshape(-1, 3), rgb[-band:].reshape(-1, 3),
        rgb[:, :band].reshape(-1, 3), rgb[:, -band:].reshape(-1, 3),
    ), axis=0)
    background = np.median(border, axis=0)
    border_dist = np.linalg.norm(border - background, axis=1)
    border_mad = float(np.median(np.abs(border_dist - np.median(border_dist))))
    color_dist = np.linalg.norm(rgb - background[None, None, :], axis=2)
    color_threshold = float(np.clip(np.percentile(border_dist, 95) + max(16.0, 3.5 * border_mad), 20.0, 68.0))
    mask_color = color_dist > color_threshold

    # Local contrast is useful when the background is not a flat color.
    sigma = max(max(h, w) // 12, 5)
    bg = ndi.gaussian_filter(gray, sigma=sigma)
    diff = np.abs(gray - bg)
    local_samples = diff[:band].ravel()
    thr_local = max(float(np.percentile(local_samples, 98)) + 8.0, 12.0)
    mask_local = diff > min(thr_local, 40.0)
    mask = mask_color | mask_local

    # Remove border-connected pixels (the estimated background), then retain
    # the strongest interior component and close small gaps in its silhouette.
    mask = ndi.binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
    mask = ndi.binary_closing(mask, structure=np.ones((5, 5), dtype=bool))
    labeled, n = ndi.label(mask)
    if n == 0:
        return np.zeros((h, w), dtype=bool)
    sizes = ndi.sum(mask, labeled, range(1, n + 1))
    # Ignore tiny contrast specks; among the remaining components, favor area
    # while mildly preferring central objects over edge clutter.
    candidates = []
    for component_id, size in enumerate(sizes, start=1):
        if size < max(30, int(h * w * 0.002)):
            continue
        component = labeled == component_id
        cy, cx = np.mean(np.nonzero(component), axis=1)
        center_penalty = 1.0 - 0.12 * min(1.0, math.hypot(cx / w - 0.5, cy / h - 0.5) / 0.707)
        candidates.append((float(size) * center_penalty, component_id))
    if not candidates:
        return np.zeros((h, w), dtype=bool)
    mask = labeled == max(candidates)[1]

    # --- 6. Preserve interior holes (background-like regions fully enclosed)
    background_ref = float(np.median(np.asarray(img.convert("L"), dtype=np.float32)[:band]))
    holes = mask & (np.abs(gray - background_ref) < 15)
    holes = ndi.binary_opening(holes, structure=np.ones((5, 5), dtype=bool))
    to_remove = np.zeros_like(mask)
    if holes.any():
        h_labeled, n_h = ndi.label(holes)
        for i in range(1, n_h + 1):
            single = h_labeled == i
            if single.sum() < 30:
                continue
            ring = ndi.binary_dilation(single, iterations=2) & ~single & ~mask
            if ring.sum() == 0:
                to_remove |= single
    return mask & ~to_remove


def _analyze_fallback(img: Image.Image, reference_scale: float) -> CVAnalysisResult:
    """Geometric baseline used when foundation models are unavailable.

    Documented limitation: this path is not intended for production real-image
    use. Install `transformers` + `torch` for the model-based pipeline.
    """
    w, h = img.size
    backend = "classical segmentation"
    try:
        from rembg import remove
        output = remove(img, only_mask=True)
        if isinstance(output, bytes):
            output = Image.open(io.BytesIO(output))
        mask = np.asarray(output.convert("L")) > 128
        if mask.mean() < .01 or mask.mean() > .98:
            raise ValueError("Ambiguous foreground mask")
        backend = "neural foreground segmentation (rembg)"
    except Exception:
        mask = _fallback_segment(img)

    if not mask.any():
        return CVAnalysisResult(
            object_type="beam",
            friendly_name=_FRIENDLY_NAMES["beam"],
            confidence=0.30,
            detected_features=["Fallback could not isolate an object; install transformers for real-image support."],
            parameters={
                "length": reference_scale,
                "height": round(reference_scale * 0.2, 3),
                "width": round(reference_scale * 0.1, 3),
            },
            pixel_dimensions=(w, h),
            annotated_image_b64=_render_annotation(
                img, np.zeros((h, w), dtype=bool), [0, 0, w, h], "beam", 0.30
            ),
            all_scores={},
        )

    inv = _geometric_invariants(mask)
    geo_type, geo_conf = _geometric_label(inv)
    params = _extract_parameters(geo_type, mask, inv, reference_scale)

    ys, xs = np.nonzero(mask)
    box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    confidence = min(0.99, 0.5 + 0.5 * geo_conf)

    features = [
        f"Fallback path using {backend} - install "
        "`pip install transformers torch` for real-image support.",
        f"Mask aspect ratio: {inv['aspect']:.2f}",
        f"Mask solidity: {inv['solidity']:.2f}",
    ]
    if inv["has_hole"]:
        features.append(f"Detected {inv['has_hole']} interior hole(s)")

    annotated_b64 = _render_annotation(img, mask, box, geo_type, confidence)
    return CVAnalysisResult(
        object_type=geo_type,
        friendly_name=_FRIENDLY_NAMES[geo_type],
        confidence=confidence,
        detected_features=features,
        parameters=params,
        pixel_dimensions=(w, h),
        annotated_image_b64=annotated_b64,
        all_scores={geo_type: geo_conf},
    )


# ===========================================================================
# Public API
# ===========================================================================
def analyze_image_object(
    image_input: Union[str, Path, Image.Image, np.ndarray, bytes],
    reference_scale: float = 0.30,
) -> CVAnalysisResult:
    """Analyze a real-world photograph and return classified CAD parameters.

    Parameters
    ----------
    image_input     Path, PIL Image, numpy array, bytes, or base64 data URI.
    reference_scale Length of the largest bounding dimension in metres.

    Returns
    -------
    CVAnalysisResult with the primary detection, confidence, and metric parameters.

    Notes
    -----
    Uses Grounding DINO (open-vocabulary detection) + SAM 2/SAM (promptable
    segmentation) + geometric parameter extraction when the foundation-model
    stack is available. Falls back to a multi-cue geometric baseline otherwise.
    """
    img = _load_image(image_input)
    orig_w, orig_h = img.size

    # ------------------------------------------------------------ 1. Model path
    fallback_reason = "Foundation model dependencies are unavailable."
    if HAS_FOUNDATION:
        detections: List[Dict[str, Any]] = []
        try:
            detections = _detect_parts(img)
        except Exception as exc:
            fallback_reason = f"Detector unavailable: {type(exc).__name__}: {exc}"
            detections = []

        if detections:
            best = detections[0]
            try:
                mask = _segment_with_sam(img, best["box"])
            except Exception:
                # SAM failed → use the detector's box interior as a coarse mask
                x1, y1, x2, y2 = (int(v) for v in best["box"])
                mask = np.zeros((orig_h, orig_w), dtype=bool)
                mask[max(0, y1):min(orig_h, y2),
                     max(0, x1):min(orig_w, x2)] = True

            inv = _geometric_invariants(mask)
            det_type = _map_label(best["label"])
            geo_type, geo_conf = _geometric_label(inv)

            # Reconcile the zero-shot detector label with geometric evidence
            if det_type is not None:
                obj_type = det_type
                if det_type == geo_type:
                    confidence = min(0.99, 0.5 + 0.5 * best["score"] + 0.2 * geo_conf)
                else:
                    confidence = min(0.99, 0.5 * best["score"] + 0.3 * geo_conf + 0.2)
            else:
                obj_type = geo_type
                confidence = min(0.99, 0.5 + 0.5 * geo_conf)

            params = _extract_parameters(obj_type, mask, inv, reference_scale)

            features = [
                f"Detector label: '{best['label']}' (score {best['score']:.2f})",
                f"Mask aspect ratio: {inv['aspect']:.2f}",
                f"Mask solidity: {inv['solidity']:.2f}",
            ]
            if inv["has_hole"]:
                features.append(
                    f"{inv['has_hole']} interior hole(s) verified via RANSAC"
                )
            if inv["n_concave"]:
                features.append(f"{inv['n_concave']} concave corner(s)")

            annotated = _render_annotation(
                img, mask, best["box"], obj_type, confidence
            )
            return CVAnalysisResult(
                object_type=obj_type,
                friendly_name=_FRIENDLY_NAMES[obj_type],
                confidence=confidence,
                detected_features=features,
                parameters=params,
                pixel_dimensions=(orig_w, orig_h),
                annotated_image_b64=annotated,
                all_scores={d["label"]: d["score"] for d in detections[:5]},
            )

    # ------------------------------------------------------------ 2. Fallback
    result = _analyze_fallback(img, reference_scale)
    result.detected_features.insert(0, fallback_reason)
    return result


def cv_image_to_cad_step(
    image_input: Union[str, Path, Image.Image, np.ndarray, bytes],
    output_step_path: Union[str, Path],
    reference_scale: float = 0.30,
    override_params: Optional[Dict[str, float]] = None,
) -> Tuple[Path, CVAnalysisResult]:
    """End-to-end pipeline: real-life photo -> detection -> parametric CAD STEP.

    Convenience wrapper around :func:`analyze_image_object` and
    :func:`mos.core.cad_parametric.generate_cad_from_parameters`.
    """
    result = analyze_image_object(image_input, reference_scale=reference_scale)
    params = dict(result.parameters)
    if override_params:
        params.update(override_params)
    step_path = generate_cad_from_parameters(result.object_type, params, output_step_path)
    return step_path, result
