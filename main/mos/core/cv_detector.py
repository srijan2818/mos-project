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
``MOS_SAM2_MODEL``, ``MOS_SAM_MODEL``. Without CUDA the defaults are the small
variants (Grounding DINO tiny, SAM 2.1 hiera-tiny), which are several times faster.

Order of work (cheapest first):
  1. Classical foreground segmentation. When the background is uniform (studio
     shot, sheet of paper, bench mat) this mask is already exact and the
     foundation models are skipped entirely -- milliseconds instead of ~20 s on CPU.
  2. Grounding DINO + SAM for cluttered scenes. If SAM cannot run, the classical
     segmenter is applied inside the detector box; the box itself is never used
     as a mask (that made every hole/concavity measurement meaningless).
  3. The geometric classifier on a real mask outranks the zero-shot label, which
     is unreliable for these fine-grained part categories.
"""
from __future__ import annotations

import logging
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
_HAS_CUDA = bool(HAS_TORCH and torch.cuda.is_available())
DETECTOR_MODEL_ID = os.environ.get(
    "MOS_DETECTOR_MODEL",
    "IDEA-Research/grounding-dino-base" if _HAS_CUDA else "IDEA-Research/grounding-dino-tiny",
)
SAM2_MODEL_ID = os.environ.get(
    "MOS_SAM2_MODEL",
    "facebook/sam2.1-hiera-large" if _HAS_CUDA else "facebook/sam2.1-hiera-tiny",
)
DETECTION_MAX_SIDE = 800       # Grounding DINO input bound; boxes are mapped back
SAM1_MODEL_ID = os.environ.get(
    "MOS_SAM_MODEL", "facebook/sam-vit-base"
)

# Grounding DINO prompt: period-separated noun phrases covering all target types.
DETECTION_PROMPT = (
    "structural l-bracket. plate with hole. rectangular beam. i-beam. "
    "stepped shaft. hollow pipe. mechanical part."
)

log = logging.getLogger(__name__)

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
    mask              (H, W) bool foreground mask the parameters were measured on;
                      feed it to :func:`mos.core.image_to_mesh.mesh_from_mask` to mesh
                      the photographed outline directly.
    segmentation      Which segmenter produced ``mask``.
    prompt_points     Clicks that selected the object: (x, y, 1 = part / 0 = not part), image pixels.
    candidate_index   Which of SAM's nested masks (0 = smallest) was used; None if not prompted.
    n_candidates      How many nested masks SAM offered for the prompt.
    """
    object_type: Optional[str]
    friendly_name: str
    confidence: float
    detected_features: List[str]
    parameters: Dict[str, float]
    pixel_dimensions: Tuple[int, int]
    annotated_image_b64: str
    all_scores: Dict[str, float] = field(default_factory=dict)
    mask: Optional[np.ndarray] = field(default=None, repr=False)
    segmentation: str = ""
    prompt_points: List[Tuple[float, float, int]] = field(default_factory=list)
    candidate_index: Optional[int] = None
    n_candidates: int = 0


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
    """Lazy-load SAM 2 (preferred) or SAM 1. Returns (processor, model, device, kind).

    SAM 2's image processor needs torchvision; when that (or the SAM 2 weights) is
    unavailable, SAM 1 is tried before giving up.
    """
    if "segmenter" not in _MODELS:
        with _MODELS_LOCK:
            if "segmenter" not in _MODELS:
                errors = []
                candidates = []
                if HAS_SAM2:
                    candidates.append(("sam2", Sam2Processor, Sam2Model, SAM2_MODEL_ID))
                if HAS_SAM1:
                    candidates.append(("sam1", SamProcessor, SamModel, SAM1_MODEL_ID))
                for kind, proc_cls, model_cls, model_id in candidates:
                    try:
                        proc = proc_cls.from_pretrained(model_id)
                        mdl = model_cls.from_pretrained(model_id)
                        dev = _device_str()
                        mdl.to(dev).eval()
                        _MODELS["segmenter"] = (proc, mdl, dev, kind)
                        break
                    except Exception as exc:
                        first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
                        errors.append(f"{model_id}: {first_line}")
                else:
                    raise CVModelUnavailableError(
                        "Could not load a SAM segmenter (" + "; ".join(errors or ["transformers has no SAM classes"]) + ")."
                    )
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
    # Detection quality saturates well below phone resolution; bound the cost.
    small = img.convert("RGB")
    small.thumbnail((DETECTION_MAX_SIDE, DETECTION_MAX_SIDE), Image.Resampling.BILINEAR)
    sx, sy = img.width / small.width, img.height / small.height
    inputs = proc(images=small, text=DETECTION_PROMPT, return_tensors="pt").to(dev)
    with torch.inference_mode():
        outputs = mdl(**inputs)

    results = _post_process_detections(
        proc,
        outputs,
        inputs.input_ids,
        target_sizes=[small.size[::-1]],
        threshold=0.20,
        text_threshold=0.25,
    )[0]

    # transformers >= 4.51 returns phrase strings under "text_labels"
    # ("labels" became integer ids).
    labels = results["text_labels"] if "text_labels" in results else results.get("labels", [])
    detections: List[Dict[str, Any]] = []
    for box, score, label in zip(results["boxes"], results["scores"], labels):
        b = [float(v) for v in box]
        x1, y1, x2, y2 = b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        detections.append({
            "label": _normalise_label(label),
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


def _sam_masks(proc, kind, outputs, inputs) -> np.ndarray:
    """SAM output masks at the original image size, as a bool array (..., H, W)."""
    # SAM 1 (and SAM 2 in transformers 4.x) also needs the resized input size;
    # SAM 2 in transformers 5 takes only original sizes.
    post = proc.post_process_masks if kind == "sam2" else proc.image_processor.post_process_masks
    args = [outputs.pred_masks.cpu(), inputs["original_sizes"].cpu()]
    if "reshaped_input_sizes" in inputs and "reshaped_input_sizes" in inspect.signature(post).parameters:
        args.append(inputs["reshaped_input_sizes"].cpu())
    try:
        masks = post(*args)
    except (AttributeError, TypeError, KeyError) as exc:
        raise CVModelUnavailableError(
            f"SAM mask post-processing API not recognised: {exc}"
        ) from exc
    m = masks[0] if isinstance(masks, (list, tuple)) else masks
    if isinstance(m, torch.Tensor):
        m = m.cpu().numpy()
    return np.asarray(m).astype(bool)


def _segment_with_sam(img: Image.Image, box: List[float]) -> np.ndarray:
    """Segment the object inside `box` using SAM 2 or SAM 1; returns (H, W) bool mask."""
    proc, mdl, dev, kind = _get_segmenter()
    inputs = proc(
        images=img.convert("RGB"), input_boxes=[[box]], return_tensors="pt"
    ).to(dev)
    with torch.inference_mode():
        outputs = mdl(**inputs)
    m = _sam_masks(proc, kind, outputs, inputs)
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
    # Intensities are 0-255: a spread under 1e-9 is a constant image.
    if len(vals) < 2 or float(np.ptp(vals)) < 1e-9:
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
    """Ramer-Douglas-Peucker simplification (shared vectorised implementation)."""
    from mos.core.image_to_cad import _rdp_simplify as rdp
    return rdp(pts, epsilon)


def _count_concave(pts: np.ndarray) -> int:
    """Count re-entrant (concave) corners of a closed polygon.

    With v1 = prev - curr and v2 = next - curr, a convex corner of a counter-clockwise
    polygon has cross(v1, v2) < 0 and a re-entrant one > 0; clockwise flips both signs.
    Only turns sharper than 45 degrees count: measured on rasterised parts, real corners turn
    ~90 deg while RDP leftovers (e.g. the closed contour's start point) turn <= 27 deg.
    Assumes re-entrant corners are sharp; one chamfered into two <= 45 deg turns is missed.
    """
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
        norm = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        if norm < 1e-12:
            continue
        sin_turn = (v1[0] * v2[1] - v1[1] * v2[0]) / norm
        if (sin_turn if ccw else -sin_turn) > math.sin(math.radians(45.0)):
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
    except Exception as exc:
        log.warning("_fit_circle_ransac: %s: %s", type(exc).__name__, exc)
        return (*best_circle, float(best_inliers / n))


def _geometric_invariants(mask: np.ndarray) -> Dict[str, Any]:
    """Compute scale-invariant geometric descriptors from a clean binary mask."""
    if not mask.any():
        raise ValueError("the mask is empty: nothing to measure")

    ys, xs = np.nonzero(mask)
    area = float(mask.sum())
    # Outline descriptors use the hole-free silhouette: a hole is reported separately
    # and must not make a pipe look "non-solid" like an I-beam.
    filled = ndi.binary_fill_holes(mask)
    area_filled = float(filled.sum())
    cy, cx = float(ys.mean()), float(xs.mean())

    cov = np.cov(np.vstack([xs - cx, ys - cy]))
    eigvecs = np.linalg.eigh(cov)[1]
    major = eigvecs[:, 1]
    angle = math.atan2(major[1], major[0])
    ca, sa = math.cos(angle), math.sin(angle)
    proj_major = (xs - cx) * ca + (ys - cy) * sa
    proj_minor = -(xs - cx) * sa + (ys - cy) * ca
    # Near-full extents: the mask is clean, and 2-98 % clipped ~4 % off every length
    # while hole diameters were measured in full, biasing the ratios.
    len_major = float(np.percentile(proj_major, 99.5) - np.percentile(proj_major, 0.5))
    len_minor = float(np.percentile(proj_minor, 99.5) - np.percentile(proj_minor, 0.5))
    aspect = max(len_major, len_minor) / max(min(len_major, len_minor), 1.0)

    contour = _trace_contour(mask)
    if len(contour) < 4:
        contour = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)

    perim = float(np.sum(np.linalg.norm(
        np.diff(np.vstack([contour, contour[0]]), axis=0), axis=1)))
    circularity = (4.0 * math.pi * area_filled / (perim ** 2)) if perim > 1e-3 else 0.0

    from scipy.spatial import ConvexHull
    try:
        hull_area = float(ConvexHull(contour).volume)
    except Exception as exc:  # degenerate (collinear) outline
        log.warning("convex hull failed (%s: %s); solidity not measured", type(exc).__name__, exc)
        hull_area = 0.0
    solidity = min(1.0, area_filled / hull_area) if hull_area > 1e-3 else None

    # Bilateral symmetry in the principal-axis frame
    if abs(angle) > 1e-3:
        # angle is measured with y pointing down; rotating by +angle (not -angle) brings
        # the major axis onto the image x axis.
        rot = ndi.rotate(mask.astype(float), math.degrees(angle),
                         reshape=True, order=0) > 0.5
    else:
        rot = mask.astype(bool)
    coords = np.argwhere(rot)
    if len(coords) > 20:
        r0, r1 = coords[:, 0].min(), coords[:, 0].max()
        c0, c1 = coords[:, 1].min(), coords[:, 1].max()
        crop = rot[r0:r1 + 1, c0:c1 + 1]
        sym_h = float(np.mean(crop == np.fliplr(crop)))
        sym_v = float(np.mean(crop == np.flipud(crop)))
        # Cross-section height along the major axis: a stepped shaft has two levels,
        # a prismatic beam one. Ratio of the 10th to 90th percentile height (ends trimmed).
        heights = ndi.binary_fill_holes(crop).sum(axis=0).astype(float)
        trim = max(1, len(heights) // 20)
        core = heights[trim:-trim] if len(heights) > 2 * trim + 4 else heights
        step_ratio = float(np.percentile(core, 10) / max(np.percentile(core, 90), 1.0))
    else:
        sym_h = sym_v = 0.5
        step_ratio = 1.0

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
        "step_ratio": step_ratio,
        "n_concave": n_concave,
        "has_hole": len(holes_info),
        "holes": holes_info,
        "len_major": len_major,
        "len_minor": len_minor,
        "centroid": (cx, cy),
        "contour": contour,
    }


def _geometric_label(g: Dict[str, Any]) -> Tuple[Optional[str], float]:
    """Heuristic label from geometric invariants (cross-check / fallback).

    Returns ``(None, 0.0)`` when no rule matches. An unmeasured solidity (``None``) is
    NaN here, so every rule that needs it is false.
    """
    scores = {k: 0.0 for k in ("bracket", "plate_hole", "beam", "ibeam", "shaft", "pipe")}
    circ = g["circularity"]
    sol = math.nan if g["solidity"] is None else g["solidity"]
    asp = g["aspect"]
    nh = g["has_hole"]
    nc = g["n_concave"]
    hole_circ_max = max(
        (h["circularity"] for h in g.get("holes", [])), default=0.0
    )

    step = g.get("step_ratio", 1.0)
    if circ > 0.75 and nh >= 1 and hole_circ_max > 0.55:
        scores["pipe"] += 1.0
    if g["sym_h"] > 0.82 and g["sym_v"] > 0.82 and nc >= 4 and sol < 0.92 and nh == 0:
        scores["ibeam"] += 1.0
    if (g["sym_h"] < 0.80 or g["sym_v"] < 0.80) and nc >= 1 and 0.45 < sol < 0.95:
        scores["bracket"] += 0.9
    if nh >= 1 and circ < 0.75 and 1.0 <= asp <= 3.5 and sol > 0.9:
        scores["plate_hole"] += 0.9
    if asp >= 2.0 and sol > 0.8 and step < 0.85:
        scores["shaft"] += 0.95          # two diameter levels along the axis
    if asp >= 2.0 and nh == 0 and sol > 0.88 and step >= 0.85:
        scores["beam"] += 0.9

    best = max(scores, key=scores.get)
    if scores[best] <= 0.0:
        return None, 0.0
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


def _normalise_label(raw: Any) -> str:
    """Lower-case a detector phrase and undo tokenizer spacing (``"l - bracket"`` -> ``"l-bracket"``)."""
    text = str(raw).strip().lower()
    text = " ".join(text.split())
    return text.replace(" - ", "-").replace(" -", "-").replace("- ", "-")


def _map_label(raw: str) -> Optional[str]:
    """Map a Grounding DINO phrase to a canonical type, or None when it is ambiguous.

    With low text thresholds Grounding DINO can emit several prompt phrases glued
    together ("structural l-bracket rectangular beam shaft"). Such a label names
    more than one category, so it is treated as no label rather than resolved by
    whichever key happens to come first.
    """
    raw = _normalise_label(raw)
    if raw in _LABEL_TO_TYPE:
        return _LABEL_TO_TYPE[raw]
    found = []
    for key in sorted(_LABEL_TO_TYPE, key=len, reverse=True):   # longest phrase first
        if key in raw:
            found.append(_LABEL_TO_TYPE[key])
            raw = raw.replace(key, " ")
    types = set(found)
    return found[0] if len(types) == 1 else None


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

    return {}  # unclassified outline: no template parameters


# ===========================================================================
# Annotation rendering
# ===========================================================================
def _render_annotation(
    img: Image.Image,
    mask: np.ndarray,
    box: List[float],
    obj_type: str,
    confidence: float,
    prompts: Optional[List[Tuple[float, float, int]]] = None,
) -> str:
    """Return a base64 PNG data URI with mask outline + box + label badge + click markers."""
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated, "RGBA")

    contour = _trace_contour(mask)
    if len(contour) > 2:
        # The tracer returns y-up coordinates; the overlay is drawn in image rows.
        h = mask.shape[0]
        pts = [(int(p[0]), int(h - 1 - p[1])) for p in contour]
        draw.polygon(pts, outline=(0, 255, 120, 240), width=3)

    x1, y1, x2, y2 = (int(v) for v in box)
    draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 180, 0, 220), width=2)

    badge = f"{_FRIENDLY_NAMES.get(obj_type, 'Unclassified')} | {int(confidence * 100)}%"
    bw = len(badge) * 7 + 12
    bx = max(x1, 4)
    by = max(y1 - 26, 4)
    draw.rectangle([(bx, by), (bx + bw, by + 22)], fill=(15, 23, 42, 230))
    draw.text((bx + 6, by + 4), badge, fill=(255, 255, 255, 255))
    r = max(6, int(round(0.009 * max(annotated.size))))
    for x, y, label in prompts or []:   # green dot = part, red cross = not the part
        if label:
            draw.ellipse([(x - r, y - r), (x + r, y + r)], fill=(34, 197, 94, 255), outline=(255, 255, 255, 255), width=2)
        else:
            draw.line([(x - r, y - r), (x + r, y + r)], fill=(239, 68, 68, 255), width=4)
            draw.line([(x - r, y + r), (x + r, y - r)], fill=(239, 68, 68, 255), width=4)

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
    # Local contrast against a wide Gaussian also fires on the backdrop next to every
    # strong edge, inflating the silhouette by tens of pixels. Use it only when colour
    # distance alone cannot find the part (part colour close to the backdrop).
    colour_area = float(ndi.binary_opening(mask_color, structure=np.ones((3, 3), dtype=bool)).mean())
    mask = mask_color if 0.005 <= colour_area <= 0.9 else (mask_color | mask_local)

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

    return mask & ~_interior_holes(rgb, mask, background)


def _interior_holes(rgb: np.ndarray, mask: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Enclosed regions of ``mask`` whose colour is nearer the background than the part.

    A through-hole shows whatever is behind the part: rarely the exact backdrop
    colour (shadow, a different surface), but nearly always closer to it than to the
    part. The old rule (|grey - backdrop| < 15) missed such holes completely.
    """
    if not mask.any():
        return np.zeros_like(mask)
    # Sample the part just inside its outer silhouette: a large hole can be most of
    # the mask, so a median over the whole mask may be the hole's colour.
    # The mask edge carries a few pixels of backdrop halo, so use a band 4-9 px inside.
    solid = ndi.binary_fill_holes(mask)
    rim = ndi.binary_erosion(solid, iterations=4) & ~ndi.binary_erosion(solid, iterations=9) & mask
    if rim.sum() >= 20:
        part_colour = np.median(rgb[rim], axis=0)
    else:                                   # thin part: the pixels least like the backdrop
        d = np.linalg.norm(rgb[mask] - background[None, :], axis=1)
        part_colour = np.median(rgb[mask][d >= np.median(d)], axis=0)
    d_bg = np.linalg.norm(rgb - background[None, None, :], axis=2)
    d_part = np.linalg.norm(rgb - part_colour[None, None, :], axis=2)
    if float(np.linalg.norm(part_colour - background)) < 25.0:
        return np.zeros_like(mask)           # part and backdrop too similar to tell
    candidate = mask & (d_bg < d_part)
    candidate = ndi.binary_opening(candidate, structure=np.ones((3, 3), dtype=bool))
    labels, count = ndi.label(candidate)
    if count == 0:
        return np.zeros_like(mask)
    min_size = max(30, int(mask.sum() * 0.002))
    sizes = np.bincount(labels.ravel())
    holes = np.zeros_like(mask)
    outside = ~ndi.binary_dilation(mask, iterations=1)
    for k in range(1, count + 1):
        if sizes[k] < min_size:
            continue
        region = labels == k
        # Must be enclosed: its 2-px ring may not touch the exterior background.
        ring = ndi.binary_dilation(region, iterations=2) & ~region
        if not (ring & outside).any():
            holes |= region
    return holes


def _classical_segment(img: Image.Image) -> Tuple[np.ndarray, bool, str]:
    """Classical mask plus whether it can be trusted without the foundation models.

    Trusted only when the image border (the backdrop) is nearly uniform and one
    clean, interior part results. Cluttered or strongly lit scenes go to DINO + SAM.
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = rgb.shape[:2]
    band = max(2, int(round(min(h, w) * 0.035)))
    border = np.concatenate((
        rgb[:band].reshape(-1, 3), rgb[-band:].reshape(-1, 3),
        rgb[:, :band].reshape(-1, 3), rgb[:, -band:].reshape(-1, 3),
    ), axis=0)
    spread = float(np.percentile(np.linalg.norm(border - np.median(border, axis=0), axis=1), 95))
    mask = _fallback_segment(img)
    area = float(mask.mean())
    if not mask.any():
        return mask, False, "no foreground found"
    ys, xs = np.nonzero(mask)
    touches_edge = ys.min() <= 1 or xs.min() <= 1 or ys.max() >= h - 2 or xs.max() >= w - 2
    solid = float(mask.sum()) / max(1.0, float(ndi.binary_fill_holes(mask).sum()))
    trusted = spread < 30.0 and 0.01 <= area <= 0.85 and not touches_edge and solid > 0.3
    note = (f"backdrop spread {spread:.0f}, foreground {area:.0%}"
            + (", touches image edge" if touches_edge else ""))
    return mask, trusted, note



# ===========================================================================
# Point-prompted object selection (the user says which object is the part)
# ===========================================================================
def _box_iou(a, b) -> float:
    """Intersection over union of two (x1, y1, x2, y2) boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _touched_sides(mask: np.ndarray) -> int:
    """How many image borders (0-4) the mask touches."""
    return int(mask[0].any()) + int(mask[-1].any()) + int(mask[:, 0].any()) + int(mask[:, -1].any())


def _skin_fraction(rgb: np.ndarray, mask: np.ndarray) -> float:
    """Share of mask pixels with skin chroma (YCrCb Cr 135-175, Cb 85-135): hands holding parts."""
    if not mask.any():
        return 0.0
    px = rgb[mask].astype(np.float32)
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    return float(np.mean((cr >= 135) & (cr <= 175) & (cb >= 85) & (cb <= 135)))


def default_candidate(masks: List[np.ndarray], scores: Optional[List[float]] = None) -> int:
    """Index of the mask to show first among SAM's nested candidates (sorted small -> large).

    The largest candidate that touches at most one image border, covers at most 60 % of the
    frame and scores at least 70 % of SAM's best score. SAM's top score tends to prefer one face
    of a part over the whole part (measured on a real photo: whole part 0.39 vs face 0.41), while
    its junk candidates score far lower (0.40 vs 0.99 on a synthetic scene). If every candidate is
    background-like, the best-scored one is used.
    """
    top = max(scores) if scores else 0.0
    ok = [i for i, m in enumerate(masks)
          if _touched_sides(m) <= 1 and 0 < m.mean() <= 0.6
          and (not scores or scores[i] >= 0.7 * top)]
    if ok:
        return ok[-1]
    return int(np.argmax(scores)) if scores else len(masks) - 1


def _classical_point_masks(img: Image.Image, points, labels) -> List[np.ndarray]:
    """Without SAM: the classical foreground components under the positive clicks."""
    comp, _ = ndi.label(_fallback_segment(img))
    pos = {int(comp[int(y), int(x)]) for (x, y), l in zip(points, labels) if l}
    neg = {int(comp[int(y), int(x)]) for (x, y), l in zip(points, labels) if not l}
    keep = sorted((pos - neg) - {0})
    if not keep:
        raise ValueError("no foreground object under the clicked point")
    return [np.isin(comp, keep)]


def segment_from_points(
    image_input: Union[str, Path, Image.Image, np.ndarray, bytes],
    points: List[Tuple[float, float]],
    labels: List[int],
) -> Tuple[List[np.ndarray], List[float], str]:
    """Masks for the object marked by clicks (label 1 = part, 0 = not part), image pixels.

    Returns SAM's nested candidates sorted from smallest to largest, their scores, and the
    segmenter name. Without SAM, the classical foreground component under the clicks.
    """
    img = image_input if isinstance(image_input, Image.Image) else _load_image(image_input)
    if not points or not any(labels):
        raise ValueError("click at least once on the part")
    if HAS_FOUNDATION:
        try:
            proc, mdl, dev, kind = _get_segmenter()
            inputs = proc(images=img.convert("RGB"),
                          input_points=[[[[float(x), float(y)] for x, y in points]]],
                          input_labels=[[[int(v) for v in labels]]], return_tensors="pt").to(dev)
            with torch.inference_mode():
                out = mdl(**inputs, multimask_output=True)
            masks = _sam_masks(proc, kind, out, inputs).reshape(-1, img.height, img.width)
            scores = out.iou_scores.cpu().numpy().reshape(-1)[:len(masks)].tolist()
            order = np.argsort([m.sum() for m in masks])
            masks = [masks[i] for i in order if masks[i].any()]
            if masks:
                return masks, [float(scores[i]) for i in order if scores], f"{kind.upper()} point prompt"
        except CVModelUnavailableError as exc:
            log.warning("SAM unavailable for point prompts (%s); using classical components", exc)
    return _classical_point_masks(img, points, labels), [1.0], "classical component under the click"


def suggest_object_point(image_input: Union[str, Path, Image.Image, np.ndarray, bytes]) -> Tuple[float, float]:
    """A first guess at where the part is, for the user to confirm or correct.

    SAM is prompted at a 5 x 3 grid over the central half of the frame (one batched pass).
    Each grid point's default candidate is scored: central (+), touching image borders (-,
    backgrounds such as walls and whiteboards), skin-coloured (-, the hand holding the part),
    tiny (< 0.3 %) or huge (> 40 %) (-). Without SAM, the image centre.
    """
    img = image_input if isinstance(image_input, Image.Image) else _load_image(image_input)
    w, h = img.size
    grid = [(w * fx, h * fy) for fy in (0.3, 0.5, 0.7) for fx in (0.25, 0.375, 0.5, 0.625, 0.75)]
    if not HAS_FOUNDATION:
        return w / 2.0, h / 2.0
    try:
        proc, mdl, dev, kind = _get_segmenter()
        inputs = proc(images=img.convert("RGB"), input_points=[[[[x, y]] for x, y in grid]],
                      input_labels=[[[1] for _ in grid]], return_tensors="pt").to(dev)
        with torch.inference_mode():
            out = mdl(**inputs, multimask_output=True)
        masks = _sam_masks(proc, kind, out, inputs).reshape(len(grid), -1, h, w)
        all_scores = out.iou_scores.cpu().numpy().reshape(len(grid), -1)
    except CVModelUnavailableError as exc:
        log.warning("SAM unavailable for the object suggestion (%s); using the image centre", exc)
        return w / 2.0, h / 2.0
    rgb = np.asarray(img.convert("RGB"))
    best, best_score = grid[len(grid) // 2], -np.inf
    for (x, y), cands, sc in zip(grid, masks, all_scores):
        pairs = sorted(((m, float(v)) for m, v in zip(cands, sc) if m.any()), key=lambda p: p[0].sum())
        if not pairs:
            continue
        cands, sc = [p[0] for p in pairs], [p[1] for p in pairs]
        m = cands[default_candidate(cands, sc)]
        ys, xs = np.nonzero(m)
        centre = 1.0 - math.hypot(xs.mean() / w - 0.5, ys.mean() / h - 0.5) / math.hypot(0.5, 0.5)
        area = float(m.mean())
        score = (centre - 1.0 * _touched_sides(m) - 2.0 * _skin_fraction(rgb, m)
                 - (1.0 if area < 0.003 or area > 0.4 else 0.0))
        if score > best_score:
            best, best_score = (x, y), score
    return float(best[0]), float(best[1])


# ===========================================================================
# Public API
# ===========================================================================
def _result_from_mask(
    img: Image.Image,
    mask: np.ndarray,
    reference_scale: float,
    segmentation: str,
    det: Optional[Dict[str, Any]] = None,
    detections: Optional[List[Dict[str, Any]]] = None,
    mask_is_real: bool = True,
    notes: Optional[List[str]] = None,
    prompts: Optional[List[Tuple[float, float, int]]] = None,
) -> CVAnalysisResult:
    """Classify a measured mask, reconcile with an optional detector label, build the result."""
    w, h = img.size
    inv = _geometric_invariants(mask)
    geo_type, geo_conf = _geometric_label(inv)
    det_type = _map_label(det["label"]) if det else None
    features = list(notes or [])
    features.append(f"Segmentation: {segmentation}")

    if det is None or det_type is None:
        obj_type, confidence = geo_type, min(0.99, 0.5 + 0.5 * geo_conf)
        if det is not None:
            features.append(f"Detector label '{det['label']}' is ambiguous; geometry decides.")
        if geo_type is None:
            confidence = 0.0
            features.append("The outline matches none of the template parts; mesh the outline "
                            "directly (no template CAD).")
    elif det_type == geo_type:
        obj_type = det_type
        confidence = min(0.99, 0.6 + 0.25 * det["score"] + 0.2 * geo_conf)
    elif mask_is_real and geo_conf >= 0.7:
        # Zero-shot labels for these categories are weak; a clean mask is strong evidence.
        obj_type, confidence = geo_type, min(0.95, 0.45 + 0.4 * geo_conf)
        features.append(f"Geometry ({geo_type}) overrides detector label '{det['label']}'.")
    else:
        obj_type, confidence = det_type, min(0.9, 0.3 + 0.5 * det["score"] + 0.1 * geo_conf)

    if det is not None:
        features.append(f"Detector label: '{det['label']}' (score {det['score']:.2f})")
    features += [f"Mask aspect ratio: {inv['aspect']:.2f}", f"Mask solidity: {inv['solidity']:.2f}"]
    if inv["has_hole"]:
        features.append(f"{inv['has_hole']} interior hole(s) verified via RANSAC circle fit")
    if inv["n_concave"]:
        features.append(f"{inv['n_concave']} concave corner(s)")

    if mask.any():
        ys, xs = np.nonzero(mask)
        box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    else:
        box = [0.0, 0.0, float(w), float(h)]
    if det is not None:
        box = det["box"]
    params = _extract_parameters(obj_type, mask, inv, reference_scale)
    return CVAnalysisResult(
        object_type=obj_type,
        friendly_name=_FRIENDLY_NAMES.get(obj_type, "Unclassified outline"),
        confidence=float(confidence),
        detected_features=features,
        parameters=params,
        pixel_dimensions=(w, h),
        annotated_image_b64=_render_annotation(img, mask, box, obj_type, confidence, prompts),
        all_scores=({d["label"]: d["score"] for d in detections[:5]} if detections
                    else {geo_type: geo_conf}),
        mask=mask,
        segmentation=segmentation,
        prompt_points=list(prompts or []),
    )


def analyze_image_object(
    image_input: Union[str, Path, Image.Image, np.ndarray, bytes],
    reference_scale: float = 0.30,
    use_models: Optional[bool] = None,
    points: Optional[List[Tuple[float, float]]] = None,
    labels: Optional[List[int]] = None,
    candidate: Optional[int] = None,
) -> CVAnalysisResult:
    """Analyze a photograph and return a part category, a clean mask and metric parameters.

    Parameters
    ----------
    image_input     Path, PIL Image, numpy array, bytes, or base64 data URI.
    reference_scale Length of the largest bounding dimension in metres.
    use_models      None (default): use the models only when the cheap classical mask is
                    not trustworthy. True: always use them when installed. False: never.
    points, labels  Clicks in image pixels marking the part (label 1) and things that are
                    not the part (label 0). They decide the object; nothing else does.
    candidate       Which of SAM's nested masks to use (0 = smallest); default: the largest
                    one that is not background-like (:func:`default_candidate`).

    Without clicks, a plain backdrop uses the classical mask; otherwise the object is
    *suggested* by :func:`suggest_object_point` and must be confirmed by the user (the app
    does not mesh an unconfirmed mask). Grounding DINO only proposes a category name.

    Returns
    -------
    CVAnalysisResult with the primary detection, confidence, mask and parameters.
    """
    img = _load_image(image_input)
    notes: List[str] = []
    guessed = None
    if not points:
        classical, trusted, note = _classical_segment(img)
        if trusted and use_models is not True:
            return _result_from_mask(img, classical, reference_scale,
                                     "classical (uniform backdrop; foundation models skipped)",
                                     notes=[f"Fast path: {note}"])
        if HAS_FOUNDATION and use_models is not False:
            guess = suggest_object_point(img)
            try:
                guessed = segment_from_points(img, [guess], [1])
                points, labels = [guess], [1]
                notes.append("Object guessed automatically (central, not touching the frame, "
                             "not skin). Click the part to change it.")
            except ValueError as exc:
                notes.append(f"No object found automatically ({exc}); click the part.")
    if points:
        labels = list(labels) if labels is not None else [1] * len(points)
        masks, scores, how = guessed or segment_from_points(img, points, labels)
        if candidate is not None:
            idx = int(candidate) % len(masks)
        else:
            idx = default_candidate(masks, scores)
        mask = masks[idx]
        det, detections = None, []
        if HAS_FOUNDATION and use_models is not False:
            try:   # category name only: the detection that overlaps the selected mask
                detections = _detect_parts(img)
                ys, xs = np.nonzero(mask)
                mb = (xs.min(), ys.min(), xs.max(), ys.max())
                det = max(detections, key=lambda d: _box_iou(d["box"], mb), default=None)
                if det is not None and _box_iou(det["box"], mb) < 0.5:
                    det = None
            except Exception as exc:
                notes.append(f"Category detector unavailable: {type(exc).__name__}: {exc}")
        n_clicks = len(points)
        res = _result_from_mask(img, mask, reference_scale,
                                f"{how} ({n_clicks} click{'s' if n_clicks != 1 else ''}, "
                                f"mask {idx + 1} of {len(masks)})",
                                det=det, detections=detections, notes=notes, prompts=list(
                                    zip([float(x) for x, _ in points], [float(y) for _, y in points],
                                        [int(v) for v in labels])))
        res.candidate_index, res.n_candidates = idx, len(masks)
        return res
    classical, trusted, note = _classical_segment(img)
    reasons: List[str] = notes
    if use_models is not False and not HAS_FOUNDATION:
        reasons.append("Foundation models (torch + transformers) are not installed.")

    # Fallback: rembg if installed, else the classical mask computed above.
    segmentation = f"classical ({note})"
    mask = classical
    try:
        from rembg import remove
        output = remove(img, only_mask=True)
        if isinstance(output, bytes):
            output = Image.open(io.BytesIO(output))
        candidate = np.asarray(output.convert("L")) > 128
        if 0.01 < candidate.mean() < 0.98:
            mask, segmentation = candidate, "neural foreground segmentation (rembg)"
        else:
            reasons.append("rembg mask was empty or covered the whole frame; kept the classical mask.")
    except ImportError:
        reasons.append("rembg is not installed; using the classical mask.")
    except Exception as exc:
        reasons.append(f"rembg failed ({type(exc).__name__}: {exc}); using the classical mask.")
    if not mask.any():
        raise ValueError("Could not isolate an object; use a plain background. "
                         + " ".join(reasons))
    return _result_from_mask(img, mask, reference_scale, segmentation, notes=reasons)


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
