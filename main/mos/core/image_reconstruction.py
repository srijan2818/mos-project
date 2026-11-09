"""Single-image 2.5D reconstruction for the MoS CAD preview workflow.

A single photograph cannot reveal hidden/back-side geometry.  This module therefore
creates a colored, closed relief mesh from a foreground mask and (optionally) a
monocular *relative* depth estimate.  It can also export a separate silhouette-
extrusion STEP approximation when Gmsh is available.  The result is suitable for
visual inspection and rough shape studies, not dimensional inspection or direct
engineering validation without manual review.

The depth model and background-removal model are optional and loaded lazily.
"""
from __future__ import annotations

import base64
import io
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage as ndi


DEFAULT_DEPTH_MODEL = os.environ.get(
    "MOS_DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Small-hf"
)

_DEPTH_PIPELINE: Any = None
_DEPTH_LOCK = threading.Lock()
_REMBG_SESSION: Any = None
_REMBG_LOCK = threading.Lock()


@dataclass
class ImageReconstructionResult:
    """Generated files and diagnostics for one image reconstruction."""

    source_name: str
    preview_mesh_path: Path
    mask_path: Path
    depth_preview_path: Optional[Path]
    step_path: Optional[Path]
    width_m: float
    height_m: float
    thickness_m: float
    vertex_count: int
    face_count: int
    segmentation_method: str
    depth_method: str
    warnings: list[str]


def _load_image(source: Any) -> tuple[str, Image.Image]:
    """Load a path, byte stream, PIL image, browser data URI, or named image mapping."""
    name = "uploaded_image"
    if isinstance(source, Mapping):
        name = str(source.get("name") or name)
        source = source.get("data", source.get("path", source.get("image")))
        if source is None:
            raise ValueError("Image mapping must include 'data', 'path', or 'image'.")

    if isinstance(source, Image.Image):
        image = source.copy()
    elif isinstance(source, (bytes, bytearray, memoryview)):
        image = Image.open(io.BytesIO(bytes(source)))
    elif isinstance(source, (str, Path)):
        value = str(source)
        if value.startswith("data:image"):
            try:
                raw = base64.b64decode(value.split(",", 1)[1], validate=True)
            except Exception as exc:
                raise ValueError(f"Could not decode uploaded image: {exc}") from exc
            image = Image.open(io.BytesIO(raw))
        else:
            path = Path(value).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Image file not found: {path}")
            image = Image.open(path)
            if name == "uploaded_image":
                name = path.name
    else:
        raise TypeError("Image input must be a path, bytes, data URI, PIL image, or named image mapping.")

    image = ImageOps.exif_transpose(image)
    if image.width < 64 or image.height < 64:
        raise ValueError("Image must be at least 64 × 64 pixels.")
    if image.width * image.height > 40_000_000:
        raise ValueError("Image is too large (maximum 40 megapixels). Resize it before uploading.")
    return name, image


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep the largest connected foreground component while preserving holes."""
    mask = np.asarray(mask, dtype=bool)
    mask = ndi.binary_closing(mask, structure=np.ones((5, 5), dtype=bool), iterations=1)
    labels, count = ndi.label(mask)
    if count <= 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    selected = int(np.argmax(sizes))
    # Preserve enclosed transparent regions: they may be real through-holes.
    return labels == selected


def _segment_foreground(image: Image.Image, mode: str = "auto") -> tuple[np.ndarray, str, list[str]]:
    """Estimate a foreground mask using alpha, optional rembg, then border color."""
    warnings: list[str] = []
    mode = str(mode or "auto").lower()
    if mode not in {"auto", "full_frame"}:
        raise ValueError("segmentation_mode must be 'auto' or 'full_frame'.")
    rgba = image.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8)
    if mode == "auto" and np.any(alpha < 250):
        alpha_mask = alpha > 12
        alpha_area = float(alpha_mask.mean())
        if alpha_area < 0.01:
            raise ValueError("Could not isolate any visible foreground pixels from the transparent image.")
        mask = _largest_component(alpha_mask)
        if mask.mean() >= 0.01 and mask.mean() <= 0.98:
            return mask, "image alpha channel", warnings

    if mode == "auto":
        # rembg is an optional, high-quality background-removal backend.
        try:
            from rembg import new_session, remove
            global _REMBG_SESSION
            with _REMBG_LOCK:
                if _REMBG_SESSION is None:
                    _REMBG_SESSION = new_session()
                session = _REMBG_SESSION
            mask_image = remove(image.convert("RGB"), session=session, only_mask=True)
            if isinstance(mask_image, (bytes, bytearray)):
                mask_image = Image.open(io.BytesIO(mask_image))
            if isinstance(mask_image, Image.Image):
                mask = np.asarray(mask_image.convert("L"), dtype=np.uint8) > 96
                mask = _largest_component(mask)
                if 0.01 <= float(mask.mean()) <= 0.98:
                    return mask, "rembg foreground segmentation", warnings
        except ImportError:
            warnings.append("rembg is not installed; automatic masking used the image border colors. For busy backgrounds, install rembg[cpu].")
        except Exception as exc:
            warnings.append(f"rembg segmentation was unavailable ({type(exc).__name__}); used border-color masking instead.")

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    if mode == "full_frame":
        return np.ones(rgb.shape[:2], dtype=bool), "full image frame", warnings

    # Background-color estimation from the outer 4% border. This is dependable
    # for studio/white backgrounds, but deliberately reports its heuristic nature.
    h, w = rgb.shape[:2]
    band = max(2, int(round(min(h, w) * 0.04)))
    border = np.concatenate((
        rgb[:band, :, :].reshape(-1, 3),
        rgb[-band:, :, :].reshape(-1, 3),
        rgb[:, :band, :].reshape(-1, 3),
        rgb[:, -band:, :].reshape(-1, 3),
    ), axis=0)
    background = np.median(border, axis=0)
    border_distance = np.linalg.norm(border - background[None, :], axis=1)
    spread = float(np.median(np.abs(border_distance - np.median(border_distance))))
    threshold = float(np.clip(np.percentile(border_distance, 92) + max(14.0, 3.0 * spread), 22.0, 72.0))
    distance = np.linalg.norm(rgb - background[None, None, :], axis=2)
    candidate = distance > threshold
    candidate = ndi.binary_opening(candidate, structure=np.ones((3, 3), dtype=bool), iterations=1)
    mask = _largest_component(candidate)
    area = float(mask.mean())
    if not 0.01 <= area <= 0.96:
        # Predictable central fallback; the UI exposes this as an approximation.
        yy, xx = np.ogrid[:h, :w]
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        rx, ry = max(2.0, 0.40 * w), max(2.0, 0.43 * h)
        mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
        warnings.append("Foreground mask was ambiguous; used a centered elliptical mask. Review the preview before exporting CAD.")
        method = "centered fallback mask"
    else:
        warnings.append("Background removal used border-color contrast; complex backgrounds may need a transparent PNG or rembg.")
        method = "border-color foreground estimate"
    return mask, method, warnings


def _get_depth_pipeline() -> Any:
    """Load and cache the optional Depth Anything V2 model pipeline."""
    global _DEPTH_PIPELINE
    if _DEPTH_PIPELINE is None:
        with _DEPTH_LOCK:
            if _DEPTH_PIPELINE is None:
                try:
                    import torch
                    from transformers import pipeline
                except ImportError as exc:
                    raise RuntimeError(
                        "AI depth estimation needs torch and transformers. Install the project requirements, "
                        "then retry; silhouette-only reconstruction does not need these model weights."
                    ) from exc
                device = 0 if torch.cuda.is_available() else -1
                try:
                    _DEPTH_PIPELINE = pipeline(
                        "depth-estimation", model=DEFAULT_DEPTH_MODEL, device=device
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Could not load depth model {DEFAULT_DEPTH_MODEL!r}. Check network access on first run "
                        f"or choose silhouette-only mode. Details: {exc}"
                    ) from exc
    return _DEPTH_PIPELINE


def _estimate_depth(image: Image.Image, size: tuple[int, int]) -> tuple[np.ndarray, str]:
    """Return normalized relative inverse depth at the requested (width, height)."""
    pipe = _get_depth_pipeline()
    try:
        prediction = pipe(image.convert("RGB"))
        depth_image = prediction.get("depth")
        if depth_image is None:
            raise RuntimeError("Depth model did not return a depth image.")
        depth_image = depth_image.resize(size, Image.Resampling.BICUBIC)
        depth = np.asarray(depth_image, dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(f"Depth estimation failed: {exc}") from exc
    finite = np.isfinite(depth)
    if not finite.any():
        raise RuntimeError("Depth model returned no finite depth values.")
    lo, hi = np.percentile(depth[finite], [2.0, 98.0])
    if hi - lo < 1e-6:
        normalized = np.zeros_like(depth, dtype=np.float32)
    else:
        normalized = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    normalized[~finite] = 0.0
    normalized = ndi.gaussian_filter(normalized, sigma=1.0)
    return normalized.astype(np.float32), f"Depth Anything V2 ({DEFAULT_DEPTH_MODEL}; relative depth)"


def _write_mask_image(mask: np.ndarray, path: Path) -> Path:
    """Write a black-foreground, white-background mask for the STEP contourer."""
    pixels = np.where(mask, 0, 255).astype(np.uint8)
    Image.fromarray(pixels, mode="L").save(path)
    return path


def _write_mesh_ply(
    path: Path,
    rgb: np.ndarray,
    mask: np.ndarray,
    depth: np.ndarray,
    width_m: float,
    height_m: float,
    thickness_m: float,
    relief_depth_m: float,
) -> tuple[int, int]:
    """Build a closed front-relief/back-plane mesh and write colored binary PLY."""
    h, w = mask.shape
    valid = (mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:])
    cell_count = int(valid.sum())
    if cell_count < 30:
        raise ValueError("The foreground mask is too small or fragmented to create a useful mesh. Use a clearer image or transparent background.")

    # Only vertices referenced by complete foreground cells are retained.
    used = np.zeros_like(mask, dtype=bool)
    rr, cc = np.where(valid)
    used[rr, cc] = True
    used[rr + 1, cc] = True
    used[rr, cc + 1] = True
    used[rr + 1, cc + 1] = True
    rows, cols = np.where(used)
    n = len(rows)
    index_map = np.full((h, w), -1, dtype=np.int64)
    index_map[rows, cols] = np.arange(n, dtype=np.int64)

    x = (cols.astype(np.float32) / max(w - 1, 1) - 0.5) * float(width_m)
    y = (0.5 - rows.astype(np.float32) / max(h - 1, 1)) * float(height_m)
    z = depth[rows, cols].astype(np.float32) * float(relief_depth_m)
    front = np.column_stack((x, y, z)).astype(np.float32)
    back = np.column_stack((x, y, np.full(n, -float(thickness_m), dtype=np.float32))).astype(np.float32)
    vertices = np.vstack((front, back))

    colors = rgb[rows, cols].astype(np.uint8)
    colors = np.vstack((colors, colors))

    # Vectorised over all foreground cells (the per-cell Python loop was the hot spot).
    tl = index_map[rr, cc]
    tr = index_map[rr, cc + 1]
    bl = index_map[rr + 1, cc]
    br = index_map[rr + 1, cc + 1]
    # Front triangles (normal toward +Z), then reversed back triangles.
    tris = np.concatenate([
        np.column_stack((tl, bl, tr)), np.column_stack((tr, bl, br)),
        np.column_stack((tl + n, tr + n, bl + n)), np.column_stack((tr + n, br + n, bl + n)),
    ])
    padded = np.pad(valid, 1)                      # padded[r + 1, c + 1] == valid[r, c]
    up = ~padded[rr, cc + 1]                       # no foreground cell above
    down = ~padded[rr + 2, cc + 1]
    left = ~padded[rr + 1, cc]
    right = ~padded[rr + 1, cc + 2]
    # Close every boundary edge against the planar back face.
    quads = np.concatenate([
        np.column_stack((tl, tr, tr + n, tl + n))[up],
        np.column_stack((br, bl, bl + n, br + n))[down],     # wound so normals point outward
        np.column_stack((bl, tl, tl + n, bl + n))[left],
        np.column_stack((tr, br, br + n, tr + n))[right],
    ])

    path.parent.mkdir(parents=True, exist_ok=True)
    # Binary little-endian PLY supports vertex colors and polygon faces.
    vertex_dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    records = np.empty(len(vertices), dtype=vertex_dtype)
    for axis, name in enumerate(("x", "y", "z")):
        records[name] = vertices[:, axis]
    records["red"], records["green"], records["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    with path.open("wb") as stream:
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(vertices)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            f"element face {len(tris) + len(quads)}\n"
            "property list uchar int vertex_indices\nend_header\n"
        )
        stream.write(header.encode("ascii"))
        records.tofile(stream)
        for block, count in ((tris, 3), (quads, 4)):
            rec = np.empty(len(block), dtype=np.dtype([("n", "u1"), ("v", "<i4", (count,))]))
            rec["n"] = count
            rec["v"] = block
            rec.tofile(stream)
    return len(vertices), len(tris) + len(quads)


def reconstruct_single_image(
    image_source: Any,
    output_dir: Union[str, Path],
    *,
    object_width_m: float = 0.20,
    body_thickness_m: float = 0.025,
    relief_depth_m: float = 0.045,
    max_resolution: int = 192,
    geometry_mode: str = "depth_relief",
    segmentation_mode: str = "auto",
    export_step: bool = True,
) -> ImageReconstructionResult:
    """Create an inspectable single-image 2.5D mesh and optional STEP silhouette.

    ``geometry_mode`` may be ``depth_relief`` (uses Depth Anything V2 relative
    depth) or ``silhouette`` (flat extruded relief).  STEP export is always a
    silhouette extrusion and is intentionally separate from the relief preview.
    ``object_width_m`` only sets approximate scene scale; monocular depth is not
    metric depth.
    """
    name, original = _load_image(image_source)
    width_m = float(object_width_m)
    thickness_m = float(body_thickness_m)
    relief_m = float(relief_depth_m)
    if not np.isfinite([width_m, thickness_m, relief_m]).all() or width_m <= 0 or thickness_m <= 0 or relief_m < 0:
        raise ValueError("Object width and thickness must be positive; relief depth cannot be negative.")
    if not 96 <= int(max_resolution) <= 384:
        raise ValueError("Mesh resolution must be between 96 and 384 samples on the longest edge.")
    geometry_mode = str(geometry_mode).lower()
    if geometry_mode not in {"depth_relief", "silhouette"}:
        raise ValueError("geometry_mode must be 'depth_relief' or 'silhouette'.")

    mask, segmentation_method, warnings = _segment_foreground(original, segmentation_mode)
    # Downsample consistently for a bounded mesh footprint.
    # One uniform scale: the metric height is derived from this aspect ratio.
    scale = min(1.0, float(max_resolution) / max(original.size))
    out_size = (int(round(original.width * scale)), int(round(original.height * scale)))
    if min(out_size) < 8:
        raise ValueError(f"The image ({original.width} x {original.height} px) is too elongated for a "
                         f"{max_resolution}-sample mesh; crop it closer to the object.")
    image = original.convert("RGB").resize(out_size, Image.Resampling.LANCZOS)
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    mask_small = np.asarray(mask_image.resize(out_size, Image.Resampling.NEAREST), dtype=np.uint8) > 127
    mask_small = _largest_component(mask_small)
    if str(segmentation_mode).lower() != "full_frame":
        if float(mask_small.mean()) < 0.01 or float(mask_small.mean()) > 0.97:
            raise ValueError("Could not isolate the object from its background. Try a transparent PNG, a plain background, or rembg[cpu].")

    width_px, height_px = out_size
    height_m = width_m * height_px / max(width_px, 1)
    if geometry_mode == "depth_relief":
        depth, depth_method = _estimate_depth(image, out_size)
    else:
        depth = np.zeros((height_px, width_px), dtype=np.float32)
        depth_method = "flat silhouette (no learned depth)"

    rgb = np.asarray(image, dtype=np.uint8)
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stem = Path(name).stem or "image_object"
    safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)[:48] or "image_object"
    run_stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000_000:09d}"
    safe_stem = f"{safe_stem}_{run_stamp}"
    preview_path = output_root / f"{safe_stem}_2_5d_preview.ply"
    mask_path = output_root / f"{safe_stem}_foreground_mask.png"
    depth_path: Optional[Path] = None
    step_path: Optional[Path] = None

    vertices, faces = _write_mesh_ply(
        preview_path, rgb, mask_small, depth, width_m, height_m,
        thickness_m, relief_m if geometry_mode == "depth_relief" else 0.0,
    )
    _write_mask_image(mask_small, mask_path)

    if geometry_mode == "depth_relief":
        depth_path = output_root / f"{safe_stem}_relative_depth.png"
        depth_u8 = np.clip(depth * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(depth_u8, mode="L").save(depth_path)

    if export_step:
        # A silhouette extrusion is a robust CAD solid approximation. It does not
        # pretend the monocular relief mesh is a parametric STEP surface.
        try:
            from mos.core.image_to_cad import HAS_GMSH, image_to_step
            if HAS_GMSH:
                step_path = output_root / f"{safe_stem}_silhouette.step"
                image_to_step(
                    mask_path, step_path, thickness=thickness_m,
                    scale_x=width_m, scale_y=height_m, threshold=128,
                    simplify_tol=max(1.0, max(width_px, height_px) / 180.0),
                )
            else:
                warnings.append("Gmsh is unavailable; generated the colored PLY preview only. Install Gmsh to export STEP.")
        except Exception as exc:
            warnings.append(f"STEP silhouette export failed; the preview mesh is still available: {exc}")

    if geometry_mode == "depth_relief":
        warnings.append("Single-image depth is relative, not metric; the hidden/back-side shape is an extrusion approximation.")
    else:
        warnings.append("Silhouette mode creates a constant-thickness extrusion; it does not recover surface depth.")
    if step_path is not None:
        warnings.append("STEP export follows the largest silhouette and its through-holes; it is a constant-thickness extrusion.")
    warnings.append("Review and edit the geometry before using it for engineering calculations.")

    return ImageReconstructionResult(
        source_name=name,
        preview_mesh_path=preview_path,
        mask_path=mask_path,
        depth_preview_path=depth_path,
        step_path=step_path,
        width_m=width_m,
        height_m=height_m,
        thickness_m=thickness_m,
        vertex_count=vertices,
        face_count=faces,
        segmentation_method=segmentation_method,
        depth_method=depth_method,
        warnings=warnings,
    )
