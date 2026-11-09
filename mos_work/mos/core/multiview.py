"""Multi-view capture, NeRF/Gaussian-splatting dataset preparation, and PLY tools.

This module deliberately keeps neural reconstruction dependencies optional.  It
can package and assess image captures with the base project dependencies, then
optionally delegate camera-pose recovery to Nerfstudio/COLMAP when installed.
It also extracts the Gaussian means from common 3D Gaussian Splatting PLY files
as a conventional colored point cloud; those means are not a watertight CAD
surface by themselves.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage as ndi


@dataclass
class CaptureSession:
    """Description of a saved multi-view image capture."""

    session_dir: Path
    images_dir: Path
    manifest_path: Path
    image_count: int
    summary: str
    warnings: List[str]


@dataclass
class GaussianPlySummary:
    """Statistics for a Gaussian PLY and its exported center point cloud."""

    source_path: Path
    centers_path: Path
    gaussian_count: int
    bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
    centroid: Tuple[float, float, float]
    mean_opacity: Optional[float]
    mean_scale: Optional[float]
    has_gaussian_attributes: bool


_PLY_SCALARS = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}
_SH_C0 = 0.28209479177387814


def _decode_image_item(item: Any) -> Tuple[str, Image.Image]:
    """Decode one path, PIL image, bytes object, data URI, or {name, data} item."""
    name = "capture"
    source = item
    if isinstance(item, Mapping):
        name = str(item.get("name") or name)
        source = item.get("data", item.get("path", item.get("image")))
    if source is None:
        raise ValueError(f"No image data supplied for {name!r}.")

    if isinstance(source, Image.Image):
        img = source.copy()
    elif isinstance(source, (bytes, bytearray, memoryview)):
        img = Image.open(io.BytesIO(bytes(source)))
    elif isinstance(source, (str, Path)):
        value = str(source)
        if value.startswith("data:image"):
            try:
                encoded = value.split(",", 1)[1]
                raw = base64.b64decode(encoded, validate=True)
            except Exception as exc:
                raise ValueError(f"Could not decode image data for {name!r}: {exc}") from exc
            img = Image.open(io.BytesIO(raw))
        else:
            path = Path(value).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Image file not found: {path}")
            img = Image.open(path)
            if name == "capture":
                name = path.name
    else:
        raise TypeError(f"Unsupported capture image type: {type(source).__name__}")

    img = ImageOps.exif_transpose(img).convert("RGB")
    if img.width < 64 or img.height < 64:
        raise ValueError(f"{name!r} is too small ({img.width}x{img.height}); use images at least 64x64 pixels.")
    return name, img


def _slug(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value).strip()).strip("_-")
    return (token or "capture")[:48]


def _image_quality(image: Image.Image) -> Tuple[float, float]:
    """Return a simple Laplacian-variance sharpness proxy and mean brightness."""
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    lap = ndi.laplace(gray)
    return float(np.var(lap)), float(np.mean(gray))


def prepare_capture_session(
    images: Sequence[Any],
    output_root: Union[str, Path],
    session_name: str = "mechanical_part",
    max_images: int = 300,
) -> CaptureSession:
    """Validate and save uploaded views into a Nerfstudio-ready raw image folder.

    Input items may be image paths, bytes, PIL images, data URIs, or mappings
    containing ``name`` and ``data``. Exact duplicate pixel images are skipped.
    At least three unique views are required for a meaningful multi-view job.
    """
    if not isinstance(images, Sequence) or isinstance(images, (str, bytes, bytearray)):
        raise TypeError("images must be a sequence of image inputs.")
    if len(images) > max_images:
        raise ValueError(f"Too many images ({len(images)}); the limit is {max_images} per capture.")
    if len(images) < 3:
        raise ValueError("Select at least 3 different views. For a more reliable scan, aim for 20–60 overlapping photos.")

    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    base_name = f"{_slug(session_name)}_{time.strftime('%Y%m%d_%H%M%S')}"
    session_dir = root / base_name
    suffix = 2
    while session_dir.exists():
        session_dir = root / f"{base_name}_{suffix}"
        suffix += 1
    images_dir = session_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=False)

    entries: List[Dict[str, Any]] = []
    warnings: List[str] = []
    hashes: set[str] = set()
    short_edges: List[int] = []
    sharpness_values: List[float] = []
    brightness_values: List[float] = []

    try:
        for item in images:
            original_name, image = _decode_image_item(item)
            pixel_hash = hashlib.sha256(image.tobytes()).hexdigest()
            if pixel_hash in hashes:
                warnings.append(f"Skipped duplicate view: {original_name}")
                continue
            hashes.add(pixel_hash)

            sharpness, brightness = _image_quality(image)
            safe_stem = _slug(Path(original_name).stem)
            filename = f"view_{len(entries) + 1:04d}_{safe_stem}.jpg"
            target = images_dir / filename
            image.save(target, format="JPEG", quality=92, optimize=True)

            short_edge = min(image.size)
            short_edges.append(short_edge)
            sharpness_values.append(sharpness)
            brightness_values.append(brightness)
            entries.append({
                "file": f"images/{filename}",
                "source_name": original_name,
                "width": int(image.width),
                "height": int(image.height),
                "sha256_pixels": pixel_hash,
                "sharpness_laplacian_variance": round(sharpness, 3),
                "mean_brightness_0_255": round(brightness, 2),
            })

        if len(entries) < 3:
            raise ValueError("Fewer than 3 unique images remain after duplicate removal. Add more distinct viewpoints.")

        if len(entries) < 8:
            warnings.append("Small view set: 20–60 viewpoints with strong overlap usually give more stable camera alignment.")
        if short_edges and min(short_edges) < 640:
            warnings.append("Some images are low resolution; use the original camera files when possible.")
        blurry = sum(value < 20.0 for value in sharpness_values)
        if blurry:
            warnings.append(f"{blurry} image(s) have low sharpness; retake them with the camera held steady.")
        dark = sum(value < 30.0 for value in brightness_values)
        bright = sum(value > 225.0 for value in brightness_values)
        if dark or bright:
            warnings.append(f"Lighting check: {dark} very dark and {bright} over-bright image(s); keep exposure consistent.")

        manifest = {
            "format": "mos-multiview-capture-v1",
            "created_unix": time.time(),
            "session_name": _slug(session_name),
            "image_count": len(entries),
            "images": entries,
            "capture_guidance": {
                "recommended_view_count": "20-60",
                "overlap": "Keep adjacent views visually overlapping; move around the object at multiple heights.",
                "lighting": "Use diffuse, stable lighting and avoid motion blur or reflective glare.",
            },
        }
        manifest_path = session_dir / "capture_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception:
        shutil.rmtree(session_dir, ignore_errors=True)
        raise

    width_values = [entry["width"] for entry in entries]
    height_values = [entry["height"] for entry in entries]
    summary = (
        f"{len(entries)} unique views saved · "
        f"{min(width_values)}–{max(width_values)} px wide · "
        f"{min(height_values)}–{max(height_values)} px high · "
        f"average sharpness {np.mean(sharpness_values):.1f}"
    )
    return CaptureSession(session_dir, images_dir, manifest_path, len(entries), summary, warnings)


def nerfstudio_commands(
    images_dir: Union[str, Path],
    output_dir: Union[str, Path],
    method: str = "splatfacto",
) -> str:
    """Build copyable processing/training commands for supported recon methods."""
    images = Path(images_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    dataset = images.parent
    method = str(method or "splatfacto").lower()

    if os.name == "nt":
        q = lambda value: subprocess.list2cmdline([str(value)])
    else:
        q = lambda value: shlex.quote(str(value))

    if method == "2dgs":
        model = dataset / "outputs" / "2dgs"
        return (
            "# First click 'Build Standard COLMAP Dataset' in this panel.\n"
            "# Then run from a separately installed 2D Gaussian Splatting repository:\n"
            f"python train.py -s {q(dataset)} -m {q(model)}\n"
            "# Optional geometry extraction after training (TSDF mesh, not STEP):\n"
            f"python render.py -m {q(model)} -s {q(dataset)}"
        )
    if method == "graphdeco":
        model = dataset / "outputs" / "graphdeco_3dgs"
        return (
            "# First click 'Build Standard COLMAP Dataset' in this panel.\n"
            "# Then run from the Graphdeco gaussian-splatting repository:\n"
            f"python train.py -s {q(dataset)} -m {q(model)}\n"
            "# The trained Gaussian PLY is written under the model's point_cloud/iteration_* folder."
        )

    process = f"ns-process-data images --data {q(images)} --output-dir {q(output)}"
    if method == "nerfacto":
        train = f"ns-train nerfacto --data {q(output)}"
        note = "NeRF training selected (Nerfstudio nerfacto)."
        export = "\n\n# NeRF models are neural radiance fields, not Gaussian PLY files."
    else:
        train = f"ns-train splatfacto --data {q(output)}"
        note = "3D Gaussian Splatting training selected (Nerfstudio Splatfacto)."
        export = (
            "\n\n# After training, export the trained 3D Gaussian model (replace the config path):\n"
            "ns-export gaussian-splat --load-config PATH_TO_TRAINED_CONFIG.yml "
            f"--output-dir {q(output / 'exports' / 'gaussian-splat')}"
        )
    return f"{process}\n{train}{export}\n\n# {note}"


def tool_availability() -> Dict[str, bool]:
    """Report whether optional reconstruction CLIs are available on PATH."""
    return {
        "colmap": shutil.which("colmap") is not None or shutil.which("COLMAP.bat") is not None,
        "ns_process_data": shutil.which("ns-process-data") is not None,
        "ns_train": shutil.which("ns-train") is not None,
        "ffmpeg": shutil.which("ffmpeg") is not None,
    }


def run_nerfstudio_preprocess(
    images_dir: Union[str, Path],
    output_dir: Union[str, Path],
    timeout: int = 1800,
) -> Dict[str, Any]:
    """Run Nerfstudio's image parser, which estimates camera poses using COLMAP."""
    executable = shutil.which("ns-process-data")
    if not executable:
        raise RuntimeError(
            "`ns-process-data` is not installed. Install Nerfstudio in its own environment, "
            "then ensure COLMAP and FFmpeg are available on PATH. The exact command is shown in the UI."
        )
    images = Path(images_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not images.is_dir() or not any(p.is_file() for p in images.iterdir()):
        raise FileNotFoundError(f"No capture images found in {images}")
    output.mkdir(parents=True, exist_ok=True)
    command = [executable, "images", "--data", str(images), "--output-dir", str(output)]
    try:
        completed = subprocess.run(
            command, cwd=str(images.parent), capture_output=True, text=True,
            timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Camera-pose processing exceeded the {timeout}s time limit.") from exc
    log = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
    if completed.returncode != 0:
        detail = log[-5000:] if log else "No console output was returned."
        raise RuntimeError(f"Nerfstudio preprocessing failed (exit {completed.returncode}).\n{detail}")
    transforms = output / "transforms.json"
    if not transforms.is_file():
        detail = log[-3000:] if log else "No console output was returned."
        raise RuntimeError(f"Preprocessing finished but transforms.json was not created.\n{detail}")
    return {
        "output_dir": output,
        "transforms_path": transforms,
        "log_tail": log[-3000:],
    }


def _read_ply_vertices(path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[Tuple[str, str]]]:
    """Read scalar vertex properties from ASCII or binary-little-endian PLY."""
    with path.open("rb") as stream:
        header_lines: List[str] = []
        while True:
            line = stream.readline()
            if not line:
                raise ValueError("PLY header is incomplete (missing end_header).")
            decoded = line.decode("ascii", errors="strict").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break
        if not header_lines or header_lines[0] != "ply":
            raise ValueError("File does not start with a PLY header.")

        fmt = next((line.split()[1] for line in header_lines if line.startswith("format ")), None)
        if fmt not in {"ascii", "binary_little_endian"}:
            raise ValueError(f"Unsupported PLY format {fmt!r}; expected ASCII or binary little-endian.")

        vertex_count: Optional[int] = None
        in_vertex = False
        properties: List[Tuple[str, str]] = []
        for line in header_lines:
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif in_vertex and len(parts) == 3 and parts[0] == "property":
                if parts[1] not in _PLY_SCALARS:
                    raise ValueError(f"Unsupported PLY property type {parts[1]!r}.")
                properties.append((parts[2], parts[1]))
            elif in_vertex and line.startswith("property list "):
                raise ValueError("List-valued vertex properties are not supported for Gaussian PLY files.")
        if vertex_count is None or vertex_count <= 0:
            raise ValueError("PLY file has no vertex records.")
        if vertex_count > 50_000_000:
            raise ValueError("PLY contains too many vertices for this in-app conversion (limit: 50 million).")
        names = [name for name, _ in properties]
        if not {"x", "y", "z"}.issubset(names):
            raise ValueError("PLY vertex records must contain x, y, and z coordinates.")

        if fmt == "ascii":
            raw = np.loadtxt(stream, dtype=np.float64, max_rows=vertex_count, usecols=range(len(properties)))
            if raw.ndim == 1:
                raw = raw.reshape(1, -1)
            if raw.shape != (vertex_count, len(properties)):
                raise ValueError("PLY vertex table is truncated or malformed.")
            arrays = {name: raw[:, index] for index, (name, _) in enumerate(properties)}
        else:
            dtype = np.dtype([(name, "<" + _PLY_SCALARS[kind]) for name, kind in properties])
            expected = vertex_count * dtype.itemsize
            payload = stream.read(expected)
            if len(payload) != expected:
                raise ValueError("PLY binary vertex table is truncated.")
            records = np.frombuffer(payload, dtype=dtype, count=vertex_count)
            arrays = {name: records[name].astype(np.float64, copy=False) for name, _ in properties}

    return np.column_stack([arrays[axis] for axis in ("x", "y", "z")]), arrays, properties


def _write_rgb_point_cloud(path: Path, centers: np.ndarray, rgb: np.ndarray) -> Path:
    """Write a compact binary PLY with xyz and RGB vertex attributes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = len(centers)
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    point_records = np.empty(count, dtype=dtype)
    for index, axis in enumerate(("x", "y", "z")):
        point_records[axis] = centers[:, index]
    for index, axis in enumerate(("red", "green", "blue")):
        point_records[axis] = rgb[:, index]
    with path.open("wb") as stream:
        stream.write(
            ("ply\nformat binary_little_endian 1.0\n"
             f"element vertex {count}\n"
             "property float x\nproperty float y\nproperty float z\n"
             "property uchar red\nproperty uchar green\nproperty uchar blue\n"
             "end_header\n").encode("ascii")
        )
        point_records.tofile(stream)
    return path


def gaussian_ply_to_point_cloud(
    gaussian_ply_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
) -> GaussianPlySummary:
    """Extract Gaussian means from a 3DGS PLY into a conventional RGB point cloud.

    Common 3DGS fields ``f_dc_0..2`` are converted from SH DC color to RGB.
    Standard ``red/green/blue`` fields are preserved. The export intentionally
    discards covariance/orientation and view-dependent SH terms: it is a point
    cloud inspection bridge, not a surface or parametric CAD reconstruction.
    """
    source = Path(gaussian_ply_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Gaussian PLY file not found: {source}")
    centers, arrays, properties = _read_ply_vertices(source)
    if not np.isfinite(centers).all():
        raise ValueError("PLY contains non-finite Gaussian center coordinates.")

    count = len(centers)
    rgb = np.full((count, 3), 185, dtype=np.uint8)
    if all(channel in arrays for channel in ("red", "green", "blue")):
        values = np.column_stack([arrays[c] for c in ("red", "green", "blue")])
        if np.nanmax(values) <= 1.0:
            values = values * 255.0
        rgb = np.clip(values, 0, 255).astype(np.uint8)
    elif all(f"f_dc_{i}" in arrays for i in range(3)):
        sh_dc = np.column_stack([arrays[f"f_dc_{i}"] for i in range(3)])
        rgb = np.clip((0.5 + _SH_C0 * sh_dc) * 255.0, 0, 255).astype(np.uint8)

    out = Path(output_path).expanduser().resolve() if output_path else source.with_name(source.stem + "_centers.ply")
    _write_rgb_point_cloud(out, centers, rgb)

    mins = centers.min(axis=0)
    maxs = centers.max(axis=0)
    opacity = None
    if "opacity" in arrays:
        # Standard 3DGS stores opacity logits; apply sigmoid for a readable mean.
        logits = np.clip(arrays["opacity"], -40.0, 40.0)
        opacity = float(np.mean(1.0 / (1.0 + np.exp(-logits))))
    scale_fields = [f"scale_{i}" for i in range(3) if f"scale_{i}" in arrays]
    mean_scale = None
    if scale_fields:
        log_scales = np.column_stack([arrays[key] for key in scale_fields])
        mean_scale = float(np.mean(np.exp(np.clip(log_scales, -30.0, 30.0))))
    gaussian_fields = any(
        name == "opacity" or name.startswith("scale_") or name.startswith("rot_") or name.startswith("f_dc_")
        for name, _ in properties
    )
    return GaussianPlySummary(
        source_path=source,
        centers_path=out,
        gaussian_count=count,
        bounds=tuple((float(mins[i]), float(maxs[i])) for i in range(3)),
        centroid=tuple(float(value) for value in centers.mean(axis=0)),
        mean_opacity=opacity,
        mean_scale=mean_scale,
        has_gaussian_attributes=gaussian_fields,
    )


def calibrate_point_cloud_scale(
    point_cloud_path: Union[str, Path],
    known_extent_m: float,
    axis: str = "x",
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Apply a single metric scale from one known bounding-box dimension.

    Coordinates are centered at the point-cloud centroid and uniformly scaled,
    preserving the recovered shape proportions. This only fixes global scale;
    it does not repair incomplete geometry or turn points into a CAD solid.
    """
    source = Path(point_cloud_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Point cloud not found: {source}")
    target = float(known_extent_m)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("Known physical dimension must be a finite positive number of metres.")
    axis_index = {"x": 0, "y": 1, "z": 2}.get(str(axis).lower())
    if axis_index is None:
        raise ValueError("Scale axis must be x, y, or z.")

    centers, arrays, _ = _read_ply_vertices(source)
    if not np.isfinite(centers).all():
        raise ValueError("Point cloud contains non-finite coordinates.")
    extent = float(np.ptp(centers[:, axis_index]))
    if extent <= 1e-12:
        raise ValueError(f"Point cloud has zero extent along {axis.upper()}; choose another axis.")
    factor = target / extent
    calibrated = (centers - centers.mean(axis=0, keepdims=True)) * factor

    rgb = np.full((len(centers), 3), 185, dtype=np.uint8)
    if all(channel in arrays for channel in ("red", "green", "blue")):
        values = np.column_stack([arrays[c] for c in ("red", "green", "blue")])
        if np.nanmax(values) <= 1.0:
            values = values * 255.0
        rgb = np.clip(values, 0, 255).astype(np.uint8)
    elif all(f"f_dc_{i}" in arrays for i in range(3)):
        sh_dc = np.column_stack([arrays[f"f_dc_{i}"] for i in range(3)])
        rgb = np.clip((0.5 + _SH_C0 * sh_dc) * 255.0, 0, 255).astype(np.uint8)

    out = (Path(output_path).expanduser().resolve() if output_path else
           source.with_name(source.stem + "_meters.ply"))
    _write_rgb_point_cloud(out, calibrated, rgb)
    bounds = tuple((float(calibrated[:, i].min()), float(calibrated[:, i].max())) for i in range(3))
    return {
        "source_path": source,
        "output_path": out,
        "scale_factor": float(factor),
        "axis": str(axis).lower(),
        "known_extent_m": target,
        "bounds_m": bounds,
        "centroid_m": tuple(float(value) for value in calibrated.mean(axis=0)),
        "point_count": int(len(calibrated)),
    }


def _colmap_command(executable: str, *arguments: str) -> List[str]:
    """Build a platform-safe COLMAP command, including Windows batch launchers."""
    command = [str(executable), *(str(value) for value in arguments)]
    if os.name == "nt" and str(executable).lower().endswith((".bat", ".cmd")):
        return ["cmd", "/c", *command]
    return command


def run_colmap_sparse_reconstruction(
    images_dir: Union[str, Path],
    workspace_dir: Optional[Union[str, Path]] = None,
    timeout_per_step: int = 1800,
) -> Dict[str, Any]:
    """Create a standard COLMAP sparse model and PLY point cloud.

    The workspace layout is compatible with the usual Graphdeco/2DGS COLMAP
    loaders: ``images/`` and ``sparse/0/{cameras,images,points3D}.bin``. Feature
    extraction and matching use the COLMAP defaults so it can select available
    acceleration; the operation is intended to run in a background UI thread.
    """
    executable = shutil.which("colmap") or shutil.which("COLMAP.bat")
    if not executable:
        raise RuntimeError("COLMAP is not installed or is not available on PATH.")
    images = Path(images_dir).expanduser().resolve()
    if not images.is_dir() or not any(path.is_file() for path in images.iterdir()):
        raise FileNotFoundError(f"No capture images found in {images}")
    workspace = Path(workspace_dir).expanduser().resolve() if workspace_dir else images.parent
    if workspace != images.parent:
        raise ValueError("For a standard COLMAP dataset, workspace_dir must be the parent folder of images_dir.")

    database = workspace / "database.db"
    sparse_root = workspace / "sparse"
    if database.exists():
        database.unlink()
    if sparse_root.exists():
        shutil.rmtree(sparse_root)
    sparse_root.mkdir(parents=True, exist_ok=True)

    commands = [
        _colmap_command(
            executable, "feature_extractor", "--database_path", str(database),
            "--image_path", str(images), "--ImageReader.single_camera", "1",
            "--ImageReader.camera_model", "SIMPLE_PINHOLE",
            "--SiftExtraction.use_gpu", "0",
        ),
        _colmap_command(
            executable, "exhaustive_matcher", "--database_path", str(database),
            "--FeatureMatching.use_gpu", "0",
        ),
        _colmap_command(
            executable, "mapper", "--database_path", str(database),
            "--image_path", str(images), "--output_path", str(sparse_root),
        ),
    ]
    logs: List[str] = []
    for stage, command in zip(("Feature extraction", "Image matching", "Sparse mapping"), commands):
        try:
            completed = subprocess.run(
                command, cwd=str(workspace), capture_output=True, text=True,
                timeout=timeout_per_step, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"COLMAP {stage.lower()} exceeded {timeout_per_step}s.") from exc
        log = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
        if log:
            logs.append(f"[{stage}]\n{log[-3500:]}")
        if completed.returncode != 0:
            detail = log[-5000:] if log else "No console output was returned."
            raise RuntimeError(f"COLMAP {stage.lower()} failed (exit {completed.returncode}).\n{detail}")

    models = [
        path for path in sparse_root.iterdir()
        if path.is_dir() and (path / "cameras.bin").is_file() and (path / "images.bin").is_file()
        and (path / "points3D.bin").is_file()
    ]
    if not models:
        detail = logs[-1][-3000:] if logs else "COLMAP did not create a sparse model. Check image overlap and texture."
        raise RuntimeError(f"COLMAP completed without a registered sparse model.\n{detail}")
    model = sparse_root / "0" if (sparse_root / "0").is_dir() else sorted(models, key=lambda path: path.name)[0]
    if model.name != "0":
        standard_model = sparse_root / "0"
        if standard_model.exists():
            shutil.rmtree(standard_model)
        shutil.copytree(model, standard_model)
        model = standard_model

    point_cloud = workspace / "sparse_points.ply"
    if point_cloud.exists():
        point_cloud.unlink()
    command = _colmap_command(
        executable, "model_converter", "--input_path", str(model),
        "--output_path", str(point_cloud), "--output_type", "PLY",
    )
    try:
        completed = subprocess.run(
            command, cwd=str(workspace), capture_output=True, text=True,
            timeout=timeout_per_step, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"COLMAP PLY export exceeded {timeout_per_step}s.") from exc
    log = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
    if log:
        logs.append(f"[PLY export]\n{log[-3500:]}")
    if completed.returncode != 0 or not point_cloud.is_file():
        detail = log[-5000:] if log else "COLMAP did not write sparse_points.ply."
        raise RuntimeError(f"COLMAP point-cloud export failed.\n{detail}")

    return {
        "workspace_dir": workspace,
        "model_dir": model,
        "point_cloud_path": point_cloud,
        "model_count": len(models),
        "log_tail": "\n\n".join(logs)[-6000:],
    }
