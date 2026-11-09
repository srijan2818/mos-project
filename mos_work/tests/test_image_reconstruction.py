from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw
import pytest

from mos.core.image_reconstruction import reconstruct_single_image
from mos.core.multiview import _read_ply_vertices


def _transparent_object() -> Image.Image:
    image = Image.new("RGBA", (180, 140), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((28, 18, 152, 122), radius=12, fill=(220, 95, 45, 255))
    draw.ellipse((72, 42, 108, 78), fill=(255, 255, 255, 255))
    return image


def test_single_image_silhouette_reconstruction_writes_colored_closed_surface(tmp_path: Path):
    result = reconstruct_single_image(
        _transparent_object(), tmp_path, geometry_mode="silhouette",
        export_step=False, max_resolution=96,
    )
    assert result.preview_mesh_path.is_file()
    assert result.mask_path.is_file()
    assert result.depth_preview_path is None
    assert result.step_path is None
    assert result.vertex_count > 500
    assert result.face_count > result.vertex_count

    points, arrays, properties = _read_ply_vertices(result.preview_mesh_path)
    assert points.shape == (result.vertex_count, 3)
    assert {"red", "green", "blue"}.issubset(arrays)
    assert np.isfinite(points).all()
    assert len(properties) == 6


def test_single_image_scale_and_output_dimensions_are_respected(tmp_path: Path):
    result = reconstruct_single_image(
        _transparent_object(), tmp_path, geometry_mode="silhouette",
        object_width_m=0.4, body_thickness_m=0.03, export_step=False,
        max_resolution=96,
    )
    assert result.width_m == pytest.approx(0.4)
    assert result.height_m == pytest.approx(0.4 * 140 / 180, rel=0.03)
    assert result.thickness_m == pytest.approx(0.03)


def test_reconstruction_rejects_invalid_scale_and_mode(tmp_path: Path):
    image = _transparent_object()
    with pytest.raises(ValueError, match="positive"):
        reconstruct_single_image(image, tmp_path, object_width_m=0, export_step=False)
    with pytest.raises(ValueError, match="geometry_mode"):
        reconstruct_single_image(image, tmp_path, geometry_mode="magic", export_step=False)


def test_reconstruction_supports_data_uri_upload(tmp_path: Path):
    import base64
    import io

    stream = io.BytesIO()
    _transparent_object().save(stream, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")
    result = reconstruct_single_image(
        uri, tmp_path, geometry_mode="silhouette", export_step=False,
        max_resolution=96,
    )
    assert result.source_name == "uploaded_image"
    assert result.preview_mesh_path.exists()


def test_reconstruction_cannot_use_ambiguous_tiny_or_empty_mask(tmp_path: Path):
    image = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
    with pytest.raises(ValueError, match="isolate|foreground mask"):
        reconstruct_single_image(
            image, tmp_path, geometry_mode="silhouette", export_step=False,
            max_resolution=96,
        )


def test_border_color_fallback_segments_object_on_plain_background(tmp_path: Path):
    image = Image.new("RGB", (180, 140), "white")
    ImageDraw.Draw(image).rounded_rectangle((30, 18, 150, 120), radius=10, fill=(30, 70, 160))
    result = reconstruct_single_image(
        image, tmp_path, geometry_mode="silhouette", export_step=False,
        max_resolution=96,
    )
    assert result.segmentation_method == "border-color foreground estimate"
    assert result.preview_mesh_path.is_file()


def test_full_frame_mode_is_an_explicit_rectangular_relief(tmp_path: Path):
    image = Image.new("RGB", (128, 128), (80, 100, 120))
    result = reconstruct_single_image(
        image, tmp_path, geometry_mode="silhouette", segmentation_mode="full_frame",
        export_step=False, max_resolution=96,
    )
    assert result.segmentation_method == "full image frame"
    assert result.face_count > 1000


def test_named_browser_upload_preserves_source_name_and_uses_unique_outputs(tmp_path: Path):
    image = _transparent_object()
    first = reconstruct_single_image(
        {"name": "custom_part.png", "data": image}, tmp_path,
        geometry_mode="silhouette", export_step=False, max_resolution=96,
    )
    second = reconstruct_single_image(
        {"name": "custom_part.png", "data": image}, tmp_path,
        geometry_mode="silhouette", export_step=False, max_resolution=96,
    )
    assert first.source_name == "custom_part.png"
    assert second.source_name == "custom_part.png"
    assert first.preview_mesh_path != second.preview_mesh_path
    assert first.preview_mesh_path.exists() and second.preview_mesh_path.exists()


def test_transparent_cutout_is_preserved_in_preview_and_mesh_is_closed(tmp_path: Path):
    image = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((12, 12, 148, 148), fill=(30, 150, 210, 255))
    draw.ellipse((50, 50, 110, 110), fill=(0, 0, 0, 0))

    result = reconstruct_single_image(
        image, tmp_path, geometry_mode="silhouette", export_step=False,
        max_resolution=96,
    )
    mask = np.asarray(Image.open(result.mask_path).convert("L"))
    assert mask[mask.shape[0] // 2, mask.shape[1] // 2] > 200
    assert mask[mask.shape[0] // 2, int(mask.shape[1] * 0.78)] < 50

    # Inspect the binary PLY faces and verify every undirected edge is shared
    # by exactly two faces, including the boundary around the inner cutout.
    payload = result.preview_mesh_path.read_bytes()
    marker = b"end_header\n"
    header_end = payload.index(marker) + len(marker)
    header = payload[:header_end].decode("ascii")
    vertex_count = int(next(line.split()[2] for line in header.splitlines() if line.startswith("element vertex ")))
    face_count = int(next(line.split()[2] for line in header.splitlines() if line.startswith("element face ")))
    offset = header_end + vertex_count * 15  # xyz float32 + RGB uint8
    edge_counts = Counter()
    for _ in range(face_count):
        vertex_total = payload[offset]
        offset += 1
        indices = np.frombuffer(payload, dtype="<i4", count=vertex_total, offset=offset).tolist()
        offset += vertex_total * 4
        for a, b in zip(indices, indices[1:] + indices[:1]):
            edge_counts[tuple(sorted((a, b)))] += 1
    assert edge_counts
    assert set(edge_counts.values()) == {2}
