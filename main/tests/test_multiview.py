"""Tests for multi-view capture and Gaussian PLY support."""
import base64
import io
import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from mos.core.multiview import (
    calibrate_point_cloud_scale,
    gaussian_ply_to_point_cloud,
    nerfstudio_commands,
    prepare_capture_session,
    run_colmap_sparse_reconstruction,
)


def _data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _distinct_image(index: int) -> Image.Image:
    image = Image.new("RGB", (360, 280), (170 + index * 10, 180, 190))
    draw = ImageDraw.Draw(image)
    draw.rectangle((40 + index * 3, 50, 250, 220), outline=(20, 30, 40), width=5)
    draw.ellipse((80, 80 + index * 2, 125, 125 + index * 2), fill=(200, 30, 20))
    return image


def test_prepare_capture_session_saves_unique_views_and_manifest(tmp_path: Path):
    images = [
        {"name": f"view_{index}.png", "data": _data_uri(_distinct_image(index))}
        for index in range(4)
    ]
    images.append(images[0])  # exact duplicate pixels should be skipped

    session = prepare_capture_session(images, tmp_path, session_name="test part")

    assert session.image_count == 4
    assert session.images_dir.is_dir()
    assert len(list(session.images_dir.glob("*.jpg"))) == 4
    assert session.manifest_path.is_file()
    assert any("duplicate" in warning.lower() for warning in session.warnings)
    assert "4 unique views" in session.summary


def test_prepare_capture_session_requires_three_distinct_views(tmp_path: Path):
    duplicate = _data_uri(_distinct_image(0))
    with pytest.raises(ValueError, match="Fewer than 3 unique images"):
        prepare_capture_session([duplicate, duplicate, duplicate], tmp_path)


def test_nerfstudio_command_variants(tmp_path: Path):
    images = tmp_path / "images folder"
    output = tmp_path / "processed folder"
    splat = nerfstudio_commands(images, output, "splatfacto")
    nerf = nerfstudio_commands(images, output, "nerfacto")
    two_d = nerfstudio_commands(images, output, "2dgs")
    graphdeco = nerfstudio_commands(images, output, "graphdeco")

    assert "ns-process-data images" in splat
    assert "ns-train splatfacto" in splat
    assert "ns-export gaussian-splat" in splat
    assert "ns-train nerfacto" in nerf
    assert "Build Standard COLMAP Dataset" in two_d
    assert "python train.py -s" in two_d
    assert "python render.py -m" in two_d
    assert "ns-train 2dgs" not in two_d
    assert "python train.py -s" in graphdeco
    assert "Graphdeco" in graphdeco
    assert "point_cloud/iteration_*" in graphdeco


def test_gaussian_ply_ascii_exports_centers_and_sh_colors(tmp_path: Path):
    source = tmp_path / "splats_ascii.ply"
    source.write_text(
        "ply\nformat ascii 1.0\n"
        "element vertex 2\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
        "property float opacity\nproperty float scale_0\nproperty float scale_1\nproperty float scale_2\n"
        "end_header\n"
        "0 1 2 0 0 0 0 0 0 0\n"
        "2 3 4 1 0 -1 1 0.693147 0.693147 0.693147\n",
        encoding="ascii",
    )
    output = tmp_path / "centers.ply"

    summary = gaussian_ply_to_point_cloud(source, output)

    assert summary.gaussian_count == 2
    assert summary.has_gaussian_attributes
    assert summary.mean_opacity == pytest.approx((0.5 + 1 / (1 + np.exp(-1))) / 2)
    assert summary.mean_scale == pytest.approx((1.0 + 2.0) / 2)
    assert summary.bounds == ((0.0, 2.0), (1.0, 3.0), (2.0, 4.0))
    assert output.is_file()
    raw = output.read_bytes()
    assert b"format binary_little_endian 1.0" in raw[:100]
    assert b"element vertex 2" in raw[:100]


def test_gaussian_ply_binary_little_endian(tmp_path: Path):
    source = tmp_path / "splats_binary.ply"
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "element vertex 2\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float opacity\n"
        "end_header\n"
    ).encode("ascii")
    records = np.array([(1.0, 2.0, 3.0, 0.0), (-1.0, 0.5, 4.0, 1.0)],
                       dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("opacity", "<f4")])
    source.write_bytes(header + records.tobytes())

    summary = gaussian_ply_to_point_cloud(source)

    assert summary.gaussian_count == 2
    assert summary.centers_path.exists()
    assert summary.centroid == pytest.approx((0.0, 1.25, 3.5))
    assert summary.mean_opacity == pytest.approx((0.5 + 1 / (1 + np.exp(-1))) / 2)


def test_gaussian_ply_rejects_missing_coordinates(tmp_path: Path):
    source = tmp_path / "bad.ply"
    source.write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\n"
        "property float x\nproperty float y\nend_header\n0 1\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="x, y, and z"):
        gaussian_ply_to_point_cloud(source)


def test_point_cloud_scale_calibration_preserves_proportions(tmp_path: Path):
    source = tmp_path / "centers.ply"
    header = (
        "ply\nformat ascii 1.0\nelement vertex 3\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    source.write_text(header + "0 0 0 255 0 0\n2 1 0 0 255 0\n1 0 1 0 0 255\n", encoding="ascii")

    result = calibrate_point_cloud_scale(source, 0.4, axis="x")

    assert result["scale_factor"] == pytest.approx(0.2)
    assert result["bounds_m"][0] == pytest.approx((-0.2, 0.2))
    assert result["point_count"] == 3
    assert result["output_path"].is_file()


def test_point_cloud_scale_calibration_rejects_zero_extent(tmp_path: Path):
    source = tmp_path / "flat.ply"
    source.write_text(
        "ply\nformat ascii 1.0\nelement vertex 2\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n0 0 0\n0 1 1\n", encoding="ascii"
    )
    with pytest.raises(ValueError, match="zero extent"):
        calibrate_point_cloud_scale(source, 0.2, axis="x")


def test_colmap_runner_builds_standard_sparse_dataset(tmp_path: Path, monkeypatch):
    images = tmp_path / "capture" / "images"
    images.mkdir(parents=True)
    (images / "a.jpg").write_bytes(b"image-a")
    (images / "b.jpg").write_bytes(b"image-b")
    calls = []

    monkeypatch.setattr("mos.core.multiview.shutil.which", lambda name: "/fake/colmap" if name == "colmap" else None)

    class Completed:
        returncode = 0
        stdout = "stage ok"
        stderr = ""

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[1] == "mapper":
            output = Path(command[command.index("--output_path") + 1]) / "0"
            output.mkdir(parents=True)
            for filename in ("cameras.bin", "images.bin", "points3D.bin"):
                (output / filename).write_bytes(b"colmap-model")
        elif command[1] == "model_converter":
            output = Path(command[command.index("--output_path") + 1])
            output.write_text("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 0\n")
        return Completed()

    monkeypatch.setattr("mos.core.multiview.subprocess.run", fake_run)
    result = run_colmap_sparse_reconstruction(images)

    assert (result["model_dir"] / "cameras.bin").is_file()
    assert result["point_cloud_path"].is_file()
    assert len(calls) == 4
    assert calls[0][1] == "feature_extractor"
    assert "--ImageReader.single_camera" in calls[0]
    assert "--ImageReader.camera_model" in calls[0]
    assert calls[0][calls[0].index("--ImageReader.camera_model") + 1] == "SIMPLE_PINHOLE"
    assert "--SiftExtraction.use_gpu" in calls[0]
    assert calls[1][1] == "exhaustive_matcher"
    assert "--FeatureMatching.use_gpu" in calls[1]
    assert calls[2][1] == "mapper"
    assert calls[3][1] == "model_converter"


def test_colmap_runner_reports_missing_executable(tmp_path: Path, monkeypatch):
    images = tmp_path / "images"
    images.mkdir()
    (images / "one.jpg").write_bytes(b"x")
    monkeypatch.setattr("mos.core.multiview.shutil.which", lambda _name: None)
    with pytest.raises(RuntimeError, match="COLMAP is not installed"):
        run_colmap_sparse_reconstruction(images)


def test_tool_availability_detects_windows_colmap_batch(monkeypatch):
    from mos.core.multiview import tool_availability

    monkeypatch.setattr(
        "mos.core.multiview.shutil.which",
        lambda name: "C:/tools/COLMAP.bat" if name == "COLMAP.bat" else None,
    )

    assert tool_availability()["colmap"] is True
