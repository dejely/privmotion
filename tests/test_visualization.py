from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from privmotion.cli.visualize import main as visualize_main, parse_size
from privmotion.config import ProcessConfig
from privmotion.pipeline import PrivMotionPipeline
from privmotion.visualization import visualize_output_dir


def write_ppm(path: Path, image: np.ndarray) -> None:
    height, width, channels = image.shape
    assert channels == 3
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + image.astype(np.uint8).tobytes())


def synthetic_person_image() -> np.ndarray:
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[8:34, 14:26] = (230, 230, 230)
    image[4:12, 16:24] = (250, 250, 250)
    return image


def build_processed_output(tmp_path, modes=("skeleton", "silhouette", "depth-surrogate", "features")) -> Path:
    input_path = tmp_path / "person.ppm"
    output_dir = tmp_path / "out"
    write_ppm(input_path, synthetic_person_image())
    PrivMotionPipeline(
        ProcessConfig(input_path=input_path, output_dir=output_dir, output_modes=modes, pose_backend="prototype")
    ).run()
    return output_dir


def test_visualize_output_dir_writes_gif_and_png_frames(tmp_path) -> None:
    output_dir = build_processed_output(tmp_path)
    gif_path = tmp_path / "preview.gif"
    frames_dir = tmp_path / "frames"

    result = visualize_output_dir(output_dir, gif_path, frames_dir=frames_dir, size=(320, 180))

    assert result.frame_count == 1
    assert result.raw_rgb_used is False
    assert gif_path.exists()
    assert (frames_dir / "frame_000000.png").exists()
    with Image.open(gif_path) as image:
        assert image.n_frames >= 1
        assert image.size == (320, 180)


def test_visualize_cli_writes_gif(tmp_path) -> None:
    output_dir = build_processed_output(tmp_path)
    gif_path = tmp_path / "preview.gif"

    exit_code = visualize_main(
        [
            "--output",
            str(output_dir),
            "--visualization",
            str(gif_path),
            "--size",
            "320x180",
        ]
    )

    assert exit_code == 0
    assert gif_path.exists()


def test_visualize_writes_mp4(tmp_path) -> None:
    output_dir = build_processed_output(tmp_path)
    mp4_path = tmp_path / "preview.mp4"

    result = visualize_output_dir(output_dir, mp4_path, size=(320, 180), fps=4)

    assert result.frame_count == 1
    assert mp4_path.exists()
    import imageio.v3 as iio

    frame = next(iter(iio.imiter(mp4_path)))
    assert frame.shape[0] == 180
    assert frame.shape[1] == 320


def test_visualize_handles_missing_optional_artifacts(tmp_path) -> None:
    output_dir = build_processed_output(tmp_path, modes=("skeleton",))
    gif_path = tmp_path / "preview.gif"

    result = visualize_output_dir(output_dir, gif_path, size=(320, 180))

    assert result.frame_count == 1
    assert result.raw_rgb_used is False
    assert gif_path.exists()


def test_visualize_fails_for_hipaa_aggregate_output(tmp_path) -> None:
    input_path = tmp_path / "person.ppm"
    output_dir = tmp_path / "out"
    write_ppm(input_path, synthetic_person_image())
    PrivMotionPipeline(
        ProcessConfig(
            input_path=input_path,
            output_dir=output_dir,
            output_modes=("aggregate",),
            pose_backend="prototype",
            deidentification_profile="hipaa-expert-aggregate",
        )
    ).run()

    with pytest.raises(ValueError, match="hipaa-expert-aggregate"):
        visualize_output_dir(output_dir, tmp_path / "preview.gif")


def test_visualize_fails_on_missing_output_dir(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        visualize_output_dir(tmp_path / "missing", tmp_path / "preview.gif")


def test_visualize_fails_on_missing_metadata(tmp_path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="metadata.json"):
        visualize_output_dir(output_dir, tmp_path / "preview.gif")


def test_visualize_does_not_mutate_source_metadata_raw_rgb_flag(tmp_path) -> None:
    output_dir = build_processed_output(tmp_path)
    gif_path = tmp_path / "preview.gif"

    result = visualize_output_dir(output_dir, gif_path)
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))

    assert result.raw_rgb_used is False
    assert metadata["retention"]["raw_rgb_written"] is False


def test_parse_size() -> None:
    assert parse_size("640x360") == (640, 360)
    with pytest.raises(ValueError):
        parse_size("640")
