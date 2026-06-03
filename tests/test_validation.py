from __future__ import annotations

from privmotion.exporters import write_json, write_pgm
from privmotion.validation import validate_output_dir

import numpy as np


def test_validate_output_dir_passes_for_allowed_outputs(tmp_path) -> None:
    output_dir = tmp_path / "out"
    write_json(output_dir / "metadata.json", {"retention": {"raw_rgb_written": False}})
    write_pgm(output_dir / "silhouettes" / "frame_000000.pgm", np.zeros((4, 4), dtype=np.uint8))
    write_pgm(output_dir / "depth_surrogates" / "frame_000000.pgm", np.zeros((4, 4), dtype=np.uint8))

    result = validate_output_dir(output_dir)

    assert result.passed
    assert result.violations == ()


def test_validate_output_dir_fails_for_raw_rgb_like_file(tmp_path) -> None:
    output_dir = tmp_path / "out"
    write_json(output_dir / "metadata.json", {})
    (output_dir / "raw_rgb").mkdir()
    (output_dir / "raw_rgb" / "frame_000000.ppm").write_bytes(b"P6\n1 1\n255\n\x00\x00\x00")

    result = validate_output_dir(output_dir)

    assert not result.passed
    assert result.violations == ("raw_rgb/frame_000000.ppm",)

