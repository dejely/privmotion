from __future__ import annotations

import pytest

from privmotion.config import ProcessConfig, parse_output_modes


def test_parse_output_modes_normalizes_and_deduplicates() -> None:
    assert parse_output_modes("skeleton, depth_surrogate, skeleton") == (
        "skeleton",
        "depth-surrogate",
    )


def test_parse_output_modes_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unsupported output mode"):
        parse_output_modes("skeleton,raw-rgb")


def test_process_config_rejects_non_default_retention(tmp_path) -> None:
    with pytest.raises(ValueError, match="no-raw-rgb"):
        ProcessConfig(
            input_path=tmp_path / "in.ppm",
            output_dir=tmp_path / "out",
            retention_policy="keep-raw-rgb",
        )


def test_process_config_rejects_unknown_pose_backend(tmp_path) -> None:
    with pytest.raises(ValueError, match="pose_backend"):
        ProcessConfig(
            input_path=tmp_path / "in.ppm",
            output_dir=tmp_path / "out",
            pose_backend="unknown",
        )


def test_process_config_records_pose_model(tmp_path) -> None:
    config = ProcessConfig(
        input_path=tmp_path / "in.ppm",
        output_dir=tmp_path / "out",
        pose_backend="yolo",
        pose_model="custom-pose.pt",
    )

    assert config.to_json()["pose_model"] == "custom-pose.pt"
