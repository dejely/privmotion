from __future__ import annotations

from privmotion.exporters import write_json, write_pgm
from privmotion.validation import validate_output_dir

import numpy as np
import pytest


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


def write_hipaa_aggregate_output(output_dir) -> None:
    write_json(
        output_dir / "metadata.json",
        {
            "schema": "privmotion.metadata.v0",
            "deidentification_profile": "hipaa-expert-aggregate",
            "config": {
                "output_modes": ["aggregate"],
                "retention_policy": "no-raw-rgb",
                "deidentification_profile": "hipaa-expert-aggregate",
            },
            "input_summary": {
                "input_type": "video",
                "source_count": 3,
                "readable_frames": 3,
                "skipped_frames": 0,
            },
            "retention": {
                "policy": "no-raw-rgb",
                "raw_rgb_written": False,
                "person_level_artifacts_written": False,
            },
        },
    )
    write_json(
        output_dir / "aggregate_report.json",
        {
            "schema": "privmotion.aggregate.v0",
            "deidentification_profile": "hipaa-expert-aggregate",
            "utility": {"processed_frame_count": 3},
            "retention": {"raw_rgb_written": False, "person_level_artifacts_written": False},
        },
    )
    write_json(
        output_dir / "deidentification_report.json",
        {
            "schema": "privmotion.deidentification.v0",
            "deidentification_profile": "hipaa-expert-aggregate",
            "expert_determination_status": "required",
            "hipaa_deidentification_claim": False,
        },
    )
    write_json(
        output_dir / "retention_report.json",
        {
            "policy": "no-raw-rgb",
            "deidentification_profile": "hipaa-expert-aggregate",
            "passed": True,
            "violations": [],
        },
    )


def test_validate_output_dir_passes_for_hipaa_aggregate_output(tmp_path) -> None:
    output_dir = tmp_path / "out"
    write_hipaa_aggregate_output(output_dir)

    result = validate_output_dir(output_dir, deidentification_profile="hipaa-expert-aggregate")

    assert result.passed
    assert result.to_json(redacted=True)["deidentification_profile"] == "hipaa-expert-aggregate"
    assert "output_dir" not in result.to_json(redacted=True)


@pytest.mark.parametrize(
    ("path", "payload", "expected"),
    [
        ("skeletons.json", {"records": []}, "skeletons.json:person-level-artifact"),
        ("features.json", {"records": []}, "features.json:person-level-artifact"),
        ("preview.mp4", b"not a real mp4", "preview.mp4:person-level-artifact"),
        ("metadata-copy.json", {"input_path": "/tmp/patient.mp4"}, "metadata-copy.json:unsupported-file"),
        ("metadata.json", {"source_name": "patient.mp4"}, "metadata.json:source_name"),
        ("aggregate_report.json", {"track_id": 1}, "aggregate_report.json:track_id"),
        ("retention_report.json", {"timestamp_ms": 1.0}, "retention_report.json:timestamp_ms"),
    ],
)
def test_validate_output_dir_flags_hipaa_identifier_artifacts(tmp_path, path, payload, expected) -> None:
    output_dir = tmp_path / "out"
    write_hipaa_aggregate_output(output_dir)
    target = output_dir / path
    if isinstance(payload, bytes):
        target.write_bytes(payload)
    else:
        write_json(target, payload)

    result = validate_output_dir(output_dir, deidentification_profile="hipaa-expert-aggregate")

    assert not result.passed
    assert expected in result.violations


def test_validate_output_dir_flags_hipaa_person_level_dirs(tmp_path) -> None:
    output_dir = tmp_path / "out"
    write_hipaa_aggregate_output(output_dir)
    write_pgm(output_dir / "silhouettes" / "frame_000000.pgm", np.zeros((4, 4), dtype=np.uint8))

    result = validate_output_dir(output_dir, deidentification_profile="hipaa-expert-aggregate")

    assert not result.passed
    assert "silhouettes:person-level-directory" in result.violations
    assert "silhouettes/frame_000000.pgm:person-level-artifact" in result.violations
