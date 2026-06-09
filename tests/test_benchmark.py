from __future__ import annotations

import json

import numpy as np
import pytest

from privmotion.benchmark import benchmark_output_dir
from privmotion.cli.benchmark import main as benchmark_main
from privmotion.config import ProcessConfig
from privmotion.exporters import write_json
from privmotion.pipeline import PrivMotionPipeline


def write_ppm(path, image: np.ndarray) -> None:
    height, width, channels = image.shape
    assert channels == 3
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + image.astype(np.uint8).tobytes())


def synthetic_person_image() -> np.ndarray:
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[8:34, 14:26] = (230, 230, 230)
    image[4:12, 16:24] = (250, 250, 250)
    return image


def build_processed_output(tmp_path, modes=("skeleton", "silhouette", "depth-surrogate", "features")):
    input_path = tmp_path / "person.ppm"
    output_dir = tmp_path / "out"
    write_ppm(input_path, synthetic_person_image())
    PrivMotionPipeline(
        ProcessConfig(input_path=input_path, output_dir=output_dir, output_modes=modes, pose_backend="prototype")
    ).run()
    return output_dir


def test_benchmark_processed_output_dir(tmp_path) -> None:
    output_dir = build_processed_output(tmp_path)

    report = benchmark_output_dir(output_dir)

    assert report.utility["processed_frame_count"] == 1
    assert report.utility["skeleton_record_count"] == 1
    assert report.utility["keypoint_frame_coverage"] == 1.0
    assert report.utility["average_keypoint_confidence"] is not None
    assert report.utility["silhouette_count"] == 1
    assert report.utility["depth_surrogate_count"] == 1
    assert report.utility["feature_record_count"] == 1
    assert report.privacy["raw_rgb_retention_passed"] is True
    assert report.privacy["mask_only_visual_outputs"] is True


def test_benchmark_cli_writes_report(tmp_path) -> None:
    output_dir = build_processed_output(tmp_path)
    report_path = tmp_path / "benchmark_report.json"

    exit_code = benchmark_main(["--output", str(output_dir), "--report", str(report_path)])

    assert exit_code == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "privmotion.benchmark.v0"
    assert payload["utility"]["processed_frame_count"] == 1


def test_benchmark_missing_output_dir_fails(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        benchmark_output_dir(tmp_path / "missing")


def test_benchmark_missing_metadata_fails(tmp_path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="metadata.json"):
        benchmark_output_dir(output_dir)


def test_benchmark_handles_missing_optional_outputs(tmp_path) -> None:
    output_dir = build_processed_output(tmp_path, modes=("skeleton", "silhouette"))
    (output_dir / "depth_surrogates").mkdir(exist_ok=True)

    report = benchmark_output_dir(output_dir)

    assert report.utility["feature_record_count"] is None
    assert report.utility["depth_surrogate_count"] == 0
    assert report.utility["depth_surrogate_frame_coverage"] == 0.0
    assert report.privacy["surrogate_resolution_reduction"] is None
    assert report.systems["available_artifacts"]["features"] is False


def test_benchmark_rejects_hipaa_aggregate_output(tmp_path) -> None:
    input_path = tmp_path / "person.ppm"
    output_dir = tmp_path / "hipaa"
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

    with pytest.raises(ValueError, match="aggregate_report"):
        benchmark_output_dir(output_dir)


def test_benchmark_privacy_metric_detects_non_mask_visual_output(tmp_path) -> None:
    output_dir = build_processed_output(tmp_path)
    write_json(output_dir / "metadata-copy.json", {})
    (output_dir / "raw_preview.ppm").write_bytes(
        f"P6\n2 2\n255\n".encode("ascii") + np.zeros((2, 2, 3), dtype=np.uint8).tobytes()
    )

    report = benchmark_output_dir(output_dir)

    assert report.privacy["mask_only_visual_outputs"] is False
