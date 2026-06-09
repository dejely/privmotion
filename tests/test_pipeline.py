from __future__ import annotations

import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from privmotion.config import ProcessConfig
from privmotion.pipeline import PrivMotionPipeline
from privmotion.backends import Keypoint, PoseResult
from privmotion.validation import validate_output_dir


def write_ppm(path: Path, image: np.ndarray) -> None:
    height, width, channels = image.shape
    assert channels == 3
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + image.astype(np.uint8).tobytes())


def synthetic_person_image() -> np.ndarray:
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[8:34, 14:26] = (230, 230, 230)
    image[4:12, 16:24] = (250, 250, 250)
    return image


def test_pipeline_processes_synthetic_image_without_raw_rgb(tmp_path) -> None:
    input_path = tmp_path / "person.ppm"
    output_dir = tmp_path / "out"
    write_ppm(input_path, synthetic_person_image())

    result = PrivMotionPipeline(
        ProcessConfig(
            input_path=input_path,
            output_dir=output_dir,
            output_modes=("skeleton", "silhouette", "features"),
            pose_backend="prototype",
        )
    ).run()

    assert result.processed_frames == 1
    assert (output_dir / "metadata.json").exists()
    assert (output_dir / "skeletons.json").exists()
    assert (output_dir / "features.json").exists()
    assert (output_dir / "silhouettes" / "frame_000000.pgm").exists()
    assert not any(path.name.startswith("raw") for path in output_dir.rglob("*"))

    validation = validate_output_dir(output_dir)
    assert validation.passed


def test_pipeline_hipaa_aggregate_writes_only_redacted_reports(tmp_path) -> None:
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

    assert sorted(path.name for path in output_dir.iterdir()) == [
        "aggregate_report.json",
        "deidentification_report.json",
        "metadata.json",
        "retention_report.json",
    ]
    assert not (output_dir / "skeletons.json").exists()
    assert not (output_dir / "features.json").exists()
    assert not (output_dir / "silhouettes").exists()
    assert not (output_dir / "depth_surrogates").exists()

    metadata_text = (output_dir / "metadata.json").read_text(encoding="utf-8")
    aggregate_text = (output_dir / "aggregate_report.json").read_text(encoding="utf-8")
    deidentification = json.loads((output_dir / "deidentification_report.json").read_text(encoding="utf-8"))
    metadata = json.loads(metadata_text)
    aggregate = json.loads(aggregate_text)

    forbidden = (
        "input_path",
        "output_dir",
        "access_policy_path",
        "source_name",
        "track_id",
        "timestamp_ms",
        "bbox_xywh",
        "centroid_xy",
        "keypoints",
        "keypoint_velocity",
        "created_unix",
    )
    for token in forbidden:
        assert token not in metadata_text
        assert token not in aggregate_text
    assert metadata["deidentification_profile"] == "hipaa-expert-aggregate"
    assert metadata["retention"]["person_level_artifacts_written"] is False
    assert aggregate["utility"]["processed_frame_count"] == 1
    assert aggregate["retention"]["person_level_artifacts_written"] is False
    assert deidentification["expert_determination_status"] == "required"
    assert deidentification["hipaa_deidentification_claim"] is False

    validation = validate_output_dir(output_dir, deidentification_profile="hipaa-expert-aggregate")
    assert validation.passed


def test_skeleton_json_contains_required_fields(tmp_path) -> None:
    input_path = tmp_path / "person.ppm"
    output_dir = tmp_path / "out"
    write_ppm(input_path, synthetic_person_image())

    PrivMotionPipeline(
        ProcessConfig(input_path=input_path, output_dir=output_dir, output_modes=("skeleton",), pose_backend="prototype")
    ).run()

    payload = json.loads((output_dir / "skeletons.json").read_text(encoding="utf-8"))
    assert payload["schema"] == "privmotion.skeletons.v0"
    record = payload["records"][0]
    assert record["frame_index"] == 0
    assert record["track_id"] == 1
    assert record["keypoints"]
    assert {"name", "x", "y", "confidence"} <= set(record["keypoints"][0])


def test_depth_surrogate_export_creation(tmp_path) -> None:
    input_path = tmp_path / "person.ppm"
    output_dir = tmp_path / "out"
    write_ppm(input_path, synthetic_person_image())

    PrivMotionPipeline(
        ProcessConfig(
            input_path=input_path,
            output_dir=output_dir,
            output_modes=("depth-surrogate",),
            pose_backend="prototype",
        )
    ).run()

    assert (output_dir / "depth_surrogates" / "frame_000000.pgm").exists()
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["retention"]["raw_rgb_written"] is False


def test_directory_input_skips_unreadable_frames(tmp_path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    write_ppm(frames_dir / "001.ppm", synthetic_person_image())
    (frames_dir / "002.ppm").write_text("not a valid ppm", encoding="utf-8")

    output_dir = tmp_path / "out"
    result = PrivMotionPipeline(
        ProcessConfig(input_path=frames_dir, output_dir=output_dir, output_modes=("skeleton",), pose_backend="prototype")
    ).run()

    assert result.processed_frames == 1
    assert result.skipped_frames == 1


def test_pipeline_processes_mp4_video_input(tmp_path) -> None:
    input_path = tmp_path / "person.mp4"
    output_dir = tmp_path / "out"
    frames = np.stack(
        [
            np.pad(synthetic_person_image(), ((12, 12), (12, 12), (0, 0))),
            np.roll(np.pad(synthetic_person_image(), ((12, 12), (12, 12), (0, 0))), 4, axis=1),
        ]
    ).astype(np.uint8)
    iio.imwrite(input_path, frames, fps=4)

    result = PrivMotionPipeline(
        ProcessConfig(
            input_path=input_path,
            output_dir=output_dir,
            output_modes=("skeleton", "silhouette", "depth-surrogate", "features"),
            pose_backend="prototype",
        )
    ).run()

    assert result.processed_frames == 2
    assert (output_dir / "silhouettes" / "frame_000001.pgm").exists()
    payload = json.loads((output_dir / "skeletons.json").read_text(encoding="utf-8"))
    assert len(payload["records"]) == 2
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["input_summary"]["input_type"] == "video"
    assert metadata["retention"]["raw_rgb_written"] is False


def test_yolo_bbox_constrains_silhouette_mask(tmp_path) -> None:
    input_path = tmp_path / "person.ppm"
    output_dir = tmp_path / "out"
    image = np.full((40, 40, 3), 240, dtype=np.uint8)
    write_ppm(input_path, image)

    PrivMotionPipeline(
        ProcessConfig(
            input_path=input_path,
            output_dir=output_dir,
            output_modes=("silhouette", "features"),
            pose_backend="yolo",
        ),
        segmenter=FakeFullFrameSegmenter(),
        pose_estimator=FakeYoloPoseEstimator(),
    ).run()

    from privmotion.io import read_portable_anymap

    mask = read_portable_anymap(output_dir / "silhouettes" / "frame_000000.pgm")
    assert mask is not None
    assert int((mask > 0).sum()) == 20 * 30


def test_yolo_no_detection_does_not_emit_full_frame_mask(tmp_path) -> None:
    input_path = tmp_path / "person.ppm"
    output_dir = tmp_path / "out"
    image = np.full((40, 40, 3), 240, dtype=np.uint8)
    write_ppm(input_path, image)

    PrivMotionPipeline(
        ProcessConfig(
            input_path=input_path,
            output_dir=output_dir,
            output_modes=("silhouette",),
            pose_backend="yolo",
        ),
        pose_estimator=FakeNoDetectionYoloPoseEstimator(),
    ).run()

    from privmotion.io import read_portable_anymap

    mask = read_portable_anymap(output_dir / "silhouettes" / "frame_000000.pgm")
    assert mask is not None
    assert int((mask > 0).sum()) == 0


def test_pipeline_metadata_records_injected_yolo_pose_backend(tmp_path) -> None:
    input_path = tmp_path / "person.ppm"
    output_dir = tmp_path / "out"
    write_ppm(input_path, synthetic_person_image())

    PrivMotionPipeline(
        ProcessConfig(
            input_path=input_path,
            output_dir=output_dir,
            output_modes=("skeleton", "features"),
            pose_backend="yolo",
            pose_model="mock-yolo-pose.pt",
        ),
        pose_estimator=FakeYoloPoseEstimator(),
    ).run()

    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["backends"]["pose"] == "yolo-pose"
    assert metadata["backends"]["pose_model"] == "mock-yolo-pose.pt"
    assert "YOLO-Pose model-backed 2D pose is active" in metadata["limitations"][1]
    features = json.loads((output_dir / "features.json").read_text(encoding="utf-8"))
    assert features["records"][0]["feature_type"] == "yolo-pose-kinematic-geometry"


class FakeYoloPoseEstimator:
    name = "yolo-pose"
    model_name = "mock-yolo-pose.pt"

    def estimate(self, image, mask, track_id):
        return PoseResult(
            track_id=track_id,
            keypoints=(
                Keypoint("nose", 20.0, 10.0, 0.9),
                Keypoint("left_ankle", 16.0, 30.0, 0.8),
            ),
            bbox_xywh=(10, 5, 20, 30),
            centroid_xy=(18.0, 20.0),
        )


class FakeFullFrameSegmenter:
    name = "fake-full-frame-segmenter"

    def segment(self, image):
        return np.full(image.shape[:2], 255, dtype=np.uint8)


class FakeNoDetectionYoloPoseEstimator:
    name = "yolo-pose"
    model_name = "mock-yolo-pose.pt"

    def estimate(self, image, mask, track_id):
        return PoseResult(track_id=0, keypoints=(), bbox_xywh=None, centroid_xy=None)
