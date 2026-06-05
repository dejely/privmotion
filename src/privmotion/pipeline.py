from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import time

from privmotion.backends import (
    GeometryPoseEstimator,
    Keypoint,
    PrototypeSegmenter,
    SingleTrackAssigner,
    YoloPoseEstimator,
    create_pose_estimator,
    keypoint_velocity,
)
from privmotion.config import ProcessConfig
from privmotion.exporters import depth_surrogate_from_frame, write_json, write_pgm
from privmotion.io import load_frames
from privmotion.recovery import encrypt_feature_records, plaintext_feature_metadata
from privmotion.validation import validate_output_dir


@dataclass(frozen=True)
class ProcessResult:
    output_dir: Path
    processed_frames: int
    skipped_frames: int
    metadata_path: Path


class PrivMotionPipeline:
    def __init__(
        self,
        config: ProcessConfig,
        segmenter: PrototypeSegmenter | None = None,
        pose_estimator: GeometryPoseEstimator | YoloPoseEstimator | None = None,
        tracker: SingleTrackAssigner | None = None,
    ) -> None:
        self.config = config
        self.segmenter = segmenter or PrototypeSegmenter()
        self.pose_estimator = pose_estimator or create_pose_estimator(config.pose_backend, config.pose_model)
        self.tracker = tracker or SingleTrackAssigner()

    def run(self) -> ProcessResult:
        frames, input_summary = load_frames(self.config.input_path, max_frames=self.config.max_frames)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        skeleton_records: list[dict[str, object]] = []
        feature_records: list[dict[str, object]] = []
        feature_encryption_metadata = plaintext_feature_metadata()
        previous_keypoints: tuple[Keypoint, ...] | None = None

        for frame in frames:
            mask = self.segmenter.segment(frame.image)
            track_id = self.tracker.track_id_for_mask(mask)
            pose = self.pose_estimator.estimate(frame.image, mask, track_id)
            mask = self._privacy_mask_for_pose(mask, pose)
            if self.pose_estimator.name == "yolo-pose":
                track_id = pose.track_id
            velocities = keypoint_velocity(pose.keypoints, previous_keypoints)
            previous_keypoints = pose.keypoints if pose.keypoints else previous_keypoints

            frame_base = f"frame_{frame.index:06d}"
            if "silhouette" in self.config.output_modes:
                write_pgm(self.config.output_dir / "silhouettes" / f"{frame_base}.pgm", mask)

            if "depth-surrogate" in self.config.output_modes:
                surrogate = depth_surrogate_from_frame(frame.image)
                write_pgm(self.config.output_dir / "depth_surrogates" / f"{frame_base}.pgm", surrogate)

            if "skeleton" in self.config.output_modes:
                skeleton_records.append(
                    {
                        "frame_index": frame.index,
                        "timestamp_ms": frame.timestamp_ms,
                        "source_name": frame.source_name,
                        "track_id": pose.track_id,
                        "bbox_xywh": list(pose.bbox_xywh) if pose.bbox_xywh else None,
                        "centroid_xy": list(pose.centroid_xy) if pose.centroid_xy else None,
                        "keypoints": [point.to_json() for point in pose.keypoints],
                        "keypoint_velocity": velocities,
                    }
                )

            if "features" in self.config.output_modes:
                feature_records.append(
                    {
                        "frame_index": frame.index,
                        "timestamp_ms": frame.timestamp_ms,
                        "track_id": pose.track_id,
                        "feature_type": f"{self.pose_estimator.name}-kinematic-geometry",
                        "person_area_px": int((mask > 0).sum()),
                        "bbox_xywh": list(pose.bbox_xywh) if pose.bbox_xywh else None,
                        "centroid_xy": list(pose.centroid_xy) if pose.centroid_xy else None,
                    }
                )

        if "skeleton" in self.config.output_modes:
            write_json(
                self.config.output_dir / "skeletons.json",
                {
                    "schema": "privmotion.skeletons.v0",
                    "backend": self.pose_estimator.name,
                    "records": skeleton_records,
                },
            )
        if "features" in self.config.output_modes:
            if self.config.feature_encryption == "fernet":
                if self.config.access_policy_path is None:
                    raise ValueError("fernet feature encryption requires an access policy JSON file")
                encryption_result = encrypt_feature_records(
                    feature_records,
                    output_dir=self.config.output_dir,
                    access_policy_path=self.config.access_policy_path,
                    recovery_key_env=self.config.recovery_key_env,
                    audit_actor=self.config.audit_actor,
                    audit_purpose=self.config.audit_purpose or "",
                )
                feature_encryption_metadata = encryption_result.metadata
                write_json(self.config.output_dir / "features.json", encryption_result.features_payload)
            else:
                write_json(
                    self.config.output_dir / "features.json",
                    {
                        "schema": "privmotion.features.v0",
                        "encryption": feature_encryption_metadata,
                        "records": feature_records,
                    },
                )

        metadata = {
            "schema": "privmotion.metadata.v0",
            "created_unix": int(time()),
            "config": self.config.to_json(),
            "input_summary": input_summary.to_json(),
            "backends": {
                "segmentation": self.segmenter.name,
                "segmentation_mode": self._segmentation_mode(),
                "requested_pose_backend": self.config.pose_backend,
                "pose": self.pose_estimator.name,
                "pose_model": getattr(self.pose_estimator, "model_name", None),
                "pose_fallback_reason": getattr(self.pose_estimator, "fallback_reason", None),
                "tracking": self.tracker.name,
            },
            "retention": {
                "policy": self.config.retention_policy,
                "raw_rgb_written": False,
            },
            "feature_encryption": feature_encryption_metadata,
            "limitations": self._limitations(),
        }
        metadata_path = self.config.output_dir / "metadata.json"
        write_json(metadata_path, metadata)

        validation = validate_output_dir(self.config.output_dir, policy=self.config.retention_policy)
        write_json(self.config.output_dir / "retention_report.json", validation.to_json())

        return ProcessResult(
            output_dir=self.config.output_dir,
            processed_frames=input_summary.readable_frames,
            skipped_frames=input_summary.skipped_frames,
            metadata_path=metadata_path,
        )

    def _privacy_mask_for_pose(self, mask, pose):
        if self.pose_estimator.name != "yolo-pose":
            return mask
        if pose.bbox_xywh is None:
            return mask * 0

        x, y, width, height = pose.bbox_xywh
        constrained = mask * 0
        y0 = max(0, int(y))
        x0 = max(0, int(x))
        y1 = min(mask.shape[0], y0 + max(0, int(height)))
        x1 = min(mask.shape[1], x0 + max(0, int(width)))
        if y1 > y0 and x1 > x0:
            constrained[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
        return constrained

    def _segmentation_mode(self) -> str:
        if self.pose_estimator.name == "yolo-pose":
            return "luminance-constrained-to-yolo-person-bbox"
        return "prototype-luminance"

    def _limitations(self) -> list[str]:
        limitations = [
            "Phase 2 segmentation is deterministic placeholder logic, not a production person model.",
            "Skeletons, silhouettes, and depth surrogates can still leak identity through body shape or motion.",
        ]
        if self.pose_estimator.name == "yolo-pose":
            limitations.insert(
                1,
                "YOLO-Pose model-backed 2D pose is active, but detections can still be inaccurate or incomplete.",
            )
        else:
            limitations.insert(
                1,
                "Phase 2 skeletons are estimated from mask geometry, not anatomical pose inference.",
            )
        return limitations
