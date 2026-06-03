from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - depends on local environment
    cv2 = None


@dataclass(frozen=True)
class Keypoint:
    name: str
    x: float
    y: float
    confidence: float

    def to_json(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "confidence": round(self.confidence, 3),
        }


@dataclass(frozen=True)
class PoseResult:
    track_id: int
    keypoints: tuple[Keypoint, ...]
    bbox_xywh: tuple[int, int, int, int] | None
    centroid_xy: tuple[float, float] | None


class PrototypeSegmenter:
    """Deterministic placeholder segmenter for Phase 2.

    This is not a production person-segmentation model. It produces a stable
    foreground-like mask from luminance so the rest of the anonymization
    pipeline can be exercised without model downloads.
    """

    name = "prototype-luminance-segmenter"

    def segment(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            gray = (
                0.299 * image[:, :, 0]
                + 0.587 * image[:, :, 1]
                + 0.114 * image[:, :, 2]
            ).astype(np.uint8)
        else:
            gray = image.astype(np.uint8)

        threshold = max(10, int(gray.mean() + 0.25 * gray.std()))
        mask = np.where(gray > threshold, 255, 0).astype(np.uint8)

        if cv2 is not None:
            kernel = np.ones((3, 3), dtype=np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask


class SingleTrackAssigner:
    name = "prototype-single-track"

    def track_id_for_mask(self, mask: np.ndarray) -> int:
        return 1 if np.any(mask > 0) else 0


class GeometryPoseEstimator:
    """Non-production skeleton placeholder derived from mask geometry."""

    name = "prototype-geometry-pose"
    model_name: str | None = None

    def __init__(
        self,
        requested_backend: str = "prototype",
        fallback_reason: str | None = None,
    ) -> None:
        self.requested_backend = requested_backend
        self.fallback_reason = fallback_reason

    def estimate(self, image: np.ndarray, mask: np.ndarray, track_id: int) -> PoseResult:
        points = np.argwhere(mask > 0)
        if points.size == 0 or track_id == 0:
            return PoseResult(track_id=0, keypoints=(), bbox_xywh=None, centroid_xy=None)

        ys = points[:, 0]
        xs = points[:, 1]
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        width = max(1, x_max - x_min + 1)
        height = max(1, y_max - y_min + 1)
        cx = float(xs.mean())

        keypoints = (
            Keypoint("head", cx, y_min + 0.12 * height, 0.55),
            Keypoint("neck", cx, y_min + 0.25 * height, 0.6),
            Keypoint("left_shoulder", x_min + 0.30 * width, y_min + 0.30 * height, 0.5),
            Keypoint("right_shoulder", x_min + 0.70 * width, y_min + 0.30 * height, 0.5),
            Keypoint("hip_center", cx, y_min + 0.60 * height, 0.55),
            Keypoint("left_ankle", x_min + 0.40 * width, y_max, 0.45),
            Keypoint("right_ankle", x_min + 0.60 * width, y_max, 0.45),
        )
        return PoseResult(
            track_id=track_id,
            keypoints=keypoints,
            bbox_xywh=(x_min, y_min, width, height),
            centroid_xy=(float(xs.mean()), float(ys.mean())),
        )


COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


class YoloPoseEstimator:
    """Opt-in Ultralytics YOLO-Pose backend.

    Frames are passed to YOLO in memory only; raw RGB frames are not persisted by
    this backend.
    """

    name = "yolo-pose"

    def __init__(self, model_name: str = "yolo11n-pose.pt", model: Any | None = None) -> None:
        self.model_name = model_name
        self.requested_backend = "yolo"
        self.fallback_reason: str | None = None
        if model is not None:
            self.model = model
            return

        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "YOLO-Pose backend requires the optional pose extra: "
                'python -m pip install -e ".[pose]"'
            ) from exc
        self.model = YOLO(model_name)

    def estimate(self, image: np.ndarray, mask: np.ndarray, track_id: int) -> PoseResult:
        results = self.model(image, verbose=False)
        if not results:
            return PoseResult(track_id=0, keypoints=(), bbox_xywh=None, centroid_xy=None)
        return self.pose_result_from_yolo_result(results[0], track_id=track_id)

    def pose_result_from_yolo_result(self, result: Any, track_id: int) -> PoseResult:
        keypoints_xy = _to_numpy(getattr(getattr(result, "keypoints", None), "xy", None))
        if keypoints_xy is None or keypoints_xy.size == 0:
            return PoseResult(track_id=0, keypoints=(), bbox_xywh=None, centroid_xy=None)

        if keypoints_xy.ndim == 2:
            keypoints_xy = keypoints_xy[np.newaxis, :, :]

        boxes = getattr(result, "boxes", None)
        confidences = _to_numpy(getattr(boxes, "conf", None))
        selected_index = _highest_confidence_index(confidences, keypoints_xy.shape[0])

        keypoint_conf = _to_numpy(getattr(getattr(result, "keypoints", None), "conf", None))
        if keypoint_conf is not None and keypoint_conf.ndim == 1:
            keypoint_conf = keypoint_conf[np.newaxis, :]

        selected_xy = keypoints_xy[selected_index]
        selected_conf = (
            keypoint_conf[selected_index]
            if keypoint_conf is not None and selected_index < keypoint_conf.shape[0]
            else np.ones((selected_xy.shape[0],), dtype=np.float32)
        )

        keypoints: list[Keypoint] = []
        visible_points: list[tuple[float, float]] = []
        for idx, xy in enumerate(selected_xy):
            if idx >= len(COCO_KEYPOINT_NAMES):
                break
            x, y = float(xy[0]), float(xy[1])
            confidence = float(selected_conf[idx]) if idx < len(selected_conf) else 1.0
            if confidence <= 0:
                continue
            keypoints.append(Keypoint(COCO_KEYPOINT_NAMES[idx], x, y, confidence))
            visible_points.append((x, y))

        if not keypoints:
            return PoseResult(track_id=0, keypoints=(), bbox_xywh=None, centroid_xy=None)

        bbox = _bbox_from_result_or_points(boxes, selected_index, visible_points)
        centroid = (
            float(sum(point[0] for point in visible_points) / len(visible_points)),
            float(sum(point[1] for point in visible_points) / len(visible_points)),
        )
        return PoseResult(
            track_id=track_id or 1,
            keypoints=tuple(keypoints),
            bbox_xywh=bbox,
            centroid_xy=centroid,
        )


def create_pose_estimator(backend: str, model_name: str = "yolo11n-pose.pt") -> GeometryPoseEstimator | YoloPoseEstimator:
    normalized = backend.lower()
    if normalized == "prototype":
        return GeometryPoseEstimator(requested_backend="prototype")
    if normalized == "auto":
        try:
            estimator = YoloPoseEstimator(model_name=model_name)
            estimator.requested_backend = "auto"
            return estimator
        except RuntimeError as exc:
            return GeometryPoseEstimator(requested_backend="auto", fallback_reason=str(exc))
    if normalized == "yolo":
        return YoloPoseEstimator(model_name=model_name)
    raise ValueError("pose_backend must be one of: auto, prototype, yolo")


def _to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    return array if array.size else None


def _highest_confidence_index(confidences: np.ndarray | None, candidate_count: int) -> int:
    if candidate_count <= 0:
        return 0
    if confidences is None or confidences.size == 0:
        return 0
    return int(np.argmax(confidences[:candidate_count]))


def _bbox_from_result_or_points(
    boxes: Any,
    selected_index: int,
    points: list[tuple[float, float]],
) -> tuple[int, int, int, int] | None:
    xyxy = _to_numpy(getattr(boxes, "xyxy", None)) if boxes is not None else None
    if xyxy is not None and xyxy.ndim >= 2 and selected_index < xyxy.shape[0]:
        x1, y1, x2, y2 = [float(value) for value in xyxy[selected_index][:4]]
        return (
            int(round(x1)),
            int(round(y1)),
            max(1, int(round(x2 - x1))),
            max(1, int(round(y2 - y1))),
        )

    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return (
        int(round(x_min)),
        int(round(y_min)),
        max(1, int(round(x_max - x_min))),
        max(1, int(round(y_max - y_min))),
    )


def keypoint_velocity(
    current: Iterable[Keypoint],
    previous: Iterable[Keypoint] | None,
) -> dict[str, dict[str, float]]:
    if previous is None:
        return {}

    previous_by_name = {point.name: point for point in previous}
    velocities: dict[str, dict[str, float]] = {}
    for point in current:
        prior = previous_by_name.get(point.name)
        if prior is None:
            continue
        velocities[point.name] = {
            "dx": round(point.x - prior.x, 3),
            "dy": round(point.y - prior.y, 3),
        }
    return velocities
