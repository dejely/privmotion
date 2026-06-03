from __future__ import annotations

import sys

import numpy as np
import pytest

from privmotion.backends import GeometryPoseEstimator, YoloPoseEstimator, create_pose_estimator


def test_create_pose_estimator_returns_prototype_for_auto_and_prototype() -> None:
    assert isinstance(create_pose_estimator("prototype"), GeometryPoseEstimator)


def test_create_pose_estimator_auto_uses_yolo_when_available(monkeypatch) -> None:
    class FakeYoloPoseEstimator:
        name = "yolo-pose"
        model_name = "fake.pt"
        fallback_reason = None

        def __init__(self, model_name):
            self.model_name = model_name
            self.requested_backend = "yolo"

    monkeypatch.setattr("privmotion.backends.YoloPoseEstimator", FakeYoloPoseEstimator)

    estimator = create_pose_estimator("auto", model_name="fake.pt")

    assert estimator.name == "yolo-pose"
    assert estimator.requested_backend == "auto"


def test_create_pose_estimator_auto_falls_back_to_prototype(monkeypatch) -> None:
    class FailingYoloPoseEstimator:
        def __init__(self, model_name):
            raise RuntimeError("missing yolo")

    monkeypatch.setattr("privmotion.backends.YoloPoseEstimator", FailingYoloPoseEstimator)

    estimator = create_pose_estimator("auto", model_name="fake.pt")

    assert isinstance(estimator, GeometryPoseEstimator)
    assert estimator.requested_backend == "auto"
    assert estimator.fallback_reason == "missing yolo"


def test_yolo_pose_estimator_missing_dependency_has_clear_error(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "ultralytics", None)

    with pytest.raises(RuntimeError, match=r'\.\[pose\]'):
        YoloPoseEstimator()


def test_yolo_pose_estimator_converts_highest_confidence_detection() -> None:
    xy = np.zeros((2, 17, 2), dtype=np.float32)
    xy[0, :, :] = 5
    xy[1, :, 0] = np.arange(17, dtype=np.float32) + 100
    xy[1, :, 1] = np.arange(17, dtype=np.float32) + 200
    conf = np.full((2, 17), 0.8, dtype=np.float32)
    boxes = FakeBoxes(
        conf=np.array([0.1, 0.9], dtype=np.float32),
        xyxy=np.array([[0, 0, 10, 10], [90, 190, 130, 250]], dtype=np.float32),
    )
    result = FakeResult(keypoints=FakeKeypoints(xy=xy, conf=conf), boxes=boxes)

    pose = YoloPoseEstimator(model=FakeModel([result])).pose_result_from_yolo_result(result, track_id=7)

    assert pose.track_id == 7
    assert pose.bbox_xywh == (90, 190, 40, 60)
    assert pose.centroid_xy is not None
    assert pose.keypoints[0].name == "nose"
    assert pose.keypoints[0].x == 100
    assert pose.keypoints[0].y == 200
    assert pose.keypoints[0].confidence == pytest.approx(0.8)
    assert pose.keypoints[-1].name == "right_ankle"


class FakeModel:
    def __init__(self, results):
        self.results = results

    def __call__(self, image, verbose=False):
        return self.results


class FakeBoxes:
    def __init__(self, conf, xyxy):
        self.conf = conf
        self.xyxy = xyxy


class FakeKeypoints:
    def __init__(self, xy, conf):
        self.xy = xy
        self.conf = conf


class FakeResult:
    def __init__(self, keypoints, boxes):
        self.keypoints = keypoints
        self.boxes = boxes
