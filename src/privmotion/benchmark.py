from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any

import numpy as np

from privmotion.exporters import write_json
from privmotion.io import read_portable_anymap
from privmotion.validation import validate_output_dir


@dataclass(frozen=True)
class BenchmarkReport:
    output_dir: Path
    generated_unix: int
    utility: dict[str, Any]
    privacy: dict[str, Any]
    systems: dict[str, Any]
    notes: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "privmotion.benchmark.v0",
            "output_dir": str(self.output_dir),
            "generated_unix": self.generated_unix,
            "utility": self.utility,
            "privacy": self.privacy,
            "systems": self.systems,
            "notes": list(self.notes),
        }


def benchmark_output_dir(output_dir: Path, report_path: Path | None = None) -> BenchmarkReport:
    path = Path(output_dir)
    if not path.exists():
        raise FileNotFoundError(f"output directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"output path is not a directory: {path}")

    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"required metadata.json is missing from: {path}")

    metadata = _read_json(metadata_path)
    skeletons = _read_optional_json(path / "skeletons.json")
    features = _read_optional_json(path / "features.json")
    retention = _read_optional_json(path / "retention_report.json")

    skeleton_records = _records_from(skeletons)
    feature_records = _records_from(features)
    encrypted_feature_count = _encrypted_record_count(features)
    feature_record_count = encrypted_feature_count if encrypted_feature_count is not None else len(feature_records)
    features_encrypted = encrypted_feature_count is not None
    processed_frames = _input_summary_value(metadata, "readable_frames")
    skipped_frames = _input_summary_value(metadata, "skipped_frames")

    silhouette_stats = _image_dir_stats(path / "silhouettes")
    surrogate_stats = _image_dir_stats(path / "depth_surrogates")
    retention_result = _retention_result(path, retention)

    utility = {
        "processed_frame_count": processed_frames,
        "skeleton_record_count": len(skeleton_records) if skeletons is not None else None,
        "keypoint_frame_coverage": _safe_ratio(
            sum(1 for record in skeleton_records if record.get("keypoints")),
            processed_frames,
        )
        if skeletons is not None
        else None,
        "average_keypoints_per_record": _average_keypoint_count(skeleton_records)
        if skeletons is not None
        else None,
        "average_keypoint_confidence": _average_keypoint_confidence(skeleton_records)
        if skeletons is not None
        else None,
        "average_keypoint_velocity_px": _average_keypoint_velocity(skeleton_records)
        if skeletons is not None
        else None,
        "silhouette_count": silhouette_stats["count"],
        "silhouette_frame_coverage": _safe_ratio(silhouette_stats["count"], processed_frames),
        "average_silhouette_foreground_ratio": silhouette_stats["foreground_ratio"],
        "depth_surrogate_count": surrogate_stats["count"],
        "depth_surrogate_frame_coverage": _safe_ratio(surrogate_stats["count"], processed_frames),
        "feature_record_count": feature_record_count if features is not None else None,
        "encrypted_feature_record_count": encrypted_feature_count,
        "feature_frame_coverage": _safe_ratio(feature_record_count, processed_frames)
        if features is not None
        else None,
    }

    residual_risk_notes = [
        "Deterministic Phase 3 metrics are proxies, not face-recognition, re-identification, or gait-model scores.",
        "Skeletons, silhouettes, and depth surrogates can still leak identity through body shape and motion.",
    ]
    if features_encrypted:
        residual_risk_notes.append(
            "Encrypted feature payloads are counted but not decrypted or analyzed by benchmark reports."
        )

    privacy = {
        "raw_rgb_retention_passed": retention_result.get("passed"),
        "raw_rgb_retention_violations": retention_result.get("violations"),
        "mask_only_visual_outputs": _mask_only_visual_outputs(path),
        "surrogate_resolution_reduction": _surrogate_resolution_reduction(
            silhouette_stats,
            surrogate_stats,
        ),
        "feature_uniqueness_proxy": _feature_uniqueness_proxy(feature_records)
        if features is not None and not features_encrypted
        else None,
        "residual_risk_notes": residual_risk_notes,
    }

    systems = {
        "output_file_count": sum(1 for child in path.rglob("*") if child.is_file()),
        "output_byte_size": sum(child.stat().st_size for child in path.rglob("*") if child.is_file()),
        "skipped_frame_count": skipped_frames,
        "backends": metadata.get("backends", {}),
        "available_artifacts": {
            "metadata": True,
            "skeletons": skeletons is not None,
            "features": features is not None,
            "encrypted_features": features_encrypted,
            "retention_report": retention is not None,
            "silhouettes": silhouette_stats["count"] > 0,
            "depth_surrogates": surrogate_stats["count"] > 0,
        },
    }

    report = BenchmarkReport(
        output_dir=path,
        generated_unix=int(time()),
        utility=utility,
        privacy=privacy,
        systems=systems,
        notes=(
            "Phase 3 benchmark metrics are deterministic local proxies.",
            "External ORPose-Depth and Market-1501-style dataset adapters are not implemented in this phase.",
        ),
    )
    if report_path is not None:
        write_json(Path(report_path), report.to_json())
    return report


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _read_json(path)


def _records_from(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if payload is None:
        return []
    records = payload.get("records", [])
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, dict)]


def _encrypted_record_count(payload: dict[str, Any] | None) -> int | None:
    if payload is None:
        return None
    encrypted_records = payload.get("encrypted_records")
    if isinstance(encrypted_records, list):
        return len(encrypted_records)
    return None


def _input_summary_value(metadata: dict[str, Any], key: str) -> int:
    value = metadata.get("input_summary", {}).get(key, 0)
    return int(value or 0)


def _average_keypoint_count(records: list[dict[str, Any]]) -> float | None:
    if not records:
        return None
    return round(sum(len(record.get("keypoints", [])) for record in records) / len(records), 6)


def _average_keypoint_confidence(records: list[dict[str, Any]]) -> float | None:
    confidences: list[float] = []
    for record in records:
        for point in record.get("keypoints", []):
            if isinstance(point, dict) and "confidence" in point:
                confidences.append(float(point["confidence"]))
    if not confidences:
        return None
    return round(sum(confidences) / len(confidences), 6)


def _average_keypoint_velocity(records: list[dict[str, Any]]) -> float | None:
    magnitudes: list[float] = []
    for record in records:
        velocities = record.get("keypoint_velocity", {})
        if not isinstance(velocities, dict):
            continue
        for value in velocities.values():
            if not isinstance(value, dict):
                continue
            dx = float(value.get("dx", 0.0))
            dy = float(value.get("dy", 0.0))
            magnitudes.append(math.hypot(dx, dy))
    if not magnitudes:
        return None
    return round(sum(magnitudes) / len(magnitudes), 6)


def _image_dir_stats(path: Path) -> dict[str, Any]:
    files = sorted(path.glob("*.pgm")) if path.exists() else []
    pixel_counts: list[int] = []
    foreground_ratios: list[float] = []
    for image_path in files:
        image = read_portable_anymap(image_path)
        if image is None:
            continue
        array = np.asarray(image)
        pixel_counts.append(int(array.size))
        if array.size:
            foreground_ratios.append(float((array > 0).sum()) / float(array.size))

    return {
        "count": len(files),
        "average_pixels": round(sum(pixel_counts) / len(pixel_counts), 6) if pixel_counts else None,
        "foreground_ratio": round(sum(foreground_ratios) / len(foreground_ratios), 6)
        if foreground_ratios
        else None,
    }


def _retention_result(path: Path, retention: dict[str, Any] | None) -> dict[str, Any]:
    if retention is not None:
        return {
            "passed": bool(retention.get("passed")),
            "violations": retention.get("violations", []),
        }
    validation = validate_output_dir(path)
    return {
        "passed": validation.passed,
        "violations": list(validation.violations),
    }


def _mask_only_visual_outputs(path: Path) -> bool:
    allowed_dirs = {"silhouettes", "depth_surrogates"}
    image_suffixes = {".bmp", ".jpeg", ".jpg", ".pgm", ".png", ".ppm", ".tif", ".tiff"}
    for child in path.rglob("*"):
        if not child.is_file() or child.suffix.lower() not in image_suffixes:
            continue
        rel = child.relative_to(path)
        if rel.parts[0] not in allowed_dirs:
            return False
    return True


def _surrogate_resolution_reduction(
    silhouette_stats: dict[str, Any],
    surrogate_stats: dict[str, Any],
) -> float | None:
    silhouette_pixels = silhouette_stats.get("average_pixels")
    surrogate_pixels = surrogate_stats.get("average_pixels")
    if not silhouette_pixels or not surrogate_pixels:
        return None
    reduction = 1.0 - (float(surrogate_pixels) / float(silhouette_pixels))
    return round(max(0.0, min(1.0, reduction)), 6)


def _feature_uniqueness_proxy(records: list[dict[str, Any]]) -> float | None:
    if not records:
        return None
    signatures = {
        json.dumps(
            {
                "area": record.get("person_area_px"),
                "bbox": record.get("bbox_xywh"),
                "centroid": record.get("centroid_xy"),
            },
            sort_keys=True,
        )
        for record in records
    }
    return round(len(signatures) / len(records), 6)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 6)
