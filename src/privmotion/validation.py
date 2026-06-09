from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from privmotion.exporters import write_json


RAW_RGB_NAMES = {
    "raw",
    "raw_rgb",
    "rgb",
    "frames",
    "original",
    "source_rgb",
}
RAW_RGB_SUFFIXES = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".tif", ".tiff"}
ALLOWED_IMAGE_DIRS = {"silhouettes", "depth_surrogates"}
VALID_DEIDENTIFICATION_PROFILES = {"standard", "hipaa-expert-aggregate"}
HIPAA_ALLOWED_FILES = {
    "metadata.json",
    "aggregate_report.json",
    "deidentification_report.json",
    "retention_report.json",
}
HIPAA_FORBIDDEN_DIRS = {
    "silhouettes",
    "depth_surrogates",
    "preview_frames",
    "frames",
}
HIPAA_FORBIDDEN_FILES = {
    "skeletons.json",
    "features.json",
    "access_policy.json",
    "audit_log.jsonl",
    "benchmark_report.json",
    "preview.gif",
    "preview.mp4",
}
HIPAA_FORBIDDEN_SUFFIXES = {".gif", ".mp4", ".png", ".jpg", ".jpeg", ".ppm", ".bmp", ".tif", ".tiff", ".pgm"}
HIPAA_FORBIDDEN_KEYS = {
    "access_policy_path",
    "audit_actor",
    "audit_purpose",
    "bbox_xywh",
    "centroid_xy",
    "created_unix",
    "expected_frames",
    "frame_index",
    "generated_unix",
    "id",
    "input",
    "input_path",
    "keypoint_velocity",
    "keypoints",
    "label",
    "manifest_path",
    "output_dir",
    "sample_id",
    "source_name",
    "split",
    "timestamp_ms",
    "track_id",
    "visualization",
}


@dataclass(frozen=True)
class RetentionValidationResult:
    output_dir: Path
    policy: str
    deidentification_profile: str
    passed: bool
    violations: tuple[str, ...]

    def to_json(self, redacted: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "policy": self.policy,
            "deidentification_profile": self.deidentification_profile,
            "passed": self.passed,
            "violations": list(self.violations),
        }
        if not redacted:
            payload["output_dir"] = str(self.output_dir)
        return payload


def validate_output_dir(
    output_dir: Path,
    policy: str = "no-raw-rgb",
    report_path: Path | None = None,
    deidentification_profile: str = "standard",
) -> RetentionValidationResult:
    path = Path(output_dir)
    if policy != "no-raw-rgb":
        raise ValueError("Phase 2 supports only the no-raw-rgb retention policy")
    normalized_profile = deidentification_profile.lower()
    if normalized_profile not in VALID_DEIDENTIFICATION_PROFILES:
        valid = ", ".join(sorted(VALID_DEIDENTIFICATION_PROFILES))
        raise ValueError(f"deidentification_profile must be one of: {valid}")
    if not path.exists():
        raise FileNotFoundError(f"output directory does not exist: {path}")

    violations: list[str] = []
    for child in path.rglob("*"):
        if not child.is_file():
            continue
        rel = child.relative_to(path)
        parts = [part.lower() for part in rel.parts]
        stem = child.stem.lower()
        suffix = child.suffix.lower()

        if parts[0] in ALLOWED_IMAGE_DIRS:
            continue
        if stem in RAW_RGB_NAMES or any(part in RAW_RGB_NAMES for part in parts[:-1]):
            violations.append(str(rel))
            continue
        if suffix in RAW_RGB_SUFFIXES and "raw" in stem:
            violations.append(str(rel))

    if normalized_profile == "hipaa-expert-aggregate":
        violations.extend(_hipaa_aggregate_violations(path))

    result = RetentionValidationResult(
        output_dir=path,
        policy=policy,
        deidentification_profile=normalized_profile,
        passed=not violations,
        violations=tuple(sorted(violations)),
    )
    if report_path is not None:
        write_json(Path(report_path), result.to_json(redacted=normalized_profile == "hipaa-expert-aggregate"))
    return result


def _hipaa_aggregate_violations(path: Path) -> list[str]:
    violations: list[str] = []
    metadata_path = path / "metadata.json"
    aggregate_path = path / "aggregate_report.json"
    deidentification_path = path / "deidentification_report.json"
    for required in (metadata_path, aggregate_path, deidentification_path):
        if not required.exists():
            violations.append(f"{required.name}:missing")

    for child in path.rglob("*"):
        rel = child.relative_to(path)
        rel_text = str(rel)
        first_part = rel.parts[0].lower()
        name = child.name.lower()
        suffix = child.suffix.lower()

        if child.is_dir():
            if first_part in HIPAA_FORBIDDEN_DIRS:
                violations.append(f"{rel_text}:person-level-directory")
            continue

        if first_part in HIPAA_FORBIDDEN_DIRS:
            violations.append(f"{rel_text}:person-level-artifact")
        if name in HIPAA_FORBIDDEN_FILES:
            violations.append(f"{rel_text}:person-level-artifact")
        if suffix in HIPAA_FORBIDDEN_SUFFIXES and rel_text not in HIPAA_ALLOWED_FILES:
            violations.append(f"{rel_text}:rendered-or-image-artifact")
        if rel_text not in HIPAA_ALLOWED_FILES:
            violations.append(f"{rel_text}:unsupported-file")
            continue
        if suffix == ".json":
            violations.extend(_identifier_key_violations(path, child))
    return sorted(set(violations))


def _identifier_key_violations(root: Path, json_path: Path) -> list[str]:
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [f"{json_path.relative_to(root)}:invalid-json"]
    rel = json_path.relative_to(root)
    return [f"{rel}:{key}" for key in sorted(_find_forbidden_keys(payload))]


def _find_forbidden_keys(payload: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in HIPAA_FORBIDDEN_KEYS:
                found.add(key)
            found.update(_find_forbidden_keys(value))
    elif isinstance(payload, list):
        for value in payload:
            found.update(_find_forbidden_keys(value))
    return found
