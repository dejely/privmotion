from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from privmotion.benchmark import benchmark_output_dir
from privmotion.config import ProcessConfig, parse_output_modes
from privmotion.exporters import write_json
from privmotion.pipeline import PrivMotionPipeline
from privmotion.visualization import visualize_output_dir


@dataclass(frozen=True)
class DatasetEvaluationReport:
    manifest_path: Path
    output_dir: Path
    sample_count: int
    processed_frame_count: int
    retention_pass_rate: float | None
    average_keypoint_coverage: float | None
    samples: tuple[dict[str, Any], ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "privmotion.dataset_eval.v0",
            "manifest_path": str(self.manifest_path),
            "output_dir": str(self.output_dir),
            "sample_count": self.sample_count,
            "processed_frame_count": self.processed_frame_count,
            "retention_pass_rate": self.retention_pass_rate,
            "average_keypoint_coverage": self.average_keypoint_coverage,
            "samples": list(self.samples),
        }


def evaluate_dataset_manifest(
    manifest_path: Path,
    output_dir: Path,
    output_modes: tuple[str, ...] = ("skeleton", "silhouette", "depth-surrogate", "features"),
    pose_backend: str = "auto",
    pose_model: str = "yolo11n-pose.pt",
    visualize: bool = False,
    visualization_ext: str = ".gif",
    fps: int = 4,
    size: tuple[int, int] = (640, 360),
) -> DatasetEvaluationReport:
    manifest = _load_manifest(Path(manifest_path))
    samples = manifest.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError("dataset manifest must contain a 'samples' list")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_base = Path(manifest_path).parent

    sample_reports: list[dict[str, Any]] = []
    retention_passes = 0
    retention_known = 0
    coverage_values: list[float] = []
    processed_frames_total = 0

    for offset, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError("each manifest sample must be an object")
        sample_id = _sample_id(sample, offset)
        input_path = _resolve_input_path(sample, manifest_base)
        sample_dir = root / sample_id
        process_result = PrivMotionPipeline(
            ProcessConfig(
                input_path=input_path,
                output_dir=sample_dir,
                output_modes=output_modes,
                pose_backend=pose_backend,
                pose_model=pose_model,
            )
        ).run()
        benchmark_report = benchmark_output_dir(sample_dir, report_path=sample_dir / "benchmark_report.json")
        preview_path = None
        if visualize:
            ext = visualization_ext if visualization_ext.startswith(".") else f".{visualization_ext}"
            preview_path = sample_dir / f"preview{ext}"
            visualize_output_dir(sample_dir, preview_path, fps=fps, size=size)

        processed_frames = int(benchmark_report.utility.get("processed_frame_count") or 0)
        processed_frames_total += processed_frames
        retention_passed = benchmark_report.privacy.get("raw_rgb_retention_passed")
        if retention_passed is not None:
            retention_known += 1
            retention_passes += 1 if retention_passed else 0
        coverage = benchmark_report.utility.get("keypoint_frame_coverage")
        if coverage is not None:
            coverage_values.append(float(coverage))

        sample_reports.append(
            {
                "id": sample_id,
                "input": str(input_path),
                "label": sample.get("label"),
                "split": sample.get("split"),
                "expected_frames": sample.get("expected_frames"),
                "output_dir": str(sample_dir),
                "processed_frames": process_result.processed_frames,
                "benchmark_report": str(sample_dir / "benchmark_report.json"),
                "visualization": str(preview_path) if preview_path is not None else None,
                "raw_rgb_retention_passed": retention_passed,
                "keypoint_frame_coverage": coverage,
            }
        )

    report = DatasetEvaluationReport(
        manifest_path=Path(manifest_path),
        output_dir=root,
        sample_count=len(sample_reports),
        processed_frame_count=processed_frames_total,
        retention_pass_rate=round(retention_passes / retention_known, 6) if retention_known else None,
        average_keypoint_coverage=round(sum(coverage_values) / len(coverage_values), 6)
        if coverage_values
        else None,
        samples=tuple(sample_reports),
    )
    write_json(root / "dataset_report.json", report.to_json())
    return report


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"dataset manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {"samples": payload}
    if not isinstance(payload, dict):
        raise ValueError("dataset manifest must be a JSON object or sample list")
    return payload


def _resolve_input_path(sample: dict[str, Any], manifest_base: Path) -> Path:
    value = sample.get("input")
    if not value:
        raise ValueError("each manifest sample requires an input path")
    path = Path(str(value))
    if not path.is_absolute():
        path = manifest_base / path
    return path


def _sample_id(sample: dict[str, Any], offset: int) -> str:
    raw = str(sample.get("id") or f"sample_{offset:04d}")
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
    return sanitized or f"sample_{offset:04d}"
