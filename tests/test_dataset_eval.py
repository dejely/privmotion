from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from privmotion.cli.dataset_eval import main as dataset_eval_main
from privmotion.dataset_eval import evaluate_dataset_manifest


def write_ppm(path: Path, image: np.ndarray) -> None:
    height, width, channels = image.shape
    assert channels == 3
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + image.astype(np.uint8).tobytes())


def synthetic_person_image() -> np.ndarray:
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[8:34, 14:26] = (230, 230, 230)
    image[4:12, 16:24] = (250, 250, 250)
    return image


def test_evaluate_dataset_manifest_processes_samples(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_ppm(data_dir / "a.ppm", synthetic_person_image())
    write_ppm(data_dir / "b.ppm", np.roll(synthetic_person_image(), 2, axis=1))
    manifest = tmp_path / "dataset.json"
    manifest.write_text(
        json.dumps(
            {
                "samples": [
                    {"id": "sample-a", "input": "data/a.ppm", "label": "walk", "split": "test"},
                    {"id": "sample-b", "input": "data/b.ppm", "label": "wave", "expected_frames": 1},
                ]
            }
        ),
        encoding="utf-8",
    )

    report = evaluate_dataset_manifest(
        manifest,
        tmp_path / "eval",
        pose_backend="prototype",
        visualize=True,
        visualization_ext=".gif",
    )

    assert report.sample_count == 2
    assert report.processed_frame_count == 2
    assert report.retention_pass_rate == 1.0
    assert report.average_keypoint_coverage == 1.0
    assert (tmp_path / "eval" / "sample-a" / "benchmark_report.json").exists()
    assert (tmp_path / "eval" / "sample-a" / "preview.gif").exists()
    assert (tmp_path / "eval" / "dataset_report.json").exists()


def test_dataset_eval_cli_writes_report(tmp_path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_ppm(data_dir / "a.ppm", synthetic_person_image())
    manifest = tmp_path / "dataset.json"
    manifest.write_text(json.dumps([{"id": "sample-a", "input": "data/a.ppm"}]), encoding="utf-8")

    exit_code = dataset_eval_main(
        [
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "eval"),
            "--pose-backend",
            "prototype",
        ]
    )

    assert exit_code == 0
    payload = json.loads((tmp_path / "eval" / "dataset_report.json").read_text(encoding="utf-8"))
    assert payload["schema"] == "privmotion.dataset_eval.v0"
    assert payload["sample_count"] == 1
