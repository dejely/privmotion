from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from cryptography.fernet import Fernet

from privmotion.benchmark import benchmark_output_dir
from privmotion.cli.recovery_inspect import main as recovery_inspect_main
from privmotion.config import ProcessConfig
from privmotion.pipeline import PrivMotionPipeline
from privmotion.recovery import inspect_recovery_policy
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


def write_policy(path: Path, policy_id: str = "phase5-test-policy") -> None:
    path.write_text(
        json.dumps(
            {
                "policy_id": policy_id,
                "allowed_purposes": ["unit test", "inspect"],
                "requires_audit": True,
            }
        ),
        encoding="utf-8",
    )


def build_encrypted_output(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("PRIVMOTION_RECOVERY_KEY", Fernet.generate_key().decode("ascii"))
    input_path = tmp_path / "person.ppm"
    policy_path = tmp_path / "policy.json"
    output_dir = tmp_path / "out"
    write_ppm(input_path, synthetic_person_image())
    write_policy(policy_path)

    PrivMotionPipeline(
        ProcessConfig(
            input_path=input_path,
            output_dir=output_dir,
            output_modes=("features", "skeleton"),
            pose_backend="prototype",
            feature_encryption="fernet",
            access_policy_path=policy_path,
            audit_actor="tester",
            audit_purpose="unit test",
        )
    ).run()
    return output_dir


def test_pipeline_encrypts_feature_records_without_plaintext(tmp_path, monkeypatch) -> None:
    output_dir = build_encrypted_output(tmp_path, monkeypatch)

    features_text = (output_dir / "features.json").read_text(encoding="utf-8")
    features = json.loads(features_text)
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))

    assert features["schema"] == "privmotion.features.v1"
    assert features["encryption"]["mode"] == "fernet"
    assert len(features["encrypted_records"]) == 1
    assert "records" not in features
    assert "person_area_px" not in features_text
    assert "bbox_xywh" not in features_text
    assert "centroid_xy" not in features_text
    assert metadata["feature_encryption"]["mode"] == "fernet"
    assert metadata["feature_encryption"]["policy_id"] == "phase5-test-policy"
    assert metadata["feature_encryption"]["key_env"] == "PRIVMOTION_RECOVERY_KEY"
    assert metadata["feature_encryption"]["encrypted_record_count"] == 1
    assert (output_dir / "access_policy.json").exists()
    assert (output_dir / "audit_log.jsonl").exists()
    assert validate_output_dir(output_dir).passed


def test_recovery_inspect_reports_policy_and_appends_audit_event(tmp_path, monkeypatch) -> None:
    output_dir = build_encrypted_output(tmp_path, monkeypatch)
    before = (output_dir / "audit_log.jsonl").read_text(encoding="utf-8").splitlines()

    result = inspect_recovery_policy(output_dir, audit_actor="reviewer", audit_purpose="inspect")

    after = (output_dir / "audit_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert result.encrypted is True
    assert result.policy_id == "phase5-test-policy"
    assert result.encrypted_record_count == 1
    assert result.audit_event_count == len(before) + 1
    assert len(after) == len(before) + 1
    assert json.loads(after[-1])["event"] == "policy_inspected"


def test_recovery_inspect_cli_prints_status(tmp_path, monkeypatch, capsys) -> None:
    output_dir = build_encrypted_output(tmp_path, monkeypatch)

    exit_code = recovery_inspect_main(
        [
            "--output",
            str(output_dir),
            "--audit-actor",
            "reviewer",
            "--audit-purpose",
            "inspect",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "encrypted=true" in captured.out
    assert "policy_id=phase5-test-policy" in captured.out
    assert "decryption_performed=false" in captured.out


def test_recovery_inspect_fails_for_missing_metadata(tmp_path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="metadata.json"):
        inspect_recovery_policy(output_dir)


def test_recovery_inspect_fails_for_plaintext_features(tmp_path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "metadata.json").write_text(
        json.dumps({"feature_encryption": {"mode": "fernet", "policy_id": "policy"}}),
        encoding="utf-8",
    )
    (output_dir / "features.json").write_text(
        json.dumps({"schema": "privmotion.features.v0", "records": [{"person_area_px": 1}]}),
        encoding="utf-8",
    )
    (output_dir / "access_policy.json").write_text(
        json.dumps({"schema": "privmotion.access_policy.v0", "policy_id": "policy"}),
        encoding="utf-8",
    )
    (output_dir / "audit_log.jsonl").write_text(
        json.dumps({"event": "feature_records_encrypted", "policy_id": "policy"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="plaintext records"):
        inspect_recovery_policy(output_dir)


def test_benchmark_counts_encrypted_features_without_decrypting(tmp_path, monkeypatch) -> None:
    output_dir = build_encrypted_output(tmp_path, monkeypatch)

    report = benchmark_output_dir(output_dir)

    assert report.utility["feature_record_count"] == 1
    assert report.utility["encrypted_feature_record_count"] == 1
    assert report.privacy["feature_uniqueness_proxy"] is None
    assert report.systems["available_artifacts"]["features"] is True
    assert report.systems["available_artifacts"]["encrypted_features"] is True
