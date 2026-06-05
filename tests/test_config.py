from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from privmotion.config import ProcessConfig, parse_output_modes


def test_parse_output_modes_normalizes_and_deduplicates() -> None:
    assert parse_output_modes("skeleton, depth_surrogate, skeleton") == (
        "skeleton",
        "depth-surrogate",
    )


def test_parse_output_modes_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unsupported output mode"):
        parse_output_modes("skeleton,raw-rgb")


def test_process_config_rejects_non_default_retention(tmp_path) -> None:
    with pytest.raises(ValueError, match="no-raw-rgb"):
        ProcessConfig(
            input_path=tmp_path / "in.ppm",
            output_dir=tmp_path / "out",
            retention_policy="keep-raw-rgb",
        )


def test_process_config_rejects_unknown_pose_backend(tmp_path) -> None:
    with pytest.raises(ValueError, match="pose_backend"):
        ProcessConfig(
            input_path=tmp_path / "in.ppm",
            output_dir=tmp_path / "out",
            pose_backend="unknown",
        )


def test_process_config_records_pose_model(tmp_path) -> None:
    config = ProcessConfig(
        input_path=tmp_path / "in.ppm",
        output_dir=tmp_path / "out",
        pose_backend="yolo",
        pose_model="custom-pose.pt",
    )

    assert config.to_json()["pose_model"] == "custom-pose.pt"


def test_process_config_defaults_to_no_feature_encryption(tmp_path) -> None:
    config = ProcessConfig(input_path=tmp_path / "in.ppm", output_dir=tmp_path / "out")

    assert config.feature_encryption == "none"
    assert config.to_json()["feature_encryption"] == "none"


def test_process_config_rejects_unknown_feature_encryption(tmp_path) -> None:
    with pytest.raises(ValueError, match="feature_encryption"):
        ProcessConfig(
            input_path=tmp_path / "in.ppm",
            output_dir=tmp_path / "out",
            feature_encryption="rot13",
        )


def test_process_config_rejects_fernet_without_features_mode(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PRIVMOTION_RECOVERY_KEY", Fernet.generate_key().decode("ascii"))
    policy_path = tmp_path / "policy.json"
    policy_path.write_text('{"policy_id": "test-policy"}', encoding="utf-8")

    with pytest.raises(ValueError, match="features output mode"):
        ProcessConfig(
            input_path=tmp_path / "in.ppm",
            output_dir=tmp_path / "out",
            output_modes=("skeleton",),
            feature_encryption="fernet",
            access_policy_path=policy_path,
            audit_purpose="unit test",
        )


def test_process_config_rejects_fernet_without_env_key(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("PRIVMOTION_RECOVERY_KEY", raising=False)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text('{"policy_id": "test-policy"}', encoding="utf-8")

    with pytest.raises(ValueError, match="PRIVMOTION_RECOVERY_KEY"):
        ProcessConfig(
            input_path=tmp_path / "in.ppm",
            output_dir=tmp_path / "out",
            output_modes=("features",),
            feature_encryption="fernet",
            access_policy_path=policy_path,
            audit_purpose="unit test",
        )
