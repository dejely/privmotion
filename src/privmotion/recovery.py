from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any

from cryptography.fernet import Fernet

from privmotion.exporters import write_json


@dataclass(frozen=True)
class FeatureEncryptionResult:
    features_payload: dict[str, Any]
    metadata: dict[str, Any]
    policy_id: str
    encrypted_record_count: int


@dataclass(frozen=True)
class RecoveryInspectionResult:
    output_dir: Path
    encrypted: bool
    policy_id: str
    encrypted_record_count: int
    audit_event_count: int
    audit_log_path: Path

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "privmotion.recovery_inspection.v0",
            "output_dir": str(self.output_dir),
            "encrypted": self.encrypted,
            "policy_id": self.policy_id,
            "encrypted_record_count": self.encrypted_record_count,
            "audit_event_count": self.audit_event_count,
            "audit_log_path": str(self.audit_log_path),
            "raw_rgb_recovery_supported": False,
            "decryption_performed": False,
        }


def plaintext_feature_metadata() -> dict[str, Any]:
    return {
        "mode": "none",
        "encrypted_record_count": 0,
        "policy_id": None,
        "audit_log": None,
        "raw_rgb_recovery_supported": False,
    }


def encrypt_feature_records(
    records: list[dict[str, object]],
    output_dir: Path,
    access_policy_path: Path,
    recovery_key_env: str,
    audit_actor: str | None,
    audit_purpose: str,
) -> FeatureEncryptionResult:
    fernet, key_id = _fernet_from_env(recovery_key_env)
    normalized_policy = load_access_policy(access_policy_path)
    output_path = Path(output_dir)
    write_json(output_path / "access_policy.json", normalized_policy)

    encrypted_records: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        plaintext = json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        encrypted_records.append(
            {
                "record_index": index,
                "encoding": "fernet-json-v1",
                "ciphertext": fernet.encrypt(plaintext).decode("ascii"),
            }
        )

    policy_id = str(normalized_policy["policy_id"])
    metadata = {
        "mode": "fernet",
        "algorithm": "Fernet",
        "key_env": recovery_key_env,
        "key_id": key_id,
        "policy_id": policy_id,
        "access_policy": "access_policy.json",
        "audit_log": "audit_log.jsonl",
        "encrypted_record_count": len(encrypted_records),
        "raw_rgb_recovery_supported": False,
        "decryption_cli_available": False,
    }
    features_payload = {
        "schema": "privmotion.features.v1",
        "encryption": metadata,
        "encrypted_records": encrypted_records,
    }
    append_audit_event(
        output_path,
        event="feature_records_encrypted",
        actor=audit_actor,
        purpose=audit_purpose,
        policy_id=policy_id,
        details={
            "encrypted_record_count": len(encrypted_records),
            "feature_encryption": "fernet",
            "raw_rgb_recovery_supported": False,
        },
    )
    return FeatureEncryptionResult(
        features_payload=features_payload,
        metadata=metadata,
        policy_id=policy_id,
        encrypted_record_count=len(encrypted_records),
    )


def inspect_recovery_policy(
    output_dir: Path,
    audit_actor: str | None = None,
    audit_purpose: str | None = None,
) -> RecoveryInspectionResult:
    output_path = Path(output_dir)
    if not output_path.exists():
        raise FileNotFoundError(f"output directory does not exist: {output_path}")
    if not output_path.is_dir():
        raise NotADirectoryError(f"output path is not a directory: {output_path}")

    metadata = _read_required_json(output_path / "metadata.json", "metadata.json")
    features = _read_required_json(output_path / "features.json", "features.json")
    access_policy = _read_required_json(output_path / "access_policy.json", "access_policy.json")
    audit_log_path = output_path / "audit_log.jsonl"
    if not audit_log_path.exists():
        raise FileNotFoundError(f"required audit_log.jsonl is missing from: {output_path}")

    encrypted_records = features.get("encrypted_records")
    if "records" in features:
        raise ValueError("features.json contains plaintext records; expected encrypted Phase 5 features")
    if not isinstance(encrypted_records, list):
        raise ValueError("features.json is not encrypted; expected encrypted_records")

    feature_encryption = features.get("encryption", {})
    if not isinstance(feature_encryption, dict) or feature_encryption.get("mode") != "fernet":
        raise ValueError("features.json encryption metadata must declare mode=fernet")

    metadata_encryption = metadata.get("feature_encryption", {})
    if not isinstance(metadata_encryption, dict) or metadata_encryption.get("mode") != "fernet":
        raise ValueError("metadata.json feature_encryption must declare mode=fernet")

    policy_id = _policy_id_from_sources(feature_encryption, metadata_encryption, access_policy)
    audit_events = _read_audit_events(audit_log_path)
    append_audit_event(
        output_path,
        event="policy_inspected",
        actor=audit_actor,
        purpose=audit_purpose or "inspect encrypted feature policy",
        policy_id=policy_id,
        details={
            "encrypted_record_count": len(encrypted_records),
            "decryption_performed": False,
            "raw_rgb_recovery_supported": False,
        },
    )

    return RecoveryInspectionResult(
        output_dir=output_path,
        encrypted=True,
        policy_id=policy_id,
        encrypted_record_count=len(encrypted_records),
        audit_event_count=len(audit_events) + 1,
        audit_log_path=audit_log_path,
    )


def load_access_policy(path: Path) -> dict[str, Any]:
    policy_path = Path(path)
    if not policy_path.exists():
        raise FileNotFoundError(f"access policy does not exist: {policy_path}")
    try:
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"access policy must be valid JSON: {policy_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("access policy must be a JSON object")

    policy_id = str(payload.get("policy_id") or payload.get("id") or _stable_policy_id(payload)).strip()
    if not policy_id:
        raise ValueError("access policy id must not be empty")
    return {
        "schema": "privmotion.access_policy.v0",
        "policy_id": policy_id,
        "source_path": str(policy_path),
        "policy": payload,
    }


def append_audit_event(
    output_dir: Path,
    event: str,
    actor: str | None,
    purpose: str | None,
    policy_id: str,
    details: dict[str, Any] | None = None,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "privmotion.audit_event.v0",
        "created_unix": int(time()),
        "event": event,
        "actor": actor or "unspecified",
        "purpose": purpose or "unspecified",
        "policy_id": policy_id,
        "details": details or {},
    }
    audit_log_path = output_path / "audit_log.jsonl"
    with audit_log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return audit_log_path


def _fernet_from_env(env_name: str) -> tuple[Fernet, str]:
    key = os.environ.get(env_name)
    if not key:
        raise ValueError(f"environment variable {env_name} is required for fernet feature encryption")
    try:
        return Fernet(key.encode("utf-8")), _key_id(key)
    except Exception as exc:
        raise ValueError(f"environment variable {env_name} must contain a valid Fernet key") from exc


def _key_id(key: str) -> str:
    return "sha256:" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _stable_policy_id(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "policy-" + hashlib.sha256(encoded).hexdigest()[:12]


def _read_required_json(path: Path, name: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"required {name} is missing from: {path.parent}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def _read_audit_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError("audit_log.jsonl must contain JSON objects")
        events.append(payload)
    return events


def _policy_id_from_sources(
    feature_encryption: dict[str, Any],
    metadata_encryption: dict[str, Any],
    access_policy: dict[str, Any],
) -> str:
    policy_id = str(feature_encryption.get("policy_id") or "")
    metadata_policy_id = str(metadata_encryption.get("policy_id") or "")
    access_policy_id = str(access_policy.get("policy_id") or "")
    if not policy_id:
        raise ValueError("features.json encryption metadata is missing policy_id")
    if metadata_policy_id and metadata_policy_id != policy_id:
        raise ValueError("metadata.json policy_id does not match features.json")
    if access_policy_id and access_policy_id != policy_id:
        raise ValueError("access_policy.json policy_id does not match features.json")
    return policy_id
