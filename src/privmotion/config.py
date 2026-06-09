from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from cryptography.fernet import Fernet


VALID_OUTPUT_MODES = frozenset({"skeleton", "silhouette", "depth-surrogate", "features", "aggregate"})
DEFAULT_OUTPUT_MODES = ("skeleton", "silhouette")
VALID_DEIDENTIFICATION_PROFILES = frozenset({"standard", "hipaa-expert-aggregate"})


def parse_output_modes(value: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_modes = [part.strip() for part in value.split(",")]
    else:
        raw_modes = [str(part).strip() for part in value]

    modes: list[str] = []
    for mode in raw_modes:
        normalized = mode.replace("_", "-").lower()
        if not normalized:
            continue
        if normalized not in VALID_OUTPUT_MODES:
            valid = ", ".join(sorted(VALID_OUTPUT_MODES))
            raise ValueError(f"unsupported output mode {mode!r}; expected one of: {valid}")
        if normalized not in modes:
            modes.append(normalized)

    if not modes:
        raise ValueError("at least one output mode is required")
    return tuple(modes)


@dataclass(frozen=True)
class ProcessConfig:
    input_path: Path
    output_dir: Path
    output_modes: tuple[str, ...] = DEFAULT_OUTPUT_MODES
    retention_policy: str = "no-raw-rgb"
    segmentation_backend: str = "auto"
    pose_backend: str = "auto"
    pose_model: str = "yolo11n-pose.pt"
    tracking_backend: str = "single"
    max_frames: int | None = None
    deidentification_profile: str = "standard"
    feature_encryption: str = "none"
    access_policy_path: Path | None = None
    audit_actor: str | None = None
    audit_purpose: str | None = None
    recovery_key_env: str = "PRIVMOTION_RECOVERY_KEY"

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_path", Path(self.input_path))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "output_modes", parse_output_modes(self.output_modes))
        if self.retention_policy != "no-raw-rgb":
            raise ValueError("privmotion currently supports only the no-raw-rgb retention policy")
        normalized_pose_backend = self.pose_backend.lower()
        if normalized_pose_backend not in {"auto", "prototype", "yolo"}:
            raise ValueError("pose_backend must be one of: auto, prototype, yolo")
        object.__setattr__(self, "pose_backend", normalized_pose_backend)
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames must be positive when provided")
        normalized_profile = self.deidentification_profile.lower()
        if normalized_profile not in VALID_DEIDENTIFICATION_PROFILES:
            valid = ", ".join(sorted(VALID_DEIDENTIFICATION_PROFILES))
            raise ValueError(f"deidentification_profile must be one of: {valid}")
        object.__setattr__(self, "deidentification_profile", normalized_profile)
        if normalized_profile == "hipaa-expert-aggregate" and self.output_modes != ("aggregate",):
            raise ValueError("hipaa-expert-aggregate requires exactly the aggregate output mode")
        if normalized_profile == "standard" and "aggregate" in self.output_modes:
            raise ValueError("aggregate output mode requires deidentification_profile=hipaa-expert-aggregate")
        normalized_feature_encryption = self.feature_encryption.lower()
        if normalized_feature_encryption not in {"none", "fernet"}:
            raise ValueError("feature_encryption must be one of: none, fernet")
        object.__setattr__(self, "feature_encryption", normalized_feature_encryption)
        if self.access_policy_path is not None:
            object.__setattr__(self, "access_policy_path", Path(self.access_policy_path))
        object.__setattr__(
            self,
            "audit_actor",
            str(self.audit_actor).strip() if self.audit_actor is not None else None,
        )
        object.__setattr__(
            self,
            "audit_purpose",
            str(self.audit_purpose).strip() if self.audit_purpose is not None else None,
        )
        object.__setattr__(self, "recovery_key_env", str(self.recovery_key_env).strip())
        if normalized_feature_encryption == "fernet":
            self._validate_fernet_controls()

    def _validate_fernet_controls(self) -> None:
        if "features" not in self.output_modes:
            raise ValueError("fernet feature encryption requires the features output mode")
        if self.access_policy_path is None:
            raise ValueError("fernet feature encryption requires an access policy JSON file")
        if not self.access_policy_path.exists():
            raise FileNotFoundError(f"access policy does not exist: {self.access_policy_path}")
        try:
            payload = json.loads(self.access_policy_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"access policy must be valid JSON: {self.access_policy_path}") from exc
        if not isinstance(payload, dict):
            raise ValueError("access policy must be a JSON object")
        if not self.audit_purpose:
            raise ValueError("fernet feature encryption requires an audit purpose")
        if not self.recovery_key_env:
            raise ValueError("recovery_key_env must not be empty")
        key = os.environ.get(self.recovery_key_env)
        if not key:
            raise ValueError(f"environment variable {self.recovery_key_env} is required for fernet feature encryption")
        try:
            Fernet(key.encode("utf-8"))
        except Exception as exc:
            raise ValueError(f"environment variable {self.recovery_key_env} must contain a valid Fernet key") from exc

    def to_json(self, redacted: bool = False) -> dict[str, object]:
        if redacted:
            return {
                "output_modes": list(self.output_modes),
                "retention_policy": self.retention_policy,
                "segmentation_backend": self.segmentation_backend,
                "pose_backend": self.pose_backend,
                "tracking_backend": self.tracking_backend,
                "max_frames": self.max_frames,
                "deidentification_profile": self.deidentification_profile,
                "feature_encryption": self.feature_encryption,
            }
        return {
            "input_path": str(self.input_path),
            "output_dir": str(self.output_dir),
            "output_modes": list(self.output_modes),
            "retention_policy": self.retention_policy,
            "segmentation_backend": self.segmentation_backend,
            "pose_backend": self.pose_backend,
            "pose_model": self.pose_model,
            "tracking_backend": self.tracking_backend,
            "max_frames": self.max_frames,
            "deidentification_profile": self.deidentification_profile,
            "feature_encryption": self.feature_encryption,
            "access_policy_path": str(self.access_policy_path) if self.access_policy_path is not None else None,
            "audit_actor": self.audit_actor,
            "audit_purpose": self.audit_purpose,
            "recovery_key_env": self.recovery_key_env,
        }
