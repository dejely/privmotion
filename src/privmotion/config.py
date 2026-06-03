from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


VALID_OUTPUT_MODES = frozenset({"skeleton", "silhouette", "depth-surrogate", "features"})
DEFAULT_OUTPUT_MODES = ("skeleton", "silhouette")


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_path", Path(self.input_path))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "output_modes", parse_output_modes(self.output_modes))
        if self.retention_policy != "no-raw-rgb":
            raise ValueError("Phase 2 supports only the no-raw-rgb retention policy")
        normalized_pose_backend = self.pose_backend.lower()
        if normalized_pose_backend not in {"auto", "prototype", "yolo"}:
            raise ValueError("pose_backend must be one of: auto, prototype, yolo")
        object.__setattr__(self, "pose_backend", normalized_pose_backend)
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames must be positive when provided")

    def to_json(self) -> dict[str, object]:
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
        }
