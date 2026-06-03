from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class RetentionValidationResult:
    output_dir: Path
    policy: str
    passed: bool
    violations: tuple[str, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "output_dir": str(self.output_dir),
            "policy": self.policy,
            "passed": self.passed,
            "violations": list(self.violations),
        }


def validate_output_dir(
    output_dir: Path,
    policy: str = "no-raw-rgb",
    report_path: Path | None = None,
) -> RetentionValidationResult:
    path = Path(output_dir)
    if policy != "no-raw-rgb":
        raise ValueError("Phase 2 supports only the no-raw-rgb retention policy")
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

    result = RetentionValidationResult(
        output_dir=path,
        policy=policy,
        passed=not violations,
        violations=tuple(sorted(violations)),
    )
    if report_path is not None:
        write_json(Path(report_path), result.to_json())
    return result

