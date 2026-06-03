from __future__ import annotations

import argparse
import sys
from pathlib import Path

from privmotion.validation import validate_output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate privmotion output retention policy.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output directory to validate.")
    parser.add_argument(
        "--policy",
        default="no-raw-rgb",
        choices=("no-raw-rgb",),
        help="Retention policy to validate.",
    )
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON report path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = validate_output_dir(args.output, policy=args.policy, report_path=args.report)
    except Exception as exc:
        print(f"privmotion-validate: {exc}", file=sys.stderr)
        return 2

    print(f"policy={result.policy}")
    print(f"passed={str(result.passed).lower()}")
    print(f"violations={len(result.violations)}")
    for violation in result.violations:
        print(f"violation={violation}")
    return 0 if result.passed else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

