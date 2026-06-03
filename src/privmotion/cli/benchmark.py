from __future__ import annotations

import argparse
import sys
from pathlib import Path

from privmotion.benchmark import benchmark_output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark privmotion output utility/privacy proxy metrics.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Processed privmotion output directory.")
    parser.add_argument("--report", required=True, type=Path, help="Benchmark JSON report path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        report = benchmark_output_dir(args.output, report_path=args.report)
    except Exception as exc:
        print(f"privmotion-benchmark: {exc}", file=sys.stderr)
        return 2

    print(f"output_dir={report.output_dir}")
    print(f"report={args.report}")
    print(f"processed_frame_count={report.utility['processed_frame_count']}")
    print(f"raw_rgb_retention_passed={str(report.privacy['raw_rgb_retention_passed']).lower()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

