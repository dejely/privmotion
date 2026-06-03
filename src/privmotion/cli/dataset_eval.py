from __future__ import annotations

import argparse
import sys
from pathlib import Path

from privmotion.cli.visualize import parse_size
from privmotion.config import parse_output_modes
from privmotion.dataset_eval import evaluate_dataset_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a local dataset manifest with privmotion processing and benchmarks.",
    )
    parser.add_argument("--manifest", required=True, type=Path, help="Dataset manifest JSON path.")
    parser.add_argument("--output", required=True, type=Path, help="Dataset evaluation output directory.")
    parser.add_argument(
        "--mode",
        default="skeleton,silhouette,depth-surrogate,features",
        help="Comma-separated processing modes.",
    )
    parser.add_argument("--pose-backend", default="auto", choices=("auto", "prototype", "yolo"))
    parser.add_argument("--pose-model", default="yolo11n-pose.pt")
    parser.add_argument("--visualize", action="store_true", help="Create per-sample previews.")
    parser.add_argument("--visualization-ext", default=".gif", choices=(".gif", "gif", ".mp4", "mp4"))
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--size", default="640x360")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        report = evaluate_dataset_manifest(
            args.manifest,
            args.output,
            output_modes=parse_output_modes(args.mode),
            pose_backend=args.pose_backend,
            pose_model=args.pose_model,
            visualize=args.visualize,
            visualization_ext=args.visualization_ext,
            fps=args.fps,
            size=parse_size(args.size),
        )
    except Exception as exc:
        print(f"privmotion-dataset-eval: {exc}", file=sys.stderr)
        return 2

    print(f"manifest={report.manifest_path}")
    print(f"output_dir={report.output_dir}")
    print(f"sample_count={report.sample_count}")
    print(f"processed_frame_count={report.processed_frame_count}")
    print(f"dataset_report={report.output_dir / 'dataset_report.json'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
