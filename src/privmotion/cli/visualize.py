from __future__ import annotations

import argparse
import sys
from pathlib import Path

from privmotion.visualization import visualize_output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create an anonymized GIF/MP4/PNG preview from privmotion outputs.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Processed privmotion output directory.")
    parser.add_argument("--visualization", required=True, type=Path, help="GIF or MP4 visualization path.")
    parser.add_argument("--frames-dir", type=Path, default=None, help="Optional directory for PNG preview frames.")
    parser.add_argument("--fps", type=int, default=4, help="GIF frames per second.")
    parser.add_argument("--size", default="640x360", help="Canvas size as WIDTHxHEIGHT.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = visualize_output_dir(
            args.output,
            args.visualization,
            frames_dir=args.frames_dir,
            fps=args.fps,
            size=parse_size(args.size),
        )
    except Exception as exc:
        print(f"privmotion-visualize: {exc}", file=sys.stderr)
        return 2

    print(f"output_dir={result.output_dir}")
    print(f"visualization={result.visualization_path}")
    print(f"frame_count={result.frame_count}")
    print(f"raw_rgb_used={str(result.raw_rgb_used).lower()}")
    if result.frames_dir is not None:
        print(f"frames_dir={result.frames_dir}")
    return 0


def parse_size(value: str) -> tuple[int, int]:
    parts = value.lower().split("x", maxsplit=1)
    if len(parts) != 2:
        raise ValueError("size must use WIDTHxHEIGHT format")
    width, height = (int(part) for part in parts)
    if width <= 0 or height <= 0:
        raise ValueError("size dimensions must be positive")
    return width, height


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
