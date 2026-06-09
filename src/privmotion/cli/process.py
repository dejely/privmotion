from __future__ import annotations

import argparse
import sys
from pathlib import Path

from privmotion.config import ProcessConfig, parse_output_modes
from privmotion.pipeline import PrivMotionPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process RGB/RGB-D/depth inputs into anonymized motion outputs.",
    )
    parser.add_argument("--input", required=True, type=Path, help="Input image, video, or frame directory.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory.")
    parser.add_argument(
        "--mode",
        default="skeleton,silhouette",
        help="Comma-separated modes: skeleton,silhouette,depth-surrogate,features.",
    )
    parser.add_argument(
        "--retention",
        default="no-raw-rgb",
        choices=("no-raw-rgb",),
        help="Retention policy. The current prototype supports only no-raw-rgb.",
    )
    parser.add_argument("--segmentation-backend", default="auto")
    parser.add_argument("--pose-backend", default="auto")
    parser.add_argument(
        "--pose-model",
        default="yolo11n-pose.pt",
        help="Pose model name/path used when --pose-backend yolo is selected.",
    )
    parser.add_argument("--tracking-backend", default="single")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--deidentification-profile",
        default="standard",
        choices=("standard", "hipaa-expert-aggregate"),
        help="De-identification profile. Use hipaa-expert-aggregate with --mode aggregate for expert-review outputs.",
    )
    parser.add_argument(
        "--feature-encryption",
        default="none",
        choices=("none", "fernet"),
        help="Feature-record encryption mode. Default keeps features plaintext.",
    )
    parser.add_argument("--access-policy", type=Path, default=None, help="Access policy JSON for encrypted features.")
    parser.add_argument("--audit-actor", default=None, help="Actor recorded in Phase 5 audit events.")
    parser.add_argument("--audit-purpose", default=None, help="Purpose recorded in Phase 5 audit events.")
    parser.add_argument(
        "--recovery-key-env",
        default="PRIVMOTION_RECOVERY_KEY",
        help="Environment variable containing a Fernet key for encrypted features.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = ProcessConfig(
            input_path=args.input,
            output_dir=args.output,
            output_modes=parse_output_modes(args.mode),
            retention_policy=args.retention,
            segmentation_backend=args.segmentation_backend,
            pose_backend=args.pose_backend,
            pose_model=args.pose_model,
            tracking_backend=args.tracking_backend,
            max_frames=args.max_frames,
            deidentification_profile=args.deidentification_profile,
            feature_encryption=args.feature_encryption,
            access_policy_path=args.access_policy,
            audit_actor=args.audit_actor,
            audit_purpose=args.audit_purpose,
            recovery_key_env=args.recovery_key_env,
        )
        result = PrivMotionPipeline(config).run()
    except Exception as exc:
        print(f"privmotion-process: {exc}", file=sys.stderr)
        return 2

    print(f"processed_frames={result.processed_frames}")
    print(f"skipped_frames={result.skipped_frames}")
    print(f"output_dir={result.output_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
