from __future__ import annotations

import argparse
import sys
from pathlib import Path

from privmotion.recovery import inspect_recovery_policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect Phase 5 encrypted feature policy state without decrypting records.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Processed privmotion output directory.")
    parser.add_argument("--audit-actor", default=None, help="Actor recorded in the inspection audit event.")
    parser.add_argument("--audit-purpose", default=None, help="Purpose recorded in the inspection audit event.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = inspect_recovery_policy(
            args.output,
            audit_actor=args.audit_actor,
            audit_purpose=args.audit_purpose,
        )
    except Exception as exc:
        print(f"privmotion-recovery-inspect: {exc}", file=sys.stderr)
        return 2

    print(f"output_dir={result.output_dir}")
    print(f"encrypted={str(result.encrypted).lower()}")
    print(f"policy_id={result.policy_id}")
    print(f"encrypted_record_count={result.encrypted_record_count}")
    print(f"audit_event_count={result.audit_event_count}")
    print("decryption_performed=false")
    print("raw_rgb_recovery_supported=false")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
