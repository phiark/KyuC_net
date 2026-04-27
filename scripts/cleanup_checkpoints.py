#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap_repo

REPO_ROOT = bootstrap_repo(configure_runtime_cache=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run or execute archive checkpoint cleanup for generated outputs."
    )
    parser.add_argument(
        "--root",
        action="append",
        default=None,
        help="Generated-output root to scan. Defaults to artifacts and records/experiments.",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Optional CSV manifest path for the complete cleanup plan.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Delete non-best/non-last checkpoint candidates. Omit for dry-run.",
    )
    return parser.parse_args()


def _format_gib(size_bytes: int) -> str:
    return f"{size_bytes / 1024**3:.2f} GiB"


def main() -> int:
    from frcnet.maintenance import (
        DEFAULT_CHECKPOINT_CLEANUP_ROOTS,
        build_checkpoint_cleanup_plan,
        delete_checkpoint_candidates,
        write_checkpoint_manifest,
    )

    args = parse_args()
    roots = tuple(Path(root) for root in args.root) if args.root else DEFAULT_CHECKPOINT_CLEANUP_ROOTS
    plan = build_checkpoint_cleanup_plan(roots)
    print(f"scan_roots={','.join(str(root) for root in roots)}")
    print(f"model_files={len(plan.records)} total={_format_gib(plan.total_size_bytes)}")
    print(
        f"retain_files={len(plan.retained_records)} "
        f"retain_size={_format_gib(plan.retained_size_bytes)}"
    )
    print(
        f"delete_candidates={len(plan.delete_records)} "
        f"delete_size={_format_gib(plan.delete_size_bytes)}"
    )
    if args.manifest_path:
        manifest_path = write_checkpoint_manifest(plan.records, args.manifest_path)
        print(f"manifest={manifest_path}")
    if not args.execute:
        print("mode=dry-run")
        return 0
    deleted_count, deleted_size = delete_checkpoint_candidates(plan)
    print(f"mode=execute deleted_files={deleted_count} deleted_size={_format_gib(deleted_size)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
