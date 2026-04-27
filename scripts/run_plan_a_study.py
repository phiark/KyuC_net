#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap_repo, make_progress_printer

REPO_ROOT = bootstrap_repo(configure_runtime_cache=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Plan A multi-seed study workflow.")
    parser.add_argument("--study-config", default="configs/study/plan_a_next_v0_6c_near_ood_cifar100_class_holdout.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--download",
        "--allow-download",
        action="store_true",
        help="Allow torchvision to download missing CIFAR10/SVHN data before the study starts.",
    )
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Run all seed-specific experiments but skip aggregate report generation.",
    )
    return parser.parse_args()


def main() -> int:
    from frcnet.workflows import run_plan_a_study_bundle

    args = parse_args()
    progress_callback = make_progress_printer()
    outputs = run_plan_a_study_bundle(
        study_config_path=args.study_config,
        output_dir=args.output_dir,
        download_override=True if args.download else None,
        aggregate_after_run=not args.skip_aggregate,
        progress_callback=progress_callback,
    )
    print(outputs["study_paths_path"])
    if "experiment_record_path" in outputs:
        print(outputs["experiment_record_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
