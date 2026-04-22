#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
RUNTIME_CACHE_ROOT = REPO_ROOT / ".cache" / "runtime"
RUNTIME_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the v0.3debug R2 Plan A multi-seed study workflow.")
    parser.add_argument("--study-config", default="configs/study/plan_a_v0_3debug_r2_study.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Allow torchvision to download missing CIFAR10/SVHN data before the study starts.",
    )
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Run all seed-specific experiments but skip aggregate report generation.",
    )
    return parser.parse_args()


def _make_progress_printer():
    state = {"batch_active": False}

    def _print_progress(message: str) -> None:
        if message.startswith("[train-batch] "):
            payload = message.removeprefix("[train-batch] ")
            print(f"\r{payload}", end="", flush=True)
            state["batch_active"] = True
            return
        if state["batch_active"]:
            print("", flush=True)
            state["batch_active"] = False
        print(message, flush=True)

    return _print_progress


def main() -> int:
    from frcnet.workflows import run_plan_a_study_bundle

    args = parse_args()
    progress_callback = _make_progress_printer()
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
