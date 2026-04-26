#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap_repo

REPO_ROOT = bootstrap_repo(configure_runtime_cache=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate an existing Plan A study directory.")
    parser.add_argument("--study-config", default="configs/study/plan_a_next_v0_6c_near_ood_cifar100_class_holdout.yaml")
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--output-dir", default=None)
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
    from frcnet.workflows import aggregate_plan_a_study_bundle

    args = parse_args()
    progress_callback = _make_progress_printer()
    outputs = aggregate_plan_a_study_bundle(
        study_root=args.study_root,
        study_config_path=args.study_config,
        output_dir=args.output_dir,
        progress_callback=progress_callback,
    )
    print(outputs["artifact_index_path"])
    print(outputs["experiment_record_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
