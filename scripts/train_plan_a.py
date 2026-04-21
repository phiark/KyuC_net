#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
RUNTIME_CACHE_ROOT = REPO_ROOT / ".cache" / "runtime"
RUNTIME_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FRCNet on a Plan A manifest-backed protocol.")
    parser.add_argument("--protocol-config", default="configs/protocol/plan_a_v1_train.yaml")
    parser.add_argument("--model-config", default="configs/model/frcnet_resnet18_base.yaml")
    parser.add_argument("--train-config", default="configs/train/plan_a_train_base.yaml")
    parser.add_argument("--validation-protocol-config", default=None)
    parser.add_argument("--validation-manifest-path", default=None)
    parser.add_argument("--eval-config", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def _default_output_dir(train_config_path: str | Path, run_id: str) -> Path:
    train_config = yaml.safe_load(Path(train_config_path).read_text(encoding="utf-8"))["train"]
    return Path(train_config["output_root"]) / run_id


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
    from frcnet.workflows import timestamp_run_id, train_plan_a_model

    args = parse_args()
    run_id = args.run_id or timestamp_run_id()
    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(args.train_config, run_id)
    progress_callback = _make_progress_printer()
    outputs = train_plan_a_model(
        protocol_config_path=args.protocol_config,
        model_config_path=args.model_config,
        train_config_path=args.train_config,
        output_dir=output_dir,
        run_id=run_id,
        manifest_path=args.manifest_path,
        checkpoint_path=args.checkpoint_path,
        validation_protocol_config_path=args.validation_protocol_config,
        validation_manifest_path=args.validation_manifest_path,
        eval_config_path=args.eval_config,
        progress_callback=progress_callback,
    )
    print(outputs["summary_path"])
    print(outputs["best_checkpoint_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
