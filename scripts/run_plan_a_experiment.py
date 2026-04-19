#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from frcnet.workflows import timestamp_run_id, write_plan_a_experiment_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full Plan A experiment chain.")
    parser.add_argument("--train-protocol-config", default="configs/protocol/plan_a_v1_train.yaml")
    parser.add_argument("--analysis-protocol-config", default="configs/protocol/plan_a_v1.yaml")
    parser.add_argument("--model-config", default="configs/model/frcnet_resnet18_base.yaml")
    parser.add_argument("--train-config", default="configs/train/plan_a_train_base.yaml")
    parser.add_argument("--analysis-config", default="configs/analysis/plan_a_artifacts.yaml")
    parser.add_argument("--eval-config", default="configs/eval/plan_a_matched_ambiguous_vs_ood.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Allow torchvision to download missing CIFAR10/SVHN data before the run starts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or timestamp_run_id()
    output_dir = Path(args.output_dir) if args.output_dir else Path("artifacts/experiments") / run_id
    outputs = write_plan_a_experiment_bundle(
        train_protocol_config_path=args.train_protocol_config,
        analysis_protocol_config_path=args.analysis_protocol_config,
        model_config_path=args.model_config,
        train_config_path=args.train_config,
        eval_config_path=args.eval_config,
        analysis_config_path=args.analysis_config,
        output_dir=output_dir,
        run_id=run_id,
        download_override=True if args.download else None,
    )
    print(outputs["bundle_path"])
    print(outputs["experiment_record_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
