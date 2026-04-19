#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from frcnet.workflows import prepare_plan_a_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or verify Plan A source datasets.")
    parser.add_argument(
        "--protocol-config",
        action="append",
        dest="protocol_configs",
        default=[],
        help="Protocol YAML path. Repeat to check multiple protocols.",
    )
    parser.add_argument(
        "--output-path",
        default="artifacts/reports/generated/data_preflight.json",
        help="Where to write the dataset availability report.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Allow torchvision to download missing CIFAR10/SVHN data.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol_configs = args.protocol_configs or [
        "configs/protocol/plan_a_v1_train.yaml",
        "configs/protocol/plan_a_v1.yaml",
    ]
    report = prepare_plan_a_datasets(
        protocol_configs,
        output_path=Path(args.output_path),
        download_override=True if args.download else None,
    )
    print(Path(args.output_path))
    print(len(report["datasets"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
