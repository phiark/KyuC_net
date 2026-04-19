#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from frcnet.workflows import export_plan_a_inference_bundle, timestamp_run_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference export for Plan A.")
    parser.add_argument("--protocol-config", default="configs/protocol/plan_a_v1.yaml")
    parser.add_argument("--model-config", default="configs/model/frcnet_resnet18_base.yaml")
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    run_id = args.run_id or timestamp_run_id()
    output_root = Path(args.output_dir) if args.output_dir else Path("artifacts/reports/generated") / run_id
    outputs = export_plan_a_inference_bundle(
        protocol_config_path=args.protocol_config,
        model_config_path=args.model_config,
        manifest_path=args.manifest_path,
        output_dir=output_root,
        run_id=run_id,
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
    )
    print(outputs["analysis_path"])
    print(outputs["proposition_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
