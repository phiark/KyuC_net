#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from frcnet.workflows import generate_plan_a_artifact_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Plan A paper-facing artifacts.")
    parser.add_argument("--analysis-path", required=True)
    parser.add_argument("--protocol-config", default="configs/protocol/plan_a_v1.yaml")
    parser.add_argument("--analysis-config", default="configs/analysis/plan_a_artifacts.yaml")
    parser.add_argument("--eval-config", default="configs/eval/plan_a_matched_ambiguous_vs_ood.yaml")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outputs = generate_plan_a_artifact_bundle(
        analysis_path=args.analysis_path,
        protocol_config_path=args.protocol_config,
        eval_config_path=args.eval_config,
        analysis_config_path=args.analysis_config,
        output_dir=args.output_dir,
    )
    print(outputs["experiment_record_path"])
    print(outputs["artifact_index_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
