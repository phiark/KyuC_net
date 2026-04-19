#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from frcnet.workflows import build_plan_a_manifest_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Plan A manifest snapshot.")
    parser.add_argument(
        "--protocol-config",
        default="configs/protocol/plan_a_v1.yaml",
        help="Path to the protocol YAML file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory override.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol_config_path = Path(args.protocol_config)
    output_dir = Path(args.output_dir) if args.output_dir is not None else None
    if output_dir is None:
        import yaml

        payload = yaml.safe_load(protocol_config_path.read_text(encoding="utf-8"))
        protocol_config = payload["protocol"]
        output_dir = Path(protocol_config["output_root"]) / protocol_config["protocol_id"]
    outputs = build_plan_a_manifest_bundle(protocol_config_path=protocol_config_path, output_dir=output_dir)
    print(outputs["manifest_path"])
    print(outputs["manifest_summary_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
