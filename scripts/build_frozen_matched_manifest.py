#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from frcnet.evaluation import (
    build_frozen_matched_manifest,
    read_reference_score_records,
    read_sample_analysis_records,
    write_matched_manifest,
    write_matched_manifest_bin_diagnostics,
)


def _load_yaml_section(path: str | Path, section_name: str) -> dict[str, Any]:
    return dict(yaml.safe_load(Path(path).read_text(encoding="utf-8"))[section_name])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a frozen matched manifest from external reference scores.")
    parser.add_argument("--analysis-path", required=True)
    parser.add_argument("--reference-scores-path", required=True)
    parser.add_argument("--eval-config", default="configs/eval/plan_a_next_v0_4_strict_freeze.yaml")
    parser.add_argument("--benchmark-name", default="")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--diagnostics-path", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    eval_config = _load_yaml_section(args.eval_config, "eval")
    benchmark_config = {}
    if args.benchmark_name:
        for candidate in eval_config.get("benchmark_slices", []):
            if str(candidate.get("benchmark_name", "")) == args.benchmark_name:
                benchmark_config = dict(candidate)
                break
        if not benchmark_config:
            raise ValueError(f"benchmark_slices does not include benchmark_name={args.benchmark_name}")
    reference_config = dict(eval_config.get("reference", {}))
    records, diagnostics = build_frozen_matched_manifest(
        read_sample_analysis_records(args.analysis_path),
        read_reference_score_records(args.reference_scores_path),
        positive_cohort=str(benchmark_config.get("positive_cohort", eval_config.get("positive_cohort", "ambiguous_id"))),
        negative_cohort=str(benchmark_config.get("negative_cohort", eval_config.get("negative_cohort", "ood"))),
        positive_source_dataset_name=str(benchmark_config.get("positive_source_dataset_name", "")),
        negative_source_dataset_name=str(benchmark_config.get("negative_source_dataset_name", "")),
        positive_source_role=str(benchmark_config.get("positive_source_role", "")),
        negative_source_role=str(benchmark_config.get("negative_source_role", "")),
        reference_score_name=str(reference_config.get("score_name", "softmax_entropy")),
        num_bins=int(reference_config.get("num_bins", 10)),
        test_size=float(eval_config.get("test_size", 0.3)),
        random_state=int(eval_config.get("random_state", 7)),
    )
    manifest_path = write_matched_manifest(records, args.output_path)
    diagnostics_path = write_matched_manifest_bin_diagnostics(diagnostics, args.diagnostics_path)
    print(manifest_path)
    print(diagnostics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
