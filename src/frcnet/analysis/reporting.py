from __future__ import annotations

from pathlib import Path
import json
from typing import Iterable, Mapping

from frcnet.evaluation.matched_benchmark import MatchedBenchmarkSummary


def _format_optional_path(path_value: str | None) -> str:
    return "<missing>" if path_value in {None, ""} else str(path_value)


def _format_string_list(values: Iterable[str]) -> str:
    return json.dumps(list(values), ensure_ascii=False)


def write_experiment_record(
    output_path: str | Path,
    run_id: str,
    protocol_id: str,
    config_snapshot_paths: dict[str, str],
    manifest_snapshot_path: str,
    analysis_record_path: str,
    proposition_record_path: str,
    artifact_paths: dict[str, str],
    matched_summary: MatchedBenchmarkSummary,
    *,
    checkpoint_path: str | None = None,
    analysis_summary_path: str | None = None,
    sidecar_resolution_mode: str | None = None,
    integrity_overrides: Iterable[str] = (),
    source_run_ids: Iterable[str] = (),
    source_protocol_ids: Iterable[str] = (),
    resolved_eval_config: Mapping[str, str | int | float] | None = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Experiment Record: {run_id}",
        "",
        f"- run_id: `{run_id}`",
        f"- protocol_id: `{protocol_id}`",
        f"- checkpoint_path: `{_format_optional_path(checkpoint_path)}`",
        f"- analysis_summary_path: `{_format_optional_path(analysis_summary_path)}`",
        f"- sidecar_resolution_mode: `{_format_optional_path(sidecar_resolution_mode)}`",
        f"- integrity_overrides: `{_format_string_list(integrity_overrides)}`",
        f"- source_run_ids: `{_format_string_list(source_run_ids)}`",
        f"- source_protocol_ids: `{_format_string_list(source_protocol_ids)}`",
        "",
        "## Snapshots",
        "",
        f"- manifest_snapshot: `{manifest_snapshot_path}`",
    ]
    for snapshot_name, snapshot_path in sorted(config_snapshot_paths.items()):
        lines.append(f"- {snapshot_name}: `{snapshot_path}`")
    lines.extend(
        [
            "",
            "## Records",
            "",
            f"- sample_analysis_record: `{analysis_record_path}`",
            f"- top1_proposition_record: `{proposition_record_path}`",
            "",
            "## Matched Benchmark",
            "",
            f"- pair_auroc: `{matched_summary.pair_auroc:.6f}`",
            f"- scalar_auroc: `{matched_summary.scalar_auroc:.6f}`",
            f"- matched_count_per_class: `{matched_summary.matched_count_per_class}`",
            f"- positive_cohort: `{matched_summary.positive_cohort}`",
            f"- negative_cohort: `{matched_summary.negative_cohort}`",
            f"- pair_name: `{matched_summary.pair_name}`",
            f"- scalar_name: `{matched_summary.scalar_name}`",
            f"- test_size: `{matched_summary.test_size}`",
            f"- random_state: `{matched_summary.random_state}`",
            "",
        ]
    )
    if resolved_eval_config:
        lines.extend(["## Resolved Eval Config", ""])
        for config_name, config_value in sorted(resolved_eval_config.items()):
            lines.append(f"- {config_name}: `{config_value}`")
        lines.append("")
    lines.extend(["## Artifacts", ""])
    for artifact_name, artifact_path in sorted(artifact_paths.items()):
        lines.append(f"- {artifact_name}: `{artifact_path}`")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
