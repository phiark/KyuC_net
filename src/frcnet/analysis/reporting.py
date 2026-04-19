from __future__ import annotations

from pathlib import Path

from frcnet.evaluation.matched_benchmark import MatchedBenchmarkSummary


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
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Experiment Record: {run_id}",
        "",
        f"- run_id: `{run_id}`",
        f"- protocol_id: `{protocol_id}`",
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
            "",
            "## Artifacts",
            "",
        ]
    )
    for artifact_name, artifact_path in sorted(artifact_paths.items()):
        lines.append(f"- {artifact_name}: `{artifact_path}`")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output

