from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import yaml

from frcnet.evaluation import read_top1_proposition_records
from frcnet.workflows.plan_a import (
    _load_yaml_section,
    _write_json,
    build_plan_a_manifest_bundle,
    export_plan_a_inference_bundle,
    generate_plan_a_artifact_bundle,
    prepare_plan_a_datasets,
    train_plan_a_model,
)


@dataclass(slots=True)
class StudyRunMetric:
    study_id: str
    model_family: str
    run_id: str
    seed: int
    pair_auroc: float
    weighted_pair_auroc: float
    scalar_auroc: float
    easy_id_top1_accuracy: float
    hard_id_top1_accuracy: float
    ambiguous_candidate_hit_rate: float
    run_output_dir: str

    def to_csv_row(self) -> dict[str, str | int | float]:
        return {
            "study_id": self.study_id,
            "model_family": self.model_family,
            "run_id": self.run_id,
            "seed": self.seed,
            "pair_auroc": self.pair_auroc,
            "weighted_pair_auroc": self.weighted_pair_auroc,
            "scalar_auroc": self.scalar_auroc,
            "easy_id_top1_accuracy": self.easy_id_top1_accuracy,
            "hard_id_top1_accuracy": self.hard_id_top1_accuracy,
            "ambiguous_candidate_hit_rate": self.ambiguous_candidate_hit_rate,
            "run_output_dir": self.run_output_dir,
        }


@dataclass(slots=True)
class CheckpointPolicyMetric:
    study_id: str
    model_family: str
    run_id: str
    seed: int
    policy_name: str
    pair_auroc: float
    weighted_pair_auroc: float
    scalar_auroc: float
    easy_id_top1_accuracy: float
    hard_id_top1_accuracy: float
    ambiguous_candidate_hit_rate: float
    run_output_dir: str

    def to_csv_row(self) -> dict[str, str | int | float]:
        return {
            "study_id": self.study_id,
            "model_family": self.model_family,
            "run_id": self.run_id,
            "seed": self.seed,
            "policy_name": self.policy_name,
            "pair_auroc": self.pair_auroc,
            "weighted_pair_auroc": self.weighted_pair_auroc,
            "scalar_auroc": self.scalar_auroc,
            "easy_id_top1_accuracy": self.easy_id_top1_accuracy,
            "hard_id_top1_accuracy": self.hard_id_top1_accuracy,
            "ambiguous_candidate_hit_rate": self.ambiguous_candidate_hit_rate,
            "run_output_dir": self.run_output_dir,
        }


AGGREGATE_METRIC_NAMES = (
    "pair_auroc",
    "weighted_pair_auroc",
    "scalar_auroc",
    "easy_id_top1_accuracy",
    "hard_id_top1_accuracy",
    "ambiguous_candidate_hit_rate",
)


def _write_yaml_section(output_path: str | Path, section_name: str, payload: Mapping[str, Any]) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump({section_name: payload}, sort_keys=False), encoding="utf-8")
    return output


def _emit_progress(progress_callback: Callable[[str], None] | None, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)


def _load_json_file(input_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(input_path).read_text(encoding="utf-8"))


def _override_train_seed(train_config_path: str | Path, seed: int, output_path: str | Path) -> Path:
    train_config = _load_yaml_section(train_config_path, "train")
    training_config = dict(train_config.get("training", {}))
    training_config["seed"] = seed
    train_config["training"] = training_config
    return _write_yaml_section(output_path, "train", train_config)


def _load_existing_train_outputs(run_root: str | Path, run_id: str) -> dict[str, str] | None:
    training_root = Path(run_root) / "training"
    summary_path = training_root / "records" / "train_summary.json"
    if not summary_path.exists():
        return None

    summary_payload = _load_json_file(summary_path)
    model_family = str(summary_payload.get("model_family", "frcnet_explicit_unknown"))
    checkpoints = summary_payload.get("checkpoints", {})
    best_policy_name = str(checkpoints.get("best_policy", "theory"))
    best_checkpoint_path = checkpoints.get("best")
    last_checkpoint_path = checkpoints.get("last")
    if not best_checkpoint_path or not Path(best_checkpoint_path).exists():
        return None

    protocol_id = ""
    protocol_snapshot_path = training_root / "snapshots" / "protocol_config_snapshot.yaml"
    if protocol_snapshot_path.exists():
        protocol_id = str(_load_yaml_section(protocol_snapshot_path, "protocol").get("protocol_id", ""))

    manifest_path = training_root / "manifests" / "train_manifest_snapshot.jsonl"
    manifest_summary_path = training_root / "manifests" / "train_manifest_summary.json"
    history_path = training_root / "records" / "train_history.csv"
    validation_history_path = training_root / "records" / "validation_history.csv"
    validation_manifest_path = training_root / "manifests" / "validation_manifest_snapshot.jsonl"

    return {
        "run_id": run_id,
        "model_family": model_family,
        "protocol_id": protocol_id,
        "output_dir": str(training_root),
        "manifest_path": str(manifest_path),
        "manifest_summary_path": str(manifest_summary_path),
        "history_path": str(history_path),
        "summary_path": str(summary_path),
        "validation_history_path": "" if not validation_history_path.exists() else str(validation_history_path),
        "validation_manifest_path": "" if not validation_manifest_path.exists() else str(validation_manifest_path),
        "best_checkpoint_path": str(best_checkpoint_path),
        "best_policy_name": best_policy_name,
        "best_theory_checkpoint_path": str(checkpoints.get("best_theory", "")),
        "best_balanced_checkpoint_path": str(checkpoints.get("best_balanced", "")),
        "checkpoint_selection_summary_path": str(checkpoints.get("selection_summary_path", "")),
        "last_checkpoint_path": "" if not last_checkpoint_path else str(last_checkpoint_path),
    }


def _load_existing_inference_outputs(
    run_root: str | Path,
    run_id: str,
    subdir_name: str = "analysis",
) -> dict[str, str] | None:
    analysis_root = Path(run_root) / subdir_name
    analysis_path = analysis_root / "sample_analysis_records.csv"
    proposition_path = analysis_root / "top1_proposition_records.csv"
    analysis_summary_path = analysis_root / "analysis_summary.json"
    protocol_snapshot_path = analysis_root / "protocol_config_snapshot.yaml"
    model_snapshot_path = analysis_root / "model_config_snapshot.yaml"
    manifest_snapshot_path = analysis_root / "plan_a_manifest_snapshot.jsonl"

    required_paths = (
        analysis_path,
        proposition_path,
        analysis_summary_path,
        protocol_snapshot_path,
        model_snapshot_path,
        manifest_snapshot_path,
    )
    if not all(path.exists() for path in required_paths):
        return None

    protocol_id = str(_load_yaml_section(protocol_snapshot_path, "protocol").get("protocol_id", ""))
    analysis_summary_payload = _load_json_file(analysis_summary_path)
    return {
        "run_id": run_id,
        "model_family": str(analysis_summary_payload.get("model_family", "frcnet_explicit_unknown")),
        "protocol_id": protocol_id,
        "output_dir": str(analysis_root),
        "protocol_snapshot_path": str(protocol_snapshot_path),
        "model_snapshot_path": str(model_snapshot_path),
        "manifest_snapshot_path": str(manifest_snapshot_path),
        "analysis_path": str(analysis_path),
        "proposition_path": str(proposition_path),
        "analysis_summary_path": str(analysis_summary_path),
        "checkpoint_selection_summary_path": str(
            analysis_summary_payload.get("checkpoint_selection_summary_path", "")
        ),
    }


def _load_existing_artifact_outputs(
    run_root: str | Path,
    run_id: str,
    analysis_config_path: str | Path | None = None,
    subdir_name: str = "report",
) -> dict[str, str] | None:
    report_root = Path(run_root) / subdir_name
    analysis_config = {"matched_table_name": "matched_ambiguous_vs_ood_table.csv"}
    if analysis_config_path is not None:
        analysis_config.update(_load_yaml_section(analysis_config_path, "analysis"))

    matched_path = report_root / str(analysis_config["matched_table_name"])
    artifact_index_path = report_root / "artifact_paths.json"
    experiment_record_path = report_root / "experiment_record.md"
    protocol_snapshot_path = report_root / "protocol_config_snapshot.yaml"
    eval_snapshot_path = report_root / "eval_config_snapshot.yaml"
    analysis_summary_path = report_root / "analysis_summary.json"

    required_paths = (
        matched_path,
        artifact_index_path,
        experiment_record_path,
        protocol_snapshot_path,
        eval_snapshot_path,
        analysis_summary_path,
    )
    if not all(path.exists() for path in required_paths):
        return None

    matched_row = _single_csv_row(matched_path)
    artifact_paths = _load_json_file(artifact_index_path)
    return {
        "run_id": run_id,
        "model_family": str(matched_row.get("model_family", "frcnet_explicit_unknown")),
        "protocol_id": str(matched_row.get("protocol_id", "")),
        "output_dir": str(report_root),
        "protocol_snapshot_path": str(protocol_snapshot_path),
        "eval_snapshot_path": str(eval_snapshot_path),
        "analysis_summary_path": str(analysis_summary_path),
        "artifact_index_path": str(artifact_index_path),
        "experiment_record_path": str(experiment_record_path),
        **{str(key): str(value) for key, value in artifact_paths.items()},
    }


def _single_csv_row(input_path: str | Path) -> dict[str, str]:
    with Path(input_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        try:
            return next(reader)
        except StopIteration as exc:
            raise ValueError(f"{input_path} does not contain any rows.") from exc


def _checkpoint_path_for_policy(train_output: Mapping[str, Any], policy_name: str) -> str:
    if policy_name == str(train_output.get("best_policy_name", "theory")):
        return str(train_output["best_checkpoint_path"])
    if policy_name == "theory":
        return str(train_output.get("best_theory_checkpoint_path", ""))
    if policy_name == "balanced":
        return str(train_output.get("best_balanced_checkpoint_path", ""))
    if policy_name == "last":
        return str(train_output.get("last_checkpoint_path", ""))
    return ""


def _proposition_accuracy(proposition_path: str | Path, cohort_name: str) -> float:
    proposition_records = read_top1_proposition_records(proposition_path)
    cohort_records = [record for record in proposition_records if record.cohort_name == cohort_name]
    if not cohort_records:
        return 0.0
    correct_count = sum(int(record.is_top1_correct) for record in cohort_records)
    return correct_count / len(cohort_records)


def _collect_run_metric(study_id: str, seed: int, run_output: Mapping[str, Any]) -> StudyRunMetric:
    matched_row = _single_csv_row(run_output["report"]["matched_ambiguous_vs_ood_table"])
    proposition_path = run_output["analysis"]["proposition_path"]
    return StudyRunMetric(
        study_id=study_id,
        model_family=str(matched_row.get("model_family", run_output.get("model_family", "frcnet_explicit_unknown"))),
        run_id=str(run_output["run_id"]),
        seed=seed,
        pair_auroc=float(matched_row["pair_auroc"]),
        weighted_pair_auroc=float(matched_row["weighted_pair_auroc"]),
        scalar_auroc=float(matched_row["scalar_auroc"]),
        easy_id_top1_accuracy=_proposition_accuracy(proposition_path, "easy_id"),
        hard_id_top1_accuracy=_proposition_accuracy(proposition_path, "hard_id"),
        ambiguous_candidate_hit_rate=_proposition_accuracy(proposition_path, "ambiguous_id"),
        run_output_dir=str(run_output["output_dir"]),
    )


def _collect_policy_metric(
    study_id: str,
    seed: int,
    policy_name: str,
    run_id: str,
    run_root: str | Path,
    analysis_output: Mapping[str, Any],
    report_output: Mapping[str, Any],
    *,
    fallback_model_family: str,
) -> CheckpointPolicyMetric:
    matched_row = _single_csv_row(report_output["matched_ambiguous_vs_ood_table"])
    proposition_path = analysis_output["proposition_path"]
    return CheckpointPolicyMetric(
        study_id=study_id,
        model_family=str(matched_row.get("model_family", fallback_model_family)),
        run_id=run_id,
        seed=seed,
        policy_name=policy_name,
        pair_auroc=float(matched_row["pair_auroc"]),
        weighted_pair_auroc=float(matched_row["weighted_pair_auroc"]),
        scalar_auroc=float(matched_row["scalar_auroc"]),
        easy_id_top1_accuracy=_proposition_accuracy(proposition_path, "easy_id"),
        hard_id_top1_accuracy=_proposition_accuracy(proposition_path, "hard_id"),
        ambiguous_candidate_hit_rate=_proposition_accuracy(proposition_path, "ambiguous_id"),
        run_output_dir=str(run_root),
    )


def _write_seed_metrics(metrics: Sequence[StudyRunMetric], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(metrics[0].to_csv_row().keys()) if metrics else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for metric in metrics:
                writer.writerow(metric.to_csv_row())
    return output


def _write_metric_summary(metrics: Sequence[StudyRunMetric], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["metric_name", "mean", "std", "min", "max"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metric_name in AGGREGATE_METRIC_NAMES:
            values = [float(getattr(metric, metric_name)) for metric in metrics]
            writer.writerow(
                {
                    "metric_name": metric_name,
                    "mean": mean(values),
                    "std": 0.0 if len(values) == 1 else pstdev(values),
                    "min": min(values),
                    "max": max(values),
                }
            )
    return output


def _write_checkpoint_policy_metrics(metrics: Sequence[CheckpointPolicyMetric], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(metrics[0].to_csv_row().keys()) if metrics else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for metric in metrics:
                writer.writerow(metric.to_csv_row())
    return output


def _write_checkpoint_policy_summary(metrics: Sequence[CheckpointPolicyMetric], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    policy_names = sorted({metric.policy_name for metric in metrics})
    with output.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["policy_name", "metric_name", "mean", "std", "min", "max"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for policy_name in policy_names:
            policy_metrics = [metric for metric in metrics if metric.policy_name == policy_name]
            for metric_name in AGGREGATE_METRIC_NAMES:
                values = [float(getattr(metric, metric_name)) for metric in policy_metrics]
                writer.writerow(
                    {
                        "policy_name": policy_name,
                        "metric_name": metric_name,
                        "mean": mean(values),
                        "std": 0.0 if len(values) == 1 else pstdev(values),
                        "min": min(values),
                        "max": max(values),
                    }
                )
    return output


def _write_checkpoint_policy_gap_summary(
    metrics: Sequence[CheckpointPolicyMetric],
    output_path: str | Path,
    *,
    minuend_policy: str,
    subtrahend_policy: str,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    by_seed_policy = {(metric.seed, metric.policy_name): metric for metric in metrics}
    shared_seeds = sorted(
        {
            metric.seed
            for metric in metrics
            if (metric.seed, minuend_policy) in by_seed_policy and (metric.seed, subtrahend_policy) in by_seed_policy
        }
    )
    with output.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["seed", "metric_name", "minuend_policy", "subtrahend_policy", "delta"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for seed in shared_seeds:
            minuend_metric = by_seed_policy[(seed, minuend_policy)]
            subtrahend_metric = by_seed_policy[(seed, subtrahend_policy)]
            for metric_name in AGGREGATE_METRIC_NAMES:
                writer.writerow(
                    {
                        "seed": seed,
                        "metric_name": metric_name,
                        "minuend_policy": minuend_policy,
                        "subtrahend_policy": subtrahend_policy,
                        "delta": float(getattr(minuend_metric, metric_name))
                        - float(getattr(subtrahend_metric, metric_name)),
                    }
                )
    return output


def _write_metric_plot(
    metrics: Sequence[StudyRunMetric],
    *,
    output_path: str | Path,
    metric_names: Sequence[str],
    title: str,
    y_label: str,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    x_positions = list(range(len(metrics)))
    width = 0.22 if len(metric_names) > 1 else 0.45

    plt.figure(figsize=(9, 5))
    for index, metric_name in enumerate(metric_names):
        offset = (index - ((len(metric_names) - 1) / 2.0)) * width
        values = [float(getattr(metric, metric_name)) for metric in metrics]
        plt.bar([position + offset for position in x_positions], values, width=width, label=metric_name)

    plt.xticks(x_positions, [f"seed{metric.seed:03d}" for metric in metrics], rotation=15)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()
    return output


def aggregate_plan_a_study_bundle(
    *,
    study_root: str | Path,
    study_config_path: str | Path,
    output_dir: str | Path | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, str]:
    study_root_path = Path(study_root)
    output_root = Path(output_dir) if output_dir is not None else study_root_path / "aggregate"
    output_root.mkdir(parents=True, exist_ok=True)
    _emit_progress(progress_callback, f"[study] aggregate_start study_root={study_root_path}")

    study_config = _load_yaml_section(study_config_path, "study")
    study_paths_path = study_root_path / "study_paths.json"
    if not study_paths_path.exists():
        raise ValueError(f"Missing study_paths.json at {study_paths_path}. Run the study workflow first.")
    study_paths = json.loads(study_paths_path.read_text(encoding="utf-8"))
    run_outputs = list(study_paths.get("runs", []))
    if not run_outputs:
        raise ValueError("Study paths do not contain any completed runs.")

    seeds = [int(seed) for seed in study_paths["seeds"]]
    metrics = [
        _collect_run_metric(str(study_paths["study_id"]), seed, run_output)
        for seed, run_output in zip(seeds, run_outputs, strict=True)
    ]
    policy_metrics: list[CheckpointPolicyMetric] = []
    for seed, run_output in zip(seeds, run_outputs, strict=True):
        policy_outputs = dict(run_output.get("policy_outputs", {}))
        if not policy_outputs:
            primary_policy_name = str(run_output.get("primary_checkpoint_policy", "theory"))
            policy_outputs[primary_policy_name] = {
                "analysis": run_output["analysis"],
                "report": run_output["report"],
            }
        for policy_name, policy_output in sorted(policy_outputs.items()):
            policy_metrics.append(
                _collect_policy_metric(
                    str(study_paths["study_id"]),
                    seed,
                    str(policy_name),
                    str(run_output["run_id"]),
                    run_output["output_dir"],
                    policy_output["analysis"],
                    policy_output["report"],
                    fallback_model_family=str(run_output.get("model_family", study_paths.get("model_family", "frcnet_explicit_unknown"))),
                )
            )

    seed_metrics_path = _write_seed_metrics(metrics, output_root / "seed_metrics.csv")
    metric_summary_path = _write_metric_summary(metrics, output_root / "metric_summary.csv")
    checkpoint_policy_metrics_path = _write_checkpoint_policy_metrics(
        policy_metrics,
        output_root / "checkpoint_policy_metrics.csv",
    )
    checkpoint_policy_summary_path = _write_checkpoint_policy_summary(
        policy_metrics,
        output_root / "checkpoint_policy_summary.csv",
    )
    checkpoint_policy_gap_summary_path = _write_checkpoint_policy_gap_summary(
        policy_metrics,
        output_root / "checkpoint_policy_gap_summary.csv",
        minuend_policy="balanced",
        subtrahend_policy="theory",
    )

    ranking_metric = str(study_config.get("report_policy", {}).get("ranking_metric", "pair_auroc"))
    ranked_metrics = sorted(metrics, key=lambda metric: float(getattr(metric, ranking_metric)), reverse=True)
    best_metric = ranked_metrics[0]
    worst_metric = ranked_metrics[-1]
    median_metric = ranked_metrics[len(ranked_metrics) // 2]
    rankings_path = _write_json(
        {
            "ranking_metric": ranking_metric,
            "best_run_id": best_metric.run_id,
            "worst_run_id": worst_metric.run_id,
            "median_run_id": median_metric.run_id,
            "best_seed": best_metric.seed,
            "worst_seed": worst_metric.seed,
            "median_seed": median_metric.seed,
        },
        output_root / "seed_rankings.json",
    )

    auroc_plot_path = _write_metric_plot(
        metrics,
        output_path=output_root / "auroc_by_seed.png",
        metric_names=("pair_auroc", "weighted_pair_auroc", "scalar_auroc"),
        title="Matched AUROC By Seed",
        y_label="auroc",
    )
    proposition_plot_path = _write_metric_plot(
        metrics,
        output_path=output_root / "proposition_accuracy_by_seed.png",
        metric_names=("easy_id_top1_accuracy", "hard_id_top1_accuracy", "ambiguous_candidate_hit_rate"),
        title="Top-1 Proposition Accuracy By Seed",
        y_label="accuracy",
    )

    artifact_paths = {
        "seed_metrics": str(seed_metrics_path),
        "metric_summary": str(metric_summary_path),
        "checkpoint_policy_metrics": str(checkpoint_policy_metrics_path),
        "checkpoint_policy_summary": str(checkpoint_policy_summary_path),
        "checkpoint_policy_gap_summary": str(checkpoint_policy_gap_summary_path),
        "seed_rankings": str(rankings_path),
        "auroc_by_seed": str(auroc_plot_path),
        "proposition_accuracy_by_seed": str(proposition_plot_path),
    }
    artifact_index_path = _write_json(artifact_paths, output_root / "artifact_paths.json")

    lines = [
        f"# Study Record: {study_paths['study_id']}",
        "",
        f"- study_id: `{study_paths['study_id']}`",
        f"- model_family: `{study_paths.get('model_family', 'frcnet_explicit_unknown')}`",
        f"- study_config_path: `{study_config_path}`",
        f"- shared_eval_manifest: `{study_paths['shared_eval_manifest_path']}`",
        "",
        "## Matched Benchmark",
        "",
        f"- ranking_metric: `{ranking_metric}`",
        f"- primary_checkpoint_policy: `{study_paths.get('primary_checkpoint_policy', 'theory')}`",
        f"- best_run_id: `{best_metric.run_id}`",
        f"- worst_run_id: `{worst_metric.run_id}`",
        f"- median_run_id: `{median_metric.run_id}`",
        "",
        "## Proposition Diagnostics",
        "",
        f"- companion_checkpoint_policies: `{json.dumps(study_paths.get('companion_checkpoint_policies', []))}`",
        f"- checkpoint_policy_summary: `{checkpoint_policy_summary_path}`",
        "",
        "## Seeds",
        "",
    ]
    for metric in metrics:
        lines.append(
            f"- `{metric.run_id}` seed={metric.seed} "
            f"pair_auroc={metric.pair_auroc:.6f} easy_id_top1={metric.easy_id_top1_accuracy:.6f}"
        )
    lines.extend(["", "## Checkpoint Policies", ""])
    for policy_name in sorted({metric.policy_name for metric in policy_metrics}):
        policy_subset = [metric for metric in policy_metrics if metric.policy_name == policy_name]
        lines.append(
            f"- `{policy_name}` seeds={len(policy_subset)} "
            f"pair_mean={mean(metric.pair_auroc for metric in policy_subset):.6f} "
            f"hard_top1_mean={mean(metric.hard_id_top1_accuracy for metric in policy_subset):.6f}"
        )
    lines.extend(["", "## Aggregate Artifacts", ""])
    for artifact_name, artifact_path in sorted(artifact_paths.items()):
        lines.append(f"- {artifact_name}: `{artifact_path}`")
    experiment_record_path = output_root / "experiment_record.md"
    experiment_record_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _emit_progress(progress_callback, f"[study] aggregate_complete record={experiment_record_path}")

    return {
        "output_dir": str(output_root),
        "seed_metrics_path": str(seed_metrics_path),
        "metric_summary_path": str(metric_summary_path),
        "checkpoint_policy_metrics_path": str(checkpoint_policy_metrics_path),
        "checkpoint_policy_summary_path": str(checkpoint_policy_summary_path),
        "checkpoint_policy_gap_summary_path": str(checkpoint_policy_gap_summary_path),
        "seed_rankings_path": str(rankings_path),
        "artifact_index_path": str(artifact_index_path),
        "experiment_record_path": str(experiment_record_path),
        "auroc_plot_path": str(auroc_plot_path),
        "proposition_plot_path": str(proposition_plot_path),
    }


def run_plan_a_study_bundle(
    *,
    study_config_path: str | Path,
    output_dir: str | Path | None = None,
    download_override: bool | None = None,
    aggregate_after_run: bool = True,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, str]:
    study_config = _load_yaml_section(study_config_path, "study")
    study_id = str(study_config["study_id"])
    study_root = Path(output_dir) if output_dir is not None else Path(study_config["output_root"]) / study_id
    study_root.mkdir(parents=True, exist_ok=True)
    _emit_progress(progress_callback, f"[study] study_id={study_id} output_dir={study_root}")

    train_protocol_config = study_config["train_protocol_config"]
    analysis_protocol_config = study_config["analysis_protocol_config"]
    model_config = study_config["model_config"]
    train_config = study_config["train_config"]
    eval_config = study_config["eval_config"]
    analysis_config = study_config["analysis_config"]
    seeds = [int(seed) for seed in study_config.get("seeds", (7, 17, 27))]
    model_family = str(study_config.get("model_family", "frcnet_explicit_unknown"))
    report_policy = dict(study_config.get("report_policy", {}))
    primary_checkpoint_policy = str(report_policy.get("primary_checkpoint_policy", "theory"))
    companion_checkpoint_policies = [
        str(policy_name) for policy_name in report_policy.get("companion_checkpoint_policies", [])
    ]

    prepare_plan_a_datasets(
        [train_protocol_config, analysis_protocol_config],
        output_path=study_root / "data_preflight.json",
        download_override=download_override,
    )
    _emit_progress(progress_callback, f"[study] data_preflight={study_root / 'data_preflight.json'}")

    shared_manifest_outputs = build_plan_a_manifest_bundle(
        protocol_config_path=analysis_protocol_config,
        output_dir=study_root / "shared" / "analysis_manifest",
        manifest_filename="plan_a_manifest.jsonl",
        summary_filename="plan_a_manifest_summary.json",
    )
    _emit_progress(progress_callback, f"[study] shared_eval_manifest={shared_manifest_outputs['manifest_path']}")

    run_outputs: list[dict[str, Any]] = []
    for seed in seeds:
        run_id = f"{study_id}-seed{seed:03d}"
        run_root = study_root / "runs" / run_id
        _emit_progress(progress_callback, f"[study] seed_start run_id={run_id} seed={seed}")
        generated_train_config_path = _override_train_seed(
            train_config,
            seed,
            study_root / "shared" / "generated_configs" / f"train_seed{seed:03d}.yaml",
        )
        train_outputs = _load_existing_train_outputs(run_root, run_id)
        if train_outputs is None:
            train_outputs = train_plan_a_model(
                protocol_config_path=train_protocol_config,
                model_config_path=model_config,
                train_config_path=generated_train_config_path,
                output_dir=run_root / "training",
                run_id=run_id,
                validation_protocol_config_path=analysis_protocol_config,
                validation_manifest_path=shared_manifest_outputs["manifest_path"],
                eval_config_path=eval_config,
                progress_callback=progress_callback,
            )
        else:
            _emit_progress(
                progress_callback,
                f"[study] seed_resume_training run_id={run_id} best_checkpoint={train_outputs['best_checkpoint_path']}",
            )

        primary_checkpoint_path = _checkpoint_path_for_policy(train_outputs, primary_checkpoint_policy)
        if not primary_checkpoint_path or not Path(primary_checkpoint_path).exists():
            raise ValueError(
                f"Primary checkpoint policy `{primary_checkpoint_policy}` did not resolve to an existing checkpoint for {run_id}."
            )

        inference_outputs = _load_existing_inference_outputs(run_root, run_id, "analysis")
        if inference_outputs is None:
            inference_outputs = export_plan_a_inference_bundle(
                protocol_config_path=analysis_protocol_config,
                model_config_path=model_config,
                manifest_path=shared_manifest_outputs["manifest_path"],
                output_dir=run_root / "analysis",
                run_id=run_id,
                checkpoint_path=primary_checkpoint_path,
                checkpoint_selection_summary_path=train_outputs.get("checkpoint_selection_summary_path"),
                model_family=str(train_outputs.get("model_family", model_family)),
            )
        else:
            _emit_progress(
                progress_callback,
                f"[study] seed_resume_analysis run_id={run_id} analysis_csv={inference_outputs['analysis_path']}",
            )

        artifact_outputs = _load_existing_artifact_outputs(run_root, run_id, analysis_config, "report")
        if artifact_outputs is None:
            artifact_outputs = generate_plan_a_artifact_bundle(
                analysis_path=inference_outputs["analysis_path"],
                analysis_summary_path=inference_outputs["analysis_summary_path"],
                protocol_config_path=analysis_protocol_config,
                eval_config_path=eval_config,
                analysis_config_path=analysis_config,
                output_dir=run_root / "report",
            )
        else:
            _emit_progress(
                progress_callback,
                f"[study] seed_resume_report run_id={run_id} record={artifact_outputs['experiment_record_path']}",
            )
        policy_outputs: dict[str, dict[str, Any]] = {
            primary_checkpoint_policy: {
                "analysis": inference_outputs,
                "report": artifact_outputs,
            }
        }
        for companion_policy in companion_checkpoint_policies:
            companion_checkpoint_path = _checkpoint_path_for_policy(train_outputs, companion_policy)
            if not companion_checkpoint_path or not Path(companion_checkpoint_path).exists():
                raise ValueError(
                    f"Companion checkpoint policy `{companion_policy}` did not resolve to an existing checkpoint for {run_id}."
                )
            analysis_subdir = "analysis_theory" if companion_policy == "theory" else f"analysis_{companion_policy}"
            report_subdir = "report_theory" if companion_policy == "theory" else f"report_{companion_policy}"
            companion_inference_outputs = _load_existing_inference_outputs(run_root, run_id, analysis_subdir)
            if companion_inference_outputs is None:
                companion_inference_outputs = export_plan_a_inference_bundle(
                    protocol_config_path=analysis_protocol_config,
                    model_config_path=model_config,
                    manifest_path=shared_manifest_outputs["manifest_path"],
                    output_dir=run_root / analysis_subdir,
                    run_id=run_id,
                    checkpoint_path=companion_checkpoint_path,
                    checkpoint_selection_summary_path=train_outputs.get("checkpoint_selection_summary_path"),
                    model_family=str(train_outputs.get("model_family", model_family)),
                )
            companion_artifact_outputs = _load_existing_artifact_outputs(
                run_root,
                run_id,
                analysis_config,
                report_subdir,
            )
            if companion_artifact_outputs is None:
                companion_artifact_outputs = generate_plan_a_artifact_bundle(
                    analysis_path=companion_inference_outputs["analysis_path"],
                    analysis_summary_path=companion_inference_outputs["analysis_summary_path"],
                    protocol_config_path=analysis_protocol_config,
                    eval_config_path=eval_config,
                    analysis_config_path=analysis_config,
                    output_dir=run_root / report_subdir,
                )
            policy_outputs[companion_policy] = {
                "analysis": companion_inference_outputs,
                "report": companion_artifact_outputs,
            }
        run_outputs.append(
            {
                "run_id": run_id,
                "seed": seed,
                "model_family": str(train_outputs.get("model_family", model_family)),
                "output_dir": str(run_root),
                "shared_eval_manifest_path": shared_manifest_outputs["manifest_path"],
                "primary_checkpoint_policy": primary_checkpoint_policy,
                "companion_checkpoint_policies": companion_checkpoint_policies,
                "train": train_outputs,
                "analysis": inference_outputs,
                "report": artifact_outputs,
                "policy_outputs": policy_outputs,
            }
        )
        _emit_progress(
            progress_callback,
            f"[study] seed_complete run_id={run_id} best_checkpoint={primary_checkpoint_path}",
        )

    study_paths_path = _write_json(
        {
            "study_id": study_id,
            "model_family": model_family,
            "study_root": str(study_root),
            "study_config_path": str(study_config_path),
            "seeds": seeds,
            "shared_eval_manifest_path": shared_manifest_outputs["manifest_path"],
            "shared_eval_manifest_summary_path": shared_manifest_outputs["manifest_summary_path"],
            "primary_checkpoint_policy": primary_checkpoint_policy,
            "companion_checkpoint_policies": companion_checkpoint_policies,
            "runs": run_outputs,
        },
        study_root / "study_paths.json",
    )

    aggregate_outputs: dict[str, str] = {}
    if aggregate_after_run:
        aggregate_outputs = aggregate_plan_a_study_bundle(
            study_root=study_root,
            study_config_path=study_config_path,
            progress_callback=progress_callback,
        )

    return {
        "study_id": study_id,
        "output_dir": str(study_root),
        "study_paths_path": str(study_paths_path),
        "shared_eval_manifest_path": shared_manifest_outputs["manifest_path"],
        **aggregate_outputs,
    }
