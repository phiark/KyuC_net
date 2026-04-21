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
    run_id: str
    seed: int
    pair_auroc: float
    weighted_pair_auroc: float
    scalar_auroc: float
    tau_scalar_auroc: float
    easy_id_top1_accuracy: float
    hard_id_top1_accuracy: float
    ambiguous_candidate_hit_rate: float
    run_output_dir: str

    def to_csv_row(self) -> dict[str, str | int | float]:
        return {
            "study_id": self.study_id,
            "run_id": self.run_id,
            "seed": self.seed,
            "pair_auroc": self.pair_auroc,
            "weighted_pair_auroc": self.weighted_pair_auroc,
            "scalar_auroc": self.scalar_auroc,
            "tau_scalar_auroc": self.tau_scalar_auroc,
            "easy_id_top1_accuracy": self.easy_id_top1_accuracy,
            "hard_id_top1_accuracy": self.hard_id_top1_accuracy,
            "ambiguous_candidate_hit_rate": self.ambiguous_candidate_hit_rate,
            "run_output_dir": self.run_output_dir,
        }


AGGREGATE_METRIC_NAMES = (
    "pair_auroc",
    "weighted_pair_auroc",
    "scalar_auroc",
    "tau_scalar_auroc",
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


def _override_train_seed(train_config_path: str | Path, seed: int, output_path: str | Path) -> Path:
    train_config = _load_yaml_section(train_config_path, "train")
    training_config = dict(train_config.get("training", {}))
    training_config["seed"] = seed
    train_config["training"] = training_config
    return _write_yaml_section(output_path, "train", train_config)


def _single_csv_row(input_path: str | Path) -> dict[str, str]:
    with Path(input_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        try:
            return next(reader)
        except StopIteration as exc:
            raise ValueError(f"{input_path} does not contain any rows.") from exc


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
        run_id=str(run_output["run_id"]),
        seed=seed,
        pair_auroc=float(matched_row["pair_auroc"]),
        weighted_pair_auroc=float(matched_row["weighted_pair_auroc"]),
        scalar_auroc=float(matched_row["scalar_auroc"]),
        tau_scalar_auroc=float(matched_row["tau_scalar_auroc"]),
        easy_id_top1_accuracy=_proposition_accuracy(proposition_path, "easy_id"),
        hard_id_top1_accuracy=_proposition_accuracy(proposition_path, "hard_id"),
        ambiguous_candidate_hit_rate=_proposition_accuracy(proposition_path, "ambiguous_id"),
        run_output_dir=str(run_output["output_dir"]),
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

    seed_metrics_path = _write_seed_metrics(metrics, output_root / "seed_metrics.csv")
    metric_summary_path = _write_metric_summary(metrics, output_root / "metric_summary.csv")

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
        metric_names=("pair_auroc", "weighted_pair_auroc", "tau_scalar_auroc"),
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
        "seed_rankings": str(rankings_path),
        "auroc_by_seed": str(auroc_plot_path),
        "proposition_accuracy_by_seed": str(proposition_plot_path),
    }
    artifact_index_path = _write_json(artifact_paths, output_root / "artifact_paths.json")

    lines = [
        f"# Study Record: {study_paths['study_id']}",
        "",
        f"- study_id: `{study_paths['study_id']}`",
        f"- study_config_path: `{study_config_path}`",
        f"- shared_eval_manifest: `{study_paths['shared_eval_manifest_path']}`",
        f"- ranking_metric: `{ranking_metric}`",
        f"- best_run_id: `{best_metric.run_id}`",
        f"- worst_run_id: `{worst_metric.run_id}`",
        f"- median_run_id: `{median_metric.run_id}`",
        "",
        "## Seeds",
        "",
    ]
    for metric in metrics:
        lines.append(
            f"- `{metric.run_id}` seed={metric.seed} "
            f"pair_auroc={metric.pair_auroc:.6f} easy_id_top1={metric.easy_id_top1_accuracy:.6f}"
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
        inference_outputs = export_plan_a_inference_bundle(
            protocol_config_path=analysis_protocol_config,
            model_config_path=model_config,
            manifest_path=shared_manifest_outputs["manifest_path"],
            output_dir=run_root / "analysis",
            run_id=run_id,
            checkpoint_path=train_outputs["best_checkpoint_path"],
        )
        artifact_outputs = generate_plan_a_artifact_bundle(
            analysis_path=inference_outputs["analysis_path"],
            analysis_summary_path=inference_outputs["analysis_summary_path"],
            protocol_config_path=analysis_protocol_config,
            eval_config_path=eval_config,
            analysis_config_path=analysis_config,
            output_dir=run_root / "report",
        )
        run_outputs.append(
            {
                "run_id": run_id,
                "seed": seed,
                "output_dir": str(run_root),
                "shared_eval_manifest_path": shared_manifest_outputs["manifest_path"],
                "train": train_outputs,
                "analysis": inference_outputs,
                "report": artifact_outputs,
            }
        )
        _emit_progress(
            progress_callback,
            f"[study] seed_complete run_id={run_id} best_checkpoint={train_outputs['best_checkpoint_path']}",
        )

    study_paths_path = _write_json(
        {
            "study_id": study_id,
            "study_root": str(study_root),
            "study_config_path": str(study_config_path),
            "seeds": seeds,
            "shared_eval_manifest_path": shared_manifest_outputs["manifest_path"],
            "shared_eval_manifest_summary_path": shared_manifest_outputs["manifest_summary_path"],
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
