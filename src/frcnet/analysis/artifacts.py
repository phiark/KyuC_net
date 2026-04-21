from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from frcnet.evaluation import SampleAnalysisRecord
from frcnet.evaluation.matched_benchmark import (
    build_scalar_roc_curve,
    summarize_scalar_benchmarks,
    write_scalar_benchmark_summaries,
)

COHORT_COLORS = {
    "easy_id": "#1f77b4",
    "hard_id": "#ff7f0e",
    "ambiguous_id": "#2ca02c",
    "ood": "#d62728",
    "unknown_supervision": "#9467bd",
}


def write_geometry_scatter(records: list[SampleAnalysisRecord], output_path: str | Path, dpi: int = 200) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    for cohort_name in sorted({record.cohort_name for record in records}):
        cohort_records = [record for record in records if record.cohort_name == cohort_name]
        plt.scatter(
            [record.resolution_ratio for record in cohort_records],
            [record.content_entropy for record in cohort_records],
            label=cohort_name,
            s=18,
            alpha=0.7,
            color=COHORT_COLORS.get(cohort_name, "#333333"),
        )
    plt.xlabel("resolution_ratio")
    plt.ylabel("content_entropy")
    plt.title("FRCNet Geometry Scatter")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=dpi)
    plt.close()
    return output


def write_geometry_hexbin(records: list[SampleAnalysisRecord], output_path: str | Path, dpi: int = 200) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.hexbin(
        [record.resolution_ratio for record in records],
        [record.content_entropy for record in records],
        gridsize=24,
        cmap="viridis",
        mincnt=1,
    )
    plt.colorbar(label="count")
    plt.xlabel("resolution_ratio")
    plt.ylabel("content_entropy")
    plt.title("FRCNet Geometry Hexbin")
    plt.tight_layout()
    plt.savefig(output, dpi=dpi)
    plt.close()
    return output


def write_cohort_occupancy(records: list[SampleAnalysisRecord], output_path: str | Path, dpi: int = 200) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cohort_counts: dict[str, int] = defaultdict(int)
    for record in records:
        cohort_counts[record.cohort_name] += 1
    labels = list(sorted(cohort_counts))
    values = [cohort_counts[label] for label in labels]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color=[COHORT_COLORS.get(label, "#333333") for label in labels])
    plt.ylabel("count")
    plt.title("Cohort Occupancy")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output, dpi=dpi)
    plt.close()
    return output


def write_tau_cohort_boxplot(records: list[SampleAnalysisRecord], output_path: str | Path, dpi: int = 200) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = list(sorted({record.cohort_name for record in records}))
    distributions = [
        [record.top1_content_probability for record in records if record.cohort_name == cohort_name]
        for cohort_name in labels
    ]

    plt.figure(figsize=(8, 5))
    try:
        boxplot = plt.boxplot(distributions, patch_artist=True, tick_labels=labels, showfliers=False)
    except TypeError:
        boxplot = plt.boxplot(distributions, patch_artist=True, labels=labels, showfliers=False)
    for patch, label in zip(boxplot["boxes"], labels, strict=True):
        patch.set_facecolor(COHORT_COLORS.get(label, "#333333"))
        patch.set_alpha(0.75)
    plt.ylabel("top1_content_probability (tau)")
    plt.title("Top-1 Content Probability By Cohort")
    plt.xticks(rotation=20)
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(output, dpi=dpi)
    plt.close()
    return output


def write_scalar_roc_curve(
    records: list[SampleAnalysisRecord],
    output_path: str | Path,
    *,
    positive_cohort: str,
    negative_cohort: str,
    scalar_name: str,
    test_size: float,
    random_state: int,
    dpi: int = 200,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    curve = build_scalar_roc_curve(
        records,
        positive_cohort=positive_cohort,
        negative_cohort=negative_cohort,
        scalar_name=scalar_name,
        test_size=test_size,
        random_state=random_state,
    )

    plt.figure(figsize=(7, 6))
    plt.plot(
        curve.false_positive_rate,
        curve.true_positive_rate,
        color="#1f77b4",
        linewidth=2.0,
        label=f"{curve.scalar_name} (AUROC={curve.auroc:.3f})",
    )
    plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#666666", linewidth=1.2, label="chance")
    plt.xlabel("false_positive_rate")
    plt.ylabel("true_positive_rate")
    plt.title(f"Matched ROC: {curve.scalar_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output, dpi=dpi)
    plt.close()
    return output


def write_cohort_summary_table(records: list[SampleAnalysisRecord], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[SampleAnalysisRecord]] = defaultdict(list)
    for record in records:
        grouped[record.cohort_name].append(record)

    with output.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "cohort_name",
            "count",
            "mean_resolution_ratio",
            "mean_unknown_mass",
            "mean_content_entropy",
            "mean_resolution_weighted_content_entropy",
            "mean_top1_content_probability",
            "mean_completion_score_beta_0_1",
            "mean_completion_score_beta_0_25",
            "mean_completion_score_beta_0_5",
            "mean_completion_score_beta_0_75",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cohort_name in sorted(grouped):
            cohort_records = grouped[cohort_name]
            writer.writerow(
                {
                    "cohort_name": cohort_name,
                    "count": len(cohort_records),
                    "mean_resolution_ratio": mean(record.resolution_ratio for record in cohort_records),
                    "mean_unknown_mass": mean(record.unknown_mass for record in cohort_records),
                    "mean_content_entropy": mean(record.content_entropy for record in cohort_records),
                    "mean_resolution_weighted_content_entropy": mean(
                        record.resolution_weighted_content_entropy for record in cohort_records
                    ),
                    "mean_top1_content_probability": mean(
                        record.top1_content_probability for record in cohort_records
                    ),
                    "mean_completion_score_beta_0_1": mean(
                        record.completion_score_beta_0_1 for record in cohort_records
                    ),
                    "mean_completion_score_beta_0_25": mean(
                        record.completion_score_beta_0_25 for record in cohort_records
                    ),
                    "mean_completion_score_beta_0_5": mean(
                        record.completion_score_beta_0_5 for record in cohort_records
                    ),
                    "mean_completion_score_beta_0_75": mean(
                        record.completion_score_beta_0_75 for record in cohort_records
                    ),
                }
            )
    return output


def write_completion_scan_table(
    records: list[SampleAnalysisRecord],
    output_path: str | Path,
    *,
    positive_cohort: str,
    negative_cohort: str,
    scalar_names: tuple[str, ...],
    test_size: float,
    random_state: int,
) -> Path:
    summaries = summarize_scalar_benchmarks(
        records,
        scalar_names=scalar_names,
        positive_cohort=positive_cohort,
        negative_cohort=negative_cohort,
        test_size=test_size,
        random_state=random_state,
    )
    return write_scalar_benchmark_summaries(summaries, output_path)


def write_artifact_path_list(artifact_paths: dict[str, str], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact_paths, indent=2, sort_keys=True), encoding="utf-8")
    return output
