from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

from frcnet.evaluation import SampleAnalysisRecord

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
            "mean_completion_score_beta_0_1",
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
                    "mean_completion_score_beta_0_1": mean(
                        record.completion_score_beta_0_1 for record in cohort_records
                    ),
                }
            )
    return output


def write_artifact_path_list(artifact_paths: dict[str, str], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact_paths, indent=2, sort_keys=True), encoding="utf-8")
    return output
