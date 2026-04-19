from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from frcnet.evaluation.records import SampleAnalysisRecord


@dataclass(slots=True)
class MatchedBenchmarkSummary:
    protocol_id: str
    run_id: str
    matched_count_per_class: int
    num_ambiguous: int
    num_ood: int
    pair_auroc: float
    scalar_auroc: float
    pair_name: str = "resolution_ratio__content_entropy"
    scalar_name: str = "completion_score_beta_0_1"

    def to_csv_row(self) -> dict[str, str | int | float]:
        return {
            "protocol_id": self.protocol_id,
            "run_id": self.run_id,
            "matched_count_per_class": self.matched_count_per_class,
            "num_ambiguous": self.num_ambiguous,
            "num_ood": self.num_ood,
            "pair_name": self.pair_name,
            "scalar_name": self.scalar_name,
            "pair_auroc": self.pair_auroc,
            "scalar_auroc": self.scalar_auroc,
        }


def summarize_matched_ambiguous_vs_ood(
    sample_analysis_records: list[SampleAnalysisRecord],
    random_state: int = 7,
) -> MatchedBenchmarkSummary:
    ambiguous_records = [record for record in sample_analysis_records if record.cohort_name == "ambiguous_id"]
    ood_records = [record for record in sample_analysis_records if record.cohort_name == "ood"]
    matched_count = min(len(ambiguous_records), len(ood_records))
    if matched_count < 2:
        raise ValueError("Matched benchmark requires at least two ambiguous and two ood records.")

    ambiguous_records = sorted(ambiguous_records, key=lambda record: record.sample_id)[:matched_count]
    ood_records = sorted(ood_records, key=lambda record: record.sample_id)[:matched_count]
    ordered_records = ambiguous_records + ood_records

    pair_features = np.array(
        [[record.resolution_ratio, record.content_entropy] for record in ordered_records],
        dtype=np.float64,
    )
    scalar_features = np.array([record.completion_score_beta_0_1 for record in ordered_records], dtype=np.float64)
    labels = np.array([1] * matched_count + [0] * matched_count, dtype=np.int64)

    train_index, test_index = train_test_split(
        np.arange(labels.shape[0]),
        test_size=0.3,
        random_state=random_state,
        stratify=labels,
    )
    classifier = LogisticRegression(random_state=random_state, max_iter=1000)
    classifier.fit(pair_features[train_index], labels[train_index])
    pair_probability = classifier.predict_proba(pair_features[test_index])[:, 1]
    pair_auroc = float(roc_auc_score(labels[test_index], pair_probability))
    scalar_auroc = float(roc_auc_score(labels[test_index], scalar_features[test_index]))

    return MatchedBenchmarkSummary(
        protocol_id=ordered_records[0].protocol_id,
        run_id=ordered_records[0].run_id,
        matched_count_per_class=matched_count,
        num_ambiguous=len(ambiguous_records),
        num_ood=len(ood_records),
        pair_auroc=pair_auroc,
        scalar_auroc=scalar_auroc,
    )


def write_matched_benchmark_summary(summary: MatchedBenchmarkSummary, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.to_csv_row().keys()))
        writer.writeheader()
        writer.writerow(summary.to_csv_row())
    return output

