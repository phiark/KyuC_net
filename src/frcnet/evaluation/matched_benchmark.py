from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from frcnet.evaluation.records import SampleAnalysisRecord

SUPPORTED_PAIR_FEATURES = frozenset({"resolution_ratio__content_entropy"})
SUPPORTED_SCALAR_FEATURES = frozenset(
    {
        "resolution_ratio",
        "unknown_mass",
        "content_entropy",
        "top1_class_mass",
        "top1_content_probability",
        "completion_score_beta_0_1",
        "completion_score_beta_0_5",
    }
)


@dataclass(slots=True)
class MatchedBenchmarkSummary:
    protocol_id: str
    run_id: str
    matched_count_per_class: int
    num_ambiguous: int
    num_ood: int
    positive_cohort: str
    negative_cohort: str
    test_size: float
    random_state: int
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
            "positive_cohort": self.positive_cohort,
            "negative_cohort": self.negative_cohort,
            "pair_name": self.pair_name,
            "scalar_name": self.scalar_name,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "pair_auroc": self.pair_auroc,
            "scalar_auroc": self.scalar_auroc,
        }


def _build_pair_features(records: list[SampleAnalysisRecord], pair_name: str) -> np.ndarray:
    if pair_name != "resolution_ratio__content_entropy":
        raise ValueError(f"Unsupported primary_pair: {pair_name}. Supported values: {sorted(SUPPORTED_PAIR_FEATURES)}")
    return np.array([[record.resolution_ratio, record.content_entropy] for record in records], dtype=np.float64)


def _build_scalar_features(records: list[SampleAnalysisRecord], scalar_name: str) -> np.ndarray:
    if scalar_name not in SUPPORTED_SCALAR_FEATURES:
        raise ValueError(
            f"Unsupported primary_scalar: {scalar_name}. Supported values: {sorted(SUPPORTED_SCALAR_FEATURES)}"
        )
    return np.array([float(getattr(record, scalar_name)) for record in records], dtype=np.float64)


def summarize_matched_ambiguous_vs_ood(
    sample_analysis_records: list[SampleAnalysisRecord],
    positive_cohort: str = "ambiguous_id",
    negative_cohort: str = "ood",
    primary_pair: str = "resolution_ratio__content_entropy",
    primary_scalar: str = "completion_score_beta_0_1",
    test_size: float = 0.3,
    random_state: int = 7,
) -> MatchedBenchmarkSummary:
    if positive_cohort == negative_cohort:
        raise ValueError("positive_cohort and negative_cohort must be different.")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be within (0, 1).")

    positive_records = [record for record in sample_analysis_records if record.cohort_name == positive_cohort]
    negative_records = [record for record in sample_analysis_records if record.cohort_name == negative_cohort]
    matched_count = min(len(positive_records), len(negative_records))
    if matched_count < 2:
        raise ValueError(
            f"Matched benchmark requires at least two `{positive_cohort}` and two `{negative_cohort}` records."
        )

    positive_records = sorted(positive_records, key=lambda record: record.sample_id)[:matched_count]
    negative_records = sorted(negative_records, key=lambda record: record.sample_id)[:matched_count]
    ordered_records = positive_records + negative_records

    pair_features = _build_pair_features(ordered_records, primary_pair)
    scalar_features = _build_scalar_features(ordered_records, primary_scalar)
    labels = np.array([1] * matched_count + [0] * matched_count, dtype=np.int64)

    train_index, test_index = train_test_split(
        np.arange(labels.shape[0]),
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    classifier = LogisticRegression(random_state=random_state, max_iter=1000)
    classifier.fit(pair_features[train_index], labels[train_index])
    pair_probability = classifier.predict_proba(pair_features[test_index])[:, 1]
    pair_auroc = float(roc_auc_score(labels[test_index], pair_probability))
    scalar_auroc = float(roc_auc_score(labels[test_index], scalar_features[test_index]))

    return MatchedBenchmarkSummary(
        protocol_id=ordered_records[0].protocol_id
        if len({record.protocol_id for record in ordered_records}) == 1
        else "MULTIPLE",
        run_id=ordered_records[0].run_id if len({record.run_id for record in ordered_records}) == 1 else "MULTIPLE",
        matched_count_per_class=matched_count,
        num_ambiguous=len(positive_records),
        num_ood=len(negative_records),
        positive_cohort=positive_cohort,
        negative_cohort=negative_cohort,
        test_size=test_size,
        random_state=random_state,
        pair_auroc=pair_auroc,
        scalar_auroc=scalar_auroc,
        pair_name=primary_pair,
        scalar_name=primary_scalar,
    )


def write_matched_benchmark_summary(summary: MatchedBenchmarkSummary, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.to_csv_row().keys()))
        writer.writeheader()
        writer.writerow(summary.to_csv_row())
    return output
