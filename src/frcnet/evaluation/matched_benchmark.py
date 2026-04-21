from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from frcnet.evaluation.records import SampleAnalysisRecord

SUPPORTED_PAIR_FEATURES = frozenset(
    {
        "resolution_ratio__content_entropy",
        "resolution_ratio__resolution_weighted_content_entropy",
    }
)
SUPPORTED_SCALAR_FEATURES = frozenset(
    {
        "resolution_ratio",
        "unknown_mass",
        "content_entropy",
        "resolution_weighted_content_entropy",
        "top1_class_mass",
        "top1_content_probability",
        "completion_score_beta_0_1",
        "completion_score_beta_0_25",
        "completion_score_beta_0_5",
        "completion_score_beta_0_75",
    }
)
DEFAULT_TAU_SCALAR_NAME = "top1_content_probability"
DEFAULT_WEIGHTED_PAIR_NAME = "resolution_ratio__resolution_weighted_content_entropy"
DEFAULT_COMPLETION_SCAN_SCALARS = (
    "completion_score_beta_0_1",
    "completion_score_beta_0_25",
    "completion_score_beta_0_5",
    "completion_score_beta_0_75",
)


@dataclass(slots=True)
class ScalarRocCurve:
    scalar_name: str
    positive_cohort: str
    negative_cohort: str
    matched_count_per_class: int
    test_size: float
    random_state: int
    auroc: float
    false_positive_rate: tuple[float, ...]
    true_positive_rate: tuple[float, ...]


@dataclass(slots=True)
class ScalarBenchmarkSummary:
    scalar_name: str
    positive_cohort: str
    negative_cohort: str
    matched_count_per_class: int
    test_size: float
    random_state: int
    auroc: float

    def to_csv_row(self) -> dict[str, str | int | float]:
        return {
            "scalar_name": self.scalar_name,
            "positive_cohort": self.positive_cohort,
            "negative_cohort": self.negative_cohort,
            "matched_count_per_class": self.matched_count_per_class,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "auroc": self.auroc,
        }


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
    weighted_pair_auroc: float
    scalar_auroc: float
    tau_scalar_auroc: float
    pair_name: str = "resolution_ratio__content_entropy"
    weighted_pair_name: str = DEFAULT_WEIGHTED_PAIR_NAME
    scalar_name: str = "completion_score_beta_0_1"
    tau_scalar_name: str = DEFAULT_TAU_SCALAR_NAME
    completion_scan_scalar_names: tuple[str, ...] = DEFAULT_COMPLETION_SCAN_SCALARS
    completion_scan_aurocs: tuple[float, ...] = ()

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
            "weighted_pair_name": self.weighted_pair_name,
            "weighted_pair_auroc": self.weighted_pair_auroc,
            "scalar_auroc": self.scalar_auroc,
            "tau_scalar_name": self.tau_scalar_name,
            "tau_scalar_auroc": self.tau_scalar_auroc,
            "completion_scan_scalar_names_json": json.dumps(list(self.completion_scan_scalar_names)),
            "completion_scan_aurocs_json": json.dumps(list(self.completion_scan_aurocs)),
        }


def _build_pair_features(records: list[SampleAnalysisRecord], pair_name: str) -> np.ndarray:
    if pair_name == "resolution_ratio__content_entropy":
        return np.array([[record.resolution_ratio, record.content_entropy] for record in records], dtype=np.float64)
    if pair_name == DEFAULT_WEIGHTED_PAIR_NAME:
        return np.array(
            [[record.resolution_ratio, record.resolution_weighted_content_entropy] for record in records],
            dtype=np.float64,
        )
    raise ValueError(f"Unsupported primary_pair: {pair_name}. Supported values: {sorted(SUPPORTED_PAIR_FEATURES)}")


def _build_scalar_features(records: list[SampleAnalysisRecord], scalar_name: str) -> np.ndarray:
    if scalar_name not in SUPPORTED_SCALAR_FEATURES:
        raise ValueError(
            f"Unsupported primary_scalar: {scalar_name}. Supported values: {sorted(SUPPORTED_SCALAR_FEATURES)}"
        )
    return np.array([float(getattr(record, scalar_name)) for record in records], dtype=np.float64)


def _prepare_matched_records(
    sample_analysis_records: list[SampleAnalysisRecord],
    *,
    positive_cohort: str,
    negative_cohort: str,
    test_size: float,
    random_state: int,
) -> tuple[list[SampleAnalysisRecord], np.ndarray, np.ndarray]:
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
    labels = np.array([1] * matched_count + [0] * matched_count, dtype=np.int64)
    train_index, test_index = train_test_split(
        np.arange(labels.shape[0]),
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    return ordered_records, labels, test_index


def build_scalar_roc_curve(
    sample_analysis_records: list[SampleAnalysisRecord],
    *,
    positive_cohort: str = "ambiguous_id",
    negative_cohort: str = "ood",
    scalar_name: str = DEFAULT_TAU_SCALAR_NAME,
    test_size: float = 0.3,
    random_state: int = 7,
) -> ScalarRocCurve:
    ordered_records, labels, test_index = _prepare_matched_records(
        sample_analysis_records,
        positive_cohort=positive_cohort,
        negative_cohort=negative_cohort,
        test_size=test_size,
        random_state=random_state,
    )
    scalar_features = _build_scalar_features(ordered_records, scalar_name)
    false_positive_rate, true_positive_rate, _ = roc_curve(labels[test_index], scalar_features[test_index])
    auroc = float(roc_auc_score(labels[test_index], scalar_features[test_index]))
    matched_count = labels.shape[0] // 2
    return ScalarRocCurve(
        scalar_name=scalar_name,
        positive_cohort=positive_cohort,
        negative_cohort=negative_cohort,
        matched_count_per_class=matched_count,
        test_size=test_size,
        random_state=random_state,
        auroc=auroc,
        false_positive_rate=tuple(float(value) for value in false_positive_rate),
        true_positive_rate=tuple(float(value) for value in true_positive_rate),
    )


def summarize_scalar_benchmarks(
    sample_analysis_records: list[SampleAnalysisRecord],
    *,
    scalar_names: Sequence[str],
    positive_cohort: str = "ambiguous_id",
    negative_cohort: str = "ood",
    test_size: float = 0.3,
    random_state: int = 7,
) -> list[ScalarBenchmarkSummary]:
    ordered_records, labels, test_index = _prepare_matched_records(
        sample_analysis_records,
        positive_cohort=positive_cohort,
        negative_cohort=negative_cohort,
        test_size=test_size,
        random_state=random_state,
    )
    matched_count = labels.shape[0] // 2
    unique_scalar_names: list[str] = []
    for scalar_name in scalar_names:
        if scalar_name not in unique_scalar_names:
            unique_scalar_names.append(scalar_name)

    summaries: list[ScalarBenchmarkSummary] = []
    for scalar_name in unique_scalar_names:
        scalar_features = _build_scalar_features(ordered_records, scalar_name)
        summaries.append(
            ScalarBenchmarkSummary(
                scalar_name=scalar_name,
                positive_cohort=positive_cohort,
                negative_cohort=negative_cohort,
                matched_count_per_class=matched_count,
                test_size=test_size,
                random_state=random_state,
                auroc=float(roc_auc_score(labels[test_index], scalar_features[test_index])),
            )
        )
    return summaries


def summarize_matched_ambiguous_vs_ood(
    sample_analysis_records: list[SampleAnalysisRecord],
    positive_cohort: str = "ambiguous_id",
    negative_cohort: str = "ood",
    primary_pair: str = "resolution_ratio__content_entropy",
    weighted_pair: str = DEFAULT_WEIGHTED_PAIR_NAME,
    primary_scalar: str = "completion_score_beta_0_1",
    tau_scalar_name: str = DEFAULT_TAU_SCALAR_NAME,
    completion_scan_scalars: Sequence[str] = DEFAULT_COMPLETION_SCAN_SCALARS,
    test_size: float = 0.3,
    random_state: int = 7,
) -> MatchedBenchmarkSummary:
    ordered_records, labels, test_index = _prepare_matched_records(
        sample_analysis_records,
        positive_cohort=positive_cohort,
        negative_cohort=negative_cohort,
        test_size=test_size,
        random_state=random_state,
    )
    matched_count = labels.shape[0] // 2

    pair_features = _build_pair_features(ordered_records, primary_pair)
    weighted_pair_features = _build_pair_features(ordered_records, weighted_pair)
    scalar_features = _build_scalar_features(ordered_records, primary_scalar)
    tau_scalar_features = _build_scalar_features(ordered_records, tau_scalar_name)
    completion_scan_summaries = summarize_scalar_benchmarks(
        ordered_records,
        scalar_names=completion_scan_scalars,
        positive_cohort=positive_cohort,
        negative_cohort=negative_cohort,
        test_size=test_size,
        random_state=random_state,
    )

    train_index = np.setdiff1d(np.arange(labels.shape[0]), test_index, assume_unique=True)
    classifier = LogisticRegression(random_state=random_state, max_iter=1000)
    classifier.fit(pair_features[train_index], labels[train_index])
    pair_probability = classifier.predict_proba(pair_features[test_index])[:, 1]
    pair_auroc = float(roc_auc_score(labels[test_index], pair_probability))
    weighted_pair_classifier = LogisticRegression(random_state=random_state, max_iter=1000)
    weighted_pair_classifier.fit(weighted_pair_features[train_index], labels[train_index])
    weighted_pair_probability = weighted_pair_classifier.predict_proba(weighted_pair_features[test_index])[:, 1]
    weighted_pair_auroc = float(roc_auc_score(labels[test_index], weighted_pair_probability))
    scalar_auroc = float(roc_auc_score(labels[test_index], scalar_features[test_index]))
    tau_scalar_auroc = float(roc_auc_score(labels[test_index], tau_scalar_features[test_index]))

    return MatchedBenchmarkSummary(
        protocol_id=ordered_records[0].protocol_id
        if len({record.protocol_id for record in ordered_records}) == 1
        else "MULTIPLE",
        run_id=ordered_records[0].run_id if len({record.run_id for record in ordered_records}) == 1 else "MULTIPLE",
        matched_count_per_class=matched_count,
        num_ambiguous=matched_count,
        num_ood=matched_count,
        positive_cohort=positive_cohort,
        negative_cohort=negative_cohort,
        test_size=test_size,
        random_state=random_state,
        pair_auroc=pair_auroc,
        weighted_pair_auroc=weighted_pair_auroc,
        scalar_auroc=scalar_auroc,
        tau_scalar_auroc=tau_scalar_auroc,
        pair_name=primary_pair,
        weighted_pair_name=weighted_pair,
        scalar_name=primary_scalar,
        tau_scalar_name=tau_scalar_name,
        completion_scan_scalar_names=tuple(summary.scalar_name for summary in completion_scan_summaries),
        completion_scan_aurocs=tuple(summary.auroc for summary in completion_scan_summaries),
    )


def write_matched_benchmark_summary(summary: MatchedBenchmarkSummary, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary.to_csv_row().keys()))
        writer.writeheader()
        writer.writerow(summary.to_csv_row())
    return output


def write_scalar_benchmark_summaries(
    summaries: Sequence[ScalarBenchmarkSummary],
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(summaries[0].to_csv_row().keys()) if summaries else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for summary in summaries:
                writer.writerow(summary.to_csv_row())
    return output
