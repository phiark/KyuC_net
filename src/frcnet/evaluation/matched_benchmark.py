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

from frcnet.evaluation.matched_manifest import MatchedManifestRecord
from frcnet.evaluation.records import DEFAULT_MODEL_FAMILY, SampleAnalysisRecord

SUPPORTED_PAIR_FEATURES = frozenset(
    {
        "resolution_ratio__content_entropy",
        "resolution_ratio__resolution_weighted_content_entropy",
        "resolution_ratio__state_content_entropy",
        "resolution_ratio__state_weighted_content_entropy",
    }
)
SUPPORTED_PRIMARY_SCALAR_FEATURES = frozenset(
    {
        "resolution_ratio",
        "unknown_mass",
        "content_entropy",
        "resolution_weighted_content_entropy",
        "state_content_entropy",
        "state_weighted_content_entropy",
        "state_entropy",
        "resolution_entropy",
        "top1_class_mass",
        "top1_view_truth_mass",
        "top1_view_false_mass",
        "top1_view_unknown_mass",
        "top1_view_tau",
        "auxiliary_top1_content_probability",
        "top1_content_probability",
        "completion_score_beta_0_1",
        "completion_score_beta_0_25",
        "completion_score_beta_0_5",
        "completion_score_beta_0_75",
        "top1_completion_beta_0_1",
        "top1_completion_beta_0_25",
        "top1_completion_beta_0_5",
        "top1_completion_beta_0_75",
    }
)
SUPPORTED_DIAGNOSTIC_SCALAR_FEATURES = SUPPORTED_PRIMARY_SCALAR_FEATURES | frozenset(
    {
        "proposition_truth_mass",
        "proposition_false_mass",
        "proposition_unknown_mass",
        "proposition_truth_ratio",
        "ternary_entropy",
    }
)
SUPPORTED_SCALAR_FEATURES = SUPPORTED_PRIMARY_SCALAR_FEATURES
SCALAR_NAME_ALIASES = {
    "top1_content_probability": "auxiliary_top1_content_probability",
    "content_entropy": "state_content_entropy",
    "resolution_weighted_content_entropy": "state_weighted_content_entropy",
    "completion_score_beta_0_1": "top1_completion_beta_0_1",
    "completion_score_beta_0_25": "top1_completion_beta_0_25",
    "completion_score_beta_0_5": "top1_completion_beta_0_5",
    "completion_score_beta_0_75": "top1_completion_beta_0_75",
}
DEFAULT_TAU_SCALAR_NAME = "proposition_truth_ratio"
DEFAULT_WEIGHTED_PAIR_NAME = "resolution_ratio__state_weighted_content_entropy"
DEFAULT_COMPLETION_SCAN_SCALARS = (
    "top1_completion_beta_0_1",
    "top1_completion_beta_0_25",
    "top1_completion_beta_0_5",
    "top1_completion_beta_0_75",
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
    raw_auc: float
    oriented_auc: float
    one_feature_logistic_auc: float

    def to_csv_row(self) -> dict[str, str | int | float]:
        return {
            "scalar_name": self.scalar_name,
            "positive_cohort": self.positive_cohort,
            "negative_cohort": self.negative_cohort,
            "matched_count_per_class": self.matched_count_per_class,
            "test_size": self.test_size,
            "random_state": self.random_state,
            "auroc": self.auroc,
            "raw_auc": self.raw_auc,
            "oriented_auc": self.oriented_auc,
            "one_feature_logistic_auc": self.one_feature_logistic_auc,
        }


@dataclass(slots=True)
class MatchedBenchmarkSummary:
    model_family: str
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
    scalar_raw_auc: float
    scalar_oriented_auc: float
    scalar_one_feature_logistic_auc: float
    pair_name: str = "resolution_ratio__state_content_entropy"
    weighted_pair_name: str = DEFAULT_WEIGHTED_PAIR_NAME
    scalar_name: str = "top1_completion_beta_0_1"
    secondary_pair_name: str = ""
    secondary_pair_auroc: float | None = None
    completion_scan_scalar_names: tuple[str, ...] = DEFAULT_COMPLETION_SCAN_SCALARS
    completion_scan_aurocs: tuple[float, ...] = ()

    def to_csv_row(self) -> dict[str, str | int | float]:
        secondary_pair_name = self.secondary_pair_name or self.weighted_pair_name
        secondary_pair_auroc = (
            self.weighted_pair_auroc if self.secondary_pair_auroc is None else self.secondary_pair_auroc
        )
        return {
            "model_family": self.model_family,
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
            "secondary_pair_name": secondary_pair_name,
            "secondary_pair_auroc": secondary_pair_auroc,
            "weighted_pair_name": self.weighted_pair_name,
            "weighted_pair_auroc": self.weighted_pair_auroc,
            "scalar_auroc": self.scalar_auroc,
            "scalar_raw_auc": self.scalar_raw_auc,
            "scalar_oriented_auc": self.scalar_oriented_auc,
            "scalar_one_feature_logistic_auc": self.scalar_one_feature_logistic_auc,
            "completion_scan_scalar_names_json": json.dumps(list(self.completion_scan_scalar_names)),
            "completion_scan_aurocs_json": json.dumps(list(self.completion_scan_aurocs)),
        }


def _build_pair_features(records: list[SampleAnalysisRecord], pair_name: str) -> np.ndarray:
    if pair_name in {"resolution_ratio__content_entropy", "resolution_ratio__state_content_entropy"}:
        return np.array(
            [[record.resolution_ratio, record.state_content_entropy] for record in records],
            dtype=np.float64,
        )
    if pair_name in {DEFAULT_WEIGHTED_PAIR_NAME, "resolution_ratio__resolution_weighted_content_entropy"}:
        return np.array(
            [[record.resolution_ratio, record.state_weighted_content_entropy] for record in records],
            dtype=np.float64,
        )
    raise ValueError(f"Unsupported primary_pair: {pair_name}. Supported values: {sorted(SUPPORTED_PAIR_FEATURES)}")


def _build_scalar_features(
    records: list[SampleAnalysisRecord],
    scalar_name: str,
    *,
    allow_label_aware: bool = False,
) -> np.ndarray:
    supported_features = SUPPORTED_DIAGNOSTIC_SCALAR_FEATURES if allow_label_aware else SUPPORTED_PRIMARY_SCALAR_FEATURES
    if scalar_name not in supported_features:
        raise ValueError(
            f"Unsupported primary_scalar: {scalar_name}. Supported values: {sorted(supported_features)}"
        )
    resolved_scalar_name = SCALAR_NAME_ALIASES.get(scalar_name, scalar_name)
    return np.array([float(getattr(record, resolved_scalar_name)) for record in records], dtype=np.float64)


def _scalar_auc_bundle(
    scalar_features: np.ndarray,
    labels: np.ndarray,
    train_index: np.ndarray,
    test_index: np.ndarray,
    *,
    random_state: int,
) -> tuple[float, float, float]:
    raw_auc = float(roc_auc_score(labels[test_index], scalar_features[test_index]))
    oriented_auc = max(raw_auc, 1.0 - raw_auc)
    classifier = LogisticRegression(random_state=random_state, max_iter=1000)
    classifier.fit(scalar_features[train_index].reshape(-1, 1), labels[train_index])
    scalar_probability = classifier.predict_proba(scalar_features[test_index].reshape(-1, 1))[:, 1]
    one_feature_logistic_auc = float(roc_auc_score(labels[test_index], scalar_probability))
    return raw_auc, oriented_auc, one_feature_logistic_auc


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


def _prepare_frozen_matched_records(
    sample_analysis_records: list[SampleAnalysisRecord],
    *,
    matched_manifest_records: Sequence[MatchedManifestRecord],
    positive_cohort: str,
    negative_cohort: str,
    test_size: float,
    random_state: int,
) -> tuple[list[SampleAnalysisRecord], np.ndarray, np.ndarray]:
    record_by_id = {record.sample_id: record for record in sample_analysis_records}
    selected_manifest = [
        record
        for record in matched_manifest_records
        if record.cohort_name in {positive_cohort, negative_cohort} and record.sample_id in record_by_id
    ]
    if not selected_manifest:
        raise ValueError("Frozen matched manifest did not match any sample_analysis_records.")
    selected_manifest = sorted(
        selected_manifest,
        key=lambda record: (record.paired_group_id, record.cohort_name, record.sample_id),
    )
    ordered_records = [record_by_id[record.sample_id] for record in selected_manifest]
    labels = np.array([1 if record.cohort_name == positive_cohort else 0 for record in selected_manifest], dtype=np.int64)
    if len(set(labels.tolist())) != 2:
        raise ValueError("Frozen matched manifest must include both positive and negative cohorts.")

    test_index = np.array(
        [
            index
            for index, record in enumerate(selected_manifest)
            if record.manifest_role in {"test", "eval_only"}
        ],
        dtype=np.int64,
    )
    if test_index.size == 0 or test_index.size == labels.shape[0]:
        _, test_index = train_test_split(
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
    allow_label_aware: bool = True,
    matched_manifest_records: Sequence[MatchedManifestRecord] | None = None,
) -> ScalarRocCurve:
    if matched_manifest_records is None:
        ordered_records, labels, test_index = _prepare_matched_records(
            sample_analysis_records,
            positive_cohort=positive_cohort,
            negative_cohort=negative_cohort,
            test_size=test_size,
            random_state=random_state,
        )
    else:
        ordered_records, labels, test_index = _prepare_frozen_matched_records(
            sample_analysis_records,
            matched_manifest_records=matched_manifest_records,
            positive_cohort=positive_cohort,
            negative_cohort=negative_cohort,
            test_size=test_size,
            random_state=random_state,
        )
    scalar_features = _build_scalar_features(ordered_records, scalar_name, allow_label_aware=allow_label_aware)
    false_positive_rate, true_positive_rate, _ = roc_curve(labels[test_index], scalar_features[test_index])
    auroc = float(roc_auc_score(labels[test_index], scalar_features[test_index]))
    matched_count = int(min(labels.sum(), labels.shape[0] - labels.sum()))
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


def _summarize_scalar_benchmarks_from_prepared(
    ordered_records: list[SampleAnalysisRecord],
    labels: np.ndarray,
    test_index: np.ndarray,
    *,
    scalar_names: Sequence[str],
    positive_cohort: str,
    negative_cohort: str,
    test_size: float,
    random_state: int,
    allow_label_aware: bool,
) -> list[ScalarBenchmarkSummary]:
    matched_count = int(min(labels.sum(), labels.shape[0] - labels.sum()))
    unique_scalar_names: list[str] = []
    for scalar_name in scalar_names:
        if scalar_name not in unique_scalar_names:
            unique_scalar_names.append(scalar_name)

    summaries: list[ScalarBenchmarkSummary] = []
    train_index = np.setdiff1d(np.arange(labels.shape[0]), test_index, assume_unique=True)
    for scalar_name in unique_scalar_names:
        scalar_features = _build_scalar_features(ordered_records, scalar_name, allow_label_aware=allow_label_aware)
        raw_auc, oriented_auc, one_feature_logistic_auc = _scalar_auc_bundle(
            scalar_features,
            labels,
            train_index,
            test_index,
            random_state=random_state,
        )
        summaries.append(
            ScalarBenchmarkSummary(
                scalar_name=scalar_name,
                positive_cohort=positive_cohort,
                negative_cohort=negative_cohort,
                matched_count_per_class=matched_count,
                test_size=test_size,
                random_state=random_state,
                auroc=raw_auc,
                raw_auc=raw_auc,
                oriented_auc=oriented_auc,
                one_feature_logistic_auc=one_feature_logistic_auc,
            )
        )
    return summaries


def summarize_scalar_benchmarks(
    sample_analysis_records: list[SampleAnalysisRecord],
    *,
    scalar_names: Sequence[str],
    positive_cohort: str = "ambiguous_id",
    negative_cohort: str = "ood",
    test_size: float = 0.3,
    random_state: int = 7,
    allow_label_aware: bool = False,
    matched_manifest_records: Sequence[MatchedManifestRecord] | None = None,
) -> list[ScalarBenchmarkSummary]:
    if matched_manifest_records is None:
        ordered_records, labels, test_index = _prepare_matched_records(
            sample_analysis_records,
            positive_cohort=positive_cohort,
            negative_cohort=negative_cohort,
            test_size=test_size,
            random_state=random_state,
        )
    else:
        ordered_records, labels, test_index = _prepare_frozen_matched_records(
            sample_analysis_records,
            matched_manifest_records=matched_manifest_records,
            positive_cohort=positive_cohort,
            negative_cohort=negative_cohort,
            test_size=test_size,
            random_state=random_state,
        )
    return _summarize_scalar_benchmarks_from_prepared(
        ordered_records,
        labels,
        test_index,
        scalar_names=scalar_names,
        positive_cohort=positive_cohort,
        negative_cohort=negative_cohort,
        test_size=test_size,
        random_state=random_state,
        allow_label_aware=allow_label_aware,
    )


def summarize_matched_ambiguous_vs_ood(
    sample_analysis_records: list[SampleAnalysisRecord],
    positive_cohort: str = "ambiguous_id",
    negative_cohort: str = "ood",
    primary_pair: str = "resolution_ratio__state_content_entropy",
    weighted_pair: str = DEFAULT_WEIGHTED_PAIR_NAME,
    primary_scalar: str = "top1_completion_beta_0_1",
    completion_scan_scalars: Sequence[str] = DEFAULT_COMPLETION_SCAN_SCALARS,
    test_size: float = 0.3,
    random_state: int = 7,
    matched_manifest_records: Sequence[MatchedManifestRecord] | None = None,
) -> MatchedBenchmarkSummary:
    if matched_manifest_records is None:
        ordered_records, labels, test_index = _prepare_matched_records(
            sample_analysis_records,
            positive_cohort=positive_cohort,
            negative_cohort=negative_cohort,
            test_size=test_size,
            random_state=random_state,
        )
    else:
        ordered_records, labels, test_index = _prepare_frozen_matched_records(
            sample_analysis_records,
            matched_manifest_records=matched_manifest_records,
            positive_cohort=positive_cohort,
            negative_cohort=negative_cohort,
            test_size=test_size,
            random_state=random_state,
        )
    matched_count = int(min(labels.sum(), labels.shape[0] - labels.sum()))

    pair_features = _build_pair_features(ordered_records, primary_pair)
    weighted_pair_features = _build_pair_features(ordered_records, weighted_pair)
    scalar_features = _build_scalar_features(ordered_records, primary_scalar)
    completion_scan_summaries = _summarize_scalar_benchmarks_from_prepared(
        ordered_records,
        labels,
        test_index,
        scalar_names=completion_scan_scalars,
        positive_cohort=positive_cohort,
        negative_cohort=negative_cohort,
        test_size=test_size,
        random_state=random_state,
        allow_label_aware=False,
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
    scalar_raw_auc, scalar_oriented_auc, scalar_one_feature_logistic_auc = _scalar_auc_bundle(
        scalar_features,
        labels,
        train_index,
        test_index,
        random_state=random_state,
    )

    return MatchedBenchmarkSummary(
        model_family=ordered_records[0].model_family
        if len({record.model_family for record in ordered_records}) == 1
        else "MULTIPLE",
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
        scalar_auroc=scalar_raw_auc,
        scalar_raw_auc=scalar_raw_auc,
        scalar_oriented_auc=scalar_oriented_auc,
        scalar_one_feature_logistic_auc=scalar_one_feature_logistic_auc,
        pair_name=primary_pair,
        weighted_pair_name=weighted_pair,
        scalar_name=primary_scalar,
        secondary_pair_name=weighted_pair,
        secondary_pair_auroc=weighted_pair_auroc,
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
