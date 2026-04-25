from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
import random
from typing import Iterable, Mapping, Sequence

import numpy as np

from frcnet.evaluation.records import SampleAnalysisRecord


@dataclass(frozen=True, slots=True)
class MatchedManifestRecord:
    sample_id: str
    cohort_name: str
    source_dataset_name: str
    source_index: str
    reference_score_name: str
    reference_score_value: float
    match_bin_id: str
    paired_group_id: str
    manifest_role: str
    construction_config_hash: str
    manifest_hash: str = ""

    def to_dict(self) -> dict[str, str | float]:
        return {
            "sample_id": self.sample_id,
            "cohort_name": self.cohort_name,
            "source_dataset_name": self.source_dataset_name,
            "source_index": self.source_index,
            "reference_score_name": self.reference_score_name,
            "reference_score_value": self.reference_score_value,
            "match_bin_id": self.match_bin_id,
            "paired_group_id": self.paired_group_id,
            "manifest_role": self.manifest_role,
            "construction_config_hash": self.construction_config_hash,
            "manifest_hash": self.manifest_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "MatchedManifestRecord":
        return cls(
            sample_id=str(payload["sample_id"]),
            cohort_name=str(payload["cohort_name"]),
            source_dataset_name=str(payload["source_dataset_name"]),
            source_index=str(payload.get("source_index", "")),
            reference_score_name=str(payload["reference_score_name"]),
            reference_score_value=float(payload["reference_score_value"]),
            match_bin_id=str(payload["match_bin_id"]),
            paired_group_id=str(payload["paired_group_id"]),
            manifest_role=str(payload.get("manifest_role", "eval")),
            construction_config_hash=str(payload["construction_config_hash"]),
            manifest_hash=str(payload.get("manifest_hash", "")),
        )


@dataclass(frozen=True, slots=True)
class MatchedManifestBinDiagnostic:
    match_bin_id: str
    positive_cohort: str
    negative_cohort: str
    positive_count: int
    negative_count: int
    matched_pairs: int
    unmatched_positive: int
    unmatched_negative: int
    score_min: float
    score_max: float

    def to_csv_row(self) -> dict[str, str | int | float]:
        return {
            "match_bin_id": self.match_bin_id,
            "positive_cohort": self.positive_cohort,
            "negative_cohort": self.negative_cohort,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "matched_pairs": self.matched_pairs,
            "unmatched_positive": self.unmatched_positive,
            "unmatched_negative": self.unmatched_negative,
            "score_min": self.score_min,
            "score_max": self.score_max,
        }


def _stable_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def compute_construction_config_hash(config: Mapping[str, object]) -> str:
    return hashlib.sha256(_stable_json(dict(config)).encode("utf-8")).hexdigest()


def compute_manifest_hash(records: Iterable[MatchedManifestRecord]) -> str:
    rows = []
    for record in records:
        payload = record.to_dict()
        payload["manifest_hash"] = ""
        rows.append(payload)
    return hashlib.sha256(_stable_json(rows).encode("utf-8")).hexdigest()


def compute_matched_manifest_hash(records: Iterable[MatchedManifestRecord]) -> str:
    return compute_manifest_hash(records)


def _reference_score_mapping(
    reference_scores: Mapping[str, float] | Sequence[object],
    reference_score_name: str | None,
) -> tuple[dict[str, float], str]:
    if isinstance(reference_scores, Mapping):
        if reference_score_name is None:
            raise ValueError("reference_score_name is required when reference_scores is a mapping.")
        return {str(key): float(value) for key, value in reference_scores.items()}, reference_score_name

    score_by_id: dict[str, float] = {}
    score_names: set[str] = set()
    for record in reference_scores:
        sample_id = str(getattr(record, "sample_id"))
        score_name = str(getattr(record, "reference_score_name"))
        if reference_score_name is not None and score_name != reference_score_name:
            continue
        score_by_id[sample_id] = float(getattr(record, "reference_score_value"))
        score_names.add(score_name)
    if not score_by_id:
        raise ValueError("No reference score records matched the requested reference_score_name.")
    if reference_score_name is not None:
        return score_by_id, reference_score_name
    if len(score_names) != 1:
        raise ValueError("reference_score_name is required when multiple score names are present.")
    return score_by_id, next(iter(score_names))


def _quantile_bin_indices(values: np.ndarray, num_bins: int) -> np.ndarray:
    if values.size == 0:
        return np.array([], dtype=np.int64)
    if float(values.max()) == float(values.min()):
        return np.zeros(values.shape[0], dtype=np.int64)
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, num_bins + 1))
    inner_edges = quantiles[1:-1]
    bin_indices = np.searchsorted(inner_edges, values, side="right").astype(np.int64)
    return np.clip(bin_indices, 0, num_bins - 1)


def _pair_roles(pair_ids: Sequence[str], *, test_size: float | None, random_state: int, default_role: str) -> dict[str, str]:
    unique_pair_ids = sorted(set(pair_ids))
    if test_size is None:
        return {pair_id: default_role for pair_id in unique_pair_ids}
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be within (0, 1).")
    shuffled_pair_ids = list(unique_pair_ids)
    random.Random(random_state).shuffle(shuffled_pair_ids)
    if len(shuffled_pair_ids) == 1:
        test_count = 1
    else:
        test_count = max(1, int(round(len(shuffled_pair_ids) * test_size)))
        test_count = min(test_count, len(shuffled_pair_ids) - 1)
    test_pair_ids = set(shuffled_pair_ids[:test_count])
    return {pair_id: ("test" if pair_id in test_pair_ids else "train") for pair_id in unique_pair_ids}


def build_frozen_matched_manifest_with_diagnostics(
    records: list[SampleAnalysisRecord],
    reference_scores: Mapping[str, float] | Sequence[object],
    *,
    reference_score_name: str | None = None,
    positive_cohort: str = "ambiguous_id",
    negative_cohort: str = "ood",
    num_bins: int = 10,
    manifest_role: str = "eval",
    test_size: float | None = None,
    random_state: int = 7,
    construction_config_hash: str | None = None,
) -> tuple[list[MatchedManifestRecord], list[MatchedManifestBinDiagnostic]]:
    score_by_id, resolved_score_name = _reference_score_mapping(reference_scores, reference_score_name)
    matched_records = _build_frozen_matched_manifest(
        records,
        reference_scores=score_by_id,
        reference_score_name=resolved_score_name,
        positive_cohort=positive_cohort,
        negative_cohort=negative_cohort,
        num_bins=num_bins,
        manifest_role=manifest_role,
        test_size=test_size,
        random_state=random_state,
        construction_config_hash=construction_config_hash,
    )
    return matched_records


def build_frozen_matched_manifest(
    records: list[SampleAnalysisRecord],
    reference_scores: Mapping[str, float] | Sequence[object],
    *,
    reference_score_name: str | None = None,
    positive_cohort: str = "ambiguous_id",
    negative_cohort: str = "ood",
    num_bins: int = 10,
    manifest_role: str = "eval",
    test_size: float | None = None,
    random_state: int = 7,
    construction_config_hash: str | None = None,
) -> list[MatchedManifestRecord]:
    matched_records, _ = build_frozen_matched_manifest_with_diagnostics(
        records,
        reference_scores,
        reference_score_name=reference_score_name,
        positive_cohort=positive_cohort,
        negative_cohort=negative_cohort,
        num_bins=num_bins,
        manifest_role=manifest_role,
        test_size=test_size,
        random_state=random_state,
        construction_config_hash=construction_config_hash,
    )
    return matched_records


def _build_frozen_matched_manifest(
    records: list[SampleAnalysisRecord],
    *,
    reference_scores: Mapping[str, float],
    reference_score_name: str,
    positive_cohort: str,
    negative_cohort: str,
    num_bins: int,
    manifest_role: str,
    test_size: float | None,
    random_state: int,
    construction_config_hash: str | None,
) -> tuple[list[MatchedManifestRecord], list[MatchedManifestBinDiagnostic]]:
    if positive_cohort == negative_cohort:
        raise ValueError("positive_cohort and negative_cohort must be different.")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")

    cohort_records = [
        record
        for record in records
        if record.cohort_name in {positive_cohort, negative_cohort} and record.sample_id in reference_scores
    ]
    if len({record.cohort_name for record in cohort_records}) < 2:
        raise ValueError("Matched manifest requires scored records from both cohorts.")

    values = np.array([float(reference_scores[record.sample_id]) for record in cohort_records], dtype=np.float64)
    bin_indices = _quantile_bin_indices(values, num_bins)

    binned: dict[tuple[str, int], list[SampleAnalysisRecord]] = defaultdict(list)
    bin_value_ranges: dict[int, list[float]] = defaultdict(list)
    for record, bin_index in zip(cohort_records, bin_indices, strict=True):
        resolved_bin_index = int(bin_index)
        score = float(reference_scores[record.sample_id])
        bin_value_ranges[resolved_bin_index].append(score)
        binned[(record.cohort_name, resolved_bin_index)].append(record)

    config_hash = construction_config_hash or compute_construction_config_hash(
        {
            "positive_cohort": positive_cohort,
            "negative_cohort": negative_cohort,
            "reference_score_name": reference_score_name,
            "num_bins": num_bins,
            "manifest_role": manifest_role,
            "test_size": test_size,
            "random_state": random_state,
        }
    )
    matched_records: list[MatchedManifestRecord] = []
    diagnostics: list[MatchedManifestBinDiagnostic] = []
    for bin_index in range(num_bins):
        positive_records = sorted(
            binned[(positive_cohort, bin_index)],
            key=lambda record: (float(reference_scores[record.sample_id]), record.sample_id),
        )
        negative_records = sorted(
            binned[(negative_cohort, bin_index)],
            key=lambda record: (float(reference_scores[record.sample_id]), record.sample_id),
        )
        pair_count = min(len(positive_records), len(negative_records))
        bin_scores = bin_value_ranges.get(bin_index, [])
        diagnostics.append(
            MatchedManifestBinDiagnostic(
                match_bin_id=f"bin{bin_index:02d}",
                positive_cohort=positive_cohort,
                negative_cohort=negative_cohort,
                positive_count=len(positive_records),
                negative_count=len(negative_records),
                matched_pairs=pair_count,
                unmatched_positive=max(0, len(positive_records) - pair_count),
                unmatched_negative=max(0, len(negative_records) - pair_count),
                score_min=float(min(bin_scores)) if bin_scores else 0.0,
                score_max=float(max(bin_scores)) if bin_scores else 0.0,
            )
        )
        for pair_index in range(pair_count):
            paired_group_id = f"bin{bin_index:02d}-pair{pair_index:05d}"
            for record in (positive_records[pair_index], negative_records[pair_index]):
                matched_records.append(
                    MatchedManifestRecord(
                        sample_id=record.sample_id,
                        cohort_name=record.cohort_name,
                        source_dataset_name=record.source_dataset_name,
                        source_index=record.sample_id,
                        reference_score_name=reference_score_name,
                        reference_score_value=float(reference_scores[record.sample_id]),
                        match_bin_id=f"bin{bin_index:02d}",
                        paired_group_id=paired_group_id,
                        manifest_role=manifest_role,
                        construction_config_hash=config_hash,
                    )
                )

    if not matched_records:
        raise ValueError("No matched records were produced. Check reference scores and binning.")
    role_by_pair_id = _pair_roles(
        [record.paired_group_id for record in matched_records],
        test_size=test_size,
        random_state=random_state,
        default_role=manifest_role,
    )
    matched_records = [
        MatchedManifestRecord(
            sample_id=record.sample_id,
            cohort_name=record.cohort_name,
            source_dataset_name=record.source_dataset_name,
            source_index=record.source_index,
            reference_score_name=record.reference_score_name,
            reference_score_value=record.reference_score_value,
            match_bin_id=record.match_bin_id,
            paired_group_id=record.paired_group_id,
            manifest_role=role_by_pair_id[record.paired_group_id],
            construction_config_hash=record.construction_config_hash,
        )
        for record in matched_records
    ]
    manifest_hash = compute_manifest_hash(matched_records)
    return [
        MatchedManifestRecord(
            sample_id=record.sample_id,
            cohort_name=record.cohort_name,
            source_dataset_name=record.source_dataset_name,
            source_index=record.source_index,
            reference_score_name=record.reference_score_name,
            reference_score_value=record.reference_score_value,
            match_bin_id=record.match_bin_id,
            paired_group_id=record.paired_group_id,
            manifest_role=record.manifest_role,
            construction_config_hash=record.construction_config_hash,
            manifest_hash=manifest_hash,
        )
        for record in matched_records
    ], diagnostics


def summarize_matched_manifest(records: Iterable[MatchedManifestRecord]) -> dict[str, object]:
    materialized = list(records)
    cohort_counts: dict[str, int] = defaultdict(int)
    bin_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    role_counts: dict[str, int] = defaultdict(int)
    paired_group_ids: set[str] = set()
    for record in materialized:
        cohort_counts[record.cohort_name] += 1
        bin_counts[record.match_bin_id][record.cohort_name] += 1
        role_counts[record.manifest_role] += 1
        paired_group_ids.add(record.paired_group_id)
    return {
        "manifest_hash": materialized[0].manifest_hash if materialized else "",
        "construction_config_hash": materialized[0].construction_config_hash if materialized else "",
        "reference_score_name": materialized[0].reference_score_name if materialized else "",
        "cohort_counts": dict(sorted(cohort_counts.items())),
        "bin_counts": {bin_id: dict(sorted(counts.items())) for bin_id, counts in sorted(bin_counts.items())},
        "role_counts": dict(sorted(role_counts.items())),
        "paired_group_count": len(paired_group_ids),
    }


def write_matched_manifest_jsonl(records: Iterable[MatchedManifestRecord], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True))
            handle.write("\n")
    return output


def write_matched_manifest(records: Iterable[MatchedManifestRecord], output_path: str | Path) -> Path:
    return write_matched_manifest_jsonl(records, output_path)


def read_matched_manifest_jsonl(input_path: str | Path) -> list[MatchedManifestRecord]:
    matched_records: list[MatchedManifestRecord] = []
    with Path(input_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                matched_records.append(MatchedManifestRecord.from_dict(json.loads(line)))
    return matched_records


def write_matched_manifest_bin_diagnostics(
    records: Iterable[MatchedManifestBinDiagnostic],
    output_path: str | Path,
) -> Path:
    materialized = list(records)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(materialized[0].to_csv_row().keys()) if materialized else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for record in materialized:
                writer.writerow(record.to_csv_row())
    return output


def write_matched_manifest_summary(records: Iterable[MatchedManifestRecord], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summarize_matched_manifest(records), indent=2, sort_keys=True), encoding="utf-8")
    return output
