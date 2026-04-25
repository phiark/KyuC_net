from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

from frcnet.evaluation.records import SampleAnalysisRecord
from frcnet.evaluation.reference_baselines import ReferenceScoreRecord


@dataclass(slots=True)
class MatchedManifestRecord:
    sample_id: str
    cohort_name: str
    reference_score_name: str
    reference_score_value: float
    match_bin_id: str
    manifest_role: str
    paired_group_id: str
    manifest_hash: str
    construction_config_hash: str
    source_dataset_name: str = ""
    source_index: int | None = None

    def to_dict(self) -> dict[str, str | float | int | None]:
        return {
            "sample_id": self.sample_id,
            "cohort_name": self.cohort_name,
            "source_dataset_name": self.source_dataset_name,
            "source_index": self.source_index,
            "reference_score_name": self.reference_score_name,
            "reference_score_value": self.reference_score_value,
            "match_bin_id": self.match_bin_id,
            "manifest_role": self.manifest_role,
            "paired_group_id": self.paired_group_id,
            "manifest_hash": self.manifest_hash,
            "construction_config_hash": self.construction_config_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "MatchedManifestRecord":
        return cls(
            sample_id=str(payload["sample_id"]),
            cohort_name=str(payload["cohort_name"]),
            source_dataset_name=str(payload.get("source_dataset_name", "")),
            source_index=None if payload.get("source_index") in {"", None} else int(payload["source_index"]),
            reference_score_name=str(payload["reference_score_name"]),
            reference_score_value=float(payload["reference_score_value"]),
            match_bin_id=str(payload["match_bin_id"]),
            manifest_role=str(payload["manifest_role"]),
            paired_group_id=str(payload["paired_group_id"]),
            manifest_hash=str(payload.get("manifest_hash", "")),
            construction_config_hash=str(payload["construction_config_hash"]),
        )


@dataclass(slots=True)
class MatchedManifestBinDiagnostic:
    match_bin_id: str
    positive_count: int
    negative_count: int
    matched_pairs: int
    unmatched_positive: int
    unmatched_negative: int
    reference_score_min: float
    reference_score_max: float

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "match_bin_id": self.match_bin_id,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "matched_pairs": self.matched_pairs,
            "unmatched_positive": self.unmatched_positive,
            "unmatched_negative": self.unmatched_negative,
            "reference_score_min": self.reference_score_min,
            "reference_score_max": self.reference_score_max,
        }


def construction_config_hash(config: Mapping[str, object]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_matched_manifest_hash(records: Iterable[MatchedManifestRecord]) -> str:
    payload_rows = []
    for record in records:
        row = record.to_dict()
        row["manifest_hash"] = ""
        payload_rows.append(row)
    payload = json.dumps(sorted(payload_rows, key=lambda row: str(row["sample_id"])), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def with_manifest_hash(records: Iterable[MatchedManifestRecord]) -> list[MatchedManifestRecord]:
    materialized = list(records)
    manifest_hash = compute_matched_manifest_hash(materialized)
    return [replace(record, manifest_hash=manifest_hash) for record in materialized]


def write_matched_manifest(records: Iterable[MatchedManifestRecord], output_path: str | Path) -> Path:
    materialized = with_manifest_hash(records)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in materialized:
            handle.write(json.dumps(record.to_dict(), sort_keys=True))
            handle.write("\n")
    return output


def read_matched_manifest(input_path: str | Path) -> list[MatchedManifestRecord]:
    records: list[MatchedManifestRecord] = []
    with Path(input_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(MatchedManifestRecord.from_dict(json.loads(line)))
    return records


def _assign_quantile_bin_ids(
    scored_records: list[tuple[SampleAnalysisRecord, ReferenceScoreRecord]],
    *,
    num_bins: int,
) -> dict[str, str]:
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")
    ordered = sorted(
        scored_records,
        key=lambda item: (
            float(item[1].reference_score_value),
            item[0].cohort_name,
            item[0].sample_id,
        ),
    )
    sample_to_bin: dict[str, str] = {}
    total_count = len(ordered)
    for rank, (sample_record, _reference_record) in enumerate(ordered):
        bin_index = min(num_bins - 1, (rank * num_bins) // total_count)
        sample_to_bin[sample_record.sample_id] = f"bin-{bin_index:02d}"
    return sample_to_bin


def _select_test_pair_ids(
    paired_group_ids: list[str],
    *,
    test_size: float,
    random_state: int,
) -> set[str]:
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be within (0, 1).")
    import random

    unique_ids = sorted(set(paired_group_ids))
    rng = random.Random(random_state)
    shuffled_ids = list(unique_ids)
    rng.shuffle(shuffled_ids)
    if len(shuffled_ids) < 2:
        test_count = len(shuffled_ids)
    else:
        test_count = int(round(len(shuffled_ids) * test_size))
        test_count = min(len(shuffled_ids) - 1, max(1, test_count))
    return set(shuffled_ids[:test_count])


def build_frozen_matched_manifest(
    sample_analysis_records: Iterable[SampleAnalysisRecord],
    reference_score_records: Iterable[ReferenceScoreRecord],
    *,
    positive_cohort: str = "ambiguous_id",
    negative_cohort: str = "ood",
    reference_score_name: str = "softmax_entropy",
    num_bins: int = 10,
    test_size: float = 0.3,
    random_state: int = 7,
) -> tuple[list[MatchedManifestRecord], list[MatchedManifestBinDiagnostic]]:
    if positive_cohort == negative_cohort:
        raise ValueError("positive_cohort and negative_cohort must be different.")
    if num_bins <= 0:
        raise ValueError("num_bins must be positive.")

    samples_by_id = {
        record.sample_id: record
        for record in sample_analysis_records
        if record.cohort_name in {positive_cohort, negative_cohort}
    }
    references_by_id = {
        record.sample_id: record
        for record in reference_score_records
        if record.reference_score_name == reference_score_name and record.sample_id in samples_by_id
    }
    scored_records = [
        (sample_record, references_by_id[sample_record.sample_id])
        for sample_record in samples_by_id.values()
        if sample_record.sample_id in references_by_id
    ]
    if not scored_records:
        raise ValueError("No overlapping sample/reference records were available for frozen matching.")

    cohorts = {sample_record.cohort_name for sample_record, _reference_record in scored_records}
    if {positive_cohort, negative_cohort} - cohorts:
        raise ValueError("Frozen manifest construction requires both positive and negative cohorts.")

    config_payload = {
        "positive_cohort": positive_cohort,
        "negative_cohort": negative_cohort,
        "reference_score_name": reference_score_name,
        "num_bins": num_bins,
        "test_size": test_size,
        "random_state": random_state,
    }
    config_hash = construction_config_hash(config_payload)
    sample_to_bin = _assign_quantile_bin_ids(scored_records, num_bins=num_bins)

    by_bin: dict[str, list[tuple[SampleAnalysisRecord, ReferenceScoreRecord]]] = defaultdict(list)
    for sample_record, reference_record in scored_records:
        by_bin[sample_to_bin[sample_record.sample_id]].append((sample_record, reference_record))

    provisional_records: list[MatchedManifestRecord] = []
    diagnostics: list[MatchedManifestBinDiagnostic] = []
    for match_bin_id in sorted(by_bin):
        bin_items = by_bin[match_bin_id]
        positive_items = sorted(
            [item for item in bin_items if item[0].cohort_name == positive_cohort],
            key=lambda item: (float(item[1].reference_score_value), item[0].sample_id),
        )
        negative_items = sorted(
            [item for item in bin_items if item[0].cohort_name == negative_cohort],
            key=lambda item: (float(item[1].reference_score_value), item[0].sample_id),
        )
        matched_pairs = min(len(positive_items), len(negative_items))
        reference_values = [float(item[1].reference_score_value) for item in bin_items]
        diagnostics.append(
            MatchedManifestBinDiagnostic(
                match_bin_id=match_bin_id,
                positive_count=len(positive_items),
                negative_count=len(negative_items),
                matched_pairs=matched_pairs,
                unmatched_positive=len(positive_items) - matched_pairs,
                unmatched_negative=len(negative_items) - matched_pairs,
                reference_score_min=min(reference_values),
                reference_score_max=max(reference_values),
            )
        )
        for pair_offset in range(matched_pairs):
            paired_group_id = f"{match_bin_id}-pair-{pair_offset:06d}"
            for sample_record, reference_record in (
                positive_items[pair_offset],
                negative_items[pair_offset],
            ):
                provisional_records.append(
                    MatchedManifestRecord(
                        sample_id=sample_record.sample_id,
                        cohort_name=sample_record.cohort_name,
                        source_dataset_name=sample_record.source_dataset_name,
                        source_index=None,
                        reference_score_name=reference_record.reference_score_name,
                        reference_score_value=float(reference_record.reference_score_value),
                        match_bin_id=match_bin_id,
                        manifest_role="train",
                        paired_group_id=paired_group_id,
                        manifest_hash="",
                        construction_config_hash=config_hash,
                    )
                )

    if not provisional_records:
        raise ValueError("Frozen matching produced zero matched pairs.")

    test_pair_ids = _select_test_pair_ids(
        [record.paired_group_id for record in provisional_records],
        test_size=test_size,
        random_state=random_state,
    )
    role_assigned_records = [
        replace(record, manifest_role="test" if record.paired_group_id in test_pair_ids else "train")
        for record in provisional_records
    ]
    return with_manifest_hash(role_assigned_records), diagnostics


def write_matched_manifest_bin_diagnostics(
    diagnostics: Iterable[MatchedManifestBinDiagnostic],
    output_path: str | Path,
) -> Path:
    import csv

    materialized = list(diagnostics)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(materialized[0].to_dict().keys()) if materialized else []
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for diagnostic in materialized:
                writer.writerow(diagnostic.to_dict())
    return output
