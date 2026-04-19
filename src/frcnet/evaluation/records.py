from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json


@dataclass(slots=True)
class SampleAnalysisRecord:
    run_id: str
    protocol_id: str
    sample_id: str
    split_name: str
    cohort_name: str
    source_dataset_name: str
    source_class_label: int | None
    class_label: int
    predicted_class_index: int
    resolution_ratio: float
    unknown_mass: float
    content_entropy: float
    top1_class_mass: float
    top1_content_probability: float
    completion_score_beta_0_1: float
    completion_score_beta_0_5: float
    candidate_class_indices: tuple[int, ...] = ()

    def to_csv_row(self) -> dict[str, str | float | int]:
        return {
            "run_id": self.run_id,
            "protocol_id": self.protocol_id,
            "sample_id": self.sample_id,
            "split_name": self.split_name,
            "cohort_name": self.cohort_name,
            "source_dataset_name": self.source_dataset_name,
            "source_class_label": "" if self.source_class_label is None else self.source_class_label,
            "class_label": self.class_label,
            "predicted_class_index": self.predicted_class_index,
            "resolution_ratio": self.resolution_ratio,
            "unknown_mass": self.unknown_mass,
            "content_entropy": self.content_entropy,
            "top1_class_mass": self.top1_class_mass,
            "top1_content_probability": self.top1_content_probability,
            "completion_score_beta_0_1": self.completion_score_beta_0_1,
            "completion_score_beta_0_5": self.completion_score_beta_0_5,
            "candidate_class_indices_json": json.dumps(list(self.candidate_class_indices)),
        }


@dataclass(slots=True)
class Top1PropositionRecord:
    run_id: str
    protocol_id: str
    sample_id: str
    split_name: str
    cohort_name: str
    proposition_target_type: str
    predicted_class_index: int
    class_label: int
    source_class_label: int | None
    is_top1_correct: bool
    candidate_class_indices: tuple[int, ...] = ()

    def to_csv_row(self) -> dict[str, str | int | bool]:
        return {
            "run_id": self.run_id,
            "protocol_id": self.protocol_id,
            "sample_id": self.sample_id,
            "split_name": self.split_name,
            "cohort_name": self.cohort_name,
            "proposition_target_type": self.proposition_target_type,
            "predicted_class_index": self.predicted_class_index,
            "class_label": self.class_label,
            "source_class_label": "" if self.source_class_label is None else self.source_class_label,
            "is_top1_correct": int(self.is_top1_correct),
            "candidate_class_indices_json": json.dumps(list(self.candidate_class_indices)),
        }


def write_sample_analysis_records(
    records: list[SampleAnalysisRecord],
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(records[0].to_csv_row().keys()) if records else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for record in records:
                writer.writerow(record.to_csv_row())
    return output


def write_top1_proposition_records(
    records: list[Top1PropositionRecord],
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(records[0].to_csv_row().keys()) if records else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for record in records:
                writer.writerow(record.to_csv_row())
    return output


def read_sample_analysis_records(input_path: str | Path) -> list[SampleAnalysisRecord]:
    records: list[SampleAnalysisRecord] = []
    with Path(input_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                SampleAnalysisRecord(
                    run_id=row["run_id"],
                    protocol_id=row["protocol_id"],
                    sample_id=row["sample_id"],
                    split_name=row["split_name"],
                    cohort_name=row["cohort_name"],
                    source_dataset_name=row["source_dataset_name"],
                    source_class_label=None if row["source_class_label"] == "" else int(row["source_class_label"]),
                    class_label=int(row["class_label"]),
                    predicted_class_index=int(row["predicted_class_index"]),
                    resolution_ratio=float(row["resolution_ratio"]),
                    unknown_mass=float(row["unknown_mass"]),
                    content_entropy=float(row["content_entropy"]),
                    top1_class_mass=float(row["top1_class_mass"]),
                    top1_content_probability=float(row["top1_content_probability"]),
                    completion_score_beta_0_1=float(row["completion_score_beta_0_1"]),
                    completion_score_beta_0_5=float(row["completion_score_beta_0_5"]),
                    candidate_class_indices=tuple(json.loads(row["candidate_class_indices_json"])),
                )
            )
    return records

