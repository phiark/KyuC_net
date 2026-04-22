from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json

import torch

from frcnet.utils import resolution_entropy, ternary_entropy_from_masses

DEFAULT_MODEL_FAMILY = "frcnet_explicit_unknown"


@dataclass(slots=True)
class SampleAnalysisRecord:
    model_family: str
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
    resolution_weighted_content_entropy: float
    resolution_entropy: float
    top1_class_mass: float
    proposition_truth_mass: float
    proposition_false_mass: float
    proposition_unknown_mass: float
    proposition_truth_ratio: float
    ternary_entropy: float
    auxiliary_top1_content_probability: float
    completion_score_beta_0_1: float
    completion_score_beta_0_25: float
    completion_score_beta_0_5: float
    completion_score_beta_0_75: float
    candidate_class_indices: tuple[int, ...] = ()

    @property
    def top1_content_probability(self) -> float:
        return self.auxiliary_top1_content_probability

    def to_csv_row(self) -> dict[str, str | float | int]:
        return {
            "model_family": self.model_family,
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
            "resolution_weighted_content_entropy": self.resolution_weighted_content_entropy,
            "resolution_entropy": self.resolution_entropy,
            "top1_class_mass": self.top1_class_mass,
            "proposition_truth_mass": self.proposition_truth_mass,
            "proposition_false_mass": self.proposition_false_mass,
            "proposition_unknown_mass": self.proposition_unknown_mass,
            "proposition_truth_ratio": self.proposition_truth_ratio,
            "ternary_entropy": self.ternary_entropy,
            "auxiliary_top1_content_probability": self.auxiliary_top1_content_probability,
            "completion_score_beta_0_1": self.completion_score_beta_0_1,
            "completion_score_beta_0_25": self.completion_score_beta_0_25,
            "completion_score_beta_0_5": self.completion_score_beta_0_5,
            "completion_score_beta_0_75": self.completion_score_beta_0_75,
            "candidate_class_indices_json": json.dumps(list(self.candidate_class_indices)),
        }


@dataclass(slots=True)
class Top1PropositionRecord:
    model_family: str
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
    proposition_truth_mass: float
    proposition_false_mass: float
    proposition_unknown_mass: float
    proposition_truth_ratio: float
    resolution_entropy: float
    ternary_entropy: float
    auxiliary_top1_content_probability: float
    candidate_class_indices: tuple[int, ...] = ()

    @property
    def top1_content_probability(self) -> float:
        return self.auxiliary_top1_content_probability

    def to_csv_row(self) -> dict[str, str | int | bool | float]:
        return {
            "model_family": self.model_family,
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
            "proposition_truth_mass": self.proposition_truth_mass,
            "proposition_false_mass": self.proposition_false_mass,
            "proposition_unknown_mass": self.proposition_unknown_mass,
            "proposition_truth_ratio": self.proposition_truth_ratio,
            "resolution_entropy": self.resolution_entropy,
            "ternary_entropy": self.ternary_entropy,
            "auxiliary_top1_content_probability": self.auxiliary_top1_content_probability,
            "candidate_class_indices_json": json.dumps(list(self.candidate_class_indices)),
        }


@dataclass(slots=True)
class AnalysisExportSummary:
    run_id: str
    protocol_id: str
    analysis_path: str
    checkpoint_path: str | None
    manifest_snapshot_path: str
    model_config_snapshot_path: str
    proposition_path: str
    checkpoint_selection_summary_path: str | None = None
    model_family: str = DEFAULT_MODEL_FAMILY
    integrity_overrides: tuple[str, ...] = ()
    sidecar_resolution_mode: str = "analysis_summary"

    def to_dict(self) -> dict[str, str | list[str] | None]:
        return {
            "run_id": self.run_id,
            "protocol_id": self.protocol_id,
            "analysis_path": self.analysis_path,
            "checkpoint_path": self.checkpoint_path,
            "manifest_snapshot_path": self.manifest_snapshot_path,
            "model_config_snapshot_path": self.model_config_snapshot_path,
            "proposition_path": self.proposition_path,
            "checkpoint_selection_summary_path": self.checkpoint_selection_summary_path,
            "model_family": self.model_family,
            "integrity_overrides": list(self.integrity_overrides),
            "sidecar_resolution_mode": self.sidecar_resolution_mode,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "AnalysisExportSummary":
        return cls(
            run_id=str(payload["run_id"]),
            protocol_id=str(payload["protocol_id"]),
            analysis_path=str(payload["analysis_path"]),
            checkpoint_path=None if payload.get("checkpoint_path") in {None, ""} else str(payload["checkpoint_path"]),
            manifest_snapshot_path=str(payload["manifest_snapshot_path"]),
            model_config_snapshot_path=str(payload["model_config_snapshot_path"]),
            proposition_path=str(payload["proposition_path"]),
            checkpoint_selection_summary_path=None
            if payload.get("checkpoint_selection_summary_path") in {None, ""}
            else str(payload["checkpoint_selection_summary_path"]),
            model_family=str(payload.get("model_family", DEFAULT_MODEL_FAMILY)),
            integrity_overrides=tuple(str(value) for value in payload.get("integrity_overrides", [])),
            sidecar_resolution_mode=str(payload.get("sidecar_resolution_mode", "analysis_summary")),
        )


def _read_float(row: dict[str, str], key: str, default: float) -> float:
    value = row.get(key, "")
    if value in {"", None}:
        return default
    return float(value)


def _fallback_proposition_fields(
    *,
    resolution_ratio_value: float,
    unknown_mass_value: float,
    top1_class_mass_value: float,
    auxiliary_top1_content_probability_value: float,
) -> tuple[float, float, float, float, float]:
    proposition_unknown_mass = unknown_mass_value
    proposition_truth_mass = top1_class_mass_value
    proposition_false_mass = max(0.0, 1.0 - proposition_truth_mass - proposition_unknown_mass)
    proposition_truth_ratio = auxiliary_top1_content_probability_value
    resolution_entropy_value = float(
        resolution_entropy(torch.tensor([resolution_ratio_value], dtype=torch.float32))[0].item()
    )
    ternary_entropy_value = float(
        ternary_entropy_from_masses(
            torch.tensor([proposition_truth_mass], dtype=torch.float32),
            torch.tensor([proposition_false_mass], dtype=torch.float32),
            torch.tensor([proposition_unknown_mass], dtype=torch.float32),
        )[0].item()
    )
    return (
        proposition_truth_mass,
        proposition_false_mass,
        proposition_truth_ratio,
        resolution_entropy_value,
        ternary_entropy_value,
    )


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
            top1_class_mass = float(row["top1_class_mass"])
            unknown_mass = float(row["unknown_mass"])
            content_entropy_value = float(row["content_entropy"])
            resolution_ratio_value = float(row["resolution_ratio"])
            auxiliary_top1_content_probability = _read_float(
                row,
                "auxiliary_top1_content_probability",
                _read_float(row, "top1_content_probability", 0.0),
            )
            (
                fallback_truth_mass,
                fallback_false_mass,
                fallback_truth_ratio,
                fallback_resolution_entropy,
                fallback_ternary_entropy,
            ) = _fallback_proposition_fields(
                resolution_ratio_value=resolution_ratio_value,
                unknown_mass_value=unknown_mass,
                top1_class_mass_value=top1_class_mass,
                auxiliary_top1_content_probability_value=auxiliary_top1_content_probability,
            )
            records.append(
                SampleAnalysisRecord(
                    model_family=str(row.get("model_family", DEFAULT_MODEL_FAMILY)),
                    run_id=row["run_id"],
                    protocol_id=row["protocol_id"],
                    sample_id=row["sample_id"],
                    split_name=row["split_name"],
                    cohort_name=row["cohort_name"],
                    source_dataset_name=row["source_dataset_name"],
                    source_class_label=None if row["source_class_label"] == "" else int(row["source_class_label"]),
                    class_label=int(row["class_label"]),
                    predicted_class_index=int(row["predicted_class_index"]),
                    resolution_ratio=resolution_ratio_value,
                    unknown_mass=unknown_mass,
                    content_entropy=content_entropy_value,
                    resolution_weighted_content_entropy=float(
                        row.get("resolution_weighted_content_entropy", resolution_ratio_value * content_entropy_value)
                    ),
                    resolution_entropy=_read_float(row, "resolution_entropy", fallback_resolution_entropy),
                    top1_class_mass=top1_class_mass,
                    proposition_truth_mass=_read_float(row, "proposition_truth_mass", fallback_truth_mass),
                    proposition_false_mass=_read_float(row, "proposition_false_mass", fallback_false_mass),
                    proposition_unknown_mass=_read_float(row, "proposition_unknown_mass", unknown_mass),
                    proposition_truth_ratio=_read_float(row, "proposition_truth_ratio", fallback_truth_ratio),
                    ternary_entropy=_read_float(row, "ternary_entropy", fallback_ternary_entropy),
                    auxiliary_top1_content_probability=auxiliary_top1_content_probability,
                    completion_score_beta_0_1=float(row["completion_score_beta_0_1"]),
                    completion_score_beta_0_25=float(
                        row.get("completion_score_beta_0_25", top1_class_mass + (0.25 * unknown_mass))
                    ),
                    completion_score_beta_0_5=float(row["completion_score_beta_0_5"]),
                    completion_score_beta_0_75=float(
                        row.get("completion_score_beta_0_75", top1_class_mass + (0.75 * unknown_mass))
                    ),
                    candidate_class_indices=tuple(json.loads(row["candidate_class_indices_json"])),
                )
            )
    return records


def read_top1_proposition_records(input_path: str | Path) -> list[Top1PropositionRecord]:
    records: list[Top1PropositionRecord] = []
    with Path(input_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            auxiliary_top1_content_probability = _read_float(
                row,
                "auxiliary_top1_content_probability",
                _read_float(row, "top1_content_probability", 0.0),
            )
            resolution_entropy_value = _read_float(row, "resolution_entropy", 0.0)
            truth_mass = _read_float(row, "proposition_truth_mass", 0.0)
            false_mass = _read_float(row, "proposition_false_mass", 0.0)
            unknown_mass = _read_float(row, "proposition_unknown_mass", 0.0)
            ternary_entropy_value = _read_float(
                row,
                "ternary_entropy",
                float(
                    ternary_entropy_from_masses(
                        torch.tensor([truth_mass], dtype=torch.float32),
                        torch.tensor([false_mass], dtype=torch.float32),
                        torch.tensor([unknown_mass], dtype=torch.float32),
                    )[0].item()
                ),
            )
            records.append(
                Top1PropositionRecord(
                    model_family=str(row.get("model_family", DEFAULT_MODEL_FAMILY)),
                    run_id=row["run_id"],
                    protocol_id=row["protocol_id"],
                    sample_id=row["sample_id"],
                    split_name=row["split_name"],
                    cohort_name=row["cohort_name"],
                    proposition_target_type=row["proposition_target_type"],
                    predicted_class_index=int(row["predicted_class_index"]),
                    class_label=int(row["class_label"]),
                    source_class_label=None if row["source_class_label"] == "" else int(row["source_class_label"]),
                    is_top1_correct=bool(int(row["is_top1_correct"])),
                    proposition_truth_mass=truth_mass,
                    proposition_false_mass=false_mass,
                    proposition_unknown_mass=unknown_mass,
                    proposition_truth_ratio=_read_float(row, "proposition_truth_ratio", auxiliary_top1_content_probability),
                    resolution_entropy=resolution_entropy_value,
                    ternary_entropy=ternary_entropy_value,
                    auxiliary_top1_content_probability=auxiliary_top1_content_probability,
                    candidate_class_indices=tuple(json.loads(row["candidate_class_indices_json"])),
                )
            )
    return records


def write_analysis_export_summary(summary: AnalysisExportSummary, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return output


def read_analysis_export_summary(input_path: str | Path) -> AnalysisExportSummary:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    return AnalysisExportSummary.from_dict(payload)
