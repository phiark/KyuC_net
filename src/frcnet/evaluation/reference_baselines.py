from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping

import torch

from frcnet.utils import content_entropy

SOFTMAX_CE_REFERENCE_FAMILY = "softmax_ce_reference"
SOFTMAX_ENTROPY_SCORE_NAME = "softmax_entropy"


@dataclass(slots=True)
class ReferenceScoreRecord:
    sample_id: str
    split_name: str
    cohort_name: str
    source_dataset_name: str
    reference_model_family: str
    reference_run_id: str
    reference_score_name: str
    reference_score_value: float

    def to_dict(self) -> dict[str, str | float]:
        return {
            "sample_id": self.sample_id,
            "split_name": self.split_name,
            "cohort_name": self.cohort_name,
            "source_dataset_name": self.source_dataset_name,
            "reference_model_family": self.reference_model_family,
            "reference_run_id": self.reference_run_id,
            "reference_score_name": self.reference_score_name,
            "reference_score_value": self.reference_score_value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ReferenceScoreRecord":
        return cls(
            sample_id=str(payload["sample_id"]),
            split_name=str(payload["split_name"]),
            cohort_name=str(payload["cohort_name"]),
            source_dataset_name=str(payload["source_dataset_name"]),
            reference_model_family=str(payload["reference_model_family"]),
            reference_run_id=str(payload["reference_run_id"]),
            reference_score_name=str(payload["reference_score_name"]),
            reference_score_value=float(payload["reference_score_value"]),
        )


def softmax_entropy_reference_scores(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=-1)
    return content_entropy(probabilities)


def write_reference_score_records(
    records: Iterable[ReferenceScoreRecord],
    output_path: str | Path,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True))
            handle.write("\n")
    return output


def read_reference_score_records(input_path: str | Path) -> list[ReferenceScoreRecord]:
    records: list[ReferenceScoreRecord] = []
    with Path(input_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(ReferenceScoreRecord.from_dict(json.loads(line)))
    return records
