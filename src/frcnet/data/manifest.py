from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(slots=True)
class SampleManifestRecord:
    protocol_id: str
    sample_id: str
    split_name: str
    cohort_name: str
    source_dataset_name: str
    source_sample_indices: tuple[int, ...]
    source_class_label: int | None
    class_label: int
    candidate_class_indices: tuple[int, ...] = ()
    augmentation_recipe: str = "identity"
    augmentation_parameters: dict[str, Any] = field(default_factory=dict)
    source_class_labels: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "sample_id": self.sample_id,
            "split_name": self.split_name,
            "cohort_name": self.cohort_name,
            "source_dataset_name": self.source_dataset_name,
            "source_sample_indices": list(self.source_sample_indices),
            "source_class_label": self.source_class_label,
            "class_label": self.class_label,
            "candidate_class_indices": list(self.candidate_class_indices),
            "augmentation_recipe": self.augmentation_recipe,
            "augmentation_parameters": self.augmentation_parameters,
            "source_class_labels": list(self.source_class_labels),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SampleManifestRecord":
        return cls(
            protocol_id=payload["protocol_id"],
            sample_id=payload["sample_id"],
            split_name=payload["split_name"],
            cohort_name=payload["cohort_name"],
            source_dataset_name=payload["source_dataset_name"],
            source_sample_indices=tuple(payload["source_sample_indices"]),
            source_class_label=payload.get("source_class_label"),
            class_label=payload["class_label"],
            candidate_class_indices=tuple(payload.get("candidate_class_indices", [])),
            augmentation_recipe=payload.get("augmentation_recipe", "identity"),
            augmentation_parameters=dict(payload.get("augmentation_parameters", {})),
            source_class_labels=tuple(payload.get("source_class_labels", [])),
        )


def write_manifest_jsonl(records: Iterable[SampleManifestRecord], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True))
            handle.write("\n")
    return output


def read_manifest_jsonl(input_path: str | Path) -> list[SampleManifestRecord]:
    manifest_path = Path(input_path)
    records: list[SampleManifestRecord] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(SampleManifestRecord.from_dict(json.loads(line)))
    return records

