from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping


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
