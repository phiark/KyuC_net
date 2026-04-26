from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import yaml

from frcnet.data import (
    build_plan_a_manifest,
    read_manifest_jsonl,
    source_fingerprint_overlap,
    validate_manifest_records,
    write_manifest_jsonl,
)
from frcnet.evaluation import MatchedManifestRecord, compute_matched_manifest_hash


class _TargetsOnlyDataset:
    def __init__(self, labels: list[int]) -> None:
        self.targets = labels

    def __len__(self) -> int:
        return len(self.targets)


class _LabelsOnlyDataset:
    def __init__(self, labels: list[int]) -> None:
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)


def _load_protocol(path: str) -> dict:
    return dict(yaml.safe_load(Path(path).read_text(encoding="utf-8"))["protocol"])


def _source_datasets(include_cifar100: bool = False):
    datasets = {
        "cifar10": _TargetsOnlyDataset([label for label in range(10) for _ in range(5000)]),
        "svhn": _LabelsOnlyDataset([index % 10 for index in range(8000)]),
    }
    if include_cifar100:
        datasets["cifar100"] = _TargetsOnlyDataset([index % 100 for index in range(1000)])
    return datasets


def test_v0_5_validation_and_final_source_partitions_do_not_overlap():
    validation_protocol = _load_protocol("configs/protocol/plan_a_next_v0_5_validation.yaml")
    final_protocol = _load_protocol("configs/protocol/plan_a_next_v0_5_test.yaml")

    validation_records = validate_manifest_records(
        build_plan_a_manifest(validation_protocol, _source_datasets(include_cifar100=False))
    )
    final_records = validate_manifest_records(
        build_plan_a_manifest(final_protocol, _source_datasets(include_cifar100=True))
    )

    assert source_fingerprint_overlap(validation_records, final_records) == set()


def test_v0_5_cifar100_is_final_only_unseen_ood_source():
    train_records = build_plan_a_manifest(
        _load_protocol("configs/protocol/plan_a_next_v0_5_train.yaml"),
        _source_datasets(include_cifar100=False),
    )
    validation_records = build_plan_a_manifest(
        _load_protocol("configs/protocol/plan_a_next_v0_5_validation.yaml"),
        _source_datasets(include_cifar100=False),
    )
    final_records = build_plan_a_manifest(
        _load_protocol("configs/protocol/plan_a_next_v0_5_test.yaml"),
        _source_datasets(include_cifar100=True),
    )

    assert "cifar100" not in {record.source_dataset_name for record in train_records}
    assert "cifar100" not in {record.source_dataset_name for record in validation_records}
    final_ood_roles = {
        (record.source_dataset_name, record.source_role)
        for record in final_records
        if record.cohort_name == "ood"
    }
    assert ("svhn", "seen_source_ood") in final_ood_roles
    assert ("cifar100", "unseen_ood_source") in final_ood_roles


def test_manifest_roundtrip_preserves_v0_5_source_provenance(tmp_path: Path):
    final_records = build_plan_a_manifest(
        _load_protocol("configs/protocol/plan_a_next_v0_5_test.yaml"),
        _source_datasets(include_cifar100=True),
    )
    output_path = write_manifest_jsonl(final_records, tmp_path / "manifest.jsonl")

    restored_records = read_manifest_jsonl(output_path)
    cifar100_record = next(record for record in restored_records if record.source_dataset_name == "cifar100")

    assert cifar100_record.source_dataset_split == "test"
    assert cifar100_record.source_role == "unseen_ood_source"
    assert cifar100_record.source_partition_name == "cifar100_test_global_0000_1000"
    assert cifar100_record.source_sample_indices


def test_matched_manifest_hash_is_sensitive_to_source_role():
    record = MatchedManifestRecord(
        sample_id="sample-a",
        cohort_name="ood",
        reference_score_name="softmax_entropy",
        reference_score_value=0.5,
        match_bin_id="bin-00",
        manifest_role="test",
        paired_group_id="pair-000",
        manifest_hash="",
        construction_config_hash="config",
        source_dataset_name="svhn",
        source_dataset_split="test",
        source_role="seen_source_ood",
        source_partition_name="svhn_test_global_1000_3000",
    )

    changed_record = replace(record, source_role="unseen_ood_source")

    assert compute_matched_manifest_hash([record]) != compute_matched_manifest_hash([changed_record])
