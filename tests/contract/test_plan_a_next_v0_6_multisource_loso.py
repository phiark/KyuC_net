from __future__ import annotations

from pathlib import Path

import yaml

from frcnet.data import (
    SampleManifestRecord,
    SourceBalancedBatchSampler,
    build_plan_a_manifest,
    read_manifest_jsonl,
    source_fingerprint_overlap,
    validate_manifest_records,
    write_manifest_jsonl,
)


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


def _v0_6_sources(*, include_cifar100: bool) -> dict[str, object]:
    sources: dict[str, object] = {
        "cifar10": _TargetsOnlyDataset([label for label in range(10) for _ in range(5000)]),
        "svhn": _LabelsOnlyDataset([index % 10 for index in range(12000)]),
        "dtd": _TargetsOnlyDataset([index % 47 for index in range(3000)]),
        "lsun_resize": _TargetsOnlyDataset([-1 for _ in range(3000)]),
        "gaussian_noise": _TargetsOnlyDataset([-1 for _ in range(10000)]),
    }
    if include_cifar100:
        sources["cifar100"] = _TargetsOnlyDataset([index % 100 for index in range(2000)])
    return sources


def test_v0_6_train_manifest_uses_multiple_seen_unknown_sources_and_excludes_cifar100():
    records = validate_manifest_records(
        build_plan_a_manifest(
            _load_protocol("configs/protocol/plan_a_next_v0_6_train.yaml"),
            _v0_6_sources(include_cifar100=False),
        )
    )

    unknown_counts: dict[str, int] = {}
    for record in records:
        if record.cohort_name == "unknown_supervision":
            unknown_counts[record.source_dataset_name] = unknown_counts.get(record.source_dataset_name, 0) + 1

    assert unknown_counts == {
        "dtd": 1800,
        "gaussian_noise": 1800,
        "lsun_resize": 1800,
        "svhn": 1800,
    }
    assert "cifar100" not in {record.source_dataset_name for record in records}


def test_v0_6_validation_and_final_source_fingerprints_do_not_overlap():
    validation_records = validate_manifest_records(
        build_plan_a_manifest(
            _load_protocol("configs/protocol/plan_a_next_v0_6_validation.yaml"),
            _v0_6_sources(include_cifar100=False),
        )
    )
    final_records = validate_manifest_records(
        build_plan_a_manifest(
            _load_protocol("configs/protocol/plan_a_next_v0_6_test_cifar100_holdout.yaml"),
            _v0_6_sources(include_cifar100=True),
        )
    )

    assert source_fingerprint_overlap(validation_records, final_records) == set()


def test_v0_6_final_manifest_marks_cifar100_as_unseen_holdout():
    records = build_plan_a_manifest(
        _load_protocol("configs/protocol/plan_a_next_v0_6_test_cifar100_holdout.yaml"),
        _v0_6_sources(include_cifar100=True),
    )
    roles = {
        (record.source_dataset_name, record.source_role)
        for record in records
        if record.cohort_name == "ood"
    }

    assert ("cifar100", "unseen_ood_source") in roles
    assert ("svhn", "seen_source_ood") in roles
    assert ("dtd", "seen_source_ood") in roles
    assert ("lsun_resize", "seen_source_ood") in roles
    assert ("gaussian_noise", "seen_source_ood") in roles


def test_v0_6_manifest_roundtrip_preserves_source_domain_fields(tmp_path: Path):
    records = build_plan_a_manifest(
        _load_protocol("configs/protocol/plan_a_next_v0_6_test_cifar100_holdout.yaml"),
        _v0_6_sources(include_cifar100=True),
    )
    output_path = write_manifest_jsonl(records, tmp_path / "manifest.jsonl")

    restored = read_manifest_jsonl(output_path)
    cifar100_record = next(record for record in restored if record.source_dataset_name == "cifar100")

    assert cifar100_record.source_domain_name == "cifar100"
    assert cifar100_record.source_domain_label == 5


def _record(sample_id: str, cohort: str, source: str) -> SampleManifestRecord:
    return SampleManifestRecord(
        protocol_id="sampler-test",
        sample_id=sample_id,
        split_name="train",
        cohort_name=cohort,
        source_dataset_name=source,
        source_sample_indices=(0,),
        source_class_label=0,
        class_label=0 if cohort in {"easy_id", "hard_id"} else -1,
    )


def test_source_balanced_sampler_balances_ood_sources_per_batch():
    records = []
    records.extend(_record(f"easy-{index}", "easy_id", "cifar10") for index in range(8))
    records.extend(_record(f"hard-{index}", "hard_id", "cifar10") for index in range(8))
    records.extend(_record(f"amb-{index}", "ambiguous_id", "cifar10") for index in range(8))
    for source in ("svhn", "dtd", "lsun_resize", "gaussian_noise"):
        records.extend(_record(f"{source}-{index}", "unknown_supervision", source) for index in range(8))

    sampler = SourceBalancedBatchSampler(records, batch_size=16, batches_per_epoch=1, seed=7, shuffle=False)
    batch_indices = next(iter(sampler))
    batch_records = [records[index] for index in batch_indices]

    assert sum(record.cohort_name in {"easy_id", "hard_id"} for record in batch_records) == 4
    assert sum(record.cohort_name == "ambiguous_id" for record in batch_records) == 4
    source_counts: dict[str, int] = {}
    for record in batch_records:
        if record.cohort_name == "unknown_supervision":
            source_counts[record.source_dataset_name] = source_counts.get(record.source_dataset_name, 0) + 1
    assert source_counts == {"dtd": 2, "gaussian_noise": 2, "lsun_resize": 2, "svhn": 2}
