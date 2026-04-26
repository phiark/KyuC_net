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
from frcnet.workflows.plan_a import enforce_zero_source_overlap


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


def _load_study(path: str) -> dict:
    return dict(yaml.safe_load(Path(path).read_text(encoding="utf-8"))["study"])


def _load_model(path: str) -> dict:
    return dict(yaml.safe_load(Path(path).read_text(encoding="utf-8"))["model"])


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


def _v0_6b_sources(*, include_cifar100: bool) -> dict[str, object]:
    sources = _v0_6_sources(include_cifar100=include_cifar100)
    sources["tiny_imagenet"] = _TargetsOnlyDataset([index % 200 for index in range(3000)])
    return sources


def _v0_6c_sources() -> dict[str, object]:
    sources = _v0_6b_sources(include_cifar100=True)
    sources["cifar100"] = _TargetsOnlyDataset([index % 100 for index in range(10000)])
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


def test_v0_6b_train_manifest_adds_tiny_imagenet_and_excludes_cifar100():
    records = validate_manifest_records(
        build_plan_a_manifest(
            _load_protocol("configs/protocol/plan_a_next_v0_6b_train.yaml"),
            _v0_6b_sources(include_cifar100=False),
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
        "tiny_imagenet": 1800,
    }
    assert "cifar100" not in {record.source_dataset_name for record in records}


def test_v0_6b_validation_and_final_source_fingerprints_do_not_overlap():
    validation_records = validate_manifest_records(
        build_plan_a_manifest(
            _load_protocol("configs/protocol/plan_a_next_v0_6b_validation.yaml"),
            _v0_6b_sources(include_cifar100=False),
        )
    )
    final_records = validate_manifest_records(
        build_plan_a_manifest(
            _load_protocol("configs/protocol/plan_a_next_v0_6b_test_cifar100_holdout.yaml"),
            _v0_6b_sources(include_cifar100=True),
        )
    )

    assert source_fingerprint_overlap(validation_records, final_records) == set()
    enforce_zero_source_overlap({"validation": validation_records, "final": final_records})


def test_v0_6b_final_manifest_marks_tiny_as_seen_and_cifar100_as_unseen():
    records = build_plan_a_manifest(
        _load_protocol("configs/protocol/plan_a_next_v0_6b_test_cifar100_holdout.yaml"),
        _v0_6b_sources(include_cifar100=True),
    )
    roles = {
        (record.source_dataset_name, record.source_role)
        for record in records
        if record.cohort_name == "ood"
    }

    assert ("tiny_imagenet", "seen_source_ood") in roles
    assert ("cifar100", "unseen_ood_source") in roles


def test_v0_6b_supcon_study_uses_model_without_source_adversary_head():
    study_config = _load_study("configs/study/plan_a_next_v0_6b_supcon_b2_loso_cifar100_holdout.yaml")
    model_config = _load_model(study_config["model_config"])

    assert model_config["source_adversary_enabled"] is False


def test_v0_6b_cifar100_is_not_in_train_or_validation_manifest():
    train_records = build_plan_a_manifest(
        _load_protocol("configs/protocol/plan_a_next_v0_6b_train.yaml"),
        _v0_6b_sources(include_cifar100=False),
    )
    validation_records = build_plan_a_manifest(
        _load_protocol("configs/protocol/plan_a_next_v0_6b_validation.yaml"),
        _v0_6b_sources(include_cifar100=False),
    )

    assert "cifar100" not in {record.source_dataset_name for record in train_records}
    assert "cifar100" not in {record.source_dataset_name for record in validation_records}


def test_source_balanced_sampler_balances_five_ood_sources_per_batch():
    records = []
    records.extend(_record(f"easy-{index}", "easy_id", "cifar10") for index in range(10))
    records.extend(_record(f"hard-{index}", "hard_id", "cifar10") for index in range(10))
    records.extend(_record(f"amb-{index}", "ambiguous_id", "cifar10") for index in range(10))
    for source in ("svhn", "dtd", "lsun_resize", "gaussian_noise", "tiny_imagenet"):
        records.extend(_record(f"{source}-{index}", "unknown_supervision", source) for index in range(10))

    sampler = SourceBalancedBatchSampler(records, batch_size=20, batches_per_epoch=1, seed=7, shuffle=False)
    batch_records = [records[index] for index in next(iter(sampler))]

    source_counts: dict[str, int] = {}
    for record in batch_records:
        if record.cohort_name == "unknown_supervision":
            source_counts[record.source_dataset_name] = source_counts.get(record.source_dataset_name, 0) + 1
    assert source_counts == {
        "dtd": 2,
        "gaussian_noise": 2,
        "lsun_resize": 2,
        "svhn": 2,
        "tiny_imagenet": 2,
    }


def test_v0_6c_cifar100_class_holdout_sets_are_disjoint():
    train_records = build_plan_a_manifest(
        _load_protocol("configs/protocol/plan_a_next_v0_6c_train.yaml"),
        _v0_6c_sources(),
    )
    validation_records = build_plan_a_manifest(
        _load_protocol("configs/protocol/plan_a_next_v0_6c_validation.yaml"),
        _v0_6c_sources(),
    )
    final_records = build_plan_a_manifest(
        _load_protocol("configs/protocol/plan_a_next_v0_6c_test_cifar100_class_holdout.yaml"),
        _v0_6c_sources(),
    )

    train_cifar100_labels = {
        record.source_class_label for record in train_records if record.source_dataset_name == "cifar100"
    }
    validation_cifar100_labels = {
        record.source_class_label for record in validation_records if record.source_dataset_name == "cifar100"
    }
    final_seen_labels = {
        record.source_class_label
        for record in final_records
        if record.source_dataset_name == "cifar100" and record.source_role == "seen_source_ood"
    }
    final_unseen_labels = {
        record.source_class_label
        for record in final_records
        if record.source_dataset_name == "cifar100" and record.source_role == "unseen_ood_classes"
    }

    assert train_cifar100_labels <= set(range(50))
    assert validation_cifar100_labels <= set(range(50))
    assert final_seen_labels <= set(range(50))
    assert final_unseen_labels <= set(range(50, 100))
    assert final_seen_labels.isdisjoint(final_unseen_labels)
    assert source_fingerprint_overlap(validation_records, final_records) == set()


def test_v0_6c_final_heldout_records_use_unseen_class_role():
    final_records = build_plan_a_manifest(
        _load_protocol("configs/protocol/plan_a_next_v0_6c_test_cifar100_class_holdout.yaml"),
        _v0_6c_sources(),
    )

    heldout_records = [
        record
        for record in final_records
        if record.source_dataset_name == "cifar100"
        and record.source_partition_name == "cifar100_test_unseen_classes_050_100_final"
    ]

    assert heldout_records
    assert {record.source_role for record in heldout_records} == {"unseen_ood_classes"}
    assert {record.source_class_label for record in heldout_records} <= set(range(50, 100))


def test_source_weighted_sampler_overweights_near_ood_sources():
    records = []
    records.extend(_record(f"easy-{index}", "easy_id", "cifar10") for index in range(12))
    records.extend(_record(f"hard-{index}", "hard_id", "cifar10") for index in range(12))
    records.extend(_record(f"amb-{index}", "ambiguous_id", "cifar10") for index in range(12))
    for source in ("svhn", "dtd", "tiny_imagenet", "cifar100"):
        records.extend(_record(f"{source}-{index}", "unknown_supervision", source) for index in range(12))

    sampler = SourceBalancedBatchSampler(
        records,
        batch_size=24,
        batches_per_epoch=1,
        seed=7,
        shuffle=False,
        source_weights={"svhn": 1.0, "dtd": 1.0, "tiny_imagenet": 2.0, "cifar100": 3.0},
    )
    batch_records = [records[index] for index in next(iter(sampler))]
    source_counts: dict[str, int] = {}
    for record in batch_records:
        if record.cohort_name == "unknown_supervision":
            source_counts[record.source_dataset_name] = source_counts.get(record.source_dataset_name, 0) + 1

    assert sum(record.cohort_name in {"easy_id", "hard_id"} for record in batch_records) == 6
    assert sum(record.cohort_name == "ambiguous_id" for record in batch_records) == 6
    assert sum(source_counts.values()) == 12
    assert source_counts["cifar100"] > source_counts["tiny_imagenet"] > min(
        source_counts["svhn"],
        source_counts["dtd"],
    )


def test_source_overlap_enforcement_rejects_cross_split_reuse():
    left = [_record("left", "easy_id", "cifar10")]
    right = [_record("right", "hard_id", "cifar10")]

    try:
        enforce_zero_source_overlap({"left": left, "right": right})
    except ValueError as error:
        assert "Source fingerprint overlap" in str(error)
    else:
        raise AssertionError("expected source fingerprint overlap to be rejected")
