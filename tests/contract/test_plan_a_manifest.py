from __future__ import annotations

from frcnet.data import ManifestBackedVisionDataset, build_plan_a_manifest, collate_manifest_samples
from tests.conftest import FakeLabelsDataset, FakeTargetsDataset, build_protocol_config


def test_build_plan_a_manifest_generates_all_target_cohorts():
    protocol_config = build_protocol_config()
    cifar_labels = [label for label in range(10) for _ in range(8)]
    svhn_labels = [index % 10 for index in range(20)]
    source_datasets = {
        "cifar10": FakeTargetsDataset(cifar_labels),
        "svhn": FakeLabelsDataset(svhn_labels),
    }

    manifest_records = build_plan_a_manifest(protocol_config, source_datasets)
    cohort_names = {record.cohort_name for record in manifest_records}

    assert cohort_names == {"easy_id", "hard_id", "ambiguous_id", "ood", "unknown_supervision"}
    assert any(record.candidate_class_indices for record in manifest_records if record.cohort_name == "ambiguous_id")
    assert all(record.class_label == -1 for record in manifest_records if record.cohort_name in {"ood", "unknown_supervision"})


def test_manifest_backed_dataset_collates_candidate_masks():
    protocol_config = build_protocol_config()
    cifar_labels = [label for label in range(10) for _ in range(8)]
    svhn_labels = [index % 10 for index in range(20)]
    source_datasets = {
        "cifar10": FakeTargetsDataset(cifar_labels),
        "svhn": FakeLabelsDataset(svhn_labels),
    }
    manifest_records = build_plan_a_manifest(protocol_config, source_datasets)
    dataset = ManifestBackedVisionDataset(manifest_records, source_datasets, num_classes=10)
    ambiguous_dataset_index = next(
        index for index, record in enumerate(manifest_records) if record.cohort_name == "ambiguous_id"
    )

    batch_input = collate_manifest_samples([dataset[0], dataset[1], dataset[ambiguous_dataset_index]])

    assert batch_input.image.shape[0] == 3
    assert batch_input.source_class_label is not None
    assert batch_input.candidate_class_mask is not None
