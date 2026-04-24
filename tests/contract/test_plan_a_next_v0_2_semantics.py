from __future__ import annotations

import pytest
import torch

from frcnet.data.plan_a import build_plan_a_manifest
from frcnet.evaluation import (
    MatchedManifestRecord,
    SUPPORTED_PRIMARY_SCALAR_FEATURES,
    binary_pignistic_beta,
    build_proposition_view_records,
    build_sample_analysis_records,
    candidate_symmetric_beta,
    compute_matched_manifest_hash,
    state_content_entropy,
    state_entropy,
    state_weighted_content_entropy,
    summarize_matched_ambiguous_vs_ood,
    top1_symmetric_beta,
    with_manifest_hash,
)
from frcnet.models import FRCNetModel
from tests.conftest import build_synthetic_batch


def test_state_entropy_decomposition_matches_contract():
    content_distribution = torch.tensor([[0.5, 0.5], [0.9, 0.1]], dtype=torch.float32)
    resolution_ratio = torch.tensor([0.25, 0.8], dtype=torch.float32)

    content_value = state_content_entropy(content_distribution)
    weighted_value = state_weighted_content_entropy(resolution_ratio, content_value)
    state_value = state_entropy(resolution_ratio, content_value)

    assert torch.allclose(weighted_value, resolution_ratio * content_value)
    assert torch.all(state_value >= weighted_value)


def test_proposition_views_conserve_mass_and_mark_label_awareness():
    batch_input = build_synthetic_batch()
    model_output = FRCNetModel(num_classes=10)(batch_input.image)
    sample_records = build_sample_analysis_records(model_output, batch_input, run_id="RUN-1", protocol_id="plan_a_next_v0_2")

    view_records = build_proposition_view_records(sample_records)

    assert {record.view_name for record in view_records} >= {"top1_view", "candidate_set_view", "empty_set_view"}
    for record in view_records:
        total_mass = record.proposition_truth_mass + record.proposition_false_mass + record.proposition_unknown_mass
        assert total_mass == pytest.approx(1.0, abs=1e-5)
        if record.view_name == "top1_view":
            assert record.label_aware is False
        else:
            assert record.label_aware is True


def test_primary_scalar_whitelist_excludes_label_aware_proposition_fields():
    assert "proposition_truth_ratio" not in SUPPORTED_PRIMARY_SCALAR_FEATURES
    assert "proposition_truth_mass" not in SUPPORTED_PRIMARY_SCALAR_FEATURES

    batch_input = build_synthetic_batch()
    model_output = FRCNetModel(num_classes=10)(batch_input.image)
    sample_records = build_sample_analysis_records(model_output, batch_input, run_id="RUN-1", protocol_id="plan_a_next_v0_2")
    duplicated_records = [*sample_records, *sample_records]
    for index, record in enumerate(duplicated_records):
        record.sample_id = f"{record.sample_id}-{index}"

    with pytest.raises(ValueError, match="Unsupported primary_scalar"):
        summarize_matched_ambiguous_vs_ood(duplicated_records, primary_scalar="proposition_truth_ratio")


def test_beta_policy_defaults_are_view_specific():
    assert top1_symmetric_beta(10) == pytest.approx(0.1)
    assert candidate_symmetric_beta(2, 10) == pytest.approx(0.2)
    assert binary_pignistic_beta() == pytest.approx(0.5)


def test_matched_manifest_hash_is_stable_and_sensitive_to_reference_score():
    config_hash = "config-hash"
    records = [
        MatchedManifestRecord(
            sample_id="a",
            cohort_name="ambiguous_id",
            reference_score_name="softmax_entropy",
            reference_score_value=0.1,
            match_bin_id="bin-0",
            manifest_role="train",
            paired_group_id="pair-0",
            manifest_hash="",
            construction_config_hash=config_hash,
        ),
        MatchedManifestRecord(
            sample_id="b",
            cohort_name="ood",
            reference_score_name="softmax_entropy",
            reference_score_value=0.2,
            match_bin_id="bin-0",
            manifest_role="test",
            paired_group_id="pair-0",
            manifest_hash="",
            construction_config_hash=config_hash,
        ),
    ]

    first_hash = compute_matched_manifest_hash(records)
    assert first_hash == compute_matched_manifest_hash(with_manifest_hash(records))

    changed_records = [records[0], records[1]]
    changed_records[1] = MatchedManifestRecord(
        **{**records[1].to_dict(), "reference_score_value": 0.3, "manifest_hash": ""}
    )
    assert compute_matched_manifest_hash(changed_records) != first_hash


def test_hard_id_manifest_uses_protocol_recipe_parameters():
    class _FakeDataset:
        targets = [0, 0, 0, 0]

        def __len__(self):
            return len(self.targets)

    protocol_config = {
        "protocol_id": "plan_a_next_v0_2_unit",
        "seed": 7,
        "split_name": "analysis",
        "num_classes": 1,
        "analysis": {
            "easy_id_per_class": 1,
            "hard_id_per_class": 2,
            "ambiguous_per_pair": 0,
            "ood_count": 0,
            "unknown_supervision_count": 0,
        },
        "ambiguous": {"alpha_min": 0.35, "alpha_max": 0.65, "class_pairs": []},
        "hard_id": {
            "recipes": ["gaussian_blur", "low_res"],
            "blur_kernel_size": 7,
            "blur_sigma": 2.5,
            "low_res_size": 12,
        },
        "datasets": {},
    }

    records = build_plan_a_manifest(protocol_config, {"cifar10": _FakeDataset(), "svhn": _FakeDataset()})
    hard_records = [record for record in records if record.cohort_name == "hard_id"]

    assert hard_records[0].augmentation_parameters == {"kernel_size": 7, "sigma": 2.5}
    assert hard_records[1].augmentation_parameters == {"downsample_size": 12}
