from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import pytest

from frcnet.evaluation import (
    ReferenceScoreRecord,
    SampleAnalysisRecord,
    build_frozen_matched_manifest,
    compute_matched_manifest_hash,
    read_reference_score_records,
    summarize_matched_ambiguous_vs_ood,
    summarize_scalar_benchmarks,
    write_matched_manifest,
    write_matched_manifest_bin_diagnostics,
    write_reference_score_records,
)
from frcnet.workflows.plan_a import _read_optional_matched_manifest


def _sample_record(sample_id: str, cohort_name: str, *, scalar_value: float, entropy_value: float) -> SampleAnalysisRecord:
    return SampleAnalysisRecord(
        model_family="frcnet_explicit_unknown",
        run_id="RUN-V4",
        protocol_id="plan_a_next_v0_4_strict_freeze",
        sample_id=sample_id,
        split_name="test",
        cohort_name=cohort_name,
        source_dataset_name="synthetic",
        source_class_label=None,
        class_label=-1,
        predicted_class_index=0,
        resolution_ratio=0.9 if cohort_name == "ambiguous_id" else 0.1,
        unknown_mass=0.1 if cohort_name == "ambiguous_id" else 0.9,
        state_content_entropy=entropy_value,
        state_weighted_content_entropy=entropy_value * (0.9 if cohort_name == "ambiguous_id" else 0.1),
        state_entropy=entropy_value,
        resolution_entropy=0.0,
        top1_class_mass=scalar_value,
        top1_view_truth_mass=scalar_value,
        top1_view_false_mass=max(0.0, 1.0 - scalar_value),
        top1_view_unknown_mass=0.0,
        top1_view_tau=scalar_value,
        proposition_truth_mass=scalar_value,
        proposition_false_mass=max(0.0, 1.0 - scalar_value),
        proposition_unknown_mass=0.0,
        proposition_truth_ratio=scalar_value,
        ternary_entropy=0.0,
        auxiliary_top1_content_probability=scalar_value,
        top1_completion_beta_0_1=scalar_value,
        top1_completion_beta_0_25=scalar_value,
        top1_completion_beta_0_5=scalar_value,
        top1_completion_beta_0_75=scalar_value,
        candidate_class_indices=(0, 1) if cohort_name == "ambiguous_id" else (),
    )


def _reference_record(sample_id: str, cohort_name: str, score_value: float) -> ReferenceScoreRecord:
    return ReferenceScoreRecord(
        sample_id=sample_id,
        split_name="test",
        cohort_name=cohort_name,
        source_dataset_name="synthetic",
        reference_model_family="softmax_ce_reference",
        reference_run_id="REF-RUN",
        reference_score_name="softmax_entropy",
        reference_score_value=score_value,
    )


def _fixture_records():
    samples = []
    references = []
    for index, score_value in enumerate((0.1, 0.2, 0.8, 0.9)):
        samples.append(
            _sample_record(f"ambiguous-{index}", "ambiguous_id", scalar_value=0.8 - (index * 0.05), entropy_value=0.5)
        )
        references.append(_reference_record(f"ambiguous-{index}", "ambiguous_id", score_value))
        samples.append(_sample_record(f"ood-{index}", "ood", scalar_value=0.2 + (index * 0.05), entropy_value=2.0))
        references.append(_reference_record(f"ood-{index}", "ood", score_value + 0.01))
    return samples, references


def test_frozen_matched_manifest_pairs_roles_and_bin_diagnostics(tmp_path: Path):
    samples, references = _fixture_records()

    manifest_records, diagnostics = build_frozen_matched_manifest(
        samples,
        references,
        num_bins=2,
        test_size=0.5,
        random_state=7,
    )

    pair_roles: dict[str, set[str]] = defaultdict(set)
    pair_cohorts: dict[str, set[str]] = defaultdict(set)
    for record in manifest_records:
        pair_roles[record.paired_group_id].add(record.manifest_role)
        pair_cohorts[record.paired_group_id].add(record.cohort_name)

    assert manifest_records
    assert all(len(roles) == 1 for roles in pair_roles.values())
    assert all(cohorts == {"ambiguous_id", "ood"} for cohorts in pair_cohorts.values())
    assert {record.manifest_role for record in manifest_records} == {"train", "test"}
    assert sum(diagnostic.matched_pairs for diagnostic in diagnostics) == len(pair_roles)
    assert all(diagnostic.unmatched_positive >= 0 for diagnostic in diagnostics)
    assert all(diagnostic.unmatched_negative >= 0 for diagnostic in diagnostics)

    manifest_path = write_matched_manifest(manifest_records, tmp_path / "frozen.jsonl")
    diagnostics_path = write_matched_manifest_bin_diagnostics(diagnostics, tmp_path / "bins.csv")

    assert manifest_path.exists()
    assert diagnostics_path.exists()
    assert "matched_pairs" in diagnostics_path.read_text(encoding="utf-8")


def test_frozen_manifest_hash_changes_when_reference_score_changes():
    samples, references = _fixture_records()
    first_records, _ = build_frozen_matched_manifest(samples, references, num_bins=2, random_state=7)
    changed_references = list(references)
    changed_references[0] = replace(changed_references[0], reference_score_value=0.33)
    changed_records, _ = build_frozen_matched_manifest(samples, changed_references, num_bins=2, random_state=7)

    assert compute_matched_manifest_hash(first_records) != compute_matched_manifest_hash(changed_records)


def test_scalar_scan_uses_same_frozen_manifest_subset_as_pair_benchmark():
    samples, references = _fixture_records()
    extra_records = [
        _sample_record("ambiguous-extra", "ambiguous_id", scalar_value=0.0, entropy_value=3.0),
        _sample_record("ood-extra", "ood", scalar_value=1.0, entropy_value=0.0),
    ]
    manifest_records, _ = build_frozen_matched_manifest(samples, references, num_bins=2, test_size=0.5, random_state=7)

    matched_summary = summarize_matched_ambiguous_vs_ood(
        samples + extra_records,
        primary_pair="resolution_ratio__state_weighted_content_entropy",
        weighted_pair="resolution_ratio__state_content_entropy",
        completion_scan_scalars=("top1_completion_beta_0_1",),
        matched_manifest_records=manifest_records,
    )
    scalar_summary = summarize_scalar_benchmarks(
        samples + extra_records,
        scalar_names=("top1_completion_beta_0_1",),
        matched_manifest_records=manifest_records,
    )[0]

    assert matched_summary.completion_scan_aurocs == pytest.approx((scalar_summary.auroc,))
    assert scalar_summary.matched_count_per_class == matched_summary.matched_count_per_class


def test_require_matched_manifest_rejects_missing_path():
    with pytest.raises(ValueError, match="matched_manifest_path is required"):
        _read_optional_matched_manifest({"matched_manifest_path": "", "require_matched_manifest": True})


def test_reference_score_records_round_trip(tmp_path: Path):
    records = [_reference_record("sample-1", "ambiguous_id", 0.5)]
    output_path = write_reference_score_records(records, tmp_path / "reference.jsonl")

    restored = read_reference_score_records(output_path)

    assert restored == records
