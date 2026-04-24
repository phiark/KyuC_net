from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import warnings

import numpy as np
import pytest

from frcnet.evaluation import (
    AnalysisExportSummary,
    build_top1_proposition_records,
    read_analysis_export_summary,
    read_sample_analysis_records,
    summarize_matched_ambiguous_vs_ood,
    write_analysis_export_summary,
    write_sample_analysis_records,
)
from frcnet.models import FRCNetModel
from frcnet.evaluation.inference import build_sample_analysis_records
from frcnet.data.plan_a import load_plan_a_source_datasets
from tests.conftest import build_synthetic_batch


def test_sample_analysis_records_expose_paper_fields():
    batch_input = build_synthetic_batch()
    model = FRCNetModel(num_classes=10)
    model_output = model(batch_input.image)

    records = build_sample_analysis_records(
        model_output,
        batch_input,
        run_id="RUN-1",
        protocol_id="plan_a_next_v0_1_analysis",
    )

    assert len(records) == batch_input.batch_size
    first_record = records[0]
    assert first_record.run_id == "RUN-1"
    assert hasattr(first_record, "resolution_ratio")
    assert hasattr(first_record, "state_content_entropy")
    assert hasattr(first_record, "state_weighted_content_entropy")
    assert hasattr(first_record, "state_entropy")
    assert hasattr(first_record, "content_entropy")
    assert hasattr(first_record, "resolution_weighted_content_entropy")
    assert hasattr(first_record, "proposition_truth_ratio")
    assert hasattr(first_record, "top1_view_tau")
    assert hasattr(first_record, "top1_completion_beta_0_1")
    assert hasattr(first_record, "resolution_entropy")
    assert hasattr(first_record, "ternary_entropy")
    assert hasattr(first_record, "auxiliary_top1_content_probability")
    assert hasattr(first_record, "completion_score_beta_0_1")
    assert hasattr(first_record, "completion_score_beta_0_25")
    assert hasattr(first_record, "completion_score_beta_0_75")


def test_sample_analysis_csv_outputs_current_schema_and_reads_legacy_aliases(tmp_path: Path):
    batch_input = build_synthetic_batch()
    model = FRCNetModel(num_classes=10)
    model_output = model(batch_input.image)
    records = build_sample_analysis_records(
        model_output,
        batch_input,
        run_id="RUN-1",
        protocol_id="plan_a_next_v0_1_analysis",
    )

    csv_path = write_sample_analysis_records(records, tmp_path / "analysis.csv")
    header = csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")

    assert "state_content_entropy" in header
    assert "state_weighted_content_entropy" in header
    assert "top1_completion_beta_0_1" in header
    assert "content_entropy" not in header
    assert "resolution_weighted_content_entropy" not in header
    assert "completion_score_beta_0_1" not in header

    legacy_row = records[0].to_csv_row()
    legacy_row["content_entropy"] = legacy_row.pop("state_content_entropy")
    legacy_row["resolution_weighted_content_entropy"] = legacy_row.pop("state_weighted_content_entropy")
    legacy_row["completion_score_beta_0_1"] = legacy_row.pop("top1_completion_beta_0_1")
    legacy_path = tmp_path / "legacy_analysis.csv"
    legacy_path.write_text(
        ",".join(legacy_row.keys()) + "\n" + ",".join(str(value) for value in legacy_row.values()) + "\n",
        encoding="utf-8",
    )
    restored = read_sample_analysis_records(legacy_path)

    assert restored[0].state_content_entropy == pytest.approx(float(legacy_row["content_entropy"]))
    assert restored[0].state_weighted_content_entropy == pytest.approx(
        float(legacy_row["resolution_weighted_content_entropy"])
    )
    assert restored[0].top1_completion_beta_0_1 == pytest.approx(float(legacy_row["completion_score_beta_0_1"]))


def test_top1_proposition_records_cover_all_cohorts_and_preserve_proposition_masses():
    batch_input = build_synthetic_batch()
    model = FRCNetModel(num_classes=10)
    model_output = model(batch_input.image)
    sample_records = build_sample_analysis_records(
        model_output,
        batch_input,
        run_id="RUN-1",
        protocol_id="plan_a_next_v0_1_analysis",
    )

    proposition_records = build_top1_proposition_records(sample_records)

    assert len(proposition_records) == batch_input.batch_size
    ambiguous_record = next(record for record in proposition_records if record.cohort_name == "ambiguous_id")
    assert ambiguous_record.proposition_target_type == "candidate_set"
    unknown_record = next(record for record in proposition_records if record.cohort_name == "unknown_supervision")
    assert unknown_record.proposition_target_type == "empty_set"
    for record in proposition_records:
        total_mass = (
            record.proposition_truth_mass + record.proposition_false_mass + record.proposition_unknown_mass
        )
        assert total_mass == pytest.approx(1.0, abs=1e-5)
        assert 0.0 <= record.proposition_truth_ratio <= 1.0


def test_analysis_export_summary_round_trip(tmp_path: Path):
    summary = AnalysisExportSummary(
        run_id="RUN-1",
        protocol_id="plan_a_next_v0_1_analysis",
        analysis_path="sample_analysis_records.csv",
        checkpoint_path="checkpoint_best.pt",
        manifest_snapshot_path="plan_a_manifest_snapshot.jsonl",
        model_config_snapshot_path="model_config_snapshot.yaml",
        proposition_path="top1_proposition_records.csv",
        integrity_overrides=("missing_checkpoint",),
    )

    output_path = write_analysis_export_summary(summary, tmp_path / "analysis_summary.json")
    restored = read_analysis_export_summary(output_path)

    assert restored.run_id == "RUN-1"
    assert restored.integrity_overrides == ("missing_checkpoint",)


def test_matched_summary_rejects_invalid_scalar_name():
    batch_input = build_synthetic_batch()
    model = FRCNetModel(num_classes=10)
    model_output = model(batch_input.image)
    sample_records = build_sample_analysis_records(
        model_output,
        batch_input,
        run_id="RUN-1",
        protocol_id="plan_a_next_v0_1_analysis",
    )
    duplicated_records = [replace(record) for record in sample_records] + [replace(record) for record in sample_records]
    for index, record in enumerate(duplicated_records):
        record.sample_id = f"{record.sample_id}-{index}"

    with pytest.raises(ValueError, match="Unsupported primary_scalar"):
        summarize_matched_ambiguous_vs_ood(duplicated_records, primary_scalar="predicted_class_index")

    with pytest.raises(ValueError, match="Unsupported primary_scalar"):
        summarize_matched_ambiguous_vs_ood(duplicated_records, primary_scalar="completion_score_beta_0_1")

    with pytest.raises(ValueError, match="Unsupported primary_pair"):
        summarize_matched_ambiguous_vs_ood(duplicated_records, primary_pair="resolution_ratio__content_entropy")


def test_matched_summary_rejects_label_aware_primary_scalar():
    batch_input = build_synthetic_batch()
    model = FRCNetModel(num_classes=10)
    model_output = model(batch_input.image)
    sample_records = build_sample_analysis_records(
        model_output,
        batch_input,
        run_id="RUN-1",
        protocol_id="plan_a_next_v0_1_analysis",
    )
    duplicated_records = [replace(record) for record in sample_records] + [replace(record) for record in sample_records]
    for index, record in enumerate(duplicated_records):
        record.sample_id = f"{record.sample_id}-{index}"

    with pytest.raises(ValueError, match="label-aware"):
        summarize_matched_ambiguous_vs_ood(duplicated_records, primary_scalar="proposition_truth_ratio")


def test_load_plan_a_source_datasets_suppresses_numpy_visible_deprecation_warning(monkeypatch):
    protocol_config = {
        "datasets": {
            "cifar10": {"root": "data/cifar10", "train": False, "download": False},
            "svhn": {"root": "data/svhn", "split": "test", "download": False},
        }
    }

    class _FakeDataset:
        def __len__(self):
            return 1

    def _fake_cifar10(**_kwargs):
        warnings.warn(
            "dtype(): align should be passed as Python or NumPy boolean but got `align=0`.",
            category=np.exceptions.VisibleDeprecationWarning,
        )
        return _FakeDataset()

    monkeypatch.setattr("frcnet.data.plan_a.datasets.CIFAR10", _fake_cifar10)
    monkeypatch.setattr("frcnet.data.plan_a.datasets.SVHN", lambda **_kwargs: _FakeDataset())

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        datasets_payload = load_plan_a_source_datasets(protocol_config)

    assert "cifar10" in datasets_payload
    assert not any(isinstance(item.message, np.exceptions.VisibleDeprecationWarning) for item in captured)
