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
    summarize_matched_ambiguous_vs_ood,
    write_analysis_export_summary,
)
from frcnet.models import FRCNetModel
from frcnet.evaluation.inference import build_sample_analysis_records
from frcnet.data.plan_a import load_plan_a_source_datasets
from tests.conftest import build_synthetic_batch


def test_sample_analysis_records_expose_paper_fields():
    batch_input = build_synthetic_batch()
    model = FRCNetModel(num_classes=10)
    model_output = model(batch_input.image)

    records = build_sample_analysis_records(model_output, batch_input, run_id="RUN-1", protocol_id="plan_a_v1")

    assert len(records) == batch_input.batch_size
    first_record = records[0]
    assert first_record.run_id == "RUN-1"
    assert hasattr(first_record, "resolution_ratio")
    assert hasattr(first_record, "content_entropy")
    assert hasattr(first_record, "resolution_weighted_content_entropy")
    assert hasattr(first_record, "completion_score_beta_0_1")
    assert hasattr(first_record, "completion_score_beta_0_25")
    assert hasattr(first_record, "completion_score_beta_0_75")


def test_top1_proposition_records_filter_out_ood_and_unknown():
    batch_input = build_synthetic_batch()
    model = FRCNetModel(num_classes=10)
    model_output = model(batch_input.image)
    sample_records = build_sample_analysis_records(model_output, batch_input, run_id="RUN-1", protocol_id="plan_a_v1")

    proposition_records = build_top1_proposition_records(sample_records)

    assert len(proposition_records) == 3
    ambiguous_record = next(record for record in proposition_records if record.cohort_name == "ambiguous_id")
    assert ambiguous_record.proposition_target_type == "candidate_set"


def test_analysis_export_summary_round_trip(tmp_path: Path):
    summary = AnalysisExportSummary(
        run_id="RUN-1",
        protocol_id="plan_a_v1",
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
    sample_records = build_sample_analysis_records(model_output, batch_input, run_id="RUN-1", protocol_id="plan_a_v1")
    duplicated_records = [replace(record) for record in sample_records] + [replace(record) for record in sample_records]
    for index, record in enumerate(duplicated_records):
        record.sample_id = f"{record.sample_id}-{index}"

    with pytest.raises(ValueError, match="Unsupported primary_scalar"):
        summarize_matched_ambiguous_vs_ood(duplicated_records, primary_scalar="predicted_class_index")


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
