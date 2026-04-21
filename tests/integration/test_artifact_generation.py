from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from frcnet.analysis import (
    write_artifact_path_list,
    write_cohort_occupancy,
    write_cohort_summary_table,
    write_experiment_record,
    write_geometry_hexbin,
    write_geometry_scatter,
    write_scalar_roc_curve,
    write_tau_cohort_boxplot,
)
from frcnet.evaluation import summarize_matched_ambiguous_vs_ood, write_matched_benchmark_summary
from frcnet.models import FRCNetModel
from frcnet.evaluation.inference import build_sample_analysis_records
from tests.conftest import build_synthetic_batch


def _build_records():
    batch_input = build_synthetic_batch()
    model = FRCNetModel(num_classes=10)
    model_output = model(batch_input.image)
    records = build_sample_analysis_records(model_output, batch_input, run_id="RUN-1", protocol_id="plan_a_v1")
    duplicated = [replace(record) for record in records] + [replace(record) for record in records]
    for index, record in enumerate(duplicated):
        record.sample_id = f"{record.sample_id}-{index}"
    return duplicated


def test_artifact_writers_create_expected_files(tmp_path: Path):
    sample_records = _build_records()
    scatter_path = write_geometry_scatter(sample_records, tmp_path / "geometry_scatter.png")
    hexbin_path = write_geometry_hexbin(sample_records, tmp_path / "geometry_hexbin.png")
    occupancy_path = write_cohort_occupancy(sample_records, tmp_path / "cohort_occupancy.png")
    tau_path = write_tau_cohort_boxplot(sample_records, tmp_path / "tau_cohort_boxplot.png")
    tau_roc_path = write_scalar_roc_curve(
        sample_records,
        tmp_path / "tau_roc_curve.png",
        positive_cohort="ambiguous_id",
        negative_cohort="ood",
        scalar_name="top1_content_probability",
        test_size=0.3,
        random_state=7,
    )
    summary_path = write_cohort_summary_table(sample_records, tmp_path / "cohort_summary_table.csv")

    assert scatter_path.exists()
    assert hexbin_path.exists()
    assert occupancy_path.exists()
    assert tau_path.exists()
    assert tau_roc_path.exists()
    assert summary_path.exists()
    assert "mean_top1_content_probability" in summary_path.read_text(encoding="utf-8")
    assert "mean_resolution_weighted_content_entropy" in summary_path.read_text(encoding="utf-8")


def test_matched_summary_and_experiment_record(tmp_path: Path):
    sample_records = _build_records()
    matched_summary = summarize_matched_ambiguous_vs_ood(sample_records)
    matched_path = write_matched_benchmark_summary(matched_summary, tmp_path / "matched.csv")
    artifact_paths = {
        "geometry_scatter": str(tmp_path / "geometry_scatter.png"),
        "tau_cohort_boxplot": str(tmp_path / "tau_cohort_boxplot.png"),
        "tau_roc_curve": str(tmp_path / "tau_roc_curve.png"),
        "matched_ambiguous_vs_ood_table": str(matched_path),
    }
    artifact_index_path = write_artifact_path_list(artifact_paths, tmp_path / "artifact_paths.json")
    experiment_record_path = write_experiment_record(
        output_path=tmp_path / "experiment_record.md",
        run_id="RUN-1",
        protocol_id="plan_a_v1",
        config_snapshot_paths={
            "protocol_config_snapshot": str(tmp_path / "protocol_config_snapshot.yaml"),
            "eval_config_snapshot": str(tmp_path / "eval_config_snapshot.yaml"),
            "model_config_snapshot": str(tmp_path / "model_config_snapshot.yaml"),
        },
        manifest_snapshot_path=str(tmp_path / "manifest.jsonl"),
        analysis_record_path=str(tmp_path / "sample_analysis_records.csv"),
        proposition_record_path=str(tmp_path / "top1_proposition_records.csv"),
        artifact_paths={**artifact_paths, "artifact_paths": str(artifact_index_path)},
        matched_summary=matched_summary,
        checkpoint_path=str(tmp_path / "checkpoint_best.pt"),
        analysis_summary_path=str(tmp_path / "analysis_summary.json"),
        sidecar_resolution_mode="analysis_summary_explicit",
        integrity_overrides=("missing_checkpoint",),
        source_run_ids=("RUN-1",),
        source_protocol_ids=("plan_a_v1",),
        resolved_eval_config={"primary_scalar": "completion_score_beta_0_1", "random_state": 7},
    )
    record_text = experiment_record_path.read_text(encoding="utf-8")
    matched_text = matched_path.read_text(encoding="utf-8")

    assert matched_path.exists()
    assert artifact_index_path.exists()
    assert experiment_record_path.exists()
    assert "tau_scalar_auroc" in matched_text
    assert "tau_scalar_name" in matched_text
    assert "weighted_pair_auroc" in matched_text
    assert "checkpoint_path" in record_text
    assert "analysis_summary_path" in record_text
    assert "Resolved Eval Config" in record_text
    assert "tau_scalar_auroc" in record_text
    assert "weighted_pair_auroc" in record_text
