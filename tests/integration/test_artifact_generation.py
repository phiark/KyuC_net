from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from frcnet.analysis import (
    write_cohort_counts,
    write_artifact_path_list,
    write_cohort_occupancy,
    write_cohort_summary_table,
    write_experiment_record,
    write_geometry_hexbin,
    write_geometry_scatter,
    write_scalar_roc_curve,
    write_tau_cohort_boxplot,
)
from frcnet.analysis.artifacts import _build_cohort_occupancy_histograms
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
    cohort_counts_path = write_cohort_counts(sample_records, tmp_path / "cohort_counts.png")
    tau_path = write_tau_cohort_boxplot(sample_records, tmp_path / "tau_cohort_boxplot.png")
    tau_roc_path = write_scalar_roc_curve(
        sample_records,
        tmp_path / "tau_roc_curve.png",
        positive_cohort="ambiguous_id",
        negative_cohort="ood",
        scalar_name="proposition_truth_ratio",
        test_size=0.3,
        random_state=7,
    )
    summary_path = write_cohort_summary_table(sample_records, tmp_path / "cohort_summary_table.csv")

    assert scatter_path.exists()
    assert hexbin_path.exists()
    assert occupancy_path.exists()
    assert cohort_counts_path.exists()
    assert tau_path.exists()
    assert tau_roc_path.exists()
    assert summary_path.exists()
    assert "mean_proposition_truth_ratio" in summary_path.read_text(encoding="utf-8")
    assert "mean_auxiliary_top1_content_probability" in summary_path.read_text(encoding="utf-8")
    assert "mean_resolution_weighted_content_entropy" in summary_path.read_text(encoding="utf-8")


def test_matched_summary_and_experiment_record(tmp_path: Path):
    sample_records = _build_records()
    matched_summary = summarize_matched_ambiguous_vs_ood(sample_records)
    matched_path = write_matched_benchmark_summary(matched_summary, tmp_path / "matched.csv")
    artifact_paths = {
        "geometry_scatter": str(tmp_path / "geometry_scatter.png"),
        "cohort_counts": str(tmp_path / "cohort_counts.png"),
        "tau_cohort_boxplot": str(tmp_path / "tau_cohort_boxplot.png"),
        "tau_roc_curve": str(tmp_path / "tau_roc_curve.png"),
        "matched_ambiguous_vs_ood_table": str(matched_path),
        "proposition_diagnostic_table": str(tmp_path / "proposition.csv"),
    }
    artifact_index_path = write_artifact_path_list(artifact_paths, tmp_path / "artifact_paths.json")
    experiment_record_path = write_experiment_record(
        output_path=tmp_path / "experiment_record.md",
        model_family="frcnet_explicit_unknown",
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
        proposition_diagnostic_scalar_name="proposition_truth_ratio",
        proposition_diagnostic_table_path=str(tmp_path / "proposition.csv"),
        proposition_tau_roc_curve_path=str(tmp_path / "tau_roc_curve.png"),
    )
    record_text = experiment_record_path.read_text(encoding="utf-8")
    matched_text = matched_path.read_text(encoding="utf-8")

    assert matched_path.exists()
    assert artifact_index_path.exists()
    assert experiment_record_path.exists()
    assert "tau_scalar_auroc" not in matched_text
    assert "tau_scalar_name" not in matched_text
    assert "weighted_pair_auroc" in matched_text
    assert "checkpoint_path" in record_text
    assert "analysis_summary_path" in record_text
    assert "Resolved Eval Config" in record_text
    assert "Proposition Diagnostics" in record_text
    assert "tau_scalar_auroc" not in record_text
    assert "weighted_pair_auroc" in record_text


def test_cohort_occupancy_histograms_depend_on_geometry_not_only_counts():
    records = _build_records()
    histograms_a, _, _ = _build_cohort_occupancy_histograms(records)
    shifted_records = [replace(record) for record in records]
    hard_index = 0
    for record in shifted_records:
        if record.cohort_name == "hard_id":
            if hard_index % 2 == 0:
                record.resolution_ratio = 0.1
                record.content_entropy = 2.8
            else:
                record.resolution_ratio = 0.95
                record.content_entropy = 0.15
            hard_index += 1
    histograms_b, _, _ = _build_cohort_occupancy_histograms(shifted_records)

    assert histograms_a["hard_id"].shape == (12, 12)
    assert not (histograms_a["hard_id"] == histograms_b["hard_id"]).all()
