from __future__ import annotations

from pathlib import Path

from frcnet.analysis import (
    write_artifact_path_list,
    write_completion_scan_table,
    write_cohort_counts,
    write_cohort_occupancy,
    write_cohort_summary_table,
    write_experiment_record,
    write_geometry_hexbin,
    write_geometry_scatter,
    write_proposition_tau_cohort_boxplot,
    write_proposition_diagnostic_table,
    write_scalar_roc_curve,
)
from frcnet.data import (
    read_manifest_jsonl,
    validate_manifest_records,
)
from frcnet.evaluation import (
    DEFAULT_MODEL_FAMILY,
    read_matched_manifest_jsonl,
    read_sample_analysis_records,
    read_top1_proposition_records,
    summarize_matched_ambiguous_vs_ood,
    write_matched_benchmark_summary,
)

from frcnet.workflows.workflow_io import (
    _copy_optional_snapshot,
    _copy_snapshot,
    _finalize_integrity_errors,
    _inspect_proposition_records,
    _inspect_sample_analysis_records,
    _load_yaml_section,
    _resolve_analysis_sidecars,
    _resolve_eval_config,
)


def generate_plan_a_artifact_bundle(
    *,
    analysis_path: str | Path,
    protocol_config_path: str | Path,
    eval_config_path: str | Path,
    output_dir: str | Path,
    analysis_config_path: str | Path | None = None,
    analysis_summary_path: str | Path | None = None,
    allow_integrity_override: bool = False,
) -> dict[str, str]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    protocol_snapshot_path = _copy_snapshot(protocol_config_path, output_root / "protocol_config_snapshot.yaml")
    eval_snapshot_path = _copy_snapshot(eval_config_path, output_root / "eval_config_snapshot.yaml")
    resolved_eval_config = _resolve_eval_config(eval_config_path)
    analysis_config = {
        "figure_dpi": 200,
        "cohort_counts_name": "cohort_counts.png",
        "proposition_tau_cohort_boxplot_name": "proposition_tau_cohort_boxplot.png",
        "proposition_diagnostic_table_name": "proposition_diagnostic_table.csv",
        "proposition_tau_roc_curve_name": "proposition_tau_roc_curve.png",
        "proposition_diagnostic_scalars": (
            "proposition_truth_ratio",
            "resolution_entropy",
            "ternary_entropy",
            "auxiliary_top1_content_probability",
        ),
    }
    analysis_snapshot_path: Path | None = None
    if analysis_config_path is not None:
        analysis_payload = _load_yaml_section(analysis_config_path, "analysis")
        analysis_config.update(analysis_payload)
        analysis_snapshot_path = _copy_snapshot(analysis_config_path, output_root / "analysis_config_snapshot.yaml")

    analysis_record_path = Path(analysis_path)
    sample_analysis_records = read_sample_analysis_records(analysis_record_path)
    if not sample_analysis_records:
        raise ValueError("analysis-path does not contain any sample analysis records.")

    analysis_state = _inspect_sample_analysis_records(sample_analysis_records)
    integrity_errors: list[str] = []
    if analysis_state.duplicate_sample_ids:
        integrity_errors.append("analysis_duplicate_sample_ids")
    if len(analysis_state.run_ids) != 1:
        integrity_errors.append("analysis_mixed_run_ids")
    if len(analysis_state.protocol_ids) != 1:
        integrity_errors.append("analysis_mixed_protocol_ids")

    sidecars, sidecar_errors = _resolve_analysis_sidecars(analysis_record_path, analysis_summary_path)
    integrity_errors.extend(sidecar_errors)
    if sidecars.summary_run_id is not None and sidecars.summary_run_id not in analysis_state.run_ids:
        integrity_errors.append("analysis_summary_run_id_mismatch")
    if sidecars.summary_protocol_id is not None and sidecars.summary_protocol_id not in analysis_state.protocol_ids:
        integrity_errors.append("analysis_summary_protocol_id_mismatch")

    sidecar_prefix = "legacy" if sidecars.sidecar_resolution_mode == "legacy_sibling" else "analysis_summary"
    manifest_records = None
    if sidecars.manifest_snapshot_path is None or not Path(sidecars.manifest_snapshot_path).exists():
        integrity_errors.append(f"{sidecar_prefix}_manifest_snapshot_missing")
    else:
        try:
            manifest_records = validate_manifest_records(read_manifest_jsonl(sidecars.manifest_snapshot_path))
        except ValueError:
            integrity_errors.append("manifest_contract_violation")
        else:
            manifest_protocol_ids = sorted({record.protocol_id for record in manifest_records})
            if set(manifest_protocol_ids) != set(analysis_state.protocol_ids):
                integrity_errors.append("manifest_protocol_id_mismatch")
            if {record.sample_id for record in manifest_records} != set(analysis_state.sample_ids):
                integrity_errors.append("manifest_sample_id_mismatch")

    proposition_records = None
    if sidecars.proposition_path is None or not Path(sidecars.proposition_path).exists():
        integrity_errors.append(f"{sidecar_prefix}_proposition_records_missing")
    else:
        proposition_records = read_top1_proposition_records(sidecars.proposition_path)
        proposition_state = _inspect_proposition_records(proposition_records)
        if proposition_state.duplicate_sample_ids:
            integrity_errors.append("proposition_duplicate_sample_ids")
        if proposition_state.sample_ids and not proposition_state.sample_ids.issubset(analysis_state.sample_ids):
            integrity_errors.append("proposition_sample_id_outside_analysis")
        if len(proposition_state.run_ids) > 1:
            integrity_errors.append("proposition_mixed_run_ids")
        if len(proposition_state.protocol_ids) > 1:
            integrity_errors.append("proposition_mixed_protocol_ids")
        if proposition_state.run_ids and not set(proposition_state.run_ids).issubset(set(analysis_state.run_ids)):
            integrity_errors.append("proposition_run_id_mismatch")
        if proposition_state.protocol_ids and not set(proposition_state.protocol_ids).issubset(set(analysis_state.protocol_ids)):
            integrity_errors.append("proposition_protocol_id_mismatch")

    if sidecars.model_config_snapshot_path is None or not Path(sidecars.model_config_snapshot_path).exists():
        integrity_errors.append(f"{sidecar_prefix}_model_config_snapshot_missing")

    integrity_overrides = _finalize_integrity_errors(
        integrity_errors,
        allow_integrity_override=allow_integrity_override,
        inherited_overrides=sidecars.inherited_integrity_overrides,
    )

    analysis_summary_copy_path = _copy_optional_snapshot(sidecars.analysis_summary_path, output_root / "analysis_summary.json")
    manifest_snapshot_copy_path = _copy_optional_snapshot(
        sidecars.manifest_snapshot_path,
        output_root / "plan_a_manifest_snapshot.jsonl",
    )
    proposition_copy_path = _copy_optional_snapshot(
        sidecars.proposition_path,
        output_root / "top1_proposition_records.csv",
    )
    model_snapshot_copy_path = _copy_optional_snapshot(
        sidecars.model_config_snapshot_path,
        output_root / "model_config_snapshot.yaml",
    )
    checkpoint_selection_summary_copy_path = _copy_optional_snapshot(
        sidecars.checkpoint_selection_summary_path,
        output_root / "checkpoint_selection_summary.json",
    )

    run_id = analysis_state.run_ids[0] if len(analysis_state.run_ids) == 1 else "MULTIPLE"
    protocol_id = analysis_state.protocol_ids[0] if len(analysis_state.protocol_ids) == 1 else "MULTIPLE"
    model_family = (
        sidecars.model_family
        if sidecars.model_family != DEFAULT_MODEL_FAMILY or not sample_analysis_records
        else sample_analysis_records[0].model_family
    )
    figure_dpi = int(analysis_config.get("figure_dpi", 200))

    scatter_path = write_geometry_scatter(
        sample_analysis_records,
        output_root / analysis_config.get("geometry_scatter_name", "geometry_scatter.png"),
        dpi=figure_dpi,
    )
    hexbin_path = write_geometry_hexbin(
        sample_analysis_records,
        output_root / analysis_config.get("geometry_hexbin_name", "geometry_hexbin.png"),
        dpi=figure_dpi,
    )
    occupancy_path = write_cohort_occupancy(
        sample_analysis_records,
        output_root / analysis_config.get("cohort_occupancy_name", "cohort_occupancy.png"),
        dpi=figure_dpi,
    )
    cohort_counts_path = write_cohort_counts(
        sample_analysis_records,
        output_root / analysis_config.get("cohort_counts_name", "cohort_counts.png"),
        dpi=figure_dpi,
    )
    tau_boxplot_name = analysis_config.get(
        "proposition_tau_cohort_boxplot_name",
        analysis_config.get("tau_cohort_boxplot_name", "proposition_tau_cohort_boxplot.png"),
    )
    tau_boxplot_path = write_proposition_tau_cohort_boxplot(
        sample_analysis_records,
        output_root / tau_boxplot_name,
        dpi=figure_dpi,
    )
    summary_path = write_cohort_summary_table(
        sample_analysis_records,
        output_root / analysis_config.get("cohort_summary_table_name", "cohort_summary_table.csv"),
    )
    matched_manifest_records = None
    matched_manifest_path_value = str(resolved_eval_config.get("matched_manifest_path", ""))
    if matched_manifest_path_value:
        matched_manifest_path = Path(matched_manifest_path_value)
        if not matched_manifest_path.is_absolute():
            matched_manifest_path = Path.cwd() / matched_manifest_path
        matched_manifest_records = tuple(read_matched_manifest_jsonl(matched_manifest_path))
    matched_summary = summarize_matched_ambiguous_vs_ood(
        sample_analysis_records,
        positive_cohort=str(resolved_eval_config["positive_cohort"]),
        negative_cohort=str(resolved_eval_config["negative_cohort"]),
        primary_pair=str(resolved_eval_config["primary_pair"]),
        weighted_pair=str(resolved_eval_config["weighted_pair"]),
        primary_scalar=str(resolved_eval_config["primary_scalar"]),
        completion_scan_scalars=tuple(resolved_eval_config["completion_scan_scalars"]),
        test_size=float(resolved_eval_config["test_size"]),
        random_state=int(resolved_eval_config["random_state"]),
        matched_manifest_records=matched_manifest_records,
    )
    matched_path = write_matched_benchmark_summary(
        matched_summary,
        output_root / analysis_config.get("matched_table_name", "matched_ambiguous_vs_ood_table.csv"),
    )
    completion_scan_path = write_completion_scan_table(
        sample_analysis_records,
        output_root / analysis_config.get("completion_scan_table_name", "completion_scan_table.csv"),
        positive_cohort=str(resolved_eval_config["positive_cohort"]),
        negative_cohort=str(resolved_eval_config["negative_cohort"]),
        scalar_names=tuple(resolved_eval_config["completion_scan_scalars"]),
        test_size=float(resolved_eval_config["test_size"]),
        random_state=int(resolved_eval_config["random_state"]),
        matched_manifest_records=matched_manifest_records,
    )
    proposition_diagnostic_table_path: Path | None = None
    proposition_tau_roc_curve_path: Path | None = None
    if bool(resolved_eval_config["emit_proposition_diagnostics"]):
        proposition_diagnostic_table_path = write_proposition_diagnostic_table(
            sample_analysis_records,
            output_root / analysis_config.get("proposition_diagnostic_table_name", "proposition_diagnostic_table.csv"),
            positive_cohort=str(resolved_eval_config["positive_cohort"]),
            negative_cohort=str(resolved_eval_config["negative_cohort"]),
            scalar_names=tuple(analysis_config.get("proposition_diagnostic_scalars", ())),
            test_size=float(resolved_eval_config["test_size"]),
            random_state=int(resolved_eval_config["random_state"]),
            matched_manifest_records=matched_manifest_records,
        )
        proposition_tau_roc_curve_path = write_scalar_roc_curve(
            sample_analysis_records,
            output_root / analysis_config.get("proposition_tau_roc_curve_name", "proposition_tau_roc_curve.png"),
            positive_cohort=str(resolved_eval_config["positive_cohort"]),
            negative_cohort=str(resolved_eval_config["negative_cohort"]),
            scalar_name=str(resolved_eval_config["tau_scalar_name"]),
            test_size=float(resolved_eval_config["test_size"]),
            random_state=int(resolved_eval_config["random_state"]),
            matched_manifest_records=matched_manifest_records,
            dpi=figure_dpi,
        )

    artifact_paths = {
        "geometry_scatter": str(scatter_path),
        "geometry_hexbin": str(hexbin_path),
        "cohort_occupancy": str(occupancy_path),
        "cohort_counts": str(cohort_counts_path),
        "proposition_tau_cohort_boxplot": str(tau_boxplot_path),
        "cohort_summary_table": str(summary_path),
        "matched_ambiguous_vs_ood_table": str(matched_path),
        "completion_scan_table": str(completion_scan_path),
    }
    if proposition_diagnostic_table_path is not None:
        artifact_paths["proposition_diagnostic_table"] = str(proposition_diagnostic_table_path)
    if proposition_tau_roc_curve_path is not None:
        artifact_paths["proposition_tau_roc_curve"] = str(proposition_tau_roc_curve_path)
    if checkpoint_selection_summary_copy_path is not None:
        artifact_paths["checkpoint_selection_summary"] = str(checkpoint_selection_summary_copy_path)
    artifact_index_path = write_artifact_path_list(artifact_paths, output_root / "artifact_paths.json")
    config_snapshot_paths = {
        "protocol_config_snapshot": str(protocol_snapshot_path),
        "eval_config_snapshot": str(eval_snapshot_path),
        "model_config_snapshot": str(
            model_snapshot_copy_path
            or sidecars.model_config_snapshot_path
            or (output_root / "model_config_snapshot.yaml")
        ),
    }
    if analysis_snapshot_path is not None:
        config_snapshot_paths["analysis_config_snapshot"] = str(analysis_snapshot_path)
    experiment_record_path = write_experiment_record(
        output_path=output_root / "experiment_record.md",
        model_family=model_family,
        run_id=run_id,
        protocol_id=protocol_id,
        config_snapshot_paths=config_snapshot_paths,
        manifest_snapshot_path=str(
            manifest_snapshot_copy_path
            or sidecars.manifest_snapshot_path
            or (output_root / "plan_a_manifest_snapshot.jsonl")
        ),
        analysis_record_path=str(analysis_record_path),
        proposition_record_path=str(
            proposition_copy_path
            or sidecars.proposition_path
            or (output_root / "top1_proposition_records.csv")
        ),
        artifact_paths={**artifact_paths, "artifact_paths": str(artifact_index_path)},
        matched_summary=matched_summary,
        checkpoint_path=sidecars.checkpoint_path,
        checkpoint_selection_summary_path=str(
            checkpoint_selection_summary_copy_path or sidecars.checkpoint_selection_summary_path or ""
        ),
        analysis_summary_path=str(analysis_summary_copy_path or sidecars.analysis_summary_path or ""),
        sidecar_resolution_mode=sidecars.sidecar_resolution_mode,
        integrity_overrides=integrity_overrides,
        source_run_ids=analysis_state.run_ids,
        source_protocol_ids=analysis_state.protocol_ids,
        resolved_eval_config=resolved_eval_config,
        proposition_diagnostic_scalar_name=str(resolved_eval_config["tau_scalar_name"]),
        proposition_diagnostic_table_path=str(proposition_diagnostic_table_path or ""),
        proposition_tau_roc_curve_path=str(proposition_tau_roc_curve_path or ""),
    )

    return {
        "run_id": run_id,
        "protocol_id": protocol_id,
        "output_dir": str(output_root),
        "protocol_snapshot_path": str(protocol_snapshot_path),
        "eval_snapshot_path": str(eval_snapshot_path),
        "analysis_summary_path": str(analysis_summary_copy_path or sidecars.analysis_summary_path or ""),
        "artifact_index_path": str(artifact_index_path),
        "experiment_record_path": str(experiment_record_path),
        **artifact_paths,
    }
