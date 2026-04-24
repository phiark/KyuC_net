from __future__ import annotations

from pathlib import Path
from typing import Callable

from frcnet.workflows.plan_a_inference import export_plan_a_inference_bundle
from frcnet.workflows.plan_a_reporting import generate_plan_a_artifact_bundle
from frcnet.workflows.plan_a_training import (
    build_plan_a_manifest_bundle,
    prepare_plan_a_datasets,
    train_plan_a_model,
)
from frcnet.workflows.workflow_io import _emit_progress, _write_json, timestamp_run_id


def write_plan_a_experiment_bundle(
    *,
    train_protocol_config_path: str | Path,
    analysis_protocol_config_path: str | Path,
    model_config_path: str | Path,
    train_config_path: str | Path,
    eval_config_path: str | Path,
    analysis_config_path: str | Path,
    output_dir: str | Path,
    run_id: str | None = None,
    download_override: bool | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, str]:
    resolved_run_id = run_id or timestamp_run_id()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    _emit_progress(progress_callback, f"[experiment] run_id={resolved_run_id} output_dir={output_root}")
    data_preflight_path = output_root / "data_preflight.json"
    prepare_plan_a_datasets(
        [train_protocol_config_path, analysis_protocol_config_path],
        output_path=data_preflight_path,
        download_override=download_override,
    )
    _emit_progress(progress_callback, f"[experiment] data_preflight={data_preflight_path}")

    analysis_manifest_outputs = build_plan_a_manifest_bundle(
        protocol_config_path=analysis_protocol_config_path,
        output_dir=output_root / "analysis_manifest",
        manifest_filename="plan_a_manifest.jsonl",
        summary_filename="plan_a_manifest_summary.json",
    )
    _emit_progress(progress_callback, f"[experiment] analysis_manifest={analysis_manifest_outputs['manifest_path']}")

    train_outputs = train_plan_a_model(
        protocol_config_path=train_protocol_config_path,
        model_config_path=model_config_path,
        train_config_path=train_config_path,
        output_dir=output_root / "training",
        run_id=resolved_run_id,
        validation_protocol_config_path=analysis_protocol_config_path,
        validation_manifest_path=analysis_manifest_outputs["manifest_path"],
        eval_config_path=eval_config_path,
        progress_callback=progress_callback,
    )
    _emit_progress(progress_callback, f"[experiment] training_complete best_checkpoint={train_outputs['best_checkpoint_path']}")

    inference_outputs = export_plan_a_inference_bundle(
        protocol_config_path=analysis_protocol_config_path,
        model_config_path=model_config_path,
        manifest_path=analysis_manifest_outputs["manifest_path"],
        output_dir=output_root / "analysis",
        run_id=resolved_run_id,
        checkpoint_path=train_outputs["best_checkpoint_path"],
        checkpoint_selection_summary_path=train_outputs["checkpoint_selection_summary_path"],
        model_family=train_outputs["model_family"],
    )
    _emit_progress(progress_callback, f"[experiment] analysis_csv={inference_outputs['analysis_path']}")

    artifact_outputs = generate_plan_a_artifact_bundle(
        analysis_path=inference_outputs["analysis_path"],
        analysis_summary_path=inference_outputs["analysis_summary_path"],
        protocol_config_path=analysis_protocol_config_path,
        eval_config_path=eval_config_path,
        analysis_config_path=analysis_config_path,
        output_dir=output_root / "report",
    )
    _emit_progress(progress_callback, f"[experiment] report_record={artifact_outputs['experiment_record_path']}")

    bundle_path = _write_json(
        {
            "run_id": resolved_run_id,
            "data_preflight_path": str(data_preflight_path),
            "train": train_outputs,
            "analysis_manifest": analysis_manifest_outputs,
            "analysis": inference_outputs,
            "report": artifact_outputs,
        },
        output_root / "experiment_paths.json",
    )

    return {
        "run_id": resolved_run_id,
        "output_dir": str(output_root),
        "data_preflight_path": str(data_preflight_path),
        "train_summary_path": train_outputs["summary_path"],
        "validation_manifest_path": train_outputs["validation_manifest_path"],
        "best_checkpoint_path": train_outputs["best_checkpoint_path"],
        "checkpoint_selection_summary_path": train_outputs["checkpoint_selection_summary_path"],
        "analysis_manifest_path": analysis_manifest_outputs["manifest_path"],
        "analysis_path": inference_outputs["analysis_path"],
        "analysis_summary_path": inference_outputs["analysis_summary_path"],
        "artifact_index_path": artifact_outputs["artifact_index_path"],
        "experiment_record_path": artifact_outputs["experiment_record_path"],
        "bundle_path": str(bundle_path),
    }
