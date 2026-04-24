from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from frcnet.data import (
    ManifestBackedVisionDataset,
    collate_manifest_samples,
    load_plan_a_source_datasets,
    read_manifest_jsonl,
    validate_manifest_records,
)
from frcnet.evaluation import (
    AnalysisExportSummary,
    DEFAULT_MODEL_FAMILY,
    build_proposition_view_records,
    build_top1_proposition_records,
    run_inference_export,
    write_analysis_export_summary,
    write_proposition_view_records,
    write_sample_analysis_records,
    write_top1_proposition_records,
)
from frcnet.workflows.plan_a_training_core import _build_model
from frcnet.workflows.workflow_io import _copy_snapshot, _load_yaml_section, timestamp_run_id
from frcnet.utils import resolve_runtime

def export_plan_a_inference_bundle(
    *,
    protocol_config_path: str | Path,
    model_config_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    run_id: str | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_selection_summary_path: str | Path | None = None,
    batch_size: int | None = None,
    allow_missing_checkpoint: bool = False,
    model_family: str = DEFAULT_MODEL_FAMILY,
) -> dict[str, str]:
    protocol_config = _load_yaml_section(protocol_config_path, "protocol")
    model_config = _load_yaml_section(model_config_path, "model")
    resolved_run_id = run_id or timestamp_run_id()

    integrity_overrides: list[str] = []
    if checkpoint_path is None and not allow_missing_checkpoint:
        raise ValueError("analysis export requires checkpoint_path unless allow_missing_checkpoint=True.")
    if checkpoint_path is None and allow_missing_checkpoint:
        integrity_overrides.append("missing_checkpoint")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    protocol_snapshot_path = _copy_snapshot(protocol_config_path, output_root / "protocol_config_snapshot.yaml")
    model_snapshot_path = _copy_snapshot(model_config_path, output_root / "model_config_snapshot.yaml")

    runtime_spec = resolve_runtime(requested_backend="auto")
    source_datasets = load_plan_a_source_datasets(protocol_config)
    manifest_records = validate_manifest_records(read_manifest_jsonl(manifest_path))
    manifest_snapshot_path = _copy_snapshot(manifest_path, output_root / "plan_a_manifest_snapshot.jsonl")
    dataset = ManifestBackedVisionDataset(
        manifest_records=manifest_records,
        source_datasets=source_datasets,
        num_classes=int(protocol_config["num_classes"]),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size or int(protocol_config["analysis"]["dataloader"]["batch_size"]),
        shuffle=False,
        num_workers=int(protocol_config["analysis"]["dataloader"]["num_workers"]),
        drop_last=False,
        collate_fn=collate_manifest_samples,
    )

    model = _build_model(model_config)
    if checkpoint_path is not None:
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint_payload.get("model_state_dict", checkpoint_payload)
        model.load_state_dict(state_dict)

    sample_analysis_records = run_inference_export(
        model=model,
        dataloader=dataloader,
        runtime_spec=runtime_spec,
        run_id=resolved_run_id,
        protocol_id=protocol_config["protocol_id"],
        model_family=model_family,
    )
    proposition_records = build_top1_proposition_records(sample_analysis_records)
    proposition_view_records = build_proposition_view_records(sample_analysis_records)

    analysis_path = write_sample_analysis_records(sample_analysis_records, output_root / "sample_analysis_records.csv")
    proposition_path = write_top1_proposition_records(
        proposition_records,
        output_root / "top1_proposition_records.csv",
    )
    proposition_view_path = write_proposition_view_records(
        proposition_view_records,
        output_root / "proposition_view_records.csv",
    )
    analysis_summary_path = write_analysis_export_summary(
        AnalysisExportSummary(
            run_id=resolved_run_id,
            protocol_id=protocol_config["protocol_id"],
            analysis_path=str(analysis_path),
            checkpoint_path=None if checkpoint_path is None else str(checkpoint_path),
            manifest_snapshot_path=str(manifest_snapshot_path),
            model_config_snapshot_path=str(model_snapshot_path),
            proposition_path=str(proposition_path),
            checkpoint_selection_summary_path=(
                None
                if checkpoint_selection_summary_path is None
                else str(checkpoint_selection_summary_path)
            ),
            model_family=model_family,
            integrity_overrides=tuple(integrity_overrides),
            sidecar_resolution_mode="analysis_summary",
        ),
        output_root / "analysis_summary.json",
    )

    return {
        "model_family": model_family,
        "run_id": resolved_run_id,
        "protocol_id": protocol_config["protocol_id"],
        "output_dir": str(output_root),
        "protocol_snapshot_path": str(protocol_snapshot_path),
        "model_snapshot_path": str(model_snapshot_path),
        "manifest_snapshot_path": str(manifest_snapshot_path),
        "analysis_path": str(analysis_path),
        "proposition_path": str(proposition_path),
        "proposition_view_path": str(proposition_view_path),
        "analysis_summary_path": str(analysis_summary_path),
        "checkpoint_selection_summary_path": ""
        if checkpoint_selection_summary_path is None
        else str(checkpoint_selection_summary_path),
    }
