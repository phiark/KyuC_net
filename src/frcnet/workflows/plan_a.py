from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import csv
import json
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import yaml

from frcnet.analysis import (
    write_artifact_path_list,
    write_cohort_occupancy,
    write_cohort_summary_table,
    write_experiment_record,
    write_geometry_hexbin,
    write_geometry_scatter,
)
from frcnet.data import (
    ManifestBackedVisionDataset,
    build_plan_a_manifest,
    collate_manifest_samples,
    load_plan_a_source_datasets,
    read_manifest_jsonl,
    summarize_manifest,
    write_manifest_jsonl,
    write_manifest_summary,
)
from frcnet.evaluation import (
    build_top1_proposition_records,
    read_sample_analysis_records,
    run_inference_export,
    summarize_matched_ambiguous_vs_ood,
    write_matched_benchmark_summary,
    write_sample_analysis_records,
    write_top1_proposition_records,
)
from frcnet.models import FRCNetModel
from frcnet.training import run_train_step
from frcnet.utils import resolve_pin_memory, resolve_runtime

TRAINABLE_COHORT_NAMES = frozenset({"easy_id", "hard_id", "ambiguous_id", "unknown_supervision"})


@dataclass(slots=True)
class TrainEpochSummary:
    epoch: int
    batch_count: int
    optimizer_steps: int
    trainable_samples: int
    mean_loss_total: float
    mean_loss_id: float
    mean_loss_unknown: float
    mean_loss_ambiguous: float

    def to_csv_row(self) -> dict[str, int | float]:
        return asdict(self)


def timestamp_run_id(prefix: str = "RUN") -> str:
    return datetime.now().astimezone().strftime(f"{prefix}-%Y%m%dT%H%M%S%z")


def _load_yaml_section(config_path: str | Path, section_name: str) -> dict[str, Any]:
    payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if section_name not in payload:
        raise KeyError(f"{config_path} does not contain a top-level `{section_name}` section.")
    section = payload[section_name]
    if not isinstance(section, dict):
        raise TypeError(f"{config_path}:{section_name} must decode to a mapping.")
    return section


def _write_json(payload: Mapping[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def _copy_snapshot(input_path: str | Path, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output)
    return output


def _build_model(model_config: Mapping[str, Any]) -> FRCNetModel:
    return FRCNetModel(
        num_classes=int(model_config["num_classes"]),
        backbone_name=model_config["backbone"],
        resolution_temperature=float(model_config["resolution_temperature"]),
        content_temperature=float(model_config["content_temperature"]),
    )


def _build_optimizer(model: nn.Module, optimizer_config: Mapping[str, Any]) -> Optimizer:
    optimizer_name = str(optimizer_config.get("name", "adamw")).lower()
    learning_rate = float(optimizer_config["lr"])
    weight_decay = float(optimizer_config.get("weight_decay", 0.0))

    if optimizer_name == "sgd":
        momentum = float(optimizer_config.get("momentum", 0.0))
        nesterov = bool(optimizer_config.get("nesterov", False))
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
    if optimizer_name == "adam":
        betas = tuple(float(value) for value in optimizer_config.get("betas", (0.9, 0.999)))
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas)
    if optimizer_name == "adamw":
        betas = tuple(float(value) for value in optimizer_config.get("betas", (0.9, 0.999)))
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def _model_requires_batch_size_at_least_two(model: nn.Module) -> bool:
    batchnorm_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
    )
    return any(isinstance(module, batchnorm_types) for module in model.modules())


def _validate_training_batching(dataset_size: int, batch_size: int, drop_last: bool, model: nn.Module) -> None:
    if batch_size <= 0:
        raise ValueError("Training dataloader batch_size must be positive.")
    if dataset_size <= 0:
        raise ValueError("Training manifest does not contain any records.")
    if drop_last and dataset_size < batch_size:
        raise ValueError("drop_last=True would drop the entire training manifest. Reduce batch_size or disable drop_last.")
    if batch_size < 2 and _model_requires_batch_size_at_least_two(model):
        raise ValueError(
            "Training batch_size < 2 is unsupported for BatchNorm-backed models. Increase batch_size or change backbone."
        )
    if (dataset_size % batch_size) == 1 and not drop_last and _model_requires_batch_size_at_least_two(model):
        raise ValueError(
            "The current training dataloader would emit a final singleton batch for a BatchNorm-backed model. "
            "Set drop_last=True or adjust batch_size."
        )


def _write_epoch_history(epoch_history: list[TrainEpochSummary], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(epoch_history[0].to_csv_row().keys()) if epoch_history else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for record in epoch_history:
                writer.writerow(record.to_csv_row())
    return output


def _save_checkpoint(
    output_path: str | Path,
    *,
    run_id: str,
    epoch: int,
    protocol_id: str,
    model: nn.Module,
    optimizer: Optimizer,
    epoch_summary: TrainEpochSummary,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "run_id": run_id,
            "epoch": epoch,
            "protocol_id": protocol_id,
            "epoch_summary": epoch_summary.to_csv_row(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        output,
    )
    return output


def _run_training_epoch(
    *,
    epoch: int,
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    runtime_spec,
    loss_config: Mapping[str, Any] | None,
) -> TrainEpochSummary:
    batch_count = 0
    optimizer_steps = 0
    trainable_samples = 0
    loss_total_sum = 0.0
    loss_id_sum = 0.0
    loss_unknown_sum = 0.0
    loss_ambiguous_sum = 0.0

    for batch_input in dataloader:
        batch_count += 1
        loss_breakdown = run_train_step(
            model=model,
            batch_input=batch_input,
            optimizer=optimizer,
            runtime_spec=runtime_spec,
            loss_config=loss_config,
        )
        trainable_samples += loss_breakdown.num_trainable_samples
        if loss_breakdown.optimizer_step_performed:
            optimizer_steps += 1
            loss_total_sum += float(loss_breakdown.loss_total.detach().item())
            loss_id_sum += float(loss_breakdown.loss_id.detach().item())
            loss_unknown_sum += float(loss_breakdown.loss_unknown.detach().item())
            loss_ambiguous_sum += float(loss_breakdown.loss_ambiguous.detach().item())

    denominator = max(optimizer_steps, 1)
    return TrainEpochSummary(
        epoch=epoch,
        batch_count=batch_count,
        optimizer_steps=optimizer_steps,
        trainable_samples=trainable_samples,
        mean_loss_total=loss_total_sum / denominator,
        mean_loss_id=loss_id_sum / denominator,
        mean_loss_unknown=loss_unknown_sum / denominator,
        mean_loss_ambiguous=loss_ambiguous_sum / denominator,
    )


def prepare_plan_a_datasets(
    protocol_config_paths: Iterable[str | Path],
    *,
    output_path: str | Path | None = None,
    download_override: bool | None = None,
) -> dict[str, Any]:
    report_items: list[dict[str, Any]] = []
    seen_dataset_specs: set[tuple[str, str, str, bool]] = set()

    for protocol_config_path in protocol_config_paths:
        protocol_config = _load_yaml_section(protocol_config_path, "protocol")
        datasets_config = {
            dataset_name: dict(dataset_config)
            for dataset_name, dataset_config in protocol_config["datasets"].items()
        }
        if download_override is not None:
            for dataset_config in datasets_config.values():
                dataset_config["download"] = download_override
        protocol_with_override = dict(protocol_config)
        protocol_with_override["datasets"] = datasets_config
        loaded_datasets = load_plan_a_source_datasets(protocol_with_override)

        for dataset_name, dataset_object in loaded_datasets.items():
            dataset_config = datasets_config[dataset_name]
            split_marker = "train" if dataset_name == "cifar10" else str(dataset_config.get("split", "test"))
            if dataset_name == "cifar10":
                split_marker = "train" if bool(dataset_config.get("train", False)) else "test"
            dataset_key = (
                dataset_name,
                str(Path(dataset_config["root"]).resolve()),
                split_marker,
                bool(dataset_config.get("download", False)),
            )
            if dataset_key in seen_dataset_specs:
                continue
            seen_dataset_specs.add(dataset_key)
            report_items.append(
                {
                    "dataset_name": dataset_name,
                    "root": str(Path(dataset_config["root"]).resolve()),
                    "split": split_marker,
                    "download": bool(dataset_config.get("download", False)),
                    "num_samples": len(dataset_object),
                }
            )

    report = {"datasets": sorted(report_items, key=lambda item: (item["dataset_name"], item["split"], item["root"]))}
    if output_path is not None:
        _write_json(report, output_path)
    return report


def build_plan_a_manifest_bundle(
    *,
    protocol_config_path: str | Path,
    output_dir: str | Path,
    manifest_filename: str = "plan_a_manifest.jsonl",
    summary_filename: str = "plan_a_manifest_summary.json",
) -> dict[str, str]:
    protocol_config = _load_yaml_section(protocol_config_path, "protocol")
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    source_datasets = load_plan_a_source_datasets(protocol_config)
    manifest_records = build_plan_a_manifest(protocol_config, source_datasets)
    manifest_path = write_manifest_jsonl(manifest_records, output_root / manifest_filename)
    summary_path = write_manifest_summary(manifest_records, output_root / summary_filename)

    return {
        "protocol_id": protocol_config["protocol_id"],
        "manifest_path": str(manifest_path),
        "manifest_summary_path": str(summary_path),
    }


def train_plan_a_model(
    *,
    protocol_config_path: str | Path,
    model_config_path: str | Path,
    train_config_path: str | Path,
    output_dir: str | Path,
    run_id: str | None = None,
    manifest_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict[str, str]:
    protocol_config = _load_yaml_section(protocol_config_path, "protocol")
    model_config = _load_yaml_section(model_config_path, "model")
    train_config = _load_yaml_section(train_config_path, "train")
    resolved_run_id = run_id or timestamp_run_id()

    output_root = Path(output_dir)
    snapshots_dir = output_root / "snapshots"
    manifests_dir = output_root / "manifests"
    checkpoints_dir = output_root / "checkpoints"
    records_dir = output_root / "records"
    for directory in (snapshots_dir, manifests_dir, checkpoints_dir, records_dir):
        directory.mkdir(parents=True, exist_ok=True)

    protocol_snapshot_path = _copy_snapshot(protocol_config_path, snapshots_dir / "protocol_config_snapshot.yaml")
    model_snapshot_path = _copy_snapshot(model_config_path, snapshots_dir / "model_config_snapshot.yaml")
    train_snapshot_path = _copy_snapshot(train_config_path, snapshots_dir / "train_config_snapshot.yaml")

    source_datasets = load_plan_a_source_datasets(protocol_config)
    if manifest_path is None:
        manifest_records = build_plan_a_manifest(protocol_config, source_datasets)
        manifest_snapshot_path = write_manifest_jsonl(manifest_records, manifests_dir / "train_manifest_snapshot.jsonl")
    else:
        manifest_records = read_manifest_jsonl(manifest_path)
        manifest_snapshot_path = _copy_snapshot(manifest_path, manifests_dir / "train_manifest_snapshot.jsonl")
    manifest_summary_path = write_manifest_summary(manifest_records, manifests_dir / "train_manifest_summary.json")

    num_trainable_manifest_records = sum(
        1 for record in manifest_records if record.cohort_name in TRAINABLE_COHORT_NAMES
    )
    if num_trainable_manifest_records == 0:
        raise ValueError("Training manifest does not contain any trainable cohorts.")

    model = _build_model(model_config)

    runtime_config = dict(train_config.get("runtime", {}))
    runtime_spec = resolve_runtime(
        requested_backend=runtime_config.get("backend", "auto"),
        dtype=runtime_config.get("dtype", "float32"),
        amp_enabled=bool(runtime_config.get("amp_enabled", False)),
    )
    dataloader_config = dict(protocol_config.get("analysis", {}).get("dataloader", {}))
    dataloader_config.update(train_config.get("dataloader", {}))
    batch_size = int(dataloader_config.get("batch_size", 32))
    drop_last = bool(dataloader_config.get("drop_last", True))
    _validate_training_batching(len(manifest_records), batch_size, drop_last, model)

    seed = int(train_config.get("training", {}).get("seed", protocol_config.get("seed", 7)))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)

    dataset = ManifestBackedVisionDataset(
        manifest_records=manifest_records,
        source_datasets=source_datasets,
        num_classes=int(protocol_config["num_classes"]),
    )
    num_workers = int(dataloader_config.get("num_workers", 0))
    persistent_workers = bool(dataloader_config.get("persistent_workers", False) and num_workers > 0)
    pin_memory_setting = dataloader_config.get("pin_memory", runtime_config.get("pin_memory", "auto"))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(dataloader_config.get("shuffle", True)),
        num_workers=num_workers,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        pin_memory=resolve_pin_memory(pin_memory_setting, runtime_spec),
        collate_fn=collate_manifest_samples,
        generator=generator,
    )
    if len(dataloader) == 0:
        raise ValueError("Training dataloader resolved to zero batches. Check batch_size and drop_last.")

    optimizer = _build_optimizer(model, train_config["optimizer"])
    if checkpoint_path is not None:
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint_payload.get("model_state_dict", checkpoint_payload)
        model.load_state_dict(state_dict)

    epoch_count = int(train_config.get("training", {}).get("epochs", 1))
    if epoch_count <= 0:
        raise ValueError("train.training.epochs must be positive.")
    loss_config = dict(train_config.get("loss", {}))
    checkpoint_config = dict(train_config.get("checkpointing", {}))
    save_every_epochs = int(checkpoint_config.get("save_every_epochs", 1))

    epoch_history: list[TrainEpochSummary] = []
    best_checkpoint_path = checkpoints_dir / "checkpoint_best.pt"
    last_checkpoint_path = checkpoints_dir / "checkpoint_last.pt"
    best_mean_loss: float | None = None

    for epoch in range(1, epoch_count + 1):
        epoch_summary = _run_training_epoch(
            epoch=epoch,
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            runtime_spec=runtime_spec,
            loss_config=loss_config,
        )
        epoch_history.append(epoch_summary)

        if save_every_epochs > 0 and (epoch % save_every_epochs == 0):
            _save_checkpoint(
                checkpoints_dir / f"checkpoint_epoch_{epoch:03d}.pt",
                run_id=resolved_run_id,
                epoch=epoch,
                protocol_id=protocol_config["protocol_id"],
                model=model,
                optimizer=optimizer,
                epoch_summary=epoch_summary,
            )

        _save_checkpoint(
            last_checkpoint_path,
            run_id=resolved_run_id,
            epoch=epoch,
            protocol_id=protocol_config["protocol_id"],
            model=model,
            optimizer=optimizer,
            epoch_summary=epoch_summary,
        )

        if best_mean_loss is None or epoch_summary.mean_loss_total <= best_mean_loss:
            best_mean_loss = epoch_summary.mean_loss_total
            _save_checkpoint(
                best_checkpoint_path,
                run_id=resolved_run_id,
                epoch=epoch,
                protocol_id=protocol_config["protocol_id"],
                model=model,
                optimizer=optimizer,
                epoch_summary=epoch_summary,
            )

    history_path = _write_epoch_history(epoch_history, records_dir / "train_history.csv")
    summary_path = _write_json(
        {
            "run_id": resolved_run_id,
            "protocol_id": protocol_config["protocol_id"],
            "seed": seed,
            "runtime": {
                "requested_backend": runtime_spec.requested_backend,
                "resolved_backend": runtime_spec.resolved_backend,
                "device": str(runtime_spec.device),
                "dtype": str(runtime_spec.dtype),
                "amp_enabled": runtime_spec.amp_enabled,
            },
            "manifest": {
                "path": str(manifest_snapshot_path),
                "summary_path": str(manifest_summary_path),
                "num_records": len(manifest_records),
                "num_trainable_records": num_trainable_manifest_records,
                "cohort_summary": summarize_manifest(manifest_records),
            },
            "snapshots": {
                "protocol_config_snapshot": str(protocol_snapshot_path),
                "model_config_snapshot": str(model_snapshot_path),
                "train_config_snapshot": str(train_snapshot_path),
            },
            "checkpoints": {
                "best": str(best_checkpoint_path),
                "last": str(last_checkpoint_path),
            },
            "history_path": str(history_path),
            "epochs": [record.to_csv_row() for record in epoch_history],
        },
        records_dir / "train_summary.json",
    )

    return {
        "run_id": resolved_run_id,
        "protocol_id": protocol_config["protocol_id"],
        "output_dir": str(output_root),
        "manifest_path": str(manifest_snapshot_path),
        "manifest_summary_path": str(manifest_summary_path),
        "history_path": str(history_path),
        "summary_path": str(summary_path),
        "best_checkpoint_path": str(best_checkpoint_path),
        "last_checkpoint_path": str(last_checkpoint_path),
    }


def export_plan_a_inference_bundle(
    *,
    protocol_config_path: str | Path,
    model_config_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    run_id: str | None = None,
    checkpoint_path: str | Path | None = None,
    batch_size: int | None = None,
) -> dict[str, str]:
    protocol_config = _load_yaml_section(protocol_config_path, "protocol")
    model_config = _load_yaml_section(model_config_path, "model")
    resolved_run_id = run_id or timestamp_run_id()

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    protocol_snapshot_path = _copy_snapshot(protocol_config_path, output_root / "protocol_config_snapshot.yaml")
    model_snapshot_path = _copy_snapshot(model_config_path, output_root / "model_config_snapshot.yaml")
    manifest_snapshot_path = _copy_snapshot(manifest_path, output_root / "plan_a_manifest_snapshot.jsonl")

    runtime_spec = resolve_runtime(requested_backend="auto")
    source_datasets = load_plan_a_source_datasets(protocol_config)
    manifest_records = read_manifest_jsonl(manifest_path)
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
    )
    proposition_records = build_top1_proposition_records(sample_analysis_records)

    analysis_path = write_sample_analysis_records(sample_analysis_records, output_root / "sample_analysis_records.csv")
    proposition_path = write_top1_proposition_records(
        proposition_records,
        output_root / "top1_proposition_records.csv",
    )

    return {
        "run_id": resolved_run_id,
        "protocol_id": protocol_config["protocol_id"],
        "output_dir": str(output_root),
        "protocol_snapshot_path": str(protocol_snapshot_path),
        "model_snapshot_path": str(model_snapshot_path),
        "manifest_snapshot_path": str(manifest_snapshot_path),
        "analysis_path": str(analysis_path),
        "proposition_path": str(proposition_path),
    }


def generate_plan_a_artifact_bundle(
    *,
    analysis_path: str | Path,
    protocol_config_path: str | Path,
    eval_config_path: str | Path,
    output_dir: str | Path,
    analysis_config_path: str | Path | None = None,
) -> dict[str, str]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    protocol_snapshot_path = _copy_snapshot(protocol_config_path, output_root / "protocol_config_snapshot.yaml")
    eval_snapshot_path = _copy_snapshot(eval_config_path, output_root / "eval_config_snapshot.yaml")
    analysis_config = {"figure_dpi": 200}
    if analysis_config_path is not None:
        analysis_payload = _load_yaml_section(analysis_config_path, "analysis")
        analysis_config.update(analysis_payload)
        _copy_snapshot(analysis_config_path, output_root / "analysis_config_snapshot.yaml")

    analysis_record_path = Path(analysis_path)
    for sibling_name in ("model_config_snapshot.yaml", "plan_a_manifest_snapshot.jsonl", "top1_proposition_records.csv"):
        sibling_path = analysis_record_path.parent / sibling_name
        if sibling_path.exists() and sibling_path.parent != output_root:
            shutil.copy2(sibling_path, output_root / sibling_name)

    sample_analysis_records = read_sample_analysis_records(analysis_record_path)
    if not sample_analysis_records:
        raise ValueError("analysis-path does not contain any sample analysis records.")
    run_id = sample_analysis_records[0].run_id
    protocol_id = sample_analysis_records[0].protocol_id
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
    summary_path = write_cohort_summary_table(
        sample_analysis_records,
        output_root / analysis_config.get("cohort_summary_table_name", "cohort_summary_table.csv"),
    )
    matched_summary = summarize_matched_ambiguous_vs_ood(sample_analysis_records)
    matched_path = write_matched_benchmark_summary(
        matched_summary,
        output_root / analysis_config.get("matched_table_name", "matched_ambiguous_vs_ood_table.csv"),
    )

    artifact_paths = {
        "geometry_scatter": str(scatter_path),
        "geometry_hexbin": str(hexbin_path),
        "cohort_occupancy": str(occupancy_path),
        "cohort_summary_table": str(summary_path),
        "matched_ambiguous_vs_ood_table": str(matched_path),
    }
    artifact_index_path = write_artifact_path_list(artifact_paths, output_root / "artifact_paths.json")
    experiment_record_path = write_experiment_record(
        output_path=output_root / "experiment_record.md",
        run_id=run_id,
        protocol_id=protocol_id,
        config_snapshot_paths={
            "protocol_config_snapshot": str(protocol_snapshot_path),
            "eval_config_snapshot": str(eval_snapshot_path),
            "model_config_snapshot": str(output_root / "model_config_snapshot.yaml"),
        },
        manifest_snapshot_path=str(output_root / "plan_a_manifest_snapshot.jsonl"),
        analysis_record_path=str(analysis_record_path),
        proposition_record_path=str(output_root / "top1_proposition_records.csv"),
        artifact_paths={**artifact_paths, "artifact_paths": str(artifact_index_path)},
        matched_summary=matched_summary,
    )

    return {
        "run_id": run_id,
        "protocol_id": protocol_id,
        "output_dir": str(output_root),
        "protocol_snapshot_path": str(protocol_snapshot_path),
        "eval_snapshot_path": str(eval_snapshot_path),
        "artifact_index_path": str(artifact_index_path),
        "experiment_record_path": str(experiment_record_path),
        **artifact_paths,
    }


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
) -> dict[str, str]:
    resolved_run_id = run_id or timestamp_run_id()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    data_preflight_path = output_root / "data_preflight.json"
    prepare_plan_a_datasets(
        [train_protocol_config_path, analysis_protocol_config_path],
        output_path=data_preflight_path,
        download_override=download_override,
    )

    train_outputs = train_plan_a_model(
        protocol_config_path=train_protocol_config_path,
        model_config_path=model_config_path,
        train_config_path=train_config_path,
        output_dir=output_root / "training",
        run_id=resolved_run_id,
    )

    analysis_manifest_outputs = build_plan_a_manifest_bundle(
        protocol_config_path=analysis_protocol_config_path,
        output_dir=output_root / "analysis_manifest",
        manifest_filename="plan_a_manifest.jsonl",
        summary_filename="plan_a_manifest_summary.json",
    )
    inference_outputs = export_plan_a_inference_bundle(
        protocol_config_path=analysis_protocol_config_path,
        model_config_path=model_config_path,
        manifest_path=analysis_manifest_outputs["manifest_path"],
        output_dir=output_root / "analysis",
        run_id=resolved_run_id,
        checkpoint_path=train_outputs["best_checkpoint_path"],
    )
    artifact_outputs = generate_plan_a_artifact_bundle(
        analysis_path=inference_outputs["analysis_path"],
        protocol_config_path=analysis_protocol_config_path,
        eval_config_path=eval_config_path,
        analysis_config_path=analysis_config_path,
        output_dir=output_root / "report",
    )

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
        "best_checkpoint_path": train_outputs["best_checkpoint_path"],
        "analysis_manifest_path": analysis_manifest_outputs["manifest_path"],
        "analysis_path": inference_outputs["analysis_path"],
        "artifact_index_path": artifact_outputs["artifact_index_path"],
        "experiment_record_path": artifact_outputs["experiment_record_path"],
        "bundle_path": str(bundle_path),
    }
