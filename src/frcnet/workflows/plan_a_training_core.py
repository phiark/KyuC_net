from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from frcnet.data import (
    ManifestBackedVisionDataset,
    collate_manifest_samples,
)
from frcnet.evaluation import (
    build_top1_proposition_records,
    run_inference_export,
    summarize_matched_ambiguous_vs_ood,
)
from frcnet.models import FRCNetModel
from frcnet.training import run_train_step
from frcnet.utils import resolve_pin_memory
from frcnet.workflows.workflow_io import _emit_progress, _progress_bar

TRAINABLE_COHORT_NAMES = frozenset({"easy_id", "hard_id", "ambiguous_id", "unknown_supervision"})


@dataclass(slots=True)
class TrainEpochSummary:
    epoch: int
    phase_name: str
    learning_rate: float
    batch_count: int
    optimizer_steps: int
    trainable_samples: int
    mean_loss_total: float
    mean_loss_id: float
    mean_loss_unknown: float
    mean_loss_ambiguous: float

    def to_csv_row(self) -> dict[str, int | float]:
        return asdict(self)

@dataclass(slots=True)
class ValidationEpochSummary:
    epoch: int
    phase_name: str
    pair_auroc: float
    weighted_pair_auroc: float
    scalar_auroc: float
    easy_id_top1_accuracy: float
    hard_id_top1_accuracy: float
    ambiguous_candidate_hit_rate: float

    def to_csv_row(self) -> dict[str, int | float]:
        return {
            "epoch": self.epoch,
            "phase_name": self.phase_name,
            "pair_auroc": self.pair_auroc,
            "weighted_pair_auroc": self.weighted_pair_auroc,
            "scalar_auroc": self.scalar_auroc,
            "easy_id_top1_accuracy": self.easy_id_top1_accuracy,
            "hard_id_top1_accuracy": self.hard_id_top1_accuracy,
            "ambiguous_candidate_hit_rate": self.ambiguous_candidate_hit_rate,
        }


def _filter_manifest_records_by_cohorts(
    manifest_records: Sequence,
    enabled_cohorts: Sequence[str],
) -> list:
    enabled_cohort_set = set(enabled_cohorts)
    return [record for record in manifest_records if record.cohort_name in enabled_cohort_set]


def _build_manifest_dataloader(
    *,
    manifest_records: Sequence,
    source_datasets: Mapping[str, object],
    num_classes: int,
    dataloader_config: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    runtime_spec,
    model: nn.Module,
    generator_seed: int | None,
    force_shuffle: bool | None = None,
    force_drop_last: bool | None = None,
) -> DataLoader:
    batch_size = int(dataloader_config.get("batch_size", 32))
    shuffle = bool(dataloader_config.get("shuffle", True)) if force_shuffle is None else bool(force_shuffle)
    drop_last = bool(dataloader_config.get("drop_last", True)) if force_drop_last is None else bool(force_drop_last)
    _validate_training_batching(len(manifest_records), batch_size, drop_last, model)

    generator = None
    if generator_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(generator_seed)

    dataset = ManifestBackedVisionDataset(
        manifest_records=list(manifest_records),
        source_datasets=source_datasets,
        num_classes=num_classes,
    )
    num_workers = int(dataloader_config.get("num_workers", 0))
    persistent_workers = bool(dataloader_config.get("persistent_workers", False) and num_workers > 0)
    pin_memory_setting = dataloader_config.get("pin_memory", runtime_config.get("pin_memory", "auto"))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        pin_memory=resolve_pin_memory(pin_memory_setting, runtime_spec),
        collate_fn=collate_manifest_samples,
        generator=generator,
    )


def _build_validation_dataloader(
    *,
    manifest_records: Sequence,
    source_datasets: Mapping[str, object],
    num_classes: int,
    protocol_config: Mapping[str, Any],
    train_config: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    runtime_spec,
) -> DataLoader:
    validation_config = dict(train_config.get("validation", {}))
    dataloader_config = dict(protocol_config.get("analysis", {}).get("dataloader", {}))
    dataloader_config.update(train_config.get("dataloader", {}))
    dataloader_config.update(validation_config.get("dataloader", {}))
    batch_size = int(dataloader_config.get("batch_size", 32))
    num_workers = int(dataloader_config.get("num_workers", 0))
    persistent_workers = bool(dataloader_config.get("persistent_workers", False) and num_workers > 0)
    pin_memory_setting = dataloader_config.get("pin_memory", runtime_config.get("pin_memory", "auto"))
    dataset = ManifestBackedVisionDataset(
        manifest_records=list(manifest_records),
        source_datasets=source_datasets,
        num_classes=num_classes,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=persistent_workers,
        pin_memory=resolve_pin_memory(pin_memory_setting, runtime_spec),
        collate_fn=collate_manifest_samples,
    )


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


def _write_validation_history(validation_history: list[ValidationEpochSummary], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(validation_history[0].to_csv_row().keys()) if validation_history else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for record in validation_history:
                writer.writerow(record.to_csv_row())
    return output


def _proposition_accuracy(records, cohort_name: str) -> float:
    cohort_records = [record for record in records if record.cohort_name == cohort_name]
    if not cohort_records:
        return 0.0
    correct_count = sum(int(record.is_top1_correct) for record in cohort_records)
    return correct_count / len(cohort_records)

def _evaluate_validation_epoch(
    *,
    epoch: int,
    phase_name: str,
    model: nn.Module,
    dataloader: DataLoader,
    runtime_spec,
    run_id: str,
    protocol_id: str,
    resolved_eval_config: Mapping[str, str | int | float | tuple[str, ...]],
) -> ValidationEpochSummary:
    sample_analysis_records = run_inference_export(
        model=model,
        dataloader=dataloader,
        runtime_spec=runtime_spec,
        run_id=run_id,
        protocol_id=protocol_id,
    )
    proposition_records = build_top1_proposition_records(sample_analysis_records)
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
    )
    return ValidationEpochSummary(
        epoch=epoch,
        phase_name=phase_name,
        pair_auroc=matched_summary.pair_auroc,
        weighted_pair_auroc=matched_summary.weighted_pair_auroc,
        scalar_auroc=matched_summary.scalar_auroc,
        easy_id_top1_accuracy=_proposition_accuracy(proposition_records, "easy_id"),
        hard_id_top1_accuracy=_proposition_accuracy(proposition_records, "hard_id"),
        ambiguous_candidate_hit_rate=_proposition_accuracy(proposition_records, "ambiguous_id"),
    )


def _save_checkpoint(
    output_path: str | Path,
    *,
    run_id: str,
    epoch: int,
    protocol_id: str,
    model: nn.Module,
    optimizer: Optimizer,
    epoch_summary: TrainEpochSummary,
    validation_summary: ValidationEpochSummary | None = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "run_id": run_id,
            "epoch": epoch,
            "protocol_id": protocol_id,
            "epoch_summary": epoch_summary.to_csv_row(),
            "validation_summary": None if validation_summary is None else validation_summary.to_csv_row(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        output,
    )
    return output


def _run_training_epoch(
    *,
    epoch: int,
    total_epoch_count: int,
    phase_name: str,
    learning_rate: float,
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    runtime_spec,
    loss_config: Mapping[str, Any] | None,
    progress_callback: Callable[[str], None] | None = None,
) -> TrainEpochSummary:
    batch_count = 0
    optimizer_steps = 0
    trainable_samples = 0
    loss_total_sum = 0.0
    loss_id_sum = 0.0
    loss_unknown_sum = 0.0
    loss_ambiguous_sum = 0.0

    total_batches = len(dataloader)
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

        batch_loss_total = float(loss_breakdown.loss_total.detach().item())
        running_loss_total = loss_total_sum / max(optimizer_steps, 1)
        batch_status = (
            f"batch_loss={batch_loss_total:.4f}"
            if loss_breakdown.optimizer_step_performed
            else "batch_loss=skip"
        )
        _emit_progress(
            progress_callback,
            (
                f"[train-batch] epoch={epoch}/{total_epoch_count} "
                f"phase={phase_name} "
                f"{_progress_bar(batch_count, total_batches)} "
                f"batch={batch_count}/{total_batches} "
                f"{batch_status} "
                f"running_loss={running_loss_total:.4f}"
            ),
        )

    denominator = max(optimizer_steps, 1)
    return TrainEpochSummary(
        epoch=epoch,
        phase_name=phase_name,
        learning_rate=learning_rate,
        batch_count=batch_count,
        optimizer_steps=optimizer_steps,
        trainable_samples=trainable_samples,
        mean_loss_total=loss_total_sum / denominator,
        mean_loss_id=loss_id_sum / denominator,
        mean_loss_unknown=loss_unknown_sum / denominator,
        mean_loss_ambiguous=loss_ambiguous_sum / denominator,
    )
