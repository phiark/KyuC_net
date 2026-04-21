from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
import csv
import json
import math
from pathlib import Path
import shutil
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import yaml

from frcnet.analysis import (
    write_artifact_path_list,
    write_completion_scan_table,
    write_cohort_occupancy,
    write_cohort_summary_table,
    write_experiment_record,
    write_geometry_hexbin,
    write_geometry_scatter,
    write_scalar_roc_curve,
    write_tau_cohort_boxplot,
)
from frcnet.data import (
    ManifestBackedVisionDataset,
    build_plan_a_manifest,
    collate_manifest_samples,
    load_plan_a_source_datasets,
    read_manifest_jsonl,
    summarize_manifest,
    validate_manifest_records,
    write_manifest_jsonl,
    write_manifest_summary,
)
from frcnet.evaluation import (
    AnalysisExportSummary,
    build_top1_proposition_records,
    read_analysis_export_summary,
    read_sample_analysis_records,
    read_top1_proposition_records,
    run_inference_export,
    summarize_matched_ambiguous_vs_ood,
    write_analysis_export_summary,
    write_matched_benchmark_summary,
    write_sample_analysis_records,
    write_top1_proposition_records,
)
from frcnet.models import FRCNetModel
from frcnet.training import run_train_step
from frcnet.utils import resolve_pin_memory, resolve_runtime

TRAINABLE_COHORT_NAMES = frozenset({"easy_id", "hard_id", "ambiguous_id", "unknown_supervision"})


@dataclass(slots=True)
class TrainPhase:
    name: str
    epoch_count: int
    enabled_cohorts: tuple[str, ...]
    loss_overrides: dict[str, Any]
    dataloader_overrides: dict[str, Any]
    lr_override: float | None = None
    lr_scale: float = 1.0


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
class AnalysisRecordState:
    run_ids: tuple[str, ...]
    protocol_ids: tuple[str, ...]
    sample_ids: frozenset[str]
    duplicate_sample_ids: tuple[str, ...]


@dataclass(slots=True)
class PropositionRecordState:
    run_ids: tuple[str, ...]
    protocol_ids: tuple[str, ...]
    sample_ids: frozenset[str]
    duplicate_sample_ids: tuple[str, ...]


@dataclass(slots=True)
class ResolvedAnalysisSidecars:
    analysis_summary_path: str | None
    manifest_snapshot_path: str | None
    proposition_path: str | None
    model_config_snapshot_path: str | None
    checkpoint_path: str | None
    summary_run_id: str | None
    summary_protocol_id: str | None
    sidecar_resolution_mode: str
    inherited_integrity_overrides: tuple[str, ...]


@dataclass(slots=True)
class ValidationEpochSummary:
    epoch: int
    phase_name: str
    pair_auroc: float
    weighted_pair_auroc: float
    scalar_auroc: float
    tau_scalar_auroc: float
    easy_id_top1_accuracy: float
    hard_id_top1_accuracy: float
    ambiguous_candidate_hit_rate: float
    selected_as_best: bool = False

    def to_csv_row(self) -> dict[str, int | float]:
        return {
            "epoch": self.epoch,
            "phase_name": self.phase_name,
            "pair_auroc": self.pair_auroc,
            "weighted_pair_auroc": self.weighted_pair_auroc,
            "scalar_auroc": self.scalar_auroc,
            "tau_scalar_auroc": self.tau_scalar_auroc,
            "easy_id_top1_accuracy": self.easy_id_top1_accuracy,
            "hard_id_top1_accuracy": self.hard_id_top1_accuracy,
            "ambiguous_candidate_hit_rate": self.ambiguous_candidate_hit_rate,
            "selected_as_best": int(self.selected_as_best),
        }


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


def _copy_optional_snapshot(input_path: str | Path | None, output_path: str | Path) -> Path | None:
    if input_path is None:
        return None
    source_path = Path(input_path)
    if not source_path.exists():
        return None
    if source_path.resolve() == Path(output_path).resolve():
        return source_path
    return _copy_snapshot(source_path, output_path)


def _emit_progress(progress_callback: Callable[[str], None] | None, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)


def _progress_bar(current: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[" + ("-" * width) + "]"
    filled_width = int(width * current / total)
    if current >= total:
        filled_width = width
    return "[" + ("#" * filled_width) + ("-" * (width - filled_width)) + "]"


def _resolve_eval_config(eval_config_path: str | Path | None) -> dict[str, str | int | float | tuple[str, ...]]:
    if eval_config_path is None:
        return {
            "positive_cohort": "ambiguous_id",
            "negative_cohort": "ood",
            "primary_pair": "resolution_ratio__content_entropy",
            "weighted_pair": "resolution_ratio__resolution_weighted_content_entropy",
            "primary_scalar": "completion_score_beta_0_1",
            "tau_scalar_name": "top1_content_probability",
            "completion_scan_scalars": (
                "completion_score_beta_0_1",
                "completion_score_beta_0_25",
                "completion_score_beta_0_5",
                "completion_score_beta_0_75",
            ),
            "test_size": 0.3,
            "random_state": 7,
        }

    eval_config = _load_yaml_section(eval_config_path, "eval")
    completion_scan_scalars = eval_config.get(
        "completion_scan_scalars",
        (
            "completion_score_beta_0_1",
            "completion_score_beta_0_25",
            "completion_score_beta_0_5",
            "completion_score_beta_0_75",
        ),
    )
    return {
        "positive_cohort": str(eval_config.get("positive_cohort", "ambiguous_id")),
        "negative_cohort": str(eval_config.get("negative_cohort", "ood")),
        "primary_pair": str(eval_config.get("primary_pair", "resolution_ratio__content_entropy")),
        "weighted_pair": str(
            eval_config.get("weighted_pair", "resolution_ratio__resolution_weighted_content_entropy")
        ),
        "primary_scalar": str(eval_config.get("primary_scalar", "completion_score_beta_0_1")),
        "tau_scalar_name": str(eval_config.get("tau_scalar_name", "top1_content_probability")),
        "completion_scan_scalars": tuple(str(value) for value in completion_scan_scalars),
        "test_size": float(eval_config.get("test_size", 0.3)),
        "random_state": int(eval_config.get("random_state", 7)),
    }


def _normalize_training_phases(train_config: Mapping[str, Any]) -> list[TrainPhase]:
    training_config = dict(train_config.get("training", {}))
    phase_payloads = training_config.get("phases")
    if phase_payloads is None:
        epoch_count = int(training_config.get("epochs", 1))
        if epoch_count <= 0:
            raise ValueError("train.training.epochs must be positive.")
        return [
            TrainPhase(
                name="main",
                epoch_count=epoch_count,
                enabled_cohorts=tuple(sorted(TRAINABLE_COHORT_NAMES)),
                loss_overrides={},
                dataloader_overrides={},
            )
        ]

    phases: list[TrainPhase] = []
    for phase_index, phase_payload in enumerate(phase_payloads, start=1):
        phase_name = str(phase_payload.get("name", f"phase_{phase_index:02d}"))
        epoch_count = int(phase_payload.get("epoch_count", 0))
        if epoch_count <= 0:
            raise ValueError(f"train.training.phases[{phase_index - 1}].epoch_count must be positive.")
        enabled_cohorts = tuple(
            str(cohort_name) for cohort_name in phase_payload.get("enabled_cohorts", tuple(sorted(TRAINABLE_COHORT_NAMES)))
        )
        if not enabled_cohorts:
            raise ValueError(f"train.training.phases[{phase_index - 1}] must enable at least one cohort.")
        unsupported_cohorts = sorted(set(enabled_cohorts) - TRAINABLE_COHORT_NAMES)
        if unsupported_cohorts:
            raise ValueError(
                f"Unsupported enabled_cohorts in phase `{phase_name}`: {unsupported_cohorts}. "
                f"Supported values: {sorted(TRAINABLE_COHORT_NAMES)}"
            )
        phases.append(
            TrainPhase(
                name=phase_name,
                epoch_count=epoch_count,
                enabled_cohorts=enabled_cohorts,
                loss_overrides=dict(phase_payload.get("loss_weights", {})),
                dataloader_overrides=dict(phase_payload.get("dataloader", {})),
                lr_override=(
                    None
                    if phase_payload.get("lr_override") is None
                    else float(phase_payload.get("lr_override"))
                ),
                lr_scale=float(phase_payload.get("lr_scale", 1.0)),
            )
        )
    return phases


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


def _phase_learning_rate(
    *,
    base_learning_rate: float,
    phase: TrainPhase,
    global_epoch_index: int,
    total_epoch_count: int,
) -> float:
    reference_lr = phase.lr_override if phase.lr_override is not None else base_learning_rate
    if total_epoch_count <= 1:
        cosine_factor = 1.0
    else:
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * global_epoch_index / (total_epoch_count - 1)))
    return reference_lr * phase.lr_scale * cosine_factor


def _set_optimizer_learning_rate(optimizer: Optimizer, learning_rate: float) -> None:
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate


def _duplicate_values(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(value for value, count in Counter(values).items() if count > 1))


def _merge_integrity_overrides(*override_groups: Iterable[str]) -> list[str]:
    merged: list[str] = []
    for override_group in override_groups:
        for override_value in override_group:
            if override_value not in merged:
                merged.append(override_value)
    return merged


def _inspect_sample_analysis_records(records: list) -> AnalysisRecordState:
    return AnalysisRecordState(
        run_ids=tuple(sorted({record.run_id for record in records})),
        protocol_ids=tuple(sorted({record.protocol_id for record in records})),
        sample_ids=frozenset(record.sample_id for record in records),
        duplicate_sample_ids=_duplicate_values(record.sample_id for record in records),
    )


def _inspect_proposition_records(records: list) -> PropositionRecordState:
    return PropositionRecordState(
        run_ids=tuple(sorted({record.run_id for record in records})),
        protocol_ids=tuple(sorted({record.protocol_id for record in records})),
        sample_ids=frozenset(record.sample_id for record in records),
        duplicate_sample_ids=_duplicate_values(record.sample_id for record in records),
    )


def _resolve_reference_path(reference: str | None, base_dir: Path) -> Path | None:
    if reference in {None, ""}:
        return None
    candidate = Path(reference)
    if candidate.is_absolute():
        return candidate
    return base_dir / candidate


def _resolve_analysis_sidecars(
    analysis_record_path: Path,
    analysis_summary_path: str | Path | None,
) -> tuple[ResolvedAnalysisSidecars, list[str]]:
    integrity_errors: list[str] = []

    if analysis_summary_path is not None:
        summary_path = Path(analysis_summary_path)
        resolution_mode = "analysis_summary_explicit"
    else:
        auto_summary_path = analysis_record_path.parent / "analysis_summary.json"
        if auto_summary_path.exists():
            summary_path = auto_summary_path
            resolution_mode = "analysis_summary_auto"
        else:
            return (
                ResolvedAnalysisSidecars(
                    analysis_summary_path=None,
                    manifest_snapshot_path=str(analysis_record_path.parent / "plan_a_manifest_snapshot.jsonl"),
                    proposition_path=str(analysis_record_path.parent / "top1_proposition_records.csv"),
                    model_config_snapshot_path=str(analysis_record_path.parent / "model_config_snapshot.yaml"),
                    checkpoint_path=None,
                    summary_run_id=None,
                    summary_protocol_id=None,
                    sidecar_resolution_mode="legacy_sibling",
                    inherited_integrity_overrides=(),
                ),
                integrity_errors,
            )

    if not summary_path.exists():
        integrity_errors.append("analysis_summary_missing")
        return (
            ResolvedAnalysisSidecars(
                analysis_summary_path=str(summary_path),
                manifest_snapshot_path=None,
                proposition_path=None,
                model_config_snapshot_path=None,
                checkpoint_path=None,
                summary_run_id=None,
                summary_protocol_id=None,
                sidecar_resolution_mode=resolution_mode,
                inherited_integrity_overrides=(),
            ),
            integrity_errors,
        )

    summary = read_analysis_export_summary(summary_path)
    resolved_analysis_path = _resolve_reference_path(summary.analysis_path, summary_path.parent)
    if resolved_analysis_path is None or resolved_analysis_path.resolve() != analysis_record_path.resolve():
        integrity_errors.append("analysis_summary_analysis_path_mismatch")

    return (
        ResolvedAnalysisSidecars(
            analysis_summary_path=str(summary_path),
            manifest_snapshot_path=(
                None
                if _resolve_reference_path(summary.manifest_snapshot_path, summary_path.parent) is None
                else str(_resolve_reference_path(summary.manifest_snapshot_path, summary_path.parent))
            ),
            proposition_path=(
                None
                if _resolve_reference_path(summary.proposition_path, summary_path.parent) is None
                else str(_resolve_reference_path(summary.proposition_path, summary_path.parent))
            ),
            model_config_snapshot_path=(
                None
                if _resolve_reference_path(summary.model_config_snapshot_path, summary_path.parent) is None
                else str(_resolve_reference_path(summary.model_config_snapshot_path, summary_path.parent))
            ),
            checkpoint_path=(
                None
                if summary.checkpoint_path is None
                else str(_resolve_reference_path(summary.checkpoint_path, summary_path.parent))
            ),
            summary_run_id=summary.run_id,
            summary_protocol_id=summary.protocol_id,
            sidecar_resolution_mode=resolution_mode,
            inherited_integrity_overrides=summary.integrity_overrides,
        ),
        integrity_errors,
    )


def _finalize_integrity_errors(
    integrity_errors: list[str],
    *,
    allow_integrity_override: bool,
    inherited_overrides: Iterable[str] = (),
) -> list[str]:
    if integrity_errors and not allow_integrity_override:
        raise ValueError(f"Integrity validation failed: {', '.join(integrity_errors)}")
    return _merge_integrity_overrides(inherited_overrides, integrity_errors)


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
        tau_scalar_name=str(resolved_eval_config["tau_scalar_name"]),
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
        tau_scalar_auroc=matched_summary.tau_scalar_auroc,
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
    manifest_records = validate_manifest_records(build_plan_a_manifest(protocol_config, source_datasets))
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
    validation_protocol_config_path: str | Path | None = None,
    validation_manifest_path: str | Path | None = None,
    eval_config_path: str | Path | None = None,
    progress_callback: Callable[[str], None] | None = None,
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
    validation_protocol_snapshot_path = (
        None
        if validation_protocol_config_path is None
        else _copy_snapshot(validation_protocol_config_path, snapshots_dir / "validation_protocol_config_snapshot.yaml")
    )
    eval_snapshot_path = (
        None
        if eval_config_path is None
        else _copy_snapshot(eval_config_path, snapshots_dir / "eval_config_snapshot.yaml")
    )

    source_datasets = load_plan_a_source_datasets(protocol_config)
    if manifest_path is None:
        manifest_records = validate_manifest_records(build_plan_a_manifest(protocol_config, source_datasets))
        manifest_snapshot_path = write_manifest_jsonl(manifest_records, manifests_dir / "train_manifest_snapshot.jsonl")
    else:
        manifest_records = validate_manifest_records(read_manifest_jsonl(manifest_path))
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
    _emit_progress(
        progress_callback,
        (
            f"[train] run_id={resolved_run_id} "
            f"backend={runtime_spec.resolved_backend} "
            f"device={runtime_spec.device} "
            f"dtype={runtime_spec.dtype}"
        ),
    )
    seed = int(train_config.get("training", {}).get("seed", protocol_config.get("seed", 7)))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    optimizer = _build_optimizer(model, train_config["optimizer"])
    base_learning_rate = float(train_config["optimizer"]["lr"])
    if checkpoint_path is not None:
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint_payload.get("model_state_dict", checkpoint_payload)
        model.load_state_dict(state_dict)

    phases = _normalize_training_phases(train_config)
    total_epoch_count = sum(phase.epoch_count for phase in phases)
    base_loss_config = dict(train_config.get("loss", {}))
    checkpoint_config = dict(train_config.get("checkpointing", {}))
    save_every_epochs = int(checkpoint_config.get("save_every_epochs", 1))
    base_dataloader_config = dict(protocol_config.get("analysis", {}).get("dataloader", {}))
    base_dataloader_config.update(train_config.get("dataloader", {}))

    epoch_history: list[TrainEpochSummary] = []
    validation_history: list[ValidationEpochSummary] = []
    best_checkpoint_path = checkpoints_dir / "checkpoint_best.pt"
    last_checkpoint_path = checkpoints_dir / "checkpoint_last.pt"
    best_mean_loss: float | None = None
    best_validation_summary: ValidationEpochSummary | None = None
    best_epoch = 0

    validation_manifest_snapshot_path: Path | None = None
    validation_manifest_summary_path: Path | None = None
    validation_history_path: Path | None = None
    resolved_eval_config = _resolve_eval_config(eval_config_path)
    validation_protocol_id: str | None = None
    validation_dataloader: DataLoader | None = None
    if validation_manifest_path is not None or validation_protocol_config_path is not None:
        if validation_manifest_path is None and validation_protocol_config_path is None:
            raise ValueError("validation manifest selection requires either validation_manifest_path or validation_protocol_config_path.")
        if validation_protocol_config_path is not None:
            validation_protocol_config = _load_yaml_section(validation_protocol_config_path, "protocol")
            validation_protocol_id = str(validation_protocol_config["protocol_id"])
            validation_source_datasets = load_plan_a_source_datasets(validation_protocol_config)
        else:
            validation_protocol_config = protocol_config
            validation_protocol_id = str(protocol_config["protocol_id"])
            validation_source_datasets = source_datasets

        if validation_manifest_path is None:
            validation_manifest_records = validate_manifest_records(
                build_plan_a_manifest(validation_protocol_config, validation_source_datasets)
            )
            validation_manifest_snapshot_path = write_manifest_jsonl(
                validation_manifest_records,
                manifests_dir / "validation_manifest_snapshot.jsonl",
            )
        else:
            validation_manifest_records = validate_manifest_records(read_manifest_jsonl(validation_manifest_path))
            validation_manifest_snapshot_path = _copy_snapshot(
                validation_manifest_path,
                manifests_dir / "validation_manifest_snapshot.jsonl",
            )
        validation_manifest_summary_path = write_manifest_summary(
            validation_manifest_records,
            manifests_dir / "validation_manifest_summary.json",
        )
        validation_dataloader = _build_validation_dataloader(
            manifest_records=validation_manifest_records,
            source_datasets=validation_source_datasets,
            num_classes=int(validation_protocol_config["num_classes"]),
            protocol_config=validation_protocol_config,
            train_config=train_config,
            runtime_config=runtime_config,
            runtime_spec=runtime_spec,
        )
        _emit_progress(
            progress_callback,
            (
                f"[train] validation_manifest={validation_manifest_snapshot_path} "
                f"protocol_id={validation_protocol_id} "
                f"records={len(validation_manifest_records)}"
            ),
        )

    global_epoch = 0
    for phase_index, phase in enumerate(phases):
        phase_manifest_records = _filter_manifest_records_by_cohorts(manifest_records, phase.enabled_cohorts)
        if not phase_manifest_records:
            raise ValueError(f"Training phase `{phase.name}` resolved to zero manifest records.")

        phase_dataloader_config = dict(base_dataloader_config)
        phase_dataloader_config.update(phase.dataloader_overrides)
        dataloader = _build_manifest_dataloader(
            manifest_records=phase_manifest_records,
            source_datasets=source_datasets,
            num_classes=int(protocol_config["num_classes"]),
            dataloader_config=phase_dataloader_config,
            runtime_config=runtime_config,
            runtime_spec=runtime_spec,
            model=model,
            generator_seed=seed + phase_index,
        )
        if len(dataloader) == 0:
            raise ValueError(f"Training dataloader resolved to zero batches for phase `{phase.name}`.")

        phase_loss_config = dict(base_loss_config)
        phase_loss_config.update(phase.loss_overrides)
        _emit_progress(
            progress_callback,
            (
                f"[train] phase={phase.name} "
                f"epochs={phase.epoch_count} "
                f"cohorts={','.join(phase.enabled_cohorts)} "
                f"records={len(phase_manifest_records)} "
                f"batches={len(dataloader)}"
            ),
        )
        for _ in range(phase.epoch_count):
            global_epoch += 1
            learning_rate = _phase_learning_rate(
                base_learning_rate=base_learning_rate,
                phase=phase,
                global_epoch_index=global_epoch - 1,
                total_epoch_count=total_epoch_count,
            )
            _set_optimizer_learning_rate(optimizer, learning_rate)
            epoch_summary = _run_training_epoch(
                epoch=global_epoch,
                total_epoch_count=total_epoch_count,
                phase_name=phase.name,
                learning_rate=learning_rate,
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                runtime_spec=runtime_spec,
                loss_config=phase_loss_config,
                progress_callback=progress_callback,
            )
            epoch_history.append(epoch_summary)

            validation_summary: ValidationEpochSummary | None = None
            if validation_dataloader is not None:
                validation_summary = _evaluate_validation_epoch(
                    epoch=global_epoch,
                    phase_name=phase.name,
                    model=model,
                    dataloader=validation_dataloader,
                    runtime_spec=runtime_spec,
                    run_id=resolved_run_id,
                    protocol_id=validation_protocol_id or protocol_config["protocol_id"],
                    resolved_eval_config=resolved_eval_config,
                )
                validation_history.append(validation_summary)

            if save_every_epochs > 0 and (global_epoch % save_every_epochs == 0):
                _save_checkpoint(
                    checkpoints_dir / f"checkpoint_epoch_{global_epoch:03d}.pt",
                    run_id=resolved_run_id,
                    epoch=global_epoch,
                    protocol_id=protocol_config["protocol_id"],
                    model=model,
                    optimizer=optimizer,
                    epoch_summary=epoch_summary,
                    validation_summary=validation_summary,
                )

            _save_checkpoint(
                last_checkpoint_path,
                run_id=resolved_run_id,
                epoch=global_epoch,
                protocol_id=protocol_config["protocol_id"],
                model=model,
                optimizer=optimizer,
                epoch_summary=epoch_summary,
                validation_summary=validation_summary,
            )

            epoch_message = (
                f"[train] epoch={global_epoch}/{total_epoch_count} "
                f"phase={phase.name} "
                f"lr={learning_rate:.6f} "
                f"loss_total={epoch_summary.mean_loss_total:.4f} "
                f"loss_id={epoch_summary.mean_loss_id:.4f} "
                f"loss_unknown={epoch_summary.mean_loss_unknown:.4f} "
                f"loss_ambiguous={epoch_summary.mean_loss_ambiguous:.4f}"
            )
            if validation_summary is not None:
                epoch_message += (
                    f" val_pair={validation_summary.pair_auroc:.4f}"
                    f" val_easy_top1={validation_summary.easy_id_top1_accuracy:.4f}"
                    f" val_hard_top1={validation_summary.hard_id_top1_accuracy:.4f}"
                    f" val_amb_hit={validation_summary.ambiguous_candidate_hit_rate:.4f}"
                )
            _emit_progress(progress_callback, epoch_message)

            if validation_summary is None:
                if best_mean_loss is None or epoch_summary.mean_loss_total <= best_mean_loss:
                    best_mean_loss = epoch_summary.mean_loss_total
                    best_epoch = global_epoch
                    _save_checkpoint(
                        best_checkpoint_path,
                        run_id=resolved_run_id,
                        epoch=global_epoch,
                        protocol_id=protocol_config["protocol_id"],
                        model=model,
                        optimizer=optimizer,
                        epoch_summary=epoch_summary,
                        validation_summary=validation_summary,
                    )
                    _emit_progress(
                        progress_callback,
                        f"[train] best_checkpoint_updated epoch={global_epoch} criterion=train_mean_loss_total",
                    )
                continue

            candidate_rank = (
                validation_summary.pair_auroc,
                validation_summary.easy_id_top1_accuracy,
                -epoch_summary.mean_loss_total,
            )
            best_rank = None
            if best_validation_summary is not None and best_mean_loss is not None:
                best_rank = (
                    best_validation_summary.pair_auroc,
                    best_validation_summary.easy_id_top1_accuracy,
                    -best_mean_loss,
                )
            if best_rank is None or candidate_rank > best_rank:
                if best_validation_summary is not None:
                    best_validation_summary.selected_as_best = False
                best_mean_loss = epoch_summary.mean_loss_total
                best_validation_summary = validation_summary
                best_epoch = global_epoch
                validation_summary.selected_as_best = True
                _save_checkpoint(
                    best_checkpoint_path,
                    run_id=resolved_run_id,
                    epoch=global_epoch,
                    protocol_id=protocol_config["protocol_id"],
                    model=model,
                    optimizer=optimizer,
                    epoch_summary=epoch_summary,
                    validation_summary=validation_summary,
                )
                _emit_progress(
                    progress_callback,
                    (
                        f"[train] best_checkpoint_updated epoch={global_epoch} "
                        f"criterion=validation_pair_auroc_then_easy_id_top1_then_train_mean_loss"
                    ),
                )

    history_path = _write_epoch_history(epoch_history, records_dir / "train_history.csv")
    if validation_history:
        validation_history_path = _write_validation_history(
            validation_history,
            records_dir / "validation_history.csv",
        )
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
            "training_phases": [
                {
                    "name": phase.name,
                    "epoch_count": phase.epoch_count,
                    "enabled_cohorts": list(phase.enabled_cohorts),
                    "lr_override": phase.lr_override,
                    "lr_scale": phase.lr_scale,
                    "loss_overrides": phase.loss_overrides,
                    "dataloader_overrides": phase.dataloader_overrides,
                }
                for phase in phases
            ],
            "snapshots": {
                "protocol_config_snapshot": str(protocol_snapshot_path),
                "model_config_snapshot": str(model_snapshot_path),
                "train_config_snapshot": str(train_snapshot_path),
                "validation_protocol_config_snapshot": None
                if validation_protocol_snapshot_path is None
                else str(validation_protocol_snapshot_path),
                "eval_config_snapshot": None if eval_snapshot_path is None else str(eval_snapshot_path),
            },
            "checkpoints": {
                "best": str(best_checkpoint_path),
                "last": str(last_checkpoint_path),
                "best_epoch": best_epoch,
            },
            "history_path": str(history_path),
            "validation": {
                "manifest_path": None if validation_manifest_snapshot_path is None else str(validation_manifest_snapshot_path),
                "summary_path": None if validation_manifest_summary_path is None else str(validation_manifest_summary_path),
                "history_path": None if validation_history_path is None else str(validation_history_path),
                "protocol_id": validation_protocol_id,
                "checkpoint_selection": (
                    "validation_pair_auroc_then_easy_id_top1_then_train_mean_loss"
                    if validation_history
                    else "train_mean_loss_total"
                ),
                "resolved_eval_config": {
                    "positive_cohort": resolved_eval_config["positive_cohort"],
                    "negative_cohort": resolved_eval_config["negative_cohort"],
                    "primary_pair": resolved_eval_config["primary_pair"],
                    "weighted_pair": resolved_eval_config["weighted_pair"],
                    "primary_scalar": resolved_eval_config["primary_scalar"],
                    "tau_scalar_name": resolved_eval_config["tau_scalar_name"],
                    "completion_scan_scalars": list(resolved_eval_config["completion_scan_scalars"]),
                    "test_size": resolved_eval_config["test_size"],
                    "random_state": resolved_eval_config["random_state"],
                },
                "best_epoch_metrics": None
                if best_validation_summary is None
                else best_validation_summary.to_csv_row(),
            },
            "epochs": [record.to_csv_row() for record in epoch_history],
            "validation_epochs": [record.to_csv_row() for record in validation_history],
        },
        records_dir / "train_summary.json",
    )
    _emit_progress(
        progress_callback,
        (
            f"[train] completed run_id={resolved_run_id} "
            f"best_checkpoint={best_checkpoint_path} "
            f"train_summary={summary_path}"
        ),
    )

    return {
        "run_id": resolved_run_id,
        "protocol_id": protocol_config["protocol_id"],
        "output_dir": str(output_root),
        "manifest_path": str(manifest_snapshot_path),
        "manifest_summary_path": str(manifest_summary_path),
        "history_path": str(history_path),
        "summary_path": str(summary_path),
        "validation_history_path": "" if validation_history_path is None else str(validation_history_path),
        "validation_manifest_path": ""
        if validation_manifest_snapshot_path is None
        else str(validation_manifest_snapshot_path),
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
    allow_missing_checkpoint: bool = False,
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
    )
    proposition_records = build_top1_proposition_records(sample_analysis_records)

    analysis_path = write_sample_analysis_records(sample_analysis_records, output_root / "sample_analysis_records.csv")
    proposition_path = write_top1_proposition_records(
        proposition_records,
        output_root / "top1_proposition_records.csv",
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
            integrity_overrides=tuple(integrity_overrides),
            sidecar_resolution_mode="analysis_summary",
        ),
        output_root / "analysis_summary.json",
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
        "analysis_summary_path": str(analysis_summary_path),
    }


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
    analysis_config = {"figure_dpi": 200}
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

    run_id = analysis_state.run_ids[0] if len(analysis_state.run_ids) == 1 else "MULTIPLE"
    protocol_id = analysis_state.protocol_ids[0] if len(analysis_state.protocol_ids) == 1 else "MULTIPLE"
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
    tau_boxplot_path = write_tau_cohort_boxplot(
        sample_analysis_records,
        output_root / analysis_config.get("tau_cohort_boxplot_name", "tau_cohort_boxplot.png"),
        dpi=figure_dpi,
    )
    summary_path = write_cohort_summary_table(
        sample_analysis_records,
        output_root / analysis_config.get("cohort_summary_table_name", "cohort_summary_table.csv"),
    )
    matched_summary = summarize_matched_ambiguous_vs_ood(
        sample_analysis_records,
        positive_cohort=str(resolved_eval_config["positive_cohort"]),
        negative_cohort=str(resolved_eval_config["negative_cohort"]),
        primary_pair=str(resolved_eval_config["primary_pair"]),
        weighted_pair=str(resolved_eval_config["weighted_pair"]),
        primary_scalar=str(resolved_eval_config["primary_scalar"]),
        tau_scalar_name=str(resolved_eval_config["tau_scalar_name"]),
        completion_scan_scalars=tuple(resolved_eval_config["completion_scan_scalars"]),
        test_size=float(resolved_eval_config["test_size"]),
        random_state=int(resolved_eval_config["random_state"]),
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
    )
    tau_roc_curve_path = write_scalar_roc_curve(
        sample_analysis_records,
        output_root / analysis_config.get("tau_roc_curve_name", "tau_roc_curve.png"),
        positive_cohort=str(resolved_eval_config["positive_cohort"]),
        negative_cohort=str(resolved_eval_config["negative_cohort"]),
        scalar_name=str(resolved_eval_config["tau_scalar_name"]),
        test_size=float(resolved_eval_config["test_size"]),
        random_state=int(resolved_eval_config["random_state"]),
        dpi=figure_dpi,
    )

    artifact_paths = {
        "geometry_scatter": str(scatter_path),
        "geometry_hexbin": str(hexbin_path),
        "cohort_occupancy": str(occupancy_path),
        "tau_cohort_boxplot": str(tau_boxplot_path),
        "tau_roc_curve": str(tau_roc_curve_path),
        "cohort_summary_table": str(summary_path),
        "matched_ambiguous_vs_ood_table": str(matched_path),
        "completion_scan_table": str(completion_scan_path),
    }
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
        analysis_summary_path=str(analysis_summary_copy_path or sidecars.analysis_summary_path or ""),
        sidecar_resolution_mode=sidecars.sidecar_resolution_mode,
        integrity_overrides=integrity_overrides,
        source_run_ids=analysis_state.run_ids,
        source_protocol_ids=analysis_state.protocol_ids,
        resolved_eval_config=resolved_eval_config,
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
        "analysis_manifest_path": analysis_manifest_outputs["manifest_path"],
        "analysis_path": inference_outputs["analysis_path"],
        "analysis_summary_path": inference_outputs["analysis_summary_path"],
        "artifact_index_path": artifact_outputs["artifact_index_path"],
        "experiment_record_path": artifact_outputs["experiment_record_path"],
        "bundle_path": str(bundle_path),
    }
