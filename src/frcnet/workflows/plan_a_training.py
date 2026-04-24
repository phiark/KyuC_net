from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

import torch

from frcnet.data import (
    build_plan_a_manifest,
    load_plan_a_source_datasets,
    read_manifest_jsonl,
    summarize_manifest,
    validate_manifest_records,
    write_manifest_jsonl,
    write_manifest_summary,
)
from frcnet.evaluation import DEFAULT_MODEL_FAMILY
from frcnet.workflows.plan_a_training_core import (
    TRAINABLE_COHORT_NAMES,
    TrainEpochSummary,
    ValidationEpochSummary,
    _build_manifest_dataloader,
    _build_model,
    _build_optimizer,
    _build_validation_dataloader,
    _evaluate_validation_epoch,
    _filter_manifest_records_by_cohorts,
    _run_training_epoch,
    _save_checkpoint,
    _write_epoch_history,
    _write_validation_history,
)
from frcnet.workflows.workflow_io import (
    _copy_snapshot,
    _emit_progress,
    _load_yaml_section,
    _resolve_eval_config,
    _write_json,
    timestamp_run_id,
)
from frcnet.utils import resolve_runtime


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


def _selection_payload(
    *,
    policy_name: str,
    checkpoint_path: Path,
    epoch_summary: TrainEpochSummary,
    validation_summary: ValidationEpochSummary | None,
    selection_rule: str,
) -> dict[str, Any]:
    return {
        "policy_name": policy_name,
        "checkpoint_path": str(checkpoint_path),
        "epoch": epoch_summary.epoch,
        "phase_name": epoch_summary.phase_name,
        "selection_rule": selection_rule,
        "train_metrics": epoch_summary.to_csv_row(),
        "validation_metrics": None if validation_summary is None else validation_summary.to_csv_row(),
    }


def _best_rank(
    *,
    epoch_summary: TrainEpochSummary,
    validation_summary: ValidationEpochSummary | None,
) -> tuple[float, ...]:
    if validation_summary is None:
        return (-epoch_summary.mean_loss_total,)
    return (
        validation_summary.pair_auroc,
        validation_summary.easy_id_top1_accuracy,
        -epoch_summary.mean_loss_total,
    )


def _set_optimizer_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate


def _resolve_training_phases(training_config: dict[str, Any]) -> list[dict[str, Any]]:
    raw_phases = training_config.get("phases")
    if raw_phases is None:
        total_epoch_count = int(training_config.get("epochs", 1))
        if total_epoch_count <= 0:
            raise ValueError("train.training.epochs must be positive.")
        return [
            {
                "name": "train",
                "epoch_count": total_epoch_count,
                "enabled_cohorts": sorted(TRAINABLE_COHORT_NAMES),
                "lr_scale": 1.0,
                "loss_overrides": {},
            }
        ]

    if not isinstance(raw_phases, list) or not raw_phases:
        raise ValueError("train.training.phases must be a non-empty list.")

    phases: list[dict[str, Any]] = []
    for phase_index, phase_config in enumerate(raw_phases, start=1):
        phase_name = str(phase_config.get("name", f"phase_{phase_index}"))
        epoch_count = int(phase_config.get("epoch_count", 0))
        if epoch_count <= 0:
            raise ValueError(f"train.training.phases[{phase_index}].epoch_count must be positive.")
        enabled_cohorts = tuple(str(value) for value in phase_config.get("enabled_cohorts", sorted(TRAINABLE_COHORT_NAMES)))
        invalid_cohorts = sorted(set(enabled_cohorts) - set(TRAINABLE_COHORT_NAMES))
        if invalid_cohorts:
            raise ValueError(
                f"train.training.phases[{phase_index}].enabled_cohorts contains unsupported cohorts: {invalid_cohorts}"
            )
        if not enabled_cohorts:
            raise ValueError(f"train.training.phases[{phase_index}].enabled_cohorts must not be empty.")
        lr_scale = float(phase_config.get("lr_scale", 1.0))
        if lr_scale <= 0.0:
            raise ValueError(f"train.training.phases[{phase_index}].lr_scale must be positive.")
        phases.append(
            {
                "name": phase_name,
                "epoch_count": epoch_count,
                "enabled_cohorts": enabled_cohorts,
                "lr_scale": lr_scale,
                "loss_overrides": dict(phase_config.get("loss_overrides", {})),
            }
        )
    return phases


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

    train_manifest_records = _filter_manifest_records_by_cohorts(manifest_records, sorted(TRAINABLE_COHORT_NAMES))
    if not train_manifest_records:
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
    learning_rate = float(train_config["optimizer"]["lr"])
    if checkpoint_path is not None:
        checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint_payload.get("model_state_dict", checkpoint_payload)
        model.load_state_dict(state_dict)

    training_config = dict(train_config.get("training", {}))
    training_phases = _resolve_training_phases(training_config)
    total_epoch_count = sum(int(phase["epoch_count"]) for phase in training_phases)
    checkpoint_config = dict(train_config.get("checkpointing", {}))
    save_every_epochs = int(checkpoint_config.get("save_every_epochs", 1))
    base_loss_config = dict(train_config.get("loss", {}))
    dataloader_config = dict(protocol_config.get("analysis", {}).get("dataloader", {}))
    dataloader_config.update(train_config.get("dataloader", {}))
    model_family = str(train_config.get("model_family", DEFAULT_MODEL_FAMILY))

    phase_runtimes: list[dict[str, Any]] = []
    for phase_index, phase in enumerate(training_phases):
        phase_records = _filter_manifest_records_by_cohorts(train_manifest_records, phase["enabled_cohorts"])
        if not phase_records:
            raise ValueError(f"Training phase `{phase['name']}` does not contain any records.")
        phase_dataloader = _build_manifest_dataloader(
            manifest_records=phase_records,
            source_datasets=source_datasets,
            num_classes=int(protocol_config["num_classes"]),
            dataloader_config=dataloader_config,
            runtime_config=runtime_config,
            runtime_spec=runtime_spec,
            model=model,
            generator_seed=seed + phase_index,
        )
        if len(phase_dataloader) == 0:
            raise ValueError(f"Training dataloader for phase `{phase['name']}` resolved to zero batches.")
        phase_loss_config = dict(base_loss_config)
        phase_loss_config.update(phase["loss_overrides"])
        phase_runtimes.append(
            {
                **phase,
                "records": phase_records,
                "dataloader": phase_dataloader,
                "learning_rate": learning_rate * float(phase["lr_scale"]),
                "loss_config": phase_loss_config,
            }
        )
    _emit_progress(
        progress_callback,
        (
            f"[train] epochs={total_epoch_count} "
            f"phases={','.join(str(phase['name']) for phase in phase_runtimes)} "
            f"records={len(train_manifest_records)} "
            f"batches={sum(len(phase['dataloader']) * int(phase['epoch_count']) for phase in phase_runtimes)}"
        ),
    )
    for phase in phase_runtimes:
        _emit_progress(
            progress_callback,
            (
                f"[train] phase={phase['name']} "
                f"epochs={phase['epoch_count']} "
                f"lr={phase['learning_rate']:.6f} "
                f"cohorts={','.join(phase['enabled_cohorts'])} "
                f"records={len(phase['records'])} "
                f"batches_per_epoch={len(phase['dataloader'])}"
            ),
        )

    validation_manifest_snapshot_path: Path | None = None
    validation_manifest_summary_path: Path | None = None
    validation_history_path: Path | None = None
    validation_protocol_id: str | None = None
    validation_dataloader = None
    resolved_eval_config = _resolve_eval_config(eval_config_path)
    if validation_manifest_path is not None or validation_protocol_config_path is not None:
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

    epoch_history: list[TrainEpochSummary] = []
    validation_history: list[ValidationEpochSummary] = []
    best_checkpoint_path = checkpoints_dir / "checkpoint_best.pt"
    last_checkpoint_path = checkpoints_dir / "checkpoint_last.pt"
    checkpoint_selection_summary_path = records_dir / "checkpoint_selection_summary.json"
    best_epoch_summary: TrainEpochSummary | None = None
    best_validation_summary: ValidationEpochSummary | None = None
    best_rank: tuple[float, ...] | None = None
    last_epoch_summary: TrainEpochSummary | None = None
    last_validation_summary: ValidationEpochSummary | None = None

    epoch = 0
    for phase in phase_runtimes:
        _set_optimizer_learning_rate(optimizer, float(phase["learning_rate"]))
        for _ in range(int(phase["epoch_count"])):
            epoch += 1
            epoch_summary = _run_training_epoch(
                epoch=epoch,
                total_epoch_count=total_epoch_count,
                phase_name=str(phase["name"]),
                learning_rate=float(phase["learning_rate"]),
                model=model,
                dataloader=phase["dataloader"],
                optimizer=optimizer,
                runtime_spec=runtime_spec,
                loss_config=phase["loss_config"],
                progress_callback=progress_callback,
            )
            epoch_history.append(epoch_summary)

            validation_summary: ValidationEpochSummary | None = None
            if validation_dataloader is not None:
                validation_summary = _evaluate_validation_epoch(
                    epoch=epoch,
                    phase_name=str(phase["name"]),
                    model=model,
                    dataloader=validation_dataloader,
                    runtime_spec=runtime_spec,
                    run_id=resolved_run_id,
                    protocol_id=validation_protocol_id or protocol_config["protocol_id"],
                    resolved_eval_config=resolved_eval_config,
                )
                validation_history.append(validation_summary)
            last_epoch_summary = epoch_summary
            last_validation_summary = validation_summary

            if save_every_epochs > 0 and (epoch % save_every_epochs == 0):
                _save_checkpoint(
                    checkpoints_dir / f"checkpoint_epoch_{epoch:03d}.pt",
                    run_id=resolved_run_id,
                    epoch=epoch,
                    protocol_id=protocol_config["protocol_id"],
                    model=model,
                    optimizer=optimizer,
                    epoch_summary=epoch_summary,
                    validation_summary=validation_summary,
                )
            _save_checkpoint(
                last_checkpoint_path,
                run_id=resolved_run_id,
                epoch=epoch,
                protocol_id=protocol_config["protocol_id"],
                model=model,
                optimizer=optimizer,
                epoch_summary=epoch_summary,
                validation_summary=validation_summary,
            )

            epoch_message = (
                f"[train] epoch={epoch}/{total_epoch_count} "
                f"phase={phase['name']} "
                f"lr={epoch_summary.learning_rate:.6f} "
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

            candidate_rank = _best_rank(epoch_summary=epoch_summary, validation_summary=validation_summary)
            if best_rank is None or candidate_rank > best_rank:
                best_rank = candidate_rank
                best_epoch_summary = epoch_summary
                best_validation_summary = validation_summary
                _save_checkpoint(
                    best_checkpoint_path,
                    run_id=resolved_run_id,
                    epoch=epoch,
                    protocol_id=protocol_config["protocol_id"],
                    model=model,
                    optimizer=optimizer,
                    epoch_summary=epoch_summary,
                    validation_summary=validation_summary,
                )
                selection_rule = (
                    "validation_pair_auroc_then_easy_id_top1_then_train_mean_loss"
                    if validation_summary is not None
                    else "train_mean_loss_total"
                )
                _emit_progress(
                    progress_callback,
                    f"[train] best_checkpoint_updated epoch={epoch} criterion={selection_rule}",
                )

    if best_epoch_summary is None or last_epoch_summary is None:
        raise RuntimeError("Training completed without recording any epoch.")

    history_path = _write_epoch_history(epoch_history, records_dir / "train_history.csv")
    if validation_history:
        validation_history_path = _write_validation_history(
            validation_history,
            records_dir / "validation_history.csv",
        )
    selection_rule = (
        "validation_pair_auroc_then_easy_id_top1_then_train_mean_loss"
        if best_validation_summary is not None
        else "train_mean_loss_total"
    )
    checkpoint_selection_summary_path = _write_json(
        {
            "model_family": model_family,
            "best": _selection_payload(
                policy_name="best",
                checkpoint_path=best_checkpoint_path,
                epoch_summary=best_epoch_summary,
                validation_summary=best_validation_summary,
                selection_rule=selection_rule,
            ),
            "last": _selection_payload(
                policy_name="last",
                checkpoint_path=last_checkpoint_path,
                epoch_summary=last_epoch_summary,
                validation_summary=last_validation_summary,
                selection_rule="final_epoch",
            ),
        },
        checkpoint_selection_summary_path,
    )
    summary_path = _write_json(
        {
            "run_id": resolved_run_id,
            "model_family": model_family,
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
                "num_trainable_records": len(train_manifest_records),
                "cohort_summary": summarize_manifest(manifest_records),
            },
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
                "best_epoch": best_epoch_summary.epoch,
                "selection_summary_path": str(checkpoint_selection_summary_path),
            },
            "history_path": str(history_path),
            "validation": {
                "manifest_path": None if validation_manifest_snapshot_path is None else str(validation_manifest_snapshot_path),
                "summary_path": None if validation_manifest_summary_path is None else str(validation_manifest_summary_path),
                "history_path": None if validation_history_path is None else str(validation_history_path),
                "protocol_id": validation_protocol_id,
                "checkpoint_selection": selection_rule,
                "resolved_eval_config": {
                    "positive_cohort": resolved_eval_config["positive_cohort"],
                    "negative_cohort": resolved_eval_config["negative_cohort"],
                    "primary_pair": resolved_eval_config["primary_pair"],
                    "weighted_pair": resolved_eval_config["weighted_pair"],
                    "primary_scalar": resolved_eval_config["primary_scalar"],
                    "completion_scan_scalars": list(resolved_eval_config["completion_scan_scalars"]),
                    "emit_proposition_diagnostics": resolved_eval_config["emit_proposition_diagnostics"],
                    "test_size": resolved_eval_config["test_size"],
                    "random_state": resolved_eval_config["random_state"],
                },
                "best_epoch_metrics": None
                if best_validation_summary is None
                else best_validation_summary.to_csv_row(),
            },
            "epochs": [record.to_csv_row() for record in epoch_history],
            "validation_epochs": [record.to_csv_row() for record in validation_history],
            "training_schedule": [
                {
                    "name": str(phase["name"]),
                    "epoch_count": int(phase["epoch_count"]),
                    "enabled_cohorts": list(phase["enabled_cohorts"]),
                    "lr_scale": float(phase["lr_scale"]),
                    "learning_rate": float(phase["learning_rate"]),
                    "loss_overrides": dict(phase["loss_overrides"]),
                    "num_records": len(phase["records"]),
                    "batches_per_epoch": len(phase["dataloader"]),
                }
                for phase in phase_runtimes
            ],
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
        "model_family": model_family,
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
        "checkpoint_selection_summary_path": str(checkpoint_selection_summary_path),
        "last_checkpoint_path": str(last_checkpoint_path),
    }
