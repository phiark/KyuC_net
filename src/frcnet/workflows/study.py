from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import yaml

from frcnet.data import read_manifest_jsonl
from frcnet.evaluation import read_top1_proposition_records
from frcnet.workflows.plan_a import (
    _load_yaml_section,
    _write_json,
    build_plan_a_manifest_bundle,
    enforce_zero_source_overlap,
    export_plan_a_inference_bundle,
    generate_plan_a_artifact_bundle,
    prepare_plan_a_datasets,
    train_plan_a_model,
)
from frcnet.workflows.provenance import (
    build_stage_provenance,
    validate_stage_provenance,
    write_stage_provenance,
)


@dataclass(slots=True)
class StudyRunMetric:
    study_id: str
    model_family: str
    run_id: str
    seed: int
    pair_auroc: float
    weighted_pair_auroc: float
    scalar_auroc: float
    easy_id_top1_accuracy: float
    hard_id_top1_accuracy: float
    ambiguous_candidate_hit_rate: float
    run_output_dir: str
    seen_ood_pair_auroc: float = math.nan
    seen_ood_svhn_pair_auroc: float = math.nan
    seen_ood_dtd_pair_auroc: float = math.nan
    seen_ood_lsun_resize_pair_auroc: float = math.nan
    seen_ood_gaussian_noise_pair_auroc: float = math.nan
    seen_ood_tiny_imagenet_pair_auroc: float = math.nan
    seen_ood_cifar100_seen_classes_pair_auroc: float = math.nan
    seen_near_ood_pair_auroc: float = math.nan
    unseen_ood_pair_auroc: float = math.nan
    unseen_ood_cifar100_pair_auroc: float = math.nan
    unseen_ood_cifar100_heldout_classes_pair_auroc: float = math.nan
    all_ood_pair_auroc: float = math.nan
    worst_source_pair_auroc: float = math.nan
    seen_unseen_gap: float = math.nan
    near_ood_seen_unseen_gap: float = math.nan
    pair_scalar_delta: float = math.nan

    def to_csv_row(self) -> dict[str, str | int | float]:
        return {
            "study_id": self.study_id,
            "model_family": self.model_family,
            "run_id": self.run_id,
            "seed": self.seed,
            "pair_auroc": self.pair_auroc,
            "weighted_pair_auroc": self.weighted_pair_auroc,
            "scalar_auroc": self.scalar_auroc,
            "easy_id_top1_accuracy": self.easy_id_top1_accuracy,
            "hard_id_top1_accuracy": self.hard_id_top1_accuracy,
            "ambiguous_candidate_hit_rate": self.ambiguous_candidate_hit_rate,
            "seen_ood_pair_auroc": self.seen_ood_pair_auroc,
            "seen_ood_svhn_pair_auroc": self.seen_ood_svhn_pair_auroc,
            "seen_ood_dtd_pair_auroc": self.seen_ood_dtd_pair_auroc,
            "seen_ood_lsun_resize_pair_auroc": self.seen_ood_lsun_resize_pair_auroc,
            "seen_ood_gaussian_noise_pair_auroc": self.seen_ood_gaussian_noise_pair_auroc,
            "seen_ood_tiny_imagenet_pair_auroc": self.seen_ood_tiny_imagenet_pair_auroc,
            "seen_ood_cifar100_seen_classes_pair_auroc": self.seen_ood_cifar100_seen_classes_pair_auroc,
            "seen_near_ood_pair_auroc": self.seen_near_ood_pair_auroc,
            "unseen_ood_pair_auroc": self.unseen_ood_pair_auroc,
            "unseen_ood_cifar100_pair_auroc": self.unseen_ood_cifar100_pair_auroc,
            "unseen_ood_cifar100_heldout_classes_pair_auroc": self.unseen_ood_cifar100_heldout_classes_pair_auroc,
            "all_ood_pair_auroc": self.all_ood_pair_auroc,
            "worst_source_pair_auroc": self.worst_source_pair_auroc,
            "seen_unseen_gap": self.seen_unseen_gap,
            "near_ood_seen_unseen_gap": self.near_ood_seen_unseen_gap,
            "pair_scalar_delta": self.pair_scalar_delta,
            "run_output_dir": self.run_output_dir,
        }


@dataclass(slots=True)
class CheckpointPolicyMetric:
    study_id: str
    model_family: str
    run_id: str
    seed: int
    policy_name: str
    pair_auroc: float
    weighted_pair_auroc: float
    scalar_auroc: float
    easy_id_top1_accuracy: float
    hard_id_top1_accuracy: float
    ambiguous_candidate_hit_rate: float
    run_output_dir: str
    seen_ood_pair_auroc: float = math.nan
    seen_ood_svhn_pair_auroc: float = math.nan
    seen_ood_dtd_pair_auroc: float = math.nan
    seen_ood_lsun_resize_pair_auroc: float = math.nan
    seen_ood_gaussian_noise_pair_auroc: float = math.nan
    seen_ood_tiny_imagenet_pair_auroc: float = math.nan
    seen_ood_cifar100_seen_classes_pair_auroc: float = math.nan
    seen_near_ood_pair_auroc: float = math.nan
    unseen_ood_pair_auroc: float = math.nan
    unseen_ood_cifar100_pair_auroc: float = math.nan
    unseen_ood_cifar100_heldout_classes_pair_auroc: float = math.nan
    all_ood_pair_auroc: float = math.nan
    worst_source_pair_auroc: float = math.nan
    seen_unseen_gap: float = math.nan
    near_ood_seen_unseen_gap: float = math.nan
    pair_scalar_delta: float = math.nan

    def to_csv_row(self) -> dict[str, str | int | float]:
        return {
            "study_id": self.study_id,
            "model_family": self.model_family,
            "run_id": self.run_id,
            "seed": self.seed,
            "policy_name": self.policy_name,
            "pair_auroc": self.pair_auroc,
            "weighted_pair_auroc": self.weighted_pair_auroc,
            "scalar_auroc": self.scalar_auroc,
            "easy_id_top1_accuracy": self.easy_id_top1_accuracy,
            "hard_id_top1_accuracy": self.hard_id_top1_accuracy,
            "ambiguous_candidate_hit_rate": self.ambiguous_candidate_hit_rate,
            "seen_ood_pair_auroc": self.seen_ood_pair_auroc,
            "seen_ood_svhn_pair_auroc": self.seen_ood_svhn_pair_auroc,
            "seen_ood_dtd_pair_auroc": self.seen_ood_dtd_pair_auroc,
            "seen_ood_lsun_resize_pair_auroc": self.seen_ood_lsun_resize_pair_auroc,
            "seen_ood_gaussian_noise_pair_auroc": self.seen_ood_gaussian_noise_pair_auroc,
            "seen_ood_tiny_imagenet_pair_auroc": self.seen_ood_tiny_imagenet_pair_auroc,
            "seen_ood_cifar100_seen_classes_pair_auroc": self.seen_ood_cifar100_seen_classes_pair_auroc,
            "seen_near_ood_pair_auroc": self.seen_near_ood_pair_auroc,
            "unseen_ood_pair_auroc": self.unseen_ood_pair_auroc,
            "unseen_ood_cifar100_pair_auroc": self.unseen_ood_cifar100_pair_auroc,
            "unseen_ood_cifar100_heldout_classes_pair_auroc": self.unseen_ood_cifar100_heldout_classes_pair_auroc,
            "all_ood_pair_auroc": self.all_ood_pair_auroc,
            "worst_source_pair_auroc": self.worst_source_pair_auroc,
            "seen_unseen_gap": self.seen_unseen_gap,
            "near_ood_seen_unseen_gap": self.near_ood_seen_unseen_gap,
            "pair_scalar_delta": self.pair_scalar_delta,
            "run_output_dir": self.run_output_dir,
        }


AGGREGATE_METRIC_NAMES = (
    "pair_auroc",
    "weighted_pair_auroc",
    "scalar_auroc",
    "easy_id_top1_accuracy",
    "hard_id_top1_accuracy",
    "ambiguous_candidate_hit_rate",
    "seen_ood_pair_auroc",
    "seen_ood_svhn_pair_auroc",
    "seen_ood_dtd_pair_auroc",
    "seen_ood_lsun_resize_pair_auroc",
    "seen_ood_gaussian_noise_pair_auroc",
    "seen_ood_tiny_imagenet_pair_auroc",
    "seen_ood_cifar100_seen_classes_pair_auroc",
    "seen_near_ood_pair_auroc",
    "unseen_ood_pair_auroc",
    "unseen_ood_cifar100_pair_auroc",
    "unseen_ood_cifar100_heldout_classes_pair_auroc",
    "all_ood_pair_auroc",
    "worst_source_pair_auroc",
    "seen_unseen_gap",
    "near_ood_seen_unseen_gap",
    "pair_scalar_delta",
)


def _write_yaml_section(output_path: str | Path, section_name: str, payload: Mapping[str, Any]) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump({section_name: payload}, sort_keys=False), encoding="utf-8")
    return output


def _run_repo_script(script_name: str, args: Sequence[str], progress_callback: Callable[[str], None] | None) -> None:
    command = [sys.executable, f"scripts/{script_name}", *args]
    _emit_progress(progress_callback, "[study] run " + " ".join(command))
    subprocess.run(command, check=True)


def _emit_progress(progress_callback: Callable[[str], None] | None, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)


def _load_json_file(input_path: str | Path) -> dict[str, Any]:
    return json.loads(Path(input_path).read_text(encoding="utf-8"))


def _yaml_snapshot_matches(
    snapshot_path: str | Path,
    expected_config_path: str | Path | None,
    section_name: str,
) -> bool:
    if expected_config_path is None:
        return True
    snapshot = Path(snapshot_path)
    expected = Path(expected_config_path)
    if not snapshot.exists() or not expected.exists():
        return False
    return _load_yaml_section(snapshot, section_name) == _load_yaml_section(expected, section_name)


def _eval_requires_matched_manifest(eval_config_path: str | Path) -> bool:
    eval_config = _load_yaml_section(eval_config_path, "eval")
    benchmark_slices = [dict(value) for value in eval_config.get("benchmark_slices", [])]
    return bool(eval_config.get("require_matched_manifest", False)) or any(
        bool(value.get("require_matched_manifest", False)) for value in benchmark_slices
    )


def _train_uses_source_balanced_sampling(train_config_path: str | Path) -> bool:
    train_config = _load_yaml_section(train_config_path, "train")
    if bool(train_config.get("dataloader", {}).get("source_balanced_sampling", False)):
        return True
    phase_configs = train_config.get("training", {}).get("phases", [])
    return any(bool(dict(phase).get("dataloader", {}).get("source_balanced_sampling", False)) for phase in phase_configs)


def _validate_source_name_in_protocol(
    protocol_config: Mapping[str, Any],
    source_name: str,
    *,
    should_exist: bool,
    protocol_label: str,
) -> None:
    dataset_names = set(dict(protocol_config.get("datasets", {})))
    if should_exist and source_name not in dataset_names:
        raise ValueError(f"protocol_controls source `{source_name}` is missing from {protocol_label} protocol datasets.")
    if not should_exist and source_name in dataset_names:
        raise ValueError(f"protocol_controls source `{source_name}` must not appear in {protocol_label} protocol datasets.")


def _source_entries(protocol_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(value) for value in protocol_config.get("unknown_sources", [])] + [
        dict(value) for value in protocol_config.get("ood_sources", [])
    ]


def _source_entry_names(protocol_config: Mapping[str, Any]) -> set[str]:
    return {str(entry.get("dataset_name", entry.get("name", ""))) for entry in _source_entries(protocol_config)}


def _has_cifar100_class_range(
    protocol_config: Mapping[str, Any],
    *,
    start: int,
    stop: int,
    source_role: str | None = None,
) -> bool:
    for entry in _source_entries(protocol_config):
        if str(entry.get("dataset_name", entry.get("name", ""))) != "cifar100":
            continue
        if source_role is not None and str(entry.get("source_role", "")) != source_role:
            continue
        if int(entry.get("class_label_start", -1)) == start and int(entry.get("class_label_stop", -1)) == stop:
            return True
    return False


def _validate_protocol_controls(
    *,
    protocol_controls: Mapping[str, Any],
    train_protocol_config_path: str | Path,
    validation_protocol_config_path: str | Path,
    final_test_protocol_config_path: str | Path,
    train_config_path: str | Path,
    eval_config_path: str | Path,
    reference_config_path: str | Path | None,
) -> None:
    allowed_controls = {
        "source_balanced_sampling",
        "leave_one_source_out",
        "near_ood_training_source",
        "near_ood_training_sources",
        "require_source_overlap_zero",
        "require_strict_frozen_final",
        "protocol_variant",
    }
    unknown_controls = sorted(set(protocol_controls) - allowed_controls)
    if unknown_controls:
        raise ValueError(f"Unsupported study.protocol_controls keys: {', '.join(unknown_controls)}")

    if bool(protocol_controls.get("source_balanced_sampling", False)) and not _train_uses_source_balanced_sampling(
        train_config_path
    ):
        raise ValueError(
            "study.protocol_controls.source_balanced_sampling=true requires train.dataloader.source_balanced_sampling "
            "or a phase dataloader override to enable it."
        )

    if bool(protocol_controls.get("require_strict_frozen_final", False)):
        if not _eval_requires_matched_manifest(eval_config_path):
            raise ValueError(
                "study.protocol_controls.require_strict_frozen_final=true requires eval.require_matched_manifest=true "
                "or at least one strict benchmark slice."
            )
        if reference_config_path is None:
            raise ValueError("study.reference_config is required when require_strict_frozen_final=true.")

    train_protocol = _load_yaml_section(train_protocol_config_path, "protocol")
    validation_protocol = _load_yaml_section(validation_protocol_config_path, "protocol")
    final_test_protocol = _load_yaml_section(final_test_protocol_config_path, "protocol")

    holdout_source = protocol_controls.get("leave_one_source_out")
    if holdout_source:
        source_name = str(holdout_source)
        if str(protocol_controls.get("protocol_variant", "")) == "cifar100_class_holdout" and source_name == "cifar100":
            raise ValueError(
                "V0.6C protocol_variant=cifar100_class_holdout must not declare leave_one_source_out=cifar100; "
                "the claim is unseen CIFAR100 classes, not unseen CIFAR100 source."
            )
        _validate_source_name_in_protocol(train_protocol, source_name, should_exist=False, protocol_label="train")
        _validate_source_name_in_protocol(validation_protocol, source_name, should_exist=False, protocol_label="validation")
        _validate_source_name_in_protocol(final_test_protocol, source_name, should_exist=True, protocol_label="final_test")
        source_role = str(dict(final_test_protocol.get("source_roles", {})).get(source_name, ""))
        if source_role != "unseen_ood_source":
            raise ValueError(
                "study.protocol_controls.leave_one_source_out requires the held-out source to be marked "
                f"`unseen_ood_source` in the final-test protocol; got `{source_role}` for `{source_name}`."
            )

    near_ood_sources = []
    if protocol_controls.get("near_ood_training_source"):
        near_ood_sources.append(str(protocol_controls["near_ood_training_source"]))
    near_ood_sources.extend(str(value) for value in protocol_controls.get("near_ood_training_sources", []))
    for source_name in near_ood_sources:
        _validate_source_name_in_protocol(train_protocol, source_name, should_exist=True, protocol_label="train")
        _validate_source_name_in_protocol(validation_protocol, source_name, should_exist=True, protocol_label="validation")
        _validate_source_name_in_protocol(final_test_protocol, source_name, should_exist=True, protocol_label="final_test")
        source_role = str(dict(final_test_protocol.get("source_roles", {})).get(source_name, ""))
        if source_role == "unseen_ood_source":
            raise ValueError(
                "study.protocol_controls.near_ood_training_source must not be marked `unseen_ood_source` "
                f"in the final-test protocol; got `{source_name}`."
            )

    protocol_variant = str(protocol_controls.get("protocol_variant", ""))
    if protocol_variant == "cifar100_class_holdout":
        if "cifar100" not in _source_entry_names(train_protocol):
            raise ValueError("cifar100_class_holdout requires CIFAR100 seen classes in train unknown/OOD sources.")
        if "cifar100" not in _source_entry_names(validation_protocol):
            raise ValueError("cifar100_class_holdout requires CIFAR100 seen classes in validation unknown/OOD sources.")
        if not _has_cifar100_class_range(train_protocol, start=0, stop=50):
            raise ValueError("V0.6C train protocol must use CIFAR100 seen classes [0,50).")
        if not _has_cifar100_class_range(validation_protocol, start=0, stop=50):
            raise ValueError("V0.6C validation protocol must use CIFAR100 seen classes [0,50).")
        if not _has_cifar100_class_range(
            final_test_protocol,
            start=50,
            stop=100,
            source_role="unseen_ood_classes",
        ):
            raise ValueError("V0.6C final protocol must include CIFAR100 held-out classes [50,100).")


def _override_train_seed(train_config_path: str | Path, seed: int, output_path: str | Path) -> Path:
    train_config = _load_yaml_section(train_config_path, "train")
    training_config = dict(train_config.get("training", {}))
    training_config["seed"] = seed
    train_config["training"] = training_config
    return _write_yaml_section(output_path, "train", train_config)


def _load_existing_train_outputs(
    run_root: str | Path,
    run_id: str,
    *,
    expected_provenance: Mapping[str, Any],
    resume_policy: str,
    protocol_config_path: str | Path,
    model_config_path: str | Path,
    train_config_path: str | Path,
    validation_protocol_config_path: str | Path | None,
    eval_config_path: str | Path | None,
) -> dict[str, str] | None:
    training_root = Path(run_root) / "training"
    summary_path = training_root / "records" / "train_summary.json"
    if not summary_path.exists():
        return None
    if not validate_stage_provenance(
        training_root,
        expected_provenance,
        resume_policy=resume_policy,
        stage_label=f"{run_id}/training",
    ):
        return None

    snapshots_root = training_root / "snapshots"
    if not (
        _yaml_snapshot_matches(snapshots_root / "protocol_config_snapshot.yaml", protocol_config_path, "protocol")
        and _yaml_snapshot_matches(snapshots_root / "model_config_snapshot.yaml", model_config_path, "model")
        and _yaml_snapshot_matches(snapshots_root / "train_config_snapshot.yaml", train_config_path, "train")
        and _yaml_snapshot_matches(
            snapshots_root / "validation_protocol_config_snapshot.yaml",
            validation_protocol_config_path,
            "protocol",
        )
        and _yaml_snapshot_matches(snapshots_root / "eval_config_snapshot.yaml", eval_config_path, "eval")
    ):
        return None

    summary_payload = _load_json_file(summary_path)
    model_family = str(summary_payload.get("model_family", "frcnet_explicit_unknown"))
    checkpoints = summary_payload.get("checkpoints", {})
    best_policy_name = str(checkpoints.get("best_policy", "theory"))
    best_checkpoint_path = checkpoints.get("best")
    last_checkpoint_path = checkpoints.get("last")
    if not best_checkpoint_path or not Path(best_checkpoint_path).exists():
        return None

    protocol_id = ""
    protocol_snapshot_path = training_root / "snapshots" / "protocol_config_snapshot.yaml"
    if protocol_snapshot_path.exists():
        protocol_id = str(_load_yaml_section(protocol_snapshot_path, "protocol").get("protocol_id", ""))

    manifest_path = training_root / "manifests" / "train_manifest_snapshot.jsonl"
    manifest_summary_path = training_root / "manifests" / "train_manifest_summary.json"
    history_path = training_root / "records" / "train_history.csv"
    validation_history_path = training_root / "records" / "validation_history.csv"
    validation_manifest_path = training_root / "manifests" / "validation_manifest_snapshot.jsonl"

    return {
        "run_id": run_id,
        "model_family": model_family,
        "protocol_id": protocol_id,
        "output_dir": str(training_root),
        "manifest_path": str(manifest_path),
        "manifest_summary_path": str(manifest_summary_path),
        "history_path": str(history_path),
        "summary_path": str(summary_path),
        "validation_history_path": "" if not validation_history_path.exists() else str(validation_history_path),
        "validation_manifest_path": "" if not validation_manifest_path.exists() else str(validation_manifest_path),
        "best_checkpoint_path": str(best_checkpoint_path),
        "best_policy_name": best_policy_name,
        "best_theory_checkpoint_path": str(checkpoints.get("best_theory", "")),
        "best_balanced_checkpoint_path": str(checkpoints.get("best_balanced", "")),
        "best_near_ood_balanced_checkpoint_path": str(checkpoints.get("best_near_ood_balanced", "")),
        "checkpoint_selection_summary_path": str(checkpoints.get("selection_summary_path", "")),
        "last_checkpoint_path": "" if not last_checkpoint_path else str(last_checkpoint_path),
    }


def _load_existing_inference_outputs(
    run_root: str | Path,
    run_id: str,
    *,
    expected_provenance: Mapping[str, Any],
    resume_policy: str,
    protocol_config_path: str | Path,
    model_config_path: str | Path,
    checkpoint_path: str | Path,
    subdir_name: str = "analysis",
) -> dict[str, str] | None:
    analysis_root = Path(run_root) / subdir_name
    analysis_path = analysis_root / "sample_analysis_records.csv"
    proposition_path = analysis_root / "top1_proposition_records.csv"
    analysis_summary_path = analysis_root / "analysis_summary.json"
    protocol_snapshot_path = analysis_root / "protocol_config_snapshot.yaml"
    model_snapshot_path = analysis_root / "model_config_snapshot.yaml"
    manifest_snapshot_path = analysis_root / "plan_a_manifest_snapshot.jsonl"

    required_paths = (
        analysis_path,
        proposition_path,
        analysis_summary_path,
        protocol_snapshot_path,
        model_snapshot_path,
        manifest_snapshot_path,
    )
    if not all(path.exists() for path in required_paths):
        return None
    if not validate_stage_provenance(
        analysis_root,
        expected_provenance,
        resume_policy=resume_policy,
        stage_label=f"{run_id}/{subdir_name}",
    ):
        return None
    if not (
        _yaml_snapshot_matches(protocol_snapshot_path, protocol_config_path, "protocol")
        and _yaml_snapshot_matches(model_snapshot_path, model_config_path, "model")
    ):
        return None

    protocol_id = str(_load_yaml_section(protocol_snapshot_path, "protocol").get("protocol_id", ""))
    analysis_summary_payload = _load_json_file(analysis_summary_path)
    stored_checkpoint_path = analysis_summary_payload.get("checkpoint_path")
    if stored_checkpoint_path is None or Path(str(stored_checkpoint_path)).resolve() != Path(checkpoint_path).resolve():
        return None
    return {
        "run_id": run_id,
        "model_family": str(analysis_summary_payload.get("model_family", "frcnet_explicit_unknown")),
        "protocol_id": protocol_id,
        "output_dir": str(analysis_root),
        "protocol_snapshot_path": str(protocol_snapshot_path),
        "model_snapshot_path": str(model_snapshot_path),
        "manifest_snapshot_path": str(manifest_snapshot_path),
        "analysis_path": str(analysis_path),
        "proposition_path": str(proposition_path),
        "analysis_summary_path": str(analysis_summary_path),
        "checkpoint_selection_summary_path": str(
            analysis_summary_payload.get("checkpoint_selection_summary_path", "")
        ),
    }


def _load_existing_artifact_outputs(
    run_root: str | Path,
    run_id: str,
    *,
    expected_provenance: Mapping[str, Any],
    resume_policy: str,
    protocol_config_path: str | Path,
    eval_config_path: str | Path,
    analysis_config_path: str | Path | None = None,
    subdir_name: str = "report",
) -> dict[str, str] | None:
    report_root = Path(run_root) / subdir_name
    analysis_config = {"matched_table_name": "matched_ambiguous_vs_ood_table.csv"}
    if analysis_config_path is not None:
        analysis_config.update(_load_yaml_section(analysis_config_path, "analysis"))

    matched_path = report_root / str(analysis_config["matched_table_name"])
    artifact_index_path = report_root / "artifact_paths.json"
    experiment_record_path = report_root / "experiment_record.md"
    protocol_snapshot_path = report_root / "protocol_config_snapshot.yaml"
    eval_snapshot_path = report_root / "eval_config_snapshot.yaml"
    analysis_config_snapshot_path = report_root / "analysis_config_snapshot.yaml"
    analysis_summary_path = report_root / "analysis_summary.json"

    required_paths = (
        matched_path,
        artifact_index_path,
        experiment_record_path,
        protocol_snapshot_path,
        eval_snapshot_path,
        analysis_summary_path,
    )
    if not all(path.exists() for path in required_paths):
        return None
    if not validate_stage_provenance(
        report_root,
        expected_provenance,
        resume_policy=resume_policy,
        stage_label=f"{run_id}/{subdir_name}",
    ):
        return None
    if not (
        _yaml_snapshot_matches(protocol_snapshot_path, protocol_config_path, "protocol")
        and _yaml_snapshot_matches(eval_snapshot_path, eval_config_path, "eval")
        and _yaml_snapshot_matches(analysis_config_snapshot_path, analysis_config_path, "analysis")
    ):
        return None

    matched_row = _single_csv_row(matched_path)
    artifact_paths = _load_json_file(artifact_index_path)
    return {
        "run_id": run_id,
        "model_family": str(matched_row.get("model_family", "frcnet_explicit_unknown")),
        "protocol_id": str(matched_row.get("protocol_id", "")),
        "output_dir": str(report_root),
        "protocol_snapshot_path": str(protocol_snapshot_path),
        "eval_snapshot_path": str(eval_snapshot_path),
        "analysis_summary_path": str(analysis_summary_path),
        "artifact_index_path": str(artifact_index_path),
        "experiment_record_path": str(experiment_record_path),
        **{str(key): str(value) for key, value in artifact_paths.items()},
    }


def _single_csv_row(input_path: str | Path) -> dict[str, str]:
    with Path(input_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        try:
            return next(reader)
        except StopIteration as exc:
            raise ValueError(f"{input_path} does not contain any rows.") from exc


def _checkpoint_path_for_policy(train_output: Mapping[str, Any], policy_name: str) -> str:
    if policy_name == str(train_output.get("best_policy_name", "theory")):
        return str(train_output["best_checkpoint_path"])
    if policy_name == "theory":
        return str(train_output.get("best_theory_checkpoint_path", ""))
    if policy_name == "balanced":
        return str(train_output.get("best_balanced_checkpoint_path", ""))
    if policy_name == "near_ood_balanced":
        return str(train_output.get("best_near_ood_balanced_checkpoint_path", ""))
    if policy_name == "last":
        return str(train_output.get("last_checkpoint_path", ""))
    return ""


def _prepare_run_strict_eval_config(
    *,
    run_id: str,
    run_root: Path,
    train_protocol_config: str | Path,
    final_test_protocol_config: str | Path,
    model_config: str | Path,
    eval_config: str | Path,
    reference_config: str | Path | None,
    final_manifest_path: str | Path,
    analysis_path: str | Path,
    progress_callback: Callable[[str], None] | None,
) -> Path:
    eval_payload = _load_yaml_section(eval_config, "eval")
    benchmark_slices = [dict(value) for value in eval_payload.get("benchmark_slices", [])]
    strict_required = bool(eval_payload.get("require_matched_manifest", False)) or any(
        bool(value.get("require_matched_manifest", False)) for value in benchmark_slices
    )
    if not strict_required:
        return Path(eval_config)
    if reference_config is None:
        raise ValueError("study.reference_config is required when eval.require_matched_manifest=true.")

    reference_training_root = run_root / "reference" / "training"
    reference_checkpoint_path = reference_training_root / "checkpoints" / "checkpoint_best.pt"
    if not reference_checkpoint_path.exists():
        _run_repo_script(
            "train_softmax_reference.py",
            [
                "--protocol-config",
                str(train_protocol_config),
                "--model-config",
                str(model_config),
                "--reference-config",
                str(reference_config),
                "--output-dir",
                str(reference_training_root),
                "--run-id",
                f"{run_id}-softmax-reference",
            ],
            progress_callback,
        )

    reference_scores_root = run_root / "reference" / "final_test_scores"
    reference_scores_path = reference_scores_root / "reference_score_records.jsonl"
    if not reference_scores_path.exists():
        _run_repo_script(
            "run_softmax_reference_inference.py",
            [
                "--protocol-config",
                str(final_test_protocol_config),
                "--model-config",
                str(model_config),
                "--reference-config",
                str(reference_config),
                "--manifest-path",
                str(final_manifest_path),
                "--checkpoint-path",
                str(reference_checkpoint_path),
                "--output-dir",
                str(reference_scores_root),
                "--run-id",
                f"{run_id}-softmax-reference-final-test",
            ],
            progress_callback,
        )

    matched_root = run_root / "shared" / "matched_manifest"
    primary_manifest_path = matched_root / "ambiguous_vs_all_ood" / "frozen_matched_manifest.jsonl"
    primary_diagnostics_path = matched_root / "ambiguous_vs_all_ood" / "bin_diagnostics.csv"
    if not primary_manifest_path.exists():
        _run_repo_script(
            "build_frozen_matched_manifest.py",
            [
                "--analysis-path",
                str(analysis_path),
                "--reference-scores-path",
                str(reference_scores_path),
                "--eval-config",
                str(eval_config),
                "--output-path",
                str(primary_manifest_path),
                "--diagnostics-path",
                str(primary_diagnostics_path),
            ],
            progress_callback,
        )
    eval_payload["matched_manifest_path"] = str(primary_manifest_path)

    for benchmark_slice in benchmark_slices:
        benchmark_name = str(benchmark_slice["benchmark_name"])
        benchmark_stem = "".join(
            character if character.isalnum() or character in {"_", "-"} else "_" for character in benchmark_name
        )
        slice_manifest_path = matched_root / benchmark_stem / "frozen_matched_manifest.jsonl"
        slice_diagnostics_path = matched_root / benchmark_stem / "bin_diagnostics.csv"
        if not slice_manifest_path.exists():
            _run_repo_script(
                "build_frozen_matched_manifest.py",
                [
                    "--analysis-path",
                    str(analysis_path),
                    "--reference-scores-path",
                    str(reference_scores_path),
                    "--eval-config",
                    str(eval_config),
                    "--benchmark-name",
                    benchmark_name,
                    "--output-path",
                    str(slice_manifest_path),
                    "--diagnostics-path",
                    str(slice_diagnostics_path),
                ],
                progress_callback,
            )
        benchmark_slice["matched_manifest_path"] = str(slice_manifest_path)
    eval_payload["benchmark_slices"] = benchmark_slices

    generated_eval_config_path = run_root / "shared" / "generated_configs" / "eval_strict_frozen.yaml"
    return _write_yaml_section(generated_eval_config_path, "eval", eval_payload)


def _prepare_training_eval_config(eval_config: str | Path, output_path: str | Path) -> Path:
    eval_payload = _load_yaml_section(eval_config, "eval")
    benchmark_slices = [dict(value) for value in eval_payload.get("benchmark_slices", [])]
    strict_required = bool(eval_payload.get("require_matched_manifest", False)) or any(
        bool(value.get("require_matched_manifest", False)) for value in benchmark_slices
    )
    if not strict_required:
        return Path(eval_config)

    eval_payload["matched_manifest_path"] = ""
    eval_payload["require_matched_manifest"] = False
    if bool(eval_payload.get("preserve_benchmark_slices_for_validation", False)):
        for benchmark_slice in benchmark_slices:
            benchmark_slice["matched_manifest_path"] = ""
            benchmark_slice["require_matched_manifest"] = False
        eval_payload["benchmark_slices"] = benchmark_slices
    else:
        eval_payload["benchmark_slices"] = []
    return _write_yaml_section(output_path, "eval", eval_payload)


def _proposition_accuracy(proposition_path: str | Path, cohort_name: str) -> float:
    proposition_records = read_top1_proposition_records(proposition_path)
    cohort_records = [record for record in proposition_records if record.cohort_name == cohort_name]
    if not cohort_records:
        return 0.0
    correct_count = sum(int(record.is_top1_correct) for record in cohort_records)
    return correct_count / len(cohort_records)


def _optional_pair_auroc(report_output: Mapping[str, Any], artifact_key: str) -> float:
    artifact_path = report_output.get(artifact_key, "")
    if not artifact_path:
        return math.nan
    path = Path(str(artifact_path))
    if not path.exists():
        return math.nan
    return float(_single_csv_row(path)["pair_auroc"])


def _mean_non_nan(values: Sequence[float]) -> float:
    filtered = [float(value) for value in values if not math.isnan(float(value))]
    if not filtered:
        return math.nan
    return mean(filtered)


def _min_non_nan(values: Sequence[float]) -> float:
    filtered = [float(value) for value in values if not math.isnan(float(value))]
    if not filtered:
        return math.nan
    return min(filtered)


def _collect_source_slice_rows(
    *,
    study_id: str,
    seed: int,
    run_output: Mapping[str, Any],
) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    report_output = dict(run_output["report"])
    for artifact_key, artifact_path in sorted(report_output.items()):
        if artifact_key == "matched_ambiguous_vs_ood_table" or not artifact_key.endswith("_matched_table"):
            continue
        path = Path(str(artifact_path))
        if not path.exists():
            continue
        matched_row = _single_csv_row(path)
        rows.append(
            {
                "study_id": study_id,
                "run_id": str(run_output["run_id"]),
                "seed": seed,
                "benchmark_name": artifact_key.removesuffix("_matched_table"),
                "pair_auroc": float(matched_row["pair_auroc"]),
                "scalar_auroc": float(matched_row["scalar_auroc"]),
                "pair_scalar_delta": float(matched_row["pair_auroc"]) - float(matched_row["scalar_auroc"]),
                "matched_count_per_class": int(matched_row["matched_count_per_class"]),
                "num_ambiguous": int(matched_row["num_ambiguous"]),
                "num_ood": int(matched_row["num_ood"]),
                "table_path": str(path),
            }
        )
    return rows


def _collect_run_metric(study_id: str, seed: int, run_output: Mapping[str, Any]) -> StudyRunMetric:
    matched_row = _single_csv_row(run_output["report"]["matched_ambiguous_vs_ood_table"])
    proposition_path = run_output["analysis"]["proposition_path"]
    seen_svhn = _optional_pair_auroc(run_output["report"], "ambiguous_vs_seen_ood_svhn_matched_table")
    seen_dtd = _optional_pair_auroc(run_output["report"], "ambiguous_vs_seen_ood_dtd_matched_table")
    seen_lsun = _optional_pair_auroc(run_output["report"], "ambiguous_vs_seen_ood_lsun_resize_matched_table")
    seen_noise = _optional_pair_auroc(
        run_output["report"],
        "ambiguous_vs_seen_ood_gaussian_noise_matched_table",
    )
    seen_tiny = _optional_pair_auroc(
        run_output["report"],
        "ambiguous_vs_seen_ood_tiny_imagenet_matched_table",
    )
    seen_cifar100 = _optional_pair_auroc(
        run_output["report"],
        "ambiguous_vs_seen_ood_cifar100_seen_classes_matched_table",
    )
    unseen_cifar100_legacy = _optional_pair_auroc(
        run_output["report"],
        "ambiguous_vs_unseen_ood_cifar100_matched_table",
    )
    unseen_cifar100_heldout = _optional_pair_auroc(
        run_output["report"],
        "ambiguous_vs_unseen_ood_cifar100_heldout_classes_matched_table",
    )
    unseen_cifar100 = (
        unseen_cifar100_heldout if not math.isnan(unseen_cifar100_heldout) else unseen_cifar100_legacy
    )
    seen_mean = _mean_non_nan((seen_svhn, seen_dtd, seen_lsun, seen_noise, seen_tiny, seen_cifar100))
    seen_near = _mean_non_nan((seen_tiny, seen_cifar100))
    source_values = (seen_svhn, seen_dtd, seen_lsun, seen_noise, seen_tiny, seen_cifar100, unseen_cifar100)
    pair_auroc = float(matched_row["pair_auroc"])
    scalar_auroc = float(matched_row["scalar_auroc"])
    return StudyRunMetric(
        study_id=study_id,
        model_family=str(matched_row.get("model_family", run_output.get("model_family", "frcnet_explicit_unknown"))),
        run_id=str(run_output["run_id"]),
        seed=seed,
        pair_auroc=pair_auroc,
        weighted_pair_auroc=float(matched_row["weighted_pair_auroc"]),
        scalar_auroc=scalar_auroc,
        easy_id_top1_accuracy=_proposition_accuracy(proposition_path, "easy_id"),
        hard_id_top1_accuracy=_proposition_accuracy(proposition_path, "hard_id"),
        ambiguous_candidate_hit_rate=_proposition_accuracy(proposition_path, "ambiguous_id"),
        run_output_dir=str(run_output["output_dir"]),
        seen_ood_pair_auroc=seen_mean,
        seen_ood_svhn_pair_auroc=seen_svhn,
        seen_ood_dtd_pair_auroc=seen_dtd,
        seen_ood_lsun_resize_pair_auroc=seen_lsun,
        seen_ood_gaussian_noise_pair_auroc=seen_noise,
        seen_ood_tiny_imagenet_pair_auroc=seen_tiny,
        seen_ood_cifar100_seen_classes_pair_auroc=seen_cifar100,
        seen_near_ood_pair_auroc=seen_near,
        unseen_ood_pair_auroc=unseen_cifar100,
        unseen_ood_cifar100_pair_auroc=unseen_cifar100,
        unseen_ood_cifar100_heldout_classes_pair_auroc=unseen_cifar100_heldout,
        all_ood_pair_auroc=_optional_pair_auroc(run_output["report"], "ambiguous_vs_all_ood_matched_table"),
        worst_source_pair_auroc=_min_non_nan(source_values),
        seen_unseen_gap=math.nan if math.isnan(seen_mean) or math.isnan(unseen_cifar100) else seen_mean - unseen_cifar100,
        near_ood_seen_unseen_gap=math.nan
        if math.isnan(seen_near) or math.isnan(unseen_cifar100)
        else seen_near - unseen_cifar100,
        pair_scalar_delta=pair_auroc - scalar_auroc,
    )


def _collect_policy_metric(
    study_id: str,
    seed: int,
    policy_name: str,
    run_id: str,
    run_root: str | Path,
    analysis_output: Mapping[str, Any],
    report_output: Mapping[str, Any],
    *,
    fallback_model_family: str,
) -> CheckpointPolicyMetric:
    matched_row = _single_csv_row(report_output["matched_ambiguous_vs_ood_table"])
    proposition_path = analysis_output["proposition_path"]
    seen_values = (
        _optional_pair_auroc(report_output, "ambiguous_vs_seen_ood_svhn_matched_table"),
        _optional_pair_auroc(report_output, "ambiguous_vs_seen_ood_dtd_matched_table"),
        _optional_pair_auroc(report_output, "ambiguous_vs_seen_ood_lsun_resize_matched_table"),
        _optional_pair_auroc(report_output, "ambiguous_vs_seen_ood_gaussian_noise_matched_table"),
        _optional_pair_auroc(report_output, "ambiguous_vs_seen_ood_tiny_imagenet_matched_table"),
        _optional_pair_auroc(report_output, "ambiguous_vs_seen_ood_cifar100_seen_classes_matched_table"),
    )
    seen_svhn, seen_dtd, seen_lsun, seen_noise, seen_tiny, seen_cifar100 = seen_values
    unseen_cifar100_legacy = _optional_pair_auroc(report_output, "ambiguous_vs_unseen_ood_cifar100_matched_table")
    unseen_cifar100_heldout = _optional_pair_auroc(
        report_output,
        "ambiguous_vs_unseen_ood_cifar100_heldout_classes_matched_table",
    )
    unseen_cifar100 = (
        unseen_cifar100_heldout if not math.isnan(unseen_cifar100_heldout) else unseen_cifar100_legacy
    )
    seen_mean = _mean_non_nan(seen_values)
    seen_near = _mean_non_nan((seen_tiny, seen_cifar100))
    pair_auroc = float(matched_row["pair_auroc"])
    scalar_auroc = float(matched_row["scalar_auroc"])
    return CheckpointPolicyMetric(
        study_id=study_id,
        model_family=str(matched_row.get("model_family", fallback_model_family)),
        run_id=run_id,
        seed=seed,
        policy_name=policy_name,
        pair_auroc=pair_auroc,
        weighted_pair_auroc=float(matched_row["weighted_pair_auroc"]),
        scalar_auroc=scalar_auroc,
        easy_id_top1_accuracy=_proposition_accuracy(proposition_path, "easy_id"),
        hard_id_top1_accuracy=_proposition_accuracy(proposition_path, "hard_id"),
        ambiguous_candidate_hit_rate=_proposition_accuracy(proposition_path, "ambiguous_id"),
        run_output_dir=str(run_root),
        seen_ood_pair_auroc=seen_mean,
        seen_ood_svhn_pair_auroc=seen_svhn,
        seen_ood_dtd_pair_auroc=seen_dtd,
        seen_ood_lsun_resize_pair_auroc=seen_lsun,
        seen_ood_gaussian_noise_pair_auroc=seen_noise,
        seen_ood_tiny_imagenet_pair_auroc=seen_tiny,
        seen_ood_cifar100_seen_classes_pair_auroc=seen_cifar100,
        seen_near_ood_pair_auroc=seen_near,
        unseen_ood_pair_auroc=unseen_cifar100,
        unseen_ood_cifar100_pair_auroc=unseen_cifar100,
        unseen_ood_cifar100_heldout_classes_pair_auroc=unseen_cifar100_heldout,
        all_ood_pair_auroc=_optional_pair_auroc(report_output, "ambiguous_vs_all_ood_matched_table"),
        worst_source_pair_auroc=_min_non_nan((*seen_values, unseen_cifar100)),
        seen_unseen_gap=math.nan if math.isnan(seen_mean) or math.isnan(unseen_cifar100) else seen_mean - unseen_cifar100,
        near_ood_seen_unseen_gap=math.nan
        if math.isnan(seen_near) or math.isnan(unseen_cifar100)
        else seen_near - unseen_cifar100,
        pair_scalar_delta=pair_auroc - scalar_auroc,
    )


def _write_seed_metrics(metrics: Sequence[StudyRunMetric], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(metrics[0].to_csv_row().keys()) if metrics else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for metric in metrics:
                writer.writerow(metric.to_csv_row())
    return output


def _write_metric_summary(metrics: Sequence[StudyRunMetric], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["metric_name", "mean", "std", "min", "max"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metric_name in AGGREGATE_METRIC_NAMES:
            values = [
                float(getattr(metric, metric_name))
                for metric in metrics
                if not math.isnan(float(getattr(metric, metric_name)))
            ]
            if not values:
                writer.writerow({"metric_name": metric_name, "mean": "", "std": "", "min": "", "max": ""})
                continue
            writer.writerow(
                {
                    "metric_name": metric_name,
                    "mean": mean(values),
                    "std": 0.0 if len(values) == 1 else pstdev(values),
                    "min": min(values),
                    "max": max(values),
                }
            )
    return output


def _write_source_slice_metrics(rows: Sequence[Mapping[str, str | int | float]], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "study_id",
        "run_id",
        "seed",
        "benchmark_name",
        "pair_auroc",
        "scalar_auroc",
        "pair_scalar_delta",
        "matched_count_per_class",
        "num_ambiguous",
        "num_ood",
        "table_path",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({fieldname: row.get(fieldname, "") for fieldname in fieldnames})
    return output


def _write_source_slice_summary(rows: Sequence[Mapping[str, str | int | float]], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[float]] = {}
    for row in rows:
        benchmark_name = str(row["benchmark_name"])
        grouped.setdefault(benchmark_name, []).append(float(row["pair_auroc"]))
    with output.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["benchmark_name", "mean_pair_auroc", "std_pair_auroc", "min_pair_auroc", "max_pair_auroc"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for benchmark_name, values in sorted(grouped.items()):
            writer.writerow(
                {
                    "benchmark_name": benchmark_name,
                    "mean_pair_auroc": mean(values),
                    "std_pair_auroc": 0.0 if len(values) == 1 else pstdev(values),
                    "min_pair_auroc": min(values),
                    "max_pair_auroc": max(values),
                }
            )
    return output


def _write_checkpoint_policy_metrics(metrics: Sequence[CheckpointPolicyMetric], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(metrics[0].to_csv_row().keys()) if metrics else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for metric in metrics:
                writer.writerow(metric.to_csv_row())
    return output


def _write_checkpoint_policy_summary(metrics: Sequence[CheckpointPolicyMetric], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    policy_names = sorted({metric.policy_name for metric in metrics})
    with output.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["policy_name", "metric_name", "mean", "std", "min", "max"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for policy_name in policy_names:
            policy_metrics = [metric for metric in metrics if metric.policy_name == policy_name]
            for metric_name in AGGREGATE_METRIC_NAMES:
                values = [
                    float(getattr(metric, metric_name))
                    for metric in policy_metrics
                    if not math.isnan(float(getattr(metric, metric_name)))
                ]
                if not values:
                    writer.writerow(
                        {"policy_name": policy_name, "metric_name": metric_name, "mean": "", "std": "", "min": "", "max": ""}
                    )
                    continue
                writer.writerow(
                    {
                        "policy_name": policy_name,
                        "metric_name": metric_name,
                        "mean": mean(values),
                        "std": 0.0 if len(values) == 1 else pstdev(values),
                        "min": min(values),
                        "max": max(values),
                    }
                )
    return output


def _write_checkpoint_policy_gap_summary(
    metrics: Sequence[CheckpointPolicyMetric],
    output_path: str | Path,
    *,
    minuend_policy: str,
    subtrahend_policy: str,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    by_seed_policy = {(metric.seed, metric.policy_name): metric for metric in metrics}
    shared_seeds = sorted(
        {
            metric.seed
            for metric in metrics
            if (metric.seed, minuend_policy) in by_seed_policy and (metric.seed, subtrahend_policy) in by_seed_policy
        }
    )
    with output.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["seed", "metric_name", "minuend_policy", "subtrahend_policy", "delta"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for seed in shared_seeds:
            minuend_metric = by_seed_policy[(seed, minuend_policy)]
            subtrahend_metric = by_seed_policy[(seed, subtrahend_policy)]
            for metric_name in AGGREGATE_METRIC_NAMES:
                minuend_value = float(getattr(minuend_metric, metric_name))
                subtrahend_value = float(getattr(subtrahend_metric, metric_name))
                writer.writerow(
                    {
                        "seed": seed,
                        "metric_name": metric_name,
                        "minuend_policy": minuend_policy,
                        "subtrahend_policy": subtrahend_policy,
                        "delta": ""
                        if math.isnan(minuend_value) or math.isnan(subtrahend_value)
                        else minuend_value - subtrahend_value,
                    }
                )
    return output


def _validate_required_source_slices(
    *,
    rows: Sequence[Mapping[str, str | int | float]],
    required_benchmark_slices: Sequence[str],
    seeds: Sequence[int],
) -> None:
    if not required_benchmark_slices:
        return
    available = {
        (int(row["seed"]), str(row["benchmark_name"]))
        for row in rows
        if "seed" in row and "benchmark_name" in row
    }
    missing = [
        f"seed{seed:03d}:{benchmark_name}"
        for seed in seeds
        for benchmark_name in required_benchmark_slices
        if (int(seed), str(benchmark_name)) not in available
    ]
    if missing:
        raise ValueError(f"Required benchmark slices are missing: {', '.join(missing)}")


def _write_metric_plot(
    metrics: Sequence[StudyRunMetric],
    *,
    output_path: str | Path,
    metric_names: Sequence[str],
    title: str,
    y_label: str,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    x_positions = list(range(len(metrics)))
    width = 0.22 if len(metric_names) > 1 else 0.45

    plt.figure(figsize=(9, 5))
    for index, metric_name in enumerate(metric_names):
        offset = (index - ((len(metric_names) - 1) / 2.0)) * width
        values = [float(getattr(metric, metric_name)) for metric in metrics]
        plt.bar([position + offset for position in x_positions], values, width=width, label=metric_name)

    plt.xticks(x_positions, [f"seed{metric.seed:03d}" for metric in metrics], rotation=15)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()
    return output


def aggregate_plan_a_study_bundle(
    *,
    study_root: str | Path,
    study_config_path: str | Path,
    output_dir: str | Path | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, str]:
    study_root_path = Path(study_root)
    output_root = Path(output_dir) if output_dir is not None else study_root_path / "aggregate"
    output_root.mkdir(parents=True, exist_ok=True)
    _emit_progress(progress_callback, f"[study] aggregate_start study_root={study_root_path}")

    study_config = _load_yaml_section(study_config_path, "study")
    study_paths_path = study_root_path / "study_paths.json"
    if not study_paths_path.exists():
        raise ValueError(f"Missing study_paths.json at {study_paths_path}. Run the study workflow first.")
    study_paths = json.loads(study_paths_path.read_text(encoding="utf-8"))
    run_outputs = list(study_paths.get("runs", []))
    if not run_outputs:
        raise ValueError("Study paths do not contain any completed runs.")

    seeds = [int(seed) for seed in study_paths["seeds"]]
    metrics = [
        _collect_run_metric(str(study_paths["study_id"]), seed, run_output)
        for seed, run_output in zip(seeds, run_outputs, strict=True)
    ]
    source_slice_rows: list[dict[str, str | int | float]] = []
    for seed, run_output in zip(seeds, run_outputs, strict=True):
        source_slice_rows.extend(
            _collect_source_slice_rows(
                study_id=str(study_paths["study_id"]),
                seed=seed,
                run_output=run_output,
            )
        )
    required_benchmark_slices = tuple(
        str(value)
        for value in study_config.get(
            "required_benchmark_slices",
            study_config.get("report_policy", {}).get("required_benchmark_slices", ()),
        )
    )
    _validate_required_source_slices(
        rows=source_slice_rows,
        required_benchmark_slices=required_benchmark_slices,
        seeds=seeds,
    )
    policy_metrics: list[CheckpointPolicyMetric] = []
    for seed, run_output in zip(seeds, run_outputs, strict=True):
        policy_outputs = dict(run_output.get("policy_outputs", {}))
        if not policy_outputs:
            primary_policy_name = str(run_output.get("primary_checkpoint_policy", "theory"))
            policy_outputs[primary_policy_name] = {
                "analysis": run_output["analysis"],
                "report": run_output["report"],
            }
        for policy_name, policy_output in sorted(policy_outputs.items()):
            policy_metrics.append(
                _collect_policy_metric(
                    str(study_paths["study_id"]),
                    seed,
                    str(policy_name),
                    str(run_output["run_id"]),
                    run_output["output_dir"],
                    policy_output["analysis"],
                    policy_output["report"],
                    fallback_model_family=str(run_output.get("model_family", study_paths.get("model_family", "frcnet_explicit_unknown"))),
                )
            )

    seed_metrics_path = _write_seed_metrics(metrics, output_root / "seed_metrics.csv")
    metric_summary_path = _write_metric_summary(metrics, output_root / "metric_summary.csv")
    source_slice_metrics_path = _write_source_slice_metrics(
        source_slice_rows,
        output_root / "source_slice_metrics.csv",
    )
    source_slice_summary_path = _write_source_slice_summary(
        source_slice_rows,
        output_root / "source_slice_summary.csv",
    )
    checkpoint_policy_metrics_path = _write_checkpoint_policy_metrics(
        policy_metrics,
        output_root / "checkpoint_policy_metrics.csv",
    )
    checkpoint_policy_summary_path = _write_checkpoint_policy_summary(
        policy_metrics,
        output_root / "checkpoint_policy_summary.csv",
    )
    report_policy = dict(study_config.get("report_policy", {}))
    checkpoint_policy_gap_summary_path = _write_checkpoint_policy_gap_summary(
        policy_metrics,
        output_root / "checkpoint_policy_gap_summary.csv",
        minuend_policy=str(report_policy.get("gap_minuend_policy", "balanced")),
        subtrahend_policy=str(report_policy.get("gap_subtrahend_policy", "theory")),
    )

    ranking_metric = str(report_policy.get("ranking_metric", "pair_auroc"))
    if not hasattr(metrics[0], ranking_metric):
        raise ValueError(f"Unsupported study report_policy.ranking_metric: `{ranking_metric}`.")
    missing_ranking_runs = [
        metric.run_id for metric in metrics if math.isnan(float(getattr(metric, ranking_metric)))
    ]
    if missing_ranking_runs:
        raise ValueError(
            f"Study ranking metric `{ranking_metric}` is missing or NaN for runs: "
            f"{', '.join(missing_ranking_runs)}"
        )
    ranked_metrics = sorted(metrics, key=lambda metric: float(getattr(metric, ranking_metric)), reverse=True)
    best_metric = ranked_metrics[0]
    worst_metric = ranked_metrics[-1]
    median_metric = ranked_metrics[len(ranked_metrics) // 2]
    rankings_path = _write_json(
        {
            "ranking_metric": ranking_metric,
            "best_run_id": best_metric.run_id,
            "worst_run_id": worst_metric.run_id,
            "median_run_id": median_metric.run_id,
            "best_seed": best_metric.seed,
            "worst_seed": worst_metric.seed,
            "median_seed": median_metric.seed,
        },
        output_root / "seed_rankings.json",
    )

    auroc_plot_path = _write_metric_plot(
        metrics,
        output_path=output_root / "auroc_by_seed.png",
        metric_names=("pair_auroc", "weighted_pair_auroc", "scalar_auroc"),
        title="Matched AUROC By Seed",
        y_label="auroc",
    )
    proposition_plot_path = _write_metric_plot(
        metrics,
        output_path=output_root / "proposition_accuracy_by_seed.png",
        metric_names=("easy_id_top1_accuracy", "hard_id_top1_accuracy", "ambiguous_candidate_hit_rate"),
        title="Top-1 Proposition Accuracy By Seed",
        y_label="accuracy",
    )

    artifact_paths = {
        "seed_metrics": str(seed_metrics_path),
        "metric_summary": str(metric_summary_path),
        "source_slice_metrics": str(source_slice_metrics_path),
        "source_slice_summary": str(source_slice_summary_path),
        "checkpoint_policy_metrics": str(checkpoint_policy_metrics_path),
        "checkpoint_policy_summary": str(checkpoint_policy_summary_path),
        "checkpoint_policy_gap_summary": str(checkpoint_policy_gap_summary_path),
        "seed_rankings": str(rankings_path),
        "auroc_by_seed": str(auroc_plot_path),
        "proposition_accuracy_by_seed": str(proposition_plot_path),
    }
    artifact_index_path = _write_json(artifact_paths, output_root / "artifact_paths.json")

    lines = [
        f"# Study Record: {study_paths['study_id']}",
        "",
        f"- study_id: `{study_paths['study_id']}`",
        f"- model_family: `{study_paths.get('model_family', 'frcnet_explicit_unknown')}`",
        f"- study_config_path: `{study_config_path}`",
        f"- shared_eval_manifest: `{study_paths['shared_eval_manifest_path']}`",
        "",
        "## Matched Benchmark",
        "",
        f"- ranking_metric: `{ranking_metric}`",
        f"- primary_checkpoint_policy: `{study_paths.get('primary_checkpoint_policy', 'theory')}`",
        f"- best_run_id: `{best_metric.run_id}`",
        f"- worst_run_id: `{worst_metric.run_id}`",
        f"- median_run_id: `{median_metric.run_id}`",
        "",
        "## Proposition Diagnostics",
        "",
        f"- companion_checkpoint_policies: `{json.dumps(study_paths.get('companion_checkpoint_policies', []))}`",
        f"- checkpoint_policy_summary: `{checkpoint_policy_summary_path}`",
        "",
        "## Seeds",
        "",
    ]
    for metric in metrics:
        lines.append(
            f"- `{metric.run_id}` seed={metric.seed} "
            f"pair_auroc={metric.pair_auroc:.6f} easy_id_top1={metric.easy_id_top1_accuracy:.6f}"
        )
    lines.extend(["", "## Checkpoint Policies", ""])
    for policy_name in sorted({metric.policy_name for metric in policy_metrics}):
        policy_subset = [metric for metric in policy_metrics if metric.policy_name == policy_name]
        lines.append(
            f"- `{policy_name}` seeds={len(policy_subset)} "
            f"pair_mean={mean(metric.pair_auroc for metric in policy_subset):.6f} "
            f"hard_top1_mean={mean(metric.hard_id_top1_accuracy for metric in policy_subset):.6f}"
        )
    lines.extend(["", "## Aggregate Artifacts", ""])
    for artifact_name, artifact_path in sorted(artifact_paths.items()):
        lines.append(f"- {artifact_name}: `{artifact_path}`")
    experiment_record_path = output_root / "experiment_record.md"
    experiment_record_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _emit_progress(progress_callback, f"[study] aggregate_complete record={experiment_record_path}")

    return {
        "output_dir": str(output_root),
        "seed_metrics_path": str(seed_metrics_path),
        "metric_summary_path": str(metric_summary_path),
        "source_slice_metrics_path": str(source_slice_metrics_path),
        "source_slice_summary_path": str(source_slice_summary_path),
        "checkpoint_policy_metrics_path": str(checkpoint_policy_metrics_path),
        "checkpoint_policy_summary_path": str(checkpoint_policy_summary_path),
        "checkpoint_policy_gap_summary_path": str(checkpoint_policy_gap_summary_path),
        "seed_rankings_path": str(rankings_path),
        "artifact_index_path": str(artifact_index_path),
        "experiment_record_path": str(experiment_record_path),
        "auroc_plot_path": str(auroc_plot_path),
        "proposition_plot_path": str(proposition_plot_path),
    }


def run_plan_a_study_bundle(
    *,
    study_config_path: str | Path,
    output_dir: str | Path | None = None,
    download_override: bool | None = None,
    aggregate_after_run: bool = True,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, str]:
    study_config = _load_yaml_section(study_config_path, "study")
    study_id = str(study_config["study_id"])
    study_root = Path(output_dir) if output_dir is not None else Path(study_config["output_root"]) / study_id
    study_root.mkdir(parents=True, exist_ok=True)
    _emit_progress(progress_callback, f"[study] study_id={study_id} output_dir={study_root}")

    train_protocol_config = study_config["train_protocol_config"]
    analysis_protocol_config = study_config.get("analysis_protocol_config", study_config.get("final_test_protocol_config"))
    validation_protocol_config = study_config.get("validation_protocol_config", analysis_protocol_config)
    final_test_protocol_config = study_config.get("final_test_protocol_config", analysis_protocol_config)
    model_config = study_config["model_config"]
    train_config = study_config["train_config"]
    eval_config = study_config["eval_config"]
    analysis_config = study_config["analysis_config"]
    reference_config = study_config.get("reference_config")
    seeds = [int(seed) for seed in study_config.get("seeds", (7, 17, 27))]
    model_family = str(study_config.get("model_family", "frcnet_explicit_unknown"))
    resume_policy = str(study_config.get("resume_policy", "fail_on_stale"))
    report_policy = dict(study_config.get("report_policy", {}))
    primary_checkpoint_policy = str(report_policy.get("primary_checkpoint_policy", "theory"))
    companion_checkpoint_policies = [
        str(policy_name) for policy_name in report_policy.get("companion_checkpoint_policies", [])
    ]
    training_eval_config = study_config.get("training_eval_config")
    if training_eval_config is None:
        training_eval_config = _prepare_training_eval_config(
            eval_config,
            study_root / "shared" / "generated_configs" / "eval_validation_selection.yaml",
        )
    protocol_controls = dict(study_config.get("protocol_controls", {}))
    _validate_protocol_controls(
        protocol_controls=protocol_controls,
        train_protocol_config_path=train_protocol_config,
        validation_protocol_config_path=validation_protocol_config,
        final_test_protocol_config_path=final_test_protocol_config,
        train_config_path=train_config,
        eval_config_path=eval_config,
        reference_config_path=reference_config,
    )

    prepare_plan_a_datasets(
        list(dict.fromkeys([train_protocol_config, validation_protocol_config, final_test_protocol_config])),
        output_path=study_root / "data_preflight.json",
        download_override=download_override,
    )
    _emit_progress(progress_callback, f"[study] data_preflight={study_root / 'data_preflight.json'}")

    validation_manifest_outputs = build_plan_a_manifest_bundle(
        protocol_config_path=validation_protocol_config,
        output_dir=study_root / "shared" / "validation_manifest",
        manifest_filename="plan_a_manifest.jsonl",
        summary_filename="plan_a_manifest_summary.json",
    )
    _emit_progress(progress_callback, f"[study] shared_validation_manifest={validation_manifest_outputs['manifest_path']}")
    final_manifest_outputs = build_plan_a_manifest_bundle(
        protocol_config_path=final_test_protocol_config,
        output_dir=study_root / "shared" / "final_test_manifest",
        manifest_filename="plan_a_manifest.jsonl",
        summary_filename="plan_a_manifest_summary.json",
    )
    _emit_progress(progress_callback, f"[study] shared_final_test_manifest={final_manifest_outputs['manifest_path']}")
    require_source_overlap_zero = bool(protocol_controls.get("require_source_overlap_zero", False))
    validation_manifest_records = read_manifest_jsonl(validation_manifest_outputs["manifest_path"])
    final_manifest_records = read_manifest_jsonl(final_manifest_outputs["manifest_path"])
    if require_source_overlap_zero:
        enforce_zero_source_overlap(
            {
                "validation": validation_manifest_records,
                "final_test": final_manifest_records,
            }
        )

    run_outputs: list[dict[str, Any]] = []
    for seed in seeds:
        run_id = f"{study_id}-seed{seed:03d}"
        run_root = study_root / "runs" / run_id
        _emit_progress(progress_callback, f"[study] seed_start run_id={run_id} seed={seed}")
        generated_train_config_path = _override_train_seed(
            train_config,
            seed,
            study_root / "shared" / "generated_configs" / f"train_seed{seed:03d}.yaml",
        )
        training_provenance = build_stage_provenance(
            stage_name="training",
            input_files=(
                study_config_path,
                train_protocol_config,
                validation_protocol_config,
                model_config,
                generated_train_config_path,
                training_eval_config,
            ),
            input_values={
                "run_id": run_id,
                "seed": seed,
                "primary_checkpoint_policy": primary_checkpoint_policy,
            },
        )
        train_outputs = _load_existing_train_outputs(
            run_root,
            run_id,
            expected_provenance=training_provenance,
            resume_policy=resume_policy,
            protocol_config_path=train_protocol_config,
            model_config_path=model_config,
            train_config_path=generated_train_config_path,
            validation_protocol_config_path=validation_protocol_config,
            eval_config_path=training_eval_config,
        )
        training_resumed = train_outputs is not None
        if train_outputs is None:
            train_outputs = train_plan_a_model(
                protocol_config_path=train_protocol_config,
                model_config_path=model_config,
                train_config_path=generated_train_config_path,
                output_dir=run_root / "training",
                run_id=run_id,
                validation_protocol_config_path=validation_protocol_config,
                validation_manifest_path=validation_manifest_outputs["manifest_path"],
                eval_config_path=training_eval_config,
                progress_callback=progress_callback,
            )
            write_stage_provenance(run_root / "training", training_provenance)
        else:
            _emit_progress(
                progress_callback,
                f"[study] seed_resume_training run_id={run_id} best_checkpoint={train_outputs['best_checkpoint_path']}",
            )
        if require_source_overlap_zero:
            train_manifest_records = read_manifest_jsonl(train_outputs["manifest_path"])
            enforce_zero_source_overlap(
                {
                    "train": train_manifest_records,
                    "validation": validation_manifest_records,
                    "final_test": final_manifest_records,
                }
            )

        primary_checkpoint_path = _checkpoint_path_for_policy(train_outputs, primary_checkpoint_policy)
        if not primary_checkpoint_path or not Path(primary_checkpoint_path).exists():
            raise ValueError(
                f"Primary checkpoint policy `{primary_checkpoint_policy}` did not resolve to an existing checkpoint for {run_id}."
            )

        analysis_provenance = build_stage_provenance(
            stage_name="analysis",
            input_files=(
                final_test_protocol_config,
                model_config,
                final_manifest_outputs["manifest_path"],
                primary_checkpoint_path,
            ),
            input_values={
                "run_id": run_id,
                "checkpoint_policy": primary_checkpoint_policy,
            },
        )
        inference_outputs = (
            _load_existing_inference_outputs(
                run_root,
                run_id,
                expected_provenance=analysis_provenance,
                resume_policy=resume_policy,
                protocol_config_path=final_test_protocol_config,
                model_config_path=model_config,
                checkpoint_path=primary_checkpoint_path,
                subdir_name="analysis",
            )
            if training_resumed
            else None
        )
        inference_resumed = inference_outputs is not None
        if inference_outputs is None:
            inference_outputs = export_plan_a_inference_bundle(
                protocol_config_path=final_test_protocol_config,
                model_config_path=model_config,
                manifest_path=final_manifest_outputs["manifest_path"],
                output_dir=run_root / "analysis",
                run_id=run_id,
                checkpoint_path=primary_checkpoint_path,
                checkpoint_selection_summary_path=train_outputs.get("checkpoint_selection_summary_path"),
                model_family=str(train_outputs.get("model_family", model_family)),
            )
            write_stage_provenance(run_root / "analysis", analysis_provenance)
        else:
            _emit_progress(
                progress_callback,
                f"[study] seed_resume_analysis run_id={run_id} analysis_csv={inference_outputs['analysis_path']}",
            )

        run_eval_config = _prepare_run_strict_eval_config(
            run_id=run_id,
            run_root=run_root,
            train_protocol_config=train_protocol_config,
            final_test_protocol_config=final_test_protocol_config,
            model_config=model_config,
            eval_config=eval_config,
            reference_config=reference_config,
            final_manifest_path=final_manifest_outputs["manifest_path"],
            analysis_path=inference_outputs["analysis_path"],
            progress_callback=progress_callback,
        )

        report_provenance = build_stage_provenance(
            stage_name="report",
            input_files=(
                final_test_protocol_config,
                run_eval_config,
                analysis_config,
                inference_outputs["analysis_summary_path"],
                inference_outputs["analysis_path"],
            ),
            input_values={
                "run_id": run_id,
                "checkpoint_policy": primary_checkpoint_policy,
            },
        )
        artifact_outputs = (
            _load_existing_artifact_outputs(
                run_root,
                run_id,
                expected_provenance=report_provenance,
                resume_policy=resume_policy,
                protocol_config_path=final_test_protocol_config,
                eval_config_path=run_eval_config,
                analysis_config_path=analysis_config,
                subdir_name="report",
            )
            if inference_resumed
            else None
        )
        if artifact_outputs is None:
            artifact_outputs = generate_plan_a_artifact_bundle(
                analysis_path=inference_outputs["analysis_path"],
                analysis_summary_path=inference_outputs["analysis_summary_path"],
                protocol_config_path=final_test_protocol_config,
                eval_config_path=run_eval_config,
                analysis_config_path=analysis_config,
                output_dir=run_root / "report",
            )
            write_stage_provenance(run_root / "report", report_provenance)
        else:
            _emit_progress(
                progress_callback,
                f"[study] seed_resume_report run_id={run_id} record={artifact_outputs['experiment_record_path']}",
            )
        policy_outputs: dict[str, dict[str, Any]] = {
            primary_checkpoint_policy: {
                "analysis": inference_outputs,
                "report": artifact_outputs,
            }
        }
        for companion_policy in companion_checkpoint_policies:
            companion_checkpoint_path = _checkpoint_path_for_policy(train_outputs, companion_policy)
            if not companion_checkpoint_path or not Path(companion_checkpoint_path).exists():
                raise ValueError(
                    f"Companion checkpoint policy `{companion_policy}` did not resolve to an existing checkpoint for {run_id}."
                )
            analysis_subdir = "analysis_theory" if companion_policy == "theory" else f"analysis_{companion_policy}"
            report_subdir = "report_theory" if companion_policy == "theory" else f"report_{companion_policy}"
            companion_analysis_provenance = build_stage_provenance(
                stage_name=analysis_subdir,
                input_files=(
                    final_test_protocol_config,
                    model_config,
                    final_manifest_outputs["manifest_path"],
                    companion_checkpoint_path,
                ),
                input_values={
                    "run_id": run_id,
                    "checkpoint_policy": companion_policy,
                },
            )
            companion_inference_outputs = (
                _load_existing_inference_outputs(
                    run_root,
                    run_id,
                    expected_provenance=companion_analysis_provenance,
                    resume_policy=resume_policy,
                    protocol_config_path=final_test_protocol_config,
                    model_config_path=model_config,
                    checkpoint_path=companion_checkpoint_path,
                    subdir_name=analysis_subdir,
                )
                if training_resumed
                else None
            )
            companion_inference_resumed = companion_inference_outputs is not None
            if companion_inference_outputs is None:
                companion_inference_outputs = export_plan_a_inference_bundle(
                    protocol_config_path=final_test_protocol_config,
                    model_config_path=model_config,
                    manifest_path=final_manifest_outputs["manifest_path"],
                    output_dir=run_root / analysis_subdir,
                    run_id=run_id,
                    checkpoint_path=companion_checkpoint_path,
                    checkpoint_selection_summary_path=train_outputs.get("checkpoint_selection_summary_path"),
                    model_family=str(train_outputs.get("model_family", model_family)),
                )
                write_stage_provenance(run_root / analysis_subdir, companion_analysis_provenance)
            companion_report_provenance = build_stage_provenance(
                stage_name=report_subdir,
                input_files=(
                    final_test_protocol_config,
                    run_eval_config,
                    analysis_config,
                    companion_inference_outputs["analysis_summary_path"],
                    companion_inference_outputs["analysis_path"],
                ),
                input_values={
                    "run_id": run_id,
                    "checkpoint_policy": companion_policy,
                },
            )
            companion_artifact_outputs = (
                _load_existing_artifact_outputs(
                    run_root,
                    run_id,
                    expected_provenance=companion_report_provenance,
                    resume_policy=resume_policy,
                    protocol_config_path=final_test_protocol_config,
                    eval_config_path=run_eval_config,
                    analysis_config_path=analysis_config,
                    subdir_name=report_subdir,
                )
                if companion_inference_resumed
                else None
            )
            if companion_artifact_outputs is None:
                companion_artifact_outputs = generate_plan_a_artifact_bundle(
                    analysis_path=companion_inference_outputs["analysis_path"],
                    analysis_summary_path=companion_inference_outputs["analysis_summary_path"],
                    protocol_config_path=final_test_protocol_config,
                    eval_config_path=run_eval_config,
                    analysis_config_path=analysis_config,
                    output_dir=run_root / report_subdir,
                )
                write_stage_provenance(run_root / report_subdir, companion_report_provenance)
            policy_outputs[companion_policy] = {
                "analysis": companion_inference_outputs,
                "report": companion_artifact_outputs,
            }
        run_outputs.append(
            {
                "run_id": run_id,
                "seed": seed,
                "model_family": str(train_outputs.get("model_family", model_family)),
                "output_dir": str(run_root),
                "shared_validation_manifest_path": validation_manifest_outputs["manifest_path"],
                "shared_eval_manifest_path": final_manifest_outputs["manifest_path"],
                "primary_checkpoint_policy": primary_checkpoint_policy,
                "companion_checkpoint_policies": companion_checkpoint_policies,
                "train": train_outputs,
                "analysis": inference_outputs,
                "report": artifact_outputs,
                "policy_outputs": policy_outputs,
            }
        )
        _emit_progress(
            progress_callback,
            f"[study] seed_complete run_id={run_id} best_checkpoint={primary_checkpoint_path}",
        )

    study_paths_path = _write_json(
        {
            "study_id": study_id,
            "model_family": model_family,
            "study_root": str(study_root),
            "study_config_path": str(study_config_path),
            "seeds": seeds,
            "shared_validation_manifest_path": validation_manifest_outputs["manifest_path"],
            "shared_validation_manifest_summary_path": validation_manifest_outputs["manifest_summary_path"],
            "shared_eval_manifest_path": final_manifest_outputs["manifest_path"],
            "shared_eval_manifest_summary_path": final_manifest_outputs["manifest_summary_path"],
            "primary_checkpoint_policy": primary_checkpoint_policy,
            "companion_checkpoint_policies": companion_checkpoint_policies,
            "runs": run_outputs,
        },
        study_root / "study_paths.json",
    )

    aggregate_outputs: dict[str, str] = {}
    if aggregate_after_run:
        aggregate_outputs = aggregate_plan_a_study_bundle(
            study_root=study_root,
            study_config_path=study_config_path,
            progress_callback=progress_callback,
        )

    return {
        "study_id": study_id,
        "output_dir": str(study_root),
        "study_paths_path": str(study_paths_path),
        "shared_validation_manifest_path": validation_manifest_outputs["manifest_path"],
        "shared_eval_manifest_path": final_manifest_outputs["manifest_path"],
        **aggregate_outputs,
    }
