from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any, Callable, Iterable, Mapping

import yaml

from frcnet.evaluation import (
    DEFAULT_MODEL_FAMILY,
    read_analysis_export_summary,
)

TRAINABLE_COHORT_NAMES = frozenset({"easy_id", "hard_id", "ambiguous_id", "unknown_supervision"})


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
    checkpoint_selection_summary_path: str | None
    model_family: str
    summary_run_id: str | None
    summary_protocol_id: str | None
    sidecar_resolution_mode: str
    inherited_integrity_overrides: tuple[str, ...]


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
    if candidate.exists():
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
                    checkpoint_selection_summary_path=None,
                    model_family=DEFAULT_MODEL_FAMILY,
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
                checkpoint_selection_summary_path=None,
                model_family=DEFAULT_MODEL_FAMILY,
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
            checkpoint_selection_summary_path=(
                None
                if summary.checkpoint_selection_summary_path is None
                else str(_resolve_reference_path(summary.checkpoint_selection_summary_path, summary_path.parent))
            ),
            model_family=summary.model_family,
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
            "primary_pair": "resolution_ratio__state_content_entropy",
            "weighted_pair": "resolution_ratio__state_weighted_content_entropy",
            "primary_scalar": "top1_completion_beta_0_1",
            "tau_scalar_name": "proposition_truth_ratio",
            "completion_scan_scalars": (
                "top1_completion_beta_0_1",
                "top1_completion_beta_0_25",
                "top1_completion_beta_0_5",
                "top1_completion_beta_0_75",
            ),
            "matched_manifest_path": "",
            "emit_proposition_diagnostics": True,
            "test_size": 0.3,
            "random_state": 7,
        }

    eval_config = _load_yaml_section(eval_config_path, "eval")
    completion_scan_scalars = eval_config.get(
        "completion_scan_scalars",
        (
            "top1_completion_beta_0_1",
            "top1_completion_beta_0_25",
            "top1_completion_beta_0_5",
            "top1_completion_beta_0_75",
        ),
    )
    return {
        "positive_cohort": str(eval_config.get("positive_cohort", "ambiguous_id")),
        "negative_cohort": str(eval_config.get("negative_cohort", "ood")),
        "primary_pair": str(eval_config.get("primary_pair", "resolution_ratio__state_content_entropy")),
        "weighted_pair": str(
            eval_config.get("weighted_pair", "resolution_ratio__state_weighted_content_entropy")
        ),
        "primary_scalar": str(eval_config.get("primary_scalar", "top1_completion_beta_0_1")),
        "tau_scalar_name": str(eval_config.get("tau_scalar_name", "proposition_truth_ratio")),
        "completion_scan_scalars": tuple(str(value) for value in completion_scan_scalars),
        "matched_manifest_path": str(eval_config.get("matched_manifest_path", "")),
        "emit_proposition_diagnostics": bool(eval_config.get("emit_proposition_diagnostics", True)),
        "test_size": float(eval_config.get("test_size", 0.3)),
        "random_state": int(eval_config.get("random_state", 7)),
    }
