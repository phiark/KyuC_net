from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
import yaml

from frcnet.data import build_plan_a_manifest, write_manifest_jsonl
from frcnet.models import FRCNetModel
from frcnet.workflows.plan_a import (
    export_plan_a_inference_bundle,
    generate_plan_a_artifact_bundle,
    train_plan_a_model,
    write_plan_a_experiment_bundle,
)
from tests.conftest import FakeLabelsDataset, FakeTargetsDataset, build_protocol_config


def _write_yaml(path: Path, section_name: str, payload: dict) -> Path:
    path.write_text(yaml.safe_dump({section_name: payload}, sort_keys=False), encoding="utf-8")
    return path


def _build_protocol(split_name: str, *, cifar_train: bool, svhn_split: str) -> dict:
    protocol = build_protocol_config()
    protocol["split_name"] = split_name
    protocol["protocol_id"] = f"plan_a_v1_{split_name}"
    protocol["datasets"]["cifar10"]["train"] = cifar_train
    protocol["datasets"]["svhn"]["split"] = svhn_split
    protocol["datasets"]["cifar10"]["download"] = False
    protocol["datasets"]["svhn"]["download"] = False
    return protocol


def _fake_source_datasets():
    cifar_labels: list[int] = []
    for class_label in range(10):
        cifar_labels.extend([class_label] * 6)
    svhn_labels = [index % 10 for index in range(40)]
    return {
        "cifar10": FakeTargetsDataset(cifar_labels),
        "svhn": FakeLabelsDataset(svhn_labels),
    }


def _build_model_payload() -> dict:
    return {
        "name": "frcnet_resnet18_base",
        "backbone": "resnet18",
        "num_classes": 10,
        "resolution_temperature": 1.0,
        "content_temperature": 1.0,
    }


def _build_train_payload(tmp_path: Path, *, epochs: int = 1) -> dict:
    return {
        "output_root": str(tmp_path / "runs"),
        "training": {"epochs": epochs, "seed": 7},
        "runtime": {"backend": "cpu", "dtype": "float32", "amp_enabled": False, "pin_memory": False},
        "dataloader": {
            "batch_size": 4,
            "shuffle": False,
            "drop_last": True,
            "num_workers": 0,
            "persistent_workers": False,
            "pin_memory": False,
        },
        "optimizer": {"name": "sgd", "lr": 0.01, "momentum": 0.0, "weight_decay": 0.0},
        "loss": {
            "weight_id": 1.0,
            "weight_unknown": 1.0,
            "weight_ambiguous": 1.0,
            "ambiguous_resolution_target": 0.8,
            "ambiguous_resolution_weight": 1.0,
        },
        "checkpointing": {"save_every_epochs": 1},
    }


def _build_eval_payload(*, primary_scalar: str = "completion_score_beta_0_1", random_state: int = 7) -> dict:
    return {
        "benchmark_name": "plan_a_matched_ambiguous_vs_ood",
        "positive_cohort": "ambiguous_id",
        "negative_cohort": "ood",
        "primary_pair": "resolution_ratio__content_entropy",
        "primary_scalar": primary_scalar,
        "test_size": 0.3,
        "random_state": random_state,
    }


def _build_analysis_payload() -> dict:
    return {
        "figure_dpi": 120,
        "geometry_scatter_name": "scatter.png",
        "geometry_hexbin_name": "hexbin.png",
        "cohort_occupancy_name": "occupancy.png",
        "cohort_summary_table_name": "summary.csv",
        "matched_table_name": "matched.csv",
    }


def _write_checkpoint(path: Path) -> Path:
    model = FRCNetModel(num_classes=10)
    torch.save({"model_state_dict": model.state_dict()}, path)
    return path


def _write_manifest(path: Path, protocol: dict) -> Path:
    manifest_records = build_plan_a_manifest(protocol, _fake_source_datasets())
    return write_manifest_jsonl(manifest_records, path)


def test_train_plan_a_model_writes_records_and_checkpoints(tmp_path: Path, monkeypatch):
    protocol_config_path = _write_yaml(
        tmp_path / "protocol_train.yaml",
        "protocol",
        _build_protocol("train", cifar_train=True, svhn_split="train"),
    )
    model_config_path = _write_yaml(
        tmp_path / "model.yaml",
        "model",
        _build_model_payload(),
    )
    train_config_path = _write_yaml(
        tmp_path / "train.yaml",
        "train",
        _build_train_payload(tmp_path, epochs=2),
    )

    monkeypatch.setattr("frcnet.workflows.plan_a.load_plan_a_source_datasets", lambda _: _fake_source_datasets())

    outputs = train_plan_a_model(
        protocol_config_path=protocol_config_path,
        model_config_path=model_config_path,
        train_config_path=train_config_path,
        output_dir=tmp_path / "train_run",
        run_id="RUN-TRAIN-TEST",
    )

    assert Path(outputs["manifest_path"]).exists()
    assert Path(outputs["manifest_summary_path"]).exists()
    assert Path(outputs["history_path"]).exists()
    assert Path(outputs["summary_path"]).exists()
    assert Path(outputs["best_checkpoint_path"]).exists()
    assert Path(outputs["last_checkpoint_path"]).exists()

    summary_payload = json.loads(Path(outputs["summary_path"]).read_text(encoding="utf-8"))
    assert summary_payload["run_id"] == "RUN-TRAIN-TEST"
    assert summary_payload["manifest"]["num_trainable_records"] > 0
    assert len(summary_payload["epochs"]) == 2


def test_write_plan_a_experiment_bundle_creates_end_to_end_outputs(tmp_path: Path, monkeypatch):
    train_protocol_config_path = _write_yaml(
        tmp_path / "protocol_train.yaml",
        "protocol",
        _build_protocol("train", cifar_train=True, svhn_split="train"),
    )
    analysis_protocol_config_path = _write_yaml(
        tmp_path / "protocol_analysis.yaml",
        "protocol",
        _build_protocol("analysis", cifar_train=False, svhn_split="test"),
    )
    model_config_path = _write_yaml(
        tmp_path / "model.yaml",
        "model",
        _build_model_payload(),
    )
    train_config_path = _write_yaml(
        tmp_path / "train.yaml",
        "train",
        _build_train_payload(tmp_path, epochs=1),
    )
    eval_config_path = _write_yaml(
        tmp_path / "eval.yaml",
        "eval",
        _build_eval_payload(),
    )
    analysis_config_path = _write_yaml(
        tmp_path / "analysis.yaml",
        "analysis",
        _build_analysis_payload(),
    )

    monkeypatch.setattr("frcnet.workflows.plan_a.load_plan_a_source_datasets", lambda _: _fake_source_datasets())

    outputs = write_plan_a_experiment_bundle(
        train_protocol_config_path=train_protocol_config_path,
        analysis_protocol_config_path=analysis_protocol_config_path,
        model_config_path=model_config_path,
        train_config_path=train_config_path,
        eval_config_path=eval_config_path,
        analysis_config_path=analysis_config_path,
        output_dir=tmp_path / "experiment_run",
        run_id="RUN-PIPELINE-TEST",
    )

    assert Path(outputs["data_preflight_path"]).exists()
    assert Path(outputs["best_checkpoint_path"]).exists()
    assert Path(outputs["analysis_manifest_path"]).exists()
    assert Path(outputs["analysis_path"]).exists()
    assert Path(outputs["analysis_summary_path"]).exists()
    assert Path(outputs["artifact_index_path"]).exists()
    assert Path(outputs["experiment_record_path"]).exists()
    assert Path(outputs["bundle_path"]).exists()


def test_train_plan_a_model_rejects_duplicate_manifest_sample_ids(tmp_path: Path, monkeypatch):
    protocol = _build_protocol("train", cifar_train=True, svhn_split="train")
    protocol_config_path = _write_yaml(tmp_path / "protocol_train.yaml", "protocol", protocol)
    model_config_path = _write_yaml(tmp_path / "model.yaml", "model", _build_model_payload())
    train_config_path = _write_yaml(tmp_path / "train.yaml", "train", _build_train_payload(tmp_path))
    manifest_path = _write_manifest(tmp_path / "duplicate_manifest.jsonl", protocol)
    manifest_lines = manifest_path.read_text(encoding="utf-8").splitlines()
    manifest_lines[1] = manifest_lines[0]
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    monkeypatch.setattr("frcnet.workflows.plan_a.load_plan_a_source_datasets", lambda _: _fake_source_datasets())

    with pytest.raises(ValueError, match="unique sample_id"):
        train_plan_a_model(
            protocol_config_path=protocol_config_path,
            model_config_path=model_config_path,
            train_config_path=train_config_path,
            output_dir=tmp_path / "train_run",
            manifest_path=manifest_path,
        )


def test_export_plan_a_inference_requires_checkpoint_by_default(tmp_path: Path, monkeypatch):
    protocol = _build_protocol("analysis", cifar_train=False, svhn_split="test")
    protocol_config_path = _write_yaml(tmp_path / "protocol_analysis.yaml", "protocol", protocol)
    model_config_path = _write_yaml(tmp_path / "model.yaml", "model", _build_model_payload())
    manifest_path = _write_manifest(tmp_path / "manifest.jsonl", protocol)

    monkeypatch.setattr("frcnet.workflows.plan_a.load_plan_a_source_datasets", lambda _: _fake_source_datasets())

    with pytest.raises(ValueError, match="checkpoint_path"):
        export_plan_a_inference_bundle(
            protocol_config_path=protocol_config_path,
            model_config_path=model_config_path,
            manifest_path=manifest_path,
            output_dir=tmp_path / "analysis_run",
            run_id="RUN-NO-CKPT",
        )


def test_export_plan_a_inference_allows_missing_checkpoint_with_summary(tmp_path: Path, monkeypatch):
    protocol = _build_protocol("analysis", cifar_train=False, svhn_split="test")
    protocol_config_path = _write_yaml(tmp_path / "protocol_analysis.yaml", "protocol", protocol)
    model_config_path = _write_yaml(tmp_path / "model.yaml", "model", _build_model_payload())
    manifest_path = _write_manifest(tmp_path / "manifest.jsonl", protocol)

    monkeypatch.setattr("frcnet.workflows.plan_a.load_plan_a_source_datasets", lambda _: _fake_source_datasets())

    outputs = export_plan_a_inference_bundle(
        protocol_config_path=protocol_config_path,
        model_config_path=model_config_path,
        manifest_path=manifest_path,
        output_dir=tmp_path / "analysis_run",
        run_id="RUN-ALLOW-NO-CKPT",
        allow_missing_checkpoint=True,
    )

    summary_payload = json.loads(Path(outputs["analysis_summary_path"]).read_text(encoding="utf-8"))
    assert summary_payload["integrity_overrides"] == ["missing_checkpoint"]
    assert summary_payload["sidecar_resolution_mode"] == "analysis_summary"


def test_generate_plan_a_artifact_bundle_uses_analysis_summary_explicit(tmp_path: Path, monkeypatch):
    protocol = _build_protocol("analysis", cifar_train=False, svhn_split="test")
    protocol_config_path = _write_yaml(tmp_path / "protocol_analysis.yaml", "protocol", protocol)
    model_config_path = _write_yaml(tmp_path / "model.yaml", "model", _build_model_payload())
    eval_config_path = _write_yaml(tmp_path / "eval.yaml", "eval", _build_eval_payload(primary_scalar="completion_score_beta_0_5"))
    analysis_config_path = _write_yaml(tmp_path / "analysis.yaml", "analysis", _build_analysis_payload())
    manifest_path = _write_manifest(tmp_path / "manifest.jsonl", protocol)
    checkpoint_path = _write_checkpoint(tmp_path / "checkpoint_best.pt")

    monkeypatch.setattr("frcnet.workflows.plan_a.load_plan_a_source_datasets", lambda _: _fake_source_datasets())

    inference_outputs = export_plan_a_inference_bundle(
        protocol_config_path=protocol_config_path,
        model_config_path=model_config_path,
        manifest_path=manifest_path,
        output_dir=tmp_path / "analysis_run",
        run_id="RUN-EXPLICIT-SUMMARY",
        checkpoint_path=checkpoint_path,
    )
    report_outputs = generate_plan_a_artifact_bundle(
        analysis_path=inference_outputs["analysis_path"],
        analysis_summary_path=inference_outputs["analysis_summary_path"],
        protocol_config_path=protocol_config_path,
        eval_config_path=eval_config_path,
        analysis_config_path=analysis_config_path,
        output_dir=tmp_path / "report_run",
    )

    record_text = Path(report_outputs["experiment_record_path"]).read_text(encoding="utf-8")
    matched_rows = list(csv.DictReader(Path(report_outputs["matched_ambiguous_vs_ood_table"]).open("r", encoding="utf-8")))

    assert "analysis_summary_explicit" in record_text
    assert matched_rows[0]["scalar_name"] == "completion_score_beta_0_5"


def test_generate_plan_a_artifact_bundle_auto_and_legacy_modes(tmp_path: Path, monkeypatch):
    protocol = _build_protocol("analysis", cifar_train=False, svhn_split="test")
    protocol_config_path = _write_yaml(tmp_path / "protocol_analysis.yaml", "protocol", protocol)
    model_config_path = _write_yaml(tmp_path / "model.yaml", "model", _build_model_payload())
    eval_config_path = _write_yaml(tmp_path / "eval.yaml", "eval", _build_eval_payload())
    analysis_config_path = _write_yaml(tmp_path / "analysis.yaml", "analysis", _build_analysis_payload())
    manifest_path = _write_manifest(tmp_path / "manifest.jsonl", protocol)
    checkpoint_path = _write_checkpoint(tmp_path / "checkpoint_best.pt")

    monkeypatch.setattr("frcnet.workflows.plan_a.load_plan_a_source_datasets", lambda _: _fake_source_datasets())

    inference_outputs = export_plan_a_inference_bundle(
        protocol_config_path=protocol_config_path,
        model_config_path=model_config_path,
        manifest_path=manifest_path,
        output_dir=tmp_path / "analysis_run",
        run_id="RUN-AUTO-SUMMARY",
        checkpoint_path=checkpoint_path,
    )
    auto_outputs = generate_plan_a_artifact_bundle(
        analysis_path=inference_outputs["analysis_path"],
        protocol_config_path=protocol_config_path,
        eval_config_path=eval_config_path,
        analysis_config_path=analysis_config_path,
        output_dir=tmp_path / "report_auto",
    )
    Path(inference_outputs["analysis_summary_path"]).unlink()
    legacy_outputs = generate_plan_a_artifact_bundle(
        analysis_path=inference_outputs["analysis_path"],
        protocol_config_path=protocol_config_path,
        eval_config_path=eval_config_path,
        analysis_config_path=analysis_config_path,
        output_dir=tmp_path / "report_legacy",
    )

    assert "analysis_summary_auto" in Path(auto_outputs["experiment_record_path"]).read_text(encoding="utf-8")
    assert "legacy_sibling" in Path(legacy_outputs["experiment_record_path"]).read_text(encoding="utf-8")


def test_generate_plan_a_artifact_bundle_rejects_mixed_runs_by_default_and_marks_override(tmp_path: Path, monkeypatch):
    protocol = _build_protocol("analysis", cifar_train=False, svhn_split="test")
    protocol_config_path = _write_yaml(tmp_path / "protocol_analysis.yaml", "protocol", protocol)
    model_config_path = _write_yaml(tmp_path / "model.yaml", "model", _build_model_payload())
    eval_config_path = _write_yaml(tmp_path / "eval.yaml", "eval", _build_eval_payload(random_state=11))
    analysis_config_path = _write_yaml(tmp_path / "analysis.yaml", "analysis", _build_analysis_payload())
    manifest_path = _write_manifest(tmp_path / "manifest.jsonl", protocol)
    checkpoint_path = _write_checkpoint(tmp_path / "checkpoint_best.pt")

    monkeypatch.setattr("frcnet.workflows.plan_a.load_plan_a_source_datasets", lambda _: _fake_source_datasets())

    inference_outputs = export_plan_a_inference_bundle(
        protocol_config_path=protocol_config_path,
        model_config_path=model_config_path,
        manifest_path=manifest_path,
        output_dir=tmp_path / "analysis_run",
        run_id="RUN-MIXED",
        checkpoint_path=checkpoint_path,
    )
    analysis_rows = list(csv.DictReader(Path(inference_outputs["analysis_path"]).open("r", encoding="utf-8")))
    analysis_rows[0]["run_id"] = "RUN-MIXED-A"
    analysis_rows[1]["run_id"] = "RUN-MIXED-B"
    with Path(inference_outputs["analysis_path"]).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(analysis_rows[0].keys()))
        writer.writeheader()
        writer.writerows(analysis_rows)

    with pytest.raises(ValueError, match="analysis_mixed_run_ids"):
        generate_plan_a_artifact_bundle(
            analysis_path=inference_outputs["analysis_path"],
            analysis_summary_path=inference_outputs["analysis_summary_path"],
            protocol_config_path=protocol_config_path,
            eval_config_path=eval_config_path,
            analysis_config_path=analysis_config_path,
            output_dir=tmp_path / "report_fail",
        )

    override_outputs = generate_plan_a_artifact_bundle(
        analysis_path=inference_outputs["analysis_path"],
        analysis_summary_path=inference_outputs["analysis_summary_path"],
        protocol_config_path=protocol_config_path,
        eval_config_path=eval_config_path,
        analysis_config_path=analysis_config_path,
        output_dir=tmp_path / "report_override",
        allow_integrity_override=True,
    )
    override_text = Path(override_outputs["experiment_record_path"]).read_text(encoding="utf-8")

    assert "MULTIPLE" in override_text
    assert "analysis_mixed_run_ids" in override_text
