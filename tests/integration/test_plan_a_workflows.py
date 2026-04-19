from __future__ import annotations

import json
from pathlib import Path

import yaml

from frcnet.workflows.plan_a import train_plan_a_model, write_plan_a_experiment_bundle
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


def test_train_plan_a_model_writes_records_and_checkpoints(tmp_path: Path, monkeypatch):
    protocol_config_path = _write_yaml(
        tmp_path / "protocol_train.yaml",
        "protocol",
        _build_protocol("train", cifar_train=True, svhn_split="train"),
    )
    model_config_path = _write_yaml(
        tmp_path / "model.yaml",
        "model",
        {
            "name": "frcnet_resnet18_base",
            "backbone": "resnet18",
            "num_classes": 10,
            "resolution_temperature": 1.0,
            "content_temperature": 1.0,
        },
    )
    train_config_path = _write_yaml(
        tmp_path / "train.yaml",
        "train",
        {
            "output_root": str(tmp_path / "runs"),
            "training": {"epochs": 2, "seed": 7},
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
        },
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
        {
            "name": "frcnet_resnet18_base",
            "backbone": "resnet18",
            "num_classes": 10,
            "resolution_temperature": 1.0,
            "content_temperature": 1.0,
        },
    )
    train_config_path = _write_yaml(
        tmp_path / "train.yaml",
        "train",
        {
            "output_root": str(tmp_path / "runs"),
            "training": {"epochs": 1, "seed": 7},
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
        },
    )
    eval_config_path = _write_yaml(
        tmp_path / "eval.yaml",
        "eval",
        {
            "benchmark_name": "plan_a_matched_ambiguous_vs_ood",
            "positive_cohort": "ambiguous_id",
            "negative_cohort": "ood",
            "primary_pair": "resolution_ratio__content_entropy",
            "primary_scalar": "completion_score_beta_0_1",
            "test_size": 0.3,
            "random_state": 7,
        },
    )
    analysis_config_path = _write_yaml(
        tmp_path / "analysis.yaml",
        "analysis",
        {
            "figure_dpi": 120,
            "geometry_scatter_name": "scatter.png",
            "geometry_hexbin_name": "hexbin.png",
            "cohort_occupancy_name": "occupancy.png",
            "cohort_summary_table_name": "summary.csv",
            "matched_table_name": "matched.csv",
        },
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
    assert Path(outputs["artifact_index_path"]).exists()
    assert Path(outputs["experiment_record_path"]).exists()
    assert Path(outputs["bundle_path"]).exists()
