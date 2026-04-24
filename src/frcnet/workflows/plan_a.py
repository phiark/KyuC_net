"""Compatibility facade for active Plan A workflows."""

from frcnet.workflows.plan_a_experiment import write_plan_a_experiment_bundle
from frcnet.workflows.plan_a_inference import export_plan_a_inference_bundle
from frcnet.workflows.plan_a_reporting import generate_plan_a_artifact_bundle
from frcnet.workflows.plan_a_training import (
    build_plan_a_manifest_bundle,
    prepare_plan_a_datasets,
    train_plan_a_model,
)
from frcnet.workflows.workflow_io import timestamp_run_id

__all__ = [
    "build_plan_a_manifest_bundle",
    "export_plan_a_inference_bundle",
    "generate_plan_a_artifact_bundle",
    "prepare_plan_a_datasets",
    "timestamp_run_id",
    "train_plan_a_model",
    "write_plan_a_experiment_bundle",
]
