"""Data contracts and manifest helpers for FRCNet."""

from frcnet.data.contracts import ALLOWED_COHORT_NAMES, BatchInput, validate_batch_input
from frcnet.data.manifest import SampleManifestRecord, read_manifest_jsonl, write_manifest_jsonl
from frcnet.data.plan_a import (
    CIFAR10_CLASS_NAMES,
    ManifestBackedVisionDataset,
    build_plan_a_manifest,
    collate_manifest_samples,
    load_plan_a_source_datasets,
    summarize_manifest,
    write_manifest_summary,
)

__all__ = [
    "ALLOWED_COHORT_NAMES",
    "BatchInput",
    "CIFAR10_CLASS_NAMES",
    "ManifestBackedVisionDataset",
    "SampleManifestRecord",
    "build_plan_a_manifest",
    "collate_manifest_samples",
    "load_plan_a_source_datasets",
    "read_manifest_jsonl",
    "summarize_manifest",
    "validate_batch_input",
    "write_manifest_jsonl",
    "write_manifest_summary",
]
