"""FRCNet public interfaces."""

from frcnet.data import (
    ALLOWED_COHORT_NAMES,
    BatchInput,
    ManifestBackedVisionDataset,
    SampleManifestRecord,
    build_plan_a_manifest,
    collate_manifest_samples,
    load_plan_a_source_datasets,
    read_manifest_jsonl,
    validate_batch_input,
    write_manifest_jsonl,
)
from frcnet.evaluation import (
    MatchedBenchmarkSummary,
    SampleAnalysisRecord,
    Top1PropositionRecord,
    build_top1_proposition_records,
    read_sample_analysis_records,
    run_inference_export,
    summarize_matched_ambiguous_vs_ood,
    write_matched_benchmark_summary,
    write_sample_analysis_records,
    write_top1_proposition_records,
)
from frcnet.models import FRCNetModel, ModelOutput
from frcnet.training import LossBreakdown, compute_total_loss, run_train_step
from frcnet.utils import RuntimeSpec, completion_score, content_entropy, move_batch_to_device, resolve_runtime

__all__ = [
    "__version__",
    "ALLOWED_COHORT_NAMES",
    "BatchInput",
    "FRCNetModel",
    "LossBreakdown",
    "ManifestBackedVisionDataset",
    "MatchedBenchmarkSummary",
    "ModelOutput",
    "RuntimeSpec",
    "SampleAnalysisRecord",
    "SampleManifestRecord",
    "Top1PropositionRecord",
    "build_plan_a_manifest",
    "build_top1_proposition_records",
    "collate_manifest_samples",
    "completion_score",
    "compute_total_loss",
    "content_entropy",
    "load_plan_a_source_datasets",
    "move_batch_to_device",
    "read_manifest_jsonl",
    "read_sample_analysis_records",
    "resolve_runtime",
    "run_inference_export",
    "run_train_step",
    "summarize_matched_ambiguous_vs_ood",
    "validate_batch_input",
    "write_manifest_jsonl",
    "write_matched_benchmark_summary",
    "write_sample_analysis_records",
    "write_top1_proposition_records",
]

__version__ = "0.1.0"
