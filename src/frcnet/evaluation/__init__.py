"""Evaluation components for FRCNet."""

from frcnet.evaluation.inference import (
    build_sample_analysis_records,
    build_top1_proposition_records,
    run_inference_export,
)
from frcnet.evaluation.matched_benchmark import (
    DEFAULT_COMPLETION_SCAN_SCALARS,
    DEFAULT_WEIGHTED_PAIR_NAME,
    MatchedBenchmarkSummary,
    ScalarBenchmarkSummary,
    ScalarRocCurve,
    build_scalar_roc_curve,
    summarize_matched_ambiguous_vs_ood,
    summarize_scalar_benchmarks,
    write_matched_benchmark_summary,
    write_scalar_benchmark_summaries,
)
from frcnet.evaluation.records import (
    AnalysisExportSummary,
    DEFAULT_MODEL_FAMILY,
    SampleAnalysisRecord,
    Top1PropositionRecord,
    read_analysis_export_summary,
    read_sample_analysis_records,
    read_top1_proposition_records,
    write_analysis_export_summary,
    write_sample_analysis_records,
    write_top1_proposition_records,
)

__all__ = [
    "AnalysisExportSummary",
    "DEFAULT_MODEL_FAMILY",
    "DEFAULT_COMPLETION_SCAN_SCALARS",
    "DEFAULT_WEIGHTED_PAIR_NAME",
    "MatchedBenchmarkSummary",
    "SampleAnalysisRecord",
    "ScalarBenchmarkSummary",
    "ScalarRocCurve",
    "Top1PropositionRecord",
    "build_scalar_roc_curve",
    "build_sample_analysis_records",
    "build_top1_proposition_records",
    "read_analysis_export_summary",
    "read_sample_analysis_records",
    "read_top1_proposition_records",
    "run_inference_export",
    "summarize_matched_ambiguous_vs_ood",
    "summarize_scalar_benchmarks",
    "write_matched_benchmark_summary",
    "write_scalar_benchmark_summaries",
    "write_analysis_export_summary",
    "write_sample_analysis_records",
    "write_top1_proposition_records",
]
