# FRCNet

FRCNet is a document-driven research codebase for a resolution-content disentangled explicit-unknown network. The project baseline comes from two upstream documents:

- the reset manuscript for explicit unknown / resolution-aware diagnostics
- the FRCNet experiment design note

The canonical model factorization is:

```text
p_k = r * c_k
u = 1 - r
q_beta = p_top1 + beta * u
```

Where:

- `r` is the resolution ratio
- `c` is the class distribution inside the resolved subspace
- `u` is the unknown mass
- `q_beta` is a completion-dependent scalar readout

This repository is initialized so that documents are the source of truth and code follows the approved contracts.

## Runtime Support

FRCNet 0.3 targets a single training code path that runs on:

- Apple Silicon macOS with `MPS`
- Linux with `ROCm`
- Linux with `CUDA`
- `CPU` fallback

The runtime resolver prefers `mps -> rocm -> cuda -> cpu` when `backend=auto`.

Installation notes by platform are documented in [Runtime Environment Matrix](docs/architecture/runtime_environment_matrix.md).

## Development Order

1. Update or add requirements in `docs/requirements/`.
2. Record architectural decisions in `records/decisions/`.
3. Update contracts in `docs/architecture/`.
4. Implement code in `src/frcnet/`.
5. Verify against `docs/verification/`.
6. Archive experiment evidence in `records/experiments/`.

## Primary Documents

- [Document Index](docs/index.md)
- [Naming And Identifier Standard](docs/governance/naming_and_identifier_standard.md)
- [System Requirements Specification](docs/requirements/system_requirements_specification.md)
- [Architecture Description](docs/architecture/architecture_description.md)
- [Plan A Paper Linkage](docs/architecture/plan_a_paper_linkage.md)
- [Plan A v0.3debug Protocol](docs/architecture/plan_a_v0_3debug_protocol.md)
- [Plan A v0.3debug R2 Protocol](docs/architecture/plan_a_v0_3debug_r2_protocol.md)
- [Runtime Environment Matrix](docs/architecture/runtime_environment_matrix.md)
- [Project Structure](docs/architecture/project_structure.md)
- [Verification And Validation Plan](docs/verification/verification_and_validation_plan.md)
- [ADR-0001 Document-Driven Baseline](records/decisions/adr_0001_document_driven_baseline.md)
- [ADR-0002 Plan A Protocol Baseline](records/decisions/adr_0002_plan_a_protocol_baseline.md)
- [ADR-0004 v0.3debug Theory Alignment Repair](records/decisions/adr_0004_v0_3debug_theory_alignment_repair.md)
- [ADR-0005 v0.3debug R2 Benchmark And Geometry Repair](records/decisions/adr_0005_v0_3debug_r2_benchmark_and_geometry_repair.md)

## Repository Layout

```text
.
├── docs/                  # normative project documents
├── records/               # evidential records and decisions
├── src/frcnet/            # implementation package
├── configs/               # structured configuration
├── tests/                 # unit, integration, contract tests
├── scripts/               # repeatable command-line entrypoints
├── artifacts/             # generated outputs, figures, logs, checkpoints
└── notebooks/             # exploratory notebooks only
```

## Current Status

The repository now contains:

- document-driven governance and architecture baselines
- a curriculum-capable FRCNet 0.3 / v0.3debug model and workflow core
- a `v0.3debug_r2` repair line with balanced-primary dual export and resolved-side geometry regularization
- cross-platform runtime resolution for MPS / ROCm / CUDA / CPU
- contract tests and smoke training tests
- a Plan A protocol chain from manifest to proposition-aware analysis record to paper-facing artifacts
- a study workflow for frozen-manifest multi-seed evaluation, aggregate reporting, and theory-vs-balanced checkpoint diagnostics

## Quick Start

Create a virtual environment and install the package:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Run the test suite:

```bash
pytest
```

## Plan A Workflow

The repository now provides three workflow levels:

- single-step training / inference / artifact scripts
- single-run end-to-end experiment bundling
- the default v0.3debug R2 multi-seed study workflow for paper-facing results

Prepare datasets and verify local availability:

```bash
python scripts/prepare_plan_a_data.py
```

Train on the manifest-backed Plan A training protocol:

```bash
python scripts/train_plan_a.py \
  --protocol-config configs/protocol/plan_a_v1_train.yaml \
  --model-config configs/model/frcnet_resnet18_base.yaml \
  --train-config configs/train/plan_a_train_base.yaml
```

Run the full training -> analysis -> artifact bundle in one command:

```bash
python scripts/run_plan_a_experiment.py
```

Run the default v0.3debug R2 study workflow with a frozen evaluation manifest and aggregate report:

```bash
python scripts/run_plan_a_study.py \
  --study-config configs/study/plan_a_v0_3debug_r2_study.yaml
```

Rebuild aggregate outputs from an existing study root:

```bash
python scripts/aggregate_plan_a_study.py \
  --study-config configs/study/plan_a_v0_3debug_r2_study.yaml \
  --study-root artifacts/studies/plan_a_v0_3debug_r2_main
```

In `v0.3debug_r2`, `tau = proposition_truth_ratio` is exported as a proposition diagnostic only. The primary matched benchmark and aggregate AUROC plots compare `pair / weighted_pair / scalar`, while `theory` remains a companion export beside the `balanced` primary line.

Analysis-only export remains available as a separate chain:

```bash
python scripts/build_plan_a_manifest.py --protocol-config configs/protocol/plan_a_v1.yaml
python scripts/run_plan_a_inference.py \
  --protocol-config configs/protocol/plan_a_v1.yaml \
  --model-config configs/model/frcnet_resnet18_base.yaml \
  --manifest-path artifacts/reports/generated/plan_a_v1/plan_a_manifest.jsonl \
  --checkpoint-path artifacts/experiments/RUN-LOCAL/training/checkpoints/checkpoint_best.pt \
  --output-dir artifacts/reports/generated/RUN-LOCAL
python scripts/generate_plan_a_artifacts.py \
  --analysis-path artifacts/reports/generated/RUN-LOCAL/sample_analysis_records.csv \
  --analysis-summary-path artifacts/reports/generated/RUN-LOCAL/analysis_summary.json \
  --protocol-config configs/protocol/plan_a_v1.yaml \
  --analysis-config configs/analysis/plan_a_artifacts.yaml \
  --eval-config configs/eval/plan_a_matched_ambiguous_vs_ood.yaml \
  --output-dir artifacts/reports/generated/RUN-LOCAL
```

Integrity overrides remain available for explicit debug or review workflows only:

```bash
python scripts/run_plan_a_inference.py \
  --protocol-config configs/protocol/plan_a_v1.yaml \
  --model-config configs/model/frcnet_resnet18_base.yaml \
  --manifest-path artifacts/reports/generated/plan_a_v1/plan_a_manifest.jsonl \
  --allow-missing-checkpoint \
  --output-dir artifacts/reports/generated/RUN-DEBUG
python scripts/generate_plan_a_artifacts.py \
  --analysis-path artifacts/reports/generated/RUN-DEBUG/sample_analysis_records.csv \
  --allow-integrity-override \
  --protocol-config configs/protocol/plan_a_v1.yaml \
  --analysis-config configs/analysis/plan_a_artifacts.yaml \
  --eval-config configs/eval/plan_a_matched_ambiguous_vs_ood.yaml \
  --output-dir artifacts/reports/generated/RUN-DEBUG
```

These override paths are recorded in `analysis_summary.json` and `experiment_record.md` and are not the default experiment workflow.
