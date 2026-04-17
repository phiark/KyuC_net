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

FRCNet 0.1 targets a single training code path that runs on:

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
- [Runtime Environment Matrix](docs/architecture/runtime_environment_matrix.md)
- [Project Structure](docs/architecture/project_structure.md)
- [Verification And Validation Plan](docs/verification/verification_and_validation_plan.md)
- [ADR-0001 Document-Driven Baseline](records/decisions/adr_0001_document_driven_baseline.md)

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
- a minimal FRCNet 0.1 model core
- cross-platform runtime resolution for MPS / ROCm / CUDA / CPU
- contract tests and smoke training tests

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
