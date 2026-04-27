# FRCNet

FRCNet is a document-driven research codebase for a resolution-content disentangled explicit-unknown network.

Current archive state:

- package version: `0.6.0`
- project state: **V0.6C clean partial evidence archive**
- primary archive record: [V0.6C Archive Closure Review](records/reviews/2026-04-27_review_v0_6c_archive_closure.md)
- governance status: [Project Archive Status](docs/governance/project_archive_status.md)

The project is archived as a clean partial evidence milestone. It is not a strong final paper result. New model training, architecture changes, or scientific claim expansion should start from a new explicit version plan.

## Model Contract

The canonical FRCNet factorization is:

```text
p_k = r * c_k
u = 1 - r
q_beta = p_top1 + beta * u
```

Where:

- `r` is the resolution ratio.
- `c` is the class distribution inside the resolved subspace.
- `u` is the unknown mass.
- `q_beta` is a completion-dependent scalar readout.

Engineering identifiers intentionally use stable names such as `resolution_ratio`, `content_distribution`, `unknown_mass`, and `completion_score` instead of paper-local single-letter variables.

## Archive Result

V0.6C narrowed the evidence target from unseen CIFAR100 source generalization to CIFAR100 class-holdout near-OOD evidence.

Final aggregate interpretation:

- unseen CIFAR100 held-out classes pair AUROC: `0.6512 +/- 0.0112`
- all-OOD pair AUROC: `0.8019 +/- 0.0076`
- worst-source pair AUROC: `0.6435 +/- 0.0214`
- seen-unseen gap: `0.2178 +/- 0.0020`
- evidence grade: clean partial repair

Integrity checks:

- train/validation source fingerprint overlap: `0`
- train/final source fingerprint overlap: `0`
- validation/final source fingerprint overlap: `0`
- CIFAR100 seen classes: `[0, 50)`
- CIFAR100 held-out classes: `[50, 100)`

Known archived limitations:

- The result is below the strong repair target.
- Pair-vs-scalar margin remains small.
- Hard resolver quality remains limited.
- `balanced` checkpoint policy outperformed the configured primary `near_ood_balanced` policy on unseen CIFAR100.

## Repository Layout

```text
.
├── docs/                  # normative project documents
├── records/               # decisions, reviews, evidence records
├── src/frcnet/            # implementation package
├── configs/               # structured experiment configuration
├── tests/                 # unit, contract, integration tests
├── scripts/               # repeatable CLI entrypoints
├── artifacts/             # generated outputs; ignored by default
└── notebooks/             # exploratory notebooks only
```

Generated checkpoints, study outputs, logs, and caches are ignored by default. Normal commits should contain source, config, docs, tests, and compact records only.

## Primary Documents

- [Document Index](docs/index.md)
- [Project Archive Status](docs/governance/project_archive_status.md)
- [System Requirements Specification](docs/requirements/system_requirements_specification.md)
- [Architecture Description](docs/architecture/architecture_description.md)
- [Plan A Next V0.6C Protocol](docs/architecture/plan_a_next_v0_6c_near_ood_split_repair_protocol.md)
- [Verification And Validation Plan](docs/verification/verification_and_validation_plan.md)
- [ADR-0010 V0.6C Near-OOD Split Repair](records/decisions/adr_0010_plan_a_next_v0_6c_near_ood_split_repair.md)
- [V0.6C Archive Closure Review](records/reviews/2026-04-27_review_v0_6c_archive_closure.md)

Older v0.2, v0.3, v0.4, v0.5, v0.6A, and v0.6B records remain available for traceability, but V0.6C is the archived baseline.

## Environment

The validated local maintenance interpreter is `.venv313/bin/python`.

Run the test suite:

```bash
.venv313/bin/python -m pytest -q
```

For a fresh environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

Runtime support remains available for Apple Silicon `MPS`, Linux `ROCm`, Linux `CUDA`, and CPU fallback. See [Runtime Environment Matrix](docs/architecture/runtime_environment_matrix.md).

## Maintenance Commands

Dry-run checkpoint cleanup:

```bash
.venv313/bin/python scripts/cleanup_checkpoints.py
```

Execute generated checkpoint cleanup:

```bash
.venv313/bin/python scripts/cleanup_checkpoints.py --execute
```

The cleanup command only scans generated-output roots by default and retains `checkpoint_best*` and `checkpoint_last*`.

Verify artifact hygiene:

```bash
git ls-files -ci --exclude-standard
git diff --cached --name-only | rg '^artifacts/'
```

Both commands should produce no output for a normal maintenance commit.

## Historical Workflows

The previous training and study CLIs remain for reproducibility:

```bash
.venv313/bin/python scripts/run_plan_a_study.py \
  --study-config configs/study/plan_a_next_v0_6c_near_ood_cifar100_class_holdout.yaml
```

These commands should not be used to create new claims under the archived V0.6C baseline. New experiments require a new version plan and new records.
