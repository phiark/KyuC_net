# FRCNet next-v0.2

FRCNet is a document-driven research codebase for an explicit-unknown model with a native resolution/content factorization:

```text
p_k = r * c_k
u = 1 - r
```

This branch advances the `KYUC-V01-001 / plan_a_next_v0_1` semantic repair into `plan_a_next_v0_2`: a document-driven, data-semantics-clean, multi-seed research baseline. Historical `v0.3` debug checkpoint policy branches are not active in this branch.

## Active Scope

- `state layer`: `resolution_ratio`, `content_distribution`, `unknown_mass`, `state_content_entropy`, `state_weighted_content_entropy`, `state_entropy`
- `proposition layer`: proposition-specific `pT / pF / pU / tau_A`
- `completion layer`: `q_beta` readouts bound to a declared proposition view
- `matched benchmark`: label-free feature whitelist plus optional frozen matched manifest
- `softmax reference`: minimal same-backbone CE reference used only to build external reference scores
- `v0.2 study`: 30 epoch seeds `[7, 17, 27]` with validation/final-test manifest separation

## Not In This Branch

- full baseline matrix
- decision benchmark
- Transformer or teacher-distillation variants
- generated experiment artifacts or checkpoints

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Run tests:

```bash
python -m pytest -q
```

## Main Commands

Prepare CIFAR-10 / SVHN data:

```bash
python scripts/prepare_plan_a_data.py
```

Train the native FRCNet model:

```bash
python scripts/train_plan_a.py \
  --protocol-config configs/protocol/plan_a_next_v0_2_train.yaml \
  --model-config configs/model/frcnet_resnet18_base.yaml \
  --train-config configs/train/plan_a_next_v0_2_curriculum.yaml
```

Run a single end-to-end bundle:

```bash
python scripts/run_plan_a_experiment.py
```

Build an analysis manifest only:

```bash
python scripts/build_plan_a_manifest.py \
  --protocol-config configs/protocol/plan_a_next_v0_2_test.yaml
```

Generate report artifacts from an existing analysis export:

```bash
python scripts/generate_plan_a_artifacts.py \
  --analysis-path artifacts/reports/generated/RUN-LOCAL/sample_analysis_records.csv \
  --analysis-summary-path artifacts/reports/generated/RUN-LOCAL/analysis_summary.json \
  --protocol-config configs/protocol/plan_a_next_v0_2_test.yaml \
  --analysis-config configs/analysis/plan_a_next_v0_2_artifacts.yaml \
  --eval-config configs/eval/plan_a_next_v0_2_matched_manifest.yaml \
  --output-dir artifacts/reports/generated/RUN-LOCAL/report
```

## Current Documents

- [Document Index](docs/index.md)
- [Architecture Description](docs/architecture/architecture_description.md)
- [Plan A next-v0.1 Protocol](docs/architecture/plan_a_next_v0_1_protocol.md)
- [Plan A next-v0.2 Protocol](docs/architecture/plan_a_next_v0_2_protocol.md)
- [Plan A Paper Linkage](docs/architecture/plan_a_paper_linkage.md)
- [System Requirements](docs/requirements/system_requirements_specification.md)
- [Verification Plan](docs/verification/verification_and_validation_plan.md)
- [ADR-0006 next-v0.1 Semantic Repair](records/decisions/adr_0006_next_v0_1_semantic_repair.md)
- [ADR-0006 plan_a_next_v0_2 Data Semantic Baseline](records/decisions/adr_0006_plan_a_next_v0_2_data_semantic_baseline.md)
