# Plan A Next v0.2 Protocol

- document_id: arch_plan_a_next_v0_2_protocol
- status: baselined
- owner: frcnet_project
- last_updated: 2026-04-24

## 1. Goal

`plan_a_next_v0_2` upgrades `KYUC-V01-001 / plan_a_next_v0_1` from a smoke model into a reproducible research baseline. The goal is clean data semantics and model interpretation, not a final paper claim.

## 2. Canonical Layers

Model state layer:

- `resolution_ratio`
- `unknown_mass`
- `state_content_entropy`
- `state_weighted_content_entropy`
- `state_entropy`

Proposition layer:

- `top1_view`: label-free and allowed in primary benchmark
- `target_view` / `candidate_set_view`: label-aware audit only
- each view exports `pT`, `pF`, `pU`, and `tau`

Completion layer:

- `top1_completion_beta_*` belongs to the top-1 view
- high-beta completion is a policy diagnostic, not a universal confidence score

## 3. Data Protocol

Training uses CIFAR-10 train plus SVHN train, with seeds `[7, 17, 27]` and 30 default epochs. Validation and final test manifests must be separate. Validation selects checkpoints; final test produces the report.

The final test protocol must include:

- seen MixUp ambiguous pairs
- held-out class-pair ambiguous samples
- explicit source-role names for seen-source unknown/OOD
- frozen matched manifest driven by an external reference score

## 4. Benchmark Protocol

Primary matched benchmark:

- task: `ambiguous_id` vs `ood`
- primary pair: `(resolution_ratio, state_content_entropy)`
- weighted pair: `(resolution_ratio, state_weighted_content_entropy)`
- primary scalar: `top1_completion_beta_0_1`
- proposition diagnostics: `proposition_truth_ratio`, `resolution_entropy`, `ternary_entropy`

Primary benchmark features must be label-free. Label-aware target/candidate proposition fields are only audit outputs.

## 5. Release Gate

`plan_a_next_v0_2` can be called a research baseline if final test satisfies:

- pair AUROC >= `0.95`
- pair AUROC exceeds best one-feature scalar by at least `0.02`
- easy-ID top-1 >= `0.60`
- hard-ID top-1 >= `0.45`
- ambiguous candidate hit >= `0.75`

If a run misses a gate, it may still be recorded as partial or negative evidence.

## 6. Config Chain

- train protocol: `configs/protocol/plan_a_next_v0_2_train.yaml`
- validation protocol: `configs/protocol/plan_a_next_v0_2_validation.yaml`
- final test protocol: `configs/protocol/plan_a_next_v0_2_test.yaml`
- train config: `configs/train/plan_a_next_v0_2_curriculum.yaml`
- eval config: `configs/eval/plan_a_next_v0_2_matched_manifest.yaml`
- analysis config: `configs/analysis/plan_a_next_v0_2_artifacts.yaml`
- study config: `configs/study/plan_a_next_v0_2_study.yaml`
