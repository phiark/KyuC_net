# Plan A Next v0.5 Evidence Repair Protocol

- document_id: arch_plan_a_next_v0_5_evidence_repair_protocol
- status: review
- owner: frcnet_project
- last_updated: 2026-04-26

## 1. Goal

`plan_a_next_v0_5_evidence_repair` repairs the evidence chain after V4 strict-freeze. It does not change FRCNet architecture. Its release claim is limited to clean source partitioning, seen/unseen OOD separation, frozen matched benchmarks, and multi-seed aggregate reporting.

## 2. Data Protocol

- train: CIFAR-10 train plus SVHN train; SVHN is `seen_unknown_source`.
- validation: CIFAR-10 test per-class source indices `[0, 200)` and SVHN test source indices `[0, 1000)`.
- final test: CIFAR-10 test per-class source indices `[200, 1000)`, SVHN test source indices `[1000, 3000)`, and CIFAR-100 test source indices `[0, 1000)`.
- CIFAR-100 is final-only and must not appear in train or validation manifests.
- source overlap is audited by `(source_dataset_name, source_dataset_split, source_sample_index)`, not by `sample_id` alone.

## 3. Benchmark Protocol

Each seed emits frozen matched manifests for:

- `ambiguous_vs_seen_ood_svhn`
- `ambiguous_vs_unseen_ood_cifar100`
- `ambiguous_vs_all_ood`

All three use the same FRCNet final analysis records and same-backbone softmax CE reference scores. Primary pair remains `(resolution_ratio, state_weighted_content_entropy)`, with `(resolution_ratio, state_content_entropy)` kept as secondary.

## 4. Config Chain

- train protocol: `configs/protocol/plan_a_next_v0_5_train.yaml`
- validation protocol: `configs/protocol/plan_a_next_v0_5_validation.yaml`
- final test protocol: `configs/protocol/plan_a_next_v0_5_test.yaml`
- eval config: `configs/eval/plan_a_next_v0_5_evidence_repair.yaml`
- reference config: `configs/reference/plan_a_next_v0_5_softmax_ce_reference.yaml`
- study config: `configs/study/plan_a_next_v0_5_evidence_repair.yaml`

## 5. Release Gate

V0.5 can be called evidence-repaired only if:

- train/final and validation/final source fingerprint overlap are both zero.
- final report contains non-empty frozen manifest paths for seen, unseen, and all OOD slices.
- three seeds are aggregated.
- any failure in AUROC, pair-vs-scalar delta, easy/hard top-1, or ambiguous hit is explicitly recorded as partial or negative evidence.
