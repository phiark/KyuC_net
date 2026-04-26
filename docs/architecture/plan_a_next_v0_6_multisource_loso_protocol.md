# Plan A Next V0.6 Multisource LOSO Protocol

- document_id: arch_plan_a_next_v0_6_multisource_loso_protocol
- status: draft
- owner: frcnet_project
- last_updated: 2026-04-26

## Purpose

V0.6 tests whether FRCNet can learn source-invariant unknown structure. V0.6-A is intentionally protocol-only: it changes data sources and sampling, not the main FRCNet architecture.

Primary question:

Can multi-source unknown supervision lift CIFAR-100 unseen OOD AUROC from the V0.5 weak baseline while preserving ID and ambiguous behavior?

## Study

- study config: `configs/study/plan_a_next_v0_6_multisource_loso_cifar100_holdout.yaml`
- train protocol: `configs/protocol/plan_a_next_v0_6_train.yaml`
- validation protocol: `configs/protocol/plan_a_next_v0_6_validation.yaml`
- final protocol: `configs/protocol/plan_a_next_v0_6_test_cifar100_holdout.yaml`
- train config: `configs/train/plan_a_next_v0_6_source_balanced_curriculum.yaml`
- eval config: `configs/eval/plan_a_next_v0_6_multisource_loso.yaml`
- seeds: `[7, 17, 27]`

## Data Contract

Training:

- ID: CIFAR-10 train
- unknown supervision: SVHN train, DTD train, LSUN-resize train, Gaussian noise
- CIFAR-100: absent

Validation:

- ID: CIFAR-10 test partition `[0, 200)` per class
- seen-source OOD: SVHN, DTD, LSUN-resize, Gaussian noise validation partitions
- CIFAR-100: absent

Final:

- ID: CIFAR-10 test partition `[200, 1000)` per class
- seen-source OOD slices: SVHN, DTD, LSUN-resize, Gaussian noise held-out partitions
- unseen-source OOD slice: CIFAR-100 test
- primary LOSO claim: CIFAR-100 was not used in train or validation

Every manifest record carries:

- `source_dataset_name`
- `source_dataset_split`
- `source_role`
- `source_partition_name`
- `source_sample_indices_json`
- `source_domain_name`
- `source_domain_label`
- `augmentation_recipe`

## Sampling Contract

Main and stabilize phases use source-balanced batches:

- 25% easy/hard ID
- 25% ambiguous ID
- 50% unknown supervision

The unknown half is balanced across:

- SVHN
- DTD
- LSUN-resize
- Gaussian noise

Warmup keeps ordinary sampling because it only uses easy ID and unknown supervision.

## Evaluation Contract

Frozen matched reports use the same softmax CE reference mechanism as V0.5 and emit slices for:

- `ambiguous_vs_seen_ood_svhn`
- `ambiguous_vs_seen_ood_dtd`
- `ambiguous_vs_seen_ood_lsun_resize`
- `ambiguous_vs_seen_ood_gaussian_noise`
- `ambiguous_vs_unseen_ood_cifar100`
- `ambiguous_vs_all_ood`

Aggregate output must include:

- seen-source AUROC
- unseen CIFAR-100 AUROC
- all-OOD AUROC
- worst-source AUROC
- seen-unseen gap
- pair-scalar delta
- easy top-1
- hard top-1
- ambiguous hit

## V0.6-B Backlog

If V0.6-A does not reach the clean partial repair gate, the next package is V0.6-B:

- add GRL source adversary on OOD samples
- add OOD supervised contrastive loss
- add source-balanced calibration loss
- keep FRCNet backbone and primary heads otherwise stable

Decision-regret, candidate proposition repair, and hard resolver redesign remain V0.7+ work.
