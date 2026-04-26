# Plan A Next V0.6C Near-OOD Split Repair Protocol

- document_id: arch_plan_a_next_v0_6c_near_ood_split_repair_protocol
- status: draft
- owner: frcnet_project
- last_updated: 2026-04-26

## Purpose

V0.6C repairs the near-OOD evidence gap by replacing the V0.6B "CIFAR-100 final-only source" target with a CIFAR-100 class-holdout target. The claim is narrower but cleaner for near-OOD behavior:

Can FRCNet learn from CIFAR-100 seen classes and TinyImageNet as near-OOD pressure, then generalize to held-out CIFAR-100 classes?

V0.6C also hardens the study workflow so output reuse, source-slice reporting, and artifact governance cannot silently drift from the active config.

## Study

- main study config: `configs/study/plan_a_next_v0_6c_near_ood_cifar100_class_holdout.yaml`
- train protocol: `configs/protocol/plan_a_next_v0_6c_train.yaml`
- validation protocol: `configs/protocol/plan_a_next_v0_6c_validation.yaml`
- final protocol: `configs/protocol/plan_a_next_v0_6c_test_cifar100_class_holdout.yaml`
- model config: `configs/model/frcnet_resnet18_source_invariant.yaml`
- train config: `configs/train/plan_a_next_v0_6c_near_ood_weighted_curriculum.yaml`
- eval config: `configs/eval/plan_a_next_v0_6c_near_ood_loso.yaml`
- seeds: `[7, 17, 27]`
- primary checkpoint policy: `near_ood_balanced`

## Data Contract

Training:

- ID: CIFAR-10 train
- ambiguous/hard ID: inherited from V0.6B
- unknown/OOD: SVHN train, DTD train, LSUN-resize, Gaussian noise, TinyImageNet train
- seen near-OOD: CIFAR-100 train classes `[0, 50)`
- held-out CIFAR-100 classes `[50, 100)`: absent

Validation:

- ID: CIFAR-10 test validation partition
- seen OOD: SVHN, DTD, LSUN-resize, Gaussian noise, TinyImageNet validation partitions
- seen near-OOD: CIFAR-100 test classes `[0, 50)` validation partition
- held-out CIFAR-100 classes `[50, 100)`: absent

Final:

- ID: CIFAR-10 test final partition
- seen OOD slices: SVHN, DTD, LSUN-resize, Gaussian noise, TinyImageNet held-out partitions
- seen near-OOD slice: CIFAR-100 test classes `[0, 50)` final partition
- unseen class-holdout slice: CIFAR-100 test classes `[50, 100)`

Required provenance fields:

- `source_dataset_name`
- `source_dataset_split`
- `source_role`
- `source_partition_name`
- `source_sample_indices_json`
- `augmentation_recipe`
- `source_domain_name`
- `source_domain_label`

CIFAR-100 final held-out records must use:

- `source_dataset_name = cifar100`
- `source_role = unseen_ood_classes`
- `source_partition_name = cifar100_unseen_classes`

## Manifest Class Filters

Manifest source partitions support:

- `class_label_start`
- `class_label_stop`
- `allowed_class_labels`

Rules:

- `class_label_start/class_label_stop` define a half-open interval.
- `allowed_class_labels` is an explicit allow-list and takes precedence when present.
- filters apply before source partition slicing.
- CIFAR-100 seen and unseen class sets must be disjoint and exactly `[0, 50)` / `[50, 100)`.

## Sampling Contract

Batch composition remains:

- 25% ID easy/hard
- 25% ambiguous ID
- 50% OOD/unknown

OOD source-balanced sampling is source-weighted in V0.6C:

- `cifar100_seen_classes: 3.0`
- `tiny_imagenet: 2.0`
- `svhn: 1.0`
- `dtd: 1.0`
- `lsun_resize: 1.0`
- `gaussian_noise: 1.0`

Weights affect source draw frequency inside the OOD portion; they must not collapse the batch composition.

## Checkpoint Policy

The primary policy is `near_ood_balanced`.

The policy prioritizes validation performance on near-OOD slices:

- TinyImageNet seen OOD
- CIFAR-100 seen classes

Candidate epochs must also satisfy floors:

- hard top-1 >= `0.63`
- ambiguous hit >= `0.72`

If no epoch satisfies all floors, the policy falls back to the highest near-OOD score and records that the floor was not met.

All checkpoint policies must honor `selection_policies.*.checkpoint_name`.

## Strict Study Controls

`study.resume_policy`:

- `fail_on_stale`: default; stale provenance raises.
- `rebuild_stale`: stale outputs are ignored and rebuilt.

Validated protocol controls:

- `source_balanced_sampling`
- `leave_one_source_out`
- `near_ood_training_source`
- `near_ood_training_sources`
- `require_strict_frozen_final`
- `require_source_overlap_zero`
- `protocol_variant`

V0.6C uses class-holdout language. If `leave_one_source_out = cifar100`, the protocol must reject the config because CIFAR-100 seen classes are deliberately present.

## Evaluation Contract

Frozen matched reports emit at least:

- `ambiguous_vs_seen_ood_svhn`
- `ambiguous_vs_seen_ood_dtd`
- `ambiguous_vs_seen_ood_lsun_resize`
- `ambiguous_vs_seen_ood_gaussian_noise`
- `ambiguous_vs_seen_ood_tiny_imagenet`
- `ambiguous_vs_seen_ood_cifar100_seen_classes`
- `ambiguous_vs_unseen_ood_cifar100_heldout_classes`
- `ambiguous_vs_all_ood`

Aggregate output must include:

- `seen_ood_cifar100_seen_classes_pair_auroc`
- `unseen_ood_cifar100_heldout_classes_pair_auroc`
- `seen_near_ood_pair_auroc`
- `near_ood_seen_unseen_gap`
- `worst_source_pair_auroc`
- `pair_scalar_delta`
- easy top-1
- hard top-1
- ambiguous hit

Aggregate fails if any required benchmark slice is missing or if the ranking metric is absent or `NaN`.

## Cleanup Boundary

Historical V0.2, V0.3debug, V0.4, V0.5, V0.6, and V0.6B configs remain in the repository for reproducibility. Deprecated status is documented rather than implemented by deletion.

Generated artifacts and caches that are already tracked but ignored by `.gitignore` are removed from the git index only. Local files are not deleted.
