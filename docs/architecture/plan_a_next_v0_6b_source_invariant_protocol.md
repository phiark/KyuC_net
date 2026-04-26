# Plan A Next V0.6B Source-Invariant Protocol

- document_id: arch_plan_a_next_v0_6b_source_invariant_protocol
- status: draft
- owner: frcnet_project
- last_updated: 2026-04-26

## Purpose

V0.6B repairs the V0.6-A failure mode where far/source-style OOD was learned but CIFAR-like near-OOD was not. The goal is source-invariant unknown structure, measured by leave-one-source-out CIFAR-100 final evidence.

Primary question:

Can a minimal source-invariant objective raise unseen CIFAR-100 AUROC without changing the FRCNet resolution/content heads?

## Study

- main study config: `configs/study/plan_a_next_v0_6b_source_invariant_loso_cifar100_holdout.yaml`
- B1 ablation: `configs/study/plan_a_next_v0_6b_source_adv_b1_loso_cifar100_holdout.yaml`
- B2 ablation: `configs/study/plan_a_next_v0_6b_supcon_b2_loso_cifar100_holdout.yaml`
- train protocol: `configs/protocol/plan_a_next_v0_6b_train.yaml`
- validation protocol: `configs/protocol/plan_a_next_v0_6b_validation.yaml`
- final protocol: `configs/protocol/plan_a_next_v0_6b_test_cifar100_holdout.yaml`
- model config: `configs/model/frcnet_resnet18_source_invariant.yaml`
- main train config: `configs/train/plan_a_next_v0_6b_source_invariant_b3_curriculum.yaml`
- eval config: `configs/eval/plan_a_next_v0_6b_source_invariant_loso.yaml`
- seeds: `[7, 17, 27]`

## Data Contract

Training:

- ID: CIFAR-10 train
- unknown supervision: SVHN train, DTD train, LSUN-resize, Gaussian noise, TinyImageNet train
- CIFAR-100: absent

Validation:

- ID: CIFAR-10 test partition `[0, 200)` per class
- seen-source OOD: SVHN, DTD, LSUN-resize, Gaussian noise, TinyImageNet validation partitions
- CIFAR-100: absent

Final:

- ID: CIFAR-10 test partition `[200, 1000)` per class
- seen-source OOD slices: SVHN, DTD, LSUN-resize, Gaussian noise, TinyImageNet held-out partitions
- unseen-source OOD slice: CIFAR-100 test

TinyImageNet is directory-backed and must exist at `data/tiny_imagenet/tiny-imagenet-200/{train,val}`. It is not downloaded by the default preflight path.

Preparation helper:

```bash
.venv313/bin/python scripts/prepare_tiny_imagenet.py --download --extract
```

## Model And Loss Contract

The primary FRCNet contract remains:

- `resolution_ratio = sigmoid(resolution_logit)`
- `content_distribution = softmax(content_logits)`
- `unknown_mass = 1 - resolution_ratio`

V0.6B adds optional source-adversary fields:

- `ModelOutput.source_logits`
- `FRCNetModel(source_adversary_enabled=true, num_source_domains=7, grl_lambda=1.0)`

Loss additions:

- `source_adv_weight = 0.05`
- `ood_supcon_weight = 0.10`
- `source_balanced_calibration_weight = 0.05`
- `ood_supcon_temperature = 0.1`

OOD SupCon uses OOD/unknown samples from different sources as positives and ID/ambiguous samples as negatives. Same-source OOD samples are excluded from the denominator.

## Evaluation Contract

Frozen matched reports emit:

- `ambiguous_vs_seen_ood_svhn`
- `ambiguous_vs_seen_ood_dtd`
- `ambiguous_vs_seen_ood_lsun_resize`
- `ambiguous_vs_seen_ood_gaussian_noise`
- `ambiguous_vs_seen_ood_tiny_imagenet`
- `ambiguous_vs_unseen_ood_cifar100`
- `ambiguous_vs_all_ood`

Aggregate output must include:

- seen-source AUROC
- seen TinyImageNet AUROC
- unseen CIFAR-100 AUROC
- all-OOD AUROC
- worst-source AUROC
- seen-unseen gap
- pair-scalar delta
- easy top-1
- hard top-1
- ambiguous hit

## Scope Boundary

V0.6B does not introduce a 4-way gate head, does not redesign hard-ID resolver behavior, and does not add decision-regret evidence. Those remain V0.7+ work packages.
