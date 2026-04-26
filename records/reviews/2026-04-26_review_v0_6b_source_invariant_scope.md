# Review: V0.6B Source-Invariant Scope

- review_id: review_v0_6b_source_invariant_scope
- date: 2026-04-26
- owner: frcnet_project
- status: accepted
- scope: `plan_a_next_v0_6b_source_invariant_loso`

## Evidence From V0.6-A

V0.6-A showed a split outcome:

- all-OOD AUROC improved because seen/far OOD sources were separated well.
- CIFAR-100 unseen AUROC stayed weak.
- CIFAR-100 samples remained high `resolution_ratio`, close to ambiguous CIFAR-10 mixtures.
- balanced checkpoint selection improved ID and ambiguous metrics but reduced unseen CIFAR-100 relative to the theory companion checkpoint.

The result is credible negative/partial evidence, not an integrity failure. Source fingerprint overlap was checked as zero for train, validation, and final manifests.

## Root Cause

The likely failure is source/style shortcut learning:

- SVHN, DTD, LSUN-resize, and Gaussian noise are visually far from CIFAR-10.
- CIFAR-100 is near-OOD and shares CIFAR-like image statistics.
- Existing unknown loss maximizes `unknown_mass` on known OOD sources but does not remove source identity information from the representation.

## V0.6B Repair Scope

V0.6B adds only the minimum source-invariant machinery:

- optional GRL `source_head`
- source adversarial CE on OOD/unknown samples
- OOD supervised contrastive loss
- source-balanced unknown calibration
- TinyImageNet as seen near-OOD pressure

The following are excluded:

- 4-way gate head
- hard resolver redesign
- decision-regret
- proposition repair beyond existing diagnostics

## Required Audit Points

Before accepting V0.6B evidence:

- CIFAR-100 must be absent from train and validation manifests.
- TinyImageNet must be marked seen-source OOD in validation/final.
- train, validation, and final source fingerprint overlap must be zero.
- frozen matched manifests must be non-empty for seen sources, unseen CIFAR-100, and all OOD.
- aggregate must report `seen_ood_tiny_imagenet_pair_auroc` and `worst_source_pair_auroc`.
