# Review: V0.6C Near-OOD Split Repair Scope

- review_id: review_v0_6c_near_ood_split_repair_scope
- date: 2026-04-26
- owner: frcnet_project
- status: accepted
- scope: `plan_a_next_v0_6c_near_ood_cifar100_class_holdout`

## Evidence From V0.6B

V0.6B kept the FRCNet architecture narrow and added source-invariant pressure. The result still did not make CIFAR-100 final-only source evidence strong enough for a paper-facing unseen-source claim.

The useful conclusion is not "the method failed globally". The more precise conclusion is:

- far OOD and source-style OOD are already learnable.
- near-OOD with CIFAR-like statistics remains the hard slice.
- CIFAR-100 final-only source is too strict for the next repair step if the immediate goal is to learn near-OOD structure.

## V0.6C Repair Claim

V0.6C narrows the claim from unseen source to unseen classes:

- train/validation may use CIFAR-100 classes `[0, 50)`.
- final held-out evidence is CIFAR-100 classes `[50, 100)`.
- result wording must say "unseen CIFAR100 classes".

This makes the evidence less broad than V0.6B, but it is better targeted at the observed near-OOD failure.

## Workflow Risks Being Repaired

V0.6C also addresses integrity issues that can make good-looking results untrustworthy:

- stale stage resume by output existence only.
- declared-but-dead `protocol_controls`.
- aggregate ranking with missing or `NaN` source slices.
- checkpoint-name config not fully honored.
- skipped OOD-only batches despite active source losses.
- tracked ignored artifacts and caches.

## Required Acceptance Checks

Before accepting V0.6C evidence:

- CIFAR-100 seen and unseen class sets are exactly disjoint `[0, 50)` / `[50, 100)`.
- held-out final records carry `source_role=unseen_ood_classes`.
- train/validation/final source fingerprints have zero overlap.
- frozen matched manifests are non-empty for seen near-OOD, held-out CIFAR-100 classes, and all OOD.
- aggregate contains `seen_ood_cifar100_seen_classes_pair_auroc`, `unseen_ood_cifar100_heldout_classes_pair_auroc`, `worst_source_pair_auroc`, and `pair_scalar_delta`.
- stale provenance fails by default.
- `git ls-files -ci --exclude-standard` is empty or every remaining item is explicitly documented.

## Out-Of-Scope Confirmation

The following remain V0.7+ work:

- hard resolver redesign.
- 4-way gate.
- decision-regret.
- full proposition repair.
- claiming unseen CIFAR-100 source generalization from V0.6C.
