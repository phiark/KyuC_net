# ADR-0009: Plan A Next V0.6B Source-Invariant Repair

- status: accepted
- date: 2026-04-26
- owner: frcnet_project
- related_protocol: `plan_a_next_v0_6b_source_invariant_loso`

## Context

V0.6-A answered the protocol-only question negatively. Multi-source unknown supervision improved all-OOD evidence, but did not repair the CIFAR-100 leave-one-source-out slice.

Observed failure pattern:

- seen/far OOD sources were easy for the gate.
- CIFAR-100 stayed CIFAR-like in FRCNet geometry and often received high `resolution_ratio`.
- pair-vs-scalar delta stayed small, so the current pair geometry did not add enough robust unknown structure.

This indicates source/style shortcuts rather than source-invariant unknown semantics.

## Decision

Adopt `plan_a_next_v0_6b_source_invariant_loso_cifar100_holdout` as the V0.6B repair study.

The implementation scope is deliberately narrow:

1. Keep the FRCNet backbone, resolution head, and content head.
2. Add an optional GRL-backed `source_head` only.
3. Add source adversarial loss, OOD supervised contrastive loss, and source-balanced unknown calibration.
4. Add TinyImageNet as a seen near-OOD training and validation source.
5. Keep CIFAR-100 final-only as the unseen OOD source.
6. Enforce zero source fingerprint overlap for train, validation, and final manifests.

The following are out of scope:

- 4-way cohort/gate head
- hard resolver redesign
- decision-regret experiments
- candidate proposition repair

## Consequences

- V0.6B remains comparable with V0.6-A because the primary FRCNet heads are unchanged.
- TinyImageNet is used as a seen near-OOD source, so final claims must distinguish seen near-OOD from unseen CIFAR-100.
- The new model config is backward compatible because source adversary fields default off in the base model.
- Existing V0.6-A configs remain valid.

## Release Gate

Clean partial repair:

- unseen CIFAR-100 AUROC >= `0.65`
- all-OOD AUROC >= `0.75`
- worst-source AUROC >= `0.62`
- source overlap audit = `0`

Strong repair:

- unseen CIFAR-100 AUROC >= `0.70`
- all-OOD AUROC >= `0.80`
- worst-source AUROC >= `0.68`
- seen-unseen gap <= `0.25`
- ambiguous hit >= `0.73`

Very strong repair:

- unseen CIFAR-100 AUROC >= `0.80`
- all-OOD AUROC >= `0.85`
- worst-source AUROC >= `0.75`
- seen-unseen gap <= `0.20`
- hard top-1 >= `0.68`
