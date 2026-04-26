# ADR-0010: Plan A Next V0.6C Near-OOD Class-Holdout Repair

- status: accepted
- date: 2026-04-26
- owner: frcnet_project
- related_protocol: `plan_a_next_v0_6c_near_ood_cifar100_class_holdout`

## Context

V0.6-A and V0.6B both showed that FRCNet can separate far/source-style OOD sources, but CIFAR-like near-OOD remains the weak slice. V0.6B also exposed workflow risks that can contaminate evidence quality:

- study stage resume can reuse old outputs without validating the current config provenance.
- several `protocol_controls` keys are declared in YAML but not enforced at execution time.
- aggregate can silently rank seeds when required source-slice metrics are missing or `NaN`.
- OOD-only batches can be skipped even when source-invariant losses have valid gradients.
- tracked generated artifacts conflict with the documented artifact-governance policy.

The next repair must therefore address the scientific evidence gap and the workflow integrity gap together.

## Decision

Adopt `plan_a_next_v0_6c_near_ood_cifar100_class_holdout` as the V0.6C study.

The scientific change is a CIFAR-100 class-holdout protocol:

- `cifar100_seen_classes`: classes `[0, 50)` are allowed in train and validation as seen near-OOD pressure.
- `cifar100_unseen_classes`: classes `[50, 100)` are reserved for the final held-out benchmark.
- final wording must say "unseen CIFAR100 classes", not "unseen CIFAR100 source".
- TinyImageNet remains a seen near-OOD source.

The workflow change is a strict study-integrity cleanup:

- stage resume is guarded by provenance hashes instead of output existence alone.
- default `resume_policy` is `fail_on_stale`; `rebuild_stale` must be explicit.
- `protocol_controls` declarations are validated by execution code.
- aggregate ranking fails when the ranking metric or required source slices are missing.
- custom checkpoint names in `selection_policies.*.checkpoint_name` are honored.
- ignored generated artifacts are removed from the git index without deleting local files.

## Scope

In scope:

1. CIFAR-100 class filtering in manifest construction.
2. source-weighted source-balanced sampling for near-OOD pressure.
3. `near_ood_balanced` checkpoint policy.
4. stale-resume provenance checks.
5. protocol-control enforcement.
6. aggregate fail-fast checks for required slices and ranking metrics.
7. OOD-only source-loss optimizer-step fix.
8. internal workflow cleanup modules while preserving public entrypoints.

Out of scope:

- 4-way gate head.
- hard-ID resolver redesign.
- decision-regret experiments.
- full proposition repair.
- deleting historical configs, docs, or local generated artifacts.

## Consequences

- V0.6C cannot claim unseen CIFAR-100 source generalization because CIFAR-100 seen classes enter train and validation.
- V0.6C can claim a stricter near-OOD class-holdout test if source fingerprints and class sets are clean.
- V0.6B remains useful as the final-only CIFAR-100 source baseline.
- Workflow failures become louder: stale outputs, missing slices, and mismatched controls stop the study instead of producing misleading aggregate records.

## Release Gate

Clean partial repair:

- unseen CIFAR100 held-out-class AUROC >= `0.68`
- seen near-OOD AUROC >= `0.75`
- worst-source AUROC >= `0.65`
- `near_ood_seen_unseen_gap <= 0.25`
- source overlap audit = `0`

Strong repair:

- unseen CIFAR100 held-out-class AUROC >= `0.75`
- seen near-OOD AUROC >= `0.82`
- worst-source AUROC >= `0.70`
- `near_ood_seen_unseen_gap <= 0.20`
- ambiguous hit >= `0.73`

Very strong repair:

- unseen CIFAR100 held-out-class AUROC >= `0.82`
- all-OOD AUROC >= `0.86`
- worst-source AUROC >= `0.76`
- hard top-1 >= `0.68`

## Traceability

- requirements: `REQ-FN-048` through `REQ-FN-053`, `REQ-SCI-011`
- verification: `VER-CON-015` through `VER-CON-020`, `VER-INT-026` through `VER-INT-031`, `VER-SCI-008`
