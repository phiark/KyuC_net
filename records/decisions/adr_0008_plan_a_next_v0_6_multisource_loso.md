# ADR-0008: Plan A Next V0.6 Multisource LOSO

- status: accepted
- date: 2026-04-26
- owner: frcnet_project
- related_protocol: `plan_a_next_v0_6_multisource_loso`

## Context

V0.5 repaired source overlap and frozen evidence, but exposed a large generalization gap:

- seen SVHN OOD AUROC stayed strong.
- unseen CIFAR-100 OOD AUROC stayed weak.

The next version therefore targets unknown structure that is less tied to a single OOD source. V0.6 is split into two stages:

- V0.6-A: protocol-only multisource OOD training plus CIFAR-100 leave-one-source-out final evidence.
- V0.6-B: add source-invariant objectives only if V0.6-A does not repair unseen OOD enough.

## Decision

Adopt `plan_a_next_v0_6_multisource_loso_cifar100_holdout` as the V0.6-A study baseline.

The first implementation commits the following scope:

1. Train FRCNet with CIFAR-10 ID plus multi-source unknown supervision from SVHN, DTD, LSUN-resize, and Gaussian noise.
2. Keep CIFAR-100 as the leave-one-source-out unseen holdout.
3. Add source-balanced batch sampling so unknown supervision is balanced across OOD sources.
4. Preserve the existing FRCNet architecture and loss for V0.6-A.
5. Report seen-source, unseen-source, all-OOD, worst-source, seen-unseen gap, and pair-vs-scalar metrics when frozen slice manifests are present.

V0.6-B is explicitly not part of this first commit. It will add GRL source adversary, OOD supervised contrastive loss, and source-balanced calibration only after V0.6-A results are inspected.

## Consequences

- V0.6-A can answer whether data diversity alone repairs CIFAR-100 unseen OOD.
- Results remain comparable to V0.5 because the FRCNet main architecture is unchanged.
- Source-balanced sampling becomes part of the training contract, not an ad hoc remote-run option.
- DTD may use torchvision download support; LSUN-resize is directory-backed and must exist under `data/lsun_resize/{train,val,test}` or equivalent split roots.
- If V0.6-A fails to reach the clean partial repair gate, V0.6-B should be prioritized over hard top-1 resolver work.

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
