# Review Record: V0.5 Evidence Repair Scope

- date: 2026-04-26
- owner: frcnet_project
- status: review
- linked_adr: `records/decisions/adr_0007_plan_a_next_v0_5_evidence_repair.md`

## Audit Inputs

- V4 strict-freeze found no direct evidence of train/final source image contamination.
- V4 strict-freeze did find validation/final source-image reuse, so final evidence was not fully independent of checkpoint selection.
- SVHN OOD evidence is seen-source because SVHN train is used for unknown supervision.
- Pair AUROC was high, but pair-vs-best-scalar margin remained small.

## V0.5 Stable Endpoint

- explicit source provenance fields are exported through manifest, batch, analysis, and matched-manifest records.
- validation and final source partitions are disjoint by source fingerprint.
- CIFAR-100 appears only in final-test OOD records with `source_role=unseen_ood_source`.
- per-seed frozen manifests exist for seen SVHN OOD, unseen CIFAR-100 OOD, and all OOD.
- aggregate records report seen/unseen/all OOD metrics separately.

## Out Of Scope

- FRCNet architecture changes.
- decision-regret benchmark.
- candidate internal proposition repair.
- r-target / no-r-target / no-ambiguous-supervision ablation matrix.
- manuscript body rewrite beyond recording evidence limits.
