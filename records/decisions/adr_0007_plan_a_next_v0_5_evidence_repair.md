# ADR-0007 Plan A Next v0.5 Evidence Repair

- adr_id: ADR-0007
- status: accepted
- date: 2026-04-26
- decision_makers: ["project_owner"]

## Context

V4 strict-freeze showed strong three-seed ambiguous-vs-OOD separation, but the audit found two evidence limits. First, FRCNet train and final-test source images were disjoint, but validation and final-test reused underlying CIFAR-10 test / SVHN test source images. Second, SVHN was both an unknown-supervision source and an OOD evaluation source, so SVHN evidence must be described as seen-source OOD.

## Decision

Freeze `plan_a_next_v0_5_evidence_repair` as an evidence-quality milestone:

1. Keep the FRCNet dual-head architecture unchanged.
2. Retrain seeds `[7, 17, 27]` with validation-driven checkpoint selection.
3. Use explicit source partitions so validation and final-test source fingerprints do not overlap.
4. Add CIFAR-100 test as final-only `unseen_ood_source`.
5. Preserve SVHN final OOD as `seen_source_ood`.
6. Generate frozen matched benchmarks for `ambiguous_vs_seen_ood_svhn`, `ambiguous_vs_unseen_ood_cifar100`, and `ambiguous_vs_all_ood`.
7. Treat performance misses as clean partial/negative evidence rather than changing the claim.

## Consequences

- V0.5 results can support stronger paper-preparation language about evidence cleanliness.
- V0.5 still does not close decision-regret or full proposition-repair claims.
- Training and reference runs are more expensive because final evidence requires three seeds plus per-seed frozen manifests.

## Traceability

- linked_protocol: `docs/architecture/plan_a_next_v0_5_evidence_repair_protocol.md`
- linked_review: `records/reviews/2026-04-26_review_v0_5_evidence_repair_scope.md`
- affected_configs:
  - `configs/protocol/plan_a_next_v0_5_train.yaml`
  - `configs/protocol/plan_a_next_v0_5_validation.yaml`
  - `configs/protocol/plan_a_next_v0_5_test.yaml`
  - `configs/eval/plan_a_next_v0_5_evidence_repair.yaml`
