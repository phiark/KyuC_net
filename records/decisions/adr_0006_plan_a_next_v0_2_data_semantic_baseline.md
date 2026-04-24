# ADR-0006 Plan A Next v0.2 Data Semantic Baseline

- adr_id: ADR-0006
- status: accepted
- date: 2026-04-24
- decision_makers: ["project_owner"]

## Context

`KYUC-V01-001 / plan_a_next_v0_1` proved that the FRCNet training and report pipeline can run on ROCm and that the explicit-unknown gate separates ID-like samples from SVHN-like unknown/OOD samples. The large audit showed pair AUROC `0.9813` against scalar beta `0.9483`, but also showed weak easy/hard classification and several data-semantics gaps.

The old repo mainline still used names such as `content_entropy`, `resolution_weighted_content_entropy`, and `completion_score_beta_*`, while the newest artifact line used `state_content_entropy`, `state_weighted_content_entropy`, and `top1_completion_beta_*`. That mismatch made code, documents, and artifacts diverge.

## Decision

Freeze `plan_a_next_v0_2` as the next document-driven baseline:

1. Use canonical state-level fields: `state_content_entropy`, `state_weighted_content_entropy`, and `state_entropy`.
2. Use canonical top-1 completion fields: `top1_completion_beta_0_1`, `top1_completion_beta_0_25`, `top1_completion_beta_0_5`, and `top1_completion_beta_0_75`.
3. Keep legacy field names only as reader/property aliases for historical records.
4. Split proposition views by semantics. `top1_view` is label-free; target/candidate views are label-aware audit views.
5. Reject label-aware proposition fields from the primary matched benchmark feature whitelist.
6. Require a frozen matched manifest for formal v0.2 reports, with an external reference score and stable hash.
7. Treat SVHN train/test use as seen-source unknown/OOD unless a future protocol introduces a truly held-out OOD source.
8. Keep FRCNet architecture unchanged for v0.2; use v0.2 to clean data, evaluation, and interpretation first.

## Consequences

Positive:

- v0.2 artifacts can be interpreted as state/proposition/completion layers instead of one mixed metric namespace.
- Primary benchmark features are protected against label-aware leakage.
- Future paper tables can compare pair, weighted pair, oriented scalar, and one-feature logistic scalar on the same footing.

Negative:

- v0.2 is not directly column-compatible with v0.3debug CSV outputs.
- Historical reports need legacy aliases when read back into new code.
- The frozen matched manifest adds another required artifact before a result can be called formal.

## Traceability

- linked_requirements:
  - `REQ-FN-014`
  - `REQ-FN-015`
  - `REQ-FN-018`
  - `REQ-FN-035`
  - `REQ-FN-036`
  - `REQ-FN-037`
  - `REQ-FN-038`
- affected_documents:
  - `docs/architecture/plan_a_next_v0_2_protocol.md`
  - `docs/governance/naming_and_identifier_standard.md`
  - `docs/verification/verification_and_validation_plan.md`
- affected_modules:
  - `src/frcnet/evaluation/`
  - `src/frcnet/data/plan_a.py`
  - `src/frcnet/analysis/`
