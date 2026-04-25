# Review Record: V4.0 Strict Freeze Scope Control

- date: 2026-04-25
- owner: frcnet_project
- status: review
- linked_plan: `docs/architecture/plan_a_next_v0_4_semantic_repair_plan.md`

## Stable Endpoint

V4.0 first slice ends when strict freeze evidence works for the existing FRCNet seed007 final-test path:

- same-backbone softmax CE reference can train and export `softmax_entropy` reference scores.
- frozen matched manifest can be built from sample analysis records plus reference scores.
- matched manifest includes deterministic roles, pair ids, bin ids, construction config hash, and manifest hash.
- FRCNet report generation can run with non-empty `matched_manifest_path`.
- pair benchmark, completion scan, and scalar diagnostic tables use the same frozen manifest subset.
- proof artifacts may exist locally, but checkpoints and large generated artifacts are not part of normal code/doc staging.

## In Scope

- `softmax_ce_reference` model and CLIs.
- reference score JSONL read/write contract.
- frozen manifest builder and bin diagnostics.
- strict eval config with `require_matched_manifest: true`.
- manifest-aware scalar summaries and artifact writers.
- compact V4.0 configs, docs, and contract tests.
- local proof under `artifacts/studies/plan_a_next_v0_4_strict_freeze_local/`.

## Out Of Scope For This Slice

- FRCNet architecture changes.
- FRCNet retraining.
- candidate internal ambiguity and pairwise proposition views.
- held-out OOD source expansion.
- r-target/no-r-target/no-ambiguous ablation matrix.
- decision-regret benchmark.
- manuscript body rewrite.
- full three-seed frozen aggregate.

## Backlog For Later V4 Slices

1. Proposition repair: candidate internal entropy and pairwise proposition views.
2. Generalization repair: held-out recipes, held-out class-pair slices, and unseen OOD sources.
3. Ablation repair: r-target sweep, no-r-target, no-ambiguous-supervision, no-unknown-supervision.
4. Decision repair: offline utility/regret benchmark comparing `q_beta`, best scalar, pair state, and oracle.
5. Paper repair: manuscript bridge from binary proposition theory to K-class FRCNet native state.
6. Evidence repair: seeds `[17, 27]` and frozen multi-seed aggregate.

## Stop Rule

Do not add new scientific claims after strict freeze passes. Record remaining failures as partial or negative evidence and move them into the backlog above.
