# Plan A Next v0.4 Semantic Repair Plan

- document_id: arch_plan_a_next_v0_4_semantic_repair_plan
- status: review
- owner: frcnet_project
- last_updated: 2026-04-25
- upstream_review: `records/reviews/2026-04-25_review_literature_project_alignment.md`

## 1. Purpose

`plan_a_next_v0_4_semantic_repair` is the next development target after the current v0.2 implementation audit. It does not replace the FRCNet dual-head architecture. It repairs the evidence chain and semantic surface so the project can support the manuscript's completion-projection claims without overstating what the existing model has proven.

The first implementation slice is `plan_a_next_v0_4_strict_freeze`. It deliberately stops at strict frozen evidence for the existing seed007 path. The scope-control record is `records/reviews/2026-04-25_review_v0_4_strict_freeze_scope_control.md`.

The target state is:

- frozen external reference-score matched benchmark,
- multi-seed FRCNet evidence,
- label-free primary comparisons,
- stronger proposition-view semantics,
- ambiguity and unknown-source generalization checks,
- a small decision-regret experiment tied to the manuscript theorem.

## 2. Non-Goals

- Do not change the core FRCNet architecture before the current semantic/evidence gaps are closed.
- Do not claim unseen OOD generalization from SVHN-only evidence.
- Do not treat label-aware target/candidate fields as primary benchmark features.
- Do not treat high-beta completion as a universal confidence score.
- Do not stage generated checkpoints or large artifacts as part of normal code/doc commits.

## 3. Work Package A: Strict Evidence Freeze

Goal: turn the current unfrozen v0.2 diagnostic into a strict reproducible benchmark.

Required changes:

- Add a same-backbone softmax CE reference training/evaluation path.
- Emit per-sample reference scores with `model_family=softmax_ce_reference` and `score_name=softmax_entropy`.
- Implement a frozen matched-manifest builder using external reference scores and deterministic binning.
- Emit matched-manifest bin diagnostics and hash records.
- Rerun FRCNet reports with non-empty `matched_manifest_path`.
- Ensure completion scans and scalar diagnostic tables use the same frozen manifest subset and test split as the primary pair benchmark.

Likely files:

- `src/frcnet/evaluation/softmax_reference.py`
- `src/frcnet/evaluation/matched_manifest.py`
- `src/frcnet/evaluation/matched_benchmark.py`
- `src/frcnet/workflows/plan_a_reporting.py`
- `src/frcnet/workflows/workflow_io.py`
- `scripts/train_plan_a.py` or a new softmax reference script
- `configs/eval/plan_a_next_v0_4_strict_freeze.yaml`
- `tests/contract/test_plan_a_next_v0_2_semantics.py`
- `tests/contract/test_plan_a_next_v0_4_strict_freeze.py`

Acceptance checks:

- frozen manifest exists under a shared study path and has a stable hash.
- changing a reference score changes the manifest hash.
- report snapshots show a non-empty `matched_manifest_path`.
- scalar scan AUROCs are computed on the same frozen records as pair AUROC.

## 4. Work Package B: Multi-Seed Close-Out

Goal: make v0.2/v0.4 evidence comparable to v0.3's three-seed aggregate.

Required changes:

- Run seeds `[7, 17, 27]` with the same train/validation/final-test protocol.
- Aggregate per-seed metrics and include mean/std/min/max.
- Preserve seed-level report directories and sidecar summaries.
- Record missing gates as partial or negative evidence instead of hiding them.

Acceptance checks:

- aggregate CSV includes pair, weighted pair, scalar, easy top-1, hard top-1, and ambiguous candidate hit.
- final report states whether each gate passed.
- no validation split is reused as final-test evidence.

## 5. Work Package C: Proposition Semantics

Goal: make proposition outputs match the theory and audit report more closely.

Required changes:

- Keep `top1_view` as the only label-free proposition view allowed in primary benchmarks.
- Keep `single_label_view`, `candidate_set_view`, and `empty_set_view` as label-aware audit views.
- Add candidate internal ambiguity diagnostics for ambiguous samples.
- Add pairwise proposition views for candidate pairs where the data recipe supplies a class pair.
- Avoid using candidate-set `tau` as a proxy for internal ambiguity.

Likely files:

- `src/frcnet/evaluation/proposition_views.py`
- `src/frcnet/evaluation/inference.py`
- `src/frcnet/evaluation/records.py`
- `src/frcnet/analysis/reporting.py`
- `tests/contract/test_plan_a_next_v0_2_semantics.py`

Acceptance checks:

- every proposition view conserves mass: `pT + pF + pU == 1`.
- label-aware views are rejected by primary feature whitelists.
- candidate internal entropy is available only for cohorts with candidate sets.

## 6. Work Package D: Ambiguity And Unknown Generalization

Goal: separate learned semantic ambiguity from recipe/source artifacts.

Required changes:

- Add held-out recipe slices beyond the training recipes.
- Report held-out class-pair metrics separately.
- Add `r_target` sweep and no-`r_target` ablation.
- Add no-ambiguous-supervision and no-unknown-supervision ablations.
- Add at least one unseen OOD source that was not used as unknown supervision.
- Preserve SVHN labels as `seen_unknown_source` or `seen_source_ood`.

Acceptance checks:

- report tables split seen recipe, held-out recipe, seen class-pair, held-out class-pair, seen unknown source, and unseen OOD source.
- ambiguous candidate hit and resolution behavior are reported per slice.
- recipe probes are reported as risk diagnostics if available.

## 7. Work Package E: Decision-Regret Benchmark

Goal: connect the code to the manuscript's decision-blindness theorem.

Required changes:

- Define a small offline decision problem over existing analysis records.
- Compare policies using:
  - one completion score `q_beta`,
  - best oriented one-feature scalar,
  - pair features `(resolution_ratio, state_weighted_content_entropy)`,
  - oracle state or label-aware upper bound.
- Report utility/regret, not only AUROC.
- Keep this benchmark separate from the primary classification benchmark.

Acceptance checks:

- pair-state policy reduces regret versus `q_beta`-only and best scalar policies.
- if it does not, report the result as negative evidence.
- manuscript-facing wording distinguishes theorem support from empirical confirmation.

## 8. Work Package F: Manuscript And Design-Doc Sync

Goal: stop the paper and project from drifting in opposite directions.

Required changes:

- Add an FRCNet-native empirical section or appendix to the manuscript branch.
- Explain the bridge from binary proposition theory to K-class FRCNet state:
  - binary theory uses `(p_T,p_F,p_U)`;
  - FRCNet native state uses `p_k = r c_k` and `u = 1-r`;
  - proposition views are derived projections of that native state.
- Update the FRCNet design note with canonical field names and current evidence limits.
- Keep EDL corrected audit as historical motivation, not the main FRCNet result.

Acceptance checks:

- paper text no longer implies FRCNet has completed decision-regret evidence before the benchmark exists.
- project docs and manuscript use the same names for state, proposition, and completion fields.

## 9. Release Gate

The v0.4 result can be treated as manuscript-facing only if:

- final benchmark uses a frozen external reference-score matched manifest.
- three seeds are reported.
- weighted-pair AUROC is at least `0.95`.
- weighted-pair AUROC exceeds the best one-feature scalar by at least `0.02`, or the report explicitly records the result as negative/partial evidence.
- easy-ID top-1 and hard-ID top-1 do not regress below the current v0.2 seed007 evidence envelope without explanation.
- ambiguous candidate hit reaches `0.75` mean or is reported as an unresolved weakness.
- decision-regret benchmark exists and is reported, including negative results.

## 10. Validation Commands

Primary local verification:

```bash
.venv313/bin/python -m pytest -q
```

Focused checks after implementing Work Package A:

```bash
.venv313/bin/python -m pytest -q tests/contract/test_plan_a_next_v0_4_strict_freeze.py
```

Report-level checks:

```bash
rg -n "matched_manifest_path: ''|unfrozen" artifacts/studies/frcnet_results artifacts/studies/plan_a_next_v0_4_main
rg -n "seen_unknown_source|unseen_ood" configs/protocol docs records
```

## 11. Artifact Governance

- Stage docs, source, configs, tests, and compact records by default.
- Do not stage checkpoints, generated plots, large CSVs, or transferred artifact directories unless a separate artifact-release task explicitly asks for them.
- Keep local remote-transfer evidence in review records when it affects conclusions, but avoid treating local path layout as the canonical study layout.
