# Review Record: Literature And Project Alignment

- date: 2026-04-25
- owner: frcnet_project
- status: review
- source_documents:
  - `/Users/zero_lab/Documents/Hecate/FRCNet_项目审计报告_执行意见_内部版.docx`
  - `/Users/zero_lab/Documents/Hecate/Articles/v_3/V3_reset.tex`
  - `/Users/zero_lab/Documents/Hecate/Articles/Entropy_Architecture/A_FRCNet.docx`
- linked_project_docs:
  - `docs/architecture/plan_a_next_v0_2_protocol.md`
  - `records/decisions/adr_0006_plan_a_next_v0_2_data_semantic_baseline.md`
  - `records/reviews/2026-04-24_review_plan_a_next_v0_2_issue_ledger.md`

## 1. Alignment Verdict

The project is aligned to the FRCNet design document and to the first semantic-repair layer of the audit report. It is not yet aligned to the full manuscript claim surface.

Current project state is best described as:

- `plan_a_next_v0_2`: implemented as a single-seed, final-test diagnostic run with clean state/proposition/completion fields.
- strict `plan_a_next_v0_2` release gate: not yet satisfied because the current evidence is unfrozen, single-seed, and lacks the softmax CE reference baseline.
- audit-report `v0.4_semantic_repair`: partially started through v0.2 contracts, but still missing frozen evidence, stronger proposition semantics, held-out ambiguity analysis, and decision-regret experiments.
- manuscript `V3_reset`: theoretically ahead of the code on completion projection and decision blindness, but empirically behind the current project because it still centers a corrected EDL audit instead of the native FRCNet runs.

## 2. What Is Already Aligned

Model-native layer:

- FRCNet implements the design-note state form `p_k = r * c_k` and `u = 1 - r`.
- The project exports canonical state fields: `resolution_ratio`, `unknown_mass`, `state_content_entropy`, `state_weighted_content_entropy`, and `state_entropy`.
- The entropy decomposition contract is represented in `src/frcnet/evaluation/state_metrics.py` and contract-tested.

Proposition and completion layer:

- `PropositionViewRecord` exists and carries `view_name`, `label_aware`, `pT/pF/pU`, and `tau` semantics.
- Primary benchmark feature whitelist excludes label-aware proposition fields such as `proposition_truth_ratio`.
- Beta policy helpers exist for top-1 symmetric, candidate symmetric, and binary pignistic defaults.
- Legacy analysis names are kept as reader aliases instead of remaining the primary schema.

Training and data protocol:

- v0.2 curriculum training is implemented with warmup/main/stabilize phases.
- hard-ID recipe parameters are now config-driven.
- final-test protocol includes MixUp plus overlay/occlusion recipes and held-out class pairs.
- SVHN is explicitly marked as `seen_unknown_source`, which avoids overclaiming unseen OOD generalization.

Existing evidence:

- v0.2 seed007 final test was generated under `artifacts/studies/frcnet_results/plan_a_next_v0_2-seed007/`.
- The final-test report is correctly marked as `report_final_test_unfrozen`, with `matched_manifest_path: ''`.
- v0.3 has a three-seed aggregate under `artifacts/studies/plan_a_v0_3_main/aggregate/`.

## 3. Where The Literature Is Ahead

`V3_reset.tex` is ahead of the current project on theory:

- It gives the cleanest statement of binary proposition states `(p_T,p_F,p_U)`, resolution coordinates `(r,tau)`, and completion projection `q_beta`.
- It states the main negative theorem: a single completion/scalar confidence can be decision-blind.
- It cleanly separates explicit-unknown states from softmax confidence and EDL-style post-hoc uncertainty.

The audit report is ahead of the current project on semantic governance:

- It requires a strict four-layer separation: native state, state diagnostics, proposition view, and completion/decision layer.
- It correctly warns that K-class content entropy and binary proposition entropy must not be conflated.
- It requires frozen external reference-score matched manifests, scalar fairness, weighted-pair primacy, and decision-regret experiments.

`A_FRCNet.docx` is ahead in experimental intent:

- It names the core risk that artificial ambiguous samples may teach recipe artifacts.
- It asks for ablations and baselines beyond the current v0.2 evidence: softmax CE, EDL, SelectiveNet, no-ambiguous supervision, and single-completion ablation.

## 4. Where The Literature Is Behind

`V3_reset.tex` is behind the project empirically:

- Its empirical section still describes a corrected EDL audit, not the native FRCNet v0.2/v0.3 evidence.
- It does not yet use the current canonical project fields such as `state_content_entropy`, `state_weighted_content_entropy`, and `top1_completion_beta_*`.
- It does not include the current FRCNet result pattern: strong gate separation, improved easy/hard top-1 in v0.2, but weak pair-vs-scalar gap.

`A_FRCNet.docx` is behind the project contract layer:

- It predates `PropositionViewRecord`, `MatchedManifestRecord`, canonical field names, and primary-label-free whitelist rules.
- It frames the experiment as design intent, not as a reproducible manifest/report pipeline.

The audit report is partly stale:

- Several items it flagged have already been implemented or partially implemented: state/proposition/completion naming, beta policy helpers, label-aware scalar guard, hard-ID config routing, and v0.2 protocol documents.
- Its proposed `v0.4_semantic_repair` is still valid as the next target, but should now start from the current v0.2 implementation rather than from the older drift state.

## 5. Where The Project Is Ahead

The project is ahead of the manuscript on FRCNet-native evidence:

- It has a runnable FRCNet dual-head training/inference/report path.
- It has concrete v0.2 final-test artifacts and v0.3 multi-seed aggregate artifacts.
- It now exports both state metrics and proposition-view records, so the code is closer to the audit report's semantic model than the manuscript text currently is.

The project is ahead of the design note on engineering rigor:

- It has explicit protocol/config chains for train, validation, final test, analysis, and study.
- It has contract tests for state decomposition, proposition mass conservation, beta policy defaults, scalar whitelist behavior, hard-ID config routing, and manifest hash sensitivity.
- It has artifact-sidecar summaries for provenance.

## 6. Where The Project Is Behind

Evidence quality:

- No frozen external reference-score matched manifest exists in the current artifact tree.
- Current v0.2 final-test report is an unfrozen diagnostic, not a strict frozen-manifest benchmark.
- v0.2 is only available for seed007 locally; the protocol target is seeds `[7, 17, 27]`.
- The softmax CE same-backbone reference baseline is only configured/named, not yet trained and emitted as a reference-score source.

Model understanding:

- v0.2 final-test pair AUROC is high, but the gap over the best one-feature scalar is too small for the v0.2 release gate.
- The current final-test ambiguous candidate hit rate is below the v0.2 target.
- Gate separation is strong, but this alone does not prove robust resolved-side content learning.
- The current project still lacks the action/decision-regret experiment needed to support the manuscript's decision-blindness claim.

Semantic implementation:

- `MatchedManifestRecord` has a record/hash/read/write contract, but the project still needs a builder that constructs the frozen manifest from external reference scores and emits bin diagnostics.
- Completion scalar scans currently need to be made manifest-aware so auxiliary scalar tables use the same frozen split as the pair benchmark.
- Primary pair config is still raw `(resolution_ratio, state_content_entropy)` while the audit report recommends weighted content entropy as the main low-resolution-safe pair.
- Candidate-set proposition reporting is present, but candidate internal ambiguity and pairwise proposition views are not yet first-class outputs.

Artifact governance:

- Existing generated outputs are useful evidence but should not be staged with code/docs unless a separate artifact-hygiene task explicitly includes them.
- Paths still show transfer/provenance drift between remote study layout and local `artifacts/studies/frcnet_results`.

## 7. Current Results Snapshot

v0.2 seed007 final test, unfrozen diagnostic:

- pair AUROC: `0.987811`
- weighted-pair AUROC: `0.988978`
- scalar AUROC: `0.985433`
- weighted-pair minus scalar: `0.003545`
- easy-ID top-1: `0.752`
- hard-ID top-1: `0.674`
- ambiguous candidate hit: `0.730`

v0.3 main aggregate, three seeds:

- pair AUROC mean/std: `0.997257 / 0.000233`
- weighted-pair AUROC mean/std: `0.986776 / 0.003727`
- scalar AUROC mean/std: `0.990947 / 0.001902`
- easy-ID top-1 mean/std: `0.5985 / 0.0327`
- hard-ID top-1 mean/std: `0.4333 / 0.0411`
- ambiguous candidate hit mean/std: `0.7763 / 0.0683`

Interpretation:

- v0.2 is better than v0.3 on easy/hard resolved-side classification in the available seed007 evidence.
- v0.3 is better on multi-seed stability and pair AUROC.
- neither version yet closes the manuscript-level decision-regret claim.
- v0.2 should be recorded as partial evidence, not as final paper evidence.

## 8. Development Direction

The next development target should be `plan_a_next_v0_4_semantic_repair`, not another architecture rewrite.

Minimum next work:

1. Freeze evidence: train or reuse a same-backbone softmax CE reference, emit reference scores, build a frozen matched manifest, and rerun reports with non-empty `matched_manifest_path`.
2. Finish strict v0.2 evidence: run seeds 17 and 27 using the same protocol and aggregate seed metrics.
3. Repair proposition semantics: add candidate internal ambiguity and pairwise proposition views; keep label-aware views out of primary benchmarks.
4. Make scalar scans and all auxiliary tables use the same frozen manifest split as the primary pair benchmark.
5. Promote weighted pair to the paper-facing primary comparison, with raw state entropy kept as an auxiliary ablation.
6. Add ambiguity and unknown-source generalization checks: held-out recipe, held-out class-pair slices, r-target sweep, no-r-target/no-ambiguous ablations, and unseen OOD sources beyond SVHN.
7. Add a small decision-regret benchmark with policies using `q_beta` only, best scalar, pair features, and oracle state.
8. Update the manuscript bridge text so `V3_reset.tex` distinguishes binary proposition theory from FRCNet's K-class native state extension.

## 9. Planning Link

The implementation plan for this audit is recorded in:

- `docs/architecture/plan_a_next_v0_4_semantic_repair_plan.md`
