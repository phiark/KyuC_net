# Review: V0.6 Evidence Report

- review_id: review_v0_6_evidence_report
- date: 2026-04-26
- owner: frcnet_project
- status: report
- scope: `plan_a_next_v0_6*`
- covered_versions:
  - `plan_a_next_v0_6_multisource_loso`
  - `plan_a_next_v0_6b_source_invariant_loso`
  - `plan_a_next_v0_6c_near_ood_cifar100_class_holdout`

## 1. Executive Summary

V0.6 is the point where the project stopped treating "OOD separation" as a single problem.

V0.5 had already cleaned the evidence path and showed that the original FRCNet geometry could separate ambiguous ID from seen SVHN OOD, but it also exposed a weak CIFAR-100 final-only slice. V0.6 therefore asked a narrower and more important question:

Can FRCNet learn unknown structure that is not just a shortcut for a known OOD source?

The current answer is mixed:

- far/source-style OOD is learnable.
- all-OOD evidence improved after multi-source/source-invariant repair.
- CIFAR-like near-OOD remains the hard boundary.
- current two-coordinate FRCNet geometry does not yet support a broad "natural three-group geometry" claim.

V0.6 should therefore be reported as a useful negative/partial result, not as a failed project and not as a completed paper-level claim.

## 2. Scientific Question

The original three-group geometry expectation was:

- easy ID: high `resolution_ratio`, low `state_content_entropy`.
- ambiguous ID: middle/high `resolution_ratio`, high `state_content_entropy`.
- OOD/unknown: low `resolution_ratio`.

V0.6 tests whether this geometry survives when OOD is not just SVHN-style far OOD but includes near-OOD data that shares CIFAR-like image statistics.

The decisive target is CIFAR-100:

- In V0.6A and V0.6B, CIFAR-100 is final-only unseen-source OOD.
- In V0.6C, the claim is narrowed to CIFAR-100 class holdout: train/validation can use classes `[0, 50)`, final held-out evidence uses classes `[50, 100)`.

## 3. Evidence Before V0.6

V0.5 repaired source-overlap and frozen-manifest evidence quality. It showed that the previous strong results were not enough for a broad unknown-generalization claim.

V0.5 aggregate:

| metric | mean |
| --- | ---: |
| `seen_ood_pair_auroc` | 0.9842 |
| `unseen_ood_pair_auroc` | 0.5982 |
| `all_ood_pair_auroc` | 0.7142 |
| `pair_auroc` | 0.7142 |
| `scalar_auroc` | 0.7178 |

Interpretation:

- Seen SVHN OOD was already strong.
- Unseen CIFAR-100 was weak.
- Pair geometry was not clearly stronger than scalar readout.
- The failure was credible because the evidence path had been cleaned enough to rule out a simple source-overlap explanation.

## 4. V0.6A: Multisource LOSO

V0.6A changed the data protocol, not the main FRCNet architecture.

It added multi-source unknown supervision:

- SVHN
- DTD
- LSUN-resize
- Gaussian noise

It kept CIFAR-100 as a leave-one-source-out final holdout.

The scientific purpose was to test whether source diversity alone could repair unseen CIFAR-100 behavior.

Conclusion:

- Data diversity helped all-OOD behavior.
- It did not make CIFAR-100 final-only evidence strong enough.
- The likely failure mode was source/style shortcut learning: the gate learned far OOD sources better than near-OOD semantics.

This justified moving to V0.6B instead of tuning hard-ID classification first.

## 5. V0.6B: Source-Invariant Repair

V0.6B kept the FRCNet backbone, resolution head, and content head stable, but added minimal source-invariant pressure:

- optional GRL-backed `source_head`
- source adversarial loss on OOD/unknown samples
- OOD supervised contrastive loss
- source-balanced unknown calibration
- TinyImageNet as seen near-OOD pressure

V0.6B aggregate:

| metric | mean | std | interpretation |
| --- | ---: | ---: | --- |
| `pair_auroc` | 0.8314 | 0.0072 | all-OOD separation improved |
| `weighted_pair_auroc` | 0.8393 | 0.0098 | weighted pair slightly better |
| `scalar_auroc` | 0.8260 | 0.0057 | scalar remained close to pair |
| `seen_ood_pair_auroc` | 0.9116 | 0.0117 | seen sources still strong |
| `unseen_ood_cifar100_pair_auroc` | 0.6272 | 0.0207 | CIFAR-100 still weak |
| `worst_source_pair_auroc` | 0.6272 | 0.0207 | worst source is CIFAR-100 |
| `seen_unseen_gap` | 0.2844 | 0.0168 | gap remains large |
| `pair_scalar_delta` | 0.0054 | 0.0024 | pair adds little over scalar |

V0.6B source slices:

| benchmark | mean pair AUROC |
| --- | ---: |
| `ambiguous_vs_seen_ood_gaussian_noise` | 0.9981 |
| `ambiguous_vs_seen_ood_lsun_resize` | 0.9838 |
| `ambiguous_vs_seen_ood_svhn` | 0.9603 |
| `ambiguous_vs_seen_ood_dtd` | 0.9332 |
| `ambiguous_vs_seen_ood_tiny_imagenet` | 0.6826 |
| `ambiguous_vs_unseen_ood_cifar100` | 0.6272 |

Interpretation:

- The model separates far OOD very well.
- TinyImageNet and CIFAR-100 show that near-OOD is the real boundary.
- Source-invariant objectives improved the global picture but did not produce broad unseen-source semantics.
- The current pair geometry is not yet a decisive improvement over a scalar confidence-like readout.

## 6. Checkpoint Tradeoff

The V0.6B checkpoint-policy comparison shows a real scientific tradeoff.

Balanced checkpoint:

- better easy/hard/ambiguous behavior.
- weaker unseen CIFAR-100 separation.

Theory companion checkpoint:

- better OOD and near-OOD separation.
- worse hard-ID and ambiguous behavior.

This means the project is not facing a single engineering bug. The objective is genuinely multi-goal:

- keep ID resolution high.
- keep hard-ID classification correct.
- keep ambiguous candidate behavior useful.
- push near-OOD toward unknown.
- avoid learning source identity as the shortcut.

The current model/loss family has not yet found a stable point that satisfies all of these at once.

## 7. V0.6C: Near-OOD Class-Holdout Repair

V0.6C is a response to the V0.6B boundary, not a completed positive result.

It narrows the claim:

- no longer claim unseen CIFAR-100 source generalization.
- allow CIFAR-100 classes `[0, 50)` in train/validation as seen near-OOD pressure.
- reserve CIFAR-100 classes `[50, 100)` for final held-out evidence.
- final wording must say "unseen CIFAR100 classes".

This is scientifically less broad than V0.6B, but more targeted and more honest.

V0.6C also repairs workflow integrity risks:

- stale stage resume by output existence only.
- declared-but-dead protocol controls.
- aggregate ranking with missing or `NaN` source slices.
- checkpoint-name config drift.
- skipped OOD-only batches when source losses still have gradients.
- tracked ignored artifacts.

## 8. What V0.6 Proves

V0.6 supports the following claims:

1. FRCNet can learn strong separation for far/source-style OOD.
2. Multi-source unknown supervision improves all-OOD aggregate behavior.
3. Source-invariant repair helps but does not solve CIFAR-like near-OOD.
4. CIFAR-100 final-only source generalization is not yet supported.
5. A broad natural three-group geometry claim is not supported by current evidence.
6. The next defensible claim should be near-OOD class-holdout behavior, not unseen-source behavior.

## 9. What V0.6 Does Not Prove

V0.6 does not prove:

- natural three-group geometry.
- robust unknown semantics independent of source style.
- superiority over all scalar confidence baselines.
- decision-regret improvement.
- full proposition-level repair.
- hard-ID resolver robustness.

These remain V0.7+ work packages unless V0.6C produces unexpectedly strong class-holdout evidence.

## 10. Diagnosis

The main failure is not that the project cannot train or report results.

The main failure is that the original geometry assumption is too strong for current evidence:

- far OOD is visually distinct and easy for the gate.
- near-OOD shares CIFAR-like image statistics.
- CIFAR-100 can look resolved even when it is semantically outside CIFAR-10.
- artificial ambiguous samples mostly teach mixup/candidate-set behavior, not natural ambiguity.
- the current `(resolution_ratio, state_content_entropy)` plane is too small to encode all distinctions cleanly.

Therefore V0.6 is best understood as the point where the project discovered the difference between:

- protocol-induced geometry
- source-style OOD separation
- near-OOD semantic generalization
- natural three-group geometry

Only the first two are currently well supported.

## 11. Recommended Reporting Language

Use:

> V0.6 shows that FRCNet learns strong far-OOD and source-style separation, but CIFAR-like near-OOD remains the limiting case. Multi-source and source-invariant training improve all-OOD aggregate behavior, yet final-only CIFAR-100 evidence remains insufficient for a broad unseen-source claim. The next claim is therefore narrowed to CIFAR-100 class-holdout near-OOD behavior.

Avoid:

> FRCNet naturally discovers three geometric groups.

Avoid:

> V0.6 solves unknown detection.

Avoid:

> CIFAR-100 failure is only an implementation issue.

## 12. Next Gate

Accept V0.6C only if all of the following are true:

- CIFAR-100 seen and held-out classes are exactly disjoint.
- source fingerprint overlap is zero for train, validation, and final.
- frozen matched manifests are non-empty for seen near-OOD, held-out CIFAR-100 classes, and all OOD.
- aggregate includes `seen_ood_cifar100_seen_classes_pair_auroc`, `unseen_ood_cifar100_heldout_classes_pair_auroc`, `worst_source_pair_auroc`, and `pair_scalar_delta`.
- stale provenance fails by default.
- the result language says "unseen CIFAR100 classes", not "unseen CIFAR100 source".

If V0.6C passes, it becomes a clean near-OOD repair milestone.

If V0.6C fails, V0.7 should stop widening data sources and instead redesign the decision/state side:

- hard resolver redesign
- 4-way gate or explicit near-OOD head
- decision-regret benchmark
- fuller proposition repair
- density/support-set or semantic-neighborhood modeling

## 13. Artifact References

- V0.5 aggregate: `artifacts/studies/plan_a_next_v0_5_evidence_repair_main_arc/aggregate/metric_summary.csv`
- V0.6B aggregate: `artifacts/studies/plan_a_next_v0_6b_source_invariant_loso_cifar100_holdout/aggregate/metric_summary.csv`
- V0.6B source slices: `artifacts/studies/plan_a_next_v0_6b_source_invariant_loso_cifar100_holdout/aggregate/source_slice_summary.csv`
- V0.6 protocol ADR: `records/decisions/adr_0008_plan_a_next_v0_6_multisource_loso.md`
- V0.6B protocol ADR: `records/decisions/adr_0009_plan_a_next_v0_6b_source_invariant.md`
- V0.6C protocol ADR: `records/decisions/adr_0010_plan_a_next_v0_6c_near_ood_split_repair.md`
