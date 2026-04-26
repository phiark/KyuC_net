# Review: V0.6 Multisource LOSO Scope

- date: 2026-04-26
- scope: `plan_a_next_v0_6_multisource_loso`
- status: implementation-scope-record

## V0.5 Finding Being Repaired

V0.5 cleaned evidence provenance but showed a large source-generalization gap:

- seen SVHN OOD was strong.
- unseen CIFAR-100 OOD was weak.
- source overlap audit was clean enough to trust the weakness as a real boundary, not as a split leak artifact.

## V0.6-A Scope

Implemented first:

- multi-source OOD/unknown source loader support for DTD, directory-backed LSUN-resize, Gaussian noise, and CIFAR-100
- `source_domain_name` / `source_domain_label` manifest and batch provenance
- multi-source `unknown_sources` manifest allocation
- source-balanced batch sampler
- LOSO CIFAR-100 holdout study, protocol, train, and eval configs
- aggregate fields for seen, unseen, all, worst-source, seen-unseen gap, and pair-scalar delta

Not included in V0.6-A:

- GRL source adversary
- OOD supervised contrastive loss
- source-balanced calibration loss
- hard resolver redesign
- decision-regret experiments

## Expected Interpretation

If CIFAR-100 unseen AUROC improves materially, the main bottleneck was source diversity in unknown supervision.

If CIFAR-100 remains near V0.5, the next implementation should be V0.6-B source-invariant training objectives rather than hard top-1 resolver tuning.

## Local Verification

Initial contract checks:

- `tests/contract/test_plan_a_next_v0_6_multisource_loso.py`
- `tests/contract/test_plan_a_next_v0_5_evidence_repair.py`
- `tests/contract/test_plan_a_next_v0_2_semantics.py`

Use `.venv313/bin/python -m pytest -q ...` locally because the current `.venv` can hang during pytest startup on this machine.
