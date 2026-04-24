# Review Record: Plan A Next v0.2 Issue Ledger

- date: 2026-04-24
- owner: frcnet_project
- status: baselined
- linked_adrs:
  - ADR-0006

## Data And Evidence Issues

1. Artifact/code naming drift: latest KYUC artifacts use state/top1 naming while the repo mainline used legacy names.
2. `plan_a_next_v0_1` was a 5-epoch, single-seed smoke run; it is not final evidence.
3. Large analysis used 6600 samples but still had no frozen matched manifest path.
4. Primary scalar whitelist allowed label-aware proposition fields.
5. SVHN is used for both unknown supervision and OOD-like evaluation, so the result is seen-source unless future protocols add held-out sources.
6. Ambiguous supervision is mostly MixUp over fixed class pairs, so held-out recipe and class-pair tests are required.
7. hard-ID recipes must come from config, not hidden code constants.
8. Artifact provenance paths differ between transferred remote outputs and the local `artifacts/studies` layout.

## Model Understanding Issues

1. The gate is strong, but gate separation alone does not prove resolved-side content quality.
2. Pair AUROC beat low-beta scalar in the latest large audit, but multi-seed and softmax CE reference evidence are missing.
3. easy/hard top-1 remains weak; the model often resolves confidently into the wrong class.
4. hard-ID is the main cohort to inspect before claiming robust resolved-side structure.
5. High-beta completion can reverse the ranking and must remain a policy diagnostic.
6. `proposition_truth_ratio = 1.0` is label-aware diagnostic success, not a fair benchmark result.
7. Decision-regret evidence is still absent and belongs to a later protocol.
