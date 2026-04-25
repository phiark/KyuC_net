# Review Record: V4.0 Strict Freeze Local Proof

- date: 2026-04-25
- owner: frcnet_project
- status: review
- scope_control: `records/reviews/2026-04-25_review_v0_4_strict_freeze_scope_control.md`
- proof_root: `artifacts/studies/plan_a_next_v0_4_strict_freeze_local/`

## Proof Inputs

- FRCNet checkpoint: `artifacts/studies/frcnet_results/plan_a_next_v0_2-seed007/training/checkpoints/checkpoint_best.pt`
- final-test manifest: `artifacts/studies/frcnet_results/plan_a_next_v0_2-seed007/final_test_manifest/plan_a_manifest.jsonl`
- protocol config: `configs/protocol/plan_a_next_v0_2_test.yaml`
- strict eval config: `configs/eval/plan_a_next_v0_4_strict_freeze.yaml`
- proof reference config: `configs/reference/plan_a_next_v0_4_softmax_ce_reference_proof.yaml`

## Generated Proof Artifacts

- softmax proof checkpoint: `artifacts/studies/plan_a_next_v0_4_strict_freeze_local/reference/training/checkpoints/checkpoint_best.pt`
- softmax proof summary: `artifacts/studies/plan_a_next_v0_4_strict_freeze_local/reference/training/records/train_summary.json`
- reference scores: `artifacts/studies/plan_a_next_v0_4_strict_freeze_local/reference/final_test_scores/reference_score_records.jsonl`
- frozen manifest: `artifacts/studies/plan_a_next_v0_4_strict_freeze_local/shared/matched_manifest/frozen_matched_manifest.jsonl`
- bin diagnostics: `artifacts/studies/plan_a_next_v0_4_strict_freeze_local/shared/matched_manifest/bin_diagnostics.csv`
- local FRCNet final-test analysis: `artifacts/studies/plan_a_next_v0_4_strict_freeze_local/runs/plan_a_next_v0_2-seed007/analysis_final_test/analysis_summary.json`
- frozen report: `artifacts/studies/plan_a_next_v0_4_strict_freeze_local/runs/plan_a_next_v0_2-seed007/report_final_test_frozen/experiment_record.md`

## Proof Counts

- softmax reference training: 1 proof epoch, 49,920 samples, top-1 `0.4381`, mean loss `1.5595`
- reference score records: 7,000
- frozen manifest records: 1,440
- matched count per class: 720
- manifest bins: 10
- report integrity overrides: `[]`
- strict report has non-empty `matched_manifest_path`
- local strict proof has no `matched_manifest_path: ''` or `unfrozen` marker under the proof root

## Frozen Benchmark Result

- primary pair: `resolution_ratio__state_weighted_content_entropy`
- secondary pair: `resolution_ratio__state_content_entropy`
- primary scalar: `top1_completion_beta_0_1`
- primary pair AUROC: `0.990355`
- secondary pair AUROC: `0.989047`
- scalar AUROC: `0.985447`
- primary pair minus scalar: `0.004908`

## Interpretation

This proof closes the V4.0 strict-freeze engineering loop for seed007. It does not close the paper-facing evidence gate because it uses a 1-epoch proof reference and one FRCNet seed. The remaining paper-facing requirements are frozen reference training at the intended epoch budget, seeds `[17, 27]`, aggregate statistics, proposition repair, held-out source checks, and decision-regret.

## Artifact Governance

The generated proof artifacts are local evidence. Do not stage checkpoints, large CSVs, plots, or the proof artifact tree in a normal code/doc commit.
