# V0.6C Archive Closure Review

- document_id: review_v0_6c_archive_closure
- status: baselined
- owner: frcnet_project
- review_date: 2026-04-27
- version: `0.6.0`
- study_id: `plan_a_next_v0_6c_near_ood_cifar100_class_holdout`
- branch: `codex/plan-a-next-v0-6c-near-ood-split-repair`

## 1. Archive Decision

The project is frozen as **FRCNet 0.6.0 / V0.6C clean partial evidence**. This is a valid archive point for the current evidence-repair line, but it is not a strong paper-facing final result.

No further model training, architecture changes, or scientific-claim expansion is part of this archive closure. Future work must start from an explicit new version plan.

## 2. Final Evidence State

V0.6C repaired the near-OOD failure enough to pass the clean partial gate:

- unseen CIFAR100 held-out classes pair AUROC: `0.6512 +/- 0.0112`
- all-OOD pair AUROC: `0.8019 +/- 0.0076`
- worst-source pair AUROC: `0.6435 +/- 0.0214`
- seen-unseen gap: `0.2178 +/- 0.0020`
- hard ID top-1: `0.6333 +/- 0.0120`
- ambiguous candidate hit: `0.7167 +/- 0.0127`
- pair-scalar delta: `0.0030 +/- 0.0032`

Interpretation:

- This is **clean partial repair**, not strong repair.
- CIFAR100 held-out class generalization improved over V0.5 and V0.6B, but remains below the strong target.
- The pair-vs-scalar margin is too small to support a strong FRCNet-specific advantage claim.
- `balanced` checkpoint policy outperformed the configured primary `near_ood_balanced` policy on unseen CIFAR100, so checkpoint policy is a known archived limitation.

## 3. Source Integrity

The V0.6C split audit passed:

- train/validation source fingerprint overlap: `0`
- train/final source fingerprint overlap: `0`
- validation/final source fingerprint overlap: `0`
- CIFAR100 seen classes: `[0, 50)`
- CIFAR100 held-out classes: `[50, 100)`
- final held-out role: `unseen_ood_classes`

This supports treating the V0.6C result as a clean class-holdout evidence point.

## 4. Checkpoint Cleanup

Cleanup scope was local only: `/Users/zero_lab/Desktop/HardMin`.

Retention rule:

- keep `checkpoint_best*.pt`, `checkpoint_best*.pth`, `checkpoint_best*.ckpt`, `checkpoint_best*.safetensors`
- keep `checkpoint_last.pt`, `checkpoint_last.pth`, `checkpoint_last.ckpt`, `checkpoint_last.safetensors`
- delete all other model checkpoint files under `artifacts/` and `records/`

Before cleanup:

- model files: `319`
- total model size: `39.82 GiB`
- retained best/last files: `29`, `3.54 GiB`
- delete candidates: `290`, `36.28 GiB`
- delete-list best/last name violations: `0`

After cleanup:

- deleted files: `290`
- deleted size: `36.28 GiB`
- remaining model files: `29`, `3.54 GiB`
- remaining non-best/last model files: `0`

Cleanup manifests:

- delete manifest: `records/reviews/2026-04-27_v0_6c_checkpoint_cleanup_delete_manifest.csv`
- pre-cleanup keep manifest: `records/reviews/2026-04-27_v0_6c_checkpoint_cleanup_pre_keep_manifest.csv`
- retained manifest: `records/reviews/2026-04-27_v0_6c_checkpoint_cleanup_retained_manifest.csv`

## 5. Verification

Commands run after cleanup:

```bash
.venv313/bin/python -m pytest -q
```

Result:

```text
99 passed in 41.81s
```

Follow-up V0.6C archive maintenance validation after version/doc/gitignore hardening:

```text
108 passed in 43.39s
```

Git/artifact hygiene checks:

- `git ls-files -ci --exclude-standard` returned no tracked ignored artifacts.
- `git status --short --branch` showed only archive review records before staging.
- No checkpoint, generated plot, large CSV, or artifact tree was staged as part of this closure.

## 6. Archived Limitations

The following limitations remain intentionally unresolved in this archive point:

- V0.6C is not a strong paper result.
- Hard resolver quality remains limited.
- Ambiguous hit rate remains below the strong target.
- Pair-scalar delta remains small.
- `near_ood_balanced` is not the best observed checkpoint policy.
- ARC-side artifacts were not cleaned in this local-only archive task.
