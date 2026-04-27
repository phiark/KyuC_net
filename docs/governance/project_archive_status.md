# Project Archive Status

- document_id: gov_project_archive_status
- status: baselined
- owner: frcnet_project
- last_updated: 2026-04-27
- version: `0.6.0`
- archive_state: `v0_6c_clean_partial_evidence_archive`

## 1. Archive State

FRCNet is archived at **0.6.0 / V0.6C clean partial evidence**.

This archive closes the current evidence-repair line. It does not claim a strong final paper result, and it does not authorize additional training under the same version.

## 2. Allowed Maintenance

Allowed work under this archive state:

- documentation corrections that do not change scientific claims
- reproducibility notes and runbook clarification
- non-behavioral code maintenance
- dependency or runtime notes
- artifact hygiene and checkpoint cleanup using the documented retention rule

## 3. Disallowed Without New Version Plan

The following require a new explicit version plan:

- model architecture changes
- new training runs intended as new evidence
- new datasets or new OOD claims
- changes to release gates or scientific interpretation
- paper-facing claim expansion beyond clean partial V0.6C evidence

## 4. Evidence Grade

Archived evidence grade: **clean partial repair**.

Key V0.6C aggregate metrics:

- unseen CIFAR100 held-out classes pair AUROC: `0.6512 +/- 0.0112`
- all-OOD pair AUROC: `0.8019 +/- 0.0076`
- worst-source pair AUROC: `0.6435 +/- 0.0214`
- seen-unseen gap: `0.2178 +/- 0.0020`

## 5. Artifact Policy

Generated artifacts are not the source of truth. The source of truth is:

`docs -> records -> configs -> source -> tests -> compact evidence records`

Normal commits must not stage:

- checkpoints
- generated study trees
- large analysis CSVs
- generated plots
- runtime caches

## 6. Checkpoint Retention

Local archive cleanup keeps only:

- `checkpoint_best*`
- `checkpoint_last*`

All other generated checkpoint files are disposable unless a future version plan explicitly defines a different retention rule.
