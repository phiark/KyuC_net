# 方案 A 论文连接说明

- document_id: arch_plan_a_paper_linkage
- status: baselined
- owner: frcnet_project
- last_updated: 2026-04-24

## 1. 目标

本文件把论文变量映射到 `plan_a_next_v0_2` 的工程字段，防止 state、proposition、completion 三层再次混用，并明确哪些 evidence 可进入主 benchmark。

## 2. 变量映射

| paper_symbol | meaning | active repo field |
| --- | --- | --- |
| `r` | resolution ratio | `resolution_ratio` |
| `u` | explicit unknown mass | `unknown_mass` |
| `H_K(c)` | K-class state entropy | `state_content_entropy` |
| `r H_K(c)` | resolution-weighted state entropy | `state_weighted_content_entropy` |
| `tau_A` | proposition truth ratio for view A | `top1_view_tau`, `target_view_tau`, `candidate_view_tau` |
| `q_beta(A)` | completion readout under view A | `top1_completion_beta_*` |

## 3. 主 benchmark

默认 matched task:

- positive: `ambiguous_id`
- negative: `ood`

默认 features:

- raw pair: `resolution_ratio__state_content_entropy`
- primary pair: `resolution_ratio__state_weighted_content_entropy`
- scalar: `top1_completion_beta_0_1`

主 benchmark 不允许:

- `class_label`
- `candidate_class_indices`
- `target_view_*`
- `candidate_view_*`
- `proposition_truth_ratio`

## 4. 追踪链

当前最小追踪链:

`plan_a_next_v0_2_{train,validation,test}.yaml -> plan_a_manifest.jsonl -> sample_analysis_records.csv -> top1_proposition_records.csv / proposition_view_records.csv -> analysis_summary.json -> frozen matched manifest -> report artifacts`

v0.2 paper-facing claims must use final test output. Validation output is only for checkpoint selection.

## 5. 历史兼容

- `content_entropy` 继续可读，但新语义名是 `state_content_entropy`。
- `resolution_weighted_content_entropy` 继续可读，但新语义名是 `state_weighted_content_entropy`。
- `completion_score_beta_*` 继续可读，但新语义名是 `top1_completion_beta_*`。
- `proposition_truth_ratio` 保留为 legacy diagnostic，不进入主 benchmark。
