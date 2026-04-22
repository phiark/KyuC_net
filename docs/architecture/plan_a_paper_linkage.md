# 方案 A 论文连接说明

- document_id: arch_plan_a_paper_linkage
- status: baselined
- owner: frcnet_project
- last_updated: 2026-04-22

## 1. 目标

本文件把原方案 A 的论文主张映射到仓库中的协议、记录和产物，避免“论文 section / 实验协议 / 代码字段 / 输出文件”四套口径继续分离。

## 2. 变量映射

| paper_symbol | paper_meaning | repo_field |
| --- | --- | --- |
| `r` | resolution ratio | `resolution_ratio` |
| `u` | explicit unknown mass | `unknown_mass` |
| `tau` | proposition truth ratio | `proposition_truth_ratio` |
| `H_cont` | content entropy in resolved subspace | `content_entropy` |
| `H_res` | binary entropy on resolved vs unknown | `resolution_entropy` |
| `H_3` | ternary entropy on proposition truth / false / unknown | `ternary_entropy` |
| `q_beta` | completion-dependent scalar readout | `completion_score_beta_0_1`, `completion_score_beta_0_5` |

## 3. 论文结论单元到产物单元

### 3.1 几何分区

论文主张:

- easy ID 应集中在高 `resolution_ratio`、低 `content_entropy`
- ambiguous ID 应在中高 `resolution_ratio`、较高 `content_entropy`
- OOD / unknown 应在低 `resolution_ratio`

仓库产物:

- `sample_analysis_records.csv`
- `geometry_scatter.png`
- `geometry_hexbin.png`
- `cohort_occupancy.png`
- `cohort_counts.png`
- `cohort_summary_table.csv`

### 3.2 Completion sensitivity

论文主张:

- 单一样本在不同 `beta` 下的标量排序变化是可观测对象

仓库产物:

- `sample_analysis_records.csv` 中的 `completion_score_beta_0_1`
- `sample_analysis_records.csv` 中的 `completion_score_beta_0_5`

### 3.3 Ambiguous-vs-OOD 判别

论文主张:

- 主 pair 使用 `(resolution_ratio, content_entropy)`
- 主 scalar baseline 使用 `completion_score_beta_0_1`

仓库产物:

- `matched_ambiguous_vs_ood_table.csv`
- `completion_scan_table.csv`

约束:

- `tau = proposition_truth_ratio` 只进入 proposition diagnostics, 不再进入主 matched benchmark 主表

## 4. 记录层

### 4.1 Sample Manifest

作用:

- 约束 cohort 是如何生成的
- 记录 source dataset、source index、augmentation recipe、candidate class set

主文件:

- `plan_a_manifest.jsonl`
- `plan_a_manifest_snapshot.jsonl`

### 4.2 Sample Analysis Record

作用:

- 统一保存每个样本的 paper-facing 指标

主文件:

- `sample_analysis_records.csv`

### 4.2.1 Analysis Export Summary

作用:

- 把 analysis CSV 与 checkpoint、manifest snapshot、proposition view、model config snapshot 绑定成单一 sidecar
- 为 report 阶段提供规范 sidecar 入口, 避免 sibling 猜测

主文件:

- `analysis_summary.json`

### 4.3 Proposition Layer

作用:

- 把 cohort 样本协议派生成命题层 `T / F / U` 视图
- 统一导出 `proposition_truth_mass`、`proposition_false_mass`、`proposition_unknown_mass`
- 恢复论文里的规范 `tau = proposition_truth_ratio`

规则:

- `easy_id / hard_id`: truth target 是真实类
- `ambiguous_id`: truth target 是 `candidate_class_indices`
- `ood / unknown_supervision`: truth target 是空集合, 因此 `truth_mass = 0`, `false_mass = resolution_ratio`
- `is_top1_correct` 只作为附带诊断字段保留, 不再等同于规范 `tau`

主文件:

- `top1_proposition_records.csv`

### 4.4 Auxiliary Diagnostics

作用:

- 保留旧的 top-1 surrogate, 但明确标注为辅助量

主字段:

- `auxiliary_top1_content_probability`

## 5. 默认实验协议

- protocol id: `plan_a_v0_3debug_r2_*`
- default pair: `(resolution_ratio, content_entropy)`
- default scalar: `completion_score_beta_0_1`
- default diagnostic tau: `proposition_truth_ratio`
- default primary checkpoint policy: `balanced`
- default companion checkpoint policy: `theory`
- default matched task: `ambiguous_id vs ood`
- default datasets: `CIFAR-10 + SVHN`
- default ambiguous recipe: `MixUp`
- default hard-ID recipes: `gaussian_blur`, `low_res`

## 6. 追踪链

当前方案 A 的最小追踪链为:

`plan_a_paper_linkage -> plan_a_v0_3debug_r2_*.yaml -> plan_a_manifest.jsonl -> sample_analysis_records.csv + top1_proposition_records.csv + analysis_summary.json -> artifacts -> experiment_record.md`
