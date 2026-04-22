# 方案 A v0.3debug R2 协议

- document_id: arch_plan_a_v0_3debug_r2_protocol
- status: baselined
- owner: frcnet_project
- last_updated: 2026-04-22

## 1. 目标

`v0.3debug_r2` 是在 `v0.3debug` 之上对五个活跃问题的收口线:

- `tau` 保留为 proposition diagnostics, 不再进入主 matched benchmark
- 默认主线切到 `balanced` checkpoint, `theory` 保留为伴随诊断线
- 通过 resolved-side 轻量正则减弱 `hard_id` / `ambiguous_id` 的 geometry collapse
- `cohort_occupancy.png` 恢复为真正的二维几何 occupancy 图

## 2. 默认协议

- 训练协议: `configs/protocol/plan_a_v0_3debug_r2_train.yaml`
- 评估协议: `configs/protocol/plan_a_v0_3debug_r2_eval.yaml`
- 训练配置: `configs/train/plan_a_v0_3debug_r2_curriculum.yaml`
- 评估配置: `configs/eval/plan_a_v0_3debug_r2_matched_ambiguous_vs_ood.yaml`
- artifact 配置: `configs/analysis/plan_a_v0_3debug_r2_artifacts.yaml`
- study 配置: `configs/study/plan_a_v0_3debug_r2_study.yaml`

## 3. 训练口径

- `warmup`: `easy_id + unknown_supervision`
  - `unknown_content_entropy_weight = 0.5`
- `main`: 开启全部 trainable cohort
  - `hard_id_label_smoothing`
  - `hard_id_resolution_floor_loss`
  - `hard_id_entropy_ceiling_loss`
  - `ambiguous_entropy_floor_loss`
- `stabilize`: 降低 `weight_id`, 提高 ambiguity / unknown-content 与 resolved-side geometry 约束

默认 selection policy:

- `primary_policy = balanced`
- `companion_policy = theory`
- `eligible_phases = [main, stabilize]`

## 4. 评估口径

主 matched benchmark:

- pair: `(resolution_ratio, content_entropy)`
- weighted pair: `(resolution_ratio, resolution_weighted_content_entropy)`
- scalar: `completion_score_beta_0_1`
- completion scan: `beta = [0.1, 0.25, 0.5, 0.75]`

proposition diagnostics:

- `tau = proposition_truth_ratio`
- `resolution_entropy`
- `ternary_entropy`
- `auxiliary_top1_content_probability`

## 5. 导出口径

训练输出:

- `checkpoint_best.pt` 指向当前 primary policy
- `checkpoint_best_balanced.pt`
- `checkpoint_best_theory.pt`
- `checkpoint_selection_summary.json`

study / experiment 输出:

- 主线 `balanced` 写入 `analysis/` 与 `report/`
- 伴随线 `theory` 写入 `analysis_theory/` 与 `report_theory/`
- aggregate 主表只汇总 primary policy
- policy 对照单独写入:
  - `checkpoint_policy_metrics.csv`
  - `checkpoint_policy_summary.csv`
  - `checkpoint_policy_gap_summary.csv`

## 6. 几何 artifact

- `cohort_occupancy.png` 是 5-panel 二维 occupancy 图
- `cohort_counts.png` 单独表达 cohort 样本数量

## 7. 兼容性

- 保留 `v0.3debug` 原配置与产物目录, 不覆盖历史结果
- 旧 analysis record 仍可读取, 但 `top1_content_probability` 仅作为 legacy surrogate / auxiliary 字段使用
