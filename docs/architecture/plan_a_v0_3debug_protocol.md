# 方案 A v0.3debug 理论对齐协议

- document_id: arch_plan_a_v0_3debug_protocol
- status: baselined
- owner: frcnet_project
- last_updated: 2026-04-22

## 1. 目标

`v0.3debug` 是 `v0.3` 之上的理论对齐修复线。它保持原生 FRCNet 双头结构不变, 但补上:

- `unknown_supervision` 上的 content neutrality regularizer
- `hard_id` 的轻量 anti-collapse label smoothing
- dual checkpoint selection: `theory-best` 与 `balanced-best`
- proposition layer 的规范 `tau / H_res / H_3` 输出

`v0.3debug` 的后续 benchmark 语义修复与 geometry R2 收口线见 `docs/architecture/plan_a_v0_3debug_r2_protocol.md`。

## 2. 默认主协议

- 训练协议: `configs/protocol/plan_a_v0_3debug_train.yaml`
- 评估协议: `configs/protocol/plan_a_v0_3debug_eval.yaml`
- 训练配置: `configs/train/plan_a_v0_3debug_curriculum.yaml`
- study 配置: `configs/study/plan_a_v0_3debug_study.yaml`

## 3. 训练口径

- `warmup` 5 epoch: `easy_id + unknown_supervision`, 打开 `unknown_content_entropy_weight`
- `main` 25 epoch: 全 trainable cohort, 打开 `hard_id_label_smoothing`
- `stabilize` 10 epoch: 降低 `weight_id`, 提高 ambiguity / unknown-content 权重

默认 checkpoint policy:

- `checkpoint_best_theory.pt`
  - `pair_auroc -> easy_id_top1_accuracy -> lower train mean_loss_total`
- `checkpoint_best_balanced.pt`
  - `balanced_score = mean(pair_auroc, easy_id_top1_accuracy, hard_id_top1_accuracy, ambiguous_candidate_hit_rate)`
  - tie-breaker: 更高 `pair_auroc`, 再看更低 train loss

## 4. 评估口径

- 主 pair: `(resolution_ratio, content_entropy)`
- 次级 pair: `(resolution_ratio, resolution_weighted_content_entropy)`
- 主 scalar: `completion_score_beta_0_1`
- 规范 `tau`: `proposition_truth_ratio`
- proposition diagnostics:
  - `proposition_truth_ratio`
  - `resolution_entropy`
  - `ternary_entropy`
  - `auxiliary_top1_content_probability`

## 5. 正式产物

- `sample_analysis_records.csv`
- `top1_proposition_records.csv`
- `matched_ambiguous_vs_ood_table.csv`
- `proposition_diagnostic_table.csv`
- `proposition_tau_roc_curve.png`
- `checkpoint_selection_summary.json`
- `balanced_vs_theory_checkpoint_table.csv`

## 6. Baseline 预留

- 当前唯一可运行 `model_family`: `frcnet_explicit_unknown`
- 预留合法 family 名: `softmax_ce`
- 本协议不实现 baseline 训练链, 只保证 study / aggregate 表结构可承接多 family
