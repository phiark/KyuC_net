# 方案 A v0.3 研究协议

- document_id: arch_plan_a_v0_3_protocol
- status: baselined
- owner: frcnet_project
- last_updated: 2026-04-19

## 1. 目标

本协议定义 FRCNet `v0.3` 的正式研究口径。该版本继续以原生 Plan A 为主线, 但把交付单位从单次 run 升级为 study 级结果。

`v0.3debug` 的理论对齐修复协议见 `docs/architecture/plan_a_v0_3debug_protocol.md`。

## 2. 默认主协议

- 训练协议: `configs/protocol/plan_a_v0_3_train.yaml`
- 评估协议: `configs/protocol/plan_a_v0_3_eval.yaml`
- 训练配置: `configs/train/plan_a_v0_3_curriculum.yaml`
- study 配置: `configs/study/plan_a_v0_3_study.yaml`

## 3. 训练口径

- `warmup` 5 epoch: `easy_id + unknown_supervision`
- `main` 25 epoch: `easy_id + hard_id + ambiguous_id + unknown_supervision`
- `stabilize` 10 epoch: 全 cohort, 低学习率稳定
- best checkpoint 由 validation manifest 上的 `pair_auroc` 决定
- tie-breaker:
  1. `easy_id_top1_accuracy`
  2. 更低 `mean_loss_total`

## 4. 评估口径

- 主 pair: `(resolution_ratio, content_entropy)`
- 强制次级 pair: `(resolution_ratio, resolution_weighted_content_entropy)`
- 主 scalar: `completion_score_beta_0_1`
- 强制附带:
  - `top1_content_probability` (`v0.3` 历史口径)
  - `completion_score_beta_0_25`
  - `completion_score_beta_0_5`
  - `completion_score_beta_0_75`

## 5. Study 口径

- 默认 seeds: `[7, 17, 27]`
- study 期间固定一份 evaluation manifest, 全部 seed 复用
- 正式输出:
  - per-seed key metrics
  - mean/std aggregate summary
  - best/worst/median seed ranking
  - aggregate experiment record

## 6. Ablation

`v0.3` 保留一个扩展 recipe 协议:

- `configs/protocol/plan_a_v0_3_train_augmented.yaml`
- `overlay_enabled = true`
- `occlusion_enabled = true`

该协议用于方法 ablation, 不作为默认主结果。
