# 架构说明

- document_id: arch_architecture_description
- status: baselined
- owner: frcnet_project
- last_updated: 2026-04-24

## 1. 架构目标

`next-v0.1` 保留 FRCNet 原生模型结构，把复杂度限制在可审计的评估语义层。

模型层:

- `resolution_ratio`
- `content_distribution`
- `class_mass`
- `unknown_mass`

state layer:

- `state_content_entropy`
- `state_weighted_content_entropy`
- `state_entropy`

proposition layer:

- `top1_view_*` label-free
- `target_view_*` / `candidate_view_*` label-aware audit only

completion layer:

- `top1_completion_beta_*`
- legacy `completion_score_beta_*` 只作为兼容 alias

## 2. 模块边界

- `src/frcnet/models`: backbone、resolution head、content head、输出 contract。
- `src/frcnet/training`: 最小训练 step 和 Plan A loss。
- `src/frcnet/data`: cohort manifest、batch contract、dataset adapter。
- `src/frcnet/evaluation`: semantic layers、inference export、matched benchmark、matched manifest、Softmax reference。
- `src/frcnet/analysis`: figures、tables、experiment record。
- `src/frcnet/workflows`: single-run Plan A workflow only。

## 3. 当前保留的 workflow

- build manifest
- train one FRCNet run
- export analysis records
- generate report artifacts
- run single end-to-end experiment bundle

## 4. 当前不保留的 workflow

- study-level multi-seed runner
- aggregate report builder
- dual checkpoint policy comparison
- decision benchmark runner

## 5. 架构约束

- `unknown_mass` 只能由 `1 - resolution_ratio` 导出。
- 主 benchmark 只能读 label-free state / top1-view 字段。
- proposition diagnostics 可以使用 label-aware 信息，但不能进入主 benchmark 排名。
- Softmax CE reference 只能服务 frozen matching，不作为完整 baseline 结论。
- 历史字段必须能读，但新报告优先使用 `state_*` 和 `top1_*` 命名。
