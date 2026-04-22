# ADR-0005 v0.3debug R2 Benchmark And Geometry Repair

- adr_id: ADR-0005
- status: accepted
- date: 2026-04-22
- decision_makers: ["project_owner"]

## Context

`v0.3debug` 已经显著修复了 unknown-side collapse, 但仍存在五个活跃问题:

1. `tau` 被混入主 matched benchmark, 容易被误读为公平 baseline
2. `best_checkpoint_path` 默认指向 theory-best, 而且 warmup checkpoint 仍可获选
3. `hard_id` 几何仍过于松散, seed 敏感性偏高
4. resolved-side 内部厚度仍偏大, ambiguous 中间带不够稳定
5. `cohort_occupancy.png` 实际只是 cohort count 图, 不是真正的 occupancy 图

## Decision

冻结 `v0.3debug_r2` 的口径如下:

1. `tau = proposition_truth_ratio` 仅作为 proposition diagnostics 输出
2. 主 matched benchmark 只保留 `pair / weighted_pair / scalar / completion scan`
3. 默认 primary checkpoint policy 切到 `balanced`, `theory` 保留伴随导出
4. `theory` 与 `balanced` 都禁止在 `warmup` 阶段获选
5. 在现有双头结构上增加三个轻量正则:
   - `hard_id_resolution_floor_loss`
   - `hard_id_entropy_ceiling_loss`
   - `ambiguous_entropy_floor_loss`
6. `cohort_occupancy.png` 恢复为二维几何 occupancy 图, `cohort_counts.png` 独立导出

## Consequences

正向影响:

- 主 benchmark 语义更清晰
- primary result 与 companion diagnostics 分层明确
- `hard_id` 与 `ambiguous_id` 的几何约束更完整
- 报告 artifact 名称与内容重新对齐

负向影响:

- study / aggregate schema 再次扩展
- 默认 CLI 入口会切到新的 R2 study 配置
- 与 `v0.3debug` 的主表字段不能直接逐列对齐

## Traceability

- linked_requirements:
  - `REQ-FN-013`
  - `REQ-FN-026`
  - `REQ-FN-027`
  - `REQ-FN-028`
  - `REQ-FN-029`
  - `REQ-FN-031`
  - `REQ-FN-032`
  - `REQ-FN-033`
  - `REQ-FN-034`
- affected_documents:
  - `README.md`
  - `docs/governance/naming_and_identifier_standard.md`
  - `docs/architecture/plan_a_paper_linkage.md`
  - `docs/architecture/plan_a_v0_3debug_protocol.md`
  - `docs/architecture/plan_a_v0_3debug_r2_protocol.md`
  - `docs/requirements/system_requirements_specification.md`
  - `docs/verification/verification_and_validation_plan.md`
- affected_modules:
  - `src/frcnet/training/`
  - `src/frcnet/evaluation/`
  - `src/frcnet/analysis/`
  - `src/frcnet/workflows/`
