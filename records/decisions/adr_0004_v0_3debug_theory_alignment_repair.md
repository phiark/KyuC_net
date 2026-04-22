# ADR-0004 v0.3debug Theory-Alignment Repair

- adr_id: ADR-0004
- status: accepted
- date: 2026-04-21
- decision_makers: ["project_owner"]

## Context

`v0.3` 的研究结果证明了主 pair 在 `ambiguous_id vs ood` 上稳定有效, 但同时暴露出三个问题:

1. 后期训练会把 content 几何拉尖, 导致 `last checkpoint` 比 `best checkpoint` 的 pair 显著退化
2. `unknown_supervision / ood` 的 content head 缺少显式约束, 会在后期出现异常尖化
3. 仓库中的 `tau` 口径仍过度依赖 top-1 surrogate, 不足以承接论文里的 proposition-level 叙事

## Decision

冻结 `v0.3debug` 的修复口径如下:

1. 保持原生 `ResNet-18 + resolution head + content head` 架构
2. 在 `unknown_supervision` 上新增 content neutrality regularizer
3. 对 `hard_id` 引入 label smoothing, 不引入 teacher distillation
4. 保留 `theory-best` checkpoint, 同时新增 `balanced-best` 诊断 checkpoint
5. 规范 `tau` 切换为 `proposition_truth_ratio`
6. 保留 `auxiliary_top1_content_probability` 作为辅助量
7. 只预留 `softmax_ce` 的 family 接口, 本轮不实现 baseline 训练链

## Consequences

正向影响:

- 训练目标与理论几何更对齐
- report 能直接输出 proposition-level 证据
- future baseline 接口一次定死, 避免后续 aggregate 返工

负向影响:

- record / artifact schema 会发生破坏式变化
- `tau` 的历史结果与新口径不能直接横向比较
- report 和 study 的复杂度继续上升

## Traceability

- linked_requirements:
  - `REQ-FN-029`
  - `REQ-FN-030`
  - `REQ-FN-031`
  - `REQ-FN-032`
- affected_documents:
  - `docs/governance/naming_and_identifier_standard.md`
  - `docs/architecture/plan_a_paper_linkage.md`
  - `docs/architecture/plan_a_v0_3debug_protocol.md`
  - `docs/requirements/system_requirements_specification.md`
  - `docs/verification/verification_and_validation_plan.md`
- affected_modules:
  - `src/frcnet/training/`
  - `src/frcnet/evaluation/`
  - `src/frcnet/analysis/`
  - `src/frcnet/workflows/`
