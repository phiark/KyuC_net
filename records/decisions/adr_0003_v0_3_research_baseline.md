# ADR-0003 v0.3 Research Baseline

- adr_id: ADR-0003
- status: accepted
- date: 2026-04-19
- decision_makers: ["project_owner"]

## Context

FRCNet 0.2 候选已经具备单 run 的训练、analysis 和 artifact 链, 但还不满足 `v0.3` 所需的研究交付标准:

1. 训练协议仍是单阶段、短 epoch
2. best checkpoint 仍由 train loss 主导
3. study 级多 seed 聚合尚未形成正式入口
4. weighted pair 与 completion beta scan 尚未成为正式报告对象

## Decision

冻结 `v0.3` 的默认研究口径如下:

1. 默认主线继续使用原生 FRCNet / Plan A
2. 默认研究交付单位从单次 run 升级为 study
3. 默认训练采用 `warmup -> main -> stabilize` 三阶段 curriculum
4. 默认 best checkpoint 使用 validation pair AUROC 选择
5. 默认 study 使用固定 evaluation manifest 和 seeds `[7, 17, 27]`
6. 默认主 pair 保持 `(resolution_ratio, content_entropy)`
7. 强制附带 weighted pair、tau scalar 和 completion beta scan 输出

## Consequences

正向影响:

- `v0.3` 结果包直接具备论文表格和图表所需的 study 级结构
- checkpoint 选择与评估协议更加一致
- weighted pair 与 beta scan 进入正式追踪链

负向影响:

- 训练和测试耗时明显高于 `v0.2`
- repo 暂不引入完整 baseline 矩阵
- study 聚合逻辑增加了额外的配置和记录复杂度

## Traceability

- linked_requirements:
  - `REQ-FN-024`
  - `REQ-FN-025`
  - `REQ-FN-026`
  - `REQ-FN-027`
  - `REQ-FN-028`
- affected_documents:
  - `docs/architecture/plan_a_v0_3_protocol.md`
  - `docs/requirements/system_requirements_specification.md`
  - `docs/verification/verification_and_validation_plan.md`
- affected_modules:
  - `src/frcnet/workflows/`
  - `src/frcnet/evaluation/`
  - `src/frcnet/analysis/`
