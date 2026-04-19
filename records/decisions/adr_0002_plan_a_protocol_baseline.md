# ADR-0002 Plan A Protocol Baseline

- adr_id: ADR-0002
- status: accepted
- date: 2026-04-18
- decision_makers: ["project_owner"]

## Context

FRCNet 0.1 已经有训练内核，但仓库仍缺少与论文直接对接的实验协议层。没有固定协议，后续图表、matched benchmark 和 experiment record 都会失去可比性。

## Decision

冻结下一阶段的默认研究口径如下:

1. 默认协议使用 `plan_a_v1`
2. 默认数据主线使用 `CIFAR-10 + SVHN`
3. 默认主 pair 使用 `(resolution_ratio, content_entropy)`
4. 默认主 scalar baseline 使用 `completion_score_beta_0_1`
5. 默认 matched task 使用 `ambiguous_id vs ood`
6. 默认 ambiguous recipe 仅启用 `MixUp`
7. 默认 hard-ID recipe 仅启用 `gaussian_blur` 和 `low_res`

## Consequences

正向影响:

- 论文主张、配置和 artifact 输出将共享同一协议标识
- matched benchmark 将有明确的 primary pair 与 scalar baseline
- experiment record 可以稳定回链到 protocol / manifest / artifacts

负向影响:

- overlay、occlusion、noise、texture 等变体被延后
- weighted pair 暂不作为默认主指标

后续动作:

- 补齐 manifest / analysis / artifact 链
- 将 review 阻塞项作为实验门槛修复
- 在协议稳定后再扩展额外 cohort 变体

## Traceability

- linked_requirements: `REQ-FN-012`, `REQ-FN-013`, `REQ-FN-014`, `REQ-FN-015`
- affected_documents:
  - `docs/architecture/plan_a_paper_linkage.md`
  - `configs/protocol/plan_a_v1.yaml`
  - `configs/eval/plan_a_matched_ambiguous_vs_ood.yaml`
- affected_modules:
  - `src/frcnet/data/`
  - `src/frcnet/evaluation/`
  - `src/frcnet/analysis/`

