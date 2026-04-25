# 文档索引

- document_id: docs_index
- status: baselined
- owner: frcnet_project
- last_updated: 2026-04-25

## 当前基线

本分支从 `next-v0.1` 语义拆层推进到 `plan_a_next_v0_2`，当前实现切片为 `plan_a_next_v0_4_strict_freeze`。V4.0 第一轮只锁定 strict frozen evidence：softmax CE same-backbone reference、frozen matched manifest、同一 frozen subset 下的 pair/scalar/completion scan 报告。`v0.3*` debug 协议仍不作为当前主线。

## 文档地图

- [文档控制](governance/document_control.md)
- [命名与标识标准](governance/naming_and_identifier_standard.md)
- [系统需求规格说明](requirements/system_requirements_specification.md)
- [架构说明](architecture/architecture_description.md)
- [方案 A next-v0.1 协议](architecture/plan_a_next_v0_1_protocol.md)
- [方案 A next-v0.2 协议](architecture/plan_a_next_v0_2_protocol.md)
- [方案 A next-v0.4 语义修复计划](architecture/plan_a_next_v0_4_semantic_repair_plan.md)
- [方案 A 论文连接说明](architecture/plan_a_paper_linkage.md)
- [运行环境矩阵](architecture/runtime_environment_matrix.md)
- [项目结构说明](architecture/project_structure.md)
- [验证与确认计划](verification/verification_and_validation_plan.md)
- [FRCNet next-v0.1 分阶段开发计划](roadmap/frcnet_next_v0_1_phased_development.md)
- [证据与追踪策略](records/evidence_and_traceability_policy.md)
- [ADR-0001 文档驱动基线](../records/decisions/adr_0001_document_driven_baseline.md)
- [ADR-0006 next-v0.1 语义拆层与公平 matched benchmark](../records/decisions/adr_0006_next_v0_1_semantic_repair.md)
- [ADR-0006 plan_a_next_v0_2 数据语义基线](../records/decisions/adr_0006_plan_a_next_v0_2_data_semantic_baseline.md)
- [v0.2 issue ledger](../records/reviews/2026-04-24_review_plan_a_next_v0_2_issue_ledger.md)
- [文献与项目对齐审计](../records/reviews/2026-04-25_review_literature_project_alignment.md)
- [V4.0 strict-freeze scope control](../records/reviews/2026-04-25_review_v0_4_strict_freeze_scope_control.md)
- [V4.0 strict-freeze local proof](../records/reviews/2026-04-25_review_v0_4_strict_freeze_local_proof.md)

## 状态规则

- `active`: 当前开发口径
- `legacy`: 仅用于读取或解释历史记录
- `deferred`: 明确不在本分支实现
