# 文档索引

- document_id: docs_index
- status: baselined
- owner: frcnet_project
- last_updated: 2026-04-27
- standard_alignment: iso_8601, iso_iec_ieee_15289, iso_iec_11179_inspired

## 1. 目的

本文件是 FRCNet 项目的总入口。仓库采用文档驱动流程，规范性内容放在 `docs/`，证据性内容放在 `records/`，生成性内容放在 `artifacts/`。

## 2. 上游依据

当前仓库基线来自以下两份外部源文档:

1. reset manuscript: explicit unknown, resolution ratio, completion projection, corrected EDL audit
2. FRCNet experiment design note: `p_k = r * c_k`, `u = 1 - r`, ambiguity supervision, matched pair analysis

## 3. 文档地图

### 3.1 Governance

- [文档控制](governance/document_control.md)
- [命名与标识标准](governance/naming_and_identifier_standard.md)
- [项目封存状态](governance/project_archive_status.md)

### 3.2 Requirements

- [系统需求规格说明](requirements/system_requirements_specification.md)

### 3.3 Architecture

- [架构说明](architecture/architecture_description.md)
- [方案 A 论文连接说明](architecture/plan_a_paper_linkage.md)
- [Plan A Next v0.2 协议](architecture/plan_a_next_v0_2_protocol.md)
- [Plan A Next v0.4 semantic repair plan](architecture/plan_a_next_v0_4_semantic_repair_plan.md)
- [Plan A Next v0.5 evidence repair protocol](architecture/plan_a_next_v0_5_evidence_repair_protocol.md)
- [Plan A Next v0.6 multisource LOSO protocol](architecture/plan_a_next_v0_6_multisource_loso_protocol.md)
- [Plan A Next v0.6B source-invariant protocol](architecture/plan_a_next_v0_6b_source_invariant_protocol.md)
- [Plan A Next v0.6C near-OOD split repair protocol](architecture/plan_a_next_v0_6c_near_ood_split_repair_protocol.md)
- [方案 A v0.3 研究协议](architecture/plan_a_v0_3_protocol.md)
- [方案 A v0.3debug 理论对齐协议](architecture/plan_a_v0_3debug_protocol.md)
- [方案 A v0.3debug R2 协议](architecture/plan_a_v0_3debug_r2_protocol.md)
- [运行环境矩阵](architecture/runtime_environment_matrix.md)
- [项目结构说明](architecture/project_structure.md)

### 3.4 Verification

- [验证与确认计划](verification/verification_and_validation_plan.md)

### 3.5 Records

- [证据与追踪策略](records/evidence_and_traceability_policy.md)
- [ADR-0001 文档驱动基线](../records/decisions/adr_0001_document_driven_baseline.md)
- [ADR-0002 方案 A 协议基线](../records/decisions/adr_0002_plan_a_protocol_baseline.md)
- [ADR-0003 v0.3 研究基线](../records/decisions/adr_0003_v0_3_research_baseline.md)
- [ADR-0004 v0.3debug 理论对齐修复](../records/decisions/adr_0004_v0_3debug_theory_alignment_repair.md)
- [ADR-0005 v0.3debug R2 benchmark 与 geometry 修复](../records/decisions/adr_0005_v0_3debug_r2_benchmark_and_geometry_repair.md)
- [ADR-0006 Plan A Next v0.2 data semantic baseline](../records/decisions/adr_0006_plan_a_next_v0_2_data_semantic_baseline.md)
- [ADR-0007 Plan A Next v0.5 evidence repair](../records/decisions/adr_0007_plan_a_next_v0_5_evidence_repair.md)
- [ADR-0008 Plan A Next v0.6 multisource LOSO](../records/decisions/adr_0008_plan_a_next_v0_6_multisource_loso.md)
- [ADR-0009 Plan A Next v0.6B source-invariant repair](../records/decisions/adr_0009_plan_a_next_v0_6b_source_invariant.md)
- [ADR-0010 Plan A Next v0.6C near-OOD split repair](../records/decisions/adr_0010_plan_a_next_v0_6c_near_ood_split_repair.md)
- [2026-04-24 Plan A Next v0.2 issue ledger](../records/reviews/2026-04-24_review_plan_a_next_v0_2_issue_ledger.md)
- [2026-04-25 Literature and project alignment review](../records/reviews/2026-04-25_review_literature_project_alignment.md)
- [2026-04-25 V4.0 strict freeze scope control](../records/reviews/2026-04-25_review_v0_4_strict_freeze_scope_control.md)
- [2026-04-25 V4.0 strict freeze local proof](../records/reviews/2026-04-25_review_v0_4_strict_freeze_local_proof.md)
- [2026-04-26 V0.5 evidence repair scope](../records/reviews/2026-04-26_review_v0_5_evidence_repair_scope.md)
- [2026-04-26 V0.6 multisource LOSO scope](../records/reviews/2026-04-26_review_v0_6_multisource_loso_scope.md)
- [2026-04-26 V0.6B source-invariant scope](../records/reviews/2026-04-26_review_v0_6b_source_invariant_scope.md)
- [2026-04-26 V0.6C near-OOD split repair scope](../records/reviews/2026-04-26_review_v0_6c_near_ood_split_repair_scope.md)
- [2026-04-27 V0.6C archive closure](../records/reviews/2026-04-27_review_v0_6c_archive_closure.md)

### 3.6 Templates

- [ADR 模板](templates/architecture_decision_record_template.md)
- [实验记录模板](templates/experiment_record_template.md)

## 4. 文档状态规则

- `draft`: 允许快速修改, 尚未形成项目基线
- `review`: 等待技术或实验负责人确认
- `baselined`: 作为当前开发基线
- `superseded`: 已被新文档替代, 仅保留追溯价值

## 5. 当前基线结论

- 当前项目封存为 `FRCNet 0.6.0 / V0.6C clean partial evidence archive`
- V0.6C 是 clean partial repair, 不是 strong paper-facing final result
- 后续训练、架构变化或科学主张扩展必须先建立新的版本计划
- 所有保留结论必须能追溯到需求、ADR、配置、运行记录和封存记录
