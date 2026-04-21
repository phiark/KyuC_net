# 文档索引

- document_id: docs_index
- status: baselined
- owner: frcnet_project
- last_updated: 2026-04-18
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

### 3.2 Requirements

- [系统需求规格说明](requirements/system_requirements_specification.md)

### 3.3 Architecture

- [架构说明](architecture/architecture_description.md)
- [方案 A 论文连接说明](architecture/plan_a_paper_linkage.md)
- [方案 A v0.3 研究协议](architecture/plan_a_v0_3_protocol.md)
- [运行环境矩阵](architecture/runtime_environment_matrix.md)
- [项目结构说明](architecture/project_structure.md)

### 3.4 Verification

- [验证与确认计划](verification/verification_and_validation_plan.md)

### 3.5 Records

- [证据与追踪策略](records/evidence_and_traceability_policy.md)
- [ADR-0001 文档驱动基线](../records/decisions/adr_0001_document_driven_baseline.md)
- [ADR-0002 方案 A 协议基线](../records/decisions/adr_0002_plan_a_protocol_baseline.md)
- [ADR-0003 v0.3 研究基线](../records/decisions/adr_0003_v0_3_research_baseline.md)

### 3.6 Templates

- [ADR 模板](templates/architecture_decision_record_template.md)
- [实验记录模板](templates/experiment_record_template.md)

## 4. 文档状态规则

- `draft`: 允许快速修改, 尚未形成项目基线
- `review`: 等待技术或实验负责人确认
- `baselined`: 作为当前开发基线
- `superseded`: 已被新文档替代, 仅保留追溯价值

## 5. 当前基线结论

- 当前项目优先实现一个原生 explicit-unknown 架构, 而不是 softmax/EDL 的事后解释层
- 代码主命名不直接沿用论文的单字母符号, 而是使用稳定、可追踪的工程名词
- 文件树按文档生命周期、配置、实现、验证、记录分层
- 所有实验结论必须能追溯到需求、ADR、配置和运行记录
