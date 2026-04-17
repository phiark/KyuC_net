# ADR-0001 Document-Driven Baseline

- adr_id: ADR-0001
- status: accepted
- date: 2026-04-18
- decision_makers: ["project_owner"]

## Context

FRCNet 项目尚未开始实现, 但上游论文与实验设计已经给出了稳定的理论轴:

- explicit unknown 是主状态, 不是附加解释
- `resolution_ratio` 与 `content_distribution` 必须结构性解耦
- `completion_score` 只是读出策略, 不能替代主状态

如果没有统一的文档树、变量命名和记录机制, 后续很容易出现论文符号、代码实现、分析列名三套口径脱节。

## Decision

项目采用 document-driven baseline:

1. 规范性内容先写入 `docs/`
2. 证据性记录写入 `records/`
3. 实现层统一放在 `src/frcnet/`
4. 变量命名遵循 `resolution_ratio`, `content_distribution`, `unknown_mass`, `content_entropy`, `completion_score` 等规范名
5. 日期与 run_id 使用 ISO 8601 兼容格式
6. 文档分类参考 ISO/IEC/IEEE 15289, 命名语义参考 ISO/IEC 11179

## Consequences

正向影响:

- 后续实现可直接追溯到需求和架构文档
- 论文图表和分析列名更容易统一
- 实验记录具备更好的可复核性

负向影响:

- 初期文档成本增加
- 新增模块前需要先补 requirement 或 ADR

后续动作:

- 补齐基础配置文件
- 落地 `frcnet_model.py` 与 `losses.py`
- 建立首批 contract tests

## Traceability

- linked_requirements: `REQ-FN-001`, `REQ-FN-003`, `REQ-FN-016`, `REQ-FN-018`
- affected_documents:
  - `docs/governance/naming_and_identifier_standard.md`
  - `docs/requirements/system_requirements_specification.md`
  - `docs/architecture/architecture_description.md`
- affected_modules:
  - `src/frcnet/models/`
  - `src/frcnet/training/`
  - `src/frcnet/evaluation/`

