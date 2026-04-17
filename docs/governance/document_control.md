# 文档控制

- document_id: gov_document_control
- status: baselined
- owner: frcnet_project
- last_updated: 2026-04-18
- standard_alignment: iso_8601, iso_iec_ieee_15289

## 1. 采用标准与假设

由于用户要求“变量命名、文件树按照 ISO 标准”，而该场景不存在单一覆盖全部问题的 ISO 文件树标准，本项目采用以下组合约束:

1. `ISO 8601` 负责日期、时间、实验时间戳与版本日期
2. `ISO/IEC/IEEE 15289` 负责生命周期信息项分层, 用于组织 requirements, architecture, verification, records
3. `ISO/IEC 11179` 的语义命名思想作为变量与标识符命名参考, 即优先使用稳定的对象名 + 属性名 + 表示名

## 2. 文档分类

### 2.1 Normative Documents

规范性文档定义“应该是什么”, 是开发前必须先确认的内容:

- governance
- requirements
- architecture
- verification

### 2.2 Evidential Records

证据性记录定义“实际发生了什么”, 不直接替代规范:

- adr
- experiment_record
- review_record

### 2.3 Generated Artifacts

生成性产物是由代码输出的派生对象:

- figures
- tables
- reports
- checkpoints
- logs

## 3. 变更控制规则

任何功能开发都应遵循以下顺序:

1. 新需求或需求变更先落在 `docs/requirements/`
2. 影响边界、接口、命名或模块职责的选择, 先登记 ADR
3. 架构文档更新后, 才进入实现
4. 实现完成后, 需要补足验证结果与实验记录

## 4. 追踪标识规则

本项目使用以下统一标识:

- `REQ-xxx`: requirement item
- `ARCH-xxx`: architecture item
- `VER-xxx`: verification item
- `ADR-xxxx`: architecture decision record
- `EXP-YYYY-MM-DD-xx`: experiment record
- `RUN-YYYYMMDDThhmmss+zzzz-xxx`: execution run identifier

## 5. 时间与日期写法

- 文档正文日期使用 ISO 8601 extended 格式: `2026-04-18`
- 文件名中的时间戳使用 ISO 8601 basic 格式: `20260418T093000+0800`
- 所有实验记录必须包含本地时区偏移

## 6. 审批与基线

初始化阶段采用单责任人模式:

- author: 当前项目维护者
- reviewer: 待指派
- approver: 待指派

在 reviewer 和 approver 尚未固定前, `baselined` 表示“当前工程基线”, 不表示正式组织级放行。

