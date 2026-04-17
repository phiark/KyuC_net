# 证据与追踪策略

- document_id: rec_evidence_traceability_policy
- status: baselined
- owner: frcnet_project
- last_updated: 2026-04-18

## 1. 目标

项目必须把“规范是什么”和“实际发生了什么”区分开来。`docs/` 提供规范, `records/` 提供证据。

## 2. 记录类型

### 2.1 ADR

用于记录架构与命名的关键决策, 例如:

- 是否固定 ResNet-18 作为首轮 backbone
- 是否把 weighted pair 作为默认 pair
- 是否引入某种 ambiguous 数据合成方案

### 2.2 Experiment Record

每次实验至少记录:

- 目标与假设
- 关联 requirement / ADR
- 配置文件路径
- 数据说明
- run_id
- 关键结果
- 偏差与问题

### 2.3 Review Record

用于记录代码审查、实验审查、论文图表审查等活动。

## 3. 最小追踪链

任何结论都应该能回溯到:

`requirement -> architecture/adr -> config -> run -> record -> artifact`

如果链条中断, 结论只能视为临时观察, 不能进入论文主结论。

## 4. 命名规则

- ADR: `adr_0001_short_title.md`
- experiment record: `YYYY-MM-DD_experiment_short_name.md`
- review record: `YYYY-MM-DD_review_short_name.md`

## 5. Artifact 关联

实验记录中引用 artifact 时, 必须注明:

- artifact 相对路径
- 生成脚本或生成命令
- 对应 run_id

