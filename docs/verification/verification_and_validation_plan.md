# 验证与确认计划

- document_id: ver_verification_and_validation_plan
- status: baselined
- owner: frcnet_project
- last_updated: 2026-04-19

## 1. 目标

本计划用于确保 FRCNet 的实现不偏离文档基线, 并且实验结论可被证伪、可被复核。

## 2. 验证层级

### 2.1 Unit Verification

关注数学和函数不变量:

- `VER-UNIT-001`: `class_mass.sum + unknown_mass == 1`
- `VER-UNIT-002`: `unknown_mass == 1 - resolution_ratio`
- `VER-UNIT-003`: `content_distribution.sum == 1`
- `VER-UNIT-004`: `content_entropy` 数值范围正确
- `VER-UNIT-005`: `completion_score` 对不同 beta 的计算一致

### 2.2 Contract Verification

关注输入输出 schema:

- `VER-CON-001`: input batch contract 完整
- `VER-CON-002`: model output contract 字段完整
- `VER-CON-003`: analysis record contract 列名稳定
- `VER-CON-004`: cohort 名称仅来自规范集合
- `VER-CON-005`: manifest contract 保证非空、单一 `protocol_id` 与唯一 `sample_id`
- `VER-CON-006`: analysis/report bundle contract 保证 run/protocol 与 sample set 一致

### 2.3 Integration Verification

关注流程拼装:

- `VER-INT-001`: 数据加载到模型前向链路可执行
- `VER-INT-002`: 训练 step 能同时处理 known, unknown, ambiguous 样本
- `VER-INT-003`: 评估流程能导出 scalar 和 pair 指标
- `VER-INT-004`: 分析流程能生成 scatter、hexbin、occupancy 图
- `VER-INT-005`: 无 checkpoint 的 analysis 导出默认被拒绝, 显式 override 时会留下 provenance 记录
- `VER-INT-006`: report 流程优先使用 `analysis_summary.json`, 并在 legacy sibling 模式下执行完整性校验
- `VER-INT-007`: matched benchmark 实际使用的 eval 参数与快照配置一致
- `VER-INT-008`: phased training 会按阶段切换启用 cohort、loss 和学习率
- `VER-INT-009`: multi-seed study 会复用同一份 frozen evaluation manifest
- `VER-INT-010`: aggregate report 能回链到每个 seed run 的记录与 artifact
- `VER-INT-011`: best checkpoint 的选择遵循 validation pair AUROC -> easy ID top-1 -> train loss 的固定规则

### 2.4 Scientific Validation

关注论文主张:

- `VER-SCI-001`: easy ID, ambiguous ID, OOD 在几何上可分
- `VER-SCI-002`: pair 相对最佳 scalar 的 matched benchmark 表现
- `VER-SCI-003`: completion sensitivity 可观测
- `VER-SCI-004`: easy ID accuracy 不出现不可接受退化

## 3. 需求到验证映射

| requirement_id | primary_verification |
| --- | --- |
| `REQ-FN-001` | `VER-CON-002`, `VER-UNIT-001` |
| `REQ-FN-003` | `VER-UNIT-002` |
| `REQ-FN-005` | `VER-CON-004`, `VER-INT-002` |
| `REQ-FN-012` | `VER-CON-003` |
| `REQ-FN-013` | `VER-INT-004` |
| `REQ-FN-014` | `VER-INT-003`, `VER-SCI-002` |
| `REQ-FN-019` | `VER-INT-005`, `VER-CON-006` |
| `REQ-FN-020` | `VER-CON-006`, `VER-INT-006` |
| `REQ-FN-021` | `VER-CON-005`, `VER-CON-006` |
| `REQ-FN-022` | `VER-INT-007` |
| `REQ-FN-023` | `VER-INT-005`, `VER-INT-006` |
| `REQ-FN-024` | `VER-INT-008` |
| `REQ-FN-025` | `VER-INT-009` |
| `REQ-FN-026` | `VER-INT-011` |
| `REQ-FN-027` | `VER-INT-010` |
| `REQ-FN-028` | `VER-CON-003`, `VER-INT-007` |
| `REQ-SCI-001` | `VER-SCI-001` |
| `REQ-SCI-002` | `VER-SCI-002` |
| `REQ-SCI-003` | `VER-SCI-004` |
| `REQ-SCI-004` | `VER-INT-009` |

## 4. 进入实现前的门槛

以下项目满足后, 才认为“开发前初始化完成”:

1. 命名标准已基线化
2. 架构边界已基线化
3. 需求项和验证项已建立映射
4. ADR-0001 已确认
5. 目录树和包路径已落地

## 5. 首轮测试建议

初始化后的第一批测试建议按以下顺序实现:

1. 输出质量守恒测试
2. unknown 导出规则测试
3. content entropy 数值测试
4. batch/output contract 测试
5. 小样本 end-to-end smoke test
