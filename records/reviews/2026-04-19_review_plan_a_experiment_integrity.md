# Review Record: Plan A Experiment Integrity

- date: 2026-04-19
- owner: frcnet_project
- status: baselined
- linked_requirements:
  - REQ-FN-019
  - REQ-FN-020
  - REQ-FN-021
  - REQ-FN-022
  - REQ-FN-023
- linked_adrs:
  - ADR-0001
  - ADR-0002

## Findings

1. analysis 导出在缺少 checkpoint 时仍可生成正式结果
2. report 生成信任任意或混合 analysis CSV, 未校验 sidecar 一致性
3. 外部 manifest 未校验 `sample_id` 唯一性
4. eval 配置快照未实际驱动 matched benchmark

## Repair Goal

把 Plan A 实验链修复为默认严格、override 留痕的 document-driven 工作流, 重新闭合:

`requirement -> config -> run -> record -> artifact`

## Acceptance Criteria

- inference 默认要求 checkpoint, 显式 override 时会写入 sidecar metadata
- report 默认校验 analysis / manifest / proposition / sidecar 一致性
- manifest 与 analysis record 默认拒绝重复 `sample_id`
- matched benchmark 使用解析后的 eval 配置而不是内部默认值
- experiment record 明确记录 checkpoint provenance、sidecar 解析模式和所有 override
