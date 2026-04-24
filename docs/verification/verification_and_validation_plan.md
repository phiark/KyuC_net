# 验证与确认计划

- document_id: ver_verification_and_validation_plan
- status: baselined
- owner: frcnet_project
- last_updated: 2026-04-24

## 1. 目标

验证 `plan_a_next_v0_2` 在 `next-v0.1` 语义拆层基础上形成数据口径干净、模型解释可信的多 seed 研究基线，并防止 label-aware 诊断量重新混入主 benchmark。

## 2. Unit Verification

- `VER-UNIT-001`: `class_mass.sum + unknown_mass == 1`。
- `VER-UNIT-002`: `state_entropy = h(r) + r * H_K(c)`。
- `VER-UNIT-003`: proposition mass 满足 `pT + pF + pU = 1`。
- `VER-UNIT-004`: `top1_view` 不读取 label 或 candidate set。
- `VER-UNIT-005`: `target_view` / `candidate_view` 显式 `label_aware = true`。
- `VER-UNIT-006`: beta policy 满足 `top1_symmetric`, `candidate_symmetric`, `binary_pignistic` 的定义。
- `VER-UNIT-007`: frozen matched manifest 同配置 hash 稳定。
- `VER-UNIT-008`: Softmax reference score 由独立 reference pipeline 导出。
- `VER-UNIT-009`: hard-ID recipe 参数由协议配置驱动。

## 3. Contract Verification

- `VER-CON-001`: batch contract 校验 label bounds 和 cohort 字段。
- `VER-CON-002`: model output contract 字段完整。
- `VER-CON-003`: analysis record 同时导出现行 state 字段和 legacy alias。
- `VER-CON-004`: 主 benchmark 配置 label-aware feature 时必须失败。
- `VER-CON-005`: scalar fairness 表包含 raw / oriented / logistic 三列。
- `VER-CON-006`: matched manifest 记录 hash、bin 和 reference score 字段。
- `VER-CON-007`: final test cohort 必须包含 seen MixUp pairs、held-out class pairs、seen-source unknown/OOD 标记。

## 4. Integration Verification

- `VER-INT-001`: 单 run train -> inference -> report workflow 可执行。
- `VER-INT-002`: report 生成优先使用 `analysis_summary.json` 做 sidecar 校验。
- `VER-INT-003`: matched benchmark 能读取 frozen matched manifest。
- `VER-INT-004`: artifact bundle 输出 scatter、hexbin、二维 occupancy、cohort counts。
- `VER-INT-005`: proposition diagnostics 独立输出，不进入主 matched benchmark 表。
- `VER-INT-006`: 1 epoch smoke train -> inference -> report 可跑通，30 epoch seeds `[7, 17, 27]` 作为 v0.2 study 目标配置。
- `VER-INT-007`: Softmax CE same-backbone reference scores 能进入 frozen matched manifest 构造。

## 5. Deferred

以下内容不在本分支验收:

- aggregate report
- decision utility / regret benchmark
- full external baseline matrix
