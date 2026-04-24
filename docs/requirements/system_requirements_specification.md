# 系统需求规格说明

- document_id: req_system_requirements_specification
- status: baselined
- owner: frcnet_project
- last_updated: 2026-04-24

## 1. 范围

`plan_a_next_v0_2` 的范围是在 `next-v0.1` 语义拆层基础上形成可审计的多 seed 研究基线。模型主体保持原生 FRCNet 双头结构，不继续扩展 `v0.3*` debug 训练协议。

## 2. In Scope

- `ResNet-18 + resolution head + content head`
- `state / proposition / completion` 三层字段拆分
- label-free 主 benchmark feature whitelist
- frozen matched manifest 合同
- raw / oriented / one-feature-logistic scalar fairness 表
- 最小 Softmax CE reference score pipeline
- validation 与 final test manifest 分离
- seed `[7, 17, 27]` 的 30 epoch study 配置
- 旧 analysis record 的 legacy alias 读取

## 3. Out Of Scope

- full Softmax / EDL / SelectiveNet baseline matrix
- decision benchmark
- Transformer、teacher distillation、K+1 unknown softmax 类
- 继续调试 `v0.3debug_r2` geometry loss

## 4. 功能需求

- `REQ-FN-001`: 模型必须输出 `resolution_ratio`, `content_distribution`, `class_mass`, `unknown_mass`。
- `REQ-FN-002`: 输出必须满足 `class_mass.sum + unknown_mass = 1`。
- `REQ-FN-003`: `unknown_mass` 必须由 `1 - resolution_ratio` 导出。
- `REQ-FN-004`: state layer 必须导出 `state_content_entropy`, `state_weighted_content_entropy`, `state_entropy`。
- `REQ-FN-005`: proposition view 必须导出 `pT / pF / pU / tau_A`, 且满足质量守恒。
- `REQ-FN-006`: `top1_view` 必须是 label-free；`target_view` 与 `candidate_view` 必须显式标记为 label-aware。
- `REQ-FN-007`: completion readout 必须绑定具体 proposition view，不得把 `q_beta` 作为全局 confidence。
- `REQ-FN-008`: 主 matched benchmark 只能使用 label-free 字段。
- `REQ-FN-009`: 主 matched benchmark 配置 label-aware 字段时必须失败。
- `REQ-FN-010`: scalar baseline 必须同时输出 `raw_auc`, `oriented_auc`, `one_feature_logistic_auc`。
- `REQ-FN-011`: frozen matched manifest 必须记录 `manifest_hash`, `construction_config_hash`, `match_bin_id`, `reference_score_name`, `reference_score_value`。
- `REQ-FN-012`: Softmax CE reference 只用于生成 external reference score，不作为完整 baseline 主实验。
- `REQ-FN-013`: 旧字段 `content_entropy`, `resolution_weighted_content_entropy`, `completion_score_beta_*` 只能作为 legacy alias 或 auxiliary diagnostic。
- `REQ-FN-014`: validation manifest 只能用于 checkpoint selection，final test manifest 只能用于最终报告。
- `REQ-FN-015`: SVHN test 必须标为 seen-source unknown/OOD，不得写成 unseen OOD 泛化。
- `REQ-FN-016`: hard-ID recipe 参数必须来自协议配置，不得被实现中的固定 recipe 静默覆盖。

## 5. 成功判据

- 主 pair 默认使用 `resolution_ratio__state_weighted_content_entropy`，raw state pair 作为辅助。
- `proposition_truth_ratio` 不得进入主 matched benchmark 或主 scalar 排名。
- frozen matched manifest 在同配置下可复现同一 hash。
- 旧 `v0.3debug_r2` analysis CSV 仍可读，但输出必须标清 legacy alias。
- v0.2 release gate 以 final test 的 pair AUROC、best scalar gap、easy/hard top-1、ambiguous hit 为主；未达标结果必须标为 partial/negative evidence。
