# 系统需求规格说明

- document_id: req_system_requirements_specification
- status: baselined
- owner: frcnet_project
- last_updated: 2026-04-18

## 1. 范围

FRCNet 是一个面向研究验证的显式未知网络。它的核心目标不是只输出单一 confidence, 而是原生分离:

- 是否已经进入 resolved 子空间
- 在 resolved 子空间内部, 类别内容是否仍然冲突

## 2. 系统边界

### 2.1 In Scope

- 轻量 backbone 上的 resolution head + content head
- `p_k = r * c_k`, `u = 1 - r` 的结构化输出
- 对 easy ID, ambiguous ID, hard ID, OOD 的统一分析
- pair-vs-scalar 的 matched benchmark
- 面向论文图表和实验记录的可追溯产物

### 2.2 Out Of Scope

- 在线服务部署
- 分布式大规模训练
- 自动化超参数搜索平台
- 与论文无关的通用 MLOps 平台能力

## 3. 功能需求

### 3.1 Core Modeling

- `REQ-FN-001`: 模型必须输出 `resolution_ratio`、`content_distribution`、`class_mass`、`unknown_mass`
- `REQ-FN-002`: 输出必须满足 `class_mass.sum + unknown_mass = 1`
- `REQ-FN-003`: `unknown_mass` 必须由 `1 - resolution_ratio` 导出, 不能由独立 head 自由回归
- `REQ-FN-004`: 系统必须支持导出 `completion_score` 作为下游读出, 但不得把它当作唯一规范输出

### 3.2 Data And Supervision

- `REQ-FN-005`: 系统必须支持 `easy_id`、`ambiguous_id`、`hard_id`、`ood` 四类 cohort
- `REQ-FN-006`: 系统必须支持显式 unknown 监督样本, 其训练目标是提升 `unknown_mass`
- `REQ-FN-007`: 系统必须支持歧义样本监督, 包括 `candidate_class_set` 与 `ambiguous_resolution_target`

### 3.3 Training

- `REQ-FN-008`: 系统必须实现 `loss_id`
- `REQ-FN-009`: 系统必须实现 `loss_unknown`
- `REQ-FN-010`: 系统必须实现 `loss_ambiguous`
- `REQ-FN-011`: 系统应支持可选的校准损失和去相关约束

### 3.4 Evaluation And Analysis

- `REQ-FN-012`: 系统必须记录 `resolution_ratio`、`unknown_mass`、`content_entropy`
- `REQ-FN-013`: 系统必须支持生成 `(resolution_ratio, content_entropy)` 的散点图、hexbin 图与 cohort occupancy 图
- `REQ-FN-014`: 系统必须支持 matched benchmark, 检验 pair 相对最佳 scalar 的增益
- `REQ-FN-015`: 系统必须支持不同 `completion_policy_beta` 下的 completion sensitivity 分析

下一阶段默认产物命名:

- `sample_analysis_records.csv`
- `top1_proposition_records.csv`
- `geometry_scatter.png`
- `geometry_hexbin.png`
- `cohort_occupancy.png`
- `cohort_summary_table.csv`
- `matched_ambiguous_vs_ood_table.csv`

### 3.5 Documentation And Traceability

- `REQ-FN-016`: 所有实现模块必须能追溯到至少一个 requirement 或 ADR
- `REQ-FN-017`: 每次实验必须绑定配置、代码版本、输入数据说明、结果摘要
- `REQ-FN-018`: 所有核心指标名称必须遵循命名标准文档

## 4. 非功能需求

- `REQ-NF-001`: 代码应保持模块职责清晰, data / models / training / evaluation / analysis 分离
- `REQ-NF-002`: 所有实验记录必须可复现到配置级别
- `REQ-NF-003`: 分析输出应支持直接写入论文图表与表格流程
- `REQ-NF-004`: 小型首轮实验应能在单张 12-24GB GPU 上完成
- `REQ-NF-005`: 初始化阶段优先保证可观测性和可反驳性, 再追求指标最优

## 5. 成功判据

- `REQ-SCI-001`: easy ID, ambiguous ID, OOD 在几何上应出现可解释分区
- `REQ-SCI-002`: 在 matched ambiguous-vs-ood 任务上, pair 应至少不劣于最佳 scalar, 理想情况下有稳定小幅优势
- `REQ-SCI-003`: 不应以明显牺牲 easy ID accuracy 为代价换取 unknown 分离

## 6. 关键开放问题

- `OPEN-001`: 首轮 backbone 固定为 ResNet-18 还是保留 ConvNeXt-Tiny 分支
- `OPEN-002`: 歧义样本构造优先级如何在 MixUp、叠加、遮挡之间排序
- `OPEN-003`: pair 评估默认使用 `(resolution_ratio, content_entropy)` 还是 `(resolution_ratio, resolution_ratio * content_entropy)`
