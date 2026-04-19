# 架构说明

- document_id: arch_architecture_description
- status: baselined
- owner: frcnet_project
- last_updated: 2026-04-19
- standard_alignment: iso_ieee_42010_informed, iso_iec_ieee_15289

## 1. 架构目标

架构必须把论文中的 gate/content 分解变成原生工程结构, 而不是后处理附加层。模型和分析链路都围绕以下显式对象构建:

- `resolution_ratio`
- `content_distribution`
- `class_mass`
- `unknown_mass`
- `content_entropy`
- `completion_score`

## 2. 上下文视图

```text
input sample
  -> cohort-aware data builder
  -> backbone
  -> resolution head
  -> content head
  -> structured output contract
  -> loss computation / inference metrics
  -> evaluation datasets
  -> analysis and paper artifacts
```

## 3. 模块分解

### 3.1 `src/frcnet/data`

职责:

- 数据源注册
- cohort 构造
- 歧义样本 candidate set 编码
- batch 合约标准化

建议子模块:

- `dataset_registry.py`
- `cohort_builders.py`
- `sample_contracts.py`
- `transforms.py`

### 3.2 `src/frcnet/models`

职责:

- backbone 封装
- resolution head
- content head
- 输出归一化与结构化打包

建议子模块:

- `backbones.py`
- `resolution_head.py`
- `content_head.py`
- `frcnet_model.py`
- `output_contracts.py`

### 3.3 `src/frcnet/training`

职责:

- loss 实现
- 训练步骤
- 日志与 checkpoint 逻辑

建议子模块:

- `losses.py`
- `trainer.py`
- `optimizers.py`
- `schedulers.py`

### 3.4 `src/frcnet/evaluation`

职责:

- 样本级推理导出
- scalar 指标
- pair probe 指标
- selective / matched benchmark

建议子模块:

- `inference.py`
- `scalar_metrics.py`
- `pair_metrics.py`
- `matched_benchmark.py`

### 3.5 `src/frcnet/analysis`

职责:

- cohort occupancy
- scatter / hexbin 可视化
- completion sensitivity
- 论文表格聚合

建议子模块:

- `geometry_reports.py`
- `plotting.py`
- `completion_analysis.py`
- `report_tables.py`

### 3.6 `src/frcnet/utils`

职责:

- 熵计算
- 标识符生成
- 序列化
- 可追溯日志工具

## 4. 数据合约

### 4.1 Input Batch Contract

每个 batch 至少应包含:

- `image`
- `class_label`
- `sample_id`
- `split_name`
- `cohort_name`
- `source_dataset_name`
- `source_class_label` 可选
- `candidate_class_set` 可选

### 4.2 Model Output Contract

模型前向输出至少应包含:

- `backbone_feature`
- `resolution_logit`
- `resolution_ratio`
- `content_logits`
- `content_distribution`
- `class_mass`
- `unknown_mass`

### 4.3 Analysis Record Contract

样本级分析表至少应包含:

- `run_id`
- `protocol_id`
- `sample_id`
- `split_name`
- `cohort_name`
- `source_dataset_name`
- `source_class_label`
- `predicted_class_index`
- `class_label`
- `resolution_ratio`
- `unknown_mass`
- `content_entropy`
- `top1_class_mass`
- `top1_content_probability`
- `completion_score_beta_0_1`
- `completion_score_beta_0_5`

### 4.4 Analysis Export Summary Contract

analysis 阶段必须额外生成 `analysis_summary.json`, 作为 report 阶段的规范 sidecar。

该 sidecar 至少应包含:

- `run_id`
- `protocol_id`
- `analysis_path`
- `checkpoint_path`
- `manifest_snapshot_path`
- `model_config_snapshot_path`
- `proposition_path`
- `integrity_overrides`
- `sidecar_resolution_mode`

## 5. 关键架构约束

- `ARCH-001`: `unknown_mass` 只能由 `resolution_ratio` 导出
- `ARCH-002`: `content_distribution` 仅描述 resolved 子空间内部类别分布
- `ARCH-003`: `completion_score` 属于读出层, 不属于模型主状态
- `ARCH-004`: 分析层必须支持 pair 与 scalar 并行输出
- `ARCH-005`: 文档、配置、代码、实验记录之间必须有可追溯链接
- `ARCH-006`: report 阶段必须先完成 bundle integrity 校验, 再写 `experiment_record.md`
- `ARCH-007`: analysis/report 间的 sidecar 解析必须优先使用规范 `analysis_summary.json`, 不得默认依赖 sibling 猜测

## 6. 初始化阶段的默认技术选择

- 默认 backbone: `ResNet-18`
- 默认主包路径: `src/frcnet/`
- 默认配置组织: `configs/model`, `configs/data`, `configs/train`, `configs/eval`, `configs/analysis`
- 默认记录组织: `records/decisions`, `records/experiments`, `records/reviews`

## 7. 待确认架构决策

- 是否在首轮就引入可选的 `weighted_pair = resolution_ratio * content_entropy`
- 是否把 `top1 correctness proposition` 作为评估层专用对象, 而不是训练时显式对象
- 是否需要在初始化阶段就固定 analysis schema 为 parquet/csv 双输出
