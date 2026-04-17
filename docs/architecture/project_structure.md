# 项目结构说明

- document_id: arch_project_structure
- status: baselined
- owner: frcnet_project
- last_updated: 2026-04-18

## 1. 设计原则

项目结构以“文档先行、证据留痕、实现解耦”为原则组织:

- `docs/` 放规范
- `records/` 放事实
- `src/` 放实现
- `artifacts/` 放生成物

## 2. 标准目录树

```text
HardMin/
├── README.md
├── pyproject.toml
├── docs/
│   ├── index.md
│   ├── governance/
│   ├── requirements/
│   ├── architecture/
│   ├── verification/
│   ├── records/
│   └── templates/
├── records/
│   ├── decisions/
│   ├── experiments/
│   └── reviews/
├── src/
│   └── frcnet/
│       ├── data/
│       ├── models/
│       ├── training/
│       ├── evaluation/
│       ├── analysis/
│       └── utils/
├── configs/
│   ├── model/
│   ├── data/
│   ├── train/
│   ├── eval/
│   └── analysis/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── contract/
├── scripts/
├── artifacts/
│   ├── figures/
│   ├── tables/
│   ├── reports/
│   ├── checkpoints/
│   └── logs/
└── notebooks/
```

## 3. 分层语义

### 3.1 `docs/`

项目的规范性源头。这里定义术语、接口、需求、架构和验证要求。

### 3.2 `records/`

存放 ADR、实验记录、评审记录。这里不改写规范, 只记录决策和结果。

### 3.3 `src/frcnet/`

实现层。任何新增模块都应能说明它对应哪个 requirement 或 ADR。

### 3.4 `configs/`

配置与代码分离。训练、评估、分析参数都应通过配置文件落盘并被实验记录引用。

### 3.5 `tests/`

按验证粒度拆分:

- `unit`: 数学与函数级校验
- `integration`: 训练/评估流水线拼装
- `contract`: 数据和输出 schema 不变量

### 3.6 `artifacts/`

只保存可重新生成的派生对象。其真值来源是代码、配置和实验记录, 不是 artifact 自身。

## 4. 文件树约束

- 目录树优先表达职责, 不表达临时阶段
- 不建立 `misc`, `temp`, `other`, `new_files` 之类无语义目录
- 训练、评估、分析禁止混放在同一模块
- notebook 只用于探索, 结论要沉淀回 `docs/` 或 `records/`

## 5. 推荐后续补充

初始化之后, 建议下一批文件优先落在:

1. `configs/model/frcnet_resnet18_base.yaml`
2. `configs/data/cifar10_svhn_small.yaml`
3. `src/frcnet/models/frcnet_model.py`
4. `src/frcnet/training/losses.py`
5. `tests/contract/test_output_contracts.py`

