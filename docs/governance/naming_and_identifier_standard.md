# 命名与标识标准

- document_id: gov_naming_identifier_standard
- status: baselined
- owner: frcnet_project
- last_updated: 2026-04-22
- standard_alignment: iso_8601, iso_iec_11179_inspired

## 1. 目标

本标准用于固定 FRCNet 的变量命名、文件命名、目录命名和实验标识, 防止论文符号、设计书术语、代码变量三套口径并行漂移。

## 2. 总体命名规则

### 2.1 路径与文件名

- 仓库内路径统一使用 ASCII
- 目录名统一使用 `lowercase_snake_case`
- 文件名统一使用 `lowercase_snake_case`
- 不在仓库文件名中使用空格、中文、临时版本尾缀如 `final2`, `new`, `latest`
- 日期前缀使用 ISO 8601, 例如 `2026-04-18_...` 或 `20260418T093000+0800`

### 2.2 Python 标识符

- package/module/function/variable: `snake_case`
- class: `PascalCase`
- constant: `UPPER_SNAKE_CASE`
- boolean: `is_...`, `has_...`, `should_...`
- tensor/batch 显式表达语义, 不使用歧义缩写

### 2.3 单字母变量限制

仅在以下场景允许使用论文同名单字母变量:

- 数学公式
- 推导注释
- 不超过 5 行的局部向量化表达式
- 对论文公式进行逐项单元测试时

除此之外, 代码层不得把主变量长期命名为 `r`, `u`, `c`, `z`, `a`, `S`。

## 3. 规范命名映射

| paper_or_note_symbol | canonical_meaning | code_name | rule |
| --- | --- | --- | --- |
| `h(x)` | backbone feature | `backbone_feature` | 共享主干输出 |
| `a` | resolution logit | `resolution_logit` | gate head 原始输出 |
| `T_r` | resolution temperature | `resolution_temperature` | gate 温度 |
| `r` | resolution ratio | `resolution_ratio` | 已解析质量比例 |
| `z` | content logits | `content_logits` | content head 原始输出 |
| `T_c` | content temperature | `content_temperature` | content 温度 |
| `c` | content distribution | `content_distribution` | resolved 子空间类别分布 |
| `p_k` | class mass | `class_mass` | `resolution_ratio * content_distribution` |
| `u` | unknown mass | `unknown_mass` | `1 - resolution_ratio` |
| `p_top1` | top-1 class mass | `top1_class_mass` | top-1 的 `class_mass` |
| `tau` / `τ` | proposition truth ratio | `proposition_truth_ratio` | 规范口径是命题层 `p_T / (p_T + p_F)` |
| `H_cont` | content entropy | `content_entropy` | 对 `content_distribution` 求熵 |
| `r H_cont` | resolution-weighted content entropy | `resolution_weighted_content_entropy` | 对 `resolution_ratio * content_entropy` 的显式导出 |
| `H_res` | resolution entropy | `resolution_entropy` | 对 `resolution_ratio` 的二元熵 |
| `H_3` | ternary entropy | `ternary_entropy` | 对显式状态求熵 |
| auxiliary `tau` surrogate | top-1 content probability | `auxiliary_top1_content_probability` | 保留为辅助诊断量, 不再作为规范 `tau` |
| `beta` / `β` | completion policy parameter | `completion_policy_beta` | 下游读出策略参数 |
| `q_beta` | completion score | `completion_score` | 标量读出 |
| `S` | candidate class set | `candidate_class_set` | 歧义监督候选类集合 |
| `r0` | ambiguous resolution target | `ambiguous_resolution_target` | 歧义样本的目标解析度 |
| `lambda_r` | ambiguous resolution weight | `ambiguous_resolution_weight` | 歧义正则权重 |
| `L_id` | known-label loss | `loss_id` | 已知单标签损失 |
| `L_unk` | explicit unknown loss | `loss_unknown` | unknown 监督损失 |
| `L_amb` | ambiguity loss | `loss_ambiguous` | 歧义监督损失 |

## 4. 派生变量命名规则

- 多个 beta 对应的 completion score 用 `completion_score_by_beta`
- 标量表格列名使用全名, 例如 `resolution_ratio`, `content_entropy`
- 批量张量用 `_batch` 后缀, 例如 `resolution_ratio_batch`
- 掩码使用 `_mask`, 例如 `is_unknown_sample_mask`
- 索引使用 `_index` 或 `_indices`
- 日志字典键优先与分析列名一致

## 5. 数据与记录命名

### 5.1 Cohort Names

统一使用以下 cohort 名称:

- `easy_id`
- `ambiguous_id`
- `hard_id`
- `ood`
- `unknown_supervision`

### 5.2 Dataset Split Names

- `train`
- `validation`
- `test`
- `analysis`

### 5.3 Identifier Examples

- `REQ-FN-001`
- `ADR-0001`
- `EXP-2026-04-18-01`
- `RUN-20260418T093000+0800-seed007`

## 6. 禁止项

- 不把 `unknown_mass` 命名为 `uncertainty`, 因为含义过宽
- 不把 `content_distribution` 或 `auxiliary_top1_content_probability` 命名为 `tau`, 因为规范 `tau` 是命题级 `proposition_truth_ratio`
- 不把 `proposition_truth_ratio` 放回主 matched benchmark 里充当公平 scalar baseline, 它属于 proposition diagnostics
- 不把 `completion_score` 当作模型唯一主输出
- 不在不同文件中混用 `vacuity`, `unknown_mass`, `unresolved_mass` 指向同一对象而不声明
