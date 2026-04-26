# 验证与确认计划

- document_id: ver_verification_and_validation_plan
- status: baselined
- owner: frcnet_project
- last_updated: 2026-04-26

## 1. 目标

本计划用于确保 FRCNet 的实现不偏离文档基线, 并且实验结论可被证伪、可被复核。

## 2. 验证层级

### 2.1 Unit Verification

关注数学和函数不变量:

- `VER-UNIT-001`: `class_mass.sum + unknown_mass == 1`
- `VER-UNIT-002`: `unknown_mass == 1 - resolution_ratio`
- `VER-UNIT-003`: `content_distribution.sum == 1`
- `VER-UNIT-004`: `state_content_entropy` 数值范围正确
- `VER-UNIT-005`: `top1_completion_beta_*` 对不同 beta 的计算一致
- `VER-UNIT-006`: proposition mass 分解满足 `truth + false + unknown = 1`
- `VER-UNIT-007`: `resolution_entropy` 与 `ternary_entropy` 数值范围正确
- `VER-UNIT-008`: unknown-content regularizer 对 peaked `content_distribution` 给出更高惩罚
- `VER-UNIT-009`: `state_entropy = resolution_entropy + state_weighted_content_entropy`
- `VER-UNIT-010`: beta policy 默认值按 view 计算, top1=`1/K`, candidate=`|S|/K`, binary pignistic=`0.5`

### 2.2 Contract Verification

关注输入输出 schema:

- `VER-CON-001`: input batch contract 完整
- `VER-CON-002`: model output contract 字段完整
- `VER-CON-003`: analysis record contract 列名稳定
- `VER-CON-004`: cohort 名称仅来自规范集合
- `VER-CON-005`: manifest contract 保证非空、单一 `protocol_id` 与唯一 `sample_id`
- `VER-CON-006`: analysis/report bundle contract 保证 run/protocol 与 sample set 一致
- `VER-CON-007`: proposition record contract 保证规范 `tau` 已切换到 `proposition_truth_ratio`
- `VER-CON-008`: v0.2 primary scalar whitelist 不包含 label-aware proposition fields
- `VER-CON-009`: frozen matched manifest hash 对同配置稳定, 对 reference score 改动敏感
- `VER-CON-010`: V0.6 manifest roundtrip 保留 `source_domain_name` 与 `source_domain_label`
- `VER-CON-011`: V0.6 CIFAR-100 不得进入 train 或 validation manifest
- `VER-CON-012`: V0.6B GRL 必须反转 backbone feature 梯度方向
- `VER-CON-013`: V0.6B source-invariant losses 缺少 source provenance 或 source logits 时必须失败
- `VER-CON-014`: V0.6B OOD SupCon 在没有跨源 positive 时必须返回 connected zero 且不产生 NaN
- `VER-CON-015`: V0.6C CIFAR-100 seen/unseen class sets 必须严格等于 `[0,50)` / `[50,100)` 且不相交
- `VER-CON-016`: V0.6C final held-out records 必须使用 `source_role=unseen_ood_classes`
- `VER-CON-017`: V0.6C source-weighted sampler 必须提升 near-OOD 抽样权重但保持 batch composition
- `VER-CON-018`: custom checkpoint names 必须被 checkpoint writer 与 study loader 同时遵循
- `VER-CON-019`: stale resume provenance hash 默认必须失败
- `VER-CON-020`: aggregate ranking metric 或 required source slice 缺失/NaN 时必须抛错

### 2.3 Integration Verification

关注流程拼装:

- `VER-INT-001`: 数据加载到模型前向链路可执行
- `VER-INT-002`: 训练 step 能同时处理 known, unknown, ambiguous 样本
- `VER-INT-003`: 评估流程能导出 scalar 和 pair 指标
- `VER-INT-004`: 分析流程能生成 scatter、hexbin、二维 occupancy 图, 并把 cohort count 独立输出
- `VER-INT-005`: 无 checkpoint 的 analysis 导出默认被拒绝, 显式 override 时会留下 provenance 记录
- `VER-INT-006`: report 流程优先使用 `analysis_summary.json`, 并在 legacy sibling 模式下执行完整性校验
- `VER-INT-007`: matched benchmark 实际使用的 eval 参数与快照配置一致
- `VER-INT-008`: phased training 会按阶段切换启用 cohort、loss 和学习率
- `VER-INT-009`: multi-seed study 会复用同一份 frozen evaluation manifest
- `VER-INT-010`: aggregate report 能回链到每个 seed run 的记录与 artifact
- `VER-INT-011`: best checkpoint 的选择遵循 validation pair AUROC -> easy ID top-1 -> train loss 的固定规则
- `VER-INT-012`: `checkpoint_best_theory` 与 `checkpoint_best_balanced` 会同时生成并写入 selection summary
- `VER-INT-013`: proposition diagnostic report 能从 analysis sidecar 生成
- `VER-INT-014`: aggregate report 能识别并保留 `model_family`
- `VER-INT-015`: study / experiment 会同时导出 primary policy 与 companion policy 的 `analysis* / report*`
- `VER-INT-016`: 主 matched benchmark 与 aggregate 主图不会再包含 proposition diagnostic `tau`
- `VER-INT-017`: hard-ID manifest recipe 参数来自协议配置, 不被实现层静默覆盖
- `VER-INT-018`: formal matched benchmark 可使用 frozen matched manifest, 不依赖临时数量截断
- `VER-INT-019`: manifest overlap audit 使用 `(source_dataset_name, source_dataset_split, source_sample_index)` 验证 validation/final 零交叉
- `VER-INT-020`: V0.5 final report 为 seen SVHN OOD、unseen CIFAR-100 OOD 和 all OOD 分别生成 frozen matched benchmark
- `VER-INT-021`: V0.6 source-balanced sampler 在 batch 内平衡 OOD/unknown source
- `VER-INT-022`: V0.6 aggregate 输出 source slice metrics、worst-source AUROC、seen-unseen gap 和 pair-scalar delta
- `VER-INT-023`: V0.6B source-balanced sampler 支持 5 个 OOD/unknown sources
- `VER-INT-024`: V0.6B study 在 `require_source_overlap_zero=true` 时强制检查 train、validation 与 final source fingerprint overlap
- `VER-INT-025`: V0.6B aggregate 输出 `seen_ood_tiny_imagenet_pair_auroc`
- `VER-INT-026`: V0.6C 小样本 manifest chain 能生成 train/validation/final class-holdout manifests
- `VER-INT-027`: OOD-only batch 在 source adversary / SupCon / source calibration 有有效样本时执行 optimizer step
- `VER-INT-028`: `protocol_controls` 声明与 protocol/train/eval 配置不一致时 study preflight 失败
- `VER-INT-029`: unchanged study 可以 resume, changed config 在 `fail_on_stale` 下失败
- `VER-INT-030`: V0.6C aggregate 比较 `near_ood_balanced`, `balanced`, and `theory`
- `VER-INT-031`: `git ls-files -ci --exclude-standard` 对普通提交应为空或仅包含显式白名单

### 2.4 Scientific Validation

关注论文主张:

- `VER-SCI-001`: easy ID, ambiguous ID, OOD 在几何上可分
- `VER-SCI-002`: pair 相对最佳 scalar 的 matched benchmark 表现
- `VER-SCI-003`: completion sensitivity 可观测
- `VER-SCI-004`: easy ID accuracy 不出现不可接受退化
- `VER-SCI-005`: v0.2 final test 达到 release gate 或明确标记 partial/negative evidence
- `VER-SCI-006`: v0.5 final test 的 unseen OOD 结论仅来自 CIFAR-100 final-only slice
- `VER-SCI-007`: v0.6 final test 以 CIFAR-100 LOSO、worst-source AUROC 与 seen-unseen gap 判断修复等级
- `VER-SCI-008`: v0.6C final test 以 unseen CIFAR100 held-out classes、seen near-OOD 与 worst-source AUROC 判断修复等级

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
| `REQ-FN-029` | `VER-UNIT-006`, `VER-CON-007`, `VER-INT-013` |
| `REQ-FN-030` | `VER-UNIT-008`, `VER-INT-002` |
| `REQ-FN-031` | `VER-INT-012` |
| `REQ-FN-032` | `VER-INT-014` |
| `REQ-FN-033` | `VER-INT-015` |
| `REQ-FN-034` | `VER-INT-004` |
| `REQ-FN-035` | `VER-CON-003`, `VER-UNIT-009` |
| `REQ-FN-036` | `VER-UNIT-006`, `VER-CON-008` |
| `REQ-FN-037` | `VER-CON-009`, `VER-INT-018` |
| `REQ-FN-038` | `VER-CON-008`, `VER-INT-016` |
| `REQ-FN-043` | `VER-CON-011`, `VER-INT-021` |
| `REQ-FN-044` | `VER-INT-021` |
| `REQ-FN-045` | `VER-INT-022` |
| `REQ-FN-046` | `VER-CON-012`, `VER-CON-013`, `VER-CON-014` |
| `REQ-FN-047` | `VER-INT-025` |
| `REQ-FN-048` | `VER-CON-015`, `VER-INT-026` |
| `REQ-FN-049` | `VER-CON-017`, `VER-INT-021` |
| `REQ-FN-050` | `VER-CON-018`, `VER-INT-030` |
| `REQ-FN-051` | `VER-CON-019`, `VER-INT-029` |
| `REQ-FN-052` | `VER-INT-028`, `VER-INT-024` |
| `REQ-FN-053` | `VER-CON-020`, `VER-INT-022` |
| `REQ-SCI-001` | `VER-SCI-001` |
| `REQ-SCI-002` | `VER-SCI-002` |
| `REQ-SCI-009` | `VER-SCI-007` |
| `REQ-SCI-011` | `VER-SCI-008` |
| `REQ-SCI-003` | `VER-SCI-004` |
| `REQ-SCI-004` | `VER-INT-009` |
| `REQ-SCI-005` | `VER-CON-007`, `VER-SCI-003` |
| `REQ-SCI-006` | `VER-SCI-002`, `VER-SCI-005` |
| `REQ-SCI-007` | `VER-SCI-004`, `VER-SCI-005` |

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
