---
name: training-sft
description: 用 training-sft 做监督微调（LoRA）并打穿垂直闭环（数据→训练→部署→推理→评估）。当用户要求"SFT 微调 / 指令微调 / LoRA 训练 / 训练一个模型 / 垂直闭环"时使用。
---

# training-sft

薄插件：**公共训练循环走 `@ai-infra/training-core`**，本插件只做 SFT 特有的适配与门禁。

## 六个工具

| 工具 | 作用 | 风险档 |
|---|---|---|
| `training_sft_prepare` | 数据校验与切分（去重 / 剔除含 PII 行 / 剔除与评估集相撞或包含的行 / 确定性切分 / 血缘） | write / 无需令牌 |
| `training_sft_train` | LoRA 训练（容器沙箱 + 公共循环 + 预算强制） | write / **需令牌** |
| `training_sft_merge` | 显式合并 LoRA → 完整 checkpoint（trained⇒合并；已合并⇒幂等；坏状态⇒拒绝）| write / **需令牌** |
| `training_sft_validate` | 合并 → serving 部署 → 双门禁 + 遗忘门禁 | write / **需令牌** |
| `training_sft_rollback` | 回退一个 run | write / **需令牌** |
| `training_sft_get_report` | 读 run 记录 + 审计链 | read |
| `training_sft_health` | 环境自检（沙箱行为探测 / 镜像 / GPU / 模型 / serving / 审计链） | read |

## 典型流程

```
1. training_sft_health                      # 先看环境
2. training_sft_prepare  {train:1000, heldout:200}
3. training_sft_train    {datasetDir, maxSteps:200, confirm}
4. training_sft_merge {runId, confirm}   # 可选：把合并作为独立可审计步骤（validate 内部也会自动合并，同一代码路径）
5. training_sft_validate {runId, thresholds:{maxF1DropPct:5, minFormatGainPct:10}, confirm}
5. training_sft_get_report {runId}          # 复核，含审计链校验
```

## 硬约束

- 训练数据与**任何**评估集的内容哈希交集必须为空；否则 `E_DATASET_LEAK` 拒绝训练。
- 单次训练：≤500 步 / ≤2 GPU 小时 / 并发 1 / 模型 ≤0.5B；超限**暂停并报告**，不降级。
- 发现 PII：**只剔除整行**并记录哈希清单，绝不改写内容，绝不"脱敏后继续"。
- 门禁阈值**必须显式给出**，插件不内置默认精度阈值。
- 评估端点由本插件自己租用并拆解；存量端点一律只读。
