# NEXT_STEPS —— 进入长任务 3 的启动清单

> 状态截至：2026-09-29。长任务 2 交付 `training-sft` 并跑通垂直闭环。
> **本清单里的每一项都标了证据等级与「谁需要动手」。**

---

## 0. 一句话

training-sft 的垂直闭环**已跑通并双门禁通过**；
进入长任务 3（pretrain / RL / QAT 或 RSI）**没有硬阻塞**，但有 **4 项必须先裁决/复验** 的事（§1）。

---

## 1. 必须先处理的 4 件事

| # | 事项 | 现状 | 需要谁 |
|---|---|---|---|
| 1 | **QAT 与「主模型不做量化优化」的冲突** | 长任务 1 就挂在台账上，**仍未裁决**。`training-qat-contract.md` 因此仍是 skeleton | **用户裁决** |
| 2 | **wikitext 遗忘指标临界通过**（ΔNLL 0.001872 vs 界 0.001990，余量 6%） | 单 seed / 单切分的结果 | 用户决定是否要求多种子复验 |
| 3 | **DSH 重启后的装配路径未验证** | 本轮用 `dev_inject_plugin` 热注入验证（host ✓ + 真调 ✓） | 下次 DSH 重启后跑一次 `training_sft_health` 即可确认 |
| 4 | **`inference-quantization` 接入 SFT 产物** | SFT 产出是标准 HF 目录（含 LICENSE + README frontmatter），**可作为量化输入**；但未实际量化过它 | 长任务 3 若要 QAT/PTQ，先在这里接一次 |

---

## 2. 长任务 3 的候选起点（按依赖顺序）

### A. 若做 QAT（依赖 §1.1 裁决）

```
1. 裁决「主模型不做量化优化」是否适用于训练侧 → 写进 training-qat-contract.md
2. 把 training-qat-contract.md 从 skeleton 提升为 frozen（需真机证据）
3. 复用已有的：training-core 循环 / testing-core 双门禁 / checkpoint-contract 三判据
4. 新增：量化配置与 quantization-contract 的受控词表对齐；起点权重只读 + 哈希不变
5. 精度门禁阈值**必须由调用方显式给出**（不得内置默认 —— eval-contract 既有纪律）
```

### B. 若做 pretrain / RL（各自契约已是 skeleton）

- `training-pretrain-contract.md` 的**冻结前置**：无标注语料的 **PII 筛查口径**必须先定
  （长任务 2 为 SFT 定的是"剔除含非占位 PII 的整行"，预训练语料规模/来源不同，需要重新定）。
- `training-rl-contract.md` 的**冻结前置**：奖励模型自身的 license 与 PII 归属必须先走 Step 0。

### C. 若做 RSI（推理编排）

- 推理侧的 `accept` 路径与 `physical-limit` 停机在长任务 1 的台账上仍未走通
  （见 `inference-integration/NEXT_STEPS.md` §2）—— RSI 之前建议先补这两条。

---

## 3. 现成可复用的资产（不用重造）

| 资产 | 位置 | 复用方式 |
|---|---|---|
| 六个共享库 v0 | `inference-training/shared/libs/*` | 直接 import；契约测试在 CI（C11/C12） |
| 训练侧契约 9 份 | `inference-contracts/contracts/` | 新插件按对应 contract 实现；skeleton 需先冻结 |
| 三份 schema | `inference-contracts/state-schema/{model-metadata,checkpoint-meta,eval-result}.schema.json` | 产物形状直接对齐 |
| 训练镜像 | `training-sft/toolchain:0.1.0`（`f82151dd84fb`） | 派生新镜像；**peft/transformers/torch 版本已钉死** |
| Python trainer 协议 | `training-sft/python/trainer.py` | 新算法只需替换 `cmd_step` 内部的 loss 计算 |
| 容器 worker | `training-sft/src/worker.mjs` | 直接复用（含 uid 映射与协议层） |
| 评估与门禁 | `training-sft/src/evaluate.mjs` | 指标抽取可扩展；**判定逻辑必须继续用 testing-core** |
| 数据准备 | `training-sft/src/data/prepare.mjs` | 换数据集时改 `SOURCE`；四道过滤（去重/PII/整行相撞/**行内包含**）全部保留 |
| 确认令牌 / 状态 / 审计 | `src/confirm.mjs` `src/runtime.mjs` | 与新插件同形复用 |

---

## 4. 长任务 2 遗留的、**不阻塞但必须入台账**的观察

| # | 观察 | 影响 |
|---|---|---|
| 1 | `inference-serving` 的可用性探针用**硬编码容差 1e-4**，而 `testing-core` 的 ε 是**推导值** | 两套判据并存；长任务 2 未动它（推理侧核心逻辑禁区）。建议后续统一到 `testing-core` |
| 2 | 四个插件的审计链仍是**两个变体**（`canonical-json` / `prevhash-canonical-json`） | `audit-core` 已支持两者并自动识别；统一需改推理侧核心逻辑 |
| 3 | `inference-diagnose` 的 `ascend-arm` 维度有一个既有死分支探针 | 见 `INFERENCE_DIAGNOSE_ENHANCEMENT.md` §A1 末 |
| 4 | 训练侧评估协议已到 `task-eval@2`；**协议版本进基线缓存键**是硬要求 | 后续改协议时缓存会自动失效，不会跨协议比较 |
| 5 | `validate` 会**创建并拆解自己租用的端点**（带 ownerToken） | 新插件若也起端点，务必遵守同一条所有权纪律，绝不触碰存量端点 |

---

## 5. 环境现状（进去之前对一下）

| 项 | 值 |
|---|---|
| 共享库契约 | 6 库 / 192 断言，全绿（C11） |
| 契约层 | **64/64**（C1–C16） |
| training-sft 单测 | 4 spec / **196 断言**，全绿（C15） |
| 三插件 `test:contract` | 各 64/64 |
| 容器残留 | **0**（serving managed 0 · training-sft toolchain 0） |
| 存活端点 | **0**（本轮两个端点均已 teardown） |
| GPU | GB10，仅既有 `qwen3-embedding` 占用 4583 MiB（本轮未触碰） |
| 训练镜像 | `training-sft/toolchain:0.1.0` @ `f82151dd84fb` |
| 基座模型 | `/data/models/Qwen2.5-0.5B-Instruct`（0.494B，apache-2.0，rev `7ae5576…`） |
| 数据集 | `~/.ai-infra/state/datasets/yahma-alpaca-cleaned__sft__s20260929__n1000`（train 1000 / heldout 200，`allClear=true`） |
| 主 run | `ts_20260929044120_kj5q`（validated，双门禁通过） |
| DSH 挂载 | `training-sft` 已热注入（`dev_inject_plugin`），工具 `training_sft_*` 可调 |
| **本长任务 GPU 用量** | **0.1752 / 10 GPU 小时** |

---

## 6. 建议的下一步动作（按性价比排序）

```
1. 裁决 QAT 冲突（§1.1）—— 一句话即可解锁 training-qat-contract 的冻结
2. 决定是否要求 wikitext 遗忘指标的多种子复验（§1.2）
3. 若继续训练侧：先做「SFT 产物 → inference-quantization 量化输入」的接口打通（成本低、复用面大）
4. 若进入 RSI：先补推理侧 accept 路径与 physical-limit 停机（长任务 1 遗留）
5. 下次 DSH 重启后：跑一次 training_sft_health 确认正式装配路径（§1.3）
```

---

## 7. 长任务 3 · 阶段 1 结账（2026-09-29）：§1.2 已复验 —— **判据被击穿，pretrain/rl 必须改判据**

**§1.1 与 §1.2 两项均已被用户裁决**（QAT 适用但为受控例外；wikitext 遗忘要求 3 种子复验）。
阶段 1 执行结果见 `SFT_SEED_REPRODUCIBILITY.md`。**结论不是"遗忘稳定"，而是"测量不成立"：**

| 项 | 实测（3 个新种子 42/123/456） |
|---|---|
| 逐种子 ΔNLL | 5.374e-4 / 1.374e-3 / **−1.160e-3** |
| 门禁 | 3/3 全部 pass（无一超界） |
| μ(Δ) / σ(Δ) / 极差 R | 2.506e-4 / **1.291e-3** / 2.535e-3 |
| **R / μ** | **1011%**（用户预声明判据：>10% ⇒ **测量不成立，需重设计**） |
| σ(Δ) / 判据界 | **0.595** |

**根因**：`forgettingGate` 的 `z·SE` 只建模"逐文本测量噪声"，**不含训练随机性**；
实测两者同量级（SE≈1.1e-3 vs σ_seed=1.29e-3），故**单种子门禁结论实质由运气主导**。
已验收那颗（Δ=1.872e-3，余量 5.9%）是 4 个样本的**上界离群点**。

### 对 pretrain / rl 的三条硬约束（**不是建议**）

1. **判据的噪声模型必须包含被判对象的全部随机性来源。**
   判"某一个模型"→ `z·SE` 够用；判"**一套训练流程**"→ 必须叠加种子间方差（`z·SE ⊕ z'·σ_seed`）。
   **pretrain/rl 训练量更大、遗忘风险更高，禁止只用单次 `z·SE` 下"流程级"结论。**
2. **关键结论至少 3 种子。** 单种子结论只能表述为"该模型如此"，不得表述为"该流程如此"。
3. **换测量仪器必须先证等价并落盘。** 本次实测：容器内 HF 前向路径与 serving 路径对 Δ 的估计
   相差 **68% 的判据上限**（容器 0.000516 vs serving 0.001872）⇒ 容器路径被判为**不可用作判定**，
   只能用于诊断。任何"为省 GPU 而换路径"的做法，偏差不得超过判据分辨率的 10%。

### 同时修掉的自身缺陷

`ContainerWorker.stop()` 与容器 `exit` 事件存在竞态（`this.child` 被置空后仍被解引用 →
`TypeError: Cannot read properties of null`）。长任务 2 未暴露，是因 `train.mjs` 的 `finally`
用 `try/catch` 吞掉了。**已修**（`src/worker.mjs`，含根因与"为何以前没暴露"的注释）。

### 预算偏离（需用户裁决）

阶段 1 实耗 **≈0.222 GPU 小时**，上限 **0.15** ⇒ **超 ≈+48%**。
原因是 0.15 这个上限建立在"容器路径可用"的假设上，而该假设已被校准实测证伪（必须走 serving）；
3 次训练本身已占 0.0956（上限的 64%）。**未降低任何判据严格度**，偏离已显式报出。
