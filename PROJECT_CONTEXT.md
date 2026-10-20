# PROJECT_CONTEXT —— training-sft 长任务 2 / 阶段 1（Step 0 只读调研）

> 执行：2026-09-29　性质：**只读**（未训练、未改任何推理侧插件、未改共享库）
> 状态：**阶段 1 完成 → 触发两条「必须暂停报告」条件 → 暂停等裁决**（见 `DATA_STEP0_REPORT.md` §裁决）
> 取证口径：`[实测]` = 本机命令/输出 · `[claimed]` = 外部来源（附 URL）· `[inferred]` = 推论

---

## 1. Step 0 八项结论速览

| # | 调研项 | 结论 |
|---|---|---|
| 1 | 训练框架可用性 | ✅ 容器镜像 `inference-quantization/toolchain:0.1.0` 自带 torch 2.11.0+cu130 / transformers 5.17.0 / accelerate 1.15.0 / datasets 5.0.1 / bitsandbytes 0.49.2；**peft 缺失但可安装**（peft 0.21.0，已实测）；宿主机另有 torch 2.11.0+cu128 / transformers 5.3.0（无 peft） |
| 2 | 数据源 | ⚠️ **两条暂停项**：`tatsu-lab/alpaca` 实测 license = **CC BY-NC 4.0（非商用）**，与题面"Apache 2.0"不符；两个候选**均含非占位 PII** |
| 3 | GPU 资源 | ✅ NVIDIA GB10；宿主内存 119 GiB 总 / **97 GiB 可用**；磁盘 **572 GiB 可用**；GPU 已被既有 `qwen3-embedding` 容器占用（不动它） |
| 4 | 沙箱能力（行为验证） | ✅ `container` 可用（`docker run --rm alpine echo ok` → exit 0 / 110 ms）；`process` 行为上可用但**策略上降级**（无 namespace 隔离保证 → 写操作提升到容器）；`vm` **不可用**（`qemu-system-aarch64` ENOENT） |
| 5 | testing-core 训练侧扩展点 | ✅ 11 个导出；训练侧**判定**逻辑可直接复用（`dualGate` / `relativeChange` / `perplexity` / `uniformSignature`）；**指标抽取**（格式合规率等任务指标）属 training-sft，不另造判定逻辑 |
| 6 | training-core 接口 | ✅ 14 个导出，接口已冻结（`createTrainingLoop` / `saveCheckpoint` / `loadCheckpoint` / `enforceLimits` / `createAuditHook` / `BUDGET_DEFAULTS`） |
| 7 | 与 serving 的接口 | ✅ `inference_serving_deploy` 接受本地目录；**要求**：目录内需含 `README.md` 的 `license:` frontmatter 或 `LICENSE` 文件，否则 license 判为 `unknown` → `decision: ask` |
| 8 | 基座模型缓存 | ✅ `Qwen/Qwen2.5-0.5B-Instruct` @ `<BASE_MODEL>`（**0.494B**、apache-2.0、rev `7ae557604adf67be50417f59c2c2f167def9a775`） |

---

## 2. 训练框架与运行时（Step 0.1）

### 2.1 宿主 `[实测]`

| 包 | 版本 |
|---|---|
| torch | 2.11.0+cu128（`cuda.is_available() = True`，`NVIDIA GB10`） |
| transformers | 5.3.0 |
| safetensors / numpy | 0.7.0 / 2.4.2 |
| peft / trl / accelerate / datasets / bitsandbytes | **缺失** |

### 2.2 容器 `[实测]`

| 镜像 | torch | transformers | accelerate | datasets | bnb | peft |
|---|---|---|---|---|---|---|
| `inference-quantization/toolchain:0.1.0`（21.7 GB，已在盘） | 2.11.0+cu130 | 5.17.0 | 1.15.0 | 5.0.1 | 0.49.2 | ✗（可装 0.21.0） |
| `vllm/vllm-openai:nightly` | 2.11.0+cu130 | 5.12.1 | 1.14.0 | ✗ | 0.49.2 | ✗ |

### 2.3 训练运行位置的决定

**决定：训练在容器沙箱（R2）内执行，基准镜像 `inference-quantization/toolchain:0.1.0` + 固定 `peft==0.21.0`。**

依据（三条，均可复核）：
1. **策略**：训练产出新权重属 `write` 档；`@ai-infra/sandbox-core` 的 `DEFAULT_DEGRADATION.process = {policy:'upgrade', to:'container'}`，
   即"process 层隔离强度不足 → 提升到容器层"。这是长任务 1 已冻结的**策略**，不是本轮新定。
2. **本机行为实测**：`container` 可用 ✓、`vm` 不可用 ✗（见 §4）。
3. **可复现**：镜像层固定（基准镜像已在盘 + 一层 `pip install peft==0.21.0`），
   与项目既有纪律「镜像 digest 固定」一致；比"运行时 pip install 到挂载卷"更可复现。

### 2.4 PEFT 可行性**已知阳性对照** `[实测]`

在容器内对真实基座做了一次最小 LoRA 冒烟（**不构成训练，未产出 checkpoint**）：

```
loaded in 4.6 s; dtype torch.bfloat16
LoRA r=8 on q/k/v/o → trainable 1,081,344 (0.218%) / total 495,114,112
loss 3.859 · grad_norm 12.36 · optimizer.step() ok · cuda max mem 0.97 GiB
```

即：**框架链路在真机上确实能跑**（加载 → 注入 LoRA → 前向 → 反向 → 优化器步），不是"装了就算"。

---

## 3. 数据侧（Step 0.2）—— 详见 `DATA_STEP0_REPORT.md`

| 候选 | 条数 | license（实测） | class | 非占位 PII | 与 wikitext-test 交集 | 与 MMLU-test 交集 |
|---|---:|---|---|---:|---:|---:|
| A `tatsu-lab/alpaca` | 52,002 | **cc-by-nc-4.0** | **restricted** | **35** | 0 | 3 |
| B `yahma/alpaca-cleaned` | 51,760 | cc-by-4.0 | permissive | **54** | 0 | 5 |

候选 A × 候选 B 内容哈希交集 = **1,832**（两者是同一上游的清洗前后版本）。

**两条「必须暂停报告」条件已触发**：① 题面 license 前提与实测严重不符；② 发现 PII。

---

## 4. 沙箱能力（Step 0.3，**行为验证**不是存在性检查）`[实测]`

用 `@ai-infra/sandbox-core` 的 `probeTiers()` 实跑（不是 `which`）：

| 层 | available | 证据 |
|---|---|---|
| `process` | **true** | 受限子进程跑 `true` → exit 0 / 18 ms |
| `container` | **true** | `docker run --rm --network=none --read-only alpine:3.20 sh -lc 'echo ok'` → exit 0 / 110 ms / stdout `ok` |
| `vm` | **false** | `spawn qemu-system-aarch64 ENOENT` |
| `none` | true | 按定义（R0 只读不需要沙箱） |

⇒ 训练（write）走容器层；`vm` 层不可用意味着**系统级动作一律拒绝**（v0 无此需求）。

---

## 5. 共享库接口确认（Step 0.5–0.6）

### 5.1 `testing-core`（判定逻辑的唯一实现）

导出面（11）：`CONTRACT_VERSION, dualGate, floatFingerprint, lnVocab, mean, meanNllFromLogprobs, nllFromLogprob, perplexity, relativeChange, stddev, uniformSignature`

**边界划分（避免"绕开契约"）**：

| 事项 | 归属 | 理由 |
|---|---|---|
| 双重门禁判定（validity + 数值） | **`testing-core.dualGate`** | 已冻结；训练侧不得自写一份 |
| 相对变化 / 方向 | **`testing-core.relativeChange`** | 同上 |
| 困惑度 / 均匀签名 | **`testing-core.perplexity` / `uniformSignature`** | 同上；ε 的推导已冻结 |
| 浮点指纹（逐位判定） | **`testing-core.floatFingerprint`** | 同上 |
| **指标抽取**（格式合规率、任务命中率） | `training-sft` | 任务相关，不是通用判据；抽出来的数**交给** `dualGate` 判定 |

### 5.2 `training-core`（公共循环的唯一实现）

导出面（14）：`BUDGET_DEFAULTS, CONTRACT_VERSION, LIMIT_KEYS, assertTrainingPlugin, canonicalJson, canonicalize, createAuditHook, createTrainingLoop, enforceLimits, listCheckpoints, loadCheckpoint, removeCheckpoint, saveCheckpoint, stateHash`

调用契约（已冻结）：`createTrainingLoop({ plugin, batchSource, limits, checkpoint, onEvent, clock, hooks, epochs })`；
插件对象 `{ id, algorithm, prepare, step(batch,state), finalize, checkpointPayload }`。
`enforceLimits` 返回 `{stop, reason, checks, policy:'pause-and-report'}` —— **超限暂停，不降级**。

> **一处需要在阶段 2 显式处理的接口空隙**（不是缺陷，是 v0 的既定边界）：
> `saveCheckpoint` 只写 `payload.json`（JSON）。真实 LoRA 权重是二进制 ——
> 按 `checkpoint-contract.md` §4 的约定，**权重文件由插件自己写到权重目录、在 `meta.files[]` 里逐文件登记哈希**。
> 本任务将按该约定实现，**不修改** `training-core`（保持 v0 冻结面不变）。

---

## 6. 与 serving 的接口（Step 0.7）`[实测]`

| 事实 | 证据 |
|---|---|
| `inference_serving_deploy` 接受本地目录（`model` 或 `path`） | `src/tools.mjs` 入参 |
| 部署成功会做**可用性探针**（加载 + 前向 + 非均匀输出） | 长任务 1 的 deploy report：`modelUsability.ok = true`，容差 `1e-4`（硬编码） |
| license 判定来源 = `README.md` 的 `license:` frontmatter **或** `LICENSE` 文件 | `src/probe/model.mjs:151-161` |
| license 缺失 → `class: unknown` → `decision: ask`；permissive → `allow` | `src/probe/model.mjs:141-148` |

⇒ **合并后的模型目录必须自带 `LICENSE` + 含 `license:` frontmatter 的 `README.md`**，
否则 serving 的许可闸门会落到 `ask`。这是一条**必须写进实现**的接口约束。

---

## 7. 基座模型缓存（Step 0.8）

| 项 | 值 |
|---|---|
| 模型 | `Qwen/Qwen2.5-0.5B-Instruct` |
| 路径 | `<BASE_MODEL>` |
| 参数量 | **494,032,768 = 0.494 B**（现场统计） |
| 磁盘 | 999,605,158 字节（954 MiB） |
| license | `apache-2.0`（`LICENSE` 文件在） |
| revision | `7ae557604adf67be50417f59c2c2f167def9a775` |

### 与长任务 1 checkpoint 兼容测试模型的**异同**（报告要求第 1 项）

| 维度 | 长任务 1（B2） | 长任务 2 |
|---|---|---|
| 基座 | `Qwen2.5-0.5B-Instruct` | **同一个**（同一路径、同一 revision） |
| 用途 | **只读加载 + 前向 + 分布检查**（不改权重） | **LoRA 微调 → 合并 → 产出新 checkpoint** |
| 产物 | 无（不产新模型） | 新的合并模型目录（权重与基座**不同**） |
| LT1 冻结证据可否沿用 | —— | **不可**。`checkpoint-contract.md` 的冻结证据绑定的是**基座**的哈希；新 checkpoint 是**新产物**，必须按同三项判据（加载 / 前向 / 非均匀）**重新验证**，并登记自己的 `payloadHash` 与 `hashes.weightsSha256` |
| vocab_size | 151936 | 同（合并不换 tokenizer） |

---

## 8. 阶段 2 的既定技术路线（**待裁决后执行，尚未动工**）

1. **薄插件**：`training-sft` npm 包（`src/` `bin/` `mcp/` `skills/` `tests/`），
   工具面 `training_sft_prepare|train|validate|rollback|get_report|health`，路由 `/__training_sft/health`，
   状态 `~/.ai-infra/state/sft-registry.yaml`（走 `state-core` 方言 + schema），
   审计 `~/.ai-infra/logs/audit/training-sft/{date}.jsonl`（走 `audit-core` 链）。
2. **LoRA 而非全参**（建议理由）：可训练参数 **0.218%**（1,081,344 / 495,114,112），
   单次显存 ~1 GiB（冒烟实测），合并后即为可被 serving 直接加载的完整 HF 模型；
   全参微调在 500 步 / ≤2 GPU 小时的预算内既无必要也更容易破坏通用能力（灾难性遗忘风险更高）。
3. **数据切分**：训练 ≤1000 条 + 留出评估 ≤200 条（**同源不同行，内容哈希验证不相交**），
   且**剔除任何与评估集（wikitext-test / MMLU-test）内容哈希相撞的行** —— 这是 P0 #1 的实现方式。
4. **门禁**：精度/格式/安全 + 灾难性遗忘（通用能力保持率阈值 **须显式给出**，
   不得内置默认值 —— `eval-contract.md` 的既有纪律）。

---

## 9. 本轮**未做**的事（范围自查）

- 未训练（未消耗 GPU 小时；仅一次 4.6 s 的 LoRA 冒烟用于验证框架链路）
- 未产出任何 checkpoint
- 未修改推理侧三插件 `src/`
- 未修改 `dsh-tuning-engine`
- 未修改六个共享库（v0 冻结面保持不变）
- 未合成任何数据、未对 PII 做脱敏
- 未创建任何常驻容器（冒烟容器 `--rm` 已回收）
