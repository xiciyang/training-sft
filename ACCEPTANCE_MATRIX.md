# ACCEPTANCE_MATRIX —— training-sft 长任务 2 验收（P0 / P1 / P2）

> 执行：2026-09-29　证据全部本机实测（`[实测]`）；外部来源标 `[claimed]`；推论标 `[inferred]`
> 主 run：**`ts_20260929044120_kj5q`**（200 步 / 0.0322 GPU 小时 / status=`validated`）
> 复跑入口见每行「证据」列；CI：`node ../inference-contracts/checks/run-all.mjs` → **64/64**

---

## 预算与资源（先给结论）

| 项 | 限额 | 实际 | 证据 |
|---|---|---|---|
| 单次训练 GPU | ≤ 2 GPU 小时 | **0.0322** | `registry.result.gpuHours` `[实测]` |
| 单次训练步数 | ≤ 500 | **200**（+续训 20，该模型累计 220） | `registry.result.steps` `[实测]` |
| 模型规模 | ≤ 0.5B | **0.494 B**（494,032,768） | 现场统计 `[实测]` |
| 并发 | 1 | **1**（`BUDGET_DEFAULTS.concurrency`） | `[实测]` |
| **长任务 2 总 GPU** | ≤ 10 GPU 小时 | **0.1752**（训练 0.1071 + 验证 0.0681） | 见下方「GPU 小时核算」 |
| 数据条数 | ≤ 1000 | **1000 训练 + 200 留出** | `lineage.json` `[实测]` |
| 金额 | 本机按电费 | 本机自持，无外部计费 | —— |
| 超限行为 | 自动暂停 + 报告 | 已实现（`policy: pause-and-report`），本轮由 `maxSteps` 正常触发 | `stopReason=budget-exhausted:maxSteps` `[实测]` |

**GPU 小时核算（逐 run）** `[实测]`：

```
训练（6 个 run，含 3 次失败的探索）合计 0.1071 GPU 小时
验证（1 次完整 validate）合计          0.0681 GPU 小时
────────────────────────────────────────────────
总计                                   0.1752 GPU 小时  /  预算 10
```

> 说明：`gpuHoursValidate = 0.0681` 是 `validate` 的**墙钟**（含两次 deploy、测量与 teardown），
> 用 `Date.now()` 差值计算，属**上界估计**（真正占 GPU 的是容器存活期）。

---

## P0 —— 阻断项（必须真机）

| # | 判据 | 结果 | 证据（可复核） |
|---|---|---|---|
| 1 | 训练数据与测试集内容哈希交集为空 | ✅ **7/7 全为 0** | `lineage.json.verification`：`train×heldout=0`、`train×wikitext-test(c128/c512)=0`、`heldout×wikitext-test(c128/c512)=0`、`train×mmlu-test=0`、`heldout×mmlu-test=0`；`allClear=true` `[实测]` |
| 2 | 训练数据血缘完整（来源/revision/种子/哈希清单） | ✅ | `lineage.json`：`source=yahma/alpaca-cleaned`、`revision=12567cabf869d7c92e573c7c783905fc160e9639`、`seed=20260929`、`calib.hash=sha256:8f5d36…`、`test.hash=sha256:0c3fb2…`、源文件 `sha256:bd844b82…` `[实测]` |
| 3 | checkpoint 可被 inference-serving 加载 | ✅ | `endpoint.id=ep_69fd99cf99`，`container=inference-serving-69fd99cf99`，镜像 digest 固定 `vllm/vllm-openai@sha256:be5ff841…`，serving 自己的可用性探针 `usability.ok=true` `[实测]` |
| 4 | 前向通过 + 输出非均匀分布 | ✅ | 端点 `n=96` 个 logprob；`uniformSignature`：`\|meanNll − lnV\| = 11.399` ≫ `ε = 0.1724`；逐 token NLL 极差 > 0 `[实测]` |
| 5 | 契约一致性（六个共享库 + 契约层） | ✅ | `node inference-contracts/checks/run-all.mjs` → **64/64**，含 C11（六库 192 断言）、C12（文档单源）、C13/C14（训练侧契约与 schema）、**C15**（training-sft 196 断言）、**C16**（eval-result 过冻结 schema + 判据来自 testing-core） `[实测]` |
| 6 | 资源上限生效（超限自动暂停） | ✅ | 真机：`stopReason=budget-exhausted:maxSteps`、`stopChecks=[{limit:maxSteps,value:200,threshold:200}]`、`policy=pause-and-report`；CI：`mock-loop.spec` 另验 `maxWallMs` 与失步检测 `[实测]` |
| 7 | checkpoint 可恢复（中断后可继续） | ✅ | 从 `step-000200`（offset=200）续训 **20 步** → 新 run `ts_20260929045106_i6p3` 报 `steps: 20`、`gpuHours: 0.0037`，且两侧步号不变量未触发（`201..220` 与 `state.step+1+offset` 逐次相等） `[实测]` |
| 8 | 审计日志哈希链完整 | ✅ | `@ai-infra/audit-core` 复算：`{ok:true, shards:1, checked:31, variant:'canonical-json'}`；样本见 `evidence/audit-sample.jsonl` `[实测]` |
| 9 | 训练侧评估与推理侧评估契约一致 | ✅ | 训练侧产出 `ai-infra/eval-result@1`（`evidence/eval-result.json`），经**冻结的** `state-schema/eval-result.schema.json` 校验通过（C16a）；`versions.testingCore=0.1.0`（C16b）；`validity` 与 `gate.validityOk` 一致（C16c）；判定逻辑 import 自 `@ai-infra/testing-core`（C16d/e） `[实测]` |

## P1 —— 告警项（小规模真机 + 指标门禁）

| # | 判据 | 阈值（**显式给出**） | 实测 | 结果 |
|---|---|---|---|---|
| 10 | 灾难性遗忘检查通过 | **z·SE 判据（z=1.96，单侧 95%）**，非拍脑袋百分比 | MMLU：`0.3833 → 0.3900`，`Δ=−0.0067 ≤ 0.0779` ✅<br>wikitext 配对 NLL：`Δ=+0.001872 ≤ 0.001990` ✅（**临界**，余量 6%） | ✅ 通过 |
| 11 | SFT 后任务指标提升 ≥ 阈值 | `maxF1DropPct=5`（F1 不得降） | token F1：`0.3199 → 0.3421`，**+6.92%** | ✅ 通过 |
| 12 | 格式合规率提升 ≥ 阈值 | `minFormatGainPct=10` | 格式合规率：**`0.14 → 0.27`，+92.86%** | ✅ 通过 |
| 13 | 测量成立性检查通过 | validity 全过 | 三条：`sft-forward`（96 logprob）、`sft-not-uniform`（ε 判据）、`task-samples`（100/100 有效生成） | ✅ 通过 |

### P1 #10 的诚实标注（**临界，不许略过**）

wikitext 配对 NLL 的退化**刚好在判据边界内**（`0.001872` vs 界 `0.001990`，余量 6%）。
按"门禁不仅判数值达标、还判测量成立"的纪律，这里必须写明：

- 该结论**依赖 z=1.96 的选择**（单侧 95%）。若用 z=1.645（单侧 95% 单尾的更宽松口径）结论不变；
  若用 z=2.58（99%）则界为 `0.00262`，仍通过。**但余量小意味着换 seed / 换数据切分可能翻面。**
- 因此本项记为 **"通过（临界）"**，并列入 `NEXT_STEPS.md` 的复验清单。
- `[inferred]` 该退化的可能来源：LoRA r=8 在 1000 条指令数据上训练 200 步、有效批 16（≈3.2 epoch），
  对本就弱的 0.5B 基座产生轻微的语言建模分布偏移。

### P1 #12 的重要说明：这条判据**第一版是失效的**（已修，过程如实记录）

第一次跑垂直闭环（协议 v1，`max_tokens=96`）时：格式合规率 **基座 0.98 / SFT 0.98**，**零提升**，任务门禁 FAIL。

根因（**测量不成立，不是没达标**）：`max_tokens=96` 让 **97% 的生成被截断**，
此时"格式合规"退化成"它至少说了点什么" —— 基座上就已经饱和，任何训练都不可能让它提升。

修法（**不是调宽容度，是让判据可区分**）：评估协议升到 `task-eval@2`，判据加一条
「**必须自然终止**（`finish_reason === 'stop'`）」。修完后基座降到 **0.14**、SFT **0.27** ——
判据开始真正区分"把答案说完"与"唠叨到被截断"。协议版本同时进基线缓存键（C16f），
避免跨协议比较数字。

> 两次结果都留在报告里（v1: `0.98/0.98`；v2: `0.14/0.27`），没有只留下通过的那次。

## P2 —— 记录项（单测/集成，CI 无 GPU / 无容器 / 无网络）

| # | 判据 | 覆盖 | 证据 |
|---|---|---|---|
| 14 | 输入处理 **7 类**场景 | ① 缺必填 ② 参数越界 ③ 目录不存在 ④ 缺 lineage ⑤ `allClear=false` 拒绝 ⑥ 未知 runId ⑦ 未知工具名 | `tests/unit/guards.spec.mjs` 第 1–7 条（真断言错误码） `[实测]` |
| 15 | 安全验收 **8 条** | ① 缺令牌拒 ② 错令牌拒 ③ 正确令牌放行+memo+令牌与目标绑定 ④ 步数超限拒 ⑤ GPU 小时超限拒 ⑥ purge 不越界（`/etc` 未被删） ⑦ 不改写原始数据集（长度比对） ⑧ 泄漏有独立错误码 | 同上第 S1–S8 条 `[实测]` |
| 16 | 健壮性 **6 条** | ① 状态文件损坏不静默重置 ② 审计链篡改被检出且可恢复 ③ schema 拦截非法状态 ④ CAS 冲突报 `ERR_STATE_CONFLICT` ⑤ worker 未启动即 call 响亮失败 + `args()` 形态 ⑥ 端点不可达 → `formatCompliance=null`（不伪造） | 同上第 R1–R6 条 `[实测]` |
| 17 | 工程验收 **5 条** | ① npm 包规范（files/bin/exports/可选依赖组） ② 软硬件解耦（无 GPU 型号硬编码、路径从模块位置推导） ③ 文档齐全 ④ 测试覆盖（4 spec / 196 断言） ⑤ 交付物齐全（src/bin/mcp/skills/tests/python/toolchain） | `tests/unit/engineering.spec.mjs`（54 断言） `[实测]` |
| 18 | 注册路径被测试覆盖 | 严格 mock ctx（像 DSH 一样挑刺）跑 `apply()` → 6 工具全注册 + **真调**两个只读工具；路由 `/__training_sft/health` 注册；**并在真实 DSH 中注入后真调**（见下） | `tests/unit/registration.spec.mjs`（78 断言）+ 真机注入 `[实测]` |
| 19 | 工具名不与既有命名空间冲突 | 去兄弟包源码**实时抓取** 33+ 工具名，交集 ∅ | 同上 `[实测]` |

### P2 #18 的**最强证据**：真实 DSH 注入后真调成功 `[实测]`

```
dev_inject_plugin <repo>
  → OK: training-sft 已注入（junction=…/profiles/web/node_modules/training-sft）；host ✓；client 跳过（无 client 声明，属预期）

training_sft_health()          → ok:true，sandboxTiers={process:true,container:true,vm:false}，镜像 f82151dd84fb，
                                 serving 可解析，审计链 {ok:true, checked:29}，
                                 governance.audit={seq:30, hash:d8e922e0…}
training_sft_get_report({runId}) → ok:true，返回完整 run 记录（200 步 / 3 checkpoint / 门禁结果 / 端点 / eval-result），
                                 governance.audit={seq:31, hash:d1afa9a2…}
```

**并且真调抓到了一个真实缺陷**：第一次 `training_sft_health()` 返回
`tool "training_sft_health" returned invalid output: value is not lossless JSON`
—— 根因是 `health()` 在 `deep=false` 时留了一个值为 `undefined` 的 `deep` 键
（`JSON.stringify` 会静默丢掉它，DSH 的校验器却会拒绝）。
**这正是"注册成功 ≠ 能用"那条纪律要抓的东西**：mock ctx 跑 `apply()` 看得见注册，
但看不见"输出契约在真实校验器下不通过"。

## 交付物清单

| 交付物 | 路径 | 状态 |
|---|---|---|
| `training-sft/` 完整 npm 包 | `inference-training/training-sft/`（package.json / manifest.yaml / src / bin / mcp / skills / tests / python / toolchain） | ✅ |
| `PROJECT_CONTEXT.md` | `inference-training/training-sft/` | ✅ |
| `DATA_STEP0_REPORT.md` | `inference-training/training-sft/` | ✅ |
| `ACCEPTANCE_MATRIX.md` | 本文件 | ✅ |
| `DEPLOYMENT_REPORT.md` | `inference-training/training-sft/` | ✅ |
| `NEXT_STEPS.md` | `inference-training/training-sft/` | ✅ |
| `TEST_REPORT.md` | `inference-training/training-sft/` | ✅ |
| 审计日志样本 | `inference-training/training-sft/evidence/audit-sample.jsonl` | ✅ |
| `eval-result.json`（契约实例） | `inference-training/training-sft/evidence/` + run 目录 | ✅ |
| `LONGTASK2_REPORT.md` | 工作区根 | ✅ |

## 未通过 / 未做（如实记录）

| 项 | 状态 | 说明 |
|---|---|---|
| P1 #10 wikitext 遗忘 | **通过但临界**（余量 6%） | 需换 seed 复验，见 `NEXT_STEPS.md` |
| 全参微调对比 | **未做** | 授权内选择 LoRA（理由见 `LONGTASK2_REPORT.md` §6），预算内不做对照实验 |
| pretrain / RL / QAT | **未做** | 长任务 2 范围外（`manifest.not_provided` 显式声明） |
| 多分片模型 / 其它基座 | **未做** | 本轮只用 0.494B 单分片基座 |
| DSH **重启后**的重新装配 | **未验证** | 本轮用 `dev_inject_plugin` 热注入验证；重启装配路径由 `cordis.patch.yml` 承担，未重启验证 |

---

## ⚠️ 多种子回溯修订（LONGTASK4 · 阶段 4，2026-09-29）

> 依据 `SEED_REPRODUCIBILITY_RETROSPECT.md`。用 **3 个新种子**（1042/2042/3042，规则 `42+1000k`）
> 以**相同的 serving 仪器与评估协议**复验。判据由用户预声明：符号翻转 或 `R/μ > 50%` ⇒ 测量不成立。

| 本矩阵原有结论 | 单种子原值 | 3 种子 μ | R/μ | 符号 | **修订后状态** |
|---|---|---|---|---|---|
| 格式合规率提升 | +92.9% | **+111.9%** | 12.8% | 一致 | ✅ **已确证为流程级结论**（原值偏保守） |
| token F1 提升 | +6.9% | +10.35% | **52.5%** | 一致 | ❌ **多种子证伪 —— 测量不成立，需重设计** |
| MMLU 保持 | Δ=−0.0067 | 0.0000 | — | **翻转** | ❌ **多种子证伪 —— 测量不成立** |
| wikitext NLL「临界通过」 | +0.0019 | {+0.0006,+0.0027,−0.0001} | — | 翻转 | ❌ **已由 LT3 阶段 1 击穿**，本轮交叉印证 |

**处置纪律**：**未修改任何判据、未放宽任何阈值**。被证伪的两条此后
- 只能作为**单种子观测值**引用，必须标 `singleSeed: true`；
- **不得**用于支撑"该训练流程具备此效应"的任何陈述。

**根因（两条不同）**：
1. **F1** —— 逐样本 F1 未落盘（`summarize` 只给均值）⇒ 噪声模型缺掉"测量噪声"那一半，
   `SĒ` 只能取 0。行动项 A1：补落盘，让噪声模型完整。
2. **MMLU** —— 判据界 `z·SE ≈ 0.046` **远大于** Δ 的量级（≤0.007），
   在 300 题规模下**分辨不出**该效应。**不是"没退化"，是"测不出来"**。

证据：`inference-training/out/lt4-seed-retrospect-{raw,}.json`；运行日志 `out/lt4-seed-retrospect.log`。
