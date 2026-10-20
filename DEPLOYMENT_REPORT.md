# DEPLOYMENT_REPORT —— training-sft 真机演示（长任务 2 / 垂直闭环）

> 执行：2026-09-29　机器：NVIDIA GB10（aarch64 / 统一内存 119 GiB / 驱动 580.126.09）
> 主 run：`ts_20260929044120_kj5q`　全程真机，无 mock。
> 原始日志：`evidence/validate-run.log`（完整 validate 输出）· `evidence/audit-sample.jsonl`

---

## 1. 一句话

**数据 → SFT（LoRA）→ 合并 → serving 部署 → 推理 → 评估 → 与基线对比 → 门禁，一次跑通，双门禁通过。**

---

## 2. 闭环五步（逐步证据）

### 第 0 步：环境自检（`training_sft_health`，经真实 DSH 工具调用）

```json
{ "ok": true, "sandboxTiers": { "process": true, "container": true, "vm": false },
  "sandboxEvidence": { "cmd": "echo ok", "exitCode": 0, "ms": 142, "runnerMethod": "executed" },
  "image": { "name": "training-sft/toolchain:0.1.0", "available": true, "id": "f82151dd84fb" },
  "gpu": "NVIDIA GB10, 580.126.09",
  "baseModel": { "path": "/data/models/Qwen2.5-0.5B-Instruct", "available": true },
  "serving": { "available": true, "serverPath": ".../inference-serving/mcp/server.mjs" },
  "audit": { "ok": true, "shards": 1, "detail": [{ "day": "2026-09-29", "checked": 29, "ok": true }] },
  "limits": { "maxSteps": 200, "maxGpuHours": 2, "concurrency": 1, "maxParams": 500000000 } }
```

沙箱判定是**行为验证**（真的跑了命令看退出码），不是 `which`。

### 第 1 步：数据准备（`training_sft_prepare`）

```
51760 源  → 51756 去重（−4） → 51733 剔除含非占位 PII 行（−23）
          → 51733 整行哈希与评估集相撞（−0） → 51723 行内含评估集文本（−10）
          → 切分：train 1000 / heldout 200（seed 20260929）
          → 复核 7 组内容哈希交集：全为 0  → allClear = true
```

**注意第 4 道过滤**：整行哈希相等只能判"逐行相同"，判不出"评估集文本被**嵌在**训练行内部"。
实测确实有 **10 行** 含 MMLU/维基的整段文本 —— 这是比哈希相等更严的一条判据，
由"证据采样必须防止长尾偏斜"的同类思路补上。

### 第 2 步：SFT 训练（`training_sft_train`）

```
base  = Qwen/Qwen2.5-0.5B-Instruct（0.494B，rev 7ae557604adf67be50417f59c2c2f167def9a775）
LoRA  = r=8, alpha=16, dropout=0, target=[q,k,v,o]_proj  → trainable 1,081,344（0.218%）
优化  = AdamW lr 1e-4（warmup 10 + cosine），batch 4 × gradAccum 4（有效批 16），maxSeqLen 256
循环  = @ai-infra/training-core 的 createTrainingLoop（**公共循环**，非自研）
沙箱  = 容器（training-sft/toolchain:0.1.0，容器内以宿主 uid:gid 运行）

step  25 loss=1.1183 gpuHours=0.004
step  50 loss=1.1167 gpuHours=0.009
...
step 200 loss=1.1518 gpuHours=0.033
→ steps=200  gpuHours=0.0322  stopReason=budget-exhausted:maxSteps  checkpoints=3
```

**每一步都校验两侧步号一致**（容器回 `stepIndex`，Node 断言 `== state.step+1(+offset)`）。

### 第 3 步：合并导出（`mergeRun`）

```
mergedDir        = …/ts_20260929044120_kj5q/merged
weightsSha256    = sha256:3a08a5ba363d915989bc159060057e07da8e1f85ad81343b541ee1fb36c0e975
params           = 494,032,768     bytes = 999,536,677     files = 9
```

合并目录自带 `LICENSE`（Apache-2.0，从基座复制）+ 含 `license:` frontmatter 的 `README.md`
+ `training-sft-provenance.json` —— 这是 **serving 的许可闸门可判定的前提**（实测约束，见 PROJECT_CONTEXT §6）。

### 第 4 步：部署 + 推理（经 `inference-serving` 的加载器）

```
inference_serving_deploy(path=merged, servedModelName=sft-ts_20260929044120_kj5q)
→ ok:true  endpoint=http://127.0.0.1:8100  id=ep_69fd99cf99
   imageDigest = vllm/vllm-openai@sha256:be5ff841eb156c3d1be3e318e33ae682eaaa0c967f05e96115bb39333bace392
   usability.ok = true  "usable：模型确实参与了前向（非均匀输出）"
     tokensScored 23 · meanNll 1.7096 · uniformLoss 11.9312 · itemRange 9.5512 · topMargin 6.1875
```

### 第 5 步：评估 + 门禁（同一协议测 SFT 与基座）

| 指标 | 基座 | SFT | 变化 | 判据 |
|---|---|---|---|---|
| 格式合规率（`task-eval@2`） | **0.14** | **0.27** | **+92.86%** | ≥ +10% ✅ |
| token F1（heldout 100 条） | 0.3199 | 0.3421 | **+6.92%** | 不降超 5% ✅ |
| MMLU 准确率（300 题） | 0.3833 | 0.3900 | Δ=−0.0067（界 0.0779） | z·SE ✅ |
| wikitext NLL（200 段，配对） | 3.25026 | 3.25213 | Δ=+0.001872（界 0.001990） | z·SE ✅（**临界**） |
| 均匀签名 | `ok`，n=96，bitIdenticalAcrossRuns=true | 同 | —— | 有效性 ✅ |

```
taskGate      = { pass:true,  validityOk:true, reason:"达标" }
forgettingGate= { pass:true,  z:1.96, reasons:[] }
→ run status = validated
```

---

## 3. 端点生命周期（零容器泄漏）

```
SFT 端点  ep_69fd99cf99 → deploy → 测量 → teardown（带 ownerToken；只拆自己租用的端点）
基座端点             → deploy → 测量 → teardown
```

事后核对 `[实测]`：

```
docker ps -a --filter label=inference-serving.managed=true  → 0
docker ps -a --filter ancestor=training-sft/toolchain:0.1.0 → 0
nvidia-smi --query-compute-apps → 仅剩既有的 qwen3-embedding（15512，4583 MiB，**本轮未触碰**）
```

---

## 4. 审计链（真机产物）

落点：`~/.ai-infra/logs/audit/training-sft/2026-09-29.jsonl`，**31 条**，事件类型：

```
prepare.begin / prepare.end
train.begin / train.end（runId / steps / wallMs / gpuHours / finalLoss / checkpoints）
merge.end（weightsSha256 / params / bytes）
validate.begin / validate.end（taskGate / validityOk / forgettingGate / 各项指标）
rollback.begin / rollback.end
capability-call（每次工具调用的摘要留痕，含失败）
```

首条样本（`evidence/audit-sample.jsonl`）：

```json
{"schema":"training-sft/audit@1","seq":1,"ts":"2026-09-29T04:26:27.937Z","pkg":"training-sft","pkgVersion":"0.1.0",
 "capability":"prepare","action":"prepare","phase":"begin",
 "argsDigest":"sha256:bfb19d4014da463ad02a47e6b52544c705abacdb1040f93b797fd942bddc5447",
 "argsKeys":["dataset","heldout","seed","train"],
 "prevHash":"0000…0000","hash":"c5cd041db79c954ef7ffa3a1451e9b8028dc42db29e90346368c4113c97ce696"}
```

复算（`@ai-infra/audit-core`，独立实现）：`{ok:true, shards:1, checked:31, ok:true, variant:'canonical-json'}`。

**只追加**：`O_APPEND` + `0600` + API 层无删改导出。记录**只存参数摘要**（`argsDigest`）与参数名，不存参数值。

---

## 5. 本轮真机踩到并修掉的坑（全部有日志）

| # | 现象 | 根因 | 修法 |
|---|---|---|---|
| 1 | `python3: can't open file '/work/trainer.py'` | 训练脚本目录没挂进容器 | `trainingMounts` 增加 `${pkg}/python:/work:ro` |
| 2 | `Repo id must be in the form 'repo_name'…` | 模型目录挂成了 `/models`（容器内看到的是**目录内容**，transformers 找不到 config.json 就当 HF repo id） | 挂成 `/models/<basename>` |
| 3 | 宿主写 `payload.json` 报 `EACCES` | 容器以 root 建目录，宿主（uid 1000）改不动 | 容器加 `--user $(id -u):$(id -g)`，并把 `HOME`/缓存指到容器内 `/tmp` |
| 4 | `checkpoints/` 只留 3 个但 `runDir` 堆了 4 个权重目录 | 插件把权重写在 `runDir/ckpt-*`，training-core 的 `keep` 只删自己的目录 | 权重与 training-core 的 checkpoint **同名同目录**（`<cpDir>/<name>/adapter`），删除时一起走 |
| 5 | "续训 20 步"实际跑了 220 步，且**不报错** | 把 `limits.maxSteps = offset+新增` 直接喂给循环 | 循环只跑"新增"步数；预算闸门单独判 `offset+新增 ≤ 500`；LR 调度用总步数 |
| 6 | `training_sft_health` 被 DSH 拒绝：`not lossless JSON` | `deep=false` 时留了值为 `undefined` 的键 | 用条件展开，不产生 `undefined` 属性 |

> 第 5 条是最危险的一条：**它不报错**，只是把"续训 20 步"变成"续训 220 步"，
> 两侧步号断言还全对。只有把"预算该等于哪一侧的步数"想清楚才会发现。

---

## 6. 复跑方式

```bash
cd inference-training/training-sft
node tests/run.mjs                                   # CI（196 断言，无 GPU）
node tests/realmachine/docker-worker.spec.mjs        # 真机 worker（需 docker + 镜像）
node -e "import('./src/core.mjs').then(m=>m.health({}))"   # 环境自检
# 完整闭环：
#   training_sft_prepare → training_sft_train → training_sft_validate → training_sft_get_report
cd ../../ && node inference-contracts/checks/run-all.mjs   # 契约层 64/64
```

---

## ⚠️ 交付后修订：多种子回溯结果（LONGTASK4 · 阶段 4）

本报告中的 **token F1 提升（+6.9%）** 与 **MMLU 保持** 两项结论，
经 3 个新种子（1042/2042/3042）以相同 serving 仪器、相同协议复验后**被证伪**：

- **token F1**：`R/μ = 52.5% > 50%` ⇒ 测量不成立（用户预声明判据）。
- **MMLU**：3 种子 Δ 符号翻转（0.0000 / +0.0067 / −0.0067）⇒ 测量不成立。

**格式合规率提升**（+92.9% → 3 种子 μ=+111.9%，`R/μ=12.8%`）**保持成立**，且原值是偏保守的。

详见 `SEED_REPRODUCIBILITY_RETROSPECT.md` 与 `ACCEPTANCE_MATRIX.md` 的「多种子回溯修订」节。
**未修改任何判据或阈值**；被证伪项降级为单种子观测值，不得作为流程级结论引用。
