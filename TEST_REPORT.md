# TEST_REPORT —— training-sft 测试报告（长任务 2）

> 执行：2026-09-29　复跑：`node tests/run.mjs`　CI 挂载点：`inference-contracts/checks/run-all.mjs` 的 **C15**

---

## 1. 汇总

| 套件 | spec | 断言 | 结果 |
|---|---:|---:|---|
| `tests/unit/registration.spec.mjs` | 1 | **78** | PASS |
| `tests/unit/guards.spec.mjs` | 1 | **49** | PASS |
| `tests/unit/mock-loop.spec.mjs` | 1 | **15** | PASS |
| `tests/unit/engineering.spec.mjs` | 1 | **54** | PASS |
| **CI 合计** | **4** | **196** | **ALL SPECS PASSED** |
| `tests/realmachine/docker-worker.spec.mjs` | 1 | （真机，不进 CI） | PASS（见 §4） |
| 契约层（含 C15/C16） | —— | **64/64** | PASS |

**CI 环境要求**：无 GPU · 无容器 · 无网络（**已实测**：CI 套件全程未起容器、未连外网）。

---

## 2. CI 套件覆盖（P2 #14–#19 的落点）

### `registration.spec.mjs`（78 断言）—— P2 #18 / #19

- 用**像 DSH 一样挑刺**的 mock ctx 跑 `apply()`：缺 `output{schema,render}` / 缺 `parameters` /
  description 过短都直接抛 → 6 个工具全部注册成功（判据来自实测教训：注入 ✓ ≠ 注册成功）。
- **真调两个只读工具**（`training_sft_health` / `training_sft_get_report`），断言信封结构与
  `governance.audit` 字段。
- 路由 `/__training_sft/health` 被注册。
- **命名空间**：去 `inference-serving` / `inference-quantization` / `inference-tuning` 的**源码**
  实时抓取工具名（33+ 个），交集必须为空；且断言兄弟包一律 `inference_` 前缀。
- **风险声明双向一致**：`manifest.yaml` ⇄ `src/index.mjs` 的 `TOOL_RISK` ⇄ 实际注册的工具名，三者逐条相等。
- 包元数据：`dsh.bundle.patch` 已声明；六个共享库依赖显式声明；**五个训练框架在 `optionalDependencies` 显式列出**。

### `guards.spec.mjs`（49 断言）—— P2 #14 / #15 / #16

- **输入 7 类**：缺必填 / 参数越界 / 目录不存在 / 缺 lineage / `allClear=false` 拒绝 / 未知 runId / 未知工具名。
- **安全 8 条**：缺令牌拒 / 错令牌拒 / 正确令牌放行并 memo（且令牌与目标绑定）/ 步数超限拒 /
  GPU 小时超限拒 / `purge` 不越界（断言 `/etc` 仍在）/ 不改写原始数据集（长度比对）/ 泄漏有独立错误码。
- **健壮性 6 条**：状态损坏不静默重置 / 审计链篡改检出并可恢复 / schema 拦截非法状态 /
  CAS 冲突 / worker 协议 fail-closed（未 start 即 call + `args()` 形态）/ 端点不可达 → `null` 而非伪造值。
- 另含指标与门禁的判据单测（格式合规 4 条、token F1 3 条、遗忘门禁 3 条、任务门禁 2 条）。

### `mock-loop.spec.mjs`（15 断言）—— 报告要求第 4 项的「训练测试特殊性」

用一个**模拟容器协议语义**的假 worker（维护自己的步计数器、会因失步报错、走 checkpoint 协议），
通过 `runTraining({workerFactory})` 这条**测试接缝**驱动**真实的 training-core 循环**。CI 因此在无 GPU 的前提下验证：

- 循环接线（step 事件 30 次、loss 逐笔记账、stop 原因正确）；
- **两侧步号一致性不变量**：故意让容器漏一步 → 必须失败；
- 容器中途"失联" → 异常上抛（不静默截断）；
- 墙钟上限（`maxWallMs`）真的会停；
- checkpoint 编排：`every`/`keep` 生效、payload 里登记权重文件哈希、`meta.payloadHash` 复核一致；
- 审计钩子（`createAuditHook`）写出的链可复算。

### `engineering.spec.mjs`（54 断言）—— P2 #17

npm 包规范 / 软硬件解耦（无 GPU 型号硬编码、绝对路径从模块位置推导且在白名单内）/
文档齐全 / 测试覆盖 / 交付物齐全。

---

## 3. 真机测试（**不进 CI**）

`tests/realmachine/docker-worker.spec.mjs`：起真容器、走协议 `ping`、断言容器内 `cuda=true`、
`shutdown` 后核对**零容器残留**。

分离机制与 inference-diagnose 同款：`tests/run.mjs` **只递归 `tests/unit/`**，本目录天然不被收集
—— 比"打标记跳过"更强（忘了带标记也不会把真机测试拖进 CI）。

---

## 4. 契约层（CI 阻断）

```
$ node inference-contracts/checks/run-all.mjs
…
PASS C15 training-sft 单测全过（4 spec / 196 断言）
PASS C16a eval-result 符合冻结 schema（…/ts_20260929044120_kj5q/eval-result.json）
PASS C16b eval-result 带 versions.testingCore（判据可溯源）
PASS C16c validity 非空且 gate.validityOk 与之一致
PASS C16d training-sft 的判定逻辑 import 自 @ai-infra/testing-core（不重写一份）
PASS C16e evaluate.mjs 未自行实现困惑度/相对变化（只有抽取与装配）
PASS C16f 基线缓存键含评估协议版本（跨协议比较会静默错判）
合计 64/64，未过 0
```

`training-sft` 的 `npm test` → `test:contract` → 这份检查；三个推理插件同样跑到它
（各 `npm run test:contract` = 64/64），**故插件改动导致契约测试失败即 CI 阻断**。

---

## 5. 缺陷清单（本轮测试/真机抓到并已修）

| # | 缺陷 | 被谁抓到 | 状态 |
|---|---|---|---|
| 1 | `getReport` 在 `root` 为 `undefined` 时抛 `auditDirFor 需要 { root, pkg }` | registration.spec | 已修（各 capability 入口归一化 root） |
| 2 | `taskGate` 的 `dropPct` 符号写反（"大幅提升"被读成"大幅下降"） | guards.spec G4 | 已修 |
| 3 | `prepare` 忽略 `root`、把数据集写到真实 `$HOME` | guards.spec S6 副作用 | 已修（新增 `datasetsRoot`） |
| 4 | CI 里起 docker 容器（违反"CI 无容器"） | 自查 | 已改：协议/参数断言进 CI，真容器移入 `tests/realmachine/` |
| 5 | 源码硬编码本机工作区路径 3 处 | engineering.spec | 已修（从模块位置推导 + 环境变量覆盖） |
| 6 | `datasetHash` 出现 `sha256:sha256:` 双前缀 | C16a 前的自查 | 已修 |
| 7 | **格式合规率判据饱和**（基座 0.98）→ 门禁永远不可能通过 | 真机第一次 validate | 已修（评估协议 v2：要求自然终止）；详见 ACCEPTANCE_MATRIX P1 #12 |
| 8 | **续训步数语义错误**（"续训 20 步"实跑 220 步且不报错） | 真机 resume 演示 | 已修（循环跑新增步数；预算单独判总量） |
| 9 | 容器内 root 写出的 checkpoint 宿主改不动（EACCES） | 真机训练 | 已修（`--user uid:gid`） |
| 10 | 模型挂载路径错（`/models` vs `/models/<name>`） | 真机训练 | 已修 |
| 11 | 训练脚本没挂进容器 | 真机训练 | 已修 |
| 12 | `keep` 只删训练元数据、权重目录成孤儿 | 真机训练 | 已修（权重与 checkpoint 同名同目录） |
| 13 | `health()` 输出含 `undefined` 键 → DSH 拒绝（`not lossless JSON`） | **真实 DSH 注入后真调** | 已修 |

> 第 13 条只可能被"真实注入 + 真调"抓到：单测走 `spec.handler()` 不经过 DSH 的输出校验器。
> 这与项目既有教训（「注入返回 host ✓ ≠ 工具注册成功」）是同一条纪律的第二个实例：
> **注册成功 ≠ 输出契约在真实校验器下成立**。

---

## 6. 未覆盖 / 已知边界

- **DSH 重启后的重新装配**未验证（本轮用热注入验证；重启路径由 `cordis.patch.yml` 承担）。
- `mcp/server.mjs` 的 stdio JSON-RPC 只做了 `tools/list` / `tools/call` 的形状检查，
  **未**用第三方 MCP 客户端做端到端握手测试。
- 真机套件只覆盖 worker 协议；`validate` 的端到端（部署→测量→拆解）在 `DEPLOYMENT_REPORT.md` 里
  有一次完整真实执行，但**未**脚本化为可复跑的真机 spec。
- 遗忘门禁的 wikitext 指标**临界通过**（余量 6%），未做多种子复验。
