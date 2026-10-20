/**
 * core —— training-sft 的 6 个 capability 实现（薄适配层）。
 *
 * 能力→源：
 *   prepare   → @xiciyang/data-core（切分/去泄漏/血缘/PII）
 *   train     → @xiciyang/training-core（公共循环/上限/checkpoint）+ 容器内 peft
 *   validate  → inference-serving（部署）+ @xiciyang/testing-core（判定）+ @xiciyang/serving 端点
 *   rollback / get_report / health → 状态与审计（state-core / audit-core / sandbox-core）
 */
import { readFileSync, writeFileSync, existsSync, readdirSync, promises as fs } from 'node:fs'
import { join, dirname, relative } from 'node:path'
import { fileURLToPath } from 'node:url'
import { createHash } from 'node:crypto'
import { execFileSync } from 'node:child_process'

import { dualGate, CONTRACT_VERSION as TESTING_CORE_VERSION } from '@xiciyang/testing-core'
import { BUDGET_DEFAULTS, enforceLimits, instrumentFromDeployReport, assertInstrumentUsable, assertArtifactChanged, assertWeightFingerprint as sharedAssertWeightFingerprint } from '@xiciyang/training-core'

import { ToolError, ERROR_CODES, PACKAGE_VERSION, SCHEMA_VERSION } from './version.mjs'
import { paths, audit, auditVerify, readRegistry, updateRegistry, sandboxProbe, requireContainerSandbox, digestArgs, mkdirp, nowId, contentHash } from './runtime.mjs'
import { runTraining } from './train.mjs'
import { evalTask, evalLM, evalMMLU, evalUniform, taskGate, forgettingGate, measurementCompleteness, attachIntegrity, EVAL_PROTOCOL, get as getJson } from './evaluate.mjs'
import { resolveServingPath, ServingMcpClient, callWithConfirm } from './serving-client.mjs'
import { prepareData, loadEvalSets } from './data/prepare.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const PKG_ROOT = join(HERE, '..')
// 契约层位置从包位置推导（training-sft 与 inference-contracts 是兄弟目录），可用环境变量覆盖
const WORKSPACE = process.env.DSH_WORKSPACE || join(PKG_ROOT, '..', '..')
const SCHEMA_DIR = join(process.env.INFERENCE_CONTRACTS_DIR || join(WORKSPACE, 'inference-contracts'), 'state-schema')

export const TRAINING_IMAGE = 'training-sft/toolchain:0.1.0'
export const DEFAULT_LIMITS = Object.freeze({ ...BUDGET_DEFAULTS, maxSteps: 200, maxGpuHours: 2, concurrency: 1, maxParams: 0.5e9 })
const SCHEMA_FILE = join(SCHEMA_DIR, 'sft-registry.schema.json')

function schemaPath() { return existsSync(SCHEMA_FILE) ? SCHEMA_FILE : null }
const P = (root) => paths(root)

async function withRun(root, runId, mutator) {
  return updateRegistry(root, (doc) => {
    const i = (doc.runs || []).findIndex((r) => r.id === runId)
    if (i < 0) throw new ToolError(ERROR_CODES.E_RUN_NOT_FOUND, { hint: `run 不存在：${runId}`, recoverable: true })
    const runs = [...(doc.runs || [])]
    runs[i] = mutator({ ...runs[i] })
    return { ...doc, runs }
  }, { schemaPath: schemaPath() })
}

/* ══════════════════════════════════════════════════════════ 1. prepare */

export async function prepare({ root = paths().root, dataset = null, seed = 20260929, train = 1000, heldout = 200, force = false, mix = 0, log = () => {} }) {
  const t0 = Date.now()
  await audit(root, { capability: 'prepare', action: 'prepare', phase: 'begin', ...digestArgs({ dataset, seed, train, heldout, mix }) })
  const out = await prepareData({ seed, train, heldout, outDir: dataset, datasetsRoot: P(root).datasetsDir, write: true, mix })
  if (!out.allClear) {
    await audit(root, { capability: 'prepare', action: 'prepare', phase: 'end', outcome: 'leak-detected', ok: false })
    throw new ToolError(ERROR_CODES.E_DATASET_LEAK, {
      hint: `数据校验失败：训练/留出/评估集之间仍有内容哈希交集 —— 拒绝训练（P0 #1）。详情：${JSON.stringify(out.verify.filter((v) => !v.ok))}`,
      details: { verify: out.verify },
      recoverable: false,
    })
  }
  const r = await audit(root, { capability: 'prepare', action: 'prepare', phase: 'end', outcome: 'ok', rows: out.trainRows, heldout: out.heldoutRows, mix, replayLanded: out.replay?.nReplay ?? 0, ms: Date.now() - t0 })
  log(`数据就绪：train=${out.trainRows} heldout=${out.heldoutRows} → ${out.out}`)
  return {
    datasetDir: out.out,
    trainRows: out.trainRows,
    heldoutRows: out.heldoutRows,
    exclusions: { piiRows: out.piiDropped, evalCollisionRows: out.evalDropped, evalContainmentRows: out.containsDropped },
    verify: out.verify,
    lineage: out.lineage,
    replay: out.replay,
    audit: { seq: r.seq, hash: r.hash },
  }
}

/* ══════════════════════════════════════════════════════════ 2. train */

export async function train({
  root = paths().root, baseModel = process.env.DSH_BASE_MODEL, datasetDir, lora = {}, lr = 1e-4, weightDecay = 0.0,
  batchSize = 4, gradAccum = 4, maxSeqLen = 256, maxSteps = 200, warmupSteps = 10, seed = 20260929,
  limits = {}, checkpointEvery = 50, checkpointKeep = 3, confirm = null, resumeRunId = null, resumeFrom = null, log = () => {},
}) {
  const p = P(root)
  await mkdirp(p.runsDir)
  const limitsResolved = { ...DEFAULT_LIMITS, ...limits, maxSteps }
  if (limitsResolved.maxSteps > 500) throw new ToolError(ERROR_CODES.E_BUDGET_EXCEEDED, { hint: `maxSteps=${limitsResolved.maxSteps} 超过单次上限 500`, recoverable: true })
  if (limitsResolved.maxGpuHours > 2) throw new ToolError(ERROR_CODES.E_BUDGET_EXCEEDED, { hint: `maxGpuHours=${limitsResolved.maxGpuHours} 超过单次上限 2`, recoverable: true })

  // 前置：数据必须已就绪且通过去泄漏复核（不信任传入路径）
  if (!datasetDir || !existsSync(join(datasetDir, 'train.jsonl'))) throw new ToolError(ERROR_CODES.E_DATASET_INVALID, { hint: `数据集未就绪：${datasetDir}（先调 training_sft_prepare）`, recoverable: true })
  const lineage = JSON.parse(readFileSync(join(datasetDir, 'lineage.json'), 'utf8'))
  if (lineage.allClear !== true) throw new ToolError(ERROR_CODES.E_DATASET_LEAK, { hint: '数据集血缘里 allClear !== true —— 拒绝训练', recoverable: false })
  if (!existsSync(join(baseModel, 'config.json'))) throw new ToolError(ERROR_CODES.E_BASE_MODEL_NOT_FOUND, { hint: `基座模型目录不合法：${baseModel}`, recoverable: true })

  // 沙箱：训练属 write 档 → 必须容器层（策略来自长任务 1 冻结的 sandbox-core）
  const probe = await sandboxProbe()
  requireContainerSandbox(probe)

  const runId = resumeRunId || nowId('ts')
  const runDir = join(p.runsDir, runId)
  // 续训：`resumeFrom` = {runId, checkpointName} —— 权重目录在容器内映射为 /out/...
  let resumeAdapter = null
  let resumeStepOffset = 0
  let lrTotalSteps = maxSteps
  if (resumeFrom?.runId && resumeFrom?.checkpointName) {
    const srcDir = join(p.runsDir, resumeFrom.runId, 'checkpoints', resumeFrom.checkpointName)
    if (!existsSync(join(srcDir, 'adapter'))) throw new ToolError(ERROR_CODES.E_RUN_NOT_FOUND, { hint: `续训源 checkpoint 缺 adapter 目录：${srcDir}`, recoverable: true })
    resumeAdapter = `/out-src/${srcDir.slice(join(p.runsDir, resumeFrom.runId).length + 1)}`
    // 从源 checkpoint 的 payload 读回真实步号（**不猜**）——预算与不变量都要用它
    try {
      const payload = JSON.parse(readFileSync(join(srcDir, 'payload.json'), 'utf8'))
      resumeStepOffset = Number(payload.step ?? 0)
    } catch (e) { throw new ToolError(ERROR_CODES.E_RUN_NOT_FOUND, { hint: `续训源 checkpoint 的 payload.json 不可读：${e.message}`, recoverable: true }) }
    // **语义**：`maxSteps` = 本次**新增**步数（不是总步数）。
    // 预算闸门判的是**这个模型的总步数** offset+新增 ≤ 500；
    // 而 training-core 的循环只跑"新增"那么多步（Node 侧计数器从 0 起算）。
    // 曾经把 `limits.maxSteps = offset+新增` 直接喂给循环 → "续训 20 步"实际跑了 220 步，
    // 且不报错（因为两侧 offset 断言是对的）。这是本轮抓到并修掉的自身缺陷。
    const total = resumeStepOffset + maxSteps
    if (total > 500) throw new ToolError(ERROR_CODES.E_BUDGET_EXCEEDED, { hint: `续训后该模型总步数 ${total}（offset ${resumeStepOffset} + 新增 ${maxSteps}）超过单次上限 500`, recoverable: true })
    lrTotalSteps = total
    log(`续训：从 ${srcDir} 恢复（offset=${resumeStepOffset}，本次新增 ${maxSteps} 步，该模型总步数 ${total}）`)
  }
  await mkdirp(runDir)
  const started = new Date().toISOString()
  await audit(root, { capability: 'train', action: 'train', phase: 'begin', runId, baseModel, datasetDir, lora, lr, batchSize, gradAccum, maxSeqLen, maxSteps, limits: limitsResolved, sandboxTier: 'container', image: TRAINING_IMAGE })

  await updateRegistry(root, (doc) => ({
    ...doc,
    runs: [...(doc.runs || []).filter((r) => r.id !== runId), {
      id: runId, status: 'running', startedAt: started,
      baseModel: { path: baseModel, revision: readRevision(baseModel) },
      dataset: { dir: datasetDir, trainRows: lineage.calib?.count ?? null, heldoutRows: lineage.test?.count ?? null, lineageHash: `sha256:${contentHash(readFileSync(join(datasetDir, 'lineage.json'), 'utf8'))}` },
      config: { lora, lr, weightDecay, batchSize, gradAccum, maxSeqLen, maxSteps, warmupSteps, seed },
      budget: limitsResolved,
      auditSeqs: [],
    }],
  }), { schemaPath: schemaPath() })

  try {
    const { result, events, checkpoints } = await runTraining({
      root, runId, runDir, image: TRAINING_IMAGE, modelDir: baseModel, datasetDir, extraMounts: resumeFrom?.runId ? [`${join(p.runsDir, resumeFrom.runId)}:/out-src:ro`] : [],
      loraConfig: { baseModelDirName: baseModel.split('/').filter(Boolean).pop(), lora: { r: 8, alpha: 16, dropout: 0.0, targetModules: ['q_proj', 'k_proj', 'v_proj', 'o_proj'], ...lora }, lr, weightDecay, batchSize, gradAccum, maxSeqLen, maxSteps: lrTotalSteps, warmupSteps, seed, resumeAdapter, resumeStepOffset },
      limits: limitsResolved,
      checkpoint: { dir: join(runDir, 'checkpoints'), every: checkpointEvery, keep: checkpointKeep },
      onEvent: () => {},
      log,
    })
    const last = checkpoints[checkpoints.length - 1] ?? null
    // **产物级断言（fail-closed）**：权重相对起点必须真的变了。放在审计之前 ⇒ 失败即落 failed 分支。
    if (last) {
      const cpPayload = JSON.parse(readFileSync(join(runDir, 'checkpoints', last.name, 'payload.json'), 'utf8'))
      assertWeightFingerprint(cpPayload)
      // `files[]` 必须**覆盖权重件**（旧实现非递归 ⇒ adapter 在子目录里，从来没被列进去）
      const weightFiles = (cpPayload.files ?? []).filter((f) => /\.(safetensors|bin|pt|pth|ckpt|gguf)$/i.test(f.name))
      if (weightFiles.length === 0) {
        throw new Error('E_WEIGHT_FILE_NOT_LISTED: checkpoint payload 的 files[] 里没有任何权重件 ⇒ 产物清单证明不了权重在场（fail-closed）')
      }
    }
    const ar = await audit(root, { capability: 'train', action: 'train', phase: 'end', runId, outcome: result.stop?.reason ?? 'ok', steps: result.state.step, wallMs: result.state.wallMs, gpuHours: Number(result.state.gpuHours.toFixed(4)), finalLoss: result.state.losses.slice(-1)[0] ?? null, checkpoints: checkpoints.length })
    await withRun(root, runId, (r) => ({
      ...r, status: 'trained', finishedAt: new Date().toISOString(),
      result: { steps: result.state.step, wallMs: result.state.wallMs, gpuHours: Number(result.state.gpuHours.toFixed(4)), stopReason: result.stop?.reason ?? null, stopChecks: result.stop?.checks ?? [], lossesTail: result.state.losses.slice(-10), checkpoints: checkpoints.map((c) => ({ name: c.name, hash: c.hash, step: c.step })) },
      checkpoint: last ? { dir: join(runDir, 'checkpoints'), name: last.name, payloadHash: last.hash, step: last.step } : null,
      runDir,
      auditSeqs: [...(r.auditSeqs || []), ar.seq],
    }))
    return { runId, runDir, status: 'trained', steps: result.state.step, gpuHours: Number(result.state.gpuHours.toFixed(4)), stopReason: result.stop?.reason ?? null, checkpoints: checkpoints.length, finalLoss: result.state.losses.slice(-1)[0] ?? null, events: events.filter((e) => e.type === 'step').length, audit: { seq: ar.seq, hash: ar.hash } }
  } catch (e) {
    const ar = await audit(root, { capability: 'train', action: 'train', phase: 'failed', runId, error: String(e.message).slice(0, 300) })
    await withRun(root, runId, (r) => ({ ...r, status: 'failed', error: String(e.message).slice(0, 500), auditSeqs: [...(r.auditSeqs || []), ar.seq] })).catch(() => {})
    throw e
  }
}

function readRevision(modelDir) {
  try {
    const m = join(modelDir, '.cache', 'huggingface', 'download', 'config.json.metadata')
    return existsSync(m) ? readFileSync(m, 'utf8').split('\n')[0].trim() : null
  } catch { return null }
}

/* ══════════════════════════════════════════════════════════ 2b. merge */

export async function mergeRun({ root = paths().root, runId, checkpointName = null, log = () => {} }) {
  const reg = await readRegistry(root, schemaPath())
  const run = (reg.runs || []).find((r) => r.id === runId)
  if (!run) throw new ToolError(ERROR_CODES.E_RUN_NOT_FOUND, { hint: `run 不存在：${runId}`, recoverable: true })
  const mergedDir = join(run.runDir, 'merged')
  const adapterDir = checkpointName ? join(run.runDir, 'checkpoints', checkpointName) : run.runDir
  // 用容器内的 worker 做合并（与训练同一镜像，避免宿主/容器版本差异）
  const worker = new (await import('./worker.mjs')).ContainerWorker({
    image: TRAINING_IMAGE,
    mounts: await trainingMountsFor({ modelDir: run.baseModel.path, datasetDir: run.dataset.dir, outDir: run.runDir }),
    log,
  }).start()
  try {
    await worker.call('init', {
      config: {
        baseModel: `/models/${run.baseModel.path.split('/').filter(Boolean).pop()}`,
        trainFile: '/data/train.jsonl',
        lora: run.config.lora, lr: run.config.lr, batchSize: run.config.batchSize, gradAccum: run.config.gradAccum,
        maxSeqLen: run.config.maxSeqLen, seed: run.config.seed,
        resumeAdapter: `/out/${checkpointName ? `checkpoints/${checkpointName}` : ''}`,
      },
    }, { timeoutMs: 20 * 60 * 1000 })
    const r = await worker.call('merge_and_export', {
      outDir: `/out/merged`,
      license: { spdx: 'apache-2.0' },
      readmeNote: `SFT (LoRA r=${run.config.lora.r}) on yahma/alpaca-cleaned (CC BY 4.0), base Qwen/Qwen2.5-0.5B-Instruct (Apache-2.0).`,
      provenance: { runId, baseModel: run.baseModel, dataset: run.dataset, config: run.config, steps: run.result?.steps ?? null },
    }, { timeoutMs: 30 * 60 * 1000 })
    const ar = await audit(root, { capability: 'merge', action: 'merge', phase: 'end', runId, mergedDir, weightsSha256: r.weightsSha256, params: r.params, bytes: r.bytes })
    await withRun(root, runId, (x) => ({ ...x, status: 'merged', merged: { dir: mergedDir, weightsSha256: r.weightsSha256, params: r.params, bytes: r.bytes, files: r.files.length }, auditSeqs: [...(x.auditSeqs || []), ar.seq] }))
    return { runId, mergedDir, weightsSha256: r.weightsSha256, params: r.params, bytes: r.bytes, vocabSize: r.vocabSize }
  } finally { await worker.stop() }
}

async function trainingMountsFor(p2) {
  const { trainingMounts } = await import('./worker.mjs')
  return trainingMounts(p2)
}

/* ══════════════════════════════════════════════════════════ 3. validate */

/**
 * **检查点权重指纹守卫**（缺陷：SFT 此前**完全没有**"空操作训练"防线）。
 *
 * 为什么需要它：本仓库的原始事故是 `lr=0` 让 500 步全是空操作，而**步号/consumedChunks/不变量全部为真**。
 * pretrain 因此有 `fingerprint`/`initFingerprint`/`weightsChanged`；**SFT 一直没有** ——
 * 也就是说"只断言过程抓不到空操作，必须断言权重"这条原则**只固化在了 pretrain 一处**。
 *
 * fail-closed 两条：
 *   ① 指纹**缺失** ⇒ 抛（"无法断言"不等于"没问题"）；
 *   ② 指纹**相同** ⇒ 抛 `E_NO_OP_TRAINING`（空操作绝不能被记成训练成功）。
 *
 * 纯函数、不碰磁盘 ⇒ 可在无 GPU 的 spec 里回归。
 */
// 判据实现**在共享库里**（`@xiciyang/training-core` 的 `assertWeightFingerprint`）——
// 本包只做默认 `what` 的绑定，不再另写一份（避免"原则固化在一处、其它处各写各的"）。
export function assertWeightFingerprint(payload, opts = {}) {
  return sharedAssertWeightFingerprint(payload, { what: 'SFT LoRA 权重', ...opts })
}

async function measureAt({ endpoint, model, heldout, wikitext, mmlu, nTask, nLm, nMmlu, vocabSize, log }) {
  const uniform = await evalUniform({ endpoint, model, prompt: 'The quick brown fox jumps over the lazy dog. In a distant country, the population of', vocabSize })
  const task = await evalTask({ endpoint, model, rows: heldout, limit: nTask, log })
  const lm = await evalLM({ endpoint, model, rows: wikitext, limit: nLm, log })
  const mmluRes = await evalMMLU({ endpoint, model, rows: mmlu, limit: nMmlu, log })
  return { uniform, task, lm, mmlu: mmluRes }
}

/**
 * fail-closed：**有损仪器不得用于验收**（`E_INSTRUMENT_LOSSY`）。
 *
 * 判据本身在 `@xiciyang/training-core`（单一实现，四个插件共用）；这里只做**错误码转译**：
 * training-core 不认识本包的 `ToolError`，若不转译，`toEnvelope` 会把一切非 ToolError
 * 归成 `E_INTERNAL`，把可执行的补救信息埋进 detail —— 那等于把"可诊断"降级成"内部错误"。
 */
function guardInstrument(instrument, runId) {
  try {
    assertInstrumentUsable(instrument, { runId })
  } catch (e) {
    if (e?.code !== ERROR_CODES.E_INSTRUMENT_LOSSY) throw e
    throw new ToolError(ERROR_CODES.E_INSTRUMENT_LOSSY, { hint: e.message, details: e.details ?? null, recoverable: false })
  }
}

/**
 * `validate` 的状态准入规则 —— **纯函数**，便于在无 GPU / 无容器的回归测试里判定。
 *
 * 缺陷 13：`train` 留下 `trained`，而 `validate` 需要 `merged`，但**公开工具面没有 merge 入口**
 * ⇒ 旧实现直接拒绝，闭环必然在 validate 处断掉。现在 `trained` 被接受，
 * 由 `validate` 自己先 `mergeRun`（**与 `rl` / `qat` 同构**）。
 * `failed` / `running` / `rolled-back` 仍然**必须拒绝** —— 这个函数同时是"闸门没被删掉"的证据。
 */
/**
 * **执行体身份**（G7 / 第五轮）：对本插件**运行时源码目录**做内容指纹。
 *
 * 由来（真实事故）：第四轮发现**运行中的插件是 09-30 的快照**，而 CI 验证的是磁盘源码
 * ⇒ "CI 绿"只是关于磁盘的断言，**不是关于执行体的断言**。此后每条测量都必须能回答
 * "**这份数是谁产出的**"。
 *
 * 只覆盖 host 侧运行时会读到的文件（`.mjs`；`.py` 一并纳入，因为容器内脚本也来自这里），
 * 跳过 `node_modules` / `tests`。内容哈希 ⇒ 与 mtime 无关，可复算、可跨机比对。
 */
export function codeVersion({ dir = fileURLToPath(new URL('.', import.meta.url)) } = {}) {
  const files = []
  const walk = (d) => {
    let ents
    try { ents = readdirSync(d, { withFileTypes: true }) } catch { return }
    for (const e of ents.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0))) {
      if (e.name === 'node_modules' || e.name === 'tests' || e.name.startsWith('.')) continue
      const full = join(d, e.name)
      if (e.isDirectory()) { walk(full); continue }
      if (!/\.(mjs|js|py)$/.test(e.name)) continue
      try { files.push(`${relative(dir, full).replace(/\\/g, '/')}:${createHash('sha256').update(readFileSync(full)).digest('hex')}`) } catch { /* 读不到则跳过（不伪造） */ }
    }
  }
  walk(dir)
  return `training-sft/src@${createHash('sha256').update(files.join('\n')).digest('hex')}`
}

export function validateStatusPlan(status) {
  if (status === 'trained') return 'auto-merge'
  if (['merged', 'validated', 'gate-failed'].includes(status)) return 'proceed'
  return 'reject'
}

/**
 * `merge` 的状态准入规则 —— **纯函数**（R-D4）。
 *
 * 缺陷（第四轮实测）：`train` 留下 `trained`、`validate` 需要 `merged`，
 * 而**公开工具面从来没有 merge 入口** ⇒ 闭环断在 validate 处，且没有可落地的补救手段。
 * 现在把 merge 显式化，并且：
 *   - `trained` ⇒ 真的 merge；
 *   - `merged`/`validated`/`gate-failed` ⇒ **幂等**（重复调用不重做、不报错）；
 *   - 其余（`failed`/`running`/`rolled-back`/未知）⇒ **拒绝**（fail-closed）。
 *
 * ⚠️ 与 `validateStatusPlan` 的**拒绝集合必须一致** ——
 * 两个入口对同一个 run 不能"一个放行、一个拒绝"（`merge-tool.spec` 有反向对照断言）。
 */
export function mergeStatusPlan(status) {
  if (status === 'trained') return 'merge'
  if (['merged', 'validated', 'gate-failed'].includes(status)) return 'idempotent'
  return 'reject'
}

/**
 * **显式 merge**（R-D4）：把 LoRA adapter 合并成完整 HF checkpoint。
 *
 * 与 `validate` 内部隐式 merge 的关系：`validate` 现在复用本函数，
 * 因此"显式 merge 后再 validate"与"直接 validate"**走同一条代码路径**，不存在行为分叉。
 */
export async function merge({ root = paths().root, runId, checkpointName = null, log = () => {} } = {}) {
  const reg = await readRegistry(root, schemaPath())
  const run = (reg.runs || []).find((r) => r.id === runId)
  if (!run) throw new ToolError(ERROR_CODES.E_RUN_NOT_FOUND, { hint: `run 不存在：${runId}`, recoverable: true })
  const plan = mergeStatusPlan(run.status)
  if (plan === 'reject') {
    throw new ToolError(ERROR_CODES.E_RUN_NOT_FOUND, { hint: `run ${runId} 状态为 ${run.status}，不能 merge（只接受 trained；merged/validated/gate-failed 幂等）`, recoverable: true })
  }
  if (plan === 'idempotent') {
    log(`  → run 状态已是 ${run.status}，merge 幂等返回`)
    return { runId, alreadyMerged: true, status: run.status, mergedDir: run.merged?.dir ?? null, weightsSha256: run.merged?.weightsSha256 ?? null, params: run.merged?.params ?? null, bytes: run.merged?.bytes ?? null }
  }
  const ckpt = checkpointName ?? run.checkpoint?.name ?? null
  if (!ckpt) throw new ToolError(ERROR_CODES.E_RUN_NOT_FOUND, { hint: `run ${runId} 状态为 trained 但没有 checkpoint 记录，无法 merge；请显式指定 checkpointName`, recoverable: true })
  const r = await mergeRun({ root, runId, checkpointName: ckpt, log })
  return { runId, alreadyMerged: false, status: 'merged', ...r }
}

export async function validate({
  root = paths().root, runId, deploy = true, nTask = 100, nLm = 200, nMmlu = 300, thresholds = null, teardown = true, log = () => {},
}) {
  const t0 = Date.now()
  const p = P(root)
  const thresholdsResolved = { maxF1DropPct: 5, minFormatGainPct: 10, ...(thresholds || {}) }
  const reg = await readRegistry(root, schemaPath())
  let run = (reg.runs || []).find((r) => r.id === runId)
  if (!run) throw new ToolError(ERROR_CODES.E_RUN_NOT_FOUND, { hint: `run 不存在：${runId}`, recoverable: true })
  // 缺陷 13：见 validateStatusPlan 的注释。`trained` ⇒ validate 自己先 merge（与 rl/qat 对齐），
  // 调用方不需要知道有这一步；这也与本工具描述（声明会 merge adapter）一致。
  if (validateStatusPlan(run.status) === 'auto-merge') {
    log('  → run 状态为 trained，validate 内部先完成 merge（与 rl/qat 对齐）')
    // R-D4：复用**显式 merge** 的同一条路径（避免"隐式/显式两套行为"）。
    // checkpoint 为空时 merge() 自己会用 `run.checkpoint.name` 兜底并在缺失时 fail-closed。
    await merge({ root, runId, log })
    const reg2 = await readRegistry(root, schemaPath())
    run = (reg2.runs || []).find((r) => r.id === runId)
    if (!run) throw new ToolError(ERROR_CODES.E_RUN_NOT_FOUND, { hint: `merge 后 run 记录消失：${runId}`, recoverable: false })
  }
  // 允许在 gate-failed 上重判（修好测量后重跑是正常闭环；不允许在 failed/running 上重判）
  if (validateStatusPlan(run.status) === 'reject') throw new ToolError(ERROR_CODES.E_RUN_NOT_FOUND, { hint: `run ${runId} 状态为 ${run.status}，不能 validate`, recoverable: true })

  await audit(root, { capability: 'validate', action: 'validate', phase: 'begin', runId, mergedDir: run.merged.dir, nTask, nLm, nMmlu })

  // 评估语料
  const heldout = readJsonl(join(run.dataset.dir, 'heldout.jsonl'))
  const sets = loadEvalSets()
  const wikitext = (sets.find((s) => s.name.startsWith('wikitext-test(c128)')) || sets.find((s) => s.kind === 'lm'))?.rows ?? []
  const mmlu = (sets.find((s) => s.kind === 'qa'))?.rows ?? []
  const vocabSize = JSON.parse(readFileSync(join(run.merged.dir, 'config.json'), 'utf8')).vocab_size

  const sp = await resolveServingPath({ root })
  const client = await new ServingMcpClient(sp.serverPath).start()
  const results = {}
  let ownerToken = null
  let sftEndpoint = null
  let baseEndpoint = null
  // ── 仪器来源（instrument provenance）：**每次用于测量的部署**各留一份归一化记录。
  //    没有它，产物只说"判决如何"，不说"判决是用什么仪器测出来的" —— 实测过的后果：
  //    部署落进 serving 的**有损**降级档位（dtype=half + quantization=fp8）时，
  //    同一份权重测出 25/48 而不是 23/48，判决翻转，而 eval-result 里没有任何字段说明这件事。
  let candidateInstrument = null
  let baseInstrument = null
  try {
    // ① 部署合并后的模型（inference-serving 的加载器 + 它自己的可用性探针）
    if (deploy) {
      log(`部署 SFT 模型 ${run.merged.dir} …`)
      const d = await callWithConfirm(client, 'inference_serving_deploy', {
        path: run.merged.dir, owner: 'training-sft', purpose: `长任务2 垂直闭环 validate ${runId}`,
        servedModelName: `sft-${runId}`, gpuMemoryUtilization: 0.18, maxNumSeqs: 4,
      }, { log })
      if (d.ok === false) throw new ToolError(ERROR_CODES.E_SERVING_UNAVAILABLE, { hint: `SFT 模型部署失败：${d.code} ${d.message ?? ''}`, details: d })
      ownerToken = d.ownerToken
      sftEndpoint = { url: d.endpoint.endpoint, model: d.endpoint.servedModelName, id: d.endpoint.id, container: d.endpoint.containerName, imageDigest: d.endpoint.imageDigest, usability: d.report?.attempts?.[0]?.modelUsability ?? null }
      // `d.report` = inference-serving 的 deploy-report@1（**真正部署的那份配置**在里面）
      candidateInstrument = instrumentFromDeployReport(d.report)
      guardInstrument(candidateInstrument, runId) // 有损 ⇒ 立刻停，不做无意义的测量
      log(`  → ${sftEndpoint.url} (usability.ok=${sftEndpoint.usability?.ok}, lossy=${candidateInstrument?.lossy ?? null})`)
    } else {
      throw new ToolError(ERROR_CODES.E_MISSING_PARAM, { hint: 'validate 需要 deploy=true（本实现的闭环要求经 serving 加载器验证）', recoverable: true })
    }

    // ② SFT 侧测量
    log('测量 SFT 模型 …')
    results.sft = await measureAt({ endpoint: sftEndpoint.url, model: sftEndpoint.model, heldout, wikitext, mmlu, nTask, nLm, nMmlu, vocabSize, log })

    // ③ 基线测量（缓存到 registry：同一基座只测一次）
    // 缓存键含**评估协议版本**：协议变了（例如修正了饱和的判据）缓存必须自动失效，
    // 否则会拿旧协议下的基线数字与新协议下的结果比较 —— 那是最隐蔽的一类错判。
    const cacheKey = `${run.baseModel.path}@${run.baseModel.revision ?? 'norev'}@${EVAL_PROTOCOL}@task${nTask}/lm${nLm}/mmlu${nMmlu}`
    const cachedRaw = reg.baselineCache?.[cacheKey]
    // **缺陷 15**：只有**能证明完整**的缓存才可复用；"未知"不等于"完整"（fail-closed）。
    const cachedCheck = cachedRaw ? measurementCompleteness(cachedRaw) : null
    if (cachedRaw && !cachedCheck.complete) log(`  基线缓存**不可复用**（${cachedCheck.reason}）⇒ 本次重新测量基线`)
    const cached = (cachedRaw && cachedCheck.complete) ? cachedRaw : null
    if (cached) {
      // 基线**没有部署**（用了缓存）⇒ 本次没有 deploy-report ⇒ `instrument.base = null`。
      // **不编造**：缓存里的基线数字是历史上某次部署测的，那份报告当时没有被留存
      // （baselineCache 只存了测量摘要）。故这里只能诚实地记 null；
      // 想补全这条，得让 baselineCache 一并缓存 instrument —— 属后续工作，本次不做。
      results.base = cached
      baseInstrument = null
      log('基线用缓存（本次无基线部署 ⇒ instrument.base = null，不编造）')
    } else {
      log(`部署基座 ${run.baseModel.path} …`)
      const b = await callWithConfirm(client, 'inference_serving_deploy', {
        path: run.baseModel.path, owner: 'training-sft', purpose: `长任务2 基线测量 ${runId}`,
        servedModelName: `base-${runId}`, gpuMemoryUtilization: 0.18, maxNumSeqs: 4,
      }, { log })
      if (b.ok === false) throw new ToolError(ERROR_CODES.E_SERVING_UNAVAILABLE, { hint: `基座部署失败：${b.code}`, details: b })
      baseEndpoint = { url: b.endpoint.endpoint, model: b.endpoint.servedModelName, id: b.endpoint.id, ownerToken: b.ownerToken, container: b.endpoint.containerName }
      baseInstrument = instrumentFromDeployReport(b.report)
      guardInstrument(baseInstrument, runId) // 基线仪器同样不得有损（基线坏了，比较就不成立）
      log(`测量基座 … (lossy=${baseInstrument?.lossy ?? null})`)
      results.base = await measureAt({ endpoint: baseEndpoint.url, model: baseEndpoint.model, heldout, wikitext, mmlu, nTask, nLm, nMmlu, vocabSize, log })
    }

    // ④ 门禁
    const validity = [
      { name: 'sft-forward', ok: results.sft.uniform.n > 0, detail: `SFT 侧拿到 ${results.sft.uniform.n} 个 logprob` },
      { name: 'sft-not-uniform', ok: results.sft.uniform.ok === true, detail: results.sft.uniform.criterion ?? '' },
      { name: 'task-samples', ok: results.sft.task.nOk >= Math.min(20, nTask), detail: `SFT 侧有效生成 ${results.sft.task.nOk}/${results.sft.task.n}` },
    ]
    const tg = taskGate({ base: results.base.task, sft: results.sft.task, thresholds: thresholdsResolved, validity })
    const fg = forgettingGate({ baseMmlu: results.base.mmlu, sftMmlu: results.sft.mmlu, baseLm: results.base.lm, sftLm: results.sft.lm })
    const pass = tg.pass && fg.pass

    // 写出**符合冻结 eval-contract 的评估结果**（`ai-infra/eval-result@1`）——
    // 这是 P0 #9「训练侧评估与推理侧评估契约一致」的可验证落点：
    // 形状由 inference-contracts/state-schema/eval-result.schema.json 校验（CI 的 C16）。
    const evalResult = {
      schema: 'ai-infra/eval-result@1',
      subject: { kind: 'checkpoint', ref: run.merged.dir, revision: null },
      measurement: {
        n: results.sft.task.nOk + results.sft.lm.texts + results.sft.mmlu.n,
        unit: 'samples',
        datasetRef: `${run.dataset.dir}#heldout+wikitext-test+mmlu-test`,
        // lineageHash 本身已带 sha256: 前缀；这里只做一次规范化，避免出现双前缀
        datasetHash: run.dataset.lineageHash ? String(run.dataset.lineageHash).replace(/^(sha256:)+/, 'sha256:') : null,
      },
      metrics: [
        { name: 'format_compliance', value: results.sft.task.formatCompliance, unit: 'ratio', direction: 'higher-is-better', baseline: results.base.task.formatCompliance, relativePct: tg.deltas.formatCompliance.rawPct },
        { name: 'token_f1', value: results.sft.task.f1, unit: 'f1', direction: 'higher-is-better', baseline: results.base.task.f1, relativePct: tg.deltas.f1.rawPct },
        { name: 'mmlu_accuracy', value: results.sft.mmlu.accuracy, unit: 'ratio', direction: 'higher-is-better', baseline: results.base.mmlu.accuracy, relativePct: null },
        { name: 'wikitext_mean_nll', value: results.sft.lm.meanNll, unit: 'nats', direction: 'lower-is-better', baseline: results.base.lm.meanNll, relativePct: null },
      ],
      validity: tg.checks.filter((c) => c.kind === 'validity').map((c) => ({ name: c.name, ok: c.ok, detail: c.detail ?? null })),
      gate: { pass: tg.pass, validityOk: tg.validityOk, reason: tg.reason, checks: tg.checks },
      versions: { testingCore: TESTING_CORE_VERSION, assets: EVAL_PROTOCOL },
      // 仪器来源（**additive**，schema 里是可选属性）：这份判决是在**哪两次部署**上测出来的、
      // 有没有走 serving 的有损降级档位。`instrument.base = null` 表示基线本次没有部署（用缓存）。
      instrument: { base: baseInstrument, candidate: candidateInstrument },
    }
    writeFileSync(join(run.runDir, 'eval-result.json'), `${JSON.stringify(evalResult, null, 2)}\n`)
    const ar = await audit(root, { capability: 'validate', action: 'validate', phase: 'end', runId, outcome: pass ? 'pass' : 'fail', taskGate: tg.pass, validityOk: tg.validityOk, forgettingGate: fg.pass, formatBase: results.base.task.formatCompliance, formatSft: results.sft.task.formatCompliance, f1Base: results.base.task.f1, f1Sft: results.sft.task.f1, mmluBase: results.base.mmlu.accuracy, mmluSft: results.sft.mmlu.accuracy, lmPplBase: results.base.lm.ppl, lmPplSft: results.sft.lm.ppl, ms: Date.now() - t0 })
    // **缺陷 15**：部分测量**不得**入缓存（fail-closed）。只有能证明完整的测量才写入。
    const baseCacheNext = (() => {
      if (cached) return null                       // 本次复用了完整缓存 ⇒ 保持原样
      const chk = measurementCompleteness(results.base)
      if (!chk.complete) { log(`  基线测量不完整（${chk.reason}）⇒ **不写入缓存**（fail-closed）`); return null }
      return attachIntegrity(results.base)
    })()
    await updateRegistry(root, (doc) => ({
      ...doc,
      baselineCache: baseCacheNext ? { ...(doc.baselineCache ?? {}), [cacheKey]: baseCacheNext } : (doc.baselineCache ?? {}),
      runs: (doc.runs || []).map((r) => (r.id === runId ? {
        ...r, status: pass ? 'validated' : 'gate-failed', validatedAt: new Date().toISOString(),
        endpoint: sftEndpoint, evaluation: { thresholds: thresholdsResolved, taskGate: tg, forgettingGate: fg, base: summarize(results.base), sft: summarize(results.sft) },
        decidedBudget: { gpuHoursTrain: r.result?.gpuHours ?? null, gpuHoursValidate: Number(((Date.now() - t0) / 3_600_000).toFixed(4)) },
        evalResult: { schema: evalResult.schema, path: join(run.runDir, 'eval-result.json') },
        auditSeqs: [...(r.auditSeqs || []), ar.seq],
      } : r)),
    }), { schemaPath: schemaPath() })

    return { runId, pass, codeVersion: codeVersion(), taskGate: tg, forgettingGate: fg, endpoint: sftEndpoint, base: summarize(results.base), sft: summarize(results.sft), audit: { seq: ar.seq, hash: ar.hash }, ms: Date.now() - t0 }
  } finally {
    if (teardown) {
      for (const [label, ep, tok] of [['sft', sftEndpoint, ownerToken], ['base', baseEndpoint, baseEndpoint?.ownerToken]]) {
        if (!ep) continue
        try {
          const r = await callWithConfirm(client, 'inference_serving_teardown', { selector: ep.id, ownerToken: tok }, { log })
          log(`  teardown ${label}: ok=${r.ok} ${r.code ?? ''}`)
        } catch (e) { log(`  teardown ${label} 异常：${e.message}`) }
      }
    }
    await client?.close?.()
  }
}

/** 持久化测量摘要里"是谁产出了这份测量"——供下游（如 `effectJudgment`）判定**不自证**。 */
export const MEASUREMENT_PRODUCER = 'training-sft.validate'

/**
 * 从仪器测量里**原样搬运**完整性证据（缺陷 14/15 的持久化面，与 pretrain 侧同源修复）。
 *
 * ⚠️ `attempted` / `failed` 一旦被摘要丢掉就**不可恢复**（`succeeded` 还能从 `perText.length` 倒推，
 * 另两个不能）⇒ "端点中途失效"与"评估集本来就小"在**落盘后**重新变得不可区分。
 * 实测（修复前，`~/.ai-infra/state/sft-registry.yaml`）：字符串 `attempted` 出现 **0 次**。
 *
 * **只在源字段存在时搬运**：源里没有就**不写**（而不是补 `null` —— `null` 会被读成"我们知道它是 null"）。
 * 回归测试：`tests/unit/persisted-integrity.spec.mjs`。
 *
 * 注：`task` 是"已知但非逐条网络测量"的部件（确定性打分），其原始测量**没有** attempted/failed，
 * 故不搬运；`measurementCompleteness` 对它的分类是 `KNOWN_OTHER`，不要求完整性证据。
 */
const INTEGRITY_FIELDS = ['attempted', 'succeeded', 'failed', 'skipped', 'partial']
const integrityOf = (x) => {
  const out = {}
  if (!x || typeof x !== 'object') return out
  for (const k of INTEGRITY_FIELDS) if (x[k] !== undefined) out[k] = x[k]
  return out
}

export const summarize = (m) => ({
  uniform: { ok: m.uniform.ok, n: m.uniform.n, epsilon: m.uniform.epsilon, delta: m.uniform.delta, bitIdenticalAcrossRuns: m.uniform.bitIdenticalAcrossRuns },
  // ⚠️ `task` 自 v0.3.0 起属于**逐条网络测量** ⇒ **必须**带完整性证据（`evalTask` 现在记录了）。
  // 这一行同样是**手写字段表**：只列数值、漏掉 attempted/failed ⇒ 落盘摘要会判 `task:no-integrity-evidence`。
  // （实测：这就是 `persisted-integrity.spec` 在本轮先红的那一条。）
  task: { ...integrityOf(m.task), n: m.task.n, nOk: m.task.nOk, formatCompliance: m.task.formatCompliance, f1: m.task.f1, f1n: m.task.f1n, truncated: m.task.truncated },
  lm: { ...integrityOf(m.lm), texts: m.lm.texts, tokens: m.lm.tokens, meanNll: m.lm.meanNll, ppl: m.lm.ppl, perText: m.lm.perText },
  mmlu: { ...integrityOf(m.mmlu), n: m.mmlu.n, correct: m.mmlu.correct, accuracy: m.mmlu.accuracy, asked: m.mmlu.asked },
  integrity: { producedBy: MEASUREMENT_PRODUCER },
})

/* ══════════════════════════════════════════════════════════ 4. rollback */

export async function rollback({ root = paths().root, runId, purge = false, log = () => {} }) {
  const reg = await readRegistry(root, schemaPath())
  const run = (reg.runs || []).find((r) => r.id === runId)
  if (!run) throw new ToolError(ERROR_CODES.E_RUN_NOT_FOUND, { hint: `run 不存在：${runId}`, recoverable: true })
  await audit(root, { capability: 'rollback', action: 'rollback', phase: 'begin', runId, purge })
  let purged = []
  if (purge) {
    for (const d of [run.runDir, join(paths(root).runsDir, runId, 'merged')]) {
      if (d && existsSync(d) && d.startsWith(paths(root).runsDir)) { await fs.rm(d, { recursive: true, force: true }); purged.push(d) }
    }
  }
  const ar = await audit(root, { capability: 'rollback', action: 'rollback', phase: 'end', runId, purged: purged.length })
  await withRun(root, runId, (r) => ({ ...r, status: 'rolled-back', rolledBackAt: new Date().toISOString(), purged, auditSeqs: [...(r.auditSeqs || []), ar.seq] }))
  log(`run ${runId} → rolled-back（purge=${purge}, 删除 ${purged.length} 个目录）`)
  return { runId, status: 'rolled-back', purged, audit: { seq: ar.seq, hash: ar.hash } }
}

/* ══════════════════════════════════════════════════════════ 5. get_report */

export async function getReport({ root = paths().root, runId = null, limit = 10 }) {
  const reg = await readRegistry(root, schemaPath())
  const chain = await auditVerify(root)
  if (!runId) {
    return { runs: (reg.runs || []).slice(-limit).map((r) => ({ id: r.id, status: r.status, startedAt: r.startedAt, steps: r.result?.steps ?? null, gpuHours: r.result?.gpuHours ?? null, formatCompliance: r.evaluation?.sft?.task?.formatCompliance ?? null, pass: r.evaluation ? (r.evaluation.taskGate?.pass && r.evaluation.forgettingGate?.pass) : null })), audit: { ok: chain.ok, shards: chain.shards } }
  }
  const run = (reg.runs || []).find((r) => r.id === runId)
  if (!run) throw new ToolError(ERROR_CODES.E_RUN_NOT_FOUND, { hint: `run 不存在：${runId}`, recoverable: true })
  return { run, audit: { ok: chain.ok, shards: chain.shards, results: chain.results.map((x) => ({ day: x.day, checked: x.checked, ok: x.ok })) } }
}

/* ══════════════════════════════════════════════════════════ 6. health */

export async function health({ root = paths().root, deep = false }) {
  const p = P(root)
  const probe = await sandboxProbe()
  const chain = await auditVerify(root)
  let gpu = null
  try {
    const out = execFileSync('nvidia-smi', ['--query-gpu=name,driver_version', '--format=csv,noheader'], { encoding: 'utf8', timeout: 20000 }).trim()
    gpu = out
  } catch { gpu = null }
  let imageOk = false
  let imageId = null
  try {
    imageId = execFileSync('docker', ['images', '-q', TRAINING_IMAGE], { encoding: 'utf8', timeout: 20000 }).trim() || null
    imageOk = Boolean(imageId)
  } catch { imageOk = false }
  const baseOk = process.env.DSH_BASE_MODEL ? existsSync(join(process.env.DSH_BASE_MODEL, 'config.json')) : false
  let servingOk = false
  let servingPath = null
  try { const sp = await resolveServingPath({ root }); servingPath = sp.serverPath; servingOk = true } catch { servingOk = false }
  const reg = await readRegistry(root, schemaPath())
  return {
    ok: probe.container.available && chain.ok && imageOk && baseOk && servingOk,
    sandboxTiers: { process: probe.process.available, container: probe.container.available, vm: probe.vm.available },
    sandboxEvidence: probe.container.evidence,
    image: { name: TRAINING_IMAGE, available: imageOk, id: imageId },
    gpu,
    baseModel: { path: process.env.DSH_BASE_MODEL ?? null, available: baseOk },
    serving: { available: servingOk, serverPath: servingPath },
    audit: { ok: chain.ok, shards: chain.shards, detail: chain.results.map((x) => ({ day: x.day, checked: x.checked, ok: x.ok })) },
    registry: { runs: (reg.runs || []).length, file: p.registryFile },
    limits: DEFAULT_LIMITS,
    version: { pkg: PACKAGE_VERSION, schema: SCHEMA_VERSION },
    // `deep` 为 false 时**不要留一个 undefined 属性**：DSH 的输出校验要求"无损 JSON"，
    // 而 `undefined` 不是 JSON 值（JSON.stringify 会静默丢掉它，校验器却会拒绝）。
    ...(deep ? { deep: { paths: p } } : {}),
  }
}

function readJsonl(file) {
  return readFileSync(file, 'utf8').split('\n').filter((l) => l.trim()).map((l) => JSON.parse(l))
}

export { enforceLimits, dualGate, PKG_ROOT }
