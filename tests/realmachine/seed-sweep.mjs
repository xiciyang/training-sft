/**
 * seed-sweep —— 长任务 3 · 阶段 1：SFT 多种子复验（用户裁决 ②）。
 *
 * ## 为什么要用**已交付的插件路径**而不是自己写循环
 * 本脚本**不实现**任何训练循环、不重算任何判据：它只按顺序调训练侧已交付的
 *   `train()`（→ training-core 公共循环 + 资源上限 + checkpoint + 审计链）
 *   `mergeRun()`（→ 容器内合并，产出 weightsSha256 指纹）
 *   `validate()`（→ inference-serving 部署 + testing-core 双重门禁 + 评估）
 * 也就是说：**仪器与已验收闭环完全同一台**。这是校准给的硬约束 ——
 *
 * ## 校准结论（为什么不省这趟 GPU 钱）
 * `calibrate-container-nll.mjs` 实测：容器内 HF 路径与 serving 路径在**逐文本**上
 * r=0.9997 高度相关，但**配对差本身**有 ±1e-3 量级（base +4.3e-4 / sft −9.3e-4），
 * 而待测的遗忘量只有 5e-4 ~ 1.9e-3。两条仪器对 Δ 的估计相差 68% 的判据上限：
 *   容器自洽 Δ=0.000516 (bound 0.002219)  vs  serving Δ=0.001872 (bound 0.001990)
 * ⇒ 已验收的"临界通过（余量 5.9%）"**不仅贴统计阈值，还贴仪器可复现带**。
 * 因此多种子复验**必须**用 serving 这一台仪器，容器路径只能用于诊断、不能用于判定。
 *
 * ## 用法
 *   node tests/realmachine/seed-sweep.mjs [--seeds 42,123,456] [--steps 200]
 */
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import { homedir } from 'node:os'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { train, mergeRun, validate, getReport } from '../../src/core.mjs'
import { evalLM } from '../../src/evaluate.mjs'
import { resolveServingPath, ServingMcpClient, callWithConfirm } from '../../src/serving-client.mjs'
import { loadEvalSets } from '../../src/data/prepare.mjs'
import { mean, stddev } from '@xiciyang/testing-core'

const HERE = dirname(fileURLToPath(import.meta.url))
const HOME = homedir()
const ROOT = process.env.INFERENCE_TRAINING_ROOT || join(HOME, '.ai-infra')
const OUT = join(HERE, '..', '..', '..', 'out', 'longtask3-seed-sweep.json')

const argv = process.argv.slice(2)
const argOf = (k, d) => { const i = argv.indexOf(k); return i >= 0 && argv[i + 1] ? argv[i + 1] : d }
const SEEDS = String(argOf('--seeds', '42,123,456')).split(',').map((s) => Number(s.trim())).filter((n) => Number.isFinite(n))
const STEPS = Number(argOf('--steps', '200'))
// `--lm-only`：只测 wikitext NLL（裁决 ② 要的正是这一项），跳过 evalTask/evalMMLU/evalUniform。
// 动因是**实测预算**：全量 validate 每个种子 0.068 GPU-h，其中大头是 evalTask（100 次 ≤256 token 生成）
// 与 evalMMLU（300 次生成），而它们回答的是任务门禁、不是「遗忘是否随种子稳定」。
const LM_ONLY = argv.includes('--lm-only')

/**
 * 遗忘门禁的 **LM 分量**（与 `src/evaluate.mjs` 的 `forgettingGate` §② **同式**）。
 * 为什么不直接调 `forgettingGate`：它同时要求 MMLU 样本，LM-only 模式下没有 MMLU，
 * 传假值就是伪造。故此处只复算 §② 那四行；并且**用 seed 42 的真实全量结果自校验**
 * （见 `selfCheckGate`）——"断言的语义必须被独立验证"，复算也要被验证。
 */
function pairedLmGate(basePerText, sftPerText, z = 1.96) {
  const n = Math.min(basePerText.length, sftPerText.length)
  const diffs = []
  for (let i = 0; i < n; i += 1) diffs.push(sftPerText[i] - basePerText[i])
  const d = mean(diffs)
  const se = stddev(diffs) / Math.sqrt(n)
  const bound = z * se
  return { n, deltaNll: d, se, bound, pass: d <= bound, margin: (bound - d) / bound }
}

/** 自校验：用 seed 42 的**已落盘全量 validate** 结果验证 pairedLmGate 与 delivered forgettingGate.lm 一致。 */
function selfCheckGate(prev, basePerText, sftPerText) {
  const mine = pairedLmGate(basePerText, sftPerText)
  const ref = (prev?.perSeed || []).find((p) => p.seed === 42 && p.validate?.forgettingGate?.lm)?.validate.forgettingGate.lm ?? null
  if (!ref) return { checked: false, reason: '无 seed 42 全量结果可作参照' }
  const c = {
    checked: true,
    refDelta: ref.deltaNll, mineDelta: mine.deltaNll, deltaEq: Math.abs(ref.deltaNll - mine.deltaNll) < 1e-12,
    refBound: ref.bound, mineBound: mine.bound, boundEq: Math.abs(ref.bound - mine.bound) < 1e-12,
    refPass: ref.pass, minePass: mine.pass, passEq: ref.pass === mine.pass,
  }
  c.ok = c.deltaEq && c.boundEq && c.passEq
  return c
}

// 与已验收 run ts_20260929044120_kj5q **逐参数相同**（唯一种子不同）——
// 否则"跨种子差异"会与"配置差异"混在一起，判据失义。
const DATASET = join(HOME, '.ai-infra/state/datasets/yahma-alpaca-cleaned__sft__s20260929__n1000')
const BASE_MODEL = '/data/models/Qwen2.5-0.5B-Instruct'
const COMMON = Object.freeze({
  root: ROOT, baseModel: BASE_MODEL, datasetDir: DATASET,
  lora: {}, lr: 1e-4, weightDecay: 0.0,
  batchSize: 4, gradAccum: 4, maxSeqLen: 256, maxSteps: STEPS, warmupSteps: 10,
  checkpointEvery: 50, checkpointKeep: 3,
})

const log = (m) => process.stderr.write(`[sweep] ${m}\n`)
const t0 = Date.now()
const report = {
  sweepStartedAt: new Date().toISOString(),
  purpose: 'SFT 多种子复验（裁决 ②）：wikitext 遗忘是否稳定，非单次运气',
  instrument: 'serving（与已验收 run 同一台）',
  config: { ...COMMON, datasetDir: DATASET, baseModel: BASE_MODEL },
  seeds: SEEDS,
  perSeed: [],
  budget: { unit: 'GPU 小时（项目既有口径：训练取 training-core 循环墙钟；validate 取整段墙钟）' },
}
mkdirSync(dirname(OUT), { recursive: true })
const flush = () => writeFileSync(OUT, `${JSON.stringify(report, null, 1)}\n`)
// **跨次运行累积**：seed 42 已单独跑过，第二次跑 123/456 不能把它的结果冲掉。
// 同时把上一版结果读进 PREV —— 它同时充当「门禁 helper 自校验」的免费参照（无需重烧 GPU）。
const PREV = (() => { try { return JSON.parse(readFileSync(OUT, 'utf8')) } catch { return null } })()

// 缓存的 base 逐文本 NLL（与已验收 run 同一台仪器、同一份 base、同一协议）。
// 取自已落盘的 seed 42 全量结果（它就是「基线用缓存」的那一份），不重算。
const BASE_LM = PREV?.perSeed?.find((p) => p.validate?.base?.lm?.perText)?.validate.base.lm ?? null
if (LM_ONLY && !BASE_LM) throw new Error('LM-only 模式需要上一版结果里的缓存 base perText（缺失则先跑一次全量 validate）')

if (LM_ONLY && PREV) {
  const e42 = PREV.perSeed.find((p) => p.seed === 42 && p.validate?.forgettingGate?.lm)
  if (e42) {
    const chk = selfCheckGate(PREV, e42.validate.base.lm.perText, e42.validate.sft.lm.perText)
    chk.vsDelivered = e42.validate.forgettingGate.lm
    report.gateHelperSelfCheck = chk
    console.error(`[sweep] 门禁 helper 自校验（vs seed 42 已交付 forgettingGate）：ok=${chk.ok} Δeq=${chk.deltaEq} boundEq=${chk.boundEq} passEq=${chk.passEq}`)
    if (!chk.ok) throw new Error('LM 门禁复算与已交付 forgettingGate 不一致 —— 仪器不等价，拒绝继续')
  }
}

// 继承上一版的既有 seed 结果
const perSeed = new Map()
for (const p of (PREV?.perSeed ?? [])) perSeed.set(p.seed, p)
let gpuTrain = Number(PREV?.budget?.gpuHoursTrain ?? 0)
let gpuValidate = Number(PREV?.budget?.gpuHoursValidate ?? 0)
report.perSeed = [...perSeed.values()]
report.mode = LM_ONLY ? 'lm-only（仅 wikitext NLL；任务/MMLU 门禁不在本模式范围）' : 'full validate'
flush()

const WIKITEXT_ROWS = (() => {
  const sets = loadEvalSets()
  const s = sets.find((x) => x.name.startsWith('wikitext-test(c128)')) || sets.find((x) => x.kind === 'lm')
  return s?.rows ?? []
})()

/** LM-only：部署合并模型 → 只跑 evalLM → 拆解。仪器与 validate 内的 evalLM **同一实现**。 */
async function measureLmOnly({ runId, mergedDir, log }) {
  const sp = await resolveServingPath({ root: ROOT })
  const client = await new ServingMcpClient(sp.serverPath).start()
  let ep = null; let ownerToken = null
  try {
    const d = await callWithConfirm(client, 'inference_serving_deploy', {
      path: mergedDir, owner: 'training-sft', purpose: `长任务3 阶段1 多种子复验 ${runId}`,
      servedModelName: `sweep-${runId}`, gpuMemoryUtilization: 0.18, maxNumSeqs: 4,
    }, { log })
    if (d.ok === false) throw new Error(`部署失败：${d.code} ${d.message ?? ''}`)
    ownerToken = d.ownerToken
    ep = { url: d.endpoint.endpoint, model: d.endpoint.servedModelName, id: d.endpoint.id, container: d.endpoint.containerName, usability: d.report?.attempts?.[0]?.modelUsability ?? null }
    const lm = await evalLM({ endpoint: ep.url, model: ep.model, rows: WIKITEXT_ROWS, limit: 200, log })
    return { endpoint: ep, lm }
  } finally {
    if (ep) { try { await callWithConfirm(client, 'inference_serving_teardown', { selector: ep.id, ownerToken }, { log }) } catch (e) { log(`teardown 异常：${e.message}`) } }
    await client?.close?.()
  }
}

for (const seed of SEEDS) {
  const entry = { seed, startedAt: new Date().toISOString() }
  const idx = report.perSeed.findIndex((p) => p.seed === seed)
  if (idx >= 0) report.perSeed[idx] = entry; else report.perSeed.push(entry)
  flush()
  try {
    log(`── seed=${seed} :: train（${STEPS} 步）`)
    const tr = await train({ ...COMMON, seed, log })
    entry.train = { runId: tr.runId, status: tr.status, steps: tr.steps, gpuHours: tr.gpuHours, stopReason: tr.stopReason, finalLoss: tr.finalLoss, checkpoints: tr.checkpoints }
    gpuTrain += Number(tr.gpuHours ?? 0)
    flush()

    log(`── seed=${seed} :: merge（产出 weightsSha256 指纹）`)
    // checkpointName **必须显式给出**：mergeRun 的缺省(null)会让容器去 `/out/adapter` 找 adapter，
    // 而真实权重在 `/out/checkpoints/<name>/adapter` ⇒ 会加载失败。名字从注册表读回（**不猜**）。
    const reg = await getReport({ root: ROOT, runId: tr.runId })
    const ckptName = reg.run?.checkpoint?.name
    if (!ckptName) throw new Error(`run ${tr.runId} 注册表里没有 checkpoint.name，无法合并`)
    const mg = await mergeRun({ root: ROOT, runId: tr.runId, checkpointName: ckptName, log })
    entry.merge = { checkpointName: ckptName, mergedDir: mg.mergedDir, weightsSha256: mg.weightsSha256, params: mg.params, bytes: mg.bytes }
    flush()

    if (LM_ONLY) {
      log(`── seed=${seed} :: evalLM（serving 仪器，仅 wikitext NLL）`)
      const t1 = Date.now()
      const r = await measureLmOnly({ runId: tr.runId, mergedDir: mg.mergedDir, log })
      const g = pairedLmGate(BASE_LM.perText, r.lm.perText)
      gpuValidate += Number(((Date.now() - t1) / 3_600_000).toFixed(4))
      entry.lmOnly = {
        endpoint: r.endpoint,
        lm: { texts: r.lm.texts, tokens: r.lm.tokens, meanNll: r.lm.meanNll, ppl: r.lm.ppl, perText: r.lm.perText },
        base: { texts: BASE_LM.texts, tokens: BASE_LM.tokens, meanNll: BASE_LM.meanNll, perText: BASE_LM.perText },
        gate: g,
        gpuHours: Number(((Date.now() - t1) / 3_600_000).toFixed(4)),
      }
      entry.validate = { pass: g.pass, forgettingGatePass: g.pass, forgettingGate: { lm: g, mmlu: null, note: 'LM-only 模式未测 MMLU' }, sft: { lm: entry.lmOnly.lm }, base: entry.lmOnly.base }
    } else {
      log(`── seed=${seed} :: validate（serving 仪器 + testing-core 双门禁）`)
      const va = await validate({ root: ROOT, runId: tr.runId, deploy: true, nTask: 100, nLm: 200, nMmlu: 300, teardown: true, log })
      gpuValidate += Number(((va.ms ?? 0) / 3_600_000).toFixed(4))
      entry.validate = {
        pass: va.pass,
        taskGatePass: va.taskGate?.pass, forgettingGatePass: va.forgettingGate?.pass,
        taskGate: va.taskGate, forgettingGate: va.forgettingGate,
        base: va.base, sft: va.sft,
        gpuHours: Number(((va.ms ?? 0) / 3_600_000).toFixed(4)),
      }
    }
    entry.status = 'ok'
  } catch (e) {
    entry.status = 'failed'
    entry.error = String(e?.message ?? e).slice(0, 600)
    log(`seed=${seed} 失败：${entry.error}`)
  }
  entry.finishedAt = new Date().toISOString()
  entry.gpuHoursCumulative = { train: gpuTrain, validate: gpuValidate, total: gpuTrain + gpuValidate }
  flush()
}

report.perSeed.sort((a, b) => a.seed - b.seed)
report.budget.gpuHoursTrain = Number(gpuTrain.toFixed(4))
report.budget.gpuHoursValidate = Number(gpuValidate.toFixed(4))
report.budget.gpuHoursTotal = Number((gpuTrain + gpuValidate).toFixed(4))
report.wallMs = Date.now() - t0
report.finishedAt = new Date().toISOString()
flush()

console.log('── 种子扫描完成 ──')
for (const e of report.perSeed) {
  if (e.status !== 'ok') { console.log(`seed=${e.seed} FAILED ${e.error}`); continue }
  const fg = e.validate.forgettingGate
  console.log(`seed=${e.seed} run=${e.train.runId} pass=${e.validate.pass} nllΔ=${fg?.lm?.deltaNll?.toFixed(6)} bound=${fg?.lm?.bound?.toFixed(6)} sftNll=${e.validate.sft?.lm?.meanNll?.toFixed(6)} weights=${e.merge.weightsSha256.slice(0, 24)}…`)
}
console.log(`预算：train ${report.budget.gpuHoursTrain} + validate ${report.budget.gpuHoursValidate} = ${report.budget.gpuHoursTotal} GPU-h   wall=${(report.wallMs / 60000).toFixed(1)}min`)
console.log(`report -> ${OUT}`)
