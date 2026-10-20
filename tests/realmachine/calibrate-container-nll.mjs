/**
 * 校准：**容器内**（HF + peft 直接前向）测得的 wikitext NLL，能否作为
 * **serving 路径**（vLLM echo+logprobs）测得 NLL 的代理？
 *
 * 为什么必须先校准（长任务 3 · 阶段 1）：
 *   已验收的遗忘判据（Δ NLL ≤ 1.96·SE，余量仅 5.9%）是用 serving 路径得到的。
 *   阶段 1 要在 ≤0.15 GPU-h 内跑 3 个种子；serving 路径每种子要起 vLLM 容器
 *   （≈0.068 GPU-h/次），3 种子必然超预算。容器内 `eval_lm` 便宜得多。
 *   但"便宜"不是理由 —— **换了仪器就必须先证明仪器等价**，否则重演
 *   "测量不成立"（长任务 2 缺陷 1）。故本脚本用**同一个已验收 checkpoint**
 *   （seed=20260929 的 step-000200 adapter）在容器内测 base/sft，与 serving 数字对比。
 *
 * **落盘纪律（第一次跑栽在这）**：每次 eval 一完成就立刻写盘。
 *   第一次跑把结果只留在内存里，随后 stop() 抛异常 → perText 全丢，白烧一次 GPU。
 *   修法：增量落盘 + stop() 包 try/catch。
 *
 * 三条**预先声明**的判据（不事后调整）：
 *   A. 逐文本数组可比：容器 perText 与 serving perText 的配对差均值 |Δ| ≤ 0.1×bound_serving
 *   B. 门禁判定一致：容器自洽配对门禁的 pass/fail 与已验收一致
 *   C. 聚合口径分离：容器 meanNll 是 token 加权、serving 是逐文本等权 ——
 *      若"容器 perText 的**等权均值**"≈ serving meanNll，则差异仅为口径，逐文本值本身一致
 */
import { readFileSync, writeFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'
import { ContainerWorker, trainingMounts } from '../../src/worker.mjs'
import { mean, stddev } from '@xiciyang/testing-core'

const HOME = homedir()
const IMAGE = 'training-sft/toolchain:0.1.0'
const BASE_MODEL = '<BASE_MODEL>'
const ACCEPTED_RUN = 'ts_20260929044120_kj5q'
const CKPT = 'step-000200'
const RUN_DIR = join(HOME, '.ai-infra/run/training-sft', ACCEPTED_RUN)
const DATASET = join(HOME, '.ai-infra/state/datasets/yahma-alpaca-cleaned__sft__s20260929__n1000')
const WIKITEXT = join(HOME, '.ai-infra/state/datasets/Salesforce-wikitext__wikitext-2-raw-v1__train__s20260923__c128/test.jsonl')
const BASELINE = join(HOME, 'deepseek_harness/inference-training/out/longtask3-baseline.json')
const OUT = join(HOME, 'deepseek_harness/inference-training/out/longtask3-calibration.json')

const N_LM = 200
const log = (m) => process.stderr.write(`[calib] ${m}\n`)

function pairedGate(basePerText, sftPerText, z = 1.96) {
  const n = Math.min(basePerText.length, sftPerText.length)
  const diffs = []
  for (let i = 0; i < n; i += 1) diffs.push(sftPerText[i] - basePerText[i])
  const d = mean(diffs)
  const se = stddev(diffs) / Math.sqrt(n)
  const bound = z * se
  return { n, deltaNll: d, se, bound, pass: d <= bound, margin: (bound - d) / bound }
}

function pearson(a, b) {
  const n = Math.min(a.length, b.length)
  const ma = mean(a.slice(0, n)); const mb = mean(b.slice(0, n))
  let num = 0; let da = 0; let db = 0
  for (let i = 0; i < n; i += 1) { const x = a[i] - ma; const y = b[i] - mb; num += x * y; da += x * x; db += y * y }
  return num / Math.sqrt(da * db)
}

const t0 = Date.now()
const report = {
  calibratedAt: new Date().toISOString(),
  purpose: '判定容器内 wikitext NLL 是否可作 serving 路径测量的代理',
  image: IMAGE, acceptedRunId: ACCEPTED_RUN, checkpoint: CKPT, nLm: N_LM,
  measurements: {},
}
const flush = () => writeFileSync(OUT, `${JSON.stringify(report, null, 1)}\n`)

const worker = new ContainerWorker({
  image: IMAGE,
  mounts: trainingMounts({ modelDir: BASE_MODEL, datasetDir: DATASET, outDir: RUN_DIR, wikitextFile: WIKITEXT }),
  log,
}).start()

try {
  const ping = await worker.call('ping')
  report.worker = { torch: ping.torch, cuda: ping.cuda }
  flush()

  // ── ① 容器内 base（无 adapter；peft LoRA B 零初始化 ⇒ 输出等同 base）
  log('init(base) …')
  const initBase = await worker.call('init', {
    config: {
      baseModel: `/models/${BASE_MODEL.split('/').filter(Boolean).pop()}`,
      trainFile: '/data/train.jsonl',
      lora: {}, lr: 1e-4, batchSize: 4, gradAccum: 4, maxSeqLen: 512,
      maxSteps: 1, warmupSteps: 0, seed: 20260929,
    },
  }, { timeoutMs: 20 * 60 * 1000 })
  log(`init(base) done loadedInS=${initBase.loadedInS} trainable=${initBase.trainable}`)
  const cb = await worker.call('eval_lm', { file: '/eval/wikitext.jsonl', maxSamples: N_LM, maxSeqLen: 512 }, { timeoutMs: 30 * 60 * 1000 })
  log(`container base: tokenWeightedMeanNll=${cb.meanNll} texts=${cb.texts} tokens=${cb.tokens}`)
  report.measurements.containerBase = { meanNllTokenWeighted: cb.meanNll, texts: cb.texts, tokens: cb.tokens, ppl: cb.ppl, perText: cb.perTextNll }
  flush()

  // ── ② 容器内 sft（加载已验收 checkpoint 的 adapter）
  log('init(sft adapter) …')
  const initSft = await worker.call('init', {
    config: {
      baseModel: `/models/${BASE_MODEL.split('/').filter(Boolean).pop()}`,
      trainFile: '/data/train.jsonl',
      lora: {}, lr: 1e-4, batchSize: 4, gradAccum: 4, maxSeqLen: 512,
      maxSteps: 1, warmupSteps: 0, seed: 20260929,
      resumeAdapter: `/out/checkpoints/${CKPT}/adapter`,
    },
  }, { timeoutMs: 20 * 60 * 1000 })
  log(`init(sft) done resumed=${initSft.resumed} loadedInS=${initSft.loadedInS}`)
  const cs = await worker.call('eval_lm', { file: '/eval/wikitext.jsonl', maxSamples: N_LM, maxSeqLen: 512 }, { timeoutMs: 30 * 60 * 1000 })
  log(`container sft: tokenWeightedMeanNll=${cs.meanNll} texts=${cs.texts} tokens=${cs.tokens}`)
  report.measurements.containerSft = { meanNllTokenWeighted: cs.meanNll, texts: cs.texts, tokens: cs.tokens, ppl: cs.ppl, perText: cs.perTextNll }
  flush()

  try { await worker.call('shutdown') } catch { /* 已退出 */ }
} catch (e) {
  report.error = String(e?.message ?? e).slice(0, 500)
  log(`测量失败：${report.error}`)
} finally {
  try { await worker.stop() } catch (e) { log(`stop() 异常（忽略，容器可能已退出）：${e.message}`) }
}

// ── 对比 serving 已验收数字
const base = JSON.parse(readFileSync(BASELINE, 'utf8'))
const sv = base.serving
const cbPer = report.measurements.containerBase?.perText
const csPer = report.measurements.containerSft?.perText
if (cbPer && csPer) {
  report.aggregationAnalysis = {
    note: '容器 meanNll = total_nll/total_tok（token 加权）；serving meanNll = mean(perText)（逐文本等权）',
    containerBaseMeanNllTokenWeighted: report.measurements.containerBase.meanNllTokenWeighted,
    containerBaseMeanNllUnweighted: mean(cbPer),
    containerSftMeanNllTokenWeighted: report.measurements.containerSft.meanNllTokenWeighted,
    containerSftMeanNllUnweighted: mean(csPer),
    servingBaseMeanNll: sv.base.meanNll,
    servingSftMeanNll: sv.sft.meanNll,
  }
  // 逐文本配对：容器 vs serving（同一权重！seed=20260929 的 adapter）
  const pa = pairedGate(sv.base.perText, cbPer)
  const ps = pairedGate(sv.sft.perText, csPer)
  report.perTextComparison = {
    baseContainerVsServing: { deltaNll: pa.deltaNll, se: pa.se, bound: pa.bound, pearson: pearson(sv.base.perText, cbPer) },
    sftContainerVsServing: { deltaNll: ps.deltaNll, se: ps.se, bound: ps.bound, pearson: pearson(sv.sft.perText, csPer) },
  }
  report.gates = {
    serving: pairedGate(sv.base.perText, sv.sft.perText),
    containerSelfConsistent: pairedGate(cbPer, csPer),
    mixedSftContainerVsServingBase: pairedGate(sv.base.perText, csPer),
  }
  report.measurements.servingBase = { meanNll: sv.base.meanNll, texts: sv.base.texts, tokens: sv.base.tokens, perText: sv.base.perText }
  report.measurements.servingSft = { meanNll: sv.sft.meanNll, texts: sv.sft.texts, tokens: sv.sft.tokens, perText: sv.sft.perText }

  const tol = 0.1 * report.gates.serving.bound
  const dAggBase = Math.abs(report.aggregationAnalysis.containerBaseMeanNllUnweighted - sv.base.meanNll)
  const dAggSft = Math.abs(report.aggregationAnalysis.containerSftMeanNllUnweighted - sv.sft.meanNll)
  report.criteria = {
    tolerance: tol,
    A_perTextBaseDelta: { value: Math.abs(report.perTextComparison.baseContainerVsServing.deltaNll), pass: Math.abs(report.perTextComparison.baseContainerVsServing.deltaNll) <= tol },
    A_perTextSftDelta: { value: Math.abs(report.perTextComparison.sftContainerVsServing.deltaNll), pass: Math.abs(report.perTextComparison.sftContainerVsServing.deltaNll) <= tol },
    B_gateVerdictMatches: { serving: report.gates.serving.pass, containerSelfConsistent: report.gates.containerSelfConsistent.pass, pass: report.gates.serving.pass === report.gates.containerSelfConsistent.pass },
    C_aggregationExplains: { unweightedBaseDelta: dAggBase, unweightedSftDelta: dAggSft, pass: dAggBase <= tol && dAggSft <= tol },
    D_pearson: { base: report.perTextComparison.baseContainerVsServing.pearson, sft: report.perTextComparison.sftContainerVsServing.pearson },
  }
  report.verdict = (report.criteria.A_perTextBaseDelta.pass && report.criteria.A_perTextSftDelta.pass && report.criteria.B_gateVerdictMatches.pass)
    ? 'PROXY_OK'
    : (report.criteria.C_aggregationExplains.pass ? 'AGGREGATION_ONLY_DIFF' : 'PROXY_REJECTED')
}
report.wallMs = Date.now() - t0
flush()

console.log('── 校准结果 ──')
const A = report.aggregationAnalysis
console.log(`容器 base : token加权 ${A.containerBaseMeanNllTokenWeighted.toFixed(6)}  逐文本等权 ${A.containerBaseMeanNllUnweighted.toFixed(6)}   serving ${A.servingBaseMeanNll.toFixed(6)}`)
console.log(`容器 sft  : token加权 ${A.containerSftMeanNllTokenWeighted.toFixed(6)}  逐文本等权 ${A.containerSftMeanNllUnweighted.toFixed(6)}   serving ${A.servingSftMeanNll.toFixed(6)}`)
console.log(`逐文本配对(容器 vs serving): base Δ=${report.perTextComparison.baseContainerVsServing.deltaNll.toExponential(2)} r=${report.perTextComparison.baseContainerVsServing.pearson.toFixed(4)} | sft Δ=${report.perTextComparison.sftContainerVsServing.deltaNll.toExponential(2)} r=${report.perTextComparison.sftContainerVsServing.pearson.toFixed(4)}`)
console.log(`门禁: serving pass=${report.gates.serving.pass} (Δ=${report.gates.serving.deltaNll.toFixed(6)} bound=${report.gates.serving.bound.toFixed(6)}) | 容器自洽 pass=${report.gates.containerSelfConsistent.pass} (Δ=${report.gates.containerSelfConsistent.deltaNll.toFixed(6)} bound=${report.gates.containerSelfConsistent.bound.toFixed(6)})`)
console.log(`verdict: ${report.verdict}   wall=${(report.wallMs / 1000).toFixed(1)}s`)
console.log(`report -> ${OUT}`)
