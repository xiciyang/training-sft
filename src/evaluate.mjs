/**
 * evaluate —— training-sft 的**评估适配层**（任务 A：合并四份 evaluate.mjs 之后）。
 *
 * ## 边界（**不绕开共享库契约**）
 * 合并前本文件是一份 351 行的**实现**，与 training-pretrain / training-rl / training-qat
 * 的同名文件是**同一算法、四个所有者**。现在：
 *
 *   - 端点原语与通用评估（`post` / `get` / `evalLM` / `evalMMLU` / `evalUniform` /
 *     `measurementCompleteness` / `attachIntegrity` / `forgettingGate`）
 *     **只有一份实现**：`@xiciyang/testing-core`（`src/evaluate.mjs`）。
 *     本文件只做**再导出**（`export … from`），不含任何实现体。
 *   - 本包**特有**的东西留在这里：提示词模板、协议版本、格式合规/F1 抽取、
 *     任务门禁 `taskGate`（用 testing-core 的 `dualGate`/`relativeChange` 判）。
 *
 * 判据（`dualGate` / `relativeChange` / `uniformSignature` / 困惑度）**全部**来自
 * `@xiciyang/testing-core` —— 本文件不重写任何一条。
 *
 * 两条**可推导**的遗忘判据、缺陷 14 的记账、缺陷 15 的 fail-closed 完整性判定，
 * 都随实现一起搬到了 `@xiciyang/testing-core/src/evaluate.mjs`；语义**一行未改**。
 */
import { post, dualGate, relativeChange, mean } from '@xiciyang/testing-core'

/* ------------------------------------------------------- 通用评估（唯一实现在 testing-core） */

export {
  post, get, evalLM, evalMMLU, evalUniform,
  measurementCompleteness, attachIntegrity, forgettingGate,
} from '@xiciyang/testing-core'

// 供调用方/测试直接取判据（合并前 sft 版本没有这几项再导出，属**追加**，不删改任何既有导出）
export { dualGate, relativeChange, perplexity, mean, stddev, uniformSignature, meanNllFromLogprobs } from '@xiciyang/testing-core'

/* ------------------------------------------------------------------ 本包特有 */

/** SFT 提示词（**与训练侧必须逐字一致**，否则格式合规率测的不是同一个东西）。 */
export const PROMPT_TMPL = (r) => '### Instruction:\n' + String(r.instruction ?? '').trim() + '\n\n'
  + (String(r.input ?? '').trim() ? '### Input:\n' + String(r.input).trim() + '\n\n' : '')
  + '### Response:\n'

/**
 * 评估协议版本 —— **进基线缓存键**：协议一变，缓存自动失效。
 * 为什么要有它：2026-09-29 第一次跑垂直闭环时，`max_tokens=96` 让 97% 的生成被截断，
 * 于是"格式合规率"在基座上就已经是 0.98（= "它至少说了点什么"），**完全不具区分度** ——
 * 任何训练结果都不可能让这个判据提升。那是**测量不成立**，不是"没达标"。
 * 修法不是调宽容度，而是让判据可区分：**要求自然终止**（`finish_reason === 'stop'`）。
 */
export const EVAL_PROTOCOL = 'task-eval@2'
export const DEFAULT_MAX_TOKENS = 256

/**
 * 格式合规率（**无阈值、无魔数**的判据）：
 *   ① 生成内容去空白后非空；
 *   ② 不含 `### Instruction:` / `### Input:`（没有把模板重新起头）；
 *   ③ 至少含一个字母/CJK 字符（不是纯符号/纯空白）；
 *   ④ 若给了 `finish`，则必须**自然终止**（`stop`）—— 被 `length` 截断说明它没有把答案说完，
 *      在 instruction 数据上这不是"格式合规"。（不给 finish 时该条不参与，保持单元测试可单测。）
 */
export function isFormatCompliant(text, finish = null) {
  const t = String(text ?? '')
  if (t.trim() === '') return false
  if (t.includes('### Instruction:') || t.includes('### Input:')) return false
  if (!/[A-Za-z\u4e00-\u9fff]/.test(t)) return false
  if (finish !== null && finish !== undefined && finish !== 'stop') return false
  return true
}

const tok = (s) => String(s ?? '').toLowerCase().match(/[a-z0-9\u4e00-\u9fff]+/g) ?? []

/** token 级 F1（SQuAD 口径）；reference 为空 → 返回 null（不参与平均，也不冒充 0）。 */
export function tokenF1(pred, ref) {
  const p = tok(pred); const r = tok(ref)
  if (r.length === 0) return null
  if (p.length === 0) return 0
  const cnt = new Map()
  for (const x of r) cnt.set(x, (cnt.get(x) || 0) + 1)
  let common = 0
  for (const x of p) { const c = cnt.get(x) || 0; if (c > 0) { common += 1; cnt.set(x, c - 1) } }
  if (common === 0) return 0
  const prec = common / p.length; const rec = common / r.length
  return (2 * prec * rec) / (prec + rec)
}

/** 跑一次生成式评估（heldout 任务指标：格式合规率 + token F1）。 */
export async function evalTask({ endpoint, model, rows, limit = 100, maxTokens = DEFAULT_MAX_TOKENS, log = () => {} }) {
  const out = []
  for (const r of rows.slice(0, limit)) {
    // eslint-disable-next-line no-await-in-loop
    const res = await post(endpoint, '/v1/completions', {
      model, prompt: PROMPT_TMPL(r), max_tokens: maxTokens, temperature: 0, seed: 0,
    }, { timeoutMs: 180000 })
    const text = res.json?.choices?.[0]?.text ?? ''
    const finish = res.json?.choices?.[0]?.finish_reason ?? null
    out.push({ compliant: isFormatCompliant(text, finish), f1: tokenF1(text, r.output), finish, status: res.status, len: text.length })
    if (out.length % 25 === 0) log(`  task ${out.length}/${Math.min(limit, rows.length)}`)
  }
  const ok = out.filter((x) => x.status === 200)
  const f1s = ok.map((x) => x.f1).filter((v) => v !== null)
  // **缺陷 14 记账**（与 pretrain 的 `evalLM`/`evalMMLU` 同构）：把"发出去多少 / 成功多少 /
  // 失败多少"落进产物。没有它，"端点中途失效（failed 高）"与"评估集本来就小"**不可区分**。
  // `okIndices` 供**按下标交集**配对（缺它就只能按位置配，一侧跳行会让其后每对都错位）。
  const okIndices = out.map((x, i) => (x.status === 200 ? i : -1)).filter((i) => i >= 0)
  return {
    protocol: EVAL_PROTOCOL,
    maxTokens,
    n: out.length,
    nOk: ok.length,
    // ---- 完整性证据（缺陷 14）----
    attempted: out.length,          // 真正发出去的请求数
    succeeded: ok.length,           // 取到可用响应的条数
    failed: out.length - ok.length, // **仪器层**失败（非 200）
    skipped: 0,
    partial: out.length > 0 && ok.length < out.length,
    okIndices,
    stopped: ok.filter((x) => x.finish === 'stop').length,
    formatCompliance: ok.length ? ok.filter((x) => x.compliant).length / ok.length : null,
    f1: f1s.length ? mean(f1s) : null,
    f1n: f1s.length,
    truncated: ok.filter((x) => x.finish === 'length').length,
    samples: out.slice(0, 5),
  }
}

/* ------------------------------------------------------------------ 门禁 */

/**
 * 任务门禁：用 `testing-core.dualGate` 判。
 * - validity：测量是否成立（token 数、端点成功率、均匀签名）
 * - 数值：格式合规率提升 + token F1 提升
 */
export function taskGate({ base, sft, thresholds, validity }) {
  const fGain = relativeChange({ baseline: base.formatCompliance, candidate: sft.formatCompliance, direction: 'higher-is-better' })
  const f1Gain = relativeChange({ baseline: base.f1, candidate: sft.f1, direction: 'higher-is-better' })
  const g = dualGate({
    validity,
    // 符号约定：两项都是 higher-is-better，故「劣化量」= worseningPct，
    // 「提升量」= −worseningPct。写反会让"大幅提升"被读成"大幅下降"（曾如此）。
    accuracy: { dropPct: f1Gain.worseningPct },
    perf: { gainPct: -fGain.worseningPct },
    maxAccuracyDropPct: thresholds.maxF1DropPct,
    minPerfGainPct: thresholds.minFormatGainPct,
  })
  return { ...g, deltas: { formatCompliance: { base: base.formatCompliance, sft: sft.formatCompliance, rawPct: fGain.rawPct }, f1: { base: base.f1, sft: sft.f1, rawPct: f1Gain.rawPct } } }
}
