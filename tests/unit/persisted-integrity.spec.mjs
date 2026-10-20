/**
 * persisted-integrity.spec —— **缺陷 15 的同类残留**（sft 侧）：*测量能被重算，但不能自证*。
 *
 * 与 pretrain 侧同源（见 `training-pretrain/tests/unit/persisted-integrity.spec.mjs`）：
 * `summarize()` 把 `attempted/succeeded/failed/partial` **从持久化摘要里剥掉**，
 * 于是落盘之后 `measurementCompleteness` **恒为 false**，
 * "端点中途失效（failed 高）"与"评估集本来就小"在**产物层面重新变得不可区分** ——
 * 那正是缺陷 14 被判为缺陷的理由，只是换了一层。
 *
 * 实测（修复前，`~/.ai-infra/state/sft-registry.yaml`）：字符串 `attempted` 出现 **0 次**。
 *
 * ## 本 spec 断言什么
 *   ① 完备测量的 `summarize()` 产物上 `measurementCompleteness` **必须 complete**（**自证**，不是"看起来完整"）；
 *   ② 每个**逐条网络测量**的部件（`lm` / `mmlu`）都逐字段带上完整性证据；
 *   ③ **反向对照（防"把失败改绿"）**：不完整测量必须仍 `complete=false` 且理由**点名** `failed=N/M`；
 *   ④ `integrity.producedBy` 在场且 **不是** `effectJudgment`（不自证）。
 *
 * `task` 属于"已知但非逐条网络测量"的部件（确定性打分），其原始测量**没有** attempted/failed，
 * 故本 spec **不**要求它带完整性证据 —— 但会断言它**没有**被误判为 `unclassified-part`。
 *
 * 不依赖 GPU / 容器 / 网络。
 */
import { summarize } from '../../src/core.mjs'
import { measurementCompleteness } from '../../src/evaluate.mjs'

export const NAME = '持久化测量的完整性证据（缺陷 15 同类残留·sft）'

const fullMeasurement = () => ({
  uniform: { ok: true, n: 12, epsilon: 0.01, delta: 0.02, bitIdenticalAcrossRuns: true },
  // `task` 自 v0.3.0 起属于 INSTRUMENTED ⇒ 夹具带完整性证据（与 `evalTask` 的新形状一致）
  task: { attempted: 100, succeeded: 100, failed: 0, partial: false, n: 100, nOk: 100, formatCompliance: 0.31, f1: 0.3578, f1n: 100, truncated: 0 },
  lm: {
    attempted: 200, succeeded: 200, failed: 0, skipped: 0, partial: false,
    texts: 200, tokens: 5000, meanNll: 3.25, ppl: 25.8, perText: [3.2, 3.3],
  },
  mmlu: {
    attempted: 300, succeeded: 300, failed: 0, skipped: 0, partial: false,
    n: 300, correct: 120, accuracy: 0.4, asked: 300, okIndices: [],
  },
})

const partialMeasurement = () => {
  const m = fullMeasurement()
  m.lm = { ...m.lm, attempted: 260, succeeded: 200, failed: 60, partial: true }
  return m
}

export async function run(t) {
  t.assert(typeof summarize === 'function', 'summarize 必须可被 import 到 —— 拿不到就是在测空气')
  const s = summarize(fullMeasurement())
  t.assert(s !== null && typeof s === 'object', 'summarize 必须返回对象')

  // ── ① 持久化摘要必须**自证**完整 ──
  const chk = measurementCompleteness(s)
  t.assert(
    chk.complete === true,
    `完备测量经 summarize() 后必须 complete（实得 ${JSON.stringify(chk)}）`
    + ' —— 非 complete 即"落盘后无法自证完整"，缺陷 15 的同类残留仍在',
  )
  t.assert(chk.reason === null, `complete 时 reason 必须为 null（实得 ${JSON.stringify(chk.reason)}）`)

  // ── ② 逐仪器、逐字段 ──
  for (const part of ['lm', 'mmlu']) {
    const p = s[part]
    const src = fullMeasurement()[part]
    t.assert(p !== undefined && p !== null, `summarize 产物必须保留仪器 ${part}`)
    for (const f of ['attempted', 'succeeded', 'failed']) {
      t.assert(typeof p[f] === 'number', `${part}.${f} 必须是数（实得 ${typeof p[f]}）`)
      t.assert(p[f] === src[f], `${part}.${f} 必须**原样保留**（期望 ${src[f]}，实得 ${p[f]}）`)
    }
    t.assert(typeof p.partial === 'boolean', `${part}.partial 必须是布尔（实得 ${typeof p.partial}）`)
  }

  // ── ②b `task` 不得被误判为未归类部件（它是已知的非逐条网络测量） ──
  t.assert(
    !String(chk.reason ?? '').includes('unclassified-part:task'),
    `task 是已知部件，不得报 unclassified-part（实得 ${JSON.stringify(chk.reason)}）`,
  )

  // ── ③ 反向对照：不完整测量不得被判为完整 ──
  const chkBad = measurementCompleteness(summarize(partialMeasurement()))
  t.assert(
    chkBad.complete === false,
    `不完整测量经 summarize() 后必须仍 complete=false（实得 ${JSON.stringify(chkBad)}）`
    + ' —— 若为 true，说明修复退化成"无条件宣称完整"，把失败改绿了',
  )
  t.assert(
    typeof chkBad.reason === 'string' && /failed=60\/260/.test(chkBad.reason),
    `理由必须**点名** failed=60/260（实得 ${JSON.stringify(chkBad.reason)}）`,
  )

  // ── ④ 不自证 ──
  t.assert(s.integrity !== undefined && s.integrity !== null, 'summarize 产物必须带 integrity 块')
  t.assert(
    typeof s.integrity?.producedBy === 'string' && s.integrity.producedBy !== '',
    `integrity.producedBy 必须是非空字符串（实得 ${JSON.stringify(s.integrity?.producedBy)}）`,
  )
  t.assert(s.integrity.producedBy !== 'effectJudgment', 'integrity.producedBy 不得是 effectJudgment（不自证）')
}
