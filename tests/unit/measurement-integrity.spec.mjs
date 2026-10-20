/**
 * measurement-integrity.spec —— **缺陷 14 回归测试**：测量完整性必须可判定。
 *
 * 背景（长任务 5 阶段 ⑥ 第 1 轮实测）：base 侧 `evalLM` 只拿到 5/100 条，
 * 而产物上 `texts=5` **与"评估集本来只有 5 条"不可区分** ⇒ 端点中途失效可以伪装成小样本。
 *
 * 本 spec 断言四件事（全部不依赖 GPU / 容器 / 网络）：
 *   ① **端点失效**（failed 高）必须被判为"测量不完整"，而不是"样本不足"；
 *   ② **评估集小**（attempted 本身就少、failed=0）必须被判为"样本不足"；
 *   ③ 两者**理由不同** —— 即产物能区分病因；
 *   ④ 配对必须按**成功行原始下标的交集**，不是按数组位置（否则一侧跳行会让其后每对都错位）。
 *
 * ══════════════════════════════════════════════════════════════════════════════
 * ⚠️ 缺陷 P1 的修复（缺陷回放 D06 实测到的**检测器空洞**）
 * ══════════════════════════════════════════════════════════════════════════════
 * 上面 ①②③ 全部喂的是**我们自己合成的** `failed/attempted` 夹具。于是当共享实现的
 * `evalLM` 停止记账（D06：把 `attempted/succeeded/failed` 从返回里删掉）时，
 * **本 spec 依然全绿** —— 因为合成的失败对象是"我们"构造的，而不是 `post()` 在真实
 * 连接失败下产出的。回放原始结果：`sft-tests` 在 D06 下 `exit=0`（绿），只有 `qat-tests` 变红。
 *
 * 处置：**把真实失败路径拉进来** —— 把 `evalLM` 指向一个确定连不上的端点
 * （`http://127.0.0.1:9`，discard 端口，本机未监听 ⇒ 连接立即被拒，无网络依赖），
 * 断言它**如实记账**，并把**真实结果**喂给 `forgettingGate` 断言它判"测量不完整"。
 * 这样"记账被删掉"就不再能伪装成通过（§⑤⑥⑦）。
 */
import { forgettingGate, evalLM } from '../../src/evaluate.mjs'

export const NAME = '测量完整性（缺陷 14 回归）'

/** discard 端口：本机未监听 ⇒ 连接立即被拒（不依赖外部网络，也不需要 GPU）。 */
const DEAD = 'http://127.0.0.1:9'
const mkRows = (n) => Array.from({ length: n }, (_, i) => ({ t: `row ${i} 这是一段用于触发前向的文本` }))

const lmOk = (n, { offset = 0, failed = 0, attempted = null, val = 3 } = {}) => ({
  attempted: attempted ?? n + failed,
  succeeded: n, failed, skipped: 0,
  partial: failed > 0,
  okIndices: Array.from({ length: n }, (_, i) => offset + i),
  perText: Array.from({ length: n }, () => val),
  meanNll: val, ppl: Math.exp(val), texts: n,
})
const mmluOk = (n, acc = 0.4) => ({ attempted: n, succeeded: n, failed: 0, n, accuracy: acc, correct: Math.round(n * acc), okIndices: [] })

export async function run(t) {
  // ── ① 端点失效：base 只成功 5/100 ──
  const a = forgettingGate({
    baseMmlu: mmluOk(100), sftMmlu: mmluOk(100),
    baseLm: lmOk(5, { failed: 95 }), sftLm: lmOk(100),
  })
  t.assert(a.pass === false, '端点失效时门禁必须**不通过**')
  const aText = a.reasons.join(' | ')
  t.assert(/测量不完整/.test(aText), `端点失效必须被报成"测量不完整"，实得：${aText}`)
  t.assert(/failed=95/.test(aText), '必须报出 failed 的具体数量（否则无法定位）')
  t.assert(!/配对样本不足/.test(aText), '端点失效**不得**被误报成"配对样本不足"（那是评估集小的说法）')

  // ── ② 评估集小：两侧都只有 10 条，failed=0 ──
  const b = forgettingGate({
    baseMmlu: mmluOk(100), sftMmlu: mmluOk(100),
    baseLm: lmOk(10), sftLm: lmOk(10),
  })
  t.assert(b.pass === false, '评估集小时门禁必须**不通过**')
  const bText = b.reasons.join(' | ')
  t.assert(/配对样本不足/.test(bText), `评估集小必须报"配对样本不足"，实得：${bText}`)
  t.assert(!/测量不完整/.test(bText), '评估集小**不得**被误报成"测量不完整"')

  // ── ③ 可区分：同一批数据下两种病因给出**不同的理由** ──
  t.assert(aText !== bText, '两种病因的理由必须不同（否则产物无法区分病因）')

  // ── ④ 配对按成功行**下标交集**，不是按位置 ──
  // base 成功于下标 0..59，sft 成功于下标 10..69 ⇒ 交集 = 10..59 = **50** 条。
  // 旧的按位置配对会得到 min(60,60)=60 —— 且其后每一对都错位。
  const c = forgettingGate({
    baseMmlu: mmluOk(100), sftMmlu: mmluOk(100),
    baseLm: lmOk(60, { offset: 0 }), sftLm: lmOk(60, { offset: 10 }),
  })
  t.assert(c.lm !== null, '交集 50 ≥ 30，LM 判据应当成立')
  t.assert(c.lm?.n === 50, `配对样本数必须等于下标交集 50，实得 ${c.lm?.n}（按位置配对会得到 60）`)
  t.assert(c.lm?.pairedOn === 'okIndices-intersection', '必须标明配对依据是下标交集')

  // ── ⑤ 完全无交集 ⇒ 不得判定为通过（配对为 0 必须 fail-closed） ──
  const d = forgettingGate({
    baseMmlu: mmluOk(100), sftMmlu: mmluOk(100),
    baseLm: lmOk(60, { offset: 0 }), sftLm: lmOk(60, { offset: 1000 }),
  })
  t.assert(d.pass === false, '两侧成功行完全不相交时必须**不通过**')
  t.assert(d.lm === null, '无交集时 lm 必须为 null（不得用 0 样本算出"通过"）')

  // ── ⑥ 向后兼容：旧产物没有 okIndices 时退化为按位置并**标明**，不得静默 ──
  const legacyBase = { perText: Array.from({ length: 40 }, () => 3), meanNll: 3, ppl: Math.exp(3) }
  const legacySft = { perText: Array.from({ length: 40 }, () => 3), meanNll: 3, ppl: Math.exp(3) }
  const e = forgettingGate({ baseMmlu: mmluOk(100), sftMmlu: mmluOk(100), baseLm: legacyBase, sftLm: legacySft })
  t.assert(e.lm?.pairedOn === 'positional-legacy', '无 okIndices 时必须标明退化为 positional-legacy')

  // ── ⑦ 完整性信息必须进入判定产物（不只是返回值） ──
  t.assert(c.lm?.attempted && typeof c.lm.attempted.base === 'number', 'lm 判据必须携带 attempted（门禁基于它判）')
  t.assert(c.lm?.failed && typeof c.lm.failed.base === 'number', 'lm 判据必须携带 failed')
  t.assert(c.mmlu?.attempted && typeof c.mmlu.attempted.base === 'number', 'mmlu 判据必须携带 attempted')

  /* ══════════════════════════════════════════════════════════════════════════
   * ⑤⑥⑦ **真实失败路径**（P1 修复）—— 端点真的连不上时，`evalLM` 必须如实记账，
   *       且下游门禁必须据此判"测量不完整"。全部不依赖 GPU / 容器 / 外部网络。
   * ══════════════════════════════════════════════════════════════════════════ */

  // ⑤ `evalLM` 在真实 ECONNREFUSED 下必须记账（不是返回一个"texts=0"了事）
  const real = await evalLM({ endpoint: DEAD, model: 'none', rows: mkRows(5), limit: 5 })
  t.assert(real.attempted === 5, `真实失败：attempted=5（真发出去 5 次），实得 ${real.attempted}`)
  t.assert(real.succeeded === 0, `真实失败：succeeded=0，实得 ${real.succeeded}`)
  t.assert(real.failed === 5, `真实失败：failed=5，实得 ${real.failed}`)
  t.assert(real.partial === true, '真实失败：partial=true（success < attempted）')
  t.assert(real.texts === 0, `真实失败：texts（向后兼容字段）=0，实得 ${real.texts}`)
  t.assert(real.considered === 5, `真实失败：considered 仍反映评估集大小 5，实得 ${real.considered}`)
  t.assert(real.firstError !== null && real.firstError !== undefined, '真实失败：firstError 必须存在（否则无法定位）')
  t.assert(real.firstError?.index === 0, `真实失败：firstError.index=0，实得 ${real.firstError?.index}`)
  t.assert(typeof real.firstError?.status === 'number', '真实失败：firstError.status 是数字（连接失败时为 0）')
  t.assert(typeof real.firstError?.detail === 'string' && real.firstError.detail.length > 0, '真实失败：firstError.detail 非空')
  t.assert(Array.isArray(real.okIndices) && real.okIndices.length === real.perText.length, '真实失败：okIndices 与 perText 等长（配对依据不脱节）')

  // ⑥ **真实结果**喂给门禁 ⇒ 必须判"测量不完整"，不得判成"配对样本不足"、更不得判通过
  const realGate = forgettingGate({
    baseMmlu: mmluOk(100), sftMmlu: mmluOk(100),
    baseLm: lmOk(100), sftLm: real,
  })
  t.assert(realGate.pass === false, '真实端点失效 ⇒ 门禁必须**不通过**（不得静默放行）')
  const realText = realGate.reasons.join(' | ')
  t.assert(/测量不完整/.test(realText), `真实端点失效必须报"测量不完整"，实得：${realText}`)
  t.assert(/failed=5/.test(realText), `必须报出真实失败的条数（failed=5），实得：${realText}`)
  t.assert(!/配对样本不足/.test(realText), '真实端点失效**不得**被误报成"评估集小"（那是另一种病因）')
  t.assert(realGate.lm === null, '测量不完整时 lm 判据必须为 null（不得用 0 样本算出一个结论）')
  t.assert(realGate.mmlu !== null, '对照：同一批里 MMLU 侧是完整的 ⇒ mmlu 判据照常产出（不是"整体拒判"，而是**点名坏了哪一侧**）')

  // ⑦ 静默吞掉记账即**变红**（D06 的注入形态：`evalLM` 不再返回 attempted/succeeded/failed）
  //    —— 这一条把"合成夹具"与"真实路径"绑定：真实路径一旦不记账，上面 ⑤⑥ 立刻失败。
  const integrityKeys = ['attempted', 'succeeded', 'failed', 'partial', 'considered', 'firstError']
  const missingKeys = integrityKeys.filter((k) => !(k in real))
  t.assert(missingKeys.length === 0,
    `真实失败结果必须携带完整记账（缺陷 14 族：${integrityKeys.join('/')}）；缺：${missingKeys.join(', ') || '无'}`)
  t.assert(typeof real.attempted === 'number' && typeof real.failed === 'number',
    'attempted/failed 必须是**数字**（非数字 ⇒ 门禁的门槛比较会静默退化）')
}
