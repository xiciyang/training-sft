/**
 * weight-fingerprint.spec —— **权重指纹守卫**回归（缺陷：SFT 此前完全没有"空操作训练"防线）。
 *
 * ## 背景
 * 本仓库的原始事故：`lr=0` 让 500 步全是空操作，而**步号 / consumedChunks / 不变量全部为真**。
 * 结论固化成原则：**"只断言过程抓不到空操作，必须断言权重"**。
 * `training-pretrain` 因此有 `fingerprint`/`initFingerprint`/`weightsChanged`；
 * **`training-sft` 一直没有**（CI `fingerprint-coverage.mjs` 实测判 FAIL：payload 键只有
 * `adapterDir,bytes,epoch,files,step`）⇒ 这条原则**只固化在了 pretrain 一处**。
 *
 * ## 本 spec 断言什么
 *   ① **判据自检**：必须**真的看到了被测对象** —— 用**线上真实 payload**（缺口期产物）
 *      作为"必须被拒"的用例。若守卫是空转的，这一条最先红。
 *   ② 指纹缺失 ⇒ 抛 `E_WEIGHT_FINGERPRINT_MISSING`（"无法断言" ≠ "没问题"）
 *   ③ 指纹相同 ⇒ 抛 `E_NO_OP_TRAINING`（**反向对照**：空操作绝不能被记成成功）
 *   ④ 指纹不同 ⇒ 通过，且回传 before/after
 *   ⑤ 两种病因的**错误码必须可区分**（否则无法定位）
 *
 * 纯函数 + 一个只读文件读取，不依赖 GPU / 容器 / 网络。
 */
import { readFileSync, existsSync } from 'node:fs'
import { join } from 'node:path'
import { homedir } from 'node:os'
import { assertWeightFingerprint } from '../../src/core.mjs'

export const NAME = '权重指纹守卫（空操作训练防线·sft）'

/** 缺口期的**真实产物**（第二轮 cycB 的 checkpoint），冻结在磁盘上，代表"没有指纹"的样子。 */
const GAP_ERA_PAYLOAD = join(homedir(), '.ai-infra/run/training-sft/ts_20261001153227_j7d9/checkpoints/step-000200/payload.json')

const expectThrow = (fn, codeOrText) => {
  try { fn(); return { ok: false, detail: '未抛出（期望抛）' } }
  catch (e) {
    const m = String(e?.message ?? '')
    return m.includes(codeOrText) ? { ok: true, detail: codeOrText } : { ok: false, detail: `错误码不符：期望含 ${codeOrText}，实得 ${m.slice(0, 120)}` }
  }
}

export async function run(t) {
  t.assert(typeof assertWeightFingerprint === 'function', '守卫必须可被 import —— 拿不到就是在测空气')

  /* ── ① 判据自检：用**线上真实 payload**（产物已被清理时显式跳过，不伪装成通过） ── */
  if (!existsSync(GAP_ERA_PAYLOAD)) {
    t.skip(`缺口期产物已随 run 清理消失（${GAP_ERA_PAYLOAD} 不存在）—— 判据自检这一条没有对象可看；守卫行为由 ②–⑤ 的合成用例继续覆盖`)
  } else {
    const real = JSON.parse(readFileSync(GAP_ERA_PAYLOAD, 'utf8'))
    t.assert(
      !Object.prototype.hasOwnProperty.call(real, 'fingerprint'),
      `缺口期真实 payload **确实没有** fingerprint（实得键：${Object.keys(real).sort().join(',')}）—— 这是要守的那个洞`,
    )
    const r1 = expectThrow(() => assertWeightFingerprint(real), 'E_WEIGHT_FINGERPRINT_MISSING')
    t.assert(r1.ok, `① 真实缺口期 payload 必须被守卫拒绝：${r1.detail}`)
  }

  /* ── ② 指纹缺失 ── */
  for (const bad of [
    { files: [] },
    { fingerprint: 'sha256:aa', files: [] },
    { initFingerprint: 'sha256:bb', files: [] },
    { fingerprint: '', initFingerprint: 'sha256:bb' },
    { fingerprint: null, initFingerprint: null },
  ]) {
    const r = expectThrow(() => assertWeightFingerprint(bad), 'E_WEIGHT_FINGERPRINT_MISSING')
    t.assert(r.ok, `② 缺指纹必须拒绝（${JSON.stringify(bad)}）：${r.detail}`)
  }

  /* ── ③ 反向对照：指纹相同 ⇒ 空操作，必须拒绝 ── */
  const same = { fingerprint: 'sha256:deadbeef', initFingerprint: 'sha256:deadbeef' }
  const r3 = expectThrow(() => assertWeightFingerprint(same), 'E_NO_OP_TRAINING')
  t.assert(r3.ok, `③ 指纹相同必须判空操作：${r3.detail}`)

  /* ── ④ 指纹不同 ⇒ 通过，且回传 before/after ── */
  const good = { fingerprint: 'sha256:1111', initFingerprint: 'sha256:2222' }
  let out = null
  try { out = assertWeightFingerprint(good) } catch (e) { out = { error: String(e.message) } }
  t.assert(out?.ok === true, `④ 指纹不同必须通过（实得 ${JSON.stringify(out)}）`)
  t.assert(out?.before === 'sha256:2222' && out?.after === 'sha256:1111', '④ 必须回传 before/after 供审计')

  /* ── ⑤ 两种病因的码必须可区分 ── */
  const missingMsg = (() => { try { assertWeightFingerprint({}) } catch (e) { return String(e.message) } return '' })()
  const noopMsg = (() => { try { assertWeightFingerprint(same) } catch (e) { return String(e.message) } return '' })()
  t.assert(missingMsg.includes('E_WEIGHT_FINGERPRINT_MISSING'), '⑤ 缺失的码')
  t.assert(noopMsg.includes('E_NO_OP_TRAINING'), '⑤ 空操作的码')
  t.assert(missingMsg !== noopMsg, '⑤ 两种病因必须给出**不同**的码与理由（否则无法定位）')
}
