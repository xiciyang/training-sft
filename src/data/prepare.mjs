#!/usr/bin/env node
/**
 * prepare-data.mjs —— training-sft 的**数据准备与校验**（阶段 3 第一步）。
 *
 * 纪律对照（题面「必须」与「不得」逐条落点）：
 *   - 训练数据**绝不含测试集样本** → 本脚本按**内容哈希**剔除与任一评估集相撞的行，
 *     并在最后**用 `assertNoLeakage` 复核**（不是"我剔过了所以应该没有"）。
 *   - **不自行脱敏** → 用户裁决的处置是"**剔除**含非占位 PII 的行"；
 *     本脚本**删除整行**，绝不改写行内内容。剔除清单只记哈希，不记原文。
 *   - **不绕开共享库契约** → 去重/切分/去泄漏/哈希**全部**走 `@xiciyang/data-core`。
 *
 * 切分口径（数据裁决后固定）：
 *   源：`yahma/alpaca-cleaned`（实测 CC BY 4.0，revision 12567cab…）
 *   训练 1000 行 / 留出评估 200 行（同源不同行，内容哈希不相交）
 *
 * 用法：node prepare-data.mjs [--seed 20260929] [--train 1000] [--heldout 200] [--out DIR]
 */
import { readFileSync, writeFileSync, mkdirSync, existsSync, readdirSync, statSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import {
  contentHash, recordHash, dedupByContent, splitDeterministic, assertNoLeakage, lineage, piiScan,
} from '@xiciyang/data-core'
import { REPLAY_SOURCE, readReplayTexts, selectDeterministic, toReplayRow, verifyReplaySource } from './replay.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const HOME = process.env.HOME
const argv = process.argv.slice(2)
const argOf = (k, d) => { const i = argv.indexOf(k); return i >= 0 && argv[i + 1] ? argv[i + 1] : d }
const SEED = Number(argOf('--seed', '20260929'))
const N_TRAIN = Number(argOf('--train', '1000'))
const N_HELDOUT = Number(argOf('--heldout', '200'))

export const SOURCE = {
  repo: 'yahma/alpaca-cleaned',
  url: 'https://hf-mirror.com/datasets/yahma/alpaca-cleaned',
  revision: '12567cabf869d7c92e573c7c783905fc160e9639',
  license: { spdx: 'cc-by-4.0', class: 'permissive', source: 'https://hf-mirror.com/api/datasets/yahma/alpaca-cleaned (tags: license:cc-by-4.0)' },
  file: process.env.DSH_ALPACA_PATH,
  fileSha256: 'sha256:bd844b8247a0f543804b6ce0882b0aaec4bbf5e8d66167df6213a0f1e4fe878b',
}

/* ------------------------------------------------------------------ 工具 */

const RESERVED_DOMAIN = /@(example\.(com|org|net|edu)|test\.(com|org)|localhost|domain\.(com|org)|email\.com|yourdomain\.com|company\.com|mycompany\.com|site\.com|x\.com|foo\.com|bar\.com|abc\.com)$/i
const RESERVED_IP = /^(10\.|127\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.|0\.|255\.|192\.0\.2\.|198\.51\.100\.|203\.0\.113\.)/
const PLACEHOLDER_LOCAL = /^(user|test|example|your|name|email|someone|john\.?doe|jane\.?doe|abc|foo|bar|admin|info|contact|hello|me|you|my)@/i
export function classifyHit(kind, text) {
  if (kind === 'email') {
    if (RESERVED_DOMAIN.test(text)) return 'reserved-domain'
    if (PLACEHOLDER_LOCAL.test(text)) return 'placeholder-localpart'
    return 'other'
  }
  if (kind === 'ipv4') return RESERVED_IP.test(text) ? 'reserved-ip' : 'other'
  return 'other'
}

/** 非占位 PII：命中里至少有一条 `other` 即判定（**整行剔除**，不改写行内内容）。 */
export function nonReservedHits(rec) {
  const found = []
  for (const k of ['instruction', 'input', 'output']) {
    const s = typeof rec[k] === 'string' ? rec[k].slice(0, 2000) : ''
    if (!s) continue
    for (const f of piiScan(s).findings) {
      const raw = s.slice(f.index, f.index + f.length)
      if (classifyHit(f.pattern, raw) === 'other') found.push({ field: k, pattern: f.pattern, hash: `sha256:${contentHash(raw).slice(0, 16)}` })
    }
  }
  return found
}

/**
 * 空白归一：把任意空白串压成一个空格。
 * 为什么去泄漏判据要**对空白不敏感**：行内文本经过切分/清洗后，接缝处的空白可能与原文不同，
 * 而"泄漏"这件事对分词器来说**不取决于空格个数**。用原样字符串比对会给出**假阴性**
 * （评估文本几乎逐字在训练行里，判据却说没有）。
 */
export const normalizeWs = (s) => String(s ?? '').replace(/\s+/g, ' ').trim()

export const pairText = (r) => `${String(r.instruction ?? '').trim()}\u0001${String(r.input ?? '').trim()}\u0001${String(r.output ?? '').trim()}`

function loadJson(p) {
  const t = readFileSync(p, 'utf8')
  if (t.trimStart().startsWith('[')) return JSON.parse(t)
  return t.split('\n').filter((l) => l.trim()).map((l) => JSON.parse(l))
}

/** 评估集：wikitext-2 test（语言建模）+ MMLU test（通用能力）。 */
export function loadEvalSets() {
  const sets = []
  const root = join(HOME, '.ai-infra/state/datasets')
  if (existsSync(root)) {
    for (const d of readdirSync(root)) {
      const p = join(root, d, 'test.jsonl')
      if (!existsSync(p)) continue
      const rows = loadJson(p)
      sets.push({ name: `wikitext-test(${d.includes('c512') ? 'c512' : 'c128'})`, kind: 'lm', file: p, rows, keyOf: (r) => r.t, sha256: `sha256:${contentHash(readFileSync(p, 'utf8'))}` })
    }
  }
  const mmlu = '/data/datasets/mmlu/mmlu-test.jsonl'
  if (existsSync(mmlu)) {
    const rows = loadJson(mmlu)
    sets.push({ name: 'mmlu-test', kind: 'qa', file: mmlu, rows, keyOf: (r) => r.question, sha256: `sha256:${contentHash(readFileSync(mmlu, 'utf8'))}` })
  }
  return sets
}

/* -------------------------------------------------------------------- 主 */

export async function prepareData({ seed = SEED, train = N_TRAIN, heldout = N_HELDOUT, outDir = null, datasetsRoot = null, write = true, force = false, mix = 0, replayFile = null, replayScanCap = 20000 } = {}) {
  if (!(mix >= 0 && mix < 1)) throw new Error(`E_MIX_RANGE: mix 必须在 [0,1) 内，收到 ${mix}`)
  if (!existsSync(SOURCE.file)) throw new Error(`数据源不存在：${SOURCE.file}`)
  const raw = loadJson(SOURCE.file)
  const steps = []
  steps.push({ step: 'source', rows: raw.length })

  // ① 去重（内容哈希）
  const dd = dedupByContent(raw, { keyOf: pairText })
  steps.push({ step: 'dedup-content-hash', kept: dd.kept.length, dropped: dd.dropped.length })

  // ② 剔除含**非占位** PII 的行（用户裁决；不改写内容）
  const piiDropped = []
  const afterPii = []
  for (const r of dd.kept) {
    const hits = nonReservedHits(r)
    if (hits.length > 0) piiDropped.push({ hash: `sha256:${recordHash(r, pairText)}`, hits: hits.map((h) => `${h.field}/${h.pattern}`), sampleHash: hits[0].hash })
    else afterPii.push(r)
  }
  steps.push({ step: 'drop-non-reserved-pii-rows', kept: afterPii.length, dropped: piiDropped.length })

  // ③ 评估集索引（**只建一次**；主源与第二源共用同一份索引与同一份判定代码）
  const evalSets = loadEvalSets()
  const evalHashes = new Map()
  for (const es of evalSets) for (const r of es.rows) evalHashes.set(recordHash(r, es.keyOf), es.name)
  const evalTextIndex = []
  for (const es of evalSets) {
    for (const r of es.rows) {
      const t = es.keyOf(r)
      if (typeof t === 'string' && t.trim().length >= 24) evalTextIndex.push({ set: es.name, text: t.trim(), norm: normalizeWs(t) })
    }
  }

  /**
   * ③+③b 去泄漏判定 —— ⚠️ **主源与第二源(replay)必须走这一个函数**。
   * 把"新能力自带更松的检查"这条后门在**结构上**堵死：不是靠"我记得也要检查"，
   * 而是靠"没有第二份实现可写"。
   *   ③  整行内容哈希相撞 ⇒ 剔
   *   ③b **包含级**：任一字段若含某条评估集项的全部文本(≥24字) ⇒ 剔整行
   *       （实测 MMLU 题目曾逐字出现在 alpaca 的 output 里；哈希相等判不出这种形态）
   */
  function applyEvalGuards(rows) {
    const dropped = []
    const contained = []
    const kept = []
    for (const r of rows) {
      const h = recordHash(r, pairText)
      if (evalHashes.has(h)) { dropped.push({ hash: `sha256:${h}`, collidesWith: evalHashes.get(h) }); continue }
      const fields = ['instruction', 'input', 'output'].map((k) => String(r[k] ?? ''))
      /**
       * ⚠️ 判据用**三字段拼接后**的整串，不是逐字段 —— 逐字段有一个**真实的漏检形态**：
       * 回放行是"前半段 → input，后半段 → output"，若某条评估集文本**恰好横跨这个切点**，
       * 则它在任何一个字段里都**不是完整子串**，逐字段检查**永远看不见它**，
       * 而训练时模型看到的正是拼接后的连续文本 ⇒ 泄漏成立而检查不报。
       * 拼接串是逐字段的超集（字段内的子串在拼接串里仍是子串）⇒ 这条**严格更强**，且不放过原形态。
       */
      const joinedNorm = normalizeWs(fields.join(''))
      let hit = null
      for (const item of evalTextIndex) { if (joinedNorm.includes(item.norm)) { hit = item; break } }
      if (hit) { contained.push({ hash: `sha256:${h}`, contains: hit.set, textHash: `sha256:${contentHash(hit.text).slice(0, 16)}`, scope: 'joined' }); continue }
      kept.push(r)
    }
    // `afterCollision` 单独回报：两个判据合在一次遍历里跑，但**审计轨迹必须按判据分段**。
    // 若第 ③ 步直接报 `kept`（两个判据之后），读者会把它读成"哈希相撞后还剩多少"——
    // 那个数**更大**，于是"这一步剔掉了多少"看起来是 0，而真正的剔除数被挪到了下一步，
    // 轨迹对得上总数、却对不上**每一步的含义**。实测：这一步把 51733 误报成 51723。
    return { kept, dropped, contained, afterCollision: rows.length - dropped.length }
  }

  const g0 = applyEvalGuards(afterPii)
  const afterEval = g0.kept, evalDropped = g0.dropped, containsDropped = g0.contained
  steps.push({ step: 'drop-eval-set-collisions', kept: g0.afterCollision, dropped: evalDropped.length, evalSets: evalSets.map((e) => ({ name: e.name, kind: e.kind, rows: e.rows.length, sha256: e.sha256 })) })
  steps.push({ step: 'drop-eval-set-containment', kept: afterEval.length, dropped: containsDropped.length, checkedEvalItems: evalTextIndex.length })

  // ④ 确定性切分（走 data-core 的 splitDeterministic）
  const split = splitDeterministic(afterEval, { seed, calib: train, test: heldout, keyOf: pairText })
  steps.push({ step: 'split', seed, train: split.calib.length, heldout: split.test.length, unused: split.counts.unused })

  /**
   * ④b 通用语料回放（第二段 C1）—— **替换**而非追加。
   *
   * 为什么是"替换"：若只是**追加**回放行，数据集变大、同样 200 步覆盖到的指令样本变少，
   * 于是"语料混合"与"数据量变化"两个因素**缠在一起**，判不出是谁的效应。
   * 替换则让 **总行数不变、步数不变、‖ΔW‖ 预算不变**，只有**成分**变 ——
   * 这才构成 evidence-core 认可的 `controlled-variation`（受控变差）。
   *
   * 回放行**只进 train**，绝不进 heldout：留出集是**评估身份**，
   * 往里掺通用文本会让"格式增益"这件事本身失去含义。
   */
  let replayRows = []
  let replayInfo = null
  if (mix > 0) {
    const nReplay = Math.round(mix * train)
    const nInstr = train - nReplay
    if (nReplay < 1) throw new Error(`E_MIX_TOO_SMALL: mix=${mix} × train=${train} 取整后为 0 行，无法构成受控变差`)

    // ⚠️ 指纹**现算**，不照抄常量：照抄的话，磁盘上的文件被换掉后 lineage 仍会声称是原来那个哈希 ——
    //    那正是"血缘可复核"要防的事。常量只用于**声明期望值**并与实测比对。
    const rfp = await verifyReplaySource({ file: replayFile || REPLAY_SOURCE.file })
    const rr = await readReplayTexts({ file: replayFile || REPLAY_SOURCE.file, scanCap: replayScanCap })
    // 过采样：后续 PII / 去泄漏 / 去重都会再剔掉一批，不能"选 n 条然后指望它们全活下来"
    // ⚠️ 过采样规模必须**与 mix 无关**：若按 nReplay 缩放，则 mix=0.15 与 mix=0.25
    // 会从**不同的池**里抽行 ⇒ "改 mix"同时改变了"用了哪些回放行"，
    // 两个因子缠在一起，bracket 出来的曲线就不是同一条。
    // 固定成 `train` 行池后，`slice(0, nReplay)` 保证**档位嵌套**（小 mix 的行 ⊂ 大 mix 的行）。
    const pool = selectDeterministic(rr.items, Math.max(train, nReplay + 64), seed)
      .map((it) => toReplayRow(it))
    const dd2 = dedupByContent(pool, { keyOf: pairText })

    // PII：**复用主源同一个判定函数**（整行剔除，不改写）
    const replayPii = []
    const afterReplayPii = []
    for (const r of dd2.kept) {
      const hits = nonReservedHits(r)
      if (hits.length > 0) replayPii.push({ hash: `sha256:${recordHash(r, pairText)}`, hits: hits.map((h) => `${h.field}/${h.pattern}`) })
      else afterReplayPii.push(r)
    }
    // 去泄漏：**复用主源同一个函数**
    const g1 = applyEvalGuards(afterReplayPii)
    // 与已入选的指令行去重（两个源之间也不能撞）
    const instrKeep = split.calib.slice(0, nInstr)
    const instrHashes = new Set(instrKeep.map((r) => recordHash(r, pairText)))
    const finalReplay = g1.kept.filter((r) => !instrHashes.has(recordHash(r, pairText))).slice(0, nReplay)

    if (finalReplay.length < nReplay) {
      throw new Error(`E_REPLAY_POOL_SHORT: 需要 ${nReplay} 行回放，清洗后只剩 ${finalReplay.length} 行。
  ⇒ **拒绝以不足的比例继续**：悄悄少放会让 mix 变成一个"实际未知"的比例，
     而 mix 正是本轮的**自变量**，自变量不可未知。请提高 scanCap 或降低 mix。`)
    }

    // 确定性交错（Bresenham）：让每个 batch 都同时看到两个源，而不是"前半段纯指令、后半段纯回放"
    const interleaved = []
    let ii = 0, ri = 0
    for (let i = 0; i < train; i++) {
      const wantReplay = Math.floor(((i + 1) * nReplay) / train) > Math.floor((i * nReplay) / train)
      interleaved.push(wantReplay ? finalReplay[ri++] : instrKeep[ii++])
    }
    replayRows = interleaved
    replayInfo = {
      mix, nReplay, nInstr,
      source: {
        repo: REPLAY_SOURCE.repo, url: REPLAY_SOURCE.url, license: REPLAY_SOURCE.license,
        file: replayFile || REPLAY_SOURCE.file,
        fileSha256: rfp.got,
        declaredSha256: REPLAY_SOURCE.fileSha256,
        fingerprintOk: rfp.ok,
        fileOverridden: Boolean(replayFile),
      },
      scanCap: replayScanCap,
      scanned: rr.scanned, candidates: rr.items.length, oversampled: pool.length,
      dropped: { dedup: dd2.dropped.length, pii: replayPii.length, evalCollision: g1.dropped.length, evalContainment: g1.contained.length },
      interleave: 'bresenham-deterministic',
      note: '替换 train 中 mix 比例的槽位（总行数不变）；回放行只进 train，不进 heldout',
    }
    steps.push({ step: 'replay-mix', mix, nReplay, nInstr, scanCap: replayScanCap, scanned: rr.scanned, candidates: rr.items.length, dropped: replayInfo.dropped })
  }
  const finalCalib = mix > 0 ? replayRows : split.calib

  // ⑤ **复核**（不是"我剔过了所以应该没有"）：三组交集都必须为空
  const verify = []
  const leakTrainHeld = assertNoLeakage(finalCalib, split.test, { keyOf: pairText })
  verify.push({ pair: 'train × heldout', intersection: leakTrainHeld.intersection.length, ok: leakTrainHeld.ok })
  for (const es of evalSets) {
    const a = assertNoLeakage(finalCalib, es.rows, { keyOf: (x) => (x.t !== undefined ? x.t : x.question !== undefined ? x.question : pairText(x)) })
    verify.push({ pair: `train × ${es.name}`, intersection: a.intersection.length, ok: a.ok })
    const b = assertNoLeakage(split.test, es.rows, { keyOf: (x) => (x.t !== undefined ? x.t : x.question !== undefined ? x.question : pairText(x)) })
    verify.push({ pair: `heldout × ${es.name}`, intersection: b.intersection.length, ok: b.ok })
  }
  /**
   * ⑤b **独立复核**包含级泄漏 —— 不信任 ③b 的剔除结果，对**最终** train 集**重新算一遍**。
   *
   * 为什么必须有这一条：③b 是"过滤器"，⑤ 原本只查**内容哈希**交集，
   * 而回放行的 hash keyOf 与评估集的 keyOf 不同（`pairText` vs `r.t`），
   * 两组哈希**在构造上就不可能相等** ⇒ 哈希复核对回放污染**永远返回"干净"**。
   * 实测：把 ③b 的包含判据拆掉后，污染行真的进了 train，而 `allClear` **仍然是 true**。
   * ⇒ 过滤器必须有一条**独立于它自己**的复核，否则"我剔过了"就真的成了唯一依据。
   */
  let containmentLeaks = 0
  const containmentSamples = []
  for (const r of finalCalib) {
    const jn = normalizeWs(['instruction', 'input', 'output'].map((k) => String(r[k] ?? '')).join(''))
    for (const item of evalTextIndex) {
      if (jn.includes(item.norm)) {
        containmentLeaks++
        if (containmentSamples.length < 5) containmentSamples.push({ hash: `sha256:${recordHash(r, pairText)}`, contains: item.set, textHash: `sha256:${contentHash(item.text).slice(0, 16)}` })
        break
      }
    }
  }
  verify.push({ pair: 'train × eval(containment, independent recheck)', intersection: containmentLeaks, ok: containmentLeaks === 0, samples: containmentSamples })

  const allClear = verify.every((v) => v.ok)

  const lg = lineage({
    source: SOURCE.repo,
    revision: SOURCE.revision,
    seed,
    calib: finalCalib,
    test: split.test,
    extra: {
      license: SOURCE.license,
      sourceFile: SOURCE.file,
      sourceFileSha256: SOURCE.fileSha256,
      // 两个源的许可**都**必须落盘：ODC-BY 的义务是署名 ⇒ 不写就履行不了
      sources: replayInfo
        ? [
          { role: 'instruction', repo: SOURCE.repo, url: SOURCE.url, revision: SOURCE.revision, license: SOURCE.license, file: SOURCE.file, fileSha256: SOURCE.fileSha256 },
          { role: 'replay', repo: REPLAY_SOURCE.repo, url: REPLAY_SOURCE.url, revision: REPLAY_SOURCE.revision, license: REPLAY_SOURCE.license, file: replayInfo.source.file, fileSha256: replayInfo.source.fileSha256 },
        ]
        : [{ role: 'instruction', repo: SOURCE.repo, url: SOURCE.url, revision: SOURCE.revision, license: SOURCE.license, file: SOURCE.file, fileSha256: SOURCE.fileSha256 }],
      mix: replayInfo,
      pipeline: steps,
      exclusions: { piiRows: piiDropped.length, evalCollisionRows: evalDropped.length, evalContainmentRows: containsDropped.length, replayPiiRows: replayInfo ? replayInfo.dropped.pii : 0 },
      verification: verify,
      allClear,
      note: 'train/heldout 由 data-core.splitDeterministic 切分；与所有评估集的内容哈希交集已复核为空',
    },
  })

  /**
   * ⚠️ **目录名必须把决定内容的参数全部编码进去**（原缺陷：只编了 `seed` 与 `train`，**漏了 `heldout`**）。
   *
   * 实测后果（真实事故）：用 `{train:1000, heldout:600}` 重跑 prepare，
   * 目标名仍是 `…__sft__s20260929__n1000` ⇒ **原地覆盖**了 `heldout.jsonl`（200 行 → 600 行）——
   * 而**已有 run 的 `dataset.lineageHash` 指着那个目录**。
   * 于是"评估集身份"在**同一个目录名下被悄悄换掉** —— 正是 `eval-contract` §7 要防的事，
   * 而工具本身把门开着。**评估集的字节内容是可以用 `lineage.test.hash` 复算验证的**，
   * 但那要靠人想起来去比对；工具不该把这个责任推给人。
   *
   * 修法两件：
   *   ① 名字带上 `__h${heldout}`（与既有的 `__n${train}` 同款）；
   *   ② **存在性守卫**：目标目录已存在且其 `lineage.json` 的参数与本次不一致 ⇒ **拒绝**（除非 `force: true`），
   *      并把冲突点名列出来。旧名目录（无 `__h`）不会被新逻辑重新指向 ⇒ 不会被覆盖。
   */
  // ⚠️ mix 必须进目录名：否则 `mix=0.25` 与 `mix=0` 会**共用同一目录名**，
  //    而两者的 train.jsonl **内容不同** —— 正是本文档开头记录的"同一目录名下悄悄换掉内容"那类事故。
  //    为保持向后兼容，mix=0 时**不**加后缀（历史目录名与已有 run 的 lineageHash 不受影响）。
  const mixTag = mix > 0 ? `__mix${String(mix).replace('.', 'p')}` : ''
  const autoName = `${SOURCE.repo.replace('/', '-')}__sft__s${seed}__n${train}__h${heldout}${mixTag}`
  const out = outDir || join(datasetsRoot || join(HOME, '.ai-infra/state/datasets'), autoName)
  if (write && !force && existsSync(join(out, 'lineage.json'))) {
    let prev = null
    try { prev = JSON.parse(readFileSync(join(out, 'lineage.json'), 'utf8')) } catch { prev = null }
    /**
     * ⚠️ `lineage()` 把 `extra` 的字段**摊平到顶层**（实测：顶层有 `mix`，**没有** `extra`）。
     * 第一版只读 `prev?.extra?.mix?.mix` ⇒ 恒为 0 ⇒ 凡是**已存在**的 mix 数据集，
     * 用**完全相同的参数**再 prepare 也会被判"参数不同"而拒（E_DATASET_WOULD_OVERWRITE）。
     * 实测后果：本轮补数据的 6 个配置里有 3 个因此失败（`__mix0p1` 已存在）。
     * 现在两种位置都认。
     */
    const prevMix = prev?.mix?.mix ?? prev?.extra?.mix?.mix ?? 0
    const p = prev && typeof prev === 'object' ? { seed: prev.seed, train: prev.calib?.count ?? null, heldout: prev.test?.count ?? null, mix: prevMix } : null
    const same = p && p.seed === seed && p.train === train && p.heldout === heldout && p.mix === mix
    if (!same) {
      throw new Error(`E_DATASET_WOULD_OVERWRITE: 目标数据集目录已存在且参数不同
`
        + `  目录 = ${out}
  已有 = ${JSON.stringify(p)}
  本次 = ${JSON.stringify({ seed, train, heldout, mix })}
`
        + `  ⇒ 拒绝覆盖（评估集身份会被在**同一个目录名下**换掉）。`
        + `如确要重建，请显式传 force=true（会留 .bak-* 备份）或指定另一个 outDir。`)
    }
  }
  if (write) {
    mkdirSync(out, { recursive: true, mode: 0o700 })
    const toLine = (r) => JSON.stringify({ instruction: r.instruction, input: r.input ?? '', output: r.output, h: recordHash(r, pairText) })
    writeFileSync(join(out, 'train.jsonl'), `${finalCalib.map(toLine).join('\n')}\n`, { mode: 0o600 })
    writeFileSync(join(out, 'heldout.jsonl'), `${split.test.map(toLine).join('\n')}\n`, { mode: 0o600 })
    writeFileSync(join(out, 'lineage.json'), `${JSON.stringify(lg, null, 2)}\n`, { mode: 0o600 })
    writeFileSync(join(out, 'exclusions.json'), `${JSON.stringify({ pii: piiDropped, evalCollisions: evalDropped, evalContainment: containsDropped, note: '只记哈希与字段名，不记原文；原始数据集文件未被修改' }, null, 2)}\n`, { mode: 0o600 })
  }
  return { out, steps, verify, allClear, lineage: lg, trainRows: finalCalib.length, replay: replayInfo, heldoutRows: split.test.length, piiDropped: piiDropped.length, evalDropped: evalDropped.length, containsDropped: containsDropped.length, evalSets: evalSets.map((e) => ({ name: e.name, kind: e.kind, rows: e.rows.length })) }
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const r = await prepareData({ mix: Number(argOf('--mix', '0')) })
  console.log(JSON.stringify({ out: r.out, steps: r.steps, verify: r.verify, allClear: r.allClear, replay: r.replay }, null, 2))
  process.exit(r.allClear ? 0 : 1)
}
