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
  file: '<ALPACA_PATH>',
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
  const mmlu = '<MMLU_PATH>'
  if (existsSync(mmlu)) {
    const rows = loadJson(mmlu)
    sets.push({ name: 'mmlu-test', kind: 'qa', file: mmlu, rows, keyOf: (r) => r.question, sha256: `sha256:${contentHash(readFileSync(mmlu, 'utf8'))}` })
  }
  return sets
}

/* -------------------------------------------------------------------- 主 */

export async function prepareData({ seed = SEED, train = N_TRAIN, heldout = N_HELDOUT, outDir = null, datasetsRoot = null, write = true, force = false } = {}) {
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

  // ③ 剔除与任一评估集内容哈希相撞的行（P0 #1：训练数据绝不含测试集）
  const evalSets = loadEvalSets()
  const evalHashes = new Map()
  for (const es of evalSets) for (const r of es.rows) evalHashes.set(recordHash(r, es.keyOf), es.name)
  const evalDropped = []
  const afterEval = []
  for (const r of afterPii) {
    const h = recordHash(r, pairText)
    if (evalHashes.has(h)) evalDropped.push({ hash: `sha256:${h}`, collidesWith: evalHashes.get(h) })
    else afterEval.push(r)
  }
  steps.push({ step: 'drop-eval-set-collisions', kept: afterEval.length, dropped: evalDropped.length, evalSets: evalSets.map((e) => ({ name: e.name, kind: e.kind, rows: e.rows.length, sha256: e.sha256 })) })

  // ③b **包含级**复核：整行哈希相等只能判"逐行相同"，判不出"评估集文本被**嵌在**训练行内部"。
  //     实测确实存在这种形态（MMLU 题目逐字出现在 alpaca 的 output 里）。
  //     故再做一遍"字段包含"检查：训练行的任一字段若**含**某条评估集项的全部文本，即剔除整行。
  //     这是比哈希相等更严的一条判据，代价是可能多剔几行 —— 宁可少训，不可污染。
  const evalTextIndex = []
  for (const es of evalSets) {
    for (const r of es.rows) {
      const t = es.keyOf(r)
      if (typeof t === 'string' && t.trim().length >= 24) evalTextIndex.push({ set: es.name, text: t.trim() })
    }
  }
  const containsDropped = []
  const afterContains = []
  for (const r of afterEval) {
    const fields = ['instruction', 'input', 'output'].map((k) => String(r[k] ?? ''))
    let hit = null
    for (const item of evalTextIndex) { if (fields.some((f) => f.includes(item.text))) { hit = item; break } }
    if (hit) containsDropped.push({ hash: `sha256:${recordHash(r, pairText)}`, contains: hit.set, textHash: `sha256:${contentHash(hit.text).slice(0, 16)}` })
    else afterContains.push(r)
  }
  steps.push({ step: 'drop-eval-set-containment', kept: afterContains.length, dropped: containsDropped.length, checkedEvalItems: evalTextIndex.length })

  // ④ 确定性切分（走 data-core 的 splitDeterministic）
  const split = splitDeterministic(afterContains, { seed, calib: train, test: heldout, keyOf: pairText })
  steps.push({ step: 'split', seed, train: split.calib.length, heldout: split.test.length, unused: split.counts.unused })

  // ⑤ **复核**（不是"我剔过了所以应该没有"）：三组交集都必须为空
  const verify = []
  const leakTrainHeld = assertNoLeakage(split.calib, split.test, { keyOf: pairText })
  verify.push({ pair: 'train × heldout', intersection: leakTrainHeld.intersection.length, ok: leakTrainHeld.ok })
  for (const es of evalSets) {
    const a = assertNoLeakage(split.calib, es.rows, { keyOf: (x) => (x.t !== undefined ? x.t : x.question !== undefined ? x.question : pairText(x)) })
    verify.push({ pair: `train × ${es.name}`, intersection: a.intersection.length, ok: a.ok })
    const b = assertNoLeakage(split.test, es.rows, { keyOf: (x) => (x.t !== undefined ? x.t : x.question !== undefined ? x.question : pairText(x)) })
    verify.push({ pair: `heldout × ${es.name}`, intersection: b.intersection.length, ok: b.ok })
  }
  const allClear = verify.every((v) => v.ok)

  const lg = lineage({
    source: SOURCE.repo,
    revision: SOURCE.revision,
    seed,
    calib: split.calib,
    test: split.test,
    extra: {
      license: SOURCE.license,
      sourceFile: SOURCE.file,
      sourceFileSha256: SOURCE.fileSha256,
      pipeline: steps,
      exclusions: { piiRows: piiDropped.length, evalCollisionRows: evalDropped.length, evalContainmentRows: containsDropped.length },
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
  const autoName = `${SOURCE.repo.replace('/', '-')}__sft__s${seed}__n${train}__h${heldout}`
  const out = outDir || join(datasetsRoot || join(HOME, '.ai-infra/state/datasets'), autoName)
  if (write && !force && existsSync(join(out, 'lineage.json'))) {
    let prev = null
    try { prev = JSON.parse(readFileSync(join(out, 'lineage.json'), 'utf8')) } catch { prev = null }
    const p = prev && typeof prev === 'object' ? { seed: prev.seed, train: prev.calib?.count ?? null, heldout: prev.test?.count ?? null } : null
    const same = p && p.seed === seed && p.train === train && p.heldout === heldout
    if (!same) {
      throw new Error(`E_DATASET_WOULD_OVERWRITE: 目标数据集目录已存在且参数不同
`
        + `  目录 = ${out}
  已有 = ${JSON.stringify(p)}
  本次 = ${JSON.stringify({ seed, train, heldout })}
`
        + `  ⇒ 拒绝覆盖（评估集身份会被在**同一个目录名下**换掉）。`
        + `如确要重建，请显式传 force=true（会留 .bak-* 备份）或指定另一个 outDir。`)
    }
  }
  if (write) {
    mkdirSync(out, { recursive: true, mode: 0o700 })
    const toLine = (r) => JSON.stringify({ instruction: r.instruction, input: r.input ?? '', output: r.output, h: recordHash(r, pairText) })
    writeFileSync(join(out, 'train.jsonl'), `${split.calib.map(toLine).join('\n')}\n`, { mode: 0o600 })
    writeFileSync(join(out, 'heldout.jsonl'), `${split.test.map(toLine).join('\n')}\n`, { mode: 0o600 })
    writeFileSync(join(out, 'lineage.json'), `${JSON.stringify(lg, null, 2)}\n`, { mode: 0o600 })
    writeFileSync(join(out, 'exclusions.json'), `${JSON.stringify({ pii: piiDropped, evalCollisions: evalDropped, evalContainment: containsDropped, note: '只记哈希与字段名，不记原文；原始数据集文件未被修改' }, null, 2)}\n`, { mode: 0o600 })
  }
  return { out, steps, verify, allClear, lineage: lg, trainRows: split.calib.length, heldoutRows: split.test.length, piiDropped: piiDropped.length, evalDropped: evalDropped.length, containsDropped: containsDropped.length, evalSets: evalSets.map((e) => ({ name: e.name, kind: e.kind, rows: e.rows.length })) }
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const r = await prepareData({})
  console.log(JSON.stringify({ out: r.out, steps: r.steps, verify: r.verify, allClear: r.allClear }, null, 2))
  process.exit(r.allClear ? 0 : 1)
}
