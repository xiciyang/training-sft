#!/usr/bin/env node
/**
 * step0-data.mjs —— 长任务 2 / 阶段 1 的数据侧只读检查（**不训练、不选数据**）。
 *
 * 对**两个候选**做同一套检查，供用户裁决：
 *   A. tatsu-lab/alpaca（实测 license = CC BY-NC 4.0，**非商用**）
 *   B. yahma/alpaca-cleaned（实测 license = CC BY 4.0，**permissive**）
 *
 * 检查项：条数/格式/license/PII/内容哈希/与各测试集的交集/候选之间的交集。
 * 全部复用 `@xiciyang/data-core` 的判据（不另写一套哈希或去泄漏逻辑）。
 */
import { readFileSync, writeFileSync, mkdirSync, existsSync, readdirSync, statSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { contentHash, recordHash, assertNoLeakage, dedupByContent, piiScan, PII_PATTERNS } from '../../shared/libs/data-core/src/index.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const HOME = process.env.HOME
const OUT = join(HERE, '..', 'out')
mkdirSync(OUT, { recursive: true })

const CANDIDATES = [
  { id: 'A-tatsu-lab-alpaca', name: 'tatsu-lab/alpaca', file: '/data/datasets/alpaca-nc/alpaca_data.jsonl', license: { spdx: 'cc-by-nc-4.0', class: 'restricted', source: 'https://hf-mirror.com/api/datasets/tatsu-lab/alpaca (tags: license:cc-by-nc-4.0) + 数据集卡片正文"available under the Creative Commons NonCommercial (CC BY-NC 4.0)"' } },
  { id: 'B-yahma-alpaca-cleaned', name: 'yahma/alpaca-cleaned', file: '/data/datasets/alpaca-cleaned/alpaca_data_cleaned.json', license: { spdx: 'cc-by-4.0', class: 'permissive', source: 'https://hf-mirror.com/api/datasets/yahma/alpaca-cleaned (tags: license:cc-by-4.0)' } },
]

const TEST_SETS = []
// ① wikitext-2 test（长任务 1 已制备；仅作**评估集**，不训练）
for (const d of readdirSync(join(HOME, '.ai-infra/state/datasets'))) {
  const p = join(HOME, '.ai-infra/state/datasets', d, 'test.jsonl')
  if (existsSync(p)) TEST_SETS.push({ name: `wikitext-test(${d.includes('c512') ? 'c512' : 'c128'})`, file: p, keyOf: (r) => r.t })
}
// ② MMLU（HF datasets 缓存的 parquet，permissive MIT）
const mmluDirs = []
const mmluRoot = join(HOME, '.cache/huggingface/datasets/cais___mmlu')
if (existsSync(mmluRoot)) {
  for (const sub of readdirSync(mmluRoot)) {
    for (const ver of readdirSync(join(mmluRoot, sub))) {
      const sd = join(mmluRoot, sub, ver)
      if (!statSync(sd).isDirectory()) continue
      for (const f of readdirSync(sd)) if (f.endsWith('.parquet')) mmluDirs.push(join(sd, f))
    }
  }
}

function loadJsonl(p, limit = Infinity) {
  const text = readFileSync(p, 'utf8')
  const rows = []
  if (text.trimStart().startsWith('[')) {
    const arr = JSON.parse(text)
    for (const r of arr) { rows.push(r); if (rows.length >= limit) break }
  } else {
    for (const line of text.split('\n')) {
      if (line.trim() === '') continue
      rows.push(JSON.parse(line))
      if (rows.length >= limit) break
    }
  }
  return { rows, bytes: Buffer.byteLength(text), sha256: `sha256:${contentHash(text)}` }
}

/** prompt/response 对 → 规范化内容键（与将来训练用的整理方式一致） */
function pairText(r) {
  const inst = String(r.instruction ?? '').trim()
  const inp = String(r.input ?? '').trim()
  const out = String(r.output ?? '').trim()
  return `${inst}\u0001${inp}\u0001${out}`
}

const R = { generatedAt: new Date().toISOString(), candidates: [], testSets: [], intersections: [], piiPatterns: PII_PATTERNS.map((p) => p.id) }

const loaded = {}
for (const c of CANDIDATES) {
  if (!existsSync(c.file)) { R.candidates.push({ ...c, error: '文件不存在' }); continue }
  const { rows, bytes, sha256 } = loadJsonl(c.file)
  loaded[c.id] = rows
  const fields = [...new Set(rows.flatMap((r) => Object.keys(r)))]
  // PII：全量扫（有界：每条最多 2000 字符 × 3 字段），记录**命中字段名**，不脱敏
  const byPattern = {}
  const fieldHits = {}
  let piiHits = 0
  for (const r of rows) {
    for (const k of ['instruction', 'input', 'output']) {
      if (typeof r[k] !== 'string' || r[k] === '') continue
      const s = piiScan(r[k].slice(0, 2000))
      if (s.count > 0) {
        piiHits += s.count
        fieldHits[k] = (fieldHits[k] || 0) + s.count
        for (const f of s.findings) byPattern[f.pattern] = (byPattern[f.pattern] || 0) + 1
      }
    }
  }
  const dd = dedupByContent(rows, { keyOf: pairText })
  R.candidates.push({
    id: c.id, name: c.name, file: c.file, bytes, fileSha256: sha256,
    rows: rows.length, fields, license: c.license,
    dedup: { unique: dd.kept.length, duplicates: dd.dropped.length },
    pii: { hits: piiHits, byPattern, fieldHits, scannedRecords: rows.length, note: '命中即记录字段，不自行脱敏' },
    sample: rows.slice(0, 2).map((r) => ({ instruction: String(r.instruction).slice(0, 120), input: String(r.input).slice(0, 60), output: String(r.output).slice(0, 120) })),
  })
  console.log(`loaded ${c.name}: ${rows.length} 行 / ${(bytes / 1e6).toFixed(1)} MB / PII 命中 ${piiHits}`)
}

// ── 测试集：加载 wikitext（jsonl）；（MMLU 的 parquet 交由容器侧另跑，见 mmlu-scan） ──
for (const t of TEST_SETS) {
  const { rows, sha256 } = loadJsonl(t.file)
  R.testSets.push({ name: t.name, file: t.file, rows: rows.length, sha256 })
  t.rows = rows
}

// ── 交集（内容哈希）──
const keyOfCand = pairText
for (const c of CANDIDATES) {
  if (!loaded[c.id]) continue
  for (const t of TEST_SETS) {
    if (!t.rows) continue
    const r = assertNoLeakage(loaded[c.id], t.rows, { keyOf: (x) => (x.t !== undefined ? x.t : keyOfCand(x)) })
    R.intersections.push({ kind: 'train×test', a: c.id, b: t.name, aRows: loaded[c.id].length, bRows: t.rows.length, intersection: r.intersection.length, ok: r.ok, sample: r.intersection.slice(0, 3) })
    console.log(`交集 ${c.id} × ${t.name}: ${r.intersection.length} ${r.ok ? 'OK' : 'LEAK'}`)
  }
  // 候选之间的交集（两个 alpaca 变体，若混用会泄漏）
  for (const c2 of CANDIDATES) {
    if (c2.id <= c.id || !loaded[c2.id]) continue
    const r = assertNoLeakage(loaded[c.id], loaded[c2.id], { keyOf: keyOfCand })
    R.intersections.push({ kind: 'candidate×candidate', a: c.id, b: c2.id, aRows: loaded[c.id].length, bRows: loaded[c2.id].length, intersection: r.intersection.length, ok: r.ok, note: '两者是同一上游的清洗前后版本；混用会泄漏' })
    console.log(`交集 ${c.id} × ${c2.id}: ${r.intersection.length}`)
  }
}

// ── 交集：内部自洽子集（训练 vs 留出）预览 ──
for (const c of CANDIDATES) {
  if (!loaded[c.id]) continue
  const rows = loaded[c.id]
  const hashes = rows.map((r) => recordHash(r, keyOfCand))
  // 按内容哈希排重后的规模，以及"能否切出 1000 训练 + 200 留出且互不重叠"
  const uniqueKeys = new Set(hashes)
  R.candidates.find((x) => x.id === c.id).uniqueContentKeys = uniqueKeys.size
}

writeFileSync(join(OUT, 'step0-data.json'), `${JSON.stringify(R, null, 2)}\n`)
console.log('\nwritten', join(OUT, 'step0-data.json'))
console.log(JSON.stringify({ candidates: R.candidates.map((c) => ({ id: c.id, rows: c.rows, license: c.license.class, pii: c.pii.hits, piiFields: c.pii.fieldHits, dup: c.dedup.duplicates, unique: c.uniqueContentKeys })), intersections: R.intersections }, null, 1))
