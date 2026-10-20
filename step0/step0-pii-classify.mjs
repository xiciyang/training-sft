#!/usr/bin/env node
/**
 * step0-pii-classify.mjs —— 对 PII 命中做**证据性分类**（只读，内存内，明文不落盘）。
 *
 * 为什么需要分类：`piiScan` 报的是"命中了什么模式"，但指令数据里的邮箱/IP 很可能是
 * **保留示例**（`example.com` / `192.168.x.x` / `10.x.x.x`），与真实个人信息是两回事。
 * 纪律要求"发现 PII → 暂停报告、不自行脱敏"，**也要求报告要可裁决** ——
 * 只报"119 条命中"无法让人判断，故这里补上"其中多少是保留示例、多少不是"。
 *
 * 明文纪律：命中原文**只在内存里**用于分类，落盘只写计数与哈希前缀（与 diagnose 的
 * "审计只存摘要"同一条口径）。
 */
import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { piiScan, contentHash } from '../../shared/libs/data-core/src/index.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const OUT = join(HERE, '..', 'out')
mkdirSync(OUT, { recursive: true })

const CANDIDATES = [
  { id: 'A-tatsu-lab-alpaca', file: '/data/datasets/alpaca-nc/alpaca_data.jsonl' },
  { id: 'B-yahma-alpaca-cleaned', file: '/data/datasets/alpaca-cleaned/alpaca_data_cleaned.json' },
]

/** 保留/示例空间判定 —— 这些**不构成个人信息**（RFC 2606 / RFC 1918 / RFC 5737 / 常见占位）。 */
const RESERVED_DOMAIN = /@(example\.(com|org|net|edu)|test\.(com|org)|localhost|domain\.(com|org)|email\.com|yourdomain\.com|company\.com|mycompany\.com|site\.com|x\.com|foo\.com|bar\.com|abc\.com)$/i
const RESERVED_IP = /^(10\.|127\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.|0\.|255\.|192\.0\.2\.|198\.51\.100\.|203\.0\.113\.)/
const PLACEHOLDER_LOCAL = /^(user|test|example|your|name|email|someone|john\.?doe|jane\.?doe|abc|foo|bar|admin|info|contact|hello|me|you|my)@/i

function classify(kind, text) {
  if (kind === 'email') {
    if (RESERVED_DOMAIN.test(text)) return 'reserved-domain'
    if (PLACEHOLDER_LOCAL.test(text)) return 'placeholder-localpart'
    return 'other'
  }
  if (kind === 'ipv4') return RESERVED_IP.test(text) ? 'reserved-ip' : 'other'
  if (kind === 'phone-intl') return 'other'
  return 'other'
}

function load(p) {
  const text = readFileSync(p, 'utf8')
  if (text.trimStart().startsWith('[')) return JSON.parse(text)
  return text.split('\n').filter((l) => l.trim()).map((l) => JSON.parse(l))
}

const R = { generatedAt: new Date().toISOString(), note: '命中原文只在内存中用于分类；落盘仅计数与哈希前缀', candidates: [] }
for (const c of CANDIDATES) {
  if (!existsSync(c.file)) { R.candidates.push({ id: c.id, error: 'missing' }); continue }
  const rows = load(c.file)
  const buckets = {}
  const fieldBuckets = {}
  const samples = []
  let total = 0
  for (const r of rows) {
    for (const k of ['instruction', 'input', 'output']) {
      const s = typeof r[k] === 'string' ? r[k].slice(0, 2000) : ''
      if (!s) continue
      const scan = piiScan(s)
      for (const f of scan.findings) {
        const raw = s.slice(f.index, f.index + f.length)
        const cls = classify(f.pattern, raw)
        buckets[`${f.pattern}/${cls}`] = (buckets[`${f.pattern}/${cls}`] || 0) + 1
        fieldBuckets[`${k}/${cls}`] = (fieldBuckets[`${k}/${cls}`] || 0) + 1
        total += 1
        if (cls === 'other' && samples.length < 8) {
          // 只留哈希前缀与长度，不留原文
          samples.push({ field: k, pattern: f.pattern, len: raw.length, hash: `sha256:${contentHash(raw).slice(0, 16)}` })
        }
      }
    }
  }
  const other = Object.entries(buckets).filter(([k]) => k.endsWith('/other')).reduce((a, [, v]) => a + v, 0)
  R.candidates.push({ id: c.id, rows: rows.length, totalHits: total, buckets, fieldBuckets, nonReservedHits: other, nonReservedSamples: samples })
  console.log(`${c.id}: 命中 ${total}；其中非保留/非占位 ${other}`)
  console.log('  buckets:', JSON.stringify(buckets))
}

writeFileSync(join(OUT, 'step0-pii-classified.json'), `${JSON.stringify(R, null, 2)}\n`)
console.log('written', join(OUT, 'step0-pii-classified.json'))
