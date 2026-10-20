#!/usr/bin/env node
/**
 * replay.mjs —— **通用语料回放（replay）** 的第二语料源与采样（第二段 C1 能力）。
 *
 * ── 为什么要它（由潜变量导出，见 LATENT_VARIABLE.md）────────────────────────
 * 实测 gain=f(‖ΔW‖) 与 cost=g(‖ΔW‖) **同向递增** ⇒ 在同一条曲线上取点，
 * 改变不了"增益/代价"的比值。要同时拿到高增益与低代价，必须**移动曲线本身**。
 * replay 的作用是让同一段适配位移**同时**服务"格式"与"通用语言建模"，
 * 从而改变"每单位适配幅度换来多少格式增益"。
 *
 * ── 为什么**不**用 wikitext 当第二源（取证结论，见下方 license）─────────────
 * 本项目自己的血缘里已记录：wikitext 许可为 **cc-by-sa-3.0（share-alike）**，
 * 且明确写着"在本项目只作量化校准集与遗忘探针评估集，**不参与主模型训练**"。
 * ⇒ 拿它做训练语料会**违反本项目自己记录在案的许可立场**。故改用 dolma。
 *
 * ── 许可（odc-by）────────────────────────────────────────────────────────
 * dolma 血缘（training-pretrain 侧）已记录：spdx=odc-by、class=permissive-with-attribution、
 * commercialUse=true、requiresAttribution=true，并附 HF API 取证。
 * ⇒ 可训练，**义务是署名** ⇒ 必须把 repo/url/spdx 一起写进数据集 lineage。
 *
 * ── 本模块的职责边界（很重要）────────────────────────────────────────────
 * 本模块**只**负责"从第二源取出干净文本窗口"。
 *   **不**做 PII 判定、**不**做去泄漏判定 —— 那两条**必须**复用 prepare.mjs 里
 *   与主源**完全相同**的那份代码（`nonReservedHits` / 评估集索引）。
 *   否则"新能力自带一套更松的检查"就成了绕过保障的后门。
 */
import { createGunzip } from 'node:zlib'
import { createReadStream } from 'node:fs'
import { createInterface } from 'node:readline'
import { contentHash } from '@xiciyang/data-core'

export const REPLAY_SOURCE = Object.freeze({
  repo: 'allenai/dolma',
  url: 'https://hf-mirror.com/datasets/allenai/dolma',
  revision: null,
  license: {
    spdx: 'odc-by',
    class: 'permissive-with-attribution',
    requiresAttribution: true,
    commercialUse: true,
    source: 'https://hf-mirror.com/datasets/allenai/dolma',
    evidence: [
      'HF API tags 含 "license:odc-by"（https://hf-mirror.com/api/datasets/allenai/dolma）',
      '与 training-pretrain 侧 dolma__pretrain__* 血缘中记录的许可一致（同源同结论）',
    ],
    note: 'ODC-BY 的义务是**署名**；训练权重不受 share-alike 传染 ⇒ 可参与主模型训练（与 wikitext 相反）。',
  },
  file: process.env.DSH_SHARD_PATH,
  fileSha256: 'sha256:71f2ab92baf0f8a9073dc573ac61d3b28546516849ec460d1052471c5e47ee59',
  textField: 'text',
})

/** 清洗：只做**空白规整**，不改写词句（与本项目"不自行改写语料"的一贯口径一致）。 */
export function cleanText(s) {
  return String(s ?? '').replace(/\r\n?/g, '\n').replace(/[ \t]+/g, ' ').replace(/\n{3,}/g, '\n\n').replace(/[ ]+\n/g, '\n').trim()
}

/**
 * 从长文本里取一个**定长窗口**，起点由内容哈希决定。
 * 理由：dolma 的 gutenberg 行开头常是书名/版权样板；固定取开头会让回放样本**系统性偏样板**。
 * 用内容哈希定位起点 ⇒ 采样自正文，且**完全确定**（同文本同窗口，不依赖进程状态)。
 */
export function windowOf(text, { maxChars = 1600 } = {}) {
  if (text.length <= maxChars) return text
  const h = parseInt(contentHash(text).slice(0, 8), 16)
  const start = h % (text.length - maxChars)
  return text.slice(start, start + maxChars)
}

/** 在句子边界附近把窗口切成 (prefix, continuation)，让续写部分尽量是完整句子。 */
export function splitAtSentence(text, { ratio = 0.5 } = {}) {
  const target = Math.floor(text.length * ratio)
  const win = 160
  const lo = Math.max(1, target - win), hi = Math.min(text.length - 1, target + win)
  let best = -1, bestD = Infinity
  for (let i = lo; i < hi; i++) {
    const c = text[i]
    if (c === '.' || c === '!' || c === '?' || c === '\n') { const d = Math.abs(i - target); if (d < bestD) { bestD = d; best = i } }
  }
  if (best < 0) best = target
  // ⚠️ **不 trim**：两边各 trim 一次会把接缝处的空白**吃掉**，于是 input+output ≠ 原文。
  // 后果不只是"少一个空格"—— 去泄漏判据里那条"拼接串包含评估文本"会因为这一个空格而**判不出**，
  // 而模型看到的仍然是几乎逐字的评估文本。接缝处不丢字符，判据才是准的。
  return [text.slice(0, best + 1), text.slice(best + 1)]
}

/**
 * 流式读第二源，产出**候选文本窗口**（未经 PII / 去泄漏判定）。
 * 返回 `{ items, scanned, rejected: {tooShort, tooLong, empty} }`。
 */
export async function readReplayTexts({ file = REPLAY_SOURCE.file, scanCap = 20000, minChars = 400, maxChars = 1600, maxWindow = 1600 } = {}) {
  if (!file.endsWith('.gz')) throw new Error(`第二语料源必须是 .json.gz：${file}`)
  const rl = createInterface({ input: createReadStream(file).pipe(createGunzip()), crlfDelay: Infinity })
  const items = []
  const rejected = { tooShort: 0, empty: 0, malformed: 0 }
  let scanned = 0
  for await (const line of rl) {
    if (!line.trim()) continue
    scanned++
    let o = null
    try { o = JSON.parse(line) } catch { rejected.malformed++; continue }
    const t = cleanText(o?.[REPLAY_SOURCE.textField])
    if (!t) { rejected.empty++; continue }
    if (t.length < minChars) { rejected.tooShort++; continue }
    // 比窗口短的**照用**（windowOf 会原样返回）；只有短于 minChars 才丢。
    items.push({ id: o.id ?? null, src: o.source ?? null, window: windowOf(t, { maxChars: maxWindow }) })
    if (scanned >= scanCap) break
  }
  rl.close()
  return { items, scanned, rejected }
}

/**
 * 确定性抽样：按 `contentHash(seed + ':' + id)` 排序取前 n。
 * **不用 PRNG**（无内部状态、可跨进程复现），seed 只作为排序键的一部分。
 */
export function selectDeterministic(items, n, seed) {
  return [...items]
    .map((it) => ({ it, k: contentHash(`${seed}:${it.id ?? it.window.slice(0, 64)}`) }))
    .sort((a, b) => (a.k < b.k ? -1 : a.k > b.k ? 1 : 0))
    .slice(0, n)
    .map((x) => x.it)
}

/** 窗口 → 与主源同构的 `{instruction, input, output}` 行（纯续写，instruction 留空）。 */
export function toReplayRow(item, { instruction = '', ratio = 0.5 } = {}) {
  const [prefix, continuation] = splitAtSentence(item.window, { ratio })
  return { instruction, input: prefix, output: continuation, replayOf: item.id ?? null }
}

/** 计算第二源文件指纹并与记录值比对（**不信任缓存**，每次现算）。 */
export async function verifyReplaySource({ file = REPLAY_SOURCE.file } = {}) {
  const { createHash } = await import('node:crypto')
  return await new Promise((resolve, reject) => {
    const h = createHash('sha256')
    createReadStream(file).on('data', (d) => h.update(d)).on('end', () => {
      const got = `sha256:${h.digest('hex')}`
      resolve({ ok: got === REPLAY_SOURCE.fileSha256, got, expected: REPLAY_SOURCE.fileSha256 })
    }).on('error', reject)
  })
}

if (process.argv[1] && import.meta.url === `file://${process.argv[1]}`) {
  const v = await verifyReplaySource()
  const r = await readReplayTexts({ scanCap: 2000 })
  console.log(JSON.stringify({ source: REPLAY_SOURCE.repo, license: REPLAY_SOURCE.license.spdx, fingerprint: v, scanned: r.scanned, candidates: r.items.length, rejected: r.rejected, sample: r.items[0] ? { id: r.items[0].id, src: r.items[0].src, chars: r.items[0].window.length } : null }, null, 2))
  process.exit(v.ok ? 0 : 1)
}
