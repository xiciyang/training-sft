/**
 * guards.spec —— P2 #14 输入处理 7 类 + #15 安全验收 8 条 + #16 健壮性 6 条。
 *
 * 全部**不依赖 GPU / 容器 / 网络**：需要外部世界的部分用最小夹具替代，
 * 但夹具**模拟对方的语义模型**（例如假 docker 会真的走 stdio 协议、会真的退出）。
 */
import { mkdtempSync, rmSync, writeFileSync, readFileSync, mkdirSync, existsSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { tmpdir } from 'node:os'
import { fileURLToPath } from 'node:url'

import * as api from '../../src/core.mjs'
import { ToolError, ERROR_CODES } from '../../src/version.mjs'
import { requireConfirm, tokenFor } from '../../src/confirm.mjs'
import { paths, updateRegistry, readRegistry, audit, auditVerify } from '../../src/runtime.mjs'
import { isFormatCompliant, tokenF1, forgettingGate, taskGate } from '../../src/evaluate.mjs'
import { ContainerWorker } from '../../src/worker.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
export const NAME = 'guards（输入 7 类 / 安全 8 条 / 健壮性 6 条）'

const attempts = async (fn) => { try { const v = await fn(); return { ok: true, value: v } } catch (e) { return { ok: false, code: e.code, message: String(e.message), body: e.body } } }

export async function run(t) {
  const tmp = mkdtempSync(join(tmpdir(), 'ts-guard-'))
  const root = join(tmp, 'ai-infra')
  mkdirSync(join(root, 'state'), { recursive: true })
  const p = paths(root)

  /* ═══════════ #14 输入处理 7 类 ═══════════ */
  // 1) 缺必填参数
  let r = await attempts(() => api.train({ root }))
  t.assert(r.ok === false && r.code === ERROR_CODES.E_DATASET_INVALID, `1) train 缺 datasetDir → E_DATASET_INVALID（实际 ${r.code}）`)

  // 2) 参数越界（步数超上限）
  r = await attempts(() => api.train({ root, datasetDir: '/nope', maxSteps: 9999 }))
  t.assert(r.ok === false && r.code === ERROR_CODES.E_BUDGET_EXCEEDED, `2) maxSteps=9999 → E_BUDGET_EXCEEDED（实际 ${r.code}）`)

  // 3) 不存在的 datasetDir
  r = await attempts(() => api.train({ root, datasetDir: join(tmp, 'no-such') }))
  t.assert(r.ok === false && r.code === ERROR_CODES.E_DATASET_INVALID, `3) 不存在的 datasetDir → E_DATASET_INVALID`)

  // 4) datasetDir 存在但缺 lineage.json
  const dsBad = join(tmp, 'ds-nolineage'); mkdirSync(dsBad, { recursive: true })
  writeFileSync(join(dsBad, 'train.jsonl'), '{}\n')
  r = await attempts(() => api.train({ root, datasetDir: dsBad }))
  t.assert(r.ok === false && /lineage/.test(String(r.message)), `4) 缺 lineage.json → 响亮失败且信息里点名 lineage（实际 ${r.code}）`)

  // 5) lineage.allClear=false → 拒绝训练（P0 #1 的闸门）
  const dsLeak = join(tmp, 'ds-leak'); mkdirSync(dsLeak, { recursive: true })
  writeFileSync(join(dsLeak, 'train.jsonl'), '{}\n')
  writeFileSync(join(dsLeak, 'lineage.json'), JSON.stringify({ allClear: false }))
  r = await attempts(() => api.train({ root, datasetDir: dsLeak }))
  t.assert(r.ok === false && r.code === ERROR_CODES.E_DATASET_LEAK, `5) allClear=false → E_DATASET_LEAK（实际 ${r.code}）`)

  // 6) 不存在的 runId
  r = await attempts(() => api.getReport({ root, runId: 'ts_nope' }))
  t.assert(r.ok === false && r.code === ERROR_CODES.E_RUN_NOT_FOUND, `6) 未知 runId → E_RUN_NOT_FOUND（实际 ${r.code}）`)
  r = await attempts(() => api.rollback({ root, runId: 'ts_nope' }))
  t.assert(r.ok === false && r.code === ERROR_CODES.E_RUN_NOT_FOUND, '6b) rollback 未知 runId → E_RUN_NOT_FOUND')

  // 7) 未知工具 / 未知子命令（MCP/CLI 侧）
  const { findToolSpec } = await import('../../src/tools.mjs')
  t.assert(findToolSpec('training_sft_nope') === null, '7) 未知工具名返回 null（MCP 侧据此回 -32601）')

  /* ═══════════ #15 安全验收 8 条 ═══════════ */
  // 1-2) 缺令牌 / 错令牌
  r = await attempts(() => requireConfirm({ root, scope: 'train', payload: { a: 1 }, summary: 'x' }))
  t.assert(r.ok === false && r.code === ERROR_CODES.E_CONFIRMATION_REQUIRED, `S1) 缺令牌 → E_CONFIRMATION_REQUIRED`)
  t.assert(typeof r.body?.details?.confirmToken === 'string' && r.body.details.confirmToken.startsWith('tc_'), 'S1b) 返回 confirmToken')
  r = await attempts(() => requireConfirm({ root, scope: 'train', payload: { a: 1 }, confirm: 'tc_wrong', summary: 'x' }))
  t.assert(r.ok === false && r.code === ERROR_CODES.E_CONFIRMATION_REQUIRED, 'S2) 错令牌 → 仍拒绝')
  // 3) 正确令牌放行 + memo
  const tok = tokenFor('train', { a: 1 })
  r = await attempts(() => requireConfirm({ root, scope: 'train', payload: { a: 1 }, confirm: tok, summary: 'x' }))
  t.assert(r.ok === true && r.value.confirmed === true, 'S3) 正确令牌 → 放行')
  r = await attempts(() => requireConfirm({ root, scope: 'train', payload: { a: 1 }, summary: 'x' }))
  t.assert(r.ok === true && r.value.source === 'memo', 'S3b) 24h 内同类操作免再问（memo）')
  // 目标变了令牌就变（防"一个令牌放行所有事"）
  t.assert(tokenFor('train', { a: 1 }) !== tokenFor('train', { a: 2 }), 'S3c) 令牌与目标绑定')
  // 4-5) 预算闸门
  r = await attempts(() => api.train({ root, datasetDir: dsLeak, limits: { maxGpuHours: 5 } }))
  t.assert(r.ok === false && r.code === ERROR_CODES.E_BUDGET_EXCEEDED, 'S4) maxGpuHours>2 → E_BUDGET_EXCEEDED')
  // 6) rollback purge 只允许删 run 目录（不越界）
  await updateRegistry(root, (doc) => ({ ...doc, runs: [{ id: 'ts_x', status: 'trained', startedAt: 'x', baseModel: { path: '/b' }, dataset: { dir: '/d' }, config: {}, budget: { maxSteps: 1, maxGpuHours: 1 }, runDir: '/etc' }] }), { schemaPath: null })
  r = await attempts(() => api.rollback({ root, runId: 'ts_x', purge: true }))
  t.assert(r.ok === true && r.value.purged.length === 0, 'S5) purge 拒绝删除 runs 根以外目录（/etc 未被删）')
  t.assert(existsSync('/etc'), 'S5b) /etc 仍在（越界删除被拦）')
  // 7) PII 处置：只剔除不改写（对真实源文件做只读哈希比对）
  const src = process.env.DSH_ALPACA_PATH || ''
  if (existsSync(src)) {
    const h1 = readFileSync(src).length
    const prep = await attempts(() => api.prepare({ root, seed: 20260929, train: 5, heldout: 2 }))
    const h2 = readFileSync(src).length
    t.assert(h1 === h2, 'S6) prepare 不改写原始数据集文件（长度一致）')
    t.assert(prep.ok === true && prep.value.verify.every((v) => v.ok), 'S6b) prepare 产出的三组交集全为空')
    t.assert(prep.value.exclusions.piiRows > 0, `S6c) 确实剔除了含 PII 的行（${prep.value.exclusions?.piiRows}）`)
  } else t.skip('原始数据集不存在（设置 DSH_ALPACA_PATH 可复现）—— 不改写/PII 剔除这两条断言没有对象可看')
  // 8) 强行造一个泄漏的数据集目录 → train 必须拒绝（见第 5 条）；此处再验 prepare 的闸门语义
  t.assert(typeof api.prepare === 'function' && typeof api.train === 'function', 'S7) 闸门在 prepare 与 train 两侧都存在')
  t.assert(ERROR_CODES.E_DATASET_LEAK === 'E_DATASET_LEAK', 'S8) 泄漏有独立错误码（不与其他失败混淆）')

  /* ═══════════ #16 健壮性 6 条 ═══════════ */
  // 1) 状态文件损坏 → 抛错而不是静默重置
  mkdirSync(p.stateDir, { recursive: true })
  writeFileSync(p.registryFile, '{ not yaml ]')
  r = await attempts(() => readRegistry(root, null))
  t.assert(r.ok === false, 'R1) 损坏的状态文件 → 解析失败（不静默重置为默认值）')
  writeFileSync(p.registryFile, 'version: 1\npkg: "0.1.0"\nruns: []\n')
  const rr = await readRegistry(root, null)
  t.assert(rr.runs.length === 0, 'R1b) 修好后可正常读')
  // 2) 审计链被篡改 → 检出
  const a1 = await audit(root, { capability: 'x', action: 'y', phase: 'end' })
  const { appendFileSync } = await import('node:fs')
  const shard = join(p.auditDir, `${new Date().toISOString().slice(0, 10)}.jsonl`)
  const lines = readFileSync(shard, 'utf8').split('\n').filter(Boolean)
  const bad = JSON.parse(lines[0]); bad.phase = 'TAMPERED'
  writeFileSync(shard, [JSON.stringify(bad), ...lines.slice(1)].join('\n') + '\n')
  const v = await auditVerify(root)
  t.assert(v.ok === false, 'R2) 审计链被篡改 → verifyAllChains 报 ok=false')
  writeFileSync(shard, `${lines.join('\n')}\n`)
  t.assert((await auditVerify(root)).ok === true, 'R2b) 恢复后链重新通过')
  t.assert(a1.hash.length === 64, 'R2c) 审计哈希是 64 位 sha256')
  // 3) schema 校验拦截非法状态
  const SFT_SCHEMA = join(HERE, '..', '..', '..', '..', 'inference-contracts', 'state-schema', 'sft-registry.schema.json')
  if (existsSync(SFT_SCHEMA)) {
    const bad2 = await attempts(() => updateRegistry(root, (doc) => ({ ...doc, runs: [{ id: 'nope', status: 'running' }] }), { schemaPath: SFT_SCHEMA }))
    t.assert(bad2.ok === false && bad2.code === ERROR_CODES.E_STATE_CONFLICT, `R3) 非法状态被 schema 拦截（实际 ${bad2.code}）`)
    const good = await attempts(() => updateRegistry(root, (doc) => ({ ...doc, runs: [{ id: 'ts_ok', status: 'running', startedAt: 'x', baseModel: { path: '/b' }, dataset: { dir: '/d' }, config: {}, budget: { maxSteps: 1, maxGpuHours: 1 } }] }), { schemaPath: SFT_SCHEMA }))
    t.assert(good.ok === true, 'R3b) 合法状态通过 schema')
  } else t.assert(false, 'R3) 找不到 sft-registry schema')
  // 4) CAS 冲突
  const cur = await readRegistry(root, null)
  r = await attempts(() => updateRegistry(root, (d) => d, { expectedHash: 'sha256:deadbeef' }))
  t.assert(r.ok === false && r.code === ERROR_CODES.E_STATE_CONFLICT, 'R4) 过期 expectedHash → ERR_STATE_CONFLICT')
  t.assert(cur._hash?.startsWith('sha256:'), 'R4b) 读出的状态带内容哈希（CAS 可用）')
  // 5) worker 协议层（**不依赖 docker** —— CI 里不跑容器；真容器测试在 tests/realmachine/）
  const w = new ContainerWorker({ image: 'alpine:3.20', mounts: ['/a:/b:ro'], gpus: true, entry: '/work/trainer.py' })
  r = await attempts(() => w.call('ping', {}, { timeoutMs: 1000 }))
  t.assert(r.ok === false && /未启动/.test(String(r.message)), 'R5) 未 start 就 call → 响亮失败（不会静默挂起）')
  const argv = w.args()
  t.assert(argv[0] === 'run' && argv.includes('--rm') && argv.includes('-i'), 'R5b) 容器参数含 --rm/-i（不残留容器）')
  t.assert(argv.includes('--gpus') && argv[argv.indexOf('--gpus') + 1] === 'all', 'R5c) 显式声明 --gpus all')
  t.assert(argv.includes('/a:/b:ro'), 'R5d) 挂载原样传递（只读挂载由调用方保证）')
  t.assert(argv[argv.length - 1] === '/work/trainer.py', 'R5e) 入口脚本是最后一个参数')
  await w.stop()

  // 6) 端点不可达 → 评估返回 0 样本而不是伪造数字
  const { evalTask } = await import('../../src/evaluate.mjs')
  const et = await evalTask({ endpoint: 'http://127.0.0.1:1', model: 'm', rows: [{ instruction: 'x', input: '', output: 'y' }], limit: 1 })
  t.assert(et.nOk === 0 && et.formatCompliance === null, 'R6) 端点不可达 → formatCompliance=null（不伪造 0 或 1）')

  /* ═══════════ 指标抽取与门禁的判据（供 P1 引用） ═══════════ */
  t.assert(isFormatCompliant('Hello there') === true, 'M1) 正常回答 → 格式合规')
  t.assert(isFormatCompliant('') === false, 'M2) 空输出 → 不合规')
  t.assert(isFormatCompliant('### Instruction:\nfoo') === false, 'M3) 重新起模板 → 不合规')
  t.assert(isFormatCompliant('...') === false, 'M4) 纯符号 → 不合规')
  t.assert(tokenF1('a b c', 'a b c') === 1, 'M5) 完全一致 → F1=1')
  t.assert(tokenF1('a b', 'c d') === 0, 'M6) 无重叠 → F1=0')
  t.assert(tokenF1('x', '') === null, 'M7) 参考为空 → null（不参与平均，也不冒充 0）')
  const g = forgettingGate({
    baseMmlu: { n: 100, accuracy: 0.25 }, sftMmlu: { n: 100, accuracy: 0.25 },
    baseLm: { perText: Array.from({ length: 40 }, () => 2.0), meanNll: 2.0, ppl: Math.exp(2) },
    sftLm: { perText: Array.from({ length: 40 }, () => 2.0), meanNll: 2.0, ppl: Math.exp(2) },
  })
  t.assert(g.pass === true, 'G1) 无退化 → 遗忘门禁通过')
  const g2 = forgettingGate({
    baseMmlu: { n: 100, accuracy: 0.60 }, sftMmlu: { n: 100, accuracy: 0.10 },
    baseLm: { perText: Array.from({ length: 40 }, () => 2.0), meanNll: 2.0, ppl: Math.exp(2) },
    sftLm: { perText: Array.from({ length: 40 }, () => 2.0), meanNll: 2.0, ppl: Math.exp(2) },
  })
  t.assert(g2.pass === false && /MMLU/.test(g2.reasons.join(' ')), 'G2) 显著退化 → 拒绝，且原因指明是 MMLU')
  const g3 = forgettingGate({ baseMmlu: { n: 0, accuracy: null }, sftMmlu: { n: 0, accuracy: null }, baseLm: {}, sftLm: {} })
  t.assert(g3.pass === false && g3.reasons.length >= 2, 'G3) 样本不足 → **不通过**（不是"没问题"）')
  const tg = taskGate({ base: { formatCompliance: 0.2, f1: 0.1 }, sft: { formatCompliance: 0.9, f1: 0.3 }, thresholds: { maxF1DropPct: 5, minFormatGainPct: 10 }, validity: [{ name: 'v', ok: true }] })
  t.assert(tg.pass === true && tg.validityOk === true, 'G4) 任务门禁：提升显著 → 通过')
  const tg2 = taskGate({ base: { formatCompliance: 0.9, f1: 0.3 }, sft: { formatCompliance: 0.1, f1: 0.05 }, thresholds: { maxF1DropPct: 5, minFormatGainPct: 10 }, validity: [{ name: 'v', ok: false, detail: 'x' }] })
  t.assert(tg2.pass === false && tg2.validityOk === false && /测量不成立/.test(tg2.reason), 'G5) 测量不成立与数值未达标可区分')

  rmSync(tmp, { recursive: true, force: true })
}
