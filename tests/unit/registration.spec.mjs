/**
 * registration.spec —— P2 #18「注册路径被测试覆盖」+ #19「工具名不与既有命名空间冲突」。
 *
 * 判据来自本机实测教训：**注入返回 host ✓ ≠ 工具注册成功**。
 * mock ctx 的 `register` 必须**像 DSH 一样挑刺**（缺 output{schema,render} 就抛），
 * 然后**真调一次**只读工具 —— 两件事都做才算覆盖。
 */
import { mkdtempSync, rmSync, readFileSync, existsSync, readdirSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { tmpdir } from 'node:os'
import { fileURLToPath } from 'node:url'
import { apply, TOOL_NAMES, TOOL_RISK, existingNamespaces } from '../../src/index.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const PKG = join(HERE, '..', '..')
const WORKSPACE = '<workspace>'
export const NAME = 'registration（注册路径 / 契约 / 命名空间）'

function strictCtx(sink) {
  const routes = []
  return {
    routes,
    logger: { info() {}, warn() {}, error() {}, debug() {} },
    effect(fn) { const d = fn(); return d },
    get(n) { return n === 'webServer' ? { route(m, p, h) { routes.push({ m, p, h }); return () => {} } } : undefined },
    on() { return () => {} },
    tools: {
      register(tool) {
        if (!tool.output || !tool.output.schema || typeof tool.output.render !== 'function') {
          throw new Error(`tool "${tool.name}" must declare output { schema, render, presentationMeta? }`)
        }
        if (!tool.parameters || tool.parameters.type !== 'object') throw new Error(`tool "${tool.name}" 缺 parameters`)
        if (typeof tool.description !== 'string' || tool.description.length < 40) throw new Error(`tool "${tool.name}" description 太短`)
        sink.push(tool)
        return () => {}
      },
    },
  }
}

export async function run(t) {
  const tmp = mkdtempSync(join(tmpdir(), 'ts-reg-'))
  const prev = process.env.TRAINING_SFT_ROOT
  process.env.TRAINING_SFT_ROOT = tmp
  const registered = []
  const ctx = strictCtx(registered)
  let threw = null
  try { apply(ctx) } catch (e) { threw = String(e.message) }
  t.assert(threw === null, `apply() 在严格 mock ctx 下不抛错（实际：${threw}）`)
  t.assert(registered.length === 7, `注册了 7 个工具（实际 ${registered.length}）`)
  t.assert(JSON.stringify(registered.map((x) => x.name)) === JSON.stringify([...TOOL_NAMES]), '注册顺序与 TOOL_NAMES 一致')

  for (const tool of registered) {
    t.assert(tool.name.startsWith('training_sft_'), `${tool.name} 带 training_sft_ 前缀`)
    t.assert(typeof tool.execute === 'function', `${tool.name} 有 execute`)
    t.assert(tool.output && tool.output.schema && typeof tool.output.render === 'function', `${tool.name} 声明 output{schema,render}`)
  }

  // 真调一次（只读、无副作用）——"注册成功 ≠ 能用"
  const healthTool = registered.find((x) => x.name === 'training_sft_health')
  const out = await healthTool.execute({})
  t.assert(out && typeof out === 'object', '真调 training_sft_health 返回对象')
  t.assert(out.ok === true || out.ok === false, '返回值是标准信封（ok 字段）')
  t.assert(out.versions?.pkg === '0.1.0', `信封带 versions（实际 ${JSON.stringify(out.versions)}）`)
  t.assert(out.governance?.audit?.ok === true, '返回信封带 governance.audit（只读调用也留痕）')

  const reportTool = registered.find((x) => x.name === 'training_sft_get_report')
  const rep = await reportTool.execute({})
  t.assert(rep.ok === true && Array.isArray(rep.data.runs), '真调 training_sft_get_report 返回 runs 数组')
  t.assert(rep.data.audit && typeof rep.data.audit.ok === 'boolean', 'get_report 附审计链校验结果')

  // 健康路由
  t.assert(ctx.routes.some((r) => r.p === '/__training_sft/health'), '注册了 /__training_sft/health 路由')

  // 命名空间：去兄弟包源码实时抓名，交集必须为空
  const siblings = existingNamespaces(WORKSPACE)
  t.assert(siblings.length >= 30, `抓到兄弟包工具名（${siblings.length} 个）`)
  const overlap = TOOL_NAMES.filter((n) => siblings.includes(n))
  t.assert(overlap.length === 0, `与既有命名空间无交集（实际重叠：${overlap.join(', ') || 'none'}）`)
  t.assert(siblings.every((n) => n.startsWith('inference_')), '兄弟包一律 inference_ 前缀')

  // 风险声明与工具面双向一致
  t.assert(Object.keys(TOOL_RISK).length === TOOL_NAMES.length, '风险声明数与工具数一致')
  for (const n of TOOL_NAMES) t.assert(Boolean(TOOL_RISK[n]), `${n} 有风险声明`)
  const trainR = TOOL_RISK.training_sft_train
  t.assert(trainR.risk === 'write' && trainR.sandbox === 'container', 'train 是 write + container 沙箱')
  t.assert(trainR.confirmation === 'required', 'train 需要确认令牌')
  t.assert(TOOL_RISK.training_sft_health.risk === 'read', 'health 是 read 档')

  // manifest.yaml 与代码逐条一致（声明漂移是这类机制唯一真正的失败模式）
  const mf = readFileSync(join(PKG, 'manifest.yaml'), 'utf8')
  for (const n of TOOL_NAMES) t.assert(mf.includes(`tool: ${n}`), `manifest.yaml 声明了 ${n}`)
  for (const [n, d] of Object.entries(TOOL_RISK)) {
    const block = mf.split(`tool: ${n}`)[1]?.split('- id:')[0] ?? ''
    t.assert(block.includes(`risk: ${d.risk}`), `manifest ${n} risk 一致`)
    t.assert(block.includes(`sandbox: ${d.sandbox}`), `manifest ${n} sandbox 一致`)
    t.assert(block.includes(`confirmation: ${d.confirmation}`), `manifest ${n} confirmation 一致`)
  }

  // 包元数据：训练框架必须**显式**声明为可选依赖组
  const pkg = JSON.parse(readFileSync(join(PKG, 'package.json'), 'utf8'))
  t.assert(pkg.name === '@xiciyang/training-sft', 'package.json name')
  t.assert(pkg.dsh?.bundle?.patch, '声明 dsh.bundle.patch（否则 DSH 不会装载）')
  for (const f of ['torch', 'transformers', 'peft', 'trl', 'accelerate']) {
    t.assert(Boolean(pkg.optionalDependencies?.[f]), `optionalDependencies 显式声明 ${f}`)
  }
  for (const l of ['testing-core', 'data-core', 'state-core', 'audit-core', 'sandbox-core', 'training-core']) {
    t.assert(Boolean(pkg.dependencies?.[`@xiciyang/${l}`]), `声明共享库依赖 @xiciyang/${l}`)
  }

  if (prev === undefined) delete process.env.TRAINING_SFT_ROOT; else process.env.TRAINING_SFT_ROOT = prev
  rmSync(tmp, { recursive: true, force: true })
}
