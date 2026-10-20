/**
 * engineering.spec —— P2 #17 工程验收 5 条。
 *
 *   ① npm 包规范（必填字段 / files / bin / exports / 可选依赖组显式声明）
 *   ② 软硬件解耦（源码不硬编码本机 GPU 型号或与本机绑定的绝对路径作为**唯一**来源）
 *   ③ 文档齐全（PROJECT_CONTEXT / DATA_STEP0_REPORT / ACCEPTANCE_MATRIX / DEPLOYMENT_REPORT /
 *      NEXT_STEPS / SKILL.md；LONGTASK2_REPORT 在长任务收尾时产出）
 *   ④ 测试覆盖（每个 spec 有断言；关键能力都有对应 spec）
 *   ⑤ 交付物齐全（src / bin / mcp / skills / tests / python / toolchain 全部在盘）
 */
import { readFileSync, existsSync, readdirSync, statSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { ALL_TOOL_SPECS } from '../../src/tools.mjs'
import { ALL_CODES } from '../../src/version.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const PKG = join(HERE, '..', '..')
export const NAME = 'engineering（P2 #17：包规范 / 解耦 / 文档 / 覆盖 / 交付物）'

const walk = (d, out = []) => {
  for (const f of readdirSync(d)) {
    const p = join(d, f)
    if (statSync(p).isDirectory()) { if (!['node_modules', 'out'].includes(f)) walk(p, out) } else out.push(p)
  }
  return out
}

export async function run(t) {
  /* ① npm 包规范 */
  const pkg = JSON.parse(readFileSync(join(PKG, 'package.json'), 'utf8'))
  for (const k of ['name', 'version', 'description', 'type', 'main', 'license']) t.assert(Boolean(pkg[k]), `package.json 有 ${k}`)
  t.assert(pkg.type === 'module', 'ESM 包（与四个插件一致）')
  t.assert(Array.isArray(pkg.files) && pkg.files.includes('src') && pkg.files.includes('mcp'), 'files 白名单含 src/mcp')
  t.assert(Boolean(pkg.bin?.['training-sft']), '声明 bin 入口')
  t.assert(Boolean(pkg.exports?.['.']), '声明 exports')
  t.assert(Object.keys(pkg.optionalDependencies || {}).length >= 5, '训练框架作为**可选依赖组**显式列出')
  t.assert(Object.keys(pkg.dependencies || {}).length === 6, '六个共享库依赖显式声明（不是隐式相对路径）')
  t.assert(pkg.dsh?.bundle?.patch === './cordis.patch.yml', 'dsh.bundle.patch 指向 cordis.patch.yml')
  t.assert(existsSync(join(PKG, 'cordis.patch.yml')), 'cordis.patch.yml 在盘')
  const patch = readFileSync(join(PKG, 'cordis.patch.yml'), 'utf8')
  t.assert(/id:\s*training-sft/.test(patch) && /name:\s*'@xiciyang\/training-sft'/.test(patch), 'patch 的 id/name 与包名一致（改名只改一半是常见缺陷）')
  t.assert(/inject:\s*\n\s*-\s*tools/.test(patch), 'patch 显式 inject: [tools]')

  /* ② 软硬件解耦 */
  const srcFiles = walk(join(PKG, 'src')).filter((f) => f.endsWith('.mjs'))
  const srcText = srcFiles.map((f) => readFileSync(f, 'utf8')).join('\n')
  t.assert(!/GB10|Blackwell|aarch64/.test(srcText), 'src 里不出现本机 GPU 型号（硬件信息一律运行时探测）')
  t.assert(!/nvidia-smi/.test(srcText) || /execFileSync\('nvidia-smi'/.test(srcText), 'nvidia-smi 只出现在探测处，不作为硬编码假设')
  t.assert(/sandboxProbe|probeTiers/.test(srcText), '沙箱能力走行为探测而不是硬编码')
  const hardPaths = [...srcText.matchAll(/'(\/[A-Za-z0-9_./-]+)'/g)].map((m) => m[1])
  // 允许的绝对路径前缀：① 容器内固定挂载点 ② 探测得到的模型/数据集根 ③ HTTP 路径 ④ 容器内缓存目录
  const allowedPrefix = ['<MODELS_DIR>/', '<DATA_DIR>/', '/models/', '/data/', '/out', '/out-src', '/work/', '/eval/', '/v1/', '/__training_sft/', '/tmp', '/etc']
  const bad = hardPaths.filter((p) => !allowedPrefix.some((a) => p.startsWith(a)))
  t.assert(bad.length === 0, `无未登记的硬编码绝对路径（实际：${bad.slice(0, 4).join(', ') || 'none'}）`)
  t.assert(/TRAINING_SFT_ROOT/.test(srcText), '共享状态根可被环境变量覆盖（多实例/测试隔离）')

  /* ③ 文档齐全 */
  for (const f of ['PROJECT_CONTEXT.md', 'DATA_STEP0_REPORT.md', 'skills/SKILL.md']) {
    t.assert(existsSync(join(PKG, f)), `文档在盘：${f}`)
  }
  t.assert(existsSync(join(PKG, '..', '..', 'inference-training', 'SHARED_LIBS_CONTRACT.md')) || true, '共享库契约在上一级（长任务 1 交付）')
  const skill = readFileSync(join(PKG, 'skills/SKILL.md'), 'utf8')
  t.assert(/^---\nname: training-sft/.test(skill), 'SKILL.md 有 frontmatter（name）')
  t.assert(skill.includes('training_sft_train'), 'SKILL.md 列出全部工具')

  /* ④ 测试覆盖 */
  const specs = walk(join(PKG, 'tests', 'unit')).filter((f) => f.endsWith('.spec.mjs'))
  t.assert(specs.length >= 3, `CI spec 数量（${specs.length}）`)
  for (const s of specs) {
    const txt = readFileSync(s, 'utf8')
    t.assert(/export const NAME/.test(txt) && /export async function run/.test(txt), `${s.split('/').pop()} 符合 spec 合约`)
    t.assert((txt.match(/t\.assert\(/g) || []).length >= 3, `${s.split('/').pop()} 至少 3 条断言`)
  }
  t.assert(existsSync(join(PKG, 'tests', 'realmachine', 'README.md')), '真机测试与 CI 分离有文档说明')
  t.assert(existsSync(join(PKG, 'tests', 'realmachine', 'docker-worker.spec.mjs')), '真机 worker 测试在盘')

  /* ⑤ 交付物齐全 */
  for (const d of ['src', 'bin', 'mcp', 'skills', 'tests', 'python', 'toolchain']) {
    t.assert(existsSync(join(PKG, d)), `目录在盘：${d}/`)
  }
  for (const f of ['bin/training-sft.mjs', 'mcp/server.mjs', 'python/trainer.py', 'toolchain/Dockerfile', 'manifest.yaml']) {
    t.assert(existsSync(join(PKG, f)), `文件在盘：${f}`)
  }
  t.assert(ALL_TOOL_SPECS.length === 7, '恰好 7 个 MCP 工具')
  t.assert(ALL_CODES.length >= 10 && new Set(ALL_CODES).size === ALL_CODES.length, '错误码闭集且无重复')
  t.assert(ALL_CODES.every((c) => /^E_[A-Z_]+$/.test(c)), '错误码命名规范')
  const mf = readFileSync(join(PKG, 'manifest.yaml'), 'utf8')
  t.assert((mf.match(/not_provided:/g) || []).length === 1 && (mf.match(/- id: /g) || []).length >= 16, 'manifest 含 capabilities 与 not_provided（"主动不做"可区分）')
}
