/**
 * code-version.spec —— **执行体身份**（G7 / 第五轮）回归。
 *
 * ## 背景（真实事故）
 * 第四轮发现**运行中的插件是 09-30 的快照**，而 CI 验证的是磁盘源码
 * ⇒ "CI 绿"只是关于磁盘的断言，**不是关于执行体的断言**。
 * 那次事故里所有既有守卫都是绿的：它们检查了判据、门槛、语料，
 * 唯独没检查"**这份测量是哪一份代码产出的**"。
 *
 * ## 断言什么
 *   ① 形状：`training-sft/src@<64hex>`
 *   ② **确定性**：两次调用相同（内容哈希，与 mtime 无关）
 *   ③ **载重**：改一个运行时会读到的字节 ⇒ 指纹**必须变**（否则它不构成身份）
 *   ④ **反向对照（有界性）**：改 `tests/` 下的文件 ⇒ 指纹**必须不变**
 *      —— 证明它**有范围**，不是"把整目录都哈希一遍"的装饰
 *
 * 纯函数 + 临时目录，不依赖 GPU / 容器 / 网络。
 */
import { readFileSync, writeFileSync, mkdirSync, rmSync, cpSync } from 'node:fs'
import { join } from 'node:path'
import { homedir, tmpdir } from 'node:os'
import { codeVersion } from '../../src/core.mjs'

export const NAME = '执行体身份（G7·第五轮）'

const SRC = join(homedir(), 'deepseek_harness', 'inference-training', 'training-sft', 'src')

export async function run(t) {
  const v = codeVersion()

  /* ── ① 形状 ── */
  t.assert(typeof v === 'string' && /^training-sft\/src@[0-9a-f]{64}$/.test(v), `① 形状须为 training-sft/src@<64hex>，实得 ${v}`)

  /* ── ② 确定性 ── */
  t.assert(codeVersion() === v, '② 两次调用必须相同（内容哈希，与 mtime 无关）')

  /* ── ③ 载重：改一个字节 ⇒ 必须变 ── */
  const tmp = join(tmpdir(), `cv-${Date.now()}`)
  try {
    cpSync(SRC, join(tmp, 'src'), { recursive: true })
    const before = codeVersion({ dir: join(tmp, 'src') })
    const target = join(tmp, 'src', 'evaluate.mjs')
    writeFileSync(target, `${readFileSync(target, 'utf8')}\n// RED-PROBE: 一个字节\n`)
    const after = codeVersion({ dir: join(tmp, 'src') })
    t.assert(before !== after, `③ 改一个运行时会读到的字节，指纹必须变（before=${before.slice(0, 24)}… after=${after.slice(0, 24)}…）`)

    /* ── ④ 反向对照：改 tests/ ⇒ 必须不变（有界性） ── */
    const tmp2 = join(tmpdir(), `cv2-${Date.now()}`)
    cpSync(SRC, join(tmp2, 'src'), { recursive: true })
    mkdirSync(join(tmp2, 'src', 'tests'), { recursive: true })
    const b2 = codeVersion({ dir: join(tmp2, 'src') })
    writeFileSync(join(tmp2, 'src', 'tests', 'x.spec.mjs'), 'export const NAME = "noise"\n')
    const a2 = codeVersion({ dir: join(tmp2, 'src') })
    t.assert(b2 === a2, '④ `tests/` 下的改动**不得**改变执行体身份（证明扫描有范围，不是装饰）')
    rmSync(tmp2, { recursive: true, force: true })
  } finally {
    rmSync(tmp, { recursive: true, force: true })
  }
}
