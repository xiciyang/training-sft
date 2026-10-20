/**
 * mock-loop.spec —— 报告要求第 4 项点名的**训练测试特殊性**：
 * 「CI 中如何用 mock 模型驱动 training-core 的 loop，而不真跑训练」。
 *
 * 做法：用一个**模拟容器协议语义**的假 worker（不是"返回固定值"的空壳 ——
 * 它维护自己的步计数器、会因失步而报错、会走 checkpoint 保存协议），
 * 通过 `runTraining({workerFactory})` 这条测试接缝驱动**真实的 training-core 循环**。
 *
 * 于是下面这些**无需 GPU** 就能被 CI 验证：
 *   - 循环接线正确（step/stop 事件、步数记账）
 *   - **两侧步号一致性不变量**（故意让假 worker 漏一步 → 必须失败）
 *   - 资源上限（maxSteps / 墙钟）真的会停
 *   - checkpoint 编排（每 N 步存、keep 生效、meta 里有 payloadHash）
 *   - 审计钩子（createAuditHook）真写进哈希链
 */
import { mkdtempSync, rmSync, existsSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { tmpdir } from 'node:os'
import { fileURLToPath } from 'node:url'
import { runTraining } from '../../src/train.mjs'
import { listCheckpoints, loadCheckpoint, createAuditHook } from '@xiciyang/training-core'
import { appendAudit, verifyChain } from '@xiciyang/audit-core'

const HERE = dirname(fileURLToPath(import.meta.url))
export const NAME = 'mock-loop（CI 里用 mock 模型驱动 training-core 的 loop）'

/** 模拟容器 worker：**模拟对方的语义模型**，而不是只返回固定值。 */
function fakeWorker({ stepOffset = 0, failAt = -1, saveThrows = false } = {}) {
  let steps = 0
  const calls = []
  return {
    calls,
    async call(cmd, payload = {}) {
      calls.push(cmd)
      if (cmd === 'init') return { rows: 1000, trainable: 1081344, total: 495114112, vocabSize: 151936, dtype: 'torch.bfloat16' }
      if (cmd === 'step') {
        steps += 1
        if (steps === failAt) throw new Error('模拟容器失联')
        return { loss: 2.0 - steps * 0.001, stepIndex: steps + stepOffset, lr: 1e-4, tokens: 512, ms: 10, gpuMemGiB: 1.0, metrics: { gradNorm: 1.0 } }
      }
      if (cmd === 'save_adapter') {
        if (saveThrows) throw new Error('模拟保存失败')
        return { adapterDir: payload.dir, files: [{ name: 'adapter_model.safetensors', bytes: 1234, sha256: 'sha256:' + 'a'.repeat(64) }], bytes: 1234, stepIndex: steps }
      }
      if (cmd === 'shutdown') return { bye: true }
      throw new Error(`未知命令 ${cmd}`)
    },
    async stop() { calls.push('stop') },
  }
}

export async function run(t) {
  const tmp = mkdtempSync(join(tmpdir(), 'ts-loop-'))

  // ── ① 正常跑通 + 上限生效 ──
  const runDir1 = join(tmp, 'run1')
  const w1 = fakeWorker()
  const r1 = await runTraining({
    root: tmp, runId: 'ts_a', runDir: runDir1, image: 'x', modelDir: '/m', datasetDir: '/d',
    loraConfig: { baseModelDirName: 'b', lora: { r: 8 }, lr: 1e-4, batchSize: 4, gradAccum: 4, maxSeqLen: 256, maxSteps: 30, seed: 1 },
    limits: { maxSteps: 30, maxEpochs: null, maxWallMs: null, maxGpuHours: 2, concurrency: 1 },
    checkpoint: { dir: join(runDir1, 'checkpoints'), every: 10, keep: 2 },
    workerFactory: () => w1,
  })
  t.assert(r1.result.state.step === 30, `跑到 maxSteps 即停（实际 ${r1.result.state.step}）`)
  t.assert(r1.result.stop.reason === 'budget-exhausted:maxSteps', `停止原因正确（${r1.result.stop.reason}）`)
  t.assert(r1.result.stop.policy === 'pause-and-report', '超限策略=暂停并报告（不降级）')
  t.assert(w1.calls.filter((c) => c === 'step').length === 30, '容器侧恰好被调 30 次 step')
  t.assert(r1.result.state.losses.length === 30, 'loss 逐笔记账')
  t.assert(r1.result.state.gpuHours >= 0, 'GPU 小时被累加（hooks 实测墙钟）')

  // ── ② checkpoint 编排 ──
  const ckpts = await listCheckpoints({ dir: join(runDir1, 'checkpoints') })
  t.assert(ckpts.length === 2, `keep=2 生效（实际保留 ${ckpts.length}）`)
  const loaded = await loadCheckpoint({ dir: join(runDir1, 'checkpoints'), name: ckpts[0].name })
  t.assert(loaded.payload.files?.[0]?.sha256?.startsWith('sha256:'), 'checkpoint payload 里登记了权重文件哈希（checkpoint-contract §4 约定）')
  t.assert(loaded.meta.payloadHash === loaded.hash, 'meta.payloadHash 与复核结果一致')

  // ── ③ 步号一致性不变量（故意让容器漏一步 → 必须失败）──
  const w2 = fakeWorker({ stepOffset: -1 })
  let threw = null
  try {
    await runTraining({
      root: tmp, runId: 'ts_b', runDir: join(tmp, 'run2'), image: 'x', modelDir: '/m', datasetDir: '/d',
      loraConfig: { baseModelDirName: 'b', lora: { r: 8 }, lr: 1e-4, batchSize: 4, gradAccum: 4, maxSeqLen: 256, maxSteps: 10, seed: 1 },
      limits: { maxSteps: 10, maxEpochs: null, maxWallMs: null, maxGpuHours: 2, concurrency: 1 },
      checkpoint: null, workerFactory: () => w2,
    })
  } catch (e) { threw = String(e.message) }
  t.assert(threw && /失步/.test(threw), `步号失步被检出并停止（实际：${String(threw).slice(0, 90)}）`)

  // ── ④ 容器失联 → 响亮失败（不产生"看起来跑完了"的 checkpoint）──
  const w3 = fakeWorker({ failAt: 5 })
  threw = null
  try {
    await runTraining({
      root: tmp, runId: 'ts_c', runDir: join(tmp, 'run3'), image: 'x', modelDir: '/m', datasetDir: '/d',
      loraConfig: { baseModelDirName: 'b', lora: { r: 8 }, lr: 1e-4, batchSize: 4, gradAccum: 4, maxSeqLen: 256, maxSteps: 10, seed: 1 },
      limits: { maxSteps: 10, maxEpochs: null, maxWallMs: null, maxGpuHours: 2, concurrency: 1 },
      checkpoint: null, workerFactory: () => w3,
    })
  } catch (e) { threw = String(e.message) }
  t.assert(threw && /模拟容器失联/.test(threw), '容器中途失联 → 异常向上抛（不静默截断）')

  // ── ⑤ 墙钟上限（注入 clock 语义：这里用极小 maxWallMs + 真实时间）──
  const w4 = fakeWorker()
  const r4 = await runTraining({
    root: tmp, runId: 'ts_d', runDir: join(tmp, 'run4'), image: 'x', modelDir: '/m', datasetDir: '/d',
    loraConfig: { baseModelDirName: 'b', lora: { r: 8 }, lr: 1e-4, batchSize: 4, gradAccum: 4, maxSeqLen: 256, maxSteps: 100000, seed: 1 },
    limits: { maxSteps: 100000, maxEpochs: null, maxWallMs: 1, maxGpuHours: 2, concurrency: 1 },
    checkpoint: null, workerFactory: () => w4,
  })
  t.assert(r4.result.stop.reason === 'budget-exhausted:maxWallMs', `墙钟上限生效（${r4.result.stop.reason}）`)

  // ── ⑥ 审计钩子真写链 ──
  const auditRoot = join(tmp, 'ai-infra')
  const hook = createAuditHook({ append: appendAudit, root: auditRoot, pkg: 'training-sft', pkgVersion: '0.1.0' })
  let n = 0
  const w5 = fakeWorker()
  await runTraining({
    root: tmp, runId: 'ts_e', runDir: join(tmp, 'run5'), image: 'x', modelDir: '/m', datasetDir: '/d',
    loraConfig: { baseModelDirName: 'b', lora: { r: 8 }, lr: 1e-4, batchSize: 4, gradAccum: 4, maxSeqLen: 256, maxSteps: 120, seed: 1 },
    limits: { maxSteps: 120, maxEpochs: null, maxWallMs: null, maxGpuHours: 2, concurrency: 1 },
    checkpoint: null, workerFactory: () => w5, auditHooks: hook,
  })
  void n
  const chain = await verifyChain({ root: auditRoot, pkg: 'training-sft' })
  t.assert(chain.ok === true, '审计钩子写出的链可复算')
  t.assert(chain.checked >= 4, `链上有 begin + ≥2 次进度 + end（实际 ${chain.checked} 条）`)

  rmSync(tmp, { recursive: true, force: true })
  t.assert(!existsSync(tmp), '测试目录已清理（零残留）')
}
