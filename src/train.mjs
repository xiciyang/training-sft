/**
 * train —— 训练执行（**公共循环走 training-core**，本文件只做适配与编排）。
 *
 * 边界（报告要求第 6 项）：
 *   training-core 负责：循环、步数/墙钟/GPU 小时上限、checkpoint 编排、审计钩子接口
 *   training-sft 负责：SFT 数据整理与校验、SFT 配置、训练框架适配、指标抽取、门禁判定
 *
 * 两个**可执行的不变量校验**（不是靠注释保证）：
 *   ① 步号一致：每次 `step` 后断言 Python 回的 `stepIndex === state.step + 1`；
 *      两侧一旦失步（漏步/重放/断连重连）立刻失败，而不是"跑完了但少了几步没人发现"。
 *   ② GPU 小时：由 hooks 累加**实测墙钟**（单卡并发 1 ⇒ 墙钟 ≈ GPU 时间），
 *      喂给 `enforceLimits`；到 2 GPU 小时即由训练循环自己暂停（policy=pause-and-report）。
 */
import { createTrainingLoop, saveCheckpoint, listCheckpoints, BUDGET_DEFAULTS } from '@xiciyang/training-core'
import { ToolError, ERROR_CODES } from './version.mjs'
import { ContainerWorker, trainingMounts } from './worker.mjs'

/**
 * 构造 training-core 的插件对象（其 `step(batch, state)` 转发给容器 worker）。
 *
 * 关于 `batchSource` 的一点说明（**接口用法，不是接口变更**）：
 * 训练数据体在容器内（Python），逐微批把 token 搬过 IPC 是荒谬的。
 * 因此数据游标留在 Python 侧，`batchSource` 在这里承担的是**步预算/调度**角色，
 * 返回 `{index}`；`plugin.step` 消费"下一批"。步号一致性由不变量 ① 保证。
 */
export function makeSftPlugin({ worker, runDir, loraConfig, checkpointDir = null }) {
  return {
    id: 'sft-lora',
    algorithm: 'sft',
    async prepare({ state }) {
      const r = await worker.call('init', {
        config: {
          baseModel: '/models/' + loraConfig.baseModelDirName,
          trainFile: '/data/train.jsonl',
          lora: loraConfig.lora,
          lr: loraConfig.lr,
          weightDecay: loraConfig.weightDecay ?? 0.0,
          batchSize: loraConfig.batchSize,
          gradAccum: loraConfig.gradAccum,
          maxSeqLen: loraConfig.maxSeqLen,
          maxSteps: loraConfig.maxSteps,
          warmupSteps: loraConfig.warmupSteps ?? 10,
          seed: loraConfig.seed,
          resumeAdapter: loraConfig.resumeAdapter || undefined,
        },
      })
      return r
    },
    async step(batch, state) {
      const r = await worker.call('step')
      // 续训时容器内的步号从 offset 起算，Node 侧循环恒从 0 起算 —— 断言必须带上 offset，
      // 否则"续训"会被误判成"失步"。offset=0 时与原先完全等价。
      const expected = state.step + 1 + (loraConfig.resumeStepOffset ?? 0)
      if (r.stepIndex !== expected) {
        throw new ToolError(ERROR_CODES.E_INTERNAL, {
          hint: `训练两侧失步：容器内 stepIndex=${r.stepIndex}，期望 ${expected}（Node state.step=${state.step} + offset=${loraConfig.resumeStepOffset ?? 0}）。已停止，避免写出"看起来跑完了"的 checkpoint。`,
          details: { containerStep: r.stepIndex, nodeStep: state.step, offset: loraConfig.resumeStepOffset ?? 0 },
          recoverable: false,
        })
      }
      return { loss: r.loss, metrics: { gradNorm: r.metrics?.gradNorm, lr: r.lr, tokens: r.tokens, ms: r.ms, gpuMemGiB: r.gpuMemGiB, epoch: r.epoch } }
    },
    async finalize() {
      return { done: true }
    },
    async checkpointPayload({ state }) {
      // **与 training-core 的 checkpoint 同名同目录**：saveCheckpoint 会用
      // `step-${String(state.step).padStart(6,'0')}` 建目录，所以权重也写进那个目录的 adapter/ 下。
      // 这样 `keep` 触发的删除会连权重一起删 —— 否则会留下永远不被引用的孤儿权重目录
      // （实测踩到：keep=3 时 runDir 下仍堆了 4 个 ckpt-step-* 目录）。
      const name = `step-${String(state.step).padStart(6, '0')}`
      const dir = checkpointDir ? `${checkpointDir}/${name}` : `${runDir}/ckpt-${name}`
      const saved = await worker.call('save_adapter', { dir: `/out/${relOut(dir, runDir)}` }, { timeoutMs: 10 * 60 * 1000 })
      // ⚠️ **这里必须逐个字段列出来** —— payload 是**手写字段表**，不是透传。
      // 这正是"原则固化在规范里、没固化在代码里"的机制：
      // Python 侧早就算了指纹（`param_fingerprint()`），但本行**没有把它列进来**，
      // 于是它被**静默丢弃**，checkpoint payload 里永远看不到 —— 而没有任何地方会报错。
      // 实测（CI `fingerprint-coverage.mjs`）：四个训练插件四种 payload 形状，全靠各自手写。
      // 修法：把**指纹三元组**与**权重清单**显式列全；再被丢掉时，CI 与 Node 侧守卫都会红。
      return {
        adapterDir: saved.adapterDir, files: saved.files, step: state.step, epoch: state.epoch, bytes: saved.bytes,
        // 权重指纹三元组（与 pretrain 同名同义）：下游"空操作训练"防线依赖它
        fingerprint: saved.fingerprint ?? null,
        initFingerprint: saved.initFingerprint ?? null,
        weightsChanged: saved.weightsChanged ?? null,
        adapterSubdir: saved.adapterSubdir ?? null,
      }
    },
  }
}

function relOut(dir, runDir) {
  const marker = `${runDir}/`
  return dir.startsWith(marker) ? dir.slice(marker.length) : dir
}

/**
 * 跑一次训练。返回 training-core 的 run 结果 + 落盘的 checkpoint 清单。
 */
export async function runTraining({
  root, runId, runDir, image, modelDir, datasetDir, wikitextFile,
  loraConfig, limits, checkpoint, onEvent = () => {}, log = () => {},
  workerFactory = null, auditHooks = null, extraMounts = [],
}) {
  // `workerFactory` 是**测试接缝**：CI 里用一个模拟容器协议语义的假 worker 驱动循环，
  // 从而在不碰 GPU/容器/框架的前提下验证"循环接线 + 步号一致性 + 上限 + checkpoint 编排"。
  // 默认仍是真容器。这条接缝正是"报错要能报在 CI 里"的前提。
  const worker = workerFactory
    ? workerFactory({ runDir, loraConfig })
    : new ContainerWorker({
      image,
      mounts: trainingMounts({ modelDir, datasetDir, outDir: runDir, wikitextFile, extra: extraMounts }),
      log,
    }).start()

  const plugin = makeSftPlugin({ worker, runDir, loraConfig, checkpointDir: checkpoint?.dir ?? null })
  const gpuClock = { t0: Date.now(), acc: 0 }
  const events = []

  const hooks = {
    beforeStep: async (ctx2) => {
      const { state } = ctx2
      if (state.step === 0) { gpuClock.t0 = Date.now() }
      if (auditHooks?.beforeStep) await auditHooks.beforeStep(ctx2)
    },
    afterStep: async ({ state, out }) => {
      const dt = Date.now() - gpuClock.t0
      gpuClock.t0 = Date.now()
      gpuClock.acc += dt
      state.gpuHours = gpuClock.acc / 3_600_000
      events.push({ type: 'step', step: state.step, loss: out?.loss ?? null, ms: dt, gpuHours: Number(state.gpuHours.toFixed(4)) })
      onEvent({ type: 'step', step: state.step, loss: out?.loss ?? null, gpuHours: state.gpuHours })
      if (state.step % 25 === 0) log(`  step ${state.step} loss=${Number(out?.loss ?? NaN).toFixed(4)} gpuHours=${state.gpuHours.toFixed(3)}`)
      if (auditHooks?.afterStep) await auditHooks.afterStep({ state, out })
    },
  }

  let loop
  try {
    loop = createTrainingLoop({
      plugin,
      batchSource: (state) => (state.step < limits.maxSteps ? { index: state.step } : null),
      limits,
      checkpoint,
      onEvent: (e) => { events.push(e); onEvent(e) },
      hooks,
      epochs: 1,
    })
    const result = await loop.run()
    if (auditHooks?.onStop && result?.stop) await auditHooks.onStop({ state: result.state, stop: result.stop })
    const ckpts = await listCheckpoints({ dir: `${runDir}/checkpoints` })
    return { result, events, checkpoints: ckpts, worker }
  } finally {
    try { await worker.stop?.() } catch (e) { log(`worker 停止异常（忽略）：${e.message}`) }
  }
}

export { BUDGET_DEFAULTS, saveCheckpoint }
