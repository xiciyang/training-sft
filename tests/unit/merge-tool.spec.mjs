/**
 * merge-tool.spec —— **显式 merge 入口**（R-D4）回归。
 *
 * ## 背景（真实缺陷，不是假想）
 * `train` 留下 `trained`，而 `validate` 需要 `merged`，但**公开工具面从来没有 merge**。
 * 第四轮实测：运行中的旧代码在 `validate` 处直接抛
 * `E_RUN_NOT_FOUND: … 状态为 trained，不能直接 validate（需要先 merge）` ⇒ **闭环断在这里**，
 * 且当时连一条能落地的补救入口都没有（只能靠 `validate` 内部隐式 merge，而那是后加的）。
 *
 * ## 断言什么
 *   ① `merge` 是**导出函数**（工具面真的够得着）
 *   ② 状态计划：`trained ⇒ merge`；`merged/validated/gate-failed ⇒ idempotent`；其余 ⇒ `reject`
 *   ③ **反向对照**：merge 与 validate 对"坏状态"的拒绝集合**必须一致**
 *      —— 两个入口对同一个 run 的状态判断不能一个放行一个拒绝
 *   ④ 工具面确实注册了 `training_sft_merge`（否则"导出了但挂不上"照样等于没有）
 *
 * 纯函数，不依赖 GPU / 容器 / 网络 / 磁盘状态。
 */
import { ALL_TOOL_SPECS } from '../../src/tools.mjs'
import { merge, mergeStatusPlan, validateStatusPlan } from '../../src/core.mjs'

export const NAME = '显式 merge 入口（R-D4）'

const ALL = ['trained', 'merged', 'validated', 'gate-failed', 'failed', 'running', 'rolled-back', 'weird']

export async function run(t) {
  // ① 导出
  t.assert(typeof merge === 'function', `① merge 必须是导出函数（实得 ${typeof merge}）`)
  t.assert(typeof mergeStatusPlan === 'function', `① mergeStatusPlan 必须是导出函数（实得 ${typeof mergeStatusPlan}）`)

  // ② 状态计划（逐状态断言，不抽样）
  const expect = {
    trained: 'merge', merged: 'idempotent', validated: 'idempotent', 'gate-failed': 'idempotent',
    failed: 'reject', running: 'reject', 'rolled-back': 'reject', weird: 'reject',
  }
  for (const [st, want] of Object.entries(expect)) {
    const got = mergeStatusPlan(st)
    t.assert(got === want, `② mergeStatusPlan(${st}) 期望 ${want}，实得 ${got}`)
  }

  // ③ 反向对照：merge 与 validate 对"坏状态"的拒绝集合必须一致
  const mv = ALL.filter((st) => mergeStatusPlan(st) === 'reject').sort().join(',')
  const vv = ALL.filter((st) => validateStatusPlan(st) === 'reject').sort().join(',')
  t.assert(mv === vv, `③ 两个入口的拒绝集合必须一致（merge=${mv} / validate=${vv}）——不能一个放行一个拒绝`)

  // ④ 工具面确实挂上了（导出 ≠ 挂上）
  const names = ALL_TOOL_SPECS.map((sp) => sp.name)
  t.assert(names.includes('training_sft_merge'), `④ 工具面缺 training_sft_merge（实得 ${names.join(', ')}）`)
  const spec = ALL_TOOL_SPECS.find((sp) => sp.name === 'training_sft_merge')
  t.assert(!!(spec?.output?.schema && typeof spec?.output?.render === 'function'), '④ 必须有 output.schema + render（DSH 契约：缺 render 整条 entry 被拒）')
  t.assert(!!spec?.parameters?.properties?.confirm, '④ 必须带 confirm 参数（写操作要过确认闸）')
  t.assert(Array.isArray(spec?.parameters?.required) && spec.parameters.required.includes('runId'), '④ runId 必须是必填')
}
