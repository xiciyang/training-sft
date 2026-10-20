/**
 * closed-loop.spec —— **缺陷 13 回归测试**：`train` 的终态必须能被 `validate` 接受。
 *
 * 背景（实测）：`train` 把 run 留在 `trained`，而 `validate` 旧实现要求 `merged`，
 * 且公开工具面**没有** merge 入口 ⇒ 通过 MCP/DSH 调用 `prepare → train → validate`
 * **必然在 validate 处断掉**（实测 `E_RUN_NOT_FOUND: ... 不能直接 validate（需要先 merge）`）。
 * 这等于"闭环在工具面之外成立、在工具面之内不成立"。
 *
 * 本 spec 全部**不依赖 GPU / 容器 / 网络**：只断言状态机这一纯函数。
 * 它同时是"闸门**没有**被删掉"的证据（非法状态必须仍被拒）。
 */
import * as api from '../../src/core.mjs'

export const NAME = '闭环状态机（缺陷 13 回归）'

// `train` 成功返回的终态（见 core.mjs 中 train 的 return）
const TRAIN_TERMINAL = 'trained'
const FLOW_THROUGH = ['merged', 'validated', 'gate-failed']
const MUST_REJECT = ['failed', 'running', 'rolled-back', null, undefined, '', 'weird-state']

export async function run(t) {
  // ① 核心不变量：train 的终态**不得**被 validate 拒绝（否则闭环在工具面内不可达）
  t.assert(
    api.validateStatusPlan(TRAIN_TERMINAL) !== 'reject',
    `train 的终态 '${TRAIN_TERMINAL}' 被 validate 拒绝 ⇒ 缺陷 13 复发（闭环在公开工具面内不可达）`,
  )
  // ② 它应该被识别为"需要先自动完成中间步骤"，而不是悄悄直接跑
  t.assert(
    api.validateStatusPlan(TRAIN_TERMINAL) === 'auto-merge',
    `train 的终态 '${TRAIN_TERMINAL}' 应触发 auto-merge，实得 '${api.validateStatusPlan(TRAIN_TERMINAL)}'`,
  )
  // ③ 已合并 / 已判定 / 判失败可重判 ⇒ 直接放行（不得被要求再次 merge）
  for (const s of FLOW_THROUGH) {
    t.assert(api.validateStatusPlan(s) === 'proceed', `状态 '${s}' 应直接放行，实得 '${api.validateStatusPlan(s)}'`)
  }
  // ④ 闸门**没有被删掉**：非法状态必须仍然被拒
  for (const s of MUST_REJECT) {
    t.assert(api.validateStatusPlan(s) === 'reject', `非法状态 ${JSON.stringify(s)} 必须被拒，实得 '${api.validateStatusPlan(s)}'`)
  }
  // ⑤ 'proceed' 与 'auto-merge' 都不能是 'reject'（即：所有合法路径都不得被误拒）
  for (const s of [TRAIN_TERMINAL, ...FLOW_THROUGH]) {
    t.assert(api.validateStatusPlan(s) !== 'reject', `合法状态 '${s}' 被误拒`)
  }
}
