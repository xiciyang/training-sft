/**
 * tools —— 6 个工具的规格（**MCP 与 DSH 两条通道共用同一份**，不存在行为分叉）。
 *
 * 命名空间纪律：`training_sft_*`，与既有三插件（`inference_serving_*` /
 * `inference_quantization_*` / `inference_tuning_*`）与 diagnose 的裸名**均无交集**
 * —— 由 `tests/unit/namespace.spec.mjs` 去兄弟包源码实时抓名并断言交集为空。
 *
 * DSH 工具契约（**踩过的坑**）：`ctx.tools.register` 要求 `output { schema, render }`
 * **同时**给出；缺 render 会让**整条 entry 被拒**（inference-tuning 就这样挂不上）。
 * 故这里每个工具都带 `output`，且 `src/index.mjs` 里用"像 DSH 一样挑刺"的 mock ctx 跑过。
 */
import { CONFIRM_PARAM, ROOT_PARAM, passthroughOutput } from './output.mjs'
import * as api from './core.mjs'

const P = (extra = {}) => ({ type: 'object', additionalProperties: false, properties: { ...extra, root: ROOT_PARAM }, ...extra })
void P

export const ALL_TOOL_SPECS = [
  {
    name: 'training_sft_prepare',
    description: 'Prepare and validate an SFT dataset: content-hash dedup, drop rows containing non-reserved PII (deletions only, never rewrite), drop rows that collide with or contain any evaluation set, deterministic split into train/held-out, and emit lineage. Optionally mixes in general-domain replay text (mix>0) from a second attributed source, replacing (not appending) train slots so the total row count is unchanged. Fails closed (E_DATASET_LEAK) when any content-hash intersection is non-empty, and independently re-checks containment-level leakage over the FINAL train set.',
    parameters: {
      type: 'object', additionalProperties: false, required: [],
      properties: {
        seed: { type: 'integer', description: '切分种子（同 seed 同输入 ⇒ 逐条相同的切分）。默认 20260929。' },
        train: { type: 'integer', description: '训练行数（≤1000）。默认 1000。' },
        heldout: { type: 'integer', description: '留出评估行数。默认 200。' },
        mix: { type: 'number', description: '通用语料回放比例 [0,1)：把 train 中该比例的槽位**替换**为通用语料续写行（总行数不变 ⇒ 受控变差）。默认 0（= 旧管道，逐字节同数据）。第二源 allenai/dolma（odc-by）。回放行只进 train，不进 heldout。' },
        root: ROOT_PARAM,
      },
    },
    output: passthroughOutput(),
    handler: (a) => api.prepare({ root: a?.root, seed: a?.seed, train: a?.train, heldout: a?.heldout, mix: a?.mix }),
  },
  {
    name: 'training_sft_train',
    description: 'Run LoRA SFT through the shared training loop (@xiciyang/training-core) inside a container sandbox, with hard budget enforcement (maxSteps ≤500, maxGpuHours ≤2, concurrency 1, params ≤0.5B). Over-budget pauses and reports — it never silently degrades. Requires a prepared dataset whose lineage proves zero leak.',
    parameters: {
      type: 'object', additionalProperties: false, required: ['datasetDir'],
      properties: {
        datasetDir: { type: 'string', description: 'training_sft_prepare 产出的数据集目录（需含 train.jsonl 与 allClear=true 的 lineage.json）。' },
        baseModel: { type: 'string', description: '基座模型本地目录。默认 <BASE_MODEL>（默认取环境变量 DSH_BASE_MODEL）。' },
        lora: { type: 'object', description: '{r, alpha, dropout, targetModules}。默认 r=8 alpha=16 dropout=0 target=[q,k,v,o]_proj。' },
        lr: { type: 'number', description: '学习率。默认 1e-4（warmup+cosine 调度）。' },
        batchSize: { type: 'integer', description: '每微批样本数。默认 4。' },
        gradAccum: { type: 'integer', description: '梯度累积步数（有效批 = batchSize×gradAccum）。默认 4。' },
        maxSeqLen: { type: 'integer', description: '最大序列长度。默认 256。' },
        maxSteps: { type: 'integer', description: '优化器步数上限（≤500）。默认 200。' },
        seed: { type: 'integer', description: '训练随机种子。默认 20260929。' },
        checkpointEvery: { type: 'integer', description: '每 N 步存一个可恢复 checkpoint。默认 50。' },
        confirm: CONFIRM_PARAM,
        root: ROOT_PARAM,
      },
    },
    output: passthroughOutput(),
    handler: (a) => api.train({ ...a, root: a?.root }),
  },
  {
    name: 'training_sft_merge',
    description: '把 SFT LoRA adapter 显式合并成完整 HF checkpoint（R-D4）。为什么需要它：train 留下 trained、validate 需要 merged，此前**公开工具面没有 merge** ⇒ 闭环只能靠 validate 隐式 merge。状态准入：trained ⇒ 真合并；merged/validated/gate-failed ⇒ **幂等**返回（不重做）；failed/running/rolled-back ⇒ **拒绝**。与 validate 内部 merge 走**同一条代码路径**，不存在行为分叉。',
    parameters: {
      type: 'object', additionalProperties: false, required: ['runId'],
      properties: {
        runId: { type: 'string', description: 'training_sft_train 返回的 run id。' },
        checkpointName: { type: 'string', description: '要合并的 checkpoint 名（如 step-000200）。缺省用该 run 的终态 checkpoint。' },
        confirm: CONFIRM_PARAM,
        root: ROOT_PARAM,
      },
    },
    output: passthroughOutput(),
    handler: (a) => api.merge({ ...a, root: a?.root }),
  },
  {
    name: 'training_sft_validate',
    description: 'Close the vertical loop: if the run is still in state trained, this merges the LoRA adapter into a full HF checkpoint first an explicit `training_sft_merge` tool also exists and validate/merge share the same code path (calling train then validate is still sufficient; call merge explicitly when you want the merge as a separate, auditable step), then deploys it through inference-serving (its loader + usability probe), measures the trained model AND the frozen baseline with the same protocol, then judges with @xiciyang/testing-core dual gate (format compliance + token F1) plus a derivable forgetting gate (MMLU accuracy delta and paired wikitext NLL delta, both bounded by z·SE at z=1.96). Deploys only its own leased endpoints and tears them down.',
    parameters: {
      type: 'object', additionalProperties: false, required: ['runId'],
      properties: {
        runId: { type: 'string', description: 'training_sft_train 返回的 run id。' },
        nTask: { type: 'integer', description: '任务指标样本数（held-out）。默认 100。' },
        nLm: { type: 'integer', description: '语言建模评估文本数。默认 200。' },
        nMmlu: { type: 'integer', description: 'MMLU 题目数。默认 300。' },
        thresholds: { type: 'object', description: '{maxF1DropPct, minFormatGainPct}。**必须显式给出**，不内置默认。' },
        confirm: CONFIRM_PARAM,
        root: ROOT_PARAM,
      },
    },
    output: passthroughOutput(),
    handler: (a) => api.validate({ ...a, root: a?.root }),
  },
  {
    name: 'training_sft_rollback',
    description: 'Roll a run back to a non-promoted state. Marks it rolled-back in the shared registry and (with purge=true) deletes ONLY that run directory under the training-sft run root — never a dataset, never the base model, never another run.',
    parameters: {
      type: 'object', additionalProperties: false, required: ['runId'],
      properties: { runId: { type: 'string' }, purge: { type: 'boolean', description: '是否删除该 run 的产出目录。默认 false。' }, confirm: CONFIRM_PARAM, root: ROOT_PARAM },
    },
    output: passthroughOutput(),
    handler: (a) => api.rollback({ root: a?.root, runId: a?.runId, purge: a?.purge === true }),
  },
  {
    name: 'training_sft_get_report',
    description: 'Read the SFT run registry and the audit hash chain. Without runId it lists recent runs; with runId it returns the full run record (checkpoint hashes, budget usage, gate results) plus the chain verification result.',
    parameters: { type: 'object', additionalProperties: false, required: [], properties: { runId: { type: 'string' }, limit: { type: 'integer' }, root: ROOT_PARAM } },
    output: passthroughOutput(),
    handler: (a) => api.getReport({ root: a?.root, runId: a?.runId ?? null, limit: a?.limit ?? 10 }),
  },
  {
    name: 'training_sft_health',
    description: 'Self-check for the training side: behavioural sandbox tier probe (process/container/vm), training toolchain image presence, GPU, base model availability, inference-serving MCP resolution, audit chain integrity, and the shared registry path. Read-only.',
    parameters: { type: 'object', additionalProperties: false, required: [], properties: { deep: { type: 'boolean' }, root: ROOT_PARAM } },
    output: passthroughOutput(),
    handler: (a) => api.health({ root: a?.root, deep: a?.deep === true }),
  },
]

export function findToolSpec(name) {
  return ALL_TOOL_SPECS.find((t) => t.name === name) || null
}
