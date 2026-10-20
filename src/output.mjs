/**
 * output —— DSH 工具契约的 `output { schema, render }` 规格。
 *
 * **这是踩过的坑的落点**：`ctx.tools.register(tool)` 要求 output **同时**给出
 * `schema` 与 `render`；只给 schema 或不给 output 会让 **整条 entry 被拒**
 * （inference-tuning 曾因此完全挂不上）。故所有工具共用这里的构造器，不手写。
 */
export const ROOT_PARAM = { type: 'string', description: '共享状态根目录。默认 ~/.ai-infra，可用 TRAINING_SFT_ROOT 覆盖。' }

export const CONFIRM_PARAM = { type: 'string', description: '确认令牌。首次调用若需确认会返回 E_CONFIRMATION_REQUIRED 并在 details.confirmToken 给出该值；把原值回传即可继续。' }

export function passthroughOutput() {
  return {
    schema: { type: 'object', additionalProperties: true },
    render(_args, value) {
      return [{ type: 'text', text: JSON.stringify(value ?? null, null, 2) }]
    },
  }
}
