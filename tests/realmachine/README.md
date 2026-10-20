# tests/realmachine —— 真机测试（**不进 CI**）

与 inference-diagnose 同一条纪律：`tests/run.mjs` 只递归 `tests/unit/`，
本目录**天然不被收集** —— 比"打标记跳过"更强（忘了带标记也不会把真机测试拖进 CI）。

| 脚本 | 需要什么 | 触发方式 |
|---|---|---|
| `docker-worker.spec.mjs` | docker + `training-sft/toolchain:0.1.0` | 手动：`node tests/realmachine/docker-worker.spec.mjs` |
| `vertical-loop.mjs`（由 core 的 validate 承担） | GPU + serving + 模型 | `training_sft_validate`（真机） |
