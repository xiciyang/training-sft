# @xiciyang/training-sft

用**真实 GPU** 对一个基座模型做 LoRA 监督微调（SFT），并把每一步变成可审计的产物：数据准备、训练、合并、验收、回退，全部留下哈希链与机器可读报告。

这是 DSH（DeepSeek Harness）插件包；也可以把它当作一个普通的 Node 库调用（`src/index.mjs` 的导出面）。

## 它解决的问题

「训练跑完了」和「训练真的改变了权重」是两件事。本包在训练前后对权重取指纹，**空操作训练（lr=0、数据为空、步数没生效）会被判为失败**，而不是记成一次成功的 run。精度侧同理：验收要求先过「测量是否有效」再谈「数值是否达标」——输出退化（困惑度等于词表大小）不会被当成 0% 损失。

## 安装

```bash
dsh plugin add github:xiciyang/training-sft
# 或从 npm（元数据）。注意：本包的运行依赖是 @xiciyang/* 系列库，已发布到 npm。
```

要求：

- Node.js >= 20（DSH 侧），`engines.dsh >= 0.2.0 < 0.3.0`
- 真实训练需要：NVIDIA GPU + 容器运行时 + 训练镜像；缺失时 `training_sft_health` 会报明是哪一层不可用，而不是笼统地「不健康」
- Python 侧框架（torch / transformers / peft / trl / accelerate）在 `optionalDependencies` 里显式声明，缺哪个报哪个

## 工具面（DSH）

| 工具 | 作用 |
|---|---|
| `training_sft_prepare` | 数据准备：去重、确定性切分、去泄漏（train/val/test 三组交集必须为空）、PII 剔除（只剔除不改写源文件）、血缘与内容哈希 |
| `training_sft_train` | 训练：预算闸门（步数/GPU 小时）、权重指纹守卫、checkpoint 保留、审计链 |
| `training_sft_merge` | 合并 LoRA 适配器到基座 |
| `training_sft_validate` | 验收：双重门禁（先判测量是否有效，再判数值变化），产出机器可读的 `eval-result` |
| `training_sft_rollback` | 回退：回到指定 run 的状态；purge 只允许删 run 目录，越界拒绝 |
| `training_sft_get_report` | 取人类可读报告 |
| `training_sft_health` | 环境自检：GPU / 容器 / 镜像 / 基座模型 / 依赖，逐项给出可用性 |

## 配置（不把本机布局写死）

| 环境变量 | 用途 |
|---|---|
| `DSH_BASE_MODEL` | 基座模型本地目录（只读）。未设置时健康检查会指出缺这一项，而不是猜一个路径 |
| `DSH_ALPACA_PATH` | 原始数据集文件路径（复现数据准备测试时需要） |
| `DSH_SHARD_PATH` | 预处理语料分片路径（引用时用） |
| `TRAINING_SFT_ROOT` | 运行目录根（默认 `~/.ai-infra`） |

硬件与后端都是可替换的：基座路径、镜像名、容器运行时、设备都不写死。没有 GPU 时，纯逻辑测试与契约测试仍然可以跑。

## 测试

```bash
node tests/run.mjs          # 单元 + 契约 + 缺陷回归
```

测试分两类：**纯逻辑/契约**（任何机器都应绿）与**需要本机历史产物的判据自检**。后者在产物不存在时显式 `SKIP` 并写明原因，不伪装成通过——汇总行会打印 skipped 数。CI 上看到 skip 属预期，看到 fail 才是问题。

## 安全边界

- 训练与合并都需要显式确认令牌（每类一次、24 小时内同类操作免重复确认；高风险操作每次都要）
- 预算超限直接拒绝，不「先跑再说」
- 回退只作用于本插件自己的 run 目录

## 许可

BSD-3-Clause。见 `LICENSE`。
