# DATA_STEP0_REPORT —— 长任务 2 / 阶段 1（Step 0 数据检查）

> 执行：2026-09-29　性质：**只读**（未训练、未合成数据、未脱敏、未更换数据集）
> 原始证据：`inference-training/training-sft/out/step0-data.json` · `out/step0-pii-classified.json`
> 复跑：`node inference-training/training-sft/step0/step0-data.mjs`（+ `step0-pii-classify.mjs`）
>
> ## ⚠️ 结论（先说）
> **触发两条「必须暂停报告」条件，已暂停：**
> 1. **数据集 license 与题面严重不符** —— 题面写 `tatsu-lab/alpaca` 是 **Apache 2.0**；
>    实测（HF 卡片标签 + 卡片正文双证据）是 **CC BY-NC 4.0（非商用）**。
> 2. **发现 PII** —— 两个候选数据集都含**非占位**邮箱 / IPv4 / 国际电话。
>
> 按纪律：**不自行脱敏、不自行更换数据集、不自行合成**。裁决项见 §6。

---

## 1. 数据源探测范围（报告要求第 2 项的"缓存位置"）

| 项 | 路径 | 实测 |
|---|---|---|
| 候选 A 下载 | `/data/datasets/alpaca-nc/` | `data.parquet` 24,246,638 B + 派生 `alpaca_data.jsonl` 20,918,996 B |
| 候选 B 下载 | `/data/datasets/alpaca-cleaned/` | `alpaca_data_cleaned.json` 44,307,561 B |
| 评估集（wikitext-2 test） | `~/.ai-infra/state/datasets/Salesforce-wikitext__*__c{128,512}/test.jsonl` | 各 400 条（长任务 1 已制备） |
| 评估集（MMLU test） | `~/.cache/huggingface/datasets/cais___mmlu/<subject>/0.0.0/*/mmlu-test.arrow` | 57 科 / **14,042 条**，已抽出为 `/data/datasets/mmlu/mmlu-test.jsonl` |
| 基座模型 | `/data/models/Qwen2.5-0.5B-Instruct` | 954 MiB |

**探测的属性**：条数 · 字段 · 格式 · license（含来源） · PII（模式 + 字段 + 是否占位） ·
与各评估集的**内容哈希**交集 · 血缘（revision / 文件哈希 / 种子） · 去重规模。

## 2. 候选数据集事实

### A. `tatsu-lab/alpaca`

| 项 | 值 |
|---|---|
| 来源 URL | `[claimed]` `https://hf-mirror.com/datasets/tatsu-lab/alpaca` |
| revision | `dce01c9b08f87459cf36a430d809084718273017` |
| **license** | **`cc-by-nc-4.0` → class `restricted`（非商用）** |
| license 来源（双证据） | ① HF API `tags: ["license:cc-by-nc-4.0"]`；② **数据集卡片正文**原文：<br>"The dataset is available under the [Creative Commons NonCommercial (CC BY-NC 4.0)]" 且 "it is restricted to non-commercial uses and to uses that follow LLaMA's license agreement" |
| 条数 | **52,002** |
| 字段 | `instruction` / `input` / `output` / `text`（parquet 含 `text`，jsonl 派生时保留前三者） |
| 文件哈希 | parquet `sha256:06391b656a06fd3fb9d213160ef2398796c3b7f3dc75ef1e3ced30d461517073`；派生的 jsonl `sha256:8a752e792250510d2271085ce06eaef584ce03ecd94e47947fbc1942f949fd4b` |
| 去重（内容哈希） | 52,002 → **52,002**（0 重复） |

### B. `yahma/alpaca-cleaned`（**备选，permissive**）

| 项 | 值 |
|---|---|
| 来源 URL | `[claimed]` `https://hf-mirror.com/datasets/yahma/alpaca-cleaned` |
| revision | `12567cabf869d7c92e573c7c783905fc160e9639` |
| **license** | **`cc-by-4.0` → class `permissive`** |
| license 来源 | HF API `tags: ["license:cc-by-4.0"]` |
| 条数 | 51,760（去重后 **51,756**，4 条重复） |
| 字段 | `instruction` / `input` / `output` |
| 文件哈希 | `sha256:bd844b8247a0f543804b6ce0882b0aaec4bbf5e8d66167df6213a0f1e4fe878b` |

### 语料关系（重要）

**A × B 的内容哈希交集 = 1,832** —— B 是 A 的**清洗版本**（同一上游、同一指令风格）。
⇒ 二者**必须视为同一语料的两个版本**：只能用一个；混用即泄漏（与长任务 1 里 wikitext c128/c512 的情形同构）。

## 3. license 检查（D1）

| 候选 | SPDX | class | 商用 | 署名 | 判定 |
|---|---|---|---|---|---|
| A `tatsu-lab/alpaca` | `cc-by-nc-4.0` | **restricted** | **禁止**（NC） | 需 | ❌ 与题面"Apache 2.0"**不符** |
| B `yahma/alpaca-cleaned` | `cc-by-4.0` | permissive | 允许 | 需 | ✅ 合规 |
| 评估集 wikitext-2 test | `cc-by-sa-3.0` + `gfdl` | copyleft | —— | 需 | ⚠️ **仅作评估集**（不训练）；license 问题在长任务 1 已登记 |
| 评估集 MMLU test | `mit` | permissive | 允许 | 需 | ✅ |

**关于"Apache 2.0"的来源（推论）**：`[inferred]` 题面很可能混用了**代码仓**与**数据仓** ——
`tatsu-lab/stanford_alpaca`（代码）是 Apache-2.0，而**数据**明确标注 CC BY-NC 4.0。
这是本轮"探测结果与假设严重不符"的具体内容。

## 4. PII 检查（D2）

用 `@ai-infra/data-core` 的 `piiScan`（6 类模式：email / 大陆手机号 / 身份证 / IPv4 / 国际电话 / token 形态），
**全量扫描**（每条最多 2000 字符 × 3 字段）。命中原文**只在内存中**用于分类，落盘只有计数与哈希前缀。

| 候选 | 总命中 | 保留域名 | 占位本地部 | **非保留/非占位** | 命中字段（output / input / instruction） |
|---|---:|---:|---:|---:|---|
| A `tatsu-lab/alpaca` | 119 | 65 | 4 | **35**（email 28 / ipv4 6 / phone 1） | 83 / 31 / 5 |
| B `yahma/alpaca-cleaned` | 165 | 83 | 3 | **54**（email 37 / ipv4 16 / phone 1） | 131 / 29 / 5 |

分类口径（**可复核，不是主观判断**）：
- `reserved-domain`：`example.com|org|net|edu` / `test.*` / `localhost` / `domain.com` / `yourdomain.com` / `company.com` …
- `placeholder-localpart`：`user@` / `test@` / `john.doe@` / `admin@` / `info@` …
- `reserved-ip`：RFC1918（`10.` / `192.168.` / `172.16-31.`）+ `127.` + `0.` + `255.` + RFC5737 文档网段
- `other`：**以上都不是** —— 计入"非占位命中"

**结论**：两个候选都含**非占位**命中（A 35 条 / B 54 条）⇒ 命中「发现 PII」暂停条件。
**本轮未做任何脱敏**（纪律明确禁止"自行脱敏后继续"）。

> 补充：这两类命中里 `email/other` 可能包含机构邮箱（如 `support@<厂商>.com`）这一灰区，
> 但**本报告不做定性裁决** —— 只给出计数、字段名与哈希前缀，由用户裁决（见 §6）。

## 5. 测试集交集检查（D3，**内容哈希**，不是字符串）

判据：`@ai-infra/data-core` 的 `assertNoLeakage`（两侧**内容哈希**交集必须为空）。

| 对比 | 规模 | 交集 | 判定 |
|---|---|---:|---|
| A × wikitext-test(c128) | 52,002 × 400 | **0** | ✅ |
| A × wikitext-test(c512) | 52,002 × 400 | **0** | ✅ |
| B × wikitext-test(c128) | 51,760 × 400 | **0** | ✅ |
| B × wikitext-test(c512) | 51,760 × 400 | **0** | ✅ |
| **A × MMLU-test** | 52,002 × 14,042 | **3** | ⚠️ **基准污染** |
| **B × MMLU-test** | 51,760 × 14,042 | **5** | ⚠️ **基准污染** |
| A × B | 52,002 × 51,760 | **1,832** | ⚠️ 同源两版本，不可混用 |

**读法**：
- 与 wikitext-test 的交集为 **0** —— 指令数据与维基段落天然不同源，符合预期 `[实测]`。
- 与 **MMLU-test 有 3 / 5 条相撞**：MMLU 的少量题目**逐字**出现在这些指令数据里
  ⇒ 若把 MMLU 当评估集，**必须从训练集里剔除这几条**（内容哈希比对），否则 P0 #1 不成立。
- 这不是"无法解决"，而是**必须在切分阶段实现的一条硬规则**；本轮只报告，不擅自决定评估集构成。

## 6. 裁决项（**请用户裁决，本轮不自行处理**）

### 裁决 1：数据集 license（阻塞）

| 选项 | 事实 | 需要你决定的点 |
|---|---|---|
| **A. 维持 `tatsu-lab/alpaca`** | 实测 CC BY-NC 4.0（非商用）+ 卡片正文还提到需遵守 LLaMA license | 是否接受**非商用**限制（并流向微调后的模型权重） |
| **B. 换 `yahma/alpaca-cleaned`** | 实测 CC BY 4.0（permissive，仅需署名）；条数 51,760；**同样含非占位 PII**（54 条） | 是否同意更换（这是最接近题面意图的合规替代：同一上游的清洗版） |
| **C. 换其它 permissive 指令集** | 已实测 license 的候选：`OpenAssistant/oasst1`（apache-2.0）、`microsoft/orca-math-word-problems-200k`（mit）、`sahib2801/CodeAlpaca-20k`（cc-by-4.0）、`ise-uiuc/Magicoder-OSS-Instruct-75K`（mit）、`allenai/tulu-v2-sft-mixture`（odc-by） | 选哪一个（可指定，我会先跑同一套 license/PII/交集检查再动工） |
| **D. 自建/合成** | —— | **禁止**（纪律明确） |

### 裁决 2：PII 处置（阻塞）

| 选项 | 事实 | 需要你决定的点 |
|---|---|---|
| **A. 暂停不做** | —— | —— |
| **B. 由你指定处置规则，我执行** | 例如"剔除命中行" / "只保留非占位命中的行" / "接受占位示例" | 具体规则（我会按内容哈希执行并把剔除清单落盘，**不修改原始数据集文件**） |
| **C. 接受并记录** | 命中均为指令数据里的示例性内容 | 是否接受（并写入 model card 的数据来源声明） |
| **D. 我做脱敏后继续** | —— | **禁止**（纪律明确禁止"自行脱敏后继续"） |

### 裁决 3（非阻塞，但需你确认口径）：评估集构成

- 建议评估集：**alpaca 留出集**（任务指标）+ **wikitext-2 test**（语言建模/遗忘）+ **MMLU-test 子集**（通用能力/遗忘）。
- 其中 wikitext-2 test 是 **copyleft** 语料，**仅作评估、不训练**；MMLU 是 **permissive**。
- 训练集将**剔除**一切与上述评估集内容哈希相撞的行（这是 P0 #1 的实现方式）。
- 需要你确认：评估集是否就用这三者（或指定其它）。

## 7. 已知边界（如实记录）

- PII 扫描覆盖 6 类模式；**中文人名 / 地址 / 机构名未覆盖** `[未核]`。
- `email/other` 与 `ipv4/other` 中含"机构邮箱/公网 IP"这类灰区，本报告**只计数不裁决**。
- 交集检查用的是"行内容哈希"（`instruction\u0001input\u0001output` 与 wikitext 的 `t` 字段、
  MMLU 的 `question` 字段）；**语义级近义改写不在此判据内**（内容哈希只能判逐字相同）。
- MMLU 的 3/5 条相撞是"问题文本逐字相同"，**未**逐条判定是否同题不同问法。
- 候选 A 的 `text` 字段（parquet 自带）在派生 jsonl 时被丢弃；最终训练整理方式若与该字段有关，
  需回读 parquet（原文件保留在 `/data/datasets/alpaca-nc/data.parquet`）。
- 本轮**未**做 tokenizer 长度分布统计（属阶段 2 数据预处理的细节，待数据裁决后执行）。

## 8. 本轮已完成的动作与残留

| 项 | 状态 |
|---|---|
| 下载候选 A / B（只读检查用） | ✅ 落在 `/data/datasets/`，**未**写入 `~/.ai-infra/state/` |
| license / revision / 文件哈希 | ✅ |
| PII 全量扫描 + 分类 | ✅ |
| 测试集交集（wikitext / MMLU / 候选互交） | ✅ |
| 评估集抽取（MMLU test → jsonl） | ✅（派生文件，原始 arrow 未改） |
| **训练** | ❌ **未执行**（除一次 4.6 s 的框架链路冒烟，无 checkpoint 产出） |
| 脱敏 | ❌ **未执行**（纪律禁止） |
| 数据集更换 | ❌ **未执行**（纪律禁止） |
| 容器残留 | **0**（所有容器均 `--rm`） |
