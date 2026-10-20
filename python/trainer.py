#!/usr/bin/env python3
"""
trainer.py —— training-sft 的**计算侧**训练执行体（容器内运行的持久 worker）。

## 为什么是"持久 worker + 行分隔 JSON 协议"而不是"Python 自己跑完整训练"

题目纪律：「插件是薄适配层，**公共训练循环走 training-core**」。
而 `@ai-infra/training-core` 是 **Node** 库：训练循环、资源上限强制（`enforceLimits`）、
checkpoint 生命周期（`saveCheckpoint`）、审计钩子（`createAuditHook`）都在 Node 侧。

所以分工必须是：
  - **Node 侧（training-core）**：循环、步数/墙钟/GPU 小时上限、checkpoint 编排、审计链
  - **Python 侧（本文件）**：只做张量数学（前向/反向/优化器步/保存权重）

两者之间用**一对请求-应答**驱动每一步：
    Node: {"id":7,"cmd":"step"}            → Python: {"id":7,"ok":true,"loss":..,"stepIndex":6,...}

这样做的代价是每步一次 IPC（实测 <1 ms），换来的是**循环与上限只有一个真源**。
若让 Python 自己跑完整训练再回报，`training-core` 就沦为摆设 —— 那是本题明确禁止的"绕开共享库契约"。

## 步号一致性校验（可执行的不变量）

Python 维护自己的步计数器，并在每个 `step` 应答里回传 `stepIndex`。
Node 侧断言 `stepIndex === state.step` —— 一旦两侧不同步（漏步/重放）立刻失败，
而不是"跑完了但少了几步没人发现"。

## 协议（行分隔 JSON，stdout 只允许出现协议行）

请求 cmd：ping | init | step | save_adapter | merge_and_export | eval_lm | shutdown
"""
import json
import os
import random
import shutil
import sys
import time
import hashlib
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

# 静默 transformers 的进度条/警告：**stdout 必须只含协议行**
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402

PROTO_OUT = sys.stdout


def log(msg: str) -> None:
    """所有人类可读输出走 stderr —— stdout 是协议通道，不能被污染。"""
    sys.stderr.write(f"[trainer] {msg}\n")
    sys.stderr.flush()


def reply(obj: Dict[str, Any]) -> None:
    PROTO_OUT.write(json.dumps(obj, ensure_ascii=False) + "\n")
    PROTO_OUT.flush()


# ─────────────────────────────────────────────────────────── 数据与整理

PROMPT_TMPL = "### Instruction:\n{instruction}\n\n"
INPUT_TMPL = "### Input:\n{input}\n\n"
RESP_TMPL = "### Response:\n{output}"


def render_prompt(rec: Dict[str, str]) -> str:
    s = PROMPT_TMPL.format(instruction=str(rec.get("instruction", "")).strip())
    inp = str(rec.get("input", "") or "").strip()
    if inp:
        s += INPUT_TMPL.format(input=inp)
    return s


def render_full(rec: Dict[str, str]) -> str:
    return render_prompt(rec) + RESP_TMPL.format(output=str(rec.get("output", "")).strip())


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


class SFTDataset(Dataset):
    """SFT 数据整理（**属 training-sft，不属 training-core** —— 见 training-core 的"不包含"清单）。"""

    def __init__(self, rows: List[Dict[str, Any]], tok, max_len: int):
        self.rows = rows
        self.tok = tok
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict[str, List[int]]:
        r = self.rows[i]
        prompt = render_prompt(r)
        full = prompt + RESP_TMPL.format(output=str(r.get("output", "")).strip())
        ids = self.tok(full, add_special_tokens=False)["input_ids"]
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        ids = ids[: self.max_len]
        labels = list(ids)
        # 只在 response 上算 loss（prompt 段置 -100）—— 这是 SFT 的标准做法，
        # 也是"格式合规率"能被学出来的前提。
        for k in range(min(len(p_ids), len(ids))):
            labels[k] = -100
        if all(l == -100 for l in labels):  # 极端截断：整条都是 prompt → 丢掉
            labels[-1] = ids[-1]
        return {"input_ids": ids, "labels": labels}


def collate(batch: List[Dict[str, List[int]]], pad_id: int) -> Dict[str, torch.Tensor]:
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        pad = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * len(b["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device="cuda"),
        "labels": torch.tensor(labels, dtype=torch.long, device="cuda"),
        "attention_mask": torch.tensor(attn, dtype=torch.long, device="cuda"),
    }


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def param_fingerprint() -> str:
    """**权重指纹**：取前若干**可训练**参数的各 1024 个元素做哈希。

    为什么必须限定 `requires_grad`（与 pretrain 版本的关键差别）：
    SFT 是 **LoRA** —— 基座权重是**冻结**的。若照 pretrain 那样哈希"前 8 个具名参数"，
    哈希到的全是**冻结的基座** ⇒ `weightsChanged` **恒为 False** ⇒
    这个断言会变成一个**永远报错的假警报**（比没有更糟：它会训练出"狼来了"）。
    ⇒ 只哈希 `requires_grad=True` 的参数（LoRA 的 `lora_A`/`lora_B`）。

    为什么需要它（沿用 pretrain 的教训）：计数类断言全部为真，也可能权重一步没动。
    必须有一条**落在产物上**的绝对量断言：「相对起点，权重到底变了没有」。
    """
    h = hashlib.sha256()
    with torch.no_grad():
        ps = [(n, p) for n, p in S.model.named_parameters() if getattr(p, "requires_grad", False)]
        if not ps:
            ps = list(S.model.named_parameters())
        for name, p in ps[:8]:
            v = p.detach().float().flatten()[:1024].cpu().numpy().tobytes()
            h.update(name.encode("utf-8"))
            h.update(v)
    return "sha256:" + h.hexdigest()


def list_files(d: str) -> List[Dict[str, Any]]:
    """列出目录下**全部**文件（**递归**），name 用相对路径。

    ⚠️ 为什么必须递归（实测缺陷）：LoRA 的权重在 `d/adapter/adapter_model.safetensors` ——
    **子目录里**。旧实现用 `os.listdir` + `isfile`，**只列顶层** ⇒ checkpoint payload 的
    `files[]` 里**根本没有权重件**（只有 `optimizer.pt` / `trainer_state.json`）。
    后果：`assertArtifactChanged` 拿这份 payload 去比"权重变了没有"时，
    比的是**训练内部文件**，而不是**交付物**；`files[]` 也永远证明不了权重在场。
    """
    out = []
    for root, _dirs, names in os.walk(d):
        for name in names:
            p = os.path.join(root, name)
            if not os.path.isfile(p):
                continue
            out.append({"name": os.path.relpath(p, d), "bytes": os.path.getsize(p), "sha256": sha256_file(p)})
    out.sort(key=lambda x: x["name"])
    return out


# ─────────────────────────────────────────────────────────── 状态

class State:
    def __init__(self) -> None:
        self.model = None
        self.tok = None
        self.opt = None
        self.ds = None
        self.cfg: Dict[str, Any] = {}
        self.cursor = 0
        self.step_index = 0
        self.micro = 0
        self.batch: List[Dict[str, List[int]]] = []
        self.order: List[int] = []
        self.epoch = 0
        self.base_dir: Optional[str] = None
        self.adapters: List[str] = []
        self.init_fingerprint: Optional[str] = None


S = State()


def cmd_ping(_req: Dict[str, Any]) -> Dict[str, Any]:
    return {"pong": True, "stepIndex": S.step_index, "torch": torch.__version__, "cuda": torch.cuda.is_available()}


def cmd_init(req: Dict[str, Any]) -> Dict[str, Any]:
    cfg = req["config"]
    S.cfg = cfg
    seed = int(cfg.get("seed", 20260929))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    base = cfg["baseModel"]
    S.base_dir = base
    t0 = time.time()
    S.tok = AutoTokenizer.from_pretrained(base, trust_remote_code=False)
    if S.tok.pad_token is None:
        S.tok.pad_token = S.tok.eos_token
    S.model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager"
    )
    load_s = time.time() - t0

    lora_cfg = cfg.get("lora", {})
    # 恢复：若给了 adapter 目录，直接加载已有 LoRA 权重（checkpoint 可恢复）
    resume = cfg.get("resumeAdapter")
    if resume:
        from peft import PeftModel
        adir = os.path.join(resume, "adapter") if os.path.isdir(os.path.join(resume, "adapter")) else resume
        S.model = PeftModel.from_pretrained(S.model, adir, is_trainable=True)
    else:
        S.model = get_peft_model(S.model, LoraConfig(
            r=int(lora_cfg.get("r", 8)),
            lora_alpha=int(lora_cfg.get("alpha", 16)),
            lora_dropout=float(lora_cfg.get("dropout", 0.0)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(lora_cfg.get("targetModules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
        ))
    S.model.train()

    trainable = sum(p.numel() for p in S.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in S.model.parameters())
    S.opt = torch.optim.AdamW(
        [p for p in S.model.parameters() if p.requires_grad],
        lr=float(cfg.get("lr", 1e-4)),
        weight_decay=float(cfg.get("weightDecay", 0.0)),
    )

    rows = load_jsonl(cfg["trainFile"])
    S.ds = SFTDataset(rows, S.tok, int(cfg.get("maxSeqLen", 512)))
    S.order = list(range(len(rows)))
    rng = random.Random(seed)
    rng.shuffle(S.order)
    S.cursor = 0
    S.step_index = 0
    S.micro = 0
    S.batch = []
    S.epoch = 0
    if resume:
        ts_path = os.path.join(resume, "trainer_state.json")
        if os.path.exists(ts_path):
            with open(ts_path, "r", encoding="utf-8") as f:
                ts = json.load(f)
            S.step_index = int(ts.get("stepIndex", 0))
            S.epoch = int(ts.get("epoch", 0))
            S.cursor = int(ts.get("cursor", 0))
        opt_path = os.path.join(resume, "optimizer.pt")
        if os.path.exists(opt_path):
            S.opt.load_state_dict(torch.load(opt_path, map_location="cuda", weights_only=False))
    S.init_fingerprint = param_fingerprint()
    return {
        "loadedInS": round(load_s, 2),
        "rows": len(rows),
        "trainable": trainable,
        "total": total,
        "trainablePct": round(trainable / total * 100, 4),
        "padTokenId": S.tok.pad_token_id,
        "vocabSize": int(S.model.config.vocab_size),
        "dtype": str(next(S.model.parameters()).dtype),
        "resumed": bool(resume),
        "baseWeights": sha256_file(os.path.join(base, "model.safetensors")) if os.path.exists(os.path.join(base, "model.safetensors")) else None,
    }


def _next_micro_batch(bs: int) -> List[Dict[str, List[int]]]:
    out = []
    for _ in range(bs):
        if S.cursor >= len(S.order):
            S.epoch += 1
            rng = random.Random(int(S.cfg.get("seed", 20260929)) + S.epoch)
            rng.shuffle(S.order)
            S.cursor = 0
        out.append(S.ds[S.order[S.cursor]])
        S.cursor += 1
    return out


def _lr_at(step: int) -> float:
    """warmup + cosine 衰减（可推导，不是魔数）：warmup 步数与总步数都来自配置。"""
    base = float(S.cfg.get("lr", 1e-4))
    warm = int(S.cfg.get("warmupSteps", 10))
    total = max(int(S.cfg.get("maxSteps", 200)), 1)
    if step < warm:
        return base * (step + 1) / max(warm, 1)
    prog = min(max((step - warm) / max(total - warm, 1), 0.0), 1.0)
    return base * 0.5 * (1.0 + float(np.cos(np.pi * prog)))


def cmd_step(_req: Dict[str, Any]) -> Dict[str, Any]:
    """一次调用 = **一个优化器步**（内部做 gradAccum 次微批），
    这样 Node 侧 training-core 的 state.step 与"训练步"同义，
    「≤500 步」这条预算才是它字面上的意思（而不是微批数）。"""
    if S.model is None:
        raise RuntimeError("init 未调用")
    bs = int(S.cfg.get("batchSize", 4))
    ga = int(S.cfg.get("gradAccum", 8))
    t0 = time.time()
    lr = _lr_at(S.step_index)
    for g in S.opt.param_groups:
        g["lr"] = lr

    losses = []
    tokens = 0
    S.opt.zero_grad(set_to_none=True)
    for _ in range(ga):
        batch = collate(_next_micro_batch(bs), S.tok.pad_token_id)
        out = S.model(**batch)
        (out.loss / ga).backward()
        losses.append(float(out.loss.detach().float().item()))
        tokens += int(batch["attention_mask"].sum().item())

    gnorm = 0.0
    for p in S.model.parameters():
        if p.grad is not None:
            gnorm += float(p.grad.detach().float().norm()) ** 2
    gnorm = gnorm ** 0.5
    torch.nn.utils.clip_grad_norm_([p for p in S.model.parameters() if p.requires_grad], 1.0)
    S.opt.step()
    S.opt.zero_grad(set_to_none=True)
    S.step_index += 1

    return {
        "loss": float(np.mean(losses)),
        "lossLast": losses[-1],
        "stepIndex": S.step_index,      # Node 侧断言 == state.step + 1（防两侧失步）
        "optStep": S.step_index,
        "tokens": tokens,
        "epoch": S.epoch,
        "lr": lr,
        "ms": int((time.time() - t0) * 1000),
        "gpuMemGiB": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "metrics": {"gradNorm": round(gnorm, 6), "microBatches": ga},
    }


def cmd_save_adapter(req: Dict[str, Any]) -> Dict[str, Any]:
    """保存一个**可恢复**的 checkpoint：LoRA 权重 + 优化器状态 + 训练器状态。

    与 `checkpoint-contract.md` §4 的约定一致：权重文件由插件自己写，
    在 `meta.files[]` 里逐文件登记哈希；Node 侧只负责把这份清单写进 meta。
    """
    d = req["dir"]
    adir = os.path.join(d, "adapter")
    os.makedirs(adir, exist_ok=True)
    S.model.save_pretrained(adir, safe_serialization=True)
    torch.save(S.opt.state_dict(), os.path.join(d, "optimizer.pt"))
    with open(os.path.join(d, "trainer_state.json"), "w", encoding="utf-8") as f:
        json.dump({"stepIndex": S.step_index, "epoch": S.epoch, "cursor": S.cursor,
                   "seed": int(S.cfg.get("seed", 20260929)), "orderLen": len(S.order)}, f, indent=2)
    files = list_files(d)
    fp = param_fingerprint()
    S.adapters.append(d)
    return {"adapterDir": d, "adapterSubdir": adir, "files": files,
            "bytes": sum(f["bytes"] for f in files), "stepIndex": S.step_index,
            # 权重指纹三元组 —— 与 pretrain **同名同义**，使下游（CI/试点）能用同一套判据。
            "fingerprint": fp, "initFingerprint": S.init_fingerprint,
            "weightsChanged": fp != S.init_fingerprint}


def cmd_merge_and_export(req: Dict[str, Any]) -> Dict[str, Any]:
    out = req["outDir"]
    os.makedirs(out, exist_ok=True)
    merged = S.model.merge_and_unload()
    merged.save_pretrained(out, safe_serialization=True, max_shard_size="4GB")
    S.tok.save_pretrained(out)
    # serving 的 license 闸门读 README frontmatter 或 LICENSE 文件（见 PROJECT_CONTEXT §6）
    lic = req.get("license", {})
    src_license = os.path.join(S.base_dir, "LICENSE")
    if os.path.exists(src_license):
        shutil.copyfile(src_license, os.path.join(out, "LICENSE"))
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
        f.write("---\n")
        f.write(f"license: {lic.get('spdx', 'apache-2.0')}\n")
        f.write("base_model: Qwen/Qwen2.5-0.5B-Instruct\n")
        f.write("library_name: transformers\n")
        f.write("---\n\n")
        f.write("# training-sft LoRA-merged checkpoint\n\n")
        f.write(req.get("readmeNote", "") + "\n")
    import json as _json
    with open(os.path.join(out, "training-sft-provenance.json"), "w", encoding="utf-8") as f:
        _json.dump(req.get("provenance", {}), f, ensure_ascii=False, indent=2)
    files = list_files(out)
    weights = [f for f in files if f["name"].endswith(".safetensors")]
    wsum = hashlib.sha256(("\n".join(f["sha256"] for f in weights)).encode()).hexdigest()
    return {
        "outDir": out,
        "files": files,
        "weightsSha256": "sha256:" + wsum,
        "bytes": sum(f["bytes"] for f in files),
        "params": int(sum(p.numel() for p in merged.parameters())),
        "vocabSize": int(merged.config.vocab_size),
        "numHiddenLayers": int(merged.config.num_hidden_layers),
    }


def cmd_eval_lm(req: Dict[str, Any]) -> Dict[str, Any]:
    """语言建模评估：给定 jsonl（字段 t 或 text），算 mean NLL / 困惑度（逐 token）。"""
    path = req["file"]
    max_samples = int(req.get("maxSamples", 400))
    max_len = int(req.get("maxSeqLen", 512))
    rows = load_jsonl(path)[:max_samples]
    model = S.model
    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_tok = 0
    per_text = []
    with torch.no_grad():
        render_sft = bool(req.get("renderSft"))
        for r in rows:
            if render_sft:
                text = render_full(r)
            else:
                text = str(r.get("t") or r.get("text") or "")
            if not text.strip():
                continue
            ids = S.tok(text, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
            if len(ids) < 2:
                continue
            x = torch.tensor([ids], dtype=torch.long, device="cuda")
            out = model(input_ids=x)
            logits = out.logits[0, :-1].float()
            tgt = x[0, 1:]
            lp = torch.log_softmax(logits, dim=-1).gather(1, tgt.unsqueeze(1)).squeeze(1)
            n = lp.numel()
            total_nll += float(-lp.sum().item())
            total_tok += int(n)
            per_text.append(float(-lp.mean().item()))
    if was_training:
        model.train()
    mean_nll = total_nll / max(total_tok, 1)
    return {
        "meanNll": mean_nll,
        "ppl": float(np.exp(mean_nll)),
        "tokens": total_tok,
        "texts": len(per_text),
        "perTextNll": per_text,
    }


def main() -> None:
    log(f"worker up tid={os.getpid()} torch={torch.__version__} cuda={torch.cuda.is_available()}")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        cmd = req.get("cmd")
        rid = req.get("id")
        t0 = time.time()
        try:
            fn = {
                "ping": cmd_ping,
                "init": cmd_init,
                "step": cmd_step,
                "save_adapter": cmd_save_adapter,
                "merge_and_export": cmd_merge_and_export,
                "eval_lm": cmd_eval_lm,
            }.get(cmd)
            if cmd == "shutdown":
                reply({"id": rid, "ok": True, "bye": True})
                break
            if fn is None:
                raise ValueError(f"未知命令 {cmd}")
            res = fn(req)
            res.update({"id": rid, "ok": True, "handlerMs": int((time.time() - t0) * 1000)})
            reply(res)
        except Exception as e:  # 错误也要走协议，不能污染 stdout 也不能静默
            import traceback
            reply({"id": rid, "ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]})


if __name__ == "__main__":
    main()
