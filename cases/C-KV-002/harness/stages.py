"""C-KV-002 pipeline: long-context retrieval with a compressed key-value cache at three memory budgets, on
Qwen2.5-1.5B-Instruct (evaluator and self-check use this same file).

The evaluator does not train. One stage, "evaluate":
  timed   import cache.py (no model is loaded by your code)                                        (killed at 120 s)
  finish  the evaluator loads the model (bf16, models_dir); for each budget b in BUDGETS (bits per cached element)
          and each held-out prompt: cache = make_cache(b, config); prefill the prompt with past_key_values=cache;
          the cache's stored bits per element = 8 x (bytes of every tensor reachable from the cache object) /
          (tokens x layers x key-value heads x head dimension x 2); then 32 greedy decoding steps with the same cache;
          the answer is scored with ruler_tasks.score. The same prompts are also run with an uncompressed cache
          (transformers DynamicCache) for reference.                                               (all ≤ 2,700 s)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from abench_stage import Context, Stage
import ruler_tasks

BUDGETS = (4.5, 2.5, 2.0)
MODEL, MAX_NEW = "Qwen2.5-1.5B-Instruct", 32
LENGTHS = (4096, 8192, 16384)
PARAMS = {"budgets_bits": list(BUDGETS), "model": MODEL, "max_new_tokens": MAX_NEW, "lengths": list(LENGTHS),
          "tasks": list(ruler_tasks.TASKS)}
HELD_OUT_KEYS = ["tokens", "offsets", "task", "length"]
PIPELINE = [{"stage": "evaluate", "timeout": 120 + 2700, "finish_cap": 2700, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": True}]


def tensor_bytes(obj, seen=None) -> int:
    import torch
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(tensor_bytes(v, seen) for v in obj.values())
    if isinstance(obj, (list, tuple, set)):
        return sum(tensor_bytes(v, seen) for v in obj)
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return sum(tensor_bytes(v, seen) for v in vars(obj).values())
    return 0


def _timed(ctx: Context, _):
    return ctx.import_submission("cache.py", required=("make_cache",))


def run_prompt(model, tok, ids, make, device):
    import torch
    cache = make()
    x = torch.tensor([ids], device=device)
    out = []
    with torch.no_grad():
        logits = model(input_ids=x, past_key_values=cache, use_cache=True).logits[0, -1]
        stored = tensor_bytes(cache)
        for _ in range(MAX_NEW):
            t = int(logits.argmax())
            out.append(t)
            if t in (151643, 151645):
                break
            logits = model(input_ids=torch.tensor([[t]], device=device), past_key_values=cache,
                           use_cache=True).logits[0, -1]
    return tok.decode(out, skip_special_tokens=True), stored


def _finish(ctx: Context, mod):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
    path = str(ctx.models / MODEL)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, local_files_only=True,
                                                 attn_implementation="sdpa").to(ctx.device).eval()
    cfg = model.config
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    per_token = cfg.num_hidden_layers * cfg.num_key_value_heads * hd * 2
    out = {}
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        t, o = z["tokens"], z["offsets"]
        prompts = [t[o[i]:o[i + 1]].astype(np.int64).tolist() for i in range(len(o) - 1)]
    for name, maker in [("full", lambda: DynamicCache())] + [(f"b{b}", (lambda b=b: mod.make_cache(b, cfg))) for b in BUDGETS]:
        answers, bits = [], []
        for ids in prompts:
            ans, stored = run_prompt(model, tok, ids, maker, ctx.device)
            answers.append(ans)
            bits.append(8.0 * stored / (len(ids) * per_token))
        out[f"{name}_bits"] = np.asarray(bits)
        enc = [a.encode("utf-8")[:512] for a in answers]
        out[f"{name}_answer_utf8"] = np.frombuffer(b"".join(enc), np.uint8).copy()
        out[f"{name}_answer_offsets"] = np.concatenate([[0], np.cumsum([len(e) for e in enc])]).astype(np.int64)
    return out


STAGES = {"evaluate": Stage(timed=_timed, finish=_finish)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Dev set for selfcheck.py: the public dev prompts (generated with a public seed by the same code and tokenised the
    same way; 3 per task and length, 1 with --quick) play the held-out role."""
    with np.load(public / "dev_prompts.npz", allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    if quick:
        keep = []
        for ta in np.unique(d["task"]):
            for L in np.unique(d["length"]):
                keep.append(int(np.flatnonzero((d["task"] == ta) & (d["length"] == L))[0]))
        keep = sorted(keep)
        t, o = d["tokens"], d["offsets"]
        seqs = [t[o[i]:o[i + 1]] for i in keep]
        d["offsets"] = np.concatenate([[0], np.cumsum([len(s) for s in seqs])]).astype(np.int64)
        d["tokens"] = np.concatenate(seqs)
        exp = [d["expected"][i] for i in keep]
        d["task"], d["length"], d["expected"] = d["task"][keep], d["length"][keep], np.asarray(exp)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "held_out.npz", **{k: d[k] for k in HELD_OUT_KEYS})
    return {"public": public, "held_out": out / "held_out.npz",
            "reference": {"expected": d["expected"], "task": d["task"], "length": d["length"]}, "time_scale": 1.0}
