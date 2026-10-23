"""A-KV-004 pipeline: the evaluator takes your modified SmolLM2-135M (attention replaced by your key-value structure),
continues its training on 30M FineWeb-Edu tokens with a fixed recipe, and measures held-out loss, the key-value cache
size per token and long-context behaviour (evaluator and self-check use this same file).

One stage, "run" (imports model.py):
  setup   map the training tokens (untimed)
  timed   build(models_dir + "/SmolLM2-135M", device) -> smol_lm.SmolLM with your mixers; train STEPS steps of 16 x 2,048
          tokens with the fixed recipe of lm_train.py (AdamW peak lr 3e-4, warm-up and cosine decay, bf16, clipping)
  finish  held-out loss on windows of 2,048 and on windows of 8,192 tokens; cache growth in bytes per token between
          1,024 and 2,048 tokens of context (bf16 autocast, smol_lm.kv_bytes_per_token); consistency of prefill +
          step decoding with the full forward (smol_lm.decode_consistency)
The evaluator runs the multi-query baseline (baselines/mqa) through the same stage on the same seeds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from abench_stage import Context, Stage
import lm_train
import smol_lm

STEPS, LR, SEQ = 915, 3e-4, 2048
KV_BUDGET = 7680.0                                 # bytes per token: a third of the original 23,040
PRM = {"batch": 16, "accum": 2, "seq": SEQ, "warmup": 50, "weight_decay": 0.1, "grad_clip": 1.0}
PARAMS = {"steps": STEPS, "lr": LR, "recipe": dict(lm_train.RECIPE, **PRM), "kv_budget_bytes_per_token": KV_BUDGET,
          "train_shards": ["fwedu_train_000001.bin"]}
HELD_OUT_KEYS = ["tokens"]
PIPELINE = [{"stage": "run", "timeout": 3000 + 300, "finish_cap": 900, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": True}]


def _overrides(ctx: Context) -> dict:
    p = ctx.public / "overrides.json" if ctx.public else None        # written only by the self-check's --quick
    return json.loads(p.read_text()) if p and p.is_file() else {}


def _setup(ctx: Context):
    from token_data import read_shard
    return {"stream": np.asarray(read_shard(ctx.public / PARAMS["train_shards"][0])), "ov": _overrides(ctx)}


def _timed(ctx: Context, st):
    import torch
    build = ctx.import_submission("model.py", required=("build",)).build
    torch.manual_seed(ctx.seed)
    model = build(str(ctx.models / "SmolLM2-135M"), ctx.device)
    prm = dict(PRM, **{k: v for k, v in st["ov"].items() if k in lm_train.RECIPE})
    res = lm_train.train(model, st["stream"], int(st["ov"].get("steps", STEPS)), prm, ctx.seed, ctx.device, LR)
    return {"model": model, "res": res, "prm": prm, "ov": st["ov"]}


def _finish(ctx: Context, st):
    import torch
    model, res, prm = st["model"], st["res"], st["prm"]
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        toks = np.asarray(z["tokens"])[: int(st["ov"].get("eval_tokens", 2_097_152)) + 1]
    bad = lm_train.DIVERGED_LOSS
    out = {"val_loss": np.asarray([bad if res["diverged"] else
                                   lm_train.heldout_loss(model, toks, dict(prm, seq=SEQ, eval_batch=8), ctx.device)]),
           "val_loss_8k": np.asarray([bad if res["diverged"] else
                                      lm_train.heldout_loss(model, toks, dict(prm, seq=8192, eval_batch=2), ctx.device)]),
           "diverged": np.asarray([float(res["diverged"])]), "curve": res["curve"][::10]}
    model.eval()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=ctx.device.startswith("cuda")):
        out["kv_bytes_per_token"] = np.asarray([smol_lm.kv_bytes_per_token(model, ctx.device)])
    x = torch.from_numpy(toks[:2 * 1088].astype(np.int64).reshape(2, 1088)).to(ctx.device)
    out["decode_mismatch"] = np.asarray([smol_lm.decode_consistency(model.float(), x, 64)])
    return out


STAGES = {"run": Stage(setup=_setup, timed=_timed, finish=_finish)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Dev split for selfcheck.py: the last 2,097,153 training tokens are held out; --quick trains 60 steps and
    evaluates 262,144 tokens."""
    from token_data import read_shard, write_shard
    s = np.asarray(read_shard(public / PARAMS["train_shards"][0]))
    hold = 2_097_153
    (out / "public").mkdir(parents=True, exist_ok=True)
    write_shard(out / "public" / PARAMS["train_shards"][0], s[:-hold])
    if quick:
        (out / "public" / "overrides.json").write_text(json.dumps({"steps": 60, "warmup": 6, "eval_tokens": 262_144}))
    np.savez(out / "held_out.npz", tokens=s[-hold:])
    return {"public": out / "public", "held_out": out / "held_out.npz", "reference": {}, "time_scale": 1.0}
