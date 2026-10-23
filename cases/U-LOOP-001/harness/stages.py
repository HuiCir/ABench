"""U-LOOP-001 pipeline: the evaluator trains your recurrent-depth language model with a fixed recipe and measures
held-out loss at 1, 2, 4 and 8 loops (evaluator and self-check use this same file).

One stage, "run" (imports model.py):
  setup   map the training shards (untimed)
  timed   build(CONFIG) with the seed; train 3,000 steps of 32 x 1,024 FineWeb tokens with the fixed recipe of
          lm_train.py (AdamW lr 1e-3, warm-up and cosine decay, bf16, clipping 1.0) using model.loss(x, y) — how many
          loops are used in training is up to the model
  finish  held-out loss with model.loss(x, y, loops=k) for k in LOOPS (windows of 1,024), and at k = 8 on windows of
          2,048 (length extrapolation); parameters; forward FLOPs per token at each k (FlopCounterMode) next to the
          4-layer baseline transformer of the same width; causality check
The evaluator runs the 4-layer transformer (baselines/fourlayer: the same parameter budget, no loops) through the same
stage on the same seeds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from abench_stage import Context, Stage
import lm_train
from lm_model import Config

CONFIG = Config(n_layers=4, max_seq=2048)            # width 384; the parameter budget of a 4-layer transformer
STEPS, LR = 3000, 1e-3
LOOPS = (1, 2, 4, 8)
PARAMS = {"steps": STEPS, "lr": LR, "loops": list(LOOPS), "recipe": lm_train.RECIPE,
          "train_shards": ["fineweb_train_000001.bin", "fineweb_train_000002.bin"]}
HELD_OUT_KEYS = ["tokens"]
PIPELINE = [{"stage": "run", "timeout": 1800 + 300, "finish_cap": 900, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": False}]
EVAL_TOKENS = 2_097_152                              # held-out tokens used for each loop count (the first half)


def _overrides(ctx: Context) -> dict:
    p = ctx.public / "overrides.json" if ctx.public else None        # written only by the self-check's --quick
    return json.loads(p.read_text()) if p and p.is_file() else {}


def _setup(ctx: Context):
    from token_data import read_shard
    toks = [np.asarray(read_shard(ctx.public / s)) for s in PARAMS["train_shards"] if (ctx.public / s).is_file()]
    return {"stream": np.concatenate(toks), "ov": _overrides(ctx)}


def _timed(ctx: Context, st):
    import torch
    build = ctx.import_submission("model.py", required=("build",)).build
    torch.manual_seed(ctx.seed)
    model = build(CONFIG).to(ctx.device)
    steps = int(st["ov"].get("steps", STEPS))
    prm = {k: v for k, v in st["ov"].items() if k in lm_train.RECIPE}
    res = lm_train.train(model, st["stream"], steps, prm, ctx.seed, ctx.device, LR)
    return {"model": model, "res": res, "prm": prm}


def _finish(ctx: Context, st):
    import torch
    from lm_model import LM
    model, res, prm = st["model"], st["res"], st["prm"]
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        toks = np.asarray(z["tokens"])
    toks = toks[: EVAL_TOKENS + 1] if len(toks) > EVAL_TOKENS + 1 else toks
    out = {"diverged": np.asarray([float(res["diverged"])]), "curve": res["curve"][::10],
           "tokens_per_second": np.asarray([res["tokens_per_second"]])}
    base = LM(CONFIG).to(ctx.device)
    fb = lm_train.forward_flops_per_token(base, 1024, ctx.device)
    for k in LOOPS:
        fn = (lambda m, x, y, k=k: m.loss(x, y, loops=k))
        v = lm_train.DIVERGED_LOSS if res["diverged"] else lm_train.heldout_loss(model, toks, prm, ctx.device, fn)
        out[f"val_loss_loops{k}"] = np.asarray([v])
        f = lm_train.forward_flops_per_token(model, 1024, ctx.device, call=lambda m, x, k=k: m(x, loops=k))
        out[f"flops_ratio_loops{k}"] = np.asarray([f / fb])
    long = dict(prm, seq=2048, eval_batch=8)
    fn8 = (lambda m, x, y: m.loss(x, y, loops=8))
    out["val_loss_T2048_loops8"] = np.asarray([lm_train.DIVERGED_LOSS if res["diverged"] else
                                               lm_train.heldout_loss(model, toks, long, ctx.device, fn8)])
    out["params_ratio"] = np.asarray([sum(p.numel() for p in model.parameters()) /
                                      sum(p.numel() for p in base.parameters())])
    del base
    model.eval()
    g = torch.Generator(device="cpu").manual_seed(0)
    xa = torch.randint(1, 50257, (2, 1024), generator=g)
    xb = xa.clone()
    xb[:, 512:] = torch.randint(1, 50257, (2, 512), generator=g)
    with torch.no_grad():
        ya, yb = model(xa.to(ctx.device), loops=8).float(), model(xb.to(ctx.device), loops=8).float()
    out["causal_leak"] = np.asarray([float((ya[:, :512] - yb[:, :512]).abs().max() / (ya[:, :512].abs().max() + 1e-6))])
    return out


STAGES = {"run": Stage(setup=_setup, timed=_timed, finish=_finish)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Dev split for selfcheck.py: the last 2,097,152 tokens of the second training shard are held out; --quick trains
    300 steps."""
    from token_data import read_shard, write_shard
    s1, s2 = read_shard(public / PARAMS["train_shards"][0]), read_shard(public / PARAMS["train_shards"][1])
    hold = 2_097_152
    (out / "public").mkdir(parents=True, exist_ok=True)
    write_shard(out / "public" / PARAMS["train_shards"][0], np.asarray(s1))
    write_shard(out / "public" / PARAMS["train_shards"][1], np.asarray(s2[:-hold]))
    if quick:
        (out / "public" / "overrides.json").write_text(json.dumps({"steps": STEPS // 10, "warmup": 15}))
    np.savez(out / "held_out.npz", tokens=np.asarray(s2[-hold:]))
    return {"public": out / "public", "held_out": out / "held_out.npz", "reference": {}, "time_scale": 1.0}
