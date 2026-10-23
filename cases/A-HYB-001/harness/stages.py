"""A-HYB-001 pipeline: the evaluator builds your model twice — as a language model and as a small associative-recall
model — trains both with fixed recipes, and measures loss, recall and long-context cost (evaluator and self-check use
this same file).

Stage "lm" (imports model.py):
  setup   map the training shards (untimed)
  timed   build(LM_CONFIG) with the seed; train 3,000 steps of 32 x 1,024 FineWeb tokens with the fixed recipe of
          lm_train.py (AdamW lr 1e-3, warm-up and cosine decay, bf16, clipping 1.0)
  finish  held-out loss; forward FLOPs per token at T = 1,024 and 16,384 (FlopCounterMode) and parameters, each next
          to the transformer of lm_model.py at the same sizes; causality check; forward latency per token at
          T = 1,024, 4,096 and 16,384 (batch 1, inference only)
Stage "mqar" (imports model.py):
  timed   build(MQAR_CONFIG) with the seed; train 8,000 steps of batch 128 on mqar.TRAIN_GRID (A-LIN-002's recipe)
  finish  predictions at the query positions of the held-out MQAR sequences (longer sequences, more pairs)
The evaluator runs the transformer baseline (baselines/transformer) through the same stages on the same seeds.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from abench_stage import Context, Stage
import lm_train
import mqar
from lm_model import Config

LM_CONFIG = Config(max_seq=16384)                                      # B-OPT-001's μ2 sizes; trained at 1,024
MQAR_CONFIG = Config(vocab_size=mqar.VOCAB, d_model=128, n_layers=4, n_heads=2, n_kv_heads=2, d_ff=384, max_seq=2048)
LM_STEPS, LM_LR = 3000, 1e-3
MQAR = {"steps": 8000, "batch": 128, "lr": 1e-3, "weight_decay": 0.1, "warmup": 400}
LENGTHS = (1024, 4096, 16384)
PARAMS = {"lm_steps": LM_STEPS, "lm_lr": LM_LR, "recipe": lm_train.RECIPE, "mqar": MQAR, "train_grid": mqar.TRAIN_GRID,
          "latency_lengths": list(LENGTHS), "train_shards": ["fineweb_train_000001.bin", "fineweb_train_000002.bin"]}
HELD_OUT_KEYS = ["tokens", "mqar_x", "mqar_cell", "mqar_t_len"]
PIPELINE = [{"stage": "lm", "timeout": 1800 + 300, "finish_cap": 600, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": False},
            {"stage": "mqar", "timeout": 1500 + 120, "finish_cap": 300, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": False}]
DEV_GRID = [(256, 16), (512, 32), (1024, 64), (1024, 128)]


def _overrides(ctx: Context) -> dict:
    p = ctx.public / "overrides.json" if ctx.public else None        # written only by the self-check's --quick
    return json.loads(p.read_text()) if p and p.is_file() else {}


def _setup_lm(ctx: Context):
    from token_data import read_shard
    toks = [np.asarray(read_shard(ctx.public / s)) for s in PARAMS["train_shards"] if (ctx.public / s).is_file()]
    return {"stream": np.concatenate(toks), "ov": _overrides(ctx)}


def _timed_lm(ctx: Context, st):
    import torch
    build = ctx.import_submission("model.py", required=("build",)).build
    torch.manual_seed(ctx.seed)
    model = build(LM_CONFIG).to(ctx.device)
    steps = int(st["ov"].get("lm_steps", LM_STEPS))
    prm = {k: v for k, v in st["ov"].items() if k in lm_train.RECIPE}
    res = lm_train.train(model, st["stream"], steps, prm, ctx.seed, ctx.device, LM_LR)
    return {"model": model, "res": res, "prm": prm}


def _causal_leak(model, vocab: int, T: int, device: str) -> float:
    import torch
    g = torch.Generator(device="cpu").manual_seed(0)
    xa = torch.randint(1, vocab, (2, T), generator=g)
    xb = xa.clone()
    xb[:, T // 2:] = torch.randint(1, vocab, (2, T - T // 2), generator=g)
    with torch.no_grad():
        ya, yb = model(xa.to(device)).float(), model(xb.to(device)).float()
    h = T // 2
    return float((ya[:, :h] - yb[:, :h]).abs().max() / (ya[:, :h].abs().max() + 1e-6))


def _latency(ctx: Context, model, T: int) -> float:
    import torch
    x = torch.randint(1, 50257, (1, T), device=ctx.device)
    ts = []
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=ctx.device.startswith("cuda")):
        for rep in range(3):
            ctx.sync()
            t0 = time.perf_counter()
            model(x)
            ctx.sync()
            if rep:
                ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) / T


def _finish_lm(ctx: Context, st):
    import torch
    from lm_model import LM
    model, res = st["model"], st["res"]
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        toks = z["tokens"]
    loss = lm_train.DIVERGED_LOSS if res["diverged"] else lm_train.heldout_loss(model, toks, st["prm"], ctx.device)
    base = LM(LM_CONFIG).to(ctx.device)
    out = {"val_loss": np.asarray([loss]), "diverged": np.asarray([float(res["diverged"])]),
           "curve": res["curve"][::10], "tokens_per_second": np.asarray([res["tokens_per_second"]])}
    for T in (1024, 16384):
        f = lm_train.forward_flops_per_token(model, T, ctx.device)
        fb = lm_train.forward_flops_per_token(base, T, ctx.device)
        out[f"flops_ratio_T{T}"] = np.asarray([f / fb])
        out[f"flops_per_token_T{T}"] = np.asarray([f])
    out["params_ratio"] = np.asarray([sum(p.numel() for p in model.parameters()) /
                                      sum(p.numel() for p in base.parameters())])
    del base
    model.eval()
    out["causal_leak"] = np.asarray([_causal_leak(model, 50257, 1024, ctx.device)])
    for T in LENGTHS:
        out[f"latency_per_token_T{T}"] = np.asarray([_latency(ctx, model, T)])
    return out


def _timed_mqar(ctx: Context, _):
    import torch
    import torch.nn.functional as F
    build = ctx.import_submission("model.py", required=("build",)).build
    prm = dict(MQAR, **_overrides(ctx).get("mqar", {}))
    torch.manual_seed(ctx.seed)
    model = build(MQAR_CONFIG).to(ctx.device)
    opt = torch.optim.AdamW(lm_train.param_groups(model, prm["weight_decay"]), lr=prm["lr"], betas=(0.9, 0.95))
    rng = np.random.default_rng(ctx.seed)
    grid = [tuple(g) for g in mqar.TRAIN_GRID]
    for step in range(prm["steps"]):
        f = lm_train.schedule(step, prm["steps"], prm["warmup"], 0.0)
        for g in opt.param_groups:
            g["lr"] = prm["lr"] * f
        T, K = grid[step % len(grid)]
        x, y = mqar.batch(rng, prm["batch"], T, K)
        x, y = torch.from_numpy(x).to(ctx.device), torch.from_numpy(y).to(ctx.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=ctx.device.startswith("cuda")):
            logits = model(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1), ignore_index=mqar.IGNORE)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    return model


def _finish_mqar(ctx: Context, model):
    import torch
    model.eval()
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        X, cell, t_len = z["mqar_x"], z["mqar_cell"], z["mqar_t_len"]
    preds = np.zeros_like(X)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=ctx.device.startswith("cuda")):
        for c in np.unique(cell):
            idx = np.flatnonzero(cell == c)
            T = int(t_len[idx[0]])
            for i in range(0, len(idx), 32):
                xb = torch.from_numpy(X[idx[i:i + 32], :T]).to(ctx.device)
                preds[idx[i:i + 32], :T] = model(xb).float().argmax(-1).cpu().numpy()
    return {"pred": preds, "mqar_causal_leak": np.asarray([_causal_leak(model, mqar.VOCAB, 512, ctx.device)])}


STAGES = {"lm": Stage(setup=_setup_lm, timed=_timed_lm, finish=_finish_lm),
          "mqar": Stage(timed=_timed_mqar, finish=_finish_mqar)}


def build_mqar_split(seed: int, grid, per_cell: int) -> dict:
    rng = np.random.default_rng(seed)
    Tmax = max(T for T, _ in grid)
    xs, ys, cells = [], [], []
    for c, (T, K) in enumerate(grid):
        x, y = mqar.batch(rng, per_cell, T, K)
        xs.append(np.pad(x, ((0, 0), (0, Tmax - T))))
        ys.append(np.pad(y, ((0, 0), (0, Tmax - T)), constant_values=mqar.IGNORE))
        cells.append(np.full(per_cell, c))
    cell = np.concatenate(cells).astype(np.int64)
    return {"mqar_x": np.concatenate(xs), "mqar_y": np.concatenate(ys), "mqar_cell": cell,
            "mqar_t_len": np.asarray(grid, np.int64)[cell, 0], "mqar_grid": np.asarray(grid, np.int64)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Dev split for selfcheck.py: the last 2,097,152 tokens of the second training shard are held out for the
    language model, and MQAR sequences on DEV_GRID come from a public seed; --quick trains 300 LM and 800 MQAR steps."""
    from token_data import read_shard, write_shard
    s1, s2 = read_shard(public / PARAMS["train_shards"][0]), read_shard(public / PARAMS["train_shards"][1])
    hold = 2_097_152
    (out / "public").mkdir(parents=True, exist_ok=True)
    write_shard(out / "public" / PARAMS["train_shards"][0], np.asarray(s1))
    write_shard(out / "public" / PARAMS["train_shards"][1], np.asarray(s2[:-hold]))
    if quick:
        (out / "public" / "overrides.json").write_text(json.dumps({"lm_steps": 300, "warmup": 15,
                                                                   "mqar": {"steps": 800, "warmup": 80}}))
    d = build_mqar_split(4242, DEV_GRID, 16 if quick else 64)
    np.savez(out / "held_out.npz", tokens=np.asarray(s2[-hold:]),
             **{k: d[k] for k in ("mqar_x", "mqar_cell", "mqar_t_len")})
    return {"public": out / "public", "held_out": out / "held_out.npz", "reference": d, "time_scale": 1.0}

