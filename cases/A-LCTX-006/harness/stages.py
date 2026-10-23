"""A-LCTX-006 pipeline: the evaluator trains the fixed language model of ctx_model.py with your positions module at
1,024 tokens, extends it briefly at 4,096, and measures loss and retrieval at lengths up to 16,384 (evaluator and
self-check use this same file).

Stage "train" (imports positions.py):
  setup   map the training shards (untimed)
  timed   build ctx_model.LM with make_positions and the seed; train 3,000 steps of 32 x 1,024 FineWeb tokens with
          the fixed recipe of lm_train.py (AdamW lr 1e-3, warm-up and cosine decay, bf16), then 200 steps of 8 x 4,096
          tokens at lr 1e-4 (the context-extension phase)
  finish  held-out FineWeb loss at 1,024; PG-19 tail loss with 4,096, 8,192 and 16,384 tokens of context; copy-test
          loss at 8,192 and 16,384 (longctx.py); parameters of the positions module; divergence
The evaluator runs the RoPE baseline (baselines/rope) through the same stage on the same seeds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import lm_train
import longctx as LC
from abench_stage import Context, Stage
from ctx_model import LM, Config

CONFIG = Config(max_seq=16384)
STEPS, LR, EXT_STEPS, EXT_LR = 3000, 1e-3, 200, 1e-4
EXT = {"seq": 4096, "batch": 8, "warmup": 20}
MAX_POS_PARAMS = 65536
PARAMS = {"steps": STEPS, "lr": LR, "recipe": lm_train.RECIPE, "extension": {"steps": EXT_STEPS, "lr": EXT_LR, **EXT},
          "long_lengths": list(LC.LONG_LENGTHS), "copy_lengths": list(LC.COPY_LENGTHS), "tail": LC.TAIL,
          "max_position_params": MAX_POS_PARAMS, "train_shards": ["fineweb_train_000001.bin", "fineweb_train_000002.bin"]}
HELD_OUT_KEYS = ["tokens", "pg19_windows"] + [f"copy_x_{L}" for L in LC.COPY_LENGTHS] + [f"copy_start_{L}" for L in LC.COPY_LENGTHS]
PIPELINE = [{"stage": "train", "timeout": 2100 + 300, "finish_cap": 1200, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": False}]


def _overrides(ctx: Context) -> dict:
    p = ctx.public / "overrides.json" if ctx.public else None        # written only by the self-check's --quick
    return json.loads(p.read_text()) if p and p.is_file() else {}


def _setup(ctx: Context):
    from token_data import read_shard
    toks = [np.asarray(read_shard(ctx.public / s)) for s in PARAMS["train_shards"] if (ctx.public / s).is_file()]
    return {"stream": np.concatenate(toks), "ov": _overrides(ctx)}


def _timed(ctx: Context, st):
    import torch
    make = ctx.import_submission("positions.py", required=("make_positions",)).make_positions
    torch.manual_seed(ctx.seed)
    model = LM(CONFIG, make, train_len=lm_train.RECIPE["seq"]).to(ctx.device)
    ov = st["ov"]
    prm = {k: v for k, v in ov.items() if k in lm_train.RECIPE}
    res = lm_train.train(model, st["stream"], int(ov.get("steps", STEPS)), prm, ctx.seed, ctx.device, LR)
    ext = {"diverged": 0}
    if not res["diverged"]:
        ext = lm_train.train(model, st["stream"], int(ov.get("ext_steps", EXT_STEPS)), {**prm, **EXT}, ctx.seed + 1,
                             ctx.device, EXT_LR)
    return {"model": model, "res": res, "ext": ext, "prm": prm}


def _finish(ctx: Context, st):
    model, res, ext = st["model"], st["res"], st["ext"]
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        held = {k: z[k] for k in z.files}
    diverged = float(res["diverged"] or ext["diverged"])
    n_pos = sum(p.numel() for p in model.pos.parameters()) if hasattr(model.pos, "parameters") else 0
    out = {"diverged": np.asarray([diverged]), "position_params": np.asarray([n_pos], np.int64),
           "curve": res["curve"][::10]}
    if diverged:
        out["short_loss"] = np.asarray([lm_train.DIVERGED_LOSS])
        return out
    out["short_loss"] = np.asarray([lm_train.heldout_loss(model, held["tokens"], st["prm"], ctx.device)])
    for L in LC.LONG_LENGTHS:
        out[f"long_loss_{L}"] = LC.tail_loss(model, held["pg19_windows"], L, ctx.device)
    for L in LC.COPY_LENGTHS:
        if f"copy_x_{L}" in held:
            out[f"copy_loss_{L}"] = LC.copy_loss(model, held[f"copy_x_{L}"], held[f"copy_start_{L}"], ctx.device)
    return out


STAGES = {"train": Stage(setup=_setup, timed=_timed, finish=_finish)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Self-check: the last 2,097,152 tokens of the second training shard are held out for the short loss; the public
    PG-19 validation windows and copy sequences built from the held-out tokens with a public seed serve the long
    measures. --quick: 300 + 20 steps, 4 windows, 4 copy sequences per length."""
    from token_data import read_shard, write_shard
    s1, s2 = read_shard(public / PARAMS["train_shards"][0]), read_shard(public / PARAMS["train_shards"][1])
    hold = 2_097_152
    pub = out / "public"
    pub.mkdir(parents=True, exist_ok=True)
    write_shard(pub / PARAMS["train_shards"][0], np.asarray(s1))
    write_shard(pub / PARAMS["train_shards"][1], np.asarray(s2[:-hold]))
    with np.load(public / "pg19_dev.npz", allow_pickle=False) as z:
        windows = z["windows"]
    n_copy = 4 if quick else 16
    if quick:
        windows = windows[:4]
        (pub / "overrides.json").write_text(json.dumps({"steps": 300, "ext_steps": 20, "warmup": 30}))
    held = {"tokens": np.asarray(s2[-hold:]), "pg19_windows": windows}
    rng = np.random.default_rng(1729)
    for L in LC.COPY_LENGTHS:
        held[f"copy_x_{L}"], held[f"copy_start_{L}"] = LC.make_copy(np.asarray(s2[-hold:]).astype(np.int64), L, n_copy, rng)
    np.savez(out / "held_out.npz", **held)
    return {"public": pub, "held_out": out / "held_out.npz", "reference": {}, "time_scale": 1.0}
