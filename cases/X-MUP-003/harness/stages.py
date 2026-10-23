"""X-MUP-003 pipeline: tune the learning rate on a narrow model, transfer it to wider ones (evaluator and self-check use
this same file).

One stage, "protocol" (imports param.py), per seed:
  setup   map the training shards; the last 1,048,576 tokens of the second shard are kept out of training as the
          tuning set (untimed)
  timed   (1) width 128: train with each η in ETA_GRID (7 values, factors of 2) for STEPS steps; η* = the η with the
              lowest loss on the tuning set
          (2) width 512: train with η* · m for m in (1/4, 1/2, 1, 2, 4); tuning-set loss of each (the transfer gap)
          (3) width 1,024: train with η*
          every run: mup_lm.LM(Config(width), PARAM), AdamW (β = 0.9, 0.95, weight decay 0.1 times your wd_scale),
          per-parameter learning rate η · lr_scale, 150 warm-up steps and cosine decay to 0.1, clipping 1.0, bf16
  finish  held-out loss of the width-1,024 model and of the width-512 model trained with η*
The evaluator runs the standard parametrisation (baselines/sp) through the same stage on the same seeds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from abench_stage import Context, Stage
import lm_train

ETA_GRID = [2.0 ** e for e in range(-12, -5)]          # 2.4e-4 ... 1.6e-2
MULTS = (0.25, 0.5, 1.0, 2.0, 4.0)
STEPS = 1500
TUNE_TOKENS = 1_048_576
PARAMS = {"eta_grid": ETA_GRID, "transfer_multipliers": list(MULTS), "steps": STEPS, "recipe": lm_train.RECIPE,
          "widths": {"tune": 128, "check": 512, "target": 1024}, "tuning_tokens": TUNE_TOKENS,
          "train_shards": ["fineweb_train_000001.bin", "fineweb_train_000002.bin"]}
HELD_OUT_KEYS = ["tokens"]
PIPELINE = [{"stage": "protocol", "timeout": 3600 + 600, "finish_cap": 300, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": False}]


def _overrides(ctx: Context) -> dict:
    p = ctx.public / "overrides.json" if ctx.public else None        # written only by the self-check's --quick
    return json.loads(p.read_text()) if p and p.is_file() else {}


def _setup(ctx: Context):
    from token_data import read_shard
    toks = [np.asarray(read_shard(ctx.public / s)) for s in PARAMS["train_shards"] if (ctx.public / s).is_file()]
    stream = np.concatenate(toks)
    return {"stream": stream[:-TUNE_TOKENS], "tune": stream[-TUNE_TOKENS:], "ov": _overrides(ctx)}


def _run(ctx: Context, st, param, width: int, eta: float, seed: int):
    import torch
    from mup_lm import LM, Config
    torch.manual_seed(seed)
    model = LM(Config(width=width), param).to(ctx.device)
    prm = {k: v for k, v in st["ov"].items() if k in lm_train.RECIPE}
    steps = int(st["ov"].get("steps", STEPS))
    res = lm_train.train(model, st["stream"], steps, prm, seed, ctx.device, eta,
                         groups=model.param_groups(lm_train.RECIPE["weight_decay"]))
    tune = lm_train.DIVERGED_LOSS if res["diverged"] else lm_train.heldout_loss(model, st["tune"], prm, ctx.device)
    return model, tune, res


def _timed(ctx: Context, st):
    param = ctx.import_submission("param.py", required=("PARAM",)).PARAM
    grid = st["ov"].get("eta_grid", ETA_GRID)
    sweep = []
    for i, eta in enumerate(grid):
        _, tl, _ = _run(ctx, st, param, 128, eta, ctx.seed)
        sweep.append(tl)
    eta_star = float(grid[int(np.argmin(sweep))])
    check, keep512 = [], None
    for m in MULTS:
        model, tl, _ = _run(ctx, st, param, 512, eta_star * m, ctx.seed + 1)
        check.append(tl)
        if m == 1.0:
            keep512 = model.cpu()
        del model
    model1024, tl1024, res = _run(ctx, st, param, 1024, eta_star, ctx.seed + 2)
    return {"sweep": sweep, "eta_star": eta_star, "check": check, "m512": keep512, "m1024": model1024,
            "tune1024": tl1024, "diverged1024": res["diverged"], "prm": {k: v for k, v in st["ov"].items()
                                                                          if k in lm_train.RECIPE}}


def _finish(ctx: Context, st):
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        toks = z["tokens"]
    l1024 = lm_train.DIVERGED_LOSS if st["diverged1024"] else lm_train.heldout_loss(st["m1024"], toks, st["prm"], ctx.device)
    st["m1024"].cpu()
    l512 = lm_train.heldout_loss(st["m512"].to(ctx.device), toks, st["prm"], ctx.device)
    return {"val_loss_w1024": np.asarray([l1024]), "val_loss_w512": np.asarray([l512]),
            "tune_sweep_w128": np.asarray(st["sweep"], np.float64), "eta_star": np.asarray([st["eta_star"]]),
            "tune_check_w512": np.asarray(st["check"], np.float64), "tune_w1024": np.asarray([st["tune1024"]])}


STAGES = {"protocol": Stage(setup=_setup, timed=_timed, finish=_finish)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Dev split for selfcheck.py: the last 2,097,152 tokens of the second training shard are held out (the tuning set
    is the 1,048,576 tokens before them); --quick trains 150 steps per run over 3 learning rates."""
    from token_data import read_shard, write_shard
    s1, s2 = read_shard(public / PARAMS["train_shards"][0]), read_shard(public / PARAMS["train_shards"][1])
    hold = 2_097_152
    (out / "public").mkdir(parents=True, exist_ok=True)
    write_shard(out / "public" / PARAMS["train_shards"][0], np.asarray(s1))
    write_shard(out / "public" / PARAMS["train_shards"][1], np.asarray(s2[:-hold]))
    if quick:
        (out / "public" / "overrides.json").write_text(json.dumps({"steps": 150, "warmup": 15,
                                                                   "eta_grid": [2.0 ** -10, 2.0 ** -8, 2.0 ** -6]}))
    np.savez(out / "held_out.npz", tokens=np.asarray(s2[-hold:]))
    return {"public": out / "public", "held_out": out / "held_out.npz", "reference": {}, "time_scale": 1.0}
