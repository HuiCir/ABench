"""U-DLM-004 pipeline: the evaluator trains the fixed bidirectional denoiser (dlm.build_denoiser) with your noise
schedule and training objective at two scales and computes the masked-diffusion likelihood bound under your schedule
on held-out text (evaluator and self-check use this same file).

For each scale in SCALES, one stage "run_<scale>" (imports recipe.py):
  setup   map the training shards (untimed)
  timed   check the schedule (dlm.check_schedule); build the denoiser with the seed; train STEPS steps of 32 x 1,024
          tokens with the fixed recipe of lm_train.py (AdamW lr 1e-3, warm-up and cosine decay, bf16, clipping 1.0),
          the loss being your recipe.loss(model, x)
  finish  NELBO per token on the held-out windows under your schedule recipe.alpha (dlm.nelbo_per_token, 8 stratified
          times per window), its perplexity bound; divergence
The evaluator runs MDLM (linear schedule, baselines/mdlm) through the same stages on the same seeds; the gap to an
autoregressive model of the same size and data (AR_REFERENCE, measured by the operator at activation with
lm_model.LM and the same recipe) is reported.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from abench_stage import Context, Stage
import dlm
import lm_train
from lm_model import Config

SCALES = {"mu2": Config(), "mu3": Config(d_model=768, n_layers=10, n_heads=12, n_kv_heads=4, d_ff=2048)}
STEPS = 3000
ACCUM = {"mu2": 1, "mu3": 2}
LR, K_TIMES, EVAL_TOKENS = 1e-3, 8, 2_097_152
MIN_WINDOW = 0.2                       # every position needs >= 0.2 of diffusion time from alpha 0.9 to 0.1
PARAMS = {"steps": STEPS, "lr": LR, "recipe": lm_train.RECIPE, "eval_times": K_TIMES, "eval_tokens": EVAL_TOKENS,
          "accumulation": ACCUM, "min_window": MIN_WINDOW, "train_shards": ["fineweb_train_000001.bin", "fineweb_train_000002.bin"]}
HELD_OUT_KEYS = ["tokens"]
PIPELINE = [{"stage": f"run_{s}", "timeout": (1800 if s == "mu2" else 3600) + 300, "finish_cap": 900,
             "tail_slack": 10, "threads": 8, "held_out": True, "models": False} for s in SCALES]


def _overrides(ctx: Context) -> dict:
    p = ctx.public / "overrides.json" if ctx.public else None        # written only by the self-check's --quick
    return json.loads(p.read_text()) if p and p.is_file() else {}


def _setup(ctx: Context):
    from token_data import read_shard
    toks = [np.asarray(read_shard(ctx.public / s)) for s in PARAMS["train_shards"] if (ctx.public / s).is_file()]
    return {"stream": np.concatenate(toks), "ov": _overrides(ctx)}


def schedule_of(recipe):
    return recipe.alpha if hasattr(recipe, "alpha") else dlm.linear_alpha


def _make_timed(scale: str):
    def _timed(ctx: Context, st):
        import torch
        recipe = ctx.import_submission("recipe.py", required=("loss",))
        chk = dlm.check_schedule(schedule_of(recipe), 1024, ctx.device, min_window=MIN_WINDOW)
        if not chk["ok"]:
            raise SystemExit(f"recipe.alpha: {chk['reason']}")
        torch.manual_seed(ctx.seed)
        model = dlm.build_denoiser(SCALES[scale]).to(ctx.device)
        steps = int(st["ov"].get("steps", STEPS))
        prm = dict({k: v for k, v in st["ov"].items() if k in lm_train.RECIPE}, accum=ACCUM[scale])
        res = lm_train.train(model, st["stream"], steps, prm, ctx.seed, ctx.device, LR,
                             loss_fn=lambda m, x, y: recipe.loss(m, x))
        return {"model": model, "res": res, "ov": st["ov"], "alpha": schedule_of(recipe)}
    return _timed


def _finish(ctx: Context, st):
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        toks = np.asarray(z["tokens"])[: int(st["ov"].get("eval_tokens", EVAL_TOKENS))]
    model, res = st["model"], st["res"]
    nelbo = lm_train.DIVERGED_LOSS if res["diverged"] else dlm.nelbo_per_token(
        model, toks, 1024, ctx.device, alpha_fn=st["alpha"], k_times=K_TIMES, seed=ctx.seed)
    return {"nelbo": np.asarray([nelbo]), "diverged": np.asarray([float(res["diverged"])]),
            "curve": res["curve"][::10], "tokens_per_second": np.asarray([res["tokens_per_second"]])}


STAGES = {f"run_{s}": Stage(setup=_setup, timed=_make_timed(s), finish=_finish) for s in SCALES}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Dev split for selfcheck.py: the last 2,097,152 tokens of the second training shard are held out; --quick trains
    300 steps and evaluates 262,144 tokens."""
    from token_data import read_shard, write_shard
    s1, s2 = read_shard(public / PARAMS["train_shards"][0]), read_shard(public / PARAMS["train_shards"][1])
    hold = 2_097_152
    (out / "public").mkdir(parents=True, exist_ok=True)
    write_shard(out / "public" / PARAMS["train_shards"][0], np.asarray(s1))
    write_shard(out / "public" / PARAMS["train_shards"][1], np.asarray(s2[:-hold]))
    if quick:
        (out / "public" / "overrides.json").write_text(json.dumps({"steps": STEPS // 10, "warmup": 15,
                                                                   "eval_tokens": 262_144}))
    np.savez(out / "held_out.npz", tokens=np.asarray(s2[-hold:]))
    return {"public": out / "public", "held_out": out / "held_out.npz", "reference": {}, "time_scale": 1.0}
