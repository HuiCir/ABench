"""X-SFT-005 pipeline: instruction following of a supervised fine-tune of SmolLM2-360M, its forgetting and its
trained-parameter cost (evaluator and self-check).

One stage, "eval" (the base model is under models/SmolLM2-360M):
  timed   import model.py and call load(checkpoint_dir, models_dir, device), which returns the fine-tuned causal LM
  finish  greedy responses (ChatML template, at most 512 new tokens, lmpost.generate) to the held-out prompts;
          held-out FineWeb-Edu loss of the fine-tuned model and of the base model; numbers stored in checkpoint/
                                                                                                   (≤ 1,800 s)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import ifcheck
import lmpost
from abench_stage import Context, Stage
from textpack import pack, unpack

PARAMS = {"max_new_tokens": 512, "template": lmpost.CHAT, "decoding": "greedy", "nll_window": 1024}
HELD_OUT_KEYS = ["prompt_utf8", "prompt_offsets", "tokens"]
PIPELINE = [{"stage": "eval", "timeout": 600 + 1800, "finish_cap": 1800, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": True}]


def _timed(ctx: Context, _):
    mod = ctx.import_submission("model.py", required=("load",))
    return mod.load(str(ctx.submission / "checkpoint"), str(ctx.models), ctx.device)


def _finish(ctx: Context, model):
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        prompts = unpack(z["prompt_utf8"], z["prompt_offsets"])
        tokens = z["tokens"]
    base_dir = str(ctx.models / "SmolLM2-360M")
    tok = lmpost.tokenizer(base_dir)
    model.eval()
    resp = lmpost.generate(model, tok, prompts, PARAMS["max_new_tokens"], ctx.device)
    nll = lmpost.heldout_nll(model, tokens, ctx.device)
    del model
    base = lmpost.load_base(base_dir, ctx.device)
    nll_base = lmpost.heldout_nll(base, tokens, ctx.device)
    return {"response": np.asarray(resp, dtype=np.str_), "nll": np.asarray([nll]), "nll_base": np.asarray([nll_base]),
            "checkpoint_numbers": np.asarray([lmpost.checkpoint_numbers(ctx.submission / "checkpoint")], np.int64)}


STAGES = {"eval": Stage(timed=_timed, finish=_finish)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Self-check: the 500 public dev requests with constraints from a public seed (quick: 60), and the public
    FineWeb-Edu dev tokens (quick: the first 65,537)."""
    import json
    with np.load(public / "sft_public.npz", allow_pickle=False) as z:
        reqs = unpack(z["dev_request_utf8"], z["dev_request_offsets"])
        tokens = z["dev_tokens"]
    if quick:
        reqs, tokens = reqs[:60], tokens[:65537]
    rng = np.random.default_rng(20231114)
    items = [ifcheck.make(r, rng) for r in reqs]
    u, o = pack([it["prompt"] for it in items])
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "held_out.npz", prompt_utf8=u, prompt_offsets=o, tokens=tokens)
    cons = np.asarray([json.dumps({"kinds": it["kinds"], "params": it["params"]}) for it in items], dtype=np.str_)
    return {"public": public, "held_out": out / "held_out.npz", "reference": {"constraints": cons},
            "time_scale": 1.0}
