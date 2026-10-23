"""X-GRPO-002 pipeline: the RL-trained Qwen2.5-0.5B-Instruct answers held-out Countdown and GSM8K problems; the
evaluator formats the prompts, decodes and verifies (evaluator and self-check use this same file).

One stage, "eval" (the base model is under models/Qwen2.5-0.5B-Instruct):
  timed   import model.py and call load(checkpoint_dir, models_dir, device), which returns the trained causal LM
  finish  greedy responses (ChatML template, at most 512 new tokens, lmpost.generate) to the held-out prompts of both
          task sets, and the number of parameters stored in checkpoint/                                (≤ 1,800 s)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import lmpost
import verify_math as V
from abench_stage import Context, Stage
from textpack import pack, unpack

PARAMS = {"max_new_tokens": 512, "template": lmpost.CHAT, "decoding": "greedy", "base_model": "Qwen2.5-0.5B-Instruct"}
HELD_OUT_KEYS = ["prompt_utf8", "prompt_offsets"]
PIPELINE = [{"stage": "eval", "timeout": 600 + 1800, "finish_cap": 1800, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": True}]


def _timed(ctx: Context, _):
    mod = ctx.import_submission("model.py", required=("load",))
    return mod.load(str(ctx.submission / "checkpoint"), str(ctx.models), ctx.device)


def _finish(ctx: Context, model):
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        prompts = unpack(z["prompt_utf8"], z["prompt_offsets"])
    tok = lmpost.tokenizer(str(ctx.models / PARAMS["base_model"]))
    model.eval()
    resp = lmpost.generate(model, tok, prompts, PARAMS["max_new_tokens"], ctx.device)
    return {"response": np.asarray(resp, dtype=np.str_),
            "checkpoint_numbers": np.asarray([lmpost.checkpoint_numbers(ctx.submission / "checkpoint")], np.int64)}


STAGES = {"eval": Stage(timed=_timed, finish=_finish)}


def held_out_from(countdown: list, gsm8k: list) -> tuple[dict, dict]:
    """Inputs (prompts) and evaluator reference for lists of Countdown instances and GSM8K {question, gold}."""
    prompts = [V.countdown_prompt(c) for c in countdown] + [V.gsm8k_prompt(g["question"]) for g in gsm8k]
    u, o = pack(prompts)
    ref = {"task": np.asarray([0] * len(countdown) + [1] * len(gsm8k), np.int64),
           "spec_json": np.asarray([json.dumps(c) for c in countdown] + [json.dumps({"gold": g["gold"]}) for g in gsm8k],
                                   dtype=np.str_)}
    return {"prompt_utf8": u, "prompt_offsets": o}, ref


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Self-check: the public dev problems (500 Countdown instances from a public seed and 300 GSM8K training problems
    set aside; --quick: 40 of each)."""
    dev = json.loads((public / "dev.json").read_text(encoding="utf-8"))
    cd, gs = dev["countdown"], dev["gsm8k"]
    if quick:
        cd, gs = cd[:40], gs[:40]
    held, ref = held_out_from(cd, gs)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "held_out.npz", **held)
    return {"public": public, "held_out": out / "held_out.npz", "reference": ref, "time_scale": 1.0}
