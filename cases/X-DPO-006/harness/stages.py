"""X-DPO-006 pipeline: preference optimisation of SmolLM2-360M-Instruct, judged by a frozen reward model against the
starting model's own answers (evaluator and self-check).

Two stages, each in a new process:
  generate  (the starting model is under models/SmolLM2-360M-Instruct) timed: import model.py and call
            load(checkpoint_dir, models_dir, device). finish: greedy responses of your model and of the starting
            model to the held-out prompts (ChatML, at most 512 new tokens); log-probabilities of your responses under
            both models (KL estimate); implicit-reward accuracy on the held-out preference pairs; numbers stored in
            checkpoint/. Responses are written to responses.npz for the next stage.
  judge     (evaluator-only reward model; your code is not imported) finish: reward of every (prompt, response)
            for your responses and the starting model's. Without the reward model (self-check) the rewards are NaN.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import lmpost
from abench_stage import Context, Stage
from textpack import pack, unpack

BASE = "SmolLM2-360M-Instruct"
RM = "Skywork-Reward-V2-Qwen3-0.6B"
PARAMS = {"max_new_tokens": 512, "template": lmpost.CHAT, "decoding": "greedy", "reference_model": BASE}
HELD_OUT_KEYS = ["prompt_utf8", "prompt_offsets", "chosen_utf8", "chosen_offsets", "rejected_utf8", "rejected_offsets"]
PIPELINE = [{"stage": "generate", "timeout": 600 + 2400, "finish_cap": 2400, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": True},
            {"stage": "judge", "timeout": 60, "finish_cap": 900, "tail_slack": 10, "threads": 8, "held_out": True,
             "models": False, "judge_models": True, "carry": {"generate": ["responses.npz"]}}]


def _held(ctx):
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        return {k: unpack(z[f"{k}_utf8"], z[f"{k}_offsets"]) for k in ("prompt", "chosen", "rejected")}


def _timed(ctx: Context, _):
    mod = ctx.import_submission("model.py", required=("load",))
    return mod.load(str(ctx.submission / "checkpoint"), str(ctx.models), ctx.device)


def _finish(ctx: Context, model):
    h = _held(ctx)
    base_dir = str(ctx.models / BASE)
    tok = lmpost.tokenizer(base_dir)
    model.eval()
    resp = lmpost.generate(model, tok, h["prompt"], PARAMS["max_new_tokens"], ctx.device)
    lp_own = lmpost.response_logprob(model, tok, h["prompt"], resp, ctx.device)
    lp_ch = lmpost.response_logprob(model, tok, h["prompt"], h["chosen"], ctx.device)
    lp_rj = lmpost.response_logprob(model, tok, h["prompt"], h["rejected"], ctx.device)
    del model
    base = lmpost.load_base(base_dir, ctx.device)
    resp_base = lmpost.generate(base, tok, h["prompt"], PARAMS["max_new_tokens"], ctx.device)
    lr_own = lmpost.response_logprob(base, tok, h["prompt"], resp, ctx.device)
    lr_ch = lmpost.response_logprob(base, tok, h["prompt"], h["chosen"], ctx.device)
    lr_rj = lmpost.response_logprob(base, tok, h["prompt"], h["rejected"], ctx.device)
    n_tok = np.asarray([len(tok(r, add_special_tokens=False)["input_ids"]) + 1 for r in resp], np.float64)
    pu, po = pack(resp)
    bu, bo = pack(resp_base)
    np.savez(ctx.work / "responses.npz", policy_utf8=pu, policy_offsets=po, base_utf8=bu, base_offsets=bo)
    return {"policy_words": np.asarray([len(r.split()) for r in resp], np.float64),
            "base_words": np.asarray([len(r.split()) for r in resp_base], np.float64),
            "log_ratio_own": lp_own - lr_own, "own_tokens": n_tok,
            "margin_pairs": (lp_ch - lr_ch) - (lp_rj - lr_rj),
            "checkpoint_numbers": np.asarray([lmpost.checkpoint_numbers(ctx.submission / "checkpoint")], np.int64)}


def _judge_timed(ctx: Context, _):
    return None


def _judge_finish(ctx: Context, _):
    import torch
    h = _held(ctx)
    with np.load(ctx.work / "responses.npz", allow_pickle=False) as z:
        pol = unpack(z["policy_utf8"], z["policy_offsets"])
        bas = unpack(z["base_utf8"], z["base_offsets"])
    if ctx.models is None or not (ctx.models / RM).is_dir():
        nan = np.full(len(pol), np.nan)
        return {"reward_policy": nan, "reward_base": nan.copy()}
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(ctx.models / RM), local_files_only=True)
    rm = AutoModelForSequenceClassification.from_pretrained(str(ctx.models / RM), local_files_only=True,
                                                            torch_dtype=torch.bfloat16, num_labels=1).to(ctx.device).eval()

    def rewards(responses):
        out = []
        for p, r in zip(h["prompt"], responses):
            conv = [{"role": "user", "content": p}, {"role": "assistant", "content": r}]
            ids = tok.apply_chat_template(conv, tokenize=True, return_tensors="pt", truncation=True, max_length=4096).to(ctx.device)
            with torch.no_grad():
                out.append(float(rm(ids).logits[0][0]))
        return np.asarray(out)

    return {"reward_policy": rewards(pol), "reward_base": rewards(bas)}


STAGES = {"generate": Stage(timed=_timed, finish=_finish), "judge": Stage(timed=_judge_timed, finish=_judge_finish)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Self-check: the 500 public dev pairs (quick: 50). The reward model is not in the sandbox, so the self-check
    reports the implicit-reward accuracy, KL and lengths, but no win rate."""
    with np.load(public / "prefs_public.npz", allow_pickle=False) as z:
        d = {k: unpack(z[f"dev_{k}_utf8"], z[f"dev_{k}_offsets"]) for k in ("prompt", "chosen", "rejected")}
    n = 50 if quick else len(d["prompt"])
    arrs = {}
    for k, v in d.items():
        arrs[f"{k}_utf8"], arrs[f"{k}_offsets"] = pack(v[:n])
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "held_out.npz", **arrs)
    return {"public": public, "held_out": out / "held_out.npz", "reference": {"n": np.asarray([n])}, "time_scale": 1.0}
