"""X-DATA-007 pipeline: your data pipeline selects training documents from a raw web pool; the evaluator trains a
fixed small language model on the selection and measures its loss on held-out educational text (evaluator and
self-check).

Two stages, each in a new process:
  select  (8 CPU cores, no GPU) timed: import pipeline.py and call select(pool_dir, budget_tokens, workdir) → document
          indices in training order (repeats allowed); finish: check the indices and write selection.npy
  train   (GPU; your code is not imported) setup: the token stream = the pool tokens of the selected documents,
          concatenated; timed: train norm_lm.LM (the 30M-parameter "mu2" configuration with pre-RMSNorm) with the
          fixed recipe of lm_train.py for STEPS steps of 32 × 1,024 tokens sampled from the stream; finish: held-out
          loss on educational text (target) and on general web text
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import lm_train
from abench_stage import Context, Stage

STEPS = 3000
BUDGET_TOKENS = STEPS * 32 * 1024                    # tokens the fixed model sees (98,304,000)
MIN_SELECTED = 20_000_000
PARAMS = {"steps": STEPS, "budget_tokens": BUDGET_TOKENS, "min_selected_tokens": MIN_SELECTED, "recipe": lm_train.RECIPE,
          "lr": 2e-3, "model": "norm_lm mu2 with pre-RMSNorm"}
HELD_OUT_KEYS = ["target_tokens", "general_tokens"]
PIPELINE = [{"stage": "select", "timeout": 3600, "finish_cap": 120, "tail_slack": 10, "threads": 8, "held_out": False,
             "models": False},
            {"stage": "train", "timeout": 2400, "finish_cap": 300, "tail_slack": 10, "threads": 8, "held_out": True,
             "models": False, "carry": {"select": ["selection.npy"]}}]


def _overrides(ctx: Context) -> dict:
    p = ctx.public / "overrides.json" if ctx.public else None        # written only by the self-check's --quick
    return json.loads(p.read_text()) if p and p.is_file() else {}


def pool_offsets(public: Path) -> np.ndarray:
    with np.load(public / "pool_index.npz", allow_pickle=False) as z:
        return z["token_offsets"]


def _sel_timed(ctx: Context, _):
    mod = ctx.import_submission("pipeline.py", required=("select",))
    t0 = time.process_time()
    idx = mod.select(str(ctx.public), int(_overrides(ctx).get("budget_tokens", BUDGET_TOKENS)), str(ctx.work / "scratch"))
    return np.asarray(idx), time.process_time() - t0


def _sel_finish(ctx: Context, st):
    idx, cpu = st
    off = pool_offsets(ctx.public)
    n_docs = len(off) - 1
    if idx.ndim != 1 or idx.dtype.kind not in "iu" or (len(idx) and (idx.min() < 0 or idx.max() >= n_docs)):
        raise SystemExit(f"select() must return a 1-D integer array of document indices in [0, {n_docs})")
    toks = int(np.sum(off[idx + 1] - off[idx])) if len(idx) else 0
    need = int(_overrides(ctx).get("min_selected_tokens", MIN_SELECTED))
    if toks < need:
        raise SystemExit(f"the selection holds {toks} tokens, fewer than {need}")
    np.save(ctx.work / "selection.npy", idx.astype(np.int64))
    return {"selection": idx.astype(np.int64), "selected_tokens": np.asarray([toks], np.int64),
            "select_cpu_seconds": np.asarray([cpu])}


def _train_setup(ctx: Context):
    from token_data import read_shard
    idx = np.load(ctx.work / "selection.npy")
    off = pool_offsets(ctx.public)
    pool = read_shard(ctx.public / "pool_tokens.bin")
    return np.concatenate([pool[off[i]:off[i + 1]] for i in idx])


def _train_timed(ctx: Context, stream):
    import torch
    from fixed_norm import SCHEME
    from norm_lm import LM, SCALES
    ov = _overrides(ctx)
    torch.manual_seed(ctx.seed)
    model = LM(SCALES["mu2"], SCHEME).to(ctx.device)
    res = lm_train.train(model, stream, int(ov.get("steps", STEPS)), {}, ctx.seed, ctx.device, PARAMS["lr"])
    return model, res


def _train_finish(ctx: Context, st):
    model, res = st
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        tgt, gen = z["target_tokens"], z["general_tokens"]
    prm = {}
    lt = lm_train.DIVERGED_LOSS if res["diverged"] else lm_train.heldout_loss(model, tgt, prm, ctx.device)
    lg = lm_train.DIVERGED_LOSS if res["diverged"] else lm_train.heldout_loss(model, gen, prm, ctx.device)
    return {"target_loss": np.asarray([lt]), "general_loss": np.asarray([lg]), "diverged": np.asarray([float(res["diverged"])])}


STAGES = {"select": Stage(timed=_sel_timed, finish=_sel_finish),
          "train": Stage(setup=_train_setup, timed=_train_timed, finish=_train_finish)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Self-check: the public dev target and general tokens play the held-out role; there are no duplicate labels.
    --quick trains for 300 steps and accepts a selection of 2M tokens."""
    with np.load(public / "dev_text.npz", allow_pickle=False) as z:
        tgt, gen = z["target_tokens"], z["general_tokens"]
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "held_out.npz", target_tokens=tgt, general_tokens=gen)
    pub = public
    if quick:
        pub = out / "public"
        pub.mkdir(exist_ok=True)
        for p in public.iterdir():
            dst = pub / p.name
            if not dst.exists():
                dst.symlink_to(p.resolve())
        (pub / "overrides.json").write_text(json.dumps({"steps": 300, "min_selected_tokens": 2_000_000, "budget_tokens": 9_830_400}))
    return {"public": pub, "held_out": out / "held_out.npz", "reference": {}, "time_scale": 1.0}
