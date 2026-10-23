"""X-ROBUST-009 pipeline: an intent classifier that knows when to abstain, scored on in-scope, out-of-scope, shifted
(Banking77) and noisy queries (evaluator and self-check).

One stage, "predict" (the provided roberta-base is available under models/):
  timed   import model.py and call load(checkpoint_dir, models_dir, device)          (≤ 600 s)
  finish  predict(state, texts) on all held-out queries, in batches of 512 in a fixed shuffled order, timed per query
          → labels (best in-scope intent), confidence, abstain                         (≤ 900 s)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from abench_stage import Context, Stage
from textpack import pack, unpack

PARAMS = {"n_intents": 150, "batch": 512, "groups": {"in_scope": 0, "out_of_scope": 1, "banking77": 2, "typo": 3}}
HELD_OUT_KEYS = ["text_utf8", "text_offsets"]
PIPELINE = [{"stage": "predict", "timeout": 600 + 900, "finish_cap": 900, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": True}]


def _timed(ctx: Context, _):
    mod = ctx.import_submission("model.py", required=("load", "predict"))
    return mod, mod.load(str(ctx.submission / "checkpoint"), str(ctx.models) if ctx.models else None, ctx.device)


def _finish(ctx: Context, st):
    mod, state = st
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        texts = unpack(z["text_utf8"], z["text_offsets"])
    n = len(texts)
    order = np.random.default_rng(ctx.seed).permutation(n)          # groups are interleaved in every batch
    lab, conf, abst = np.zeros(n, np.int64), np.zeros(n), np.zeros(n, np.int64)
    t0 = time.perf_counter()
    for s in range(0, n, PARAMS["batch"]):
        idx = order[s:s + PARAMS["batch"]]
        l, c, a = mod.predict(state, [texts[i] for i in idx])
        lab[idx], conf[idx], abst[idx] = np.asarray(l), np.asarray(c, np.float64), np.asarray(a).astype(np.int64)
    ms = (time.perf_counter() - t0) * 1e3 / max(n, 1)
    ck = ctx.submission / "checkpoint"
    size = sum(p.stat().st_size for p in ck.rglob("*") if p.is_file()) if ck.is_dir() else 0
    return {"label": lab, "confidence": conf, "abstain": abst, "ms_per_query": np.asarray([ms]),
            "checkpoint_bytes": np.asarray([size], np.int64)}


STAGES = {"predict": Stage(timed=_timed, finish=_finish)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Self-check: the public validation split (in-scope and out-of-scope) plays the held-out role; its in-scope
    queries are repeated with character noise as the 'typo' group. There is no Banking77 stand-in."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    with np.load(public / "clinc_public.npz", allow_pickle=False) as z:
        t = unpack(z["val_text_utf8"], z["val_text_offsets"])
        y = z["val_label"]
    if quick:
        keep = np.r_[np.flatnonzero(y >= 0)[:300], np.flatnonzero(y < 0)[:50]]
        t, y = [t[i] for i in keep], y[keep]
    rng = np.random.default_rng(1)
    ins = [x for x, l in zip(t, y) if l >= 0]
    texts = t + [_typos(x, rng) for x in ins]
    label = np.concatenate([y, y[y >= 0]])
    group = np.concatenate([np.where(y >= 0, 0, 1), np.full(len(ins), 3)]).astype(np.int64)
    out.mkdir(parents=True, exist_ok=True)
    u, o = pack(texts)
    np.savez(out / "held_out.npz", text_utf8=u, text_offsets=o)
    return {"public": public, "held_out": out / "held_out.npz", "reference": {"label": label, "group": group},
            "time_scale": 1.0}


def _typos(text: str, rng, rate: float = 0.06) -> str:
    """The same kind of character noise as the held-out 'typo' group (different seed)."""
    out, i, s = [], 0, list(text)
    while i < len(s):
        c = s[i]
        if c.isalpha() and rng.random() < rate:
            op = rng.integers(0, 3)
            if op == 0:
                i += 1
                continue
            out += [c, c] if op == 1 else ([s[i + 1], c] if i + 1 < len(s) else [c])
            i += 2 if op == 2 and i + 1 < len(s) else 1
            continue
        out.append(c)
        i += 1
    return "".join(out)
