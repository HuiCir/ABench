"""FN-QA-004 pipeline: numerical questions over financial reports answered by a submitted pipeline around the frozen
Qwen2.5-1.5B-Instruct (evaluator and self-check use this same file).

One stage, "answer" (the frozen model is under models/Qwen2.5-1.5B-Instruct):
  timed   import pipeline.py, load(checkpoint_dir, models_dir, device) -> state                   (killed at 600 s)
  finish  answer(state, items) on the held-out questions in batches of 32, in a fixed shuffled order; each item is
          {id, question, pre_text, post_text, table} and each result an answer value and/or a FinQA program, which
          this stage executes on the item's table (finqa.execute)                                       (≤ 3,600 s)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import finqa as F
from abench_stage import Context, Stage

PARAMS = {"batch": 32, "model": "Qwen2.5-1.5B-Instruct", "tolerance": "max(5e-5, 1e-4*|gold|) after rounding to 5 decimals"}
HELD_OUT_KEYS = ["items_json"]
PIPELINE = [{"stage": "answer", "timeout": 600 + 3600, "finish_cap": 3600, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": True}]


def _timed(ctx: Context, _):
    mod = ctx.import_submission("pipeline.py", required=("load", "answer"))
    return mod, mod.load(str(ctx.submission / "checkpoint"), str(ctx.models), ctx.device)


def _as_text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _finish(ctx: Context, st):
    mod, state = st
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        items = [json.loads(s) for s in z["items_json"]]
    n = len(items)
    order = np.random.default_rng(ctx.seed).permutation(n)
    ans, prog, val = [""] * n, [""] * n, [""] * n
    t0 = time.perf_counter()
    for s in range(0, n, PARAMS["batch"]):
        idx = order[s:s + PARAMS["batch"]]
        batch = [dict(items[i]) for i in idx]
        out = mod.answer(state, batch)
        if len(out) != len(idx):
            raise ValueError(f"answer() returned {len(out)} results for {len(idx)} items")
        for i, r in zip(idx, out):
            r = r if isinstance(r, dict) else {"answer": r}
            ans[i] = _as_text(r.get("answer"))
            prog[i] = str(r.get("program") or "")
            if prog[i]:
                val[i] = _as_text(F.execute(prog[i], items[i]["table"]))
    ms = (time.perf_counter() - t0) * 1e3 / max(n, 1)
    return {"answer": np.asarray(ans, dtype=np.str_), "program": np.asarray(prog, dtype=np.str_),
            "program_value": np.asarray(val, dtype=np.str_), "ms_per_question": np.asarray([ms])}


STAGES = {"answer": Stage(timed=_timed, finish=_finish)}


def public_item(x: dict) -> dict:
    """What a pipeline sees of a FinQA example (no gold program, answer or annotated evidence)."""
    return {"id": x["id"], "question": x["qa"]["question"], "pre_text": x["pre_text"], "post_text": x["post_text"],
            "table": x["table"]}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Self-check: the FinQA dev split (883 questions; --quick: the first 64)."""
    dev = json.loads((public / "dev.json").read_text(encoding="utf-8"))
    if quick:
        dev = dev[:64]
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "held_out.npz", items_json=np.asarray([json.dumps(public_item(x)) for x in dev], dtype=np.str_))
    ref = {"gold": np.asarray([str(x["qa"]["exe_ans"]) for x in dev], dtype=np.str_),
           "steps": np.asarray([len(F.split_steps(x["qa"]["program"])) for x in dev], np.int64),
           "has_table_evidence": np.asarray([any(k.startswith("table") for k in x["qa"].get("gold_inds", {}))
                                             for x in dev], np.int64)}
    return {"public": public, "held_out": out / "held_out.npz", "reference": ref, "time_scale": 1.0}
