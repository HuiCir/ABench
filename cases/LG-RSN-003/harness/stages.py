"""LG-RSN-003 pipeline: LegalBench classification tasks answered by a submitted pipeline around the frozen
Qwen2.5-1.5B-Instruct, on tasks seen during development and on tasks it has never seen (evaluator and self-check).

One stage, "predict" (the frozen model is under models/Qwen2.5-1.5B-Instruct):
  timed   import pipeline.py, load(checkpoint_dir, models_dir, device) -> state                   (killed at 600 s)
  finish  for each held-out task, in a fixed shuffled order of tasks: predict(state, task, rows) on its test rows in
          batches of 64 -> one answer per row (legaltasks.py describes tasks and rows)               (≤ 3,600 s)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from abench_stage import Context, Stage

PARAMS = {"batch": 64, "model": "Qwen2.5-1.5B-Instruct", "score": "balanced accuracy per task, mean over tasks"}
HELD_OUT_KEYS = ["tasks_json", "rows_json", "row_task"]
PIPELINE = [{"stage": "predict", "timeout": 600 + 3600, "finish_cap": 3600, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": True}]


def _timed(ctx: Context, _):
    mod = ctx.import_submission("pipeline.py", required=("load", "predict"))
    return mod, mod.load(str(ctx.submission / "checkpoint"), str(ctx.models), ctx.device)


def _finish(ctx: Context, st):
    mod, state = st
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        tasks = [json.loads(s) for s in z["tasks_json"]]
        rows = [json.loads(s) for s in z["rows_json"]]
        row_task = z["row_task"].astype(np.int64)
    pred = [""] * len(rows)
    t0 = time.perf_counter()
    for k in np.random.default_rng(ctx.seed).permutation(len(tasks)):
        idx = np.flatnonzero(row_task == k)
        for s in range(0, len(idx), PARAMS["batch"]):
            b = idx[s:s + PARAMS["batch"]]
            out = mod.predict(state, dict(tasks[k]), [dict(rows[i]) for i in b])
            if len(out) != len(b):
                raise ValueError(f"predict() returned {len(out)} answers for {len(b)} rows")
            for i, a in zip(b, out):
                pred[i] = str(a)
    ms = (time.perf_counter() - t0) * 1e3 / max(len(rows), 1)
    return {"answer": np.asarray(pred, dtype=np.str_), "ms_per_row": np.asarray([ms])}


STAGES = {"predict": Stage(timed=_timed, finish=_finish)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Self-check: the dev rows of the twenty public tasks (--quick: four tasks, 16 rows each)."""
    pub = json.loads((public / "legalbench_public.json").read_text(encoding="utf-8"))
    tasks, rows, row_task, gold = [], [], [], []
    for t in pub["tasks"][: 4 if quick else None]:
        dev = t["dev"][: 16 if quick else None]
        spec = {k: t[k] for k in ("name", "prompt_template", "labels", "columns", "train")}
        tasks.append(spec)
        for r in dev:
            rows.append({c: r[c] for c in t["columns"]})
            row_task.append(len(tasks) - 1)
            gold.append(r["answer"])
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "held_out.npz", tasks_json=np.asarray([json.dumps(t) for t in tasks], dtype=np.str_),
             rows_json=np.asarray([json.dumps(r) for r in rows], dtype=np.str_), row_task=np.asarray(row_task, np.int64))
    ref = {"gold": np.asarray(gold, dtype=np.str_), "row_task": np.asarray(row_task, np.int64),
           "task_labels_json": np.asarray([json.dumps(t["labels"]) for t in tasks], dtype=np.str_),
           "task_group": np.asarray(["seen"] * len(tasks), dtype=np.str_),
           "task_name": np.asarray([t["name"] for t in tasks], dtype=np.str_)}
    return {"public": public, "held_out": out / "held_out.npz", "reference": ref, "time_scale": 1.0}
