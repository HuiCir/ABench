"""X-LORA-004 pipeline: score parameter-efficient fine-tuning states for frozen roberta-base on four GLUE tasks
(SST-2, CoLA, MNLI-matched, QNLI). Texts are stored packed (UTF-8 bytes + offsets, textpack.py).

The evaluator does not train. One stage, "eval" (imports peft.py; the provided roberta-base is available):
  timed   load checkpoint/states.pt ({task: {name: tensor}}, task-specific parameters only), import peft.py
  finish  count the elements per task; predict(models_dir, task, state, sentence1, sentence2, device) -> labels,
          for each task's held-out split                                                           (≤ 600 s)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from abench_stage import Context, Stage
from textpack import pack, unpack

TASKS = {"sst2": ("sentence", None, 2), "cola": ("sentence", None, 2),
         "mnli": ("premise", "hypothesis", 3), "qnli": ("question", "sentence", 2)}
PARAM_CAP = 1_000_000
PARAMS = {"tasks": list(TASKS), "param_cap_per_task": PARAM_CAP, "base_model": "roberta-base"}
HELD_OUT_KEYS = [f"{t}_{c}_{part}" for t, (a, b, _) in TASKS.items() for c in (a, b) if c for part in ("utf8", "offsets")]
PIPELINE = [{"stage": "eval", "timeout": 180 + 600, "finish_cap": 600, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": True}]


def _texts(z, key, idx=None):
    return unpack(z[f"{key}_utf8"], z[f"{key}_offsets"], idx)


def _timed(ctx: Context, _):
    import torch
    extra = sorted(str(p.relative_to(ctx.submission / "checkpoint")) for p in (ctx.submission / "checkpoint").rglob("*")
                   if p.is_file() and p.name != "states.pt")
    if extra:
        raise SystemExit(f"checkpoint/ may contain only states.pt; found {extra[:5]}")
    states = torch.load(ctx.submission / "checkpoint" / "states.pt", map_location="cpu", weights_only=True)
    mod = ctx.import_submission("peft.py", required=("predict",))
    return mod, states


def _finish(ctx: Context, st):
    import torch
    mod, states = st
    if not isinstance(states, dict) or set(states) != set(TASKS):
        raise SystemExit(f"checkpoint/states.pt must hold a dict with keys {sorted(TASKS)}")
    counts = []
    for t in TASKS:
        s = states[t]
        if not isinstance(s, dict) or not all(isinstance(k, str) and torch.is_tensor(v) for k, v in s.items()):
            raise SystemExit(f"state of {t} must be a dict of named tensors")
        counts.append(sum(v.numel() for v in s.values()))
    out = {"params_per_task": np.asarray(counts, np.int64)}
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        for t, (a, b, _) in TASKS.items():
            s1 = _texts(z, f"{t}_{a}")
            s2 = _texts(z, f"{t}_{b}") if b else None
            out[f"pred_{t}"] = np.asarray(mod.predict(str(ctx.models), t, states[t], s1, s2, ctx.device)).astype(np.int64)
    return out


STAGES = {"eval": Stage(timed=_timed, finish=_finish)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Dev split for selfcheck.py: per task, the 10% of the training split in dev_idx. For an unbiased dev estimate,
    train your states without them (the reference train.py does this unless --include-dev is given)."""
    with np.load(public / "glue_train.npz", allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    held, ref = {}, {}
    for t, (a, b, _) in TASKS.items():
        dev = d[f"{t}_dev_idx"]
        if quick:
            dev = dev[:100]
        for c in (a, b):
            if c:
                held[f"{t}_{c}_utf8"], held[f"{t}_{c}_offsets"] = pack(_texts(d, f"{t}_{c}", dev))
        ref[f"{t}_label"] = d[f"{t}_label"][dev]
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "held_out.npz", **held)
    return {"public": public, "held_out": out / "held_out.npz", "reference": ref, "time_scale": 1.0}
