"""U-LOOP-002 pipeline: accuracy of a looped model at three test-time compute budgets on three reasoning tasks
(evaluator and self-check use this same file).

The evaluator does not train. One stage, "predict":
  timed   import model.py, load(checkpoint_dir, device) -> model                     (load time, killed at 120 s)
  finish  for each task and each budget B in PARAMS["budgets"] (in reference-layer units, see below), the evaluator
          drives the loop itself:
            state = init(model, task, x, budget_flops)          # embed the problems
            repeat at most PARAMS["max_iters"] times:
              state = step(model, state, active)                 # one recurrent iteration (active: bool mask)
              logits, halt = readout(model, state)               # answers so far, and which problems stop now
              a problem's answer is argmax(logits) at the iteration where it halts (or at the last iteration)
          Every call runs under torch.utils.flop_counter.FlopCounterMode; the floating-point operations of all calls
          divided by the number of problems is the test-time compute per problem.             (all ≤ 600 s)
Budget unit: F_ref(task), the FLOPs of one application of the reference layer (refblock.py: a pre-norm transformer
layer of width 256 with 4 heads and an MLP of 1,024) to one problem of the task's sequence length.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from abench_stage import Context, Stage
import reason_tasks as T

PER_TASK = 512
BUDGETS = (4, 8, 16)
PARAMS = {"tasks": list(T.TASKS), "budgets": list(BUDGETS), "max_iters": 64, "problems_per_task": PER_TASK,
          "budget_tolerance": 1.02, "max_params": 20_000_000}
HELD_OUT_KEYS = [f"x_{t}" for t in T.TASKS]
PIPELINE = [{"stage": "predict", "timeout": 120 + 600, "finish_cap": 600, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": False}]
DEV_SEED = 2502


def build_split(seed: int, per_task: int) -> dict:
    rng = np.random.default_rng(seed)
    d = {}
    for t in T.TASKS:
        x, y, dd = T.make_mixed(t, per_task, rng)
        d[f"x_{t}"], d[f"y_{t}"], d[f"d_{t}"] = x, y, dd
    return d


class _Counter:
    """FLOPs of the code run inside it (torch FlopCounterMode); 0 when PyTorch is not installed (unit tests)."""

    def __init__(self):
        self.total = 0

    def run(self, fn, *args):
        try:
            from torch.utils.flop_counter import FlopCounterMode
        except ImportError:
            return fn(*args)
        with FlopCounterMode(display=False) as fc:
            out = fn(*args)
        self.total += int(fc.get_total_flops())
        return out


def reference_flops(task: str, device: str) -> float:
    """F_ref(task): FLOPs of the reference layer on one problem of the task (measured, not assumed)."""
    try:
        import torch
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return 1.0
    from refblock import RefLayer
    layer = RefLayer().to(device)
    x = torch.zeros(1, T.SEQ_LEN[task], 256, device=device)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        layer(x)
    return float(fc.get_total_flops())


def _to_np(a) -> np.ndarray:
    if hasattr(a, "detach"):
        a = a.detach().float().cpu().numpy()
    return np.asarray(a)


def run_loop(mod, model, task: str, x: np.ndarray, budget_flops: float, max_iters: int):
    """Returns (predictions, iterations per problem, FLOPs per problem)."""
    n = len(x)
    c = _Counter()
    state = c.run(mod.init, model, task, x.copy(), float(budget_flops))
    active = np.ones(n, bool)
    pred = np.full(n, -1, np.int64)
    iters = np.zeros(n, np.int64)
    for _ in range(max_iters):
        state = c.run(mod.step, model, state, active.copy())
        iters[active] += 1
        logits, halt = c.run(mod.readout, model, state)
        logits, halt = _to_np(logits), _to_np(halt).astype(bool).reshape(-1)
        if logits.shape != (n, T.N_CLASSES[task]) or halt.shape != (n,):
            raise ValueError(f"{task}: readout must return logits ({n}, {T.N_CLASSES[task]}) and halt ({n},)")
        if not np.isfinite(logits).all():
            raise ValueError(f"{task}: non-finite logits")
        stop = active & halt
        pred[stop] = logits[stop].argmax(1)
        active &= ~halt
        if not active.any():
            break
    if active.any():
        pred[active] = logits[active].argmax(1)
    return pred, iters, c.total / n


def _timed(ctx: Context, _):
    mod = ctx.import_submission("model.py", required=("load", "init", "step", "readout"))
    model = mod.load(str(ctx.submission / "checkpoint"), ctx.device)
    return mod, model


def _finish(ctx: Context, state):
    mod, model = state
    out = {"n_params": np.asarray([float(mod.n_params(model)) if hasattr(mod, "n_params") else _count_params(model)])}
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        for t in PARAMS["tasks"]:
            fref = reference_flops(t, ctx.device)
            out[f"fref_{t}"] = np.asarray([fref])
            for b in PARAMS["budgets"]:
                pred, iters, flops = run_loop(mod, model, t, z[f"x_{t}"], b * fref, PARAMS["max_iters"])
                out[f"pred_{t}_b{b}"] = pred
                out[f"iters_{t}_b{b}"] = iters
                out[f"flops_{t}_b{b}"] = np.asarray([flops])
    return out


def _count_params(model) -> float:
    try:
        return float(sum(p.numel() for p in model.parameters()))
    except AttributeError:
        return float("nan")


STAGES = {"predict": Stage(timed=_timed, finish=_finish)}


def make_dev(public, out: Path, quick: bool = False) -> dict:
    """Dev set for selfcheck.py: freshly generated problems (same distribution as the evaluation, public seed);
    --quick uses 64 per task."""
    out.mkdir(parents=True, exist_ok=True)
    d = build_split(DEV_SEED, 64 if quick else PER_TASK)
    np.savez(out / "held_out.npz", **{k: v for k, v in d.items() if k in HELD_OUT_KEYS})
    return {"public": None, "held_out": out / "held_out.npz",
            "reference": {k: v for k, v in d.items() if k.startswith(("y_", "d_"))}, "time_scale": 1.0}
