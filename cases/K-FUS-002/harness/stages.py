"""K-FUS-002 pipeline: kernels for 20 KernelBench Level-2 problems, correct on five input distributions, timed against
PyTorch eager and torch.compile (evaluator and self-check; K-KB-001 uses the same file with its own problem list).

Two stages, each in a new process on the same GPU:
  reference  nothing timed; finish: for every problem, Model in PyTorch eager and torch.compile(Model) (default mode),
             timed on the problem's own inputs (kb_bench.time_call), and eager peak memory. Your code is not imported.
  bench      setup: a snapshot of torch's entry points; timed: import every kernels/<problem>.py that exists (module-level
             compilation, load_inline builds and autotuning count here); finish: for every problem, correctness on
             the five input distributions (a ModelNew that modifies its inputs fails), timing and peak memory of
             ModelNew, a correctness check of the timed instance on fresh inputs and again after those buffers are
             overwritten in place, and a check that torch's entry points are unchanged.
Problem files, shape overrides and the order of problems come from `$ABENCH_DATA_DIR/problems.json`.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

import kb_bench as KB
from abench_stage import Context, Stage

CASE = "K-FUS-002"
PARAMS = {"input_kinds": list(KB.INPUT_KINDS), "atol": KB.ATOL, "rtol": KB.RTOL, "warmup": KB.WARMUP, "reps": KB.REPS,
          "seed_columns": list(KB.SEED_COLS)}
HELD_OUT_KEYS = ["seeds"]                      # (n_problems, 7): weights, native, normal, scaled, sparse, large, post_timing
PIPELINE = [{"stage": "reference", "timeout": 60, "finish_cap": 3000, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": False},
            {"stage": "bench", "timeout": 900, "finish_cap": 2400, "tail_slack": 10, "threads": 8,
             "held_out": True, "models": False}]

_SNAP: dict = {}


def _seeds(ctx: Context) -> np.ndarray:
    with np.load(ctx.post_dir / "held_out.npz", allow_pickle=False) as z:
        return z["seeds"].astype(np.int64)


def _problems(ctx: Context, n: int) -> list[dict]:
    return KB.list_problems(ctx.public)[:n]


# --------------------------------------------------------------------------- reference stage
def _ref_timed(ctx: Context, _):
    return None


def _ref_finish(ctx: Context, _):
    import torch
    import torch._dynamo
    seeds = _seeds(ctx)
    rows = []
    for p, sd in zip(_problems(ctx, len(seeds)), seeds):
        g = KB.load_problem(ctx.public / p["file"], p.get("overrides"))
        m = KB.build(g, g["Model"], int(sd[0]), ctx.device)
        x = KB.make_inputs(g, "native", int(sd[1]), ctx.device)
        t_eager = KB.time_call(m, x)
        mem = KB.peak_extra_mb(m, x)
        try:
            torch._dynamo.reset()
            t_comp = KB.time_call(torch.compile(m), x)
        except Exception:
            t_comp = float("nan")           # graphs torch.compile cannot handle: eager is the baseline
        rows.append([t_eager, t_comp, mem])
        del m, x
        torch.cuda.empty_cache()
    return {"rows": np.asarray(rows, np.float64)}


# --------------------------------------------------------------------------- bench stage
def _setup(ctx: Context):
    import torch  # noqa: F401
    KB.precision_defaults()
    _SNAP.update(KB.snapshot())


def _timed(ctx: Context, _):
    import sys
    if str(ctx.submission) not in sys.path:
        sys.path.insert(0, str(ctx.submission))      # kernels may import helpers from src/
    mods = {}
    for p in KB.list_problems(ctx.public):
        f = ctx.submission / "kernels" / f"{p['name']}.py"
        if f.is_file():
            try:
                mods[p["name"]] = KB.load_solution(f)
            except Exception as e:  # a broken file fails only its problem
                mods[p["name"]] = e
    return mods


def _scan(sub: Path) -> int:
    pat = re.compile(KB.FORBIDDEN)
    return sum(len(pat.findall(f.read_text(encoding="utf-8", errors="replace"))) for f in sub.rglob("*.py"))


def _finish(ctx: Context, mods):
    import torch
    seeds = _seeds(ctx)
    rows, touched, dead = [], set(), False
    for p, sd in zip(_problems(ctx, len(seeds)), seeds):
        mod = mods.get(p["name"])
        row = [0.0 if mod is None else 1.0, float("inf"), 0.0, float("inf"), float("nan")]
        if dead or mod is None or isinstance(mod, Exception):
            rows.append(row)
            continue
        try:
            touched.update(KB.tampered(_SNAP))
            g = KB.load_problem(ctx.public / p["file"], p.get("overrides"))
            worst, new_m = KB.check_before_timing(g, mod.ModelNew, sd, ctx.device)
            row[1] = worst
            if worst <= 1.0:
                x = KB.make_inputs(g, "native", int(sd[1]), ctx.device)
                row[3] = KB.time_call(new_m, x)
                row[4] = KB.peak_extra_mb(new_m, x)
                del x
                post = KB.check_after_timing(g, new_m, sd, ctx.device)
                row[1] = max(worst, post)
                row[2] = float(post <= 1.0)
            touched.update(KB.tampered(_SNAP))
        except Exception as e:
            if "CUDA" in str(e) or isinstance(e, getattr(torch, "AcceleratorError", ())):
                dead = True                  # the CUDA context may be corrupted: later problems count as failed
            row[2], row[3] = 0.0, float("inf")
        rows.append(row)
        torch.cuda.empty_cache()
    return {"rows": np.asarray(rows, np.float64), "forbidden_api": np.asarray([_scan(ctx.submission)]),
            "tampered": np.asarray([len(touched)])}


STAGES = {"reference": Stage(timed=_ref_timed, finish=_ref_finish),
          "bench": Stage(setup=_setup, timed=_timed, finish=_finish)}


def make_dev(public: Path, out: Path, quick: bool = False) -> dict:
    """Self-check: the same problems with public seeds (quick: the first 4 problems)."""
    out.mkdir(parents=True, exist_ok=True)
    n = len(KB.list_problems(public))
    rng = np.random.default_rng(20250218)
    seeds = rng.integers(1, 2**31 - 1, size=(4 if quick else n, len(KB.SEED_COLS)), dtype=np.int64)
    np.savez(out / "held_out.npz", seeds=seeds)
    return {"public": public, "held_out": out / "held_out.npz", "reference": {}, "time_scale": 1.0}
