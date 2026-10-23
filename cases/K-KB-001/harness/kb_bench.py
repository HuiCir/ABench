"""KernelBench-style problems: loading, input distributions, correctness and timing (evaluator code; identical copy
in each kernel case that uses it: K-KB-001, K-FUS-002).

A problem is a KernelBench file (Ouyang et al. 2025, MIT licence) defining `Model`, `get_inputs()` and
`get_init_inputs()`. A solution is `kernels/<problem>.py` defining `ModelNew` with the same constructor and forward
signature as `Model`.

Per problem the evaluator
  - loads the problem file and applies the case's shape overrides (module globals, e.g. `batch_size`) before calling
    `get_inputs()` / `get_init_inputs()`;
  - builds Model and ModelNew after the same torch.manual_seed(weight seed), copies Model's parameters and buffers
    into ModelNew where names and shapes match, and puts both in eval mode;
  - checks correctness on the five input distributions of INPUT_KINDS (one held-out seed each), once more after
    timing on fresh inputs (another seed), and again after those input buffers are overwritten in place with new
    values (the same seed column): every output element within ATOL + RTOL * |ref| of Model's fp32 output
    (KernelBench's tolerance), positions where Model's output is not finite skipped, shapes and count must match;
  - fails a problem whose ModelNew modifies any of its input tensors (the inputs are compared bit for bit with a
    copy taken before the call);
  - times one call as the wall-clock time between two torch.cuda.synchronize() (all streams), median of REPS calls
    after WARMUP calls, inputs fixed.
Matmuls and convolutions of the reference run in full fp32 (TF32 disabled); before every call of Model or ModelNew
the evaluator restores these flags, so a solution that wants TF32 or lower precision must use it in its own kernels.
"""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np

INPUT_KINDS = ("native", "normal", "scaled", "sparse", "large")   # native: the problem's own get_inputs()
WARMUP, REPS, ATOL, RTOL = 3, 10, 1e-2, 1e-2
SEED_COLS = ("weights",) + INPUT_KINDS + ("post_timing",)
FORBIDDEN = r"torch\.compile|_inductor|_dynamo|torch\.jit|torch\.cuda\.Stream\(|cudaStreamCreate|torch\.cuda\.stream\("


# --------------------------------------------------------------------------- problems and inputs
def list_problems(public: Path) -> list[dict]:
    """problems.json: [{"name": "L1_19_ReLU", "file": "L1_19_ReLU.py", "overrides": {...}}, ...]"""
    return json.loads((public / "problems.json").read_text(encoding="utf-8"))


def load_problem(path: Path, overrides: dict) -> dict:
    g: dict = {"__name__": f"kb_problem_{path.stem}"}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), g)
    g.update(overrides or {})          # functions read the module globals at call time
    return g


def load_solution(path: Path):
    spec = importlib.util.spec_from_file_location(f"kb_solution_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not callable(getattr(mod, "ModelNew", None)):
        raise ImportError(f"{path.name} must define ModelNew")
    return mod


def make_inputs(g: dict, kind: str, seed: int, device: str) -> list:
    """The problem's get_inputs() under torch.manual_seed(seed); for kinds other than native every floating-point
    tensor is replaced by fresh values of the same shape and dtype: normal N(0,1), scaled 8·N(0,1), sparse N(0,1)
    with 75% of the entries set to zero, large 100·N(0,1) (numerical range: overflow in a reduced-precision kernel
    or an unguarded exp shows here). Integer tensors and non-tensor inputs are kept."""
    import torch
    torch.manual_seed(seed)
    ins = g["get_inputs"]()
    gen = torch.Generator(device="cpu").manual_seed(seed + 1)
    out = []
    for x in ins:
        if torch.is_tensor(x) and x.is_floating_point() and kind != "native":
            z = torch.randn(x.shape, generator=gen, dtype=torch.float32)
            if kind == "scaled":
                z = z * 8.0
            elif kind == "large":
                z = z * 100.0
            elif kind == "sparse":
                z = z * (torch.rand(x.shape, generator=gen) < 0.25)
            x = z.to(x.dtype)
        out.append(x.to(device) if torch.is_tensor(x) else x)
    return out


def precision_defaults() -> None:
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def build(g: dict, cls, seed: int, device: str, like=None):
    import torch
    torch.manual_seed(seed)
    m = cls(*g["get_init_inputs"]()).to(device).eval()
    if like is not None:
        src, tgt = like.state_dict(), m.state_dict()
        m.load_state_dict({k: v for k, v in src.items() if k in tgt and tgt[k].shape == v.shape}, strict=False)
    return m


def run(m, inputs):
    import torch
    precision_defaults()
    with torch.no_grad():
        y = m(*inputs)
    return [y] if torch.is_tensor(y) else list(y)


def run_new(m, inputs) -> tuple[list, bool]:
    """Run a solution and report whether it modified any of its input tensors (bit-for-bit comparison with copies
    taken before the call)."""
    import torch
    before = [x.clone() if torch.is_tensor(x) else None for x in inputs]
    out = run(m, inputs)
    changed = any(b is not None and not torch.equal(b, x) for b, x in zip(before, inputs))
    return out, changed


def refresh_inplace(inputs: list, seed: int) -> list:
    """Overwrite every floating-point input tensor in place with fresh N(0,1) values (same storage, same data
    pointer): a solution that keys a cache on buffer addresses returns stale results here."""
    import torch
    gen = torch.Generator(device="cpu").manual_seed(seed + 7)
    for x in inputs:
        if torch.is_tensor(x) and x.is_floating_point():
            x.copy_(torch.randn(x.shape, generator=gen, dtype=torch.float32).to(x.dtype))
    return inputs


def check_before_timing(g: dict, cls, sd, device: str):
    """Correctness of a solution class on every input distribution. Returns (worst error ratio, the last ModelNew
    instance); inf if the solution modifies an input. Stops at the first failing distribution."""
    worst, new_m = 0.0, None
    for j, kind in enumerate(INPUT_KINDS):
        ref_m = build(g, g["Model"], int(sd[0]), device)
        x = make_inputs(g, kind, int(sd[1 + j]), device)
        ref = run(ref_m, [t.clone() if _is_t(t) else t for t in x])
        new_m = build(g, cls, int(sd[0]), device, like=ref_m)
        out, changed = run_new(new_m, x)
        worst = max(worst, float("inf") if changed else compare(ref, out))
        del ref_m, x, ref, out
        if worst > 1.0:
            break
    return worst, new_m


def check_after_timing(g: dict, new_m, sd, device: str) -> float:
    """The same ModelNew instance that was timed, on fresh inputs, then on the same buffers overwritten in place."""
    col = int(sd[SEED_COLS.index("post_timing")])
    ref_m = build(g, g["Model"], int(sd[0]), device)
    worst = 0.0
    x = make_inputs(g, "normal", col, device)
    for step in range(2):
        if step == 1:
            refresh_inplace(x, col)
        ref = run(ref_m, [t.clone() if _is_t(t) else t for t in x])
        out, changed = run_new(new_m, x)
        worst = max(worst, float("inf") if changed else compare(ref, out))
    return worst


def _is_t(x) -> bool:
    import torch
    return torch.is_tensor(x)


def compare(ref: list, out: list) -> float:
    """Largest |out - ref| / (ATOL + RTOL |ref|) over finite reference positions; inf on a shape or count mismatch
    or a non-finite output where the reference is finite. Correct means <= 1."""
    import torch
    if len(ref) != len(out):
        return float("inf")
    worst = 0.0
    for a, b in zip(ref, out):
        if not torch.is_tensor(b) or tuple(a.shape) != tuple(b.shape):
            return float("inf")
        a, b = a.float(), b.detach().float().to(a.device)
        fin = torch.isfinite(a)
        if not bool(torch.isfinite(b[fin]).all()):
            return float("inf")
        if fin.any():
            worst = max(worst, float(((a - b).abs()[fin] / (ATOL + RTOL * a.abs()[fin])).max()))
    return worst


def time_call(fn, inputs) -> float:
    """Median wall-clock milliseconds of one call between two device-wide synchronisations."""
    import torch
    precision_defaults()
    with torch.no_grad():
        for _ in range(WARMUP):
            fn(*inputs)
        torch.cuda.synchronize()
        ts = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            fn(*inputs)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def peak_extra_mb(fn, inputs) -> float:
    """Peak memory allocated during one call beyond what was allocated before it (MB)."""
    import torch
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    with torch.no_grad():
        y = fn(*inputs)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del y
    return peak / 2**20


# --------------------------------------------------------------------------- tamper guard
def snapshot() -> dict:
    """Identities of the torch entry points the reference relies on; a solution must not replace any of them."""
    import torch
    import torch.nn.functional as F
    snap = {}
    for owner, name in ((torch, "torch"), (F, "F"), (torch.Tensor, "Tensor"), (torch.cuda, "cuda")):
        for k in dir(owner):
            if k.startswith("__"):
                continue
            try:
                v = getattr(owner, k)
            except Exception:
                continue
            if callable(v):
                snap[f"{name}.{k}"] = id(v)
    for k in dir(torch.nn):
        c = getattr(torch.nn, k, None)
        if isinstance(c, type) and issubclass(c, torch.nn.Module):
            snap[f"nn.{k}.forward"] = id(c.__dict__.get("forward", c.forward))
    snap["time.perf_counter"] = id(time.perf_counter)
    return snap


def tampered(before: dict) -> list[str]:
    now = snapshot()
    return sorted(k for k, v in before.items() if now.get(k) != v)
